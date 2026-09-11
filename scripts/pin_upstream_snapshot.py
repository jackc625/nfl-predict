"""Capture the nflverse frames a gold rebuild consumes into an immutable pinned snapshot.

This is the ONE tool that is allowed to reach nflverse live for the datasets
``data/upstream_pin.py`` serves. It fetches, writes a timestamped snapshot into
``data/bronze/`` following the layer's existing
``<name>_raw_bronze_<season>_W00_<UTCstamp>.parquet`` convention, and records the
provenance -- source, ``nflreadpy`` version, UTC capture time, per-file sha256, seasons
and columns -- into the COMMITTED manifest at ``config/upstream_pin.json``.

The manifest is committed and the parquet is not, because ``data/`` is gitignored. That
asymmetry is the point: a fresh checkout can still say WHICH upstream revision a verdict
was measured against, and can detect that the bytes it has are not those bytes.

ROUND-TRIP FIDELITY IS VERIFIED, NOT ASSUMED. A pin that changed a dtype on the way to
parquet would move gold values for a reason that has nothing to do with the data. Every
snapshot is read back and compared to the frame that was written -- values with
``DataFrame.equals`` and dtypes column by column -- before its digest is recorded. A
mismatch aborts the capture for that season and is reported; it is never written into the
manifest as though it were faithful.

USAGE
-----
Capture everything Phase 31 pins (play-by-play 2001-2025, schedules 1999-2025, depth
charts 2002-2025)::

    .venv/Scripts/python.exe -m scripts.pin_upstream_snapshot --all

Capture or extend one dataset::

    .venv/Scripts/python.exe -m scripts.pin_upstream_snapshot --dataset pbp --seasons 2026 2026

Re-verify an existing pin's bytes against the manifest without fetching anything::

    .venv/Scripts/python.exe -m scripts.pin_upstream_snapshot --verify

Capturing a season that is already pinned REPLACES that season's manifest entry and
writes a NEW timestamped bronze file. The previous file is left on disk, because
``data/bronze/`` is append-only by contract and deleting a recorded snapshot would
destroy the evidence a superseded verdict was measured against.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas as pd

from data.storage import save_bronze_snapshot
from data.upstream_pin import (
    DATASET_COLUMNS,
    DATASET_LOADERS,
    DATASET_TABLE_NAMES,
    MANIFEST_PATH,
    MANIFEST_SCHEMA_VERSION,
    SEALED_LOCK_SCHEMA_VERSION,
    SEALED_THROUGH_SEASON,
    ZONE_SEALED,
    digest_file,
    load_manifest,
    load_sealed_lock,
    zone_for_season,
)
from utils import get_logger

logger = get_logger(__name__)

# The season spans Phase 31 pins, and why each one starts where it does.
#
# pbp 2001: gold covers 2002-2025, and BOTH play-by-play consumers reach one season back.
#   ``TeamFormCalculator.build_features`` prepends ``all_seasons[0] - 1`` and
#   ``QBTracker._load_pbp_data`` loads ``[season - 1, season]``.
# schedules 1999: nflverse schedule coverage begins in 1999 and the frames are tiny, so
#   pinning the whole span costs nothing and leaves no consumer able to fall off the edge.
# depth_charts 2002: the only consumers are ``QBTracker`` and ``InjuryBuilder``, both of
#   which ask for a gold season and never for its predecessor.
DEFAULT_SPANS: dict[str, tuple[int, int]] = {
    "pbp": (2001, 2025),
    "schedules": (1999, 2025),
    "depth_charts": (2002, 2025),
}

# Datasets that reach gold but are NOT pinned here, recorded in the manifest so the
# boundary of the reproducibility claim is legible from the artifact itself.
NOT_PINNED: list[dict[str, str]] = [
    {
        "loader": "nflreadpy.load_injuries",
        "reason": (
            "scripts/ingest_injuries.py already writes its own timestamped "
            "data/bronze/injuries_raw_bronze_*.parquet snapshot and reaches gold through "
            "data/silver/, so its input is already frozen on disk. A second pin would be "
            "a competing snapshot of the same bytes."
        ),
    },
    {
        "loader": "nflreadpy.load_snap_counts",
        "reason": (
            "scripts/ingest_snaps.py already writes its own timestamped "
            "data/bronze/snaps_raw_bronze_*.parquet snapshot; same reason as injuries."
        ),
    },
    {
        "loader": "models/blending_data.py:110 nflreadpy.load_schedules",
        "reason": (
            "NOT WIRED, and this is the one honest gap in the pin. It reaches nflverse "
            "live for the pre-2018 blend-weight tuning window. It operates on the POLARS "
            "frame directly (filter/select before to_pandas), so routing it through the "
            "pandas pin is a real refactor rather than a call-site swap. It does not "
            "feed data/gold/ -- the blend weights it tunes are already frozen in the "
            "deployed blend artifact -- so no gold column and no gate baseline depends "
            "on it. Named here rather than left for a reader to discover."
        ),
    },
    {
        "loader": "tests/integration/test_audit_freshness.py nflreadpy.load_schedules",
        "reason": (
            "NOT WIRED ON PURPOSE. That audit exists to diff LIVE nflverse against "
            "on-disk silver; reading the pin would make it compare a snapshot with "
            "itself and always pass."
        ),
    },
]


class PinCaptureError(RuntimeError):
    """A snapshot could not be captured faithfully, so nothing was recorded for it."""


def _package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:  # pragma: no cover - the package is a hard dependency
        return "unknown"


def fetch_live(dataset: str, season: int) -> pd.DataFrame:
    """Fetch one season of *dataset* from nflverse, as pandas."""
    import nflreadpy as nfl

    if dataset == "pbp":
        return nfl.load_pbp([season]).to_pandas()
    if dataset == "schedules":
        return nfl.load_schedules([season]).to_pandas()
    if dataset == "depth_charts":
        return nfl.load_depth_charts(season).to_pandas()
    msg = f"Unknown dataset {dataset!r}. Known: {sorted(DATASET_COLUMNS)}."
    raise PinCaptureError(msg)


def narrow(dataset: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Restrict *frame* to the dataset's pinned columns, PRESERVING absence.

    A column named in the allowlist but absent from this season's upstream frame is left
    absent. Materialising it as all-null would hand the builders a column the live path
    never gave them, and ``features/team_form.py`` branches on ``"cpoe" in group.columns``
    -- so inventing a column would change what gets computed.
    """
    columns = DATASET_COLUMNS[dataset]
    if columns is None:
        return frame
    present = [column for column in columns if column in frame.columns]
    return pd.DataFrame(frame.loc[:, present]).copy()


def assert_round_trip_faithful(written: pd.DataFrame, path: Path) -> None:
    """Read *path* back and prove it carries exactly the frame that was written."""
    readback = pd.read_parquet(path)

    if list(readback.columns) != list(written.columns):
        msg = (
            f"Round-trip through '{path}' changed the column set.\n"
            f"  written  {list(written.columns)}\n"
            f"  readback {list(readback.columns)}"
        )
        raise PinCaptureError(msg)

    dtype_moves = [
        f"{column}: {written[column].dtype} -> {readback[column].dtype}"
        for column in written.columns
        if written[column].dtype != readback[column].dtype
    ]
    if dtype_moves:
        msg = (
            f"Round-trip through '{path}' changed {len(dtype_moves)} dtype(s). A pin that "
            "shifts a dtype moves gold values for a reason that is not the data:\n  "
            + "\n  ".join(dtype_moves)
        )
        raise PinCaptureError(msg)

    if not written.equals(readback):
        msg = (
            f"Round-trip through '{path}' changed VALUES. The snapshot is not a faithful "
            "copy of what nflverse returned and was not recorded."
        )
        raise PinCaptureError(msg)


def capture_season(
    dataset: str,
    season: int,
    data_root: Path,
) -> dict:
    """Fetch, narrow, write, verify and describe one pinned season."""
    raw = fetch_live(dataset, season)
    frame = narrow(dataset, raw)

    if frame.empty:
        msg = (
            f"nflverse returned ZERO rows for {dataset} season {season}. An empty pin "
            "would silently starve every builder that reads it; refusing to record one."
        )
        raise PinCaptureError(msg)

    path = save_bronze_snapshot(
        frame,
        table_name=DATASET_TABLE_NAMES[dataset],
        season=season,
        week=0,
        base_path=data_root,
    )
    assert_round_trip_faithful(frame, path)

    entry = {
        "path": path.relative_to(data_root).as_posix(),
        "sha256": digest_file(path),
        "bytes": path.stat().st_size,
        "rows": len(frame),
        "columns": list(frame.columns),
        "upstream_width": int(raw.shape[1]),
        "captured_at_utc": datetime.now(UTC).isoformat(),
    }
    logger.info(
        "Pinned upstream season",
        dataset=dataset,
        season=season,
        rows=entry["rows"],
        columns=len(entry["columns"]),
        path=entry["path"],
    )
    return entry


def _empty_manifest() -> dict:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "source": "nflverse (github.com/nflverse) via nflreadpy",
        "datasets": {},
        "not_pinned": NOT_PINNED,
    }


def capture(
    datasets: dict[str, list[int]],
    *,
    manifest_path: Path,
    data_root: Path,
) -> dict:
    """Capture every requested season and return the updated manifest."""
    manifest = load_manifest(manifest_path) or _empty_manifest()
    manifest["schema_version"] = MANIFEST_SCHEMA_VERSION
    manifest["source"] = "nflverse (github.com/nflverse) via nflreadpy"
    manifest["not_pinned"] = NOT_PINNED
    manifest["captured_at_utc"] = datetime.now(UTC).isoformat()
    manifest["nflreadpy_version"] = _package_version("nflreadpy")
    manifest["pandas_version"] = _package_version("pandas")
    manifest["pyarrow_version"] = _package_version("pyarrow")
    manifest["python_version"] = sys.version.split()[0]

    for dataset, seasons in datasets.items():
        record = manifest["datasets"].setdefault(
            dataset,
            {"loader": DATASET_LOADERS[dataset], "seasons": {}},
        )
        record["loader"] = DATASET_LOADERS[dataset]
        columns = DATASET_COLUMNS[dataset]
        record["column_policy"] = (
            "full frame"
            if columns is None
            else (
                "narrowed to the columns the gold-rebuild consumers read; see "
                "data.upstream_pin.PBP_PINNED_COLUMNS for the per-consumer attribution"
            )
        )
        if columns is not None:
            record["requested_columns"] = list(columns)
        for season in seasons:
            record["seasons"][str(season)] = capture_season(dataset, season, data_root)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def verify(manifest_path: Path, data_root: Path) -> list[str]:
    """Return a list of problems with the pin on disk. Empty means the pin is intact."""
    manifest = load_manifest(manifest_path)
    if manifest is None:
        return [f"no pin manifest at '{manifest_path}'"]

    problems: list[str] = []
    for dataset, record in sorted(manifest.get("datasets", {}).items()):
        for season, entry in sorted(record.get("seasons", {}).items()):
            path = data_root / entry["path"]
            if not path.is_file():
                problems.append(f"{dataset} {season}: MISSING {path}")
                continue
            actual = digest_file(path)
            if actual != entry["sha256"]:
                problems.append(
                    f"{dataset} {season}: DIGEST MISMATCH {path}\n"
                    f"    recorded {entry['sha256']}\n"
                    f"    actual   {actual}"
                )
    return problems


_SEALED_LOCK_FORMAT_NOTE = (
    "JSON, not TOML, and deliberately so. json.dumps(indent=2) is a standard-library "
    "writer that already produces this repository's committed-record style -- it is "
    "exactly how scripts/pin_upstream_snapshot.py writes config/upstream_pin.json. The "
    "config/*.toml verdict records exist only because the standard library has NO TOML "
    "writer, so the hand-pasted block IS the discipline there "
    "(backtest/group_gate.py::render_verdict_toml). The .lock extension names the "
    "CONTRACT -- this file locks the sealed zone -- not the serialisation."
)


def build_sealed_lock(manifest: dict) -> dict:
    """Build the sealed-zone lock document from *manifest*. Pure; writes nothing.

    Only SEALED-zone seasons are locked. A live-zone season grows week by week under
    ``config/upstream_live/<season>.json``; locking it here would make the lock fail
    every week by design and would merge two records whose diffs mean opposite things.

    THE THREE NULL-SEEDED FIELDS ARE DECLARED NOW ON PURPOSE. ``upstream_updated_at``,
    ``upstream_size`` and ``acknowledgement`` are the D32-05 / D32-10 signature slots:
    plan 32-05's sealed revision probe COMPARES against the first two, and plan 32-08's
    owner-attributed run FILLS all three when a divergence is ruled on. ``null`` means
    "not yet observed", which is a different and honest claim from "absent". Declaring
    them here means no later plan has to reshape a file that is already committed and
    already bound by a suite test.
    """
    datasets: dict[str, dict[str, dict]] = {}
    for dataset, record in sorted(manifest.get("datasets", {}).items()):
        seasons: dict[str, dict] = {}
        for season, entry in sorted(
            record.get("seasons", {}).items(), key=lambda item: int(item[0])
        ):
            if zone_for_season(int(season)) != ZONE_SEALED:
                continue
            seasons[str(season)] = {
                "sha256": entry["sha256"],
                "rows": entry["rows"],
                "bytes": entry["bytes"],
                "upstream_updated_at": None,
                "upstream_size": None,
                "acknowledgement": None,
            }
        if seasons:
            datasets[dataset] = seasons

    return {
        "schema_version": SEALED_LOCK_SCHEMA_VERSION,
        "format_note": _SEALED_LOCK_FORMAT_NOTE,
        "sealed_through_season": SEALED_THROUGH_SEASON,
        "generated_from": MANIFEST_PATH.as_posix(),
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "datasets": datasets,
    }


# The three fields a regeneration must CARRY FORWARD rather than re-derive. They are not
# functions of the manifest: they record what was OBSERVED upstream and what a human
# RULED about it.
_CARRIED_FORWARD_FIELDS = ("upstream_updated_at", "upstream_size", "acknowledgement")


def refresh_sealed_lock(manifest_path: Path, lock_path: Path) -> dict:
    """Regenerate the lock at *lock_path* from the manifest at *manifest_path*.

    REGENERATION NEVER ERASES AN ACKNOWLEDGEMENT. The ``sha256``/``rows``/``bytes`` half
    is re-derived from the manifest, but ``upstream_updated_at``, ``upstream_size`` and
    ``acknowledgement`` are carried forward verbatim for every ``(dataset, season)`` that
    survives. An acknowledgement is a committed, attributed ruling (D32-10); silently
    dropping it would re-arm a CRITICAL the owner has already ruled on, and the owner
    would have no way to tell that their ruling had evaporated.

    This function is reachable ONLY through ``--refresh-sealed-lock``, which itself
    requires ``--sealed-rewrite-reason``. A lock that re-derived itself from the manifest
    it is supposed to police -- as a side effect of an ordinary capture, say -- would
    prove nothing at all, because it could never disagree with what it polices.
    """
    manifest = load_manifest(manifest_path)
    if manifest is None:
        msg = (
            f"Cannot regenerate the sealed-zone lock: no pin manifest at "
            f"'{manifest_path}'. The lock is derived from the manifest, so there is "
            "nothing to lock."
        )
        raise PinCaptureError(msg)

    existing = load_sealed_lock(lock_path) or {}
    previous = existing.get("datasets", {})

    lock = build_sealed_lock(manifest)
    for dataset, seasons in lock["datasets"].items():
        for season, entry in seasons.items():
            carried = previous.get(dataset, {}).get(season, {})
            for field in _CARRIED_FORWARD_FIELDS:
                if field in carried:
                    entry[field] = carried[field]

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "Regenerated the sealed-zone lock",
        lock=str(lock_path),
        manifest=str(manifest_path),
        datasets=sorted(lock["datasets"]),
        sealed_pairs=sum(len(seasons) for seasons in lock["datasets"].values()),
    )
    return lock


def build_parser() -> argparse.ArgumentParser:
    summary = (__doc__ or "Capture the upstream pin.").split("\n\n")[0]
    parser = argparse.ArgumentParser(description=summary)
    parser.add_argument(
        "--all",
        action="store_true",
        help=(
            "Capture every dataset over its default span: "
            + ", ".join(
                f"{name} {span[0]}-{span[1]}" for name, span in DEFAULT_SPANS.items()
            )
        ),
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_COLUMNS),
        action="append",
        default=None,
        help="Capture only this dataset (repeatable).",
    )
    parser.add_argument(
        "--seasons",
        nargs=2,
        type=int,
        metavar=("FIRST", "LAST"),
        default=None,
        help="Inclusive season span to capture. Defaults to the dataset's full span.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Re-hash the pinned files and compare them to the manifest. Fetches nothing.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST_PATH,
        help=f"Manifest path (default: {MANIFEST_PATH})",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Data lake root (default: the configured data root)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.data_root is not None:
        data_root = args.data_root
    else:
        from data.upstream_pin import default_data_root

        data_root = default_data_root()

    if args.verify:
        problems = verify(args.manifest, data_root)
        if problems:
            print("UPSTREAM PIN VERIFY FAILED:", file=sys.stderr)
            for problem in problems:
                print(f"  {problem}", file=sys.stderr)
            return 1
        manifest = load_manifest(args.manifest) or {}
        pinned = sum(
            len(record.get("seasons", {}))
            for record in manifest.get("datasets", {}).values()
        )
        print(
            f"UPSTREAM PIN INTACT: {pinned} pinned season file(s) match {args.manifest}"
        )
        return 0

    if not args.all and not args.dataset:
        print(
            "ERROR: pass --all, or --dataset NAME (optionally with --seasons FIRST LAST),"
            " or --verify",
            file=sys.stderr,
        )
        return 2

    names = sorted(DATASET_COLUMNS) if args.all else sorted(set(args.dataset))
    requested: dict[str, list[int]] = {}
    for name in names:
        first, last = args.seasons if args.seasons else DEFAULT_SPANS[name]
        requested[name] = list(range(first, last + 1))

    manifest = capture(
        requested,
        manifest_path=args.manifest,
        data_root=data_root,
    )
    for dataset, record in sorted(manifest["datasets"].items()):
        seasons = sorted(int(season) for season in record["seasons"])
        print(
            f"{dataset}: {len(seasons)} season(s) pinned "
            f"({seasons[0]}-{seasons[-1]}), loader {record['loader']}"
        )
    print(f"Wrote {args.manifest}")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main in tests
    sys.exit(main())
