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

Capture or extend one dataset with seasons it does not already cover::

    .venv/Scripts/python.exe -m scripts.pin_upstream_snapshot --dataset pbp --seasons 2001 2001

Re-verify an existing pin's bytes against the manifest without fetching anything::

    .venv/Scripts/python.exe -m scripts.pin_upstream_snapshot --verify

Rewrite an ALREADY-PINNED sealed season, on the record::

    .venv/Scripts/python.exe -m scripts.pin_upstream_snapshot --dataset pbp \
        --seasons 2024 2024 --allow-sealed-rewrite --sealed-rewrite-reason "<why>"

Regenerate the committed sealed-zone lock after a deliberate sealed change::

    .venv/Scripts/python.exe -m scripts.pin_upstream_snapshot --refresh-sealed-lock \
        --sealed-rewrite-reason "<why>"

WHICH SEASONS THIS TOOL OWNS. It owns the SEALED zone only -- seasons at or before
``data.upstream_pin.SEALED_THROUGH_SEASON``. A live-zone season is refused outright and
pointed at ``scripts/capture_live_season.py``, which captures week by week into
``config/upstream_live/<season>.json``. The refusal is raised BEFORE any fetch, so a
mis-aimed run makes no network call and leaves the manifest byte-identical.

Capturing a season that is already pinned REPLACES that season's manifest entry and
writes a NEW timestamped bronze file. For a SEALED season that is exactly the silent
rewrite the zone exists to prevent, so it is now REFUSED unless the run carries both
``--allow-sealed-rewrite`` and a non-blank ``--sealed-rewrite-reason``; the reason is
emitted to stderr and to the structured log. When a rewrite does go ahead, the previous
bronze file is left on disk, because ``data/bronze/`` is append-only by contract and
deleting a recorded snapshot would destroy the evidence a superseded verdict was
measured against.
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
    LIVE_MANIFEST_DIR_TEXT,
    MANIFEST_PATH,
    MANIFEST_SCHEMA_VERSION,
    SEALED_LOCK_PATH,
    SEALED_LOCK_SCHEMA_VERSION,
    SEALED_THROUGH_SEASON,
    ZONE_LIVE,
    ZONE_SEALED,
    ZoneWriteRefused,
    digest_file,
    load_manifest,
    load_sealed_lock,
    pinned_seasons,
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
    # The blend-tuning live schedule fetch (models/blending_data.py) that used to sit here as
    # "the one honest gap in the pin" was DELETED by Plan 33.2-24 (D33.2-03 / D33.2-10): the
    # blend is now tuned on the owned odds_timeline, already frozen on disk, and nothing in
    # that module reaches nflverse. The entry is gone because the gap is gone; the sealed
    # config/upstream_pin.json keeps it as the record of what was true when it was captured.
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


# The concrete incident every sealed-zone refusal cites. A refusal that only says "this
# is not allowed" teaches nobody why, and a rule nobody understands is a rule that gets
# routed around the first time it is inconvenient.
_WHY_THE_SEALED_ZONE_EXISTS = (
    "A live nflverse fetch is what moved sixteen opponent-adjusted and QB columns from "
    "season 2020 onward between the 2026-08-22 and 2026-09-04 gold builds, INSIDE the "
    "protected 2021-2024 holdout the deploy gate's frozen baseline was measured "
    "against. The 2026-08-22 gold is unrecoverable. Sealed bytes are what make that "
    "class of loss impossible to repeat, so replacing one is never incidental."
)

_PIN_CLI = ".venv/Scripts/python.exe -m scripts.pin_upstream_snapshot"
_LIVE_CLI = ".venv/Scripts/python.exe -m scripts.capture_live_season"


def _covered_text(dataset: str, manifest: dict | None) -> str:
    covered = pinned_seasons(dataset, manifest)
    return f"{covered[0]}-{covered[-1]}" if covered else "nothing (no pin captured)"


def assert_write_allowed(
    datasets: dict[str, list[int]],
    manifest: dict | None,
    *,
    allow_sealed_rewrite: bool,
    sealed_rewrite_reason: str | None,
) -> None:
    """Refuse a write aimed at the wrong zone, BEFORE anything is fetched or written.

    Called from :func:`capture` ahead of the season loop and ahead of every
    ``fetch_live``, for the reason ``data.upstream_pin._read_pinned_frame`` states about
    its own digest ordering: a check that arrives after the work is a check nobody can
    afford to trust. A refused run therefore makes NO network call and leaves
    ``config/upstream_pin.json`` byte-identical.

    The rule, per requested ``(dataset, season)``:

    * LIVE zone -- ALWAYS refused, with no override. The sealed manifest is simply not
      where a live season goes: the two records have opposite mutability contracts, so a
      diff on one is a red flag and a diff on the other is expected. This is the exact
      mirror of the refusal ``scripts/capture_live_season.py`` raises for a sealed
      season, and the pair of them is what makes the boundary un-crossable from BOTH
      sides.
    * Beyond the live zone -- refused. No zone owns those seasons yet.
    * SEALED and already pinned -- refused unless ``allow_sealed_rewrite`` is set AND
      ``sealed_rewrite_reason`` is non-blank. An unattributed override is not an
      override; it is the same silent rewrite with an extra flag on it.
    * SEALED and NOT already pinned -- allowed with no flag. Extending the pin is the
      ordinary use of this tool.

    Every refusal raises :class:`data.upstream_pin.ZoneWriteRefused` and NOT
    :class:`PinCaptureError`. ``PinCaptureError`` is a ``RuntimeError``, which every
    wired call site catches and converts into an empty frame; a refusal that landed in
    one of those handlers would become a silently degraded gold matrix.

    WHERE THE ATTRIBUTION GOES. An accepted override is recorded in the structured log
    line and in the commit message -- deliberately NOT as a new field on the manifest's
    season entry. ``tests/unit/test_upstream_pin.py::
    TestTheCommittedManifestIsTheProvenanceRecord`` asserts that per-entry shape, and
    widening it here would break the sealed zone's own provenance test for a reason that
    has nothing to do with provenance.
    """
    reason = (sealed_rewrite_reason or "").strip()
    if allow_sealed_rewrite and not reason:
        msg = (
            "Refusing --allow-sealed-rewrite with no --sealed-rewrite-reason.\n"
            "\n"
            "An unattributed override is not an override. It is the same silent rewrite "
            "the sealed zone exists to prevent, with one extra flag on it, and six "
            "months later nobody can say why a sealed season moved.\n"
            "\n"
            f"{_WHY_THE_SEALED_ZONE_EXISTS}\n"
            "\n"
            "Do ONE of these, deliberately:\n"
            "  1. Say why, on the record, and the rewrite proceeds:\n"
            f"       {_PIN_CLI} --dataset <D> --seasons <S> <S> "
            '--allow-sealed-rewrite --sealed-rewrite-reason "<why>"\n'
            "  2. Drop the override and extend the pin with seasons it does not "
            "already cover instead."
        )
        raise ZoneWriteRefused(msg)

    rewrites: dict[str, list[int]] = {}
    for dataset, seasons in sorted(datasets.items()):
        covered = set(pinned_seasons(dataset, manifest))
        for season in sorted(seasons):
            zone = zone_for_season(season)
            if zone == ZONE_LIVE:
                raise ZoneWriteRefused(_live_zone_refusal(dataset, season, manifest))
            if season in covered and not allow_sealed_rewrite:
                raise ZoneWriteRefused(
                    _sealed_rewrite_refusal(dataset, season, manifest)
                )
            if season in covered:
                rewrites.setdefault(dataset, []).append(season)

    for dataset, seasons in sorted(rewrites.items()):
        # Warn where a human will see it first, then log where the run record keeps it --
        # the same order data.upstream_pin uses for the live-fetch bypass.
        print(
            f"SEALED REWRITE ALLOWED: {dataset} season(s) "
            f"{', '.join(str(season) for season in seasons)} will REPLACE their pinned "
            f"manifest entries.\n  reason: {reason}",
            file=sys.stderr,
        )
        logger.warning(
            "Sealed pin rewrite allowed by explicit override",
            dataset=dataset,
            seasons=seasons,
            reason=reason,
            operator=None,
        )


def _live_zone_refusal(dataset: str, season: int, manifest: dict | None) -> str:
    return (
        f"Refusing to capture {dataset} season {season} into the SEALED pin. "
        f"{MANIFEST_PATH} covers {_covered_text(dataset, manifest)} and owns seasons at "
        f"or before {SEALED_THROUGH_SEASON} only; {season} is in the LIVE zone.\n"
        "\n"
        "There is NO override for this direction. The sealed manifest and the live "
        f"manifest ({LIVE_MANIFEST_DIR_TEXT}/{season}.json) have opposite mutability "
        "contracts -- a diff on the sealed record is always a red flag, a diff on the "
        "live record is always expected -- so a single file holding both would have "
        "diffs that mean opposite things and could never be read as evidence of "
        "anything.\n"
        "\n"
        f"{_WHY_THE_SEALED_ZONE_EXISTS}\n"
        "\n"
        "Do ONE of these, deliberately:\n"
        f"  1. Capture the LIVE season one week at a time (writes "
        f"{LIVE_MANIFEST_DIR_TEXT}/{season}.json and a timestamped snapshot under "
        "data/bronze/). <W> is the week being PREDICTED, not the last week present in "
        "the data:\n"
        f"       {_LIVE_CLI} --season {season} --week <W> --dataset {dataset}\n"
        f"  2. If season {season} has genuinely ENDED and you mean to SEAL it, that is a "
        "one-way promotion, not a capture: it is a human edit to "
        "data.upstream_pin.SEALED_THROUGH_SEASON, made once the season is over. "
        "scripts/seal_season.py is deliberately NOT built (D32-02) -- it cannot be "
        "exercised against a real live zone until the live season actually ends."
    )


def _sealed_rewrite_refusal(dataset: str, season: int, manifest: dict | None) -> str:
    return (
        f"Refusing to REWRITE the sealed pin for {dataset} season {season}. It is "
        f"already pinned in {MANIFEST_PATH}, which covers "
        f"{_covered_text(dataset, manifest)}.\n"
        "\n"
        "Capturing an already-pinned season REPLACES its manifest entry and writes a "
        "NEW timestamped bronze file, orphaning the recorded one. That is the right "
        "behaviour for EXTENDING a pin and the wrong behaviour for a SEALED season, "
        "whose bytes are immutable by definition.\n"
        "\n"
        f"{_WHY_THE_SEALED_ZONE_EXISTS}\n"
        "\n"
        "Do ONE of these, deliberately:\n"
        "  1. Rewrite it on the record, saying why. The reason is emitted to the log "
        "and to stderr, and belongs in the commit message too:\n"
        f"       {_PIN_CLI} --dataset {dataset} --seasons {season} {season} "
        '--allow-sealed-rewrite --sealed-rewrite-reason "<why this sealed season must '
        'be re-captured>"\n'
        "  2. Check the pin you already have instead of replacing it. This fetches "
        "nothing:\n"
        f"       {_PIN_CLI} --verify"
    )


def capture(
    datasets: dict[str, list[int]],
    *,
    manifest_path: Path,
    data_root: Path,
    allow_sealed_rewrite: bool = False,
    sealed_rewrite_reason: str | None = None,
) -> dict:
    """Capture every requested season and return the updated manifest.

    The zone write gate runs FIRST -- before the season loop and before any fetch -- so
    a refused run costs nothing and changes nothing. See :func:`assert_write_allowed`.
    """
    recorded = load_manifest(manifest_path)
    assert_write_allowed(
        datasets,
        recorded,
        allow_sealed_rewrite=allow_sealed_rewrite,
        sealed_rewrite_reason=sealed_rewrite_reason,
    )
    manifest = recorded or _empty_manifest()
    # A capture changes the entries of the seasons it captures and NOTHING ELSE on the record.
    # Every top-level field already recorded -- the original capture's time and versions, and
    # the not_pinned list as it stood then -- is kept; only a field the record lacks is filled.
    # Overwriting them made a two-season re-pin rewrite the whole pin's provenance (2026-09-30).
    top_level = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "source": "nflverse (github.com/nflverse) via nflreadpy",
        "not_pinned": NOT_PINNED,
        "captured_at_utc": datetime.now(UTC).isoformat(),
        "nflreadpy_version": _package_version("nflreadpy"),
        "pandas_version": _package_version("pandas"),
        "pyarrow_version": _package_version("pyarrow"),
        "python_version": sys.version.split()[0],
    }
    for field, value in top_level.items():
        manifest.setdefault(field, value)

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
    # A top-level field the rebuild does not derive -- the signature seeding record
    # (``signatures_seeded_at_utc`` / ``signatures_seeded_by``, data.sealed_probe) -- is carried
    # forward too, for the same reason: it records what was done, not what the manifest says.
    for field, value in existing.items():
        lock.setdefault(field, value)

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
    parser.add_argument(
        "--allow-sealed-rewrite",
        action="store_true",
        help=(
            "Permit REPLACING an already-pinned sealed season's manifest entry. "
            "Requires --sealed-rewrite-reason; an unattributed override is refused."
        ),
    )
    parser.add_argument(
        "--sealed-rewrite-reason",
        type=str,
        default=None,
        help=(
            "Why a sealed record is being rewritten. Emitted to stderr and to the "
            "structured log, and it belongs in the commit message too."
        ),
    )
    parser.add_argument(
        "--sealed-lock",
        type=Path,
        default=SEALED_LOCK_PATH,
        help=f"Sealed-zone lock path (default: {SEALED_LOCK_PATH})",
    )
    parser.add_argument(
        "--refresh-sealed-lock",
        action="store_true",
        help=(
            "Regenerate the sealed-zone lock from the manifest and exit. Fetches "
            "nothing. Requires --sealed-rewrite-reason: a lock that re-derived itself "
            "from the record it polices -- as a side effect of an ordinary capture, "
            "say -- could never disagree with it, and would prove nothing."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.refresh_sealed_lock:
        if not (args.sealed_rewrite_reason or "").strip():
            print(
                "ERROR: --refresh-sealed-lock requires --sealed-rewrite-reason. "
                "Regenerating the lock is a deliberate, attributed act: a lock that "
                "re-derives itself from the record it polices proves nothing.",
                file=sys.stderr,
            )
            return 2
        lock = refresh_sealed_lock(args.manifest, args.sealed_lock)
        pairs = sum(len(seasons) for seasons in lock["datasets"].values())
        print(
            f"Wrote {args.sealed_lock}: {pairs} sealed (dataset, season) pair(s) locked "
            f"from {args.manifest}\n  reason: {args.sealed_rewrite_reason.strip()}"
        )
        return 0

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

    try:
        manifest = capture(
            requested,
            manifest_path=args.manifest,
            data_root=data_root,
            allow_sealed_rewrite=args.allow_sealed_rewrite,
            sealed_rewrite_reason=args.sealed_rewrite_reason,
        )
    except ZoneWriteRefused as refusal:
        # A finding, not a usage error: the flags parsed fine and the request was
        # understood exactly -- it is the WRITE that is refused. Exit 1, matching
        # --verify's "the pin has a problem" code, so an external check can act on it.
        print("SEALED ZONE WRITE REFUSED:", file=sys.stderr)
        print(str(refusal), file=sys.stderr)
        return 1
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
