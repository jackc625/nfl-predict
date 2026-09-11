"""Capture one week of the LIVE nflverse season into the append-only live-zone pin.

This is the live-zone mirror of ``scripts/pin_upstream_snapshot.py``. That tool owns the
SEALED zone: capture a finished season once, never move it. This one owns the season
currently being played: fetch the datasets a gold rebuild reads, write a timestamped
snapshot into ``data/bronze/``, and APPEND a capture record to the committed manifest at
``config/upstream_live/<season>.json``.

The two tools refuse each other's seasons outright, before any fetch. Asking this one for
a sealed season raises ``ZoneWriteRefused`` and names the tool that does own it, because
the failure a mis-aimed capture would cause -- silently rewriting the bytes a published
verdict was measured against -- is not one a warning can undo.

WHAT THE WEEK MEANS
-------------------
``--week`` is the week being PREDICTED, not the last week present in the data. RATIFIED
by the owner on 2026-09-11 (D32-13). A capture labelled week 6 is the capture the week-6
predictions were made from; what it actually contains is recorded separately, as the
manifest's observed ``content_through_week``. Do not compute one from the other: the
Thursday night game that opens each NFL week is played before the Friday 6 PM ET freeze,
so a week-6 capture normally already carries week-6 rows.

USAGE
-----
Capture every dataset for the week being predicted::

    .venv/Scripts/python.exe -m scripts.capture_live_season --season 2026 --week 3

Capture one dataset only::

    .venv/Scripts/python.exe -m scripts.capture_live_season --season 2026 --week 3 \
        --dataset pbp

NOT WIRED INTO THE PIPELINE, DELIBERATELY (D32-04). Phase 32 ships this as a standalone,
fully-tested CLI. Phase 33 wires it as the new FIRST step of the DATA phase, ahead of
``ingest_games``, when it stands up the live run it can actually observe firing. An
unwired capture is a capture nobody runs, so that obligation is carried forward
explicitly rather than assumed.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from data.storage import save_bronze_snapshot
from data.upstream_live import (
    LIVE_MANIFEST_DIR,
    append_capture,
    build_capture_entry,
    empty_live_manifest,
    load_live_manifest,
    write_live_manifest,
)
from data.upstream_pin import (
    DATASET_COLUMNS,
    DATASET_TABLE_NAMES,
    LIVE_ZONE_FIRST_SEASON,
    SEALED_THROUGH_SEASON,
    ZONE_LIVE,
    ZONE_SEALED,
    ZoneWriteRefused,
    default_data_root,
    zone_for_season,
)
from scripts import pin_upstream_snapshot
from scripts.pin_upstream_snapshot import PinCaptureError
from utils import get_logger

logger = get_logger(__name__)

# The exit-code vocabulary, declared once and NEVER renumbered: a caller that learned
# "3 means CRITICAL" from one release must not be silently retrained by the next.
# ``0`` clean, ``1`` a real finding and ``2`` a usage error continue
# ``scripts/pin_upstream_snapshot.py``'s existing codes.
EXIT_OK = 0
EXIT_CAPTURE_FAILED = 1
EXIT_USAGE = 2
# Reached by Plan 32-07's revision detector, not by anything in this plan. Declared now
# so the numbering is fixed before two plans can pick the same value for two meanings.
EXIT_CRITICAL = 3
EXIT_UNKNOWN = 4

# How long to wait for the clock second to advance past a colliding snapshot name, and
# how many times. 15 x 0.2 s = 3 s, which comfortably outlasts the one-second window the
# collision can occupy while still failing fast if something else is wrong.
_COLLISION_POLL_SECONDS = 0.2
_COLLISION_POLL_ATTEMPTS = 15


def _utc_stamp_now() -> str:
    """The UTC stamp ``data.storage.save_bronze_snapshot`` will put in the filename.

    The format string is duplicated from ``data/storage.py:1009`` DELIBERATELY and is the
    only copy in this module, so a test can pin the clock here and get the same second
    the writer is about to use. Making this the single seam is what lets the same-second
    collision be exercised deterministically instead of raced.
    """
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%S")


def _bronze_snapshot_glob(table_name: str, season: int, week: int) -> str:
    """The filename pattern ``save_bronze_snapshot`` writes for this (season, week)."""
    return f"{table_name}_raw_bronze_{season}_W{week:02d}_*.parquet"


def _stamp_of(path: Path) -> str:
    """The UTC stamp segment of a bronze snapshot filename."""
    return path.stem.rsplit("_", 1)[-1]


def _reserve_distinct_bronze_second(
    bronze_dir: Path,
    table_name: str,
    season: int,
    week: int,
) -> None:
    """Wait until the current clock second yields a filename nothing already owns.

    ``data/storage.py::save_bronze_snapshot`` builds its filename from a SECOND-resolution
    UTC stamp, and its own comment names the resulting collision as a known defect that
    was out of scope there. The live zone is the first caller for which it is not
    academic: it passes a real ``week``, and two captures of the same ``(season, week)``
    inside one second would produce the SAME path and the second would silently overwrite
    the first -- breaking ``data/bronze/``'s append-only contract literally, and
    destroying the bytes a published week-N prediction was made from.

    So this polls: while any existing snapshot for this ``(table_name, season, week)``
    carries the stamp :func:`_utc_stamp_now` would produce, sleep and re-check. If the
    second never advances within the budget, REFUSE.

    :class:`scripts.pin_upstream_snapshot.PinCaptureError` is the right family here and
    NOT :class:`data.upstream_pin.UpstreamPinError`. This is a capture-tool failure at top
    level -- it cannot reach a builder, because builders read the manifest and never call
    the capture CLI -- and ``PinCaptureError`` is already what the sealed capture tool
    raises for "the snapshot could not be captured faithfully, so nothing was recorded
    for it". Using the read-refusal family for a write-tool failure would put a
    capture-tool bug in the same bucket as a missing pin.
    """
    pattern = _bronze_snapshot_glob(table_name, season, week)
    for _attempt in range(_COLLISION_POLL_ATTEMPTS):
        stamp = _utc_stamp_now()
        colliding = [
            candidate
            for candidate in sorted(bronze_dir.glob(pattern))
            if _stamp_of(candidate) == stamp
        ]
        if not colliding:
            return
        time.sleep(_COLLISION_POLL_SECONDS)

    stamp = _utc_stamp_now()
    colliding = [
        candidate
        for candidate in sorted(bronze_dir.glob(pattern))
        if _stamp_of(candidate) == stamp
    ]
    msg = (
        f"Refusing to capture {table_name} season {season} week {week}: the bronze "
        f"snapshot name for this clock second is already taken by "
        f"{', '.join(str(path) for path in colliding)}.\n"
        "\n"
        "CAUSE: data/storage.py::save_bronze_snapshot builds its filename from a "
        "SECOND-resolution UTC stamp "
        "(<table>_raw_bronze_<season>_W<week>_<YYYYmmddTHHMMSS>.parquet), so two captures "
        "of the same (season, week) inside one second collide and the second would "
        "OVERWRITE the first. data/bronze/ is append-only by contract: a recorded "
        "snapshot is the evidence a superseded verdict was measured against, so it is "
        "never rewritten.\n"
        "\n"
        f"Nothing was fetched into a file and the live manifest is byte-unchanged. "
        f"RECOVERY: re-run the capture; the next second yields a distinct name:\n"
        f"       .venv/Scripts/python.exe -m scripts.capture_live_season "
        f"--season {season} --week {week}"
    )
    raise PinCaptureError(msg)


def capture_live_dataset(
    dataset: str,
    season: int,
    week: int,
    *,
    data_root: Path,
    manifest_dir: Path,
) -> dict:
    """Fetch, narrow, write, verify and RECORD one live-zone dataset for one week.

    THE ORDER OF THE LAST FOUR STEPS IS THE CONTRACT, not an implementation detail. The
    bronze bytes are written, :func:`scripts.pin_upstream_snapshot.assert_round_trip_faithful`
    proves they are the frame that was fetched, the entry (with its digests) is built from
    them, and ONLY THEN is the capture appended to the manifest and the manifest written.

    The reason is the one ``data.upstream_pin._read_pinned_frame`` gives for its own
    digest-before-parse ordering: a check that arrives after the work is a check nobody
    can afford to trust. An interruption anywhere between the bronze write and the
    manifest write leaves an UNREFERENCED bronze file and a byte-unchanged
    ``config/upstream_live/<season>.json`` -- a recoverable state, because the next run
    simply writes a new snapshot and nothing ever pointed at the orphan. The reverse
    order would leave a manifest entry pointing at bytes nobody verified, which is
    indistinguishable from a faithful capture at read time and is exactly what the digest
    exists to make impossible.
    """
    zone = zone_for_season(season)
    if zone != ZONE_LIVE:
        owner = (
            "scripts.pin_upstream_snapshot, which owns the SEALED zone"
            if zone == ZONE_SEALED
            else "no tool -- that season lies beyond the live zone"
        )
        msg = (
            f"Refusing to write a LIVE capture for season {season}: it is in the "
            f"{zone!r} zone, not {ZONE_LIVE!r}. The live zone is exactly season "
            f"{LIVE_ZONE_FIRST_SEASON}; everything at or before {SEALED_THROUGH_SEASON} "
            f"is sealed and immutable. That season belongs to {owner}.\n"
            "\n"
            "Nothing was fetched and nothing was written. The two zones have opposite "
            "mutability contracts, so a capture aimed at the wrong one is refused "
            "before it can rewrite bytes a published verdict was measured against."
        )
        raise ZoneWriteRefused(msg)

    raw = pin_upstream_snapshot.fetch_live(dataset, season)
    frame = pin_upstream_snapshot.narrow(dataset, raw)

    table_name = DATASET_TABLE_NAMES[dataset]
    bronze_dir = Path(data_root) / "bronze"
    bronze_dir.mkdir(parents=True, exist_ok=True)
    _reserve_distinct_bronze_second(bronze_dir, table_name, season, week)
    existing_paths = set(
        bronze_dir.glob(_bronze_snapshot_glob(table_name, season, week))
    )

    path = save_bronze_snapshot(
        frame,
        table_name=table_name,
        season=season,
        week=week,
        base_path=data_root,
    )
    if path in existing_paths:
        msg = (
            f"The live capture for {dataset} season {season} week {week} wrote OVER an "
            f"existing bronze snapshot at '{path}'. data/bronze/ is append-only by "
            "contract, so those bytes were the evidence some earlier verdict was "
            "measured against and they are now gone.\n"
            "\n"
            "This is the second-resolution filename collision data/storage.py names as a "
            "known defect, reached despite the pre-write reservation. NOTHING was "
            "recorded: the live manifest is byte-unchanged, so no entry claims these "
            "bytes were verified. Investigate before re-running -- a reservation that "
            "passed and a write that collided anyway means the two clocks disagree."
        )
        raise PinCaptureError(msg)

    pin_upstream_snapshot.assert_round_trip_faithful(frame, path)

    entry = build_capture_entry(dataset, season, week, raw, frame, path, data_root)
    manifest = load_live_manifest(season, manifest_dir=manifest_dir) or (
        empty_live_manifest(season)
    )
    manifest = append_capture(manifest, dataset, entry)
    write_live_manifest(manifest, manifest_dir=manifest_dir)

    recorded = manifest["datasets"][dataset]["captures"][-1]
    logger.info(
        "Captured live upstream week",
        dataset=dataset,
        season=season,
        week=week,
        sequence=recorded["sequence"],
        rows=recorded["rows"],
        content_through_week=recorded["content_through_week"],
        path=recorded["path"],
    )
    return recorded


def run_capture(
    datasets: list[str],
    season: int,
    week: int,
    *,
    data_root: Path,
    manifest_dir: Path,
) -> int:
    """Capture every requested dataset for one week. Returns an exit code."""
    for dataset in datasets:
        entry = capture_live_dataset(
            dataset,
            season,
            week,
            data_root=data_root,
            manifest_dir=manifest_dir,
        )
        print(
            f"{dataset}: season {season} week {week} sequence {entry['sequence']} -- "
            f"{entry['rows']} row(s), {len(entry['columns'])} column(s), content "
            f"through week {entry['content_through_week']}, {entry['path']}"
        )
    print(f"Wrote {manifest_dir}/{season}.json")
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    summary = (__doc__ or "Capture one live-season week.").split("\n\n")[0]
    parser = argparse.ArgumentParser(description=summary)
    parser.add_argument(
        "--season",
        type=int,
        required=True,
        help=f"The live season to capture (the live zone is {LIVE_ZONE_FIRST_SEASON}).",
    )
    parser.add_argument(
        "--week",
        type=int,
        required=True,
        help=(
            "The week being PREDICTED -- not the last week present in the data. "
            "The capture records what it actually contains separately."
        ),
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASET_COLUMNS),
        action="append",
        default=None,
        help="Capture only this dataset (repeatable). Default: all of them.",
    )
    parser.add_argument(
        "--manifest-dir",
        type=Path,
        default=LIVE_MANIFEST_DIR,
        help=f"Live manifest directory (default: {LIVE_MANIFEST_DIR})",
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

    data_root = args.data_root if args.data_root is not None else default_data_root()
    datasets = sorted(set(args.dataset)) if args.dataset else sorted(DATASET_COLUMNS)

    try:
        return run_capture(
            datasets,
            args.season,
            args.week,
            data_root=Path(data_root),
            manifest_dir=Path(args.manifest_dir),
        )
    except ZoneWriteRefused as error:
        # A season this tool does not own is an operator mistake, not a capture failure:
        # nothing was fetched and nothing was written.
        print(f"LIVE CAPTURE REFUSED:\n{error}", file=sys.stderr)
        return EXIT_USAGE
    except Exception as error:  # noqa: BLE001 - the CLI boundary turns any failure into a code
        print(f"LIVE CAPTURE FAILED: {error}", file=sys.stderr)
        return EXIT_CAPTURE_FAILED


if __name__ == "__main__":  # pragma: no cover - exercised through main in tests
    sys.exit(main())
