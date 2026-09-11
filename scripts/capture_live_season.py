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


def capture_live_dataset(
    dataset: str,
    season: int,
    week: int,
    *,
    data_root: Path,
    manifest_dir: Path,
) -> dict:
    """Fetch, narrow, write, verify and RECORD one live-zone dataset for one week."""
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

    path = save_bronze_snapshot(
        frame,
        table_name=DATASET_TABLE_NAMES[dataset],
        season=season,
        week=week,
        base_path=data_root,
    )
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
