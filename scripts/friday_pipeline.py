#!/usr/bin/env python3
"""Unified Friday Pipeline Orchestrator.

Replaces both friday_data_update.py and friday_predictions_run.py with a single
process that runs all 18 data-to-prediction steps in sequence via direct Python
imports.

Usage:
    python scripts/friday_pipeline.py                    # Full pipeline
    python scripts/friday_pipeline.py --data-only        # Data phase only
    python scripts/friday_pipeline.py --predictions-only # Predictions phase only
    python scripts/friday_pipeline.py --dry-run --force  # List steps, bypass checks
"""

import argparse
import sys
from datetime import datetime, timedelta

from utils.date_utils import (
    ET,
    get_current_nfl_week,
    get_nfl_season_start,
)
from utils.logging_config import get_logger

logger = get_logger(__name__)

# Season window matches StalenessGate.check_season: season_start + 22 weeks
# (regular season + playoffs through the Super Bowl). Kept here so the D-06
# offseason no-op short-circuit and the staleness gate stay in lockstep.
_SEASON_WINDOW_WEEKS = 22


def _is_offseason(now: datetime | None = None) -> bool:
    """Return True if *now* falls outside the current NFL season window.

    Mirrors ``pipeline.staleness.StalenessGate.check_season`` so the CLI
    short-circuit and the staleness gate agree on what "offseason" means:
    the window is ``[season_start, season_start + 22 weeks]`` for the season
    resolved by ``get_current_nfl_week()``.

    Args:
        now: Reference time (defaults to ``datetime.now(ET)``).

    Returns:
        True when *now* is before the season start or after the season end.
    """
    if now is None:
        now = datetime.now(ET)

    season, _week = get_current_nfl_week()
    season_start = get_nfl_season_start(season)
    season_end = season_start + timedelta(weeks=_SEASON_WINDOW_WEEKS)

    return now < season_start or now > season_end


def main() -> int:
    """CLI entry point for the unified Friday pipeline.

    Returns:
        Exit code (0 for success, 1 for failure).
    """
    parser = argparse.ArgumentParser(
        description="Unified Friday NFL prediction pipeline orchestrator"
    )

    # Phase flags (mutually exclusive)
    phase_group = parser.add_mutually_exclusive_group()
    phase_group.add_argument(
        "--data-only",
        action="store_true",
        help="Run data ingestion phase only",
    )
    phase_group.add_argument(
        "--predictions-only",
        action="store_true",
        help="Run predictions phase only",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show pipeline steps without executing",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Bypass pre-flight staleness/season checks",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level",
    )

    args = parser.parse_args()

    # Determine mode from flags
    if args.data_only:
        mode = "data-only"
    elif args.predictions_only:
        mode = "predictions-only"
    else:
        mode = "full"

    # D-06: offseason no-op short-circuit. A live (scheduled, unforced) Friday
    # run during the offseason would otherwise hit the staleness season gate,
    # which sets status="failed", fires a CRITICAL "Pipeline Failed" alert, and
    # raises -- a false alarm that erodes trust (crying wolf). Short-circuit to a
    # clean exit-0 INFO no-op BEFORE constructing FridayPipeline so no orchestrator
    # alert path is reached. --force deliberately bypasses this so the operator can
    # still run the pipeline out of season (e.g. a one-time forced run against a
    # completed-week stand-in).
    if not args.force and _is_offseason():
        logger.info("Offseason no-op -- pipeline skipped (use --force to run anyway)")
        return 0

    try:
        from pipeline.orchestrator import FridayPipeline

        pipeline = FridayPipeline(force=args.force, mode=mode)

        if args.dry_run:
            step_list = pipeline.dry_run()
            logger.info("DRY RUN -- pipeline steps that would execute:")
            for i, step_desc in enumerate(step_list, 1):
                logger.info(f"  {i}. {step_desc}")
            print(f"Dry run: {len(step_list)} steps would execute.")
            return 0

        log = pipeline.run()

        if log.status in ("success", "degraded"):
            logger.info(
                "Pipeline finished",
                status=log.status,
                duration_ms=round(log.total_duration_ms, 1),
            )
            return 0

        logger.error("Pipeline failed", status=log.status, error=log.error)
        return 1

    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        return 130
    except Exception as exc:  # noqa: BLE001 -- top-level CLI must catch all
        logger.error("Pipeline error", error=str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
