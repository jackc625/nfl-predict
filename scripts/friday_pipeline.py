#!/usr/bin/env python3
"""Unified Friday Pipeline Orchestrator.

Replaces both friday_data_update.py and friday_predictions_run.py with a single
process that runs all 19 data-to-prediction steps in sequence via direct Python
imports.

Usage:
    python scripts/friday_pipeline.py                    # Full pipeline
    python scripts/friday_pipeline.py --data-only        # Data phase only
    python scripts/friday_pipeline.py --predictions-only # Predictions phase only
    python scripts/friday_pipeline.py --dry-run --force  # List steps, bypass checks
"""

import argparse
import sys
from datetime import datetime

from utils.current_slate import SlateResolutionError, resolve_current_slate
from utils.date_utils import ET
from utils.logging_config import get_logger

logger = get_logger(__name__)


def _is_offseason(now: datetime | None = None) -> bool:
    """Return True if the recorded schedule has no slate open at *now*.

    Reads the SAME schedule-keyed resolver as ``pipeline.staleness.StalenessGate.
    check_season``, so the CLI short-circuit and the staleness gate cannot disagree on
    what "offseason" means: it is the resolver's offseason branch -- before a season's
    opener lock day, or after a season whose Super Bowl is recorded with the next
    season not yet begun.

    Step 24c of Plan 33.2-24 replaced the calendar window this used to compute,
    ``[computed opener Thursday, + 22 weeks]`` for the season the retired week count
    resolved. That window was wrong at both ends: it opened AFTER the 2026 opener's
    Tuesday lock (the Wednesday opener kicked off before the computed Thursday), and
    it closed on 2026-02-05, three days BEFORE the 2026-02-08 Super Bowl. It also
    ignored its own ``now`` argument when resolving the season.

    Args:
        now: Reference time (defaults to ``datetime.now(ET)``).

    Returns:
        True when no slate is open.

    Raises:
        utils.current_slate.SlateResolutionError: the schedule cannot say (for example a
            season at hand whose schedule is not recorded). Never read as "offseason".
    """
    if now is None:
        now = datetime.now(ET)
    return not resolve_current_slate(now).in_season


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
        help=(
            "Bypass pre-flight staleness/season checks and the offseason no-op "
            "short-circuit."
        ),
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
    #
    # --dry-run also bypasses the short-circuit (WR-04): it is a read-only inspection
    # tool that lists the steps that WOULD run and executes nothing, so it should work
    # year-round. The crying-wolf protection (D-06) is intact because the scheduled task
    # never passes --dry-run; only a human operator inspecting steps out of season does.
    #
    # A schedule that cannot name the slate (a season at hand with no recorded schedule, a
    # store stale beyond one playoff round) is NOT the offseason: it fails the run by name,
    # with the command that records the schedule, before any orchestrator exists -- so it is
    # an honest exit 1, not a CRITICAL alert, and never a silent no-op that would skip a live
    # slate (step 24c).
    if not args.force and not args.dry_run:
        try:
            offseason = _is_offseason()
        except SlateResolutionError as refusal:
            logger.error(
                "The recorded schedule cannot name the current slate -- run refused",
                error=str(refusal),
            )
            return 1
        if offseason:
            logger.info(
                "Offseason no-op -- pipeline skipped (use --force to run anyway)"
            )
            return 0

    try:
        from pipeline.orchestrator import FridayPipeline
        from pipeline.steps import RunStatus

        pipeline = FridayPipeline(force=args.force, mode=mode)

        if args.dry_run:
            step_list = pipeline.dry_run()
            logger.info("DRY RUN -- pipeline steps that would execute:")
            for i, step_desc in enumerate(step_list, 1):
                logger.info(f"  {i}. {step_desc}")
            print(f"Dry run: {len(step_list)} steps would execute.")
            return 0

        log = pipeline.run()

        # THE COMPLETED-RUN SET. A run that FINISHED WITH SKIPS completed: every clean game was
        # predicted and the dropped ones are recorded durably (config/skip_records.jsonl), which
        # is exactly the outcome D33.2-05's live half defines as correct -- so it exits 0, and
        # the scheduled task does not report a failure for it. It is NOT reported in the words of
        # a clean run: its own line names the games it left out.
        if log.status == RunStatus.FINISHED_WITH_SKIPS.value:
            logger.warning(
                "Pipeline finished with skipped games",
                status=log.status,
                skipped_games=list(log.skipped_games),
                duration_ms=round(log.total_duration_ms, 1),
            )
            return 0

        if log.status in (RunStatus.SUCCESS.value, "degraded"):
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
