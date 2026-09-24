#!/usr/bin/env python3
"""The DAILY lock-time run: predict tomorrow's games, finishing before today's 6 PM ET lock.

Plan 33.2-27 (D33.2-01, D33.2-18 as amended by the owner on 2026-09-23). It replaces the weekly
Friday run as the scheduled entry point; ``scripts/friday_pipeline.py`` stays as a manual tool.

WHAT ONE RUN DOES
-----------------
0. BEFORE ANY REQUEST: if the run started at or after 18:00 ET on its run date, every one of
   tomorrow's games has already locked, so the run records a named refusal and exits having made
   ZERO network requests. A missed day is never back-filled.
1. Refreshes the schedule: captures what nflverse serves now and ingests the season from that
   capture, so results, moved games and each new playoff round arrive with no manual step.
2. Selects tomorrow's games -- the games whose lock is today's 18:00 ET. None: a no-op record.
3. Runs ``pipeline.daily_steps``: collection, the FULL-HISTORY gold build, then prediction and
   emission, all before the lock (owner ruling "Finish before 6 PM").

WHY THE FULL HISTORY IS REBUILT, AND WHY THAT CANNOT DENY PREDICTIONS TO TONIGHT'S GAMES
-----------------------------------------------------------------------------------------
Owner ruling 2026-09-23, "Rebuild everything nightly": a one-season build diverges from the full
build on 139 of 186 numeric columns (scaling, imputation and clipping are fitted on prior
seasons), so only the full build is the real one. RESEARCH pitfall P10 warned that a full rebuild
lets one defective historical row stop the whole build and deny predictions to that night's clean
games. The live-skip rule answers it: the gate refuses per game, and a live run drops only the
games a refusal names, so a bad historical row skips that one game, never tomorrow's slate.

``--dry-run`` runs step 1 (without the nflverse capture, which reads back its own write) and the
COLLECTION stage under ``data.write_sink.RecordingSink``: the real collection code runs and nothing
is written. It stops before the build, because a build cannot read captures that were never
saved -- the build and predictions are proved by a real run.

Usage:
    uv run python -m scripts.daily_lock_pipeline                    # today's run
    uv run python -m scripts.daily_lock_pipeline --date 2026-09-26  # run as of that ET day
    uv run python -m scripts.daily_lock_pipeline --dry-run          # collection only, no writes
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from backtest.weekly_bet_list import _with_gameday, select_games_for_decision_instant
from data.write_sink import (
    ProductionSink,
    RecordingSink,
    active_sink,
    append_jsonl,
)
from pipeline.daily_steps import (
    COLLECTION_STEP_NAMES,
    DECISION_TIME_BRANCH,
    DailySlate,
    build_daily_step_registry,
    slate_lock,
)
from utils.date_utils import ET, get_current_nfl_season
from utils.logging_config import get_logger

if TYPE_CHECKING:
    import pandas as pd

logger = get_logger(__name__)

#: One line per run that predicted nothing (no games tomorrow, or the lock had passed).
DAILY_RUN_RECORDS = Path("logs/daily_lock_runs.jsonl")

#: Collection steps a no-write run cannot execute, each with the reason. The live capture proves
#: its bronze bytes round-trip by READING BACK the file it just wrote and digests it into the
#: manifest; under ``RecordingSink`` that file was never written, so there is nothing to read.
DRY_RUN_UNSUPPORTED_STEPS: dict[str, str] = {
    "capture_live_season": "reads back the bronze file it writes, which a no-write run never has",
}


def _recorded_tomorrow_ids(run_date_et: date) -> list[str]:
    """Tomorrow's games per the RECORDED schedule, read locally. Empty if none can be read."""
    from utils.current_slate import SlateResolutionError, load_recorded_schedule

    try:
        schedule = load_recorded_schedule()
    except SlateResolutionError:
        return []
    tomorrow = run_date_et + timedelta(days=1)
    return sorted(schedule.loc[schedule["et_day"] == tomorrow, "game_id"].astype(str))


def _record_no_prediction(run_date_et: date, outcome: str, game_ids: list[str]) -> None:
    """Append the day's no-prediction record through the write sink."""
    append_jsonl(
        DAILY_RUN_RECORDS,
        {
            "run_date_et": run_date_et.isoformat(),
            "outcome": outcome,
            "game_ids": game_ids,
            "recorded_at": datetime.now(UTC).isoformat(),
        },
        kind="daily_run_record",
    )


def _refresh_schedule(run_date_et: date, *, dry_run: bool) -> int:
    """Capture what nflverse serves now, ingest the schedule from it, and return the season.

    The schedule ingest reads the LIVE-ZONE capture, never the network directly, so the capture
    must come first -- the Friday registry's order. Daily, this is what picks up results, moved
    games and the next playoff round with no manual step. A no-write run cannot capture (see
    :data:`DRY_RUN_UNSUPPORTED_STEPS`), so it ingests from the latest existing capture.
    """
    from pipeline.steps import step_capture_live_season
    from scripts.ingest_games import GameDataIngester

    if dry_run:
        print(
            "DRY_RUN_SKIPPED capture_live_season: "
            f"{DRY_RUN_UNSUPPORTED_STEPS['capture_live_season']}"
        )
    else:
        step_capture_live_season()
    tomorrow_noon = datetime.combine(
        run_date_et + timedelta(days=1), time(12), tzinfo=ET
    )
    season = get_current_nfl_season(tomorrow_noon)
    GameDataIngester().ingest_games(seasons=[season])
    return season


def select_slate(
    schedule: pd.DataFrame, run_date_et: date, *, decided_at: datetime
) -> DailySlate:
    """The games whose lock is today's 18:00 ET -- tomorrow's games -- as a :class:`DailySlate`.

    Selection goes through ``backtest.weekly_bet_list.select_games_for_decision_instant``, the
    one lock-instant selector, with *decided_at* as the decision instant (Plan 33.2-02: "the
    instant the inputs were captured, passed in explicitly, never now"). It raises
    ``LockPassedError`` when *decided_at* is after the lock.
    """
    lock = slate_lock(run_date_et)
    selected = select_games_for_decision_instant(
        _with_gameday(schedule), lock, decided_at=decided_at
    )
    wanted = set(selected["game_id"].astype(str))
    slate_games = schedule.loc[schedule["game_id"].astype(str).isin(wanted)]
    return DailySlate(
        run_date_et=run_date_et,
        lock=lock,
        schedule=slate_games.reset_index(drop=True),
    )


def _require_slate_is_current_week(slate: DailySlate, start: datetime) -> None:
    """The steps resolve the week from the schedule at the run's clock; it must be the slate's.

    MEASURED 2026-09-24 over every 2018-2026 game day: resolving at noon ET on the day before
    always names that day's week, so this never fires on a real schedule. If it ever did, the
    capture and build steps would work on another week than the one predicted -- refused.
    """
    from utils.current_slate import resolve_current_slate

    current = resolve_current_slate(start.astimezone(ET))
    if (current.season, current.week) != (slate.season, slate.week):
        msg = (
            f"tomorrow's games are {slate.season} week {slate.week}, but the schedule names "
            f"{current.season} week {current.week} as the current slate at {start.isoformat()}; "
            "the run would capture and build a different week from the one it predicts."
        )
        raise RuntimeError(msg)


def _print_contract(
    sink: ProductionSink | RecordingSink,
    *,
    dry_run: bool,
    lock_passed: int,
    next_day_games: int,
) -> None:
    """The output lines Plan 33.2-28's rehearsal keys on."""
    intended = len(sink.intended_writes) if isinstance(sink, RecordingSink) else 0
    print(f"DRY_RUN= {dry_run}")
    print(
        f"WRITES= {sink.production_write_count if isinstance(sink, RecordingSink) else 'n/a'}"
    )
    print(f"INTENDED_WRITES= {intended}")
    print(f"LOCK_PASSED_BEFORE_COLLECTION= {lock_passed}")
    print(f"DECISION_TIME_BRANCH= {DECISION_TIME_BRANCH}")
    print(f"NEXT_DAY_GAMES= {next_day_games}")
    print("BUILD_SCOPE= full-history (owner ruling 2026-09-23)")


def _run_collection_only(slate: DailySlate) -> None:
    """The dry run: the collection stage's real code, in order, under the caller's sink."""
    for step in build_daily_step_registry(slate):
        if step.name not in COLLECTION_STEP_NAMES:
            break
        logger.info("Dry-run collection step", step=step.name)
        step.callable()


def _report_skips(run_id: str) -> None:
    """Name every game this run skipped, with its recorded reason."""
    from pipeline.skip_log import read_skip_records

    for record in read_skip_records():
        if record.get("run_id") == run_id:
            print(
                f"SKIPPED {record['game_id']}: {record['reason']} "
                f"(source {record['source']})"
            )


def run_daily(run_date_et: date, *, start: datetime, dry_run: bool) -> int:
    """One daily run as of *run_date_et*, started at *start*. Returns the exit code."""
    lock = slate_lock(run_date_et)
    sink: ProductionSink | RecordingSink = (
        RecordingSink() if dry_run else ProductionSink()
    )

    with active_sink(sink):
        # 0. The passed-lock refusal, BEFORE any request.
        if start >= lock:
            recorded = _recorded_tomorrow_ids(run_date_et)
            logger.warning(
                "Tomorrow's lock has passed; nothing collected or predicted",
                lock=lock.isoformat(),
                start=start.isoformat(),
                games=recorded,
            )
            _record_no_prediction(run_date_et, "lock_passed", recorded)
            _print_contract(
                sink,
                dry_run=dry_run,
                lock_passed=len(recorded),
                next_day_games=len(recorded),
            )
            return 0

        # 1-2. Refresh the schedule, then select tomorrow's games.
        season = _refresh_schedule(run_date_et, dry_run=dry_run)
        from data.storage import load_dataframe

        games = load_dataframe("games", layer="silver")
        slate = select_slate(
            games.loc[games["season"] == season], run_date_et, decided_at=start
        )
        if slate.schedule.empty:
            logger.info("No games tomorrow; no-op", run_date_et=run_date_et.isoformat())
            _record_no_prediction(run_date_et, "no_games", [])
            _print_contract(sink, dry_run=dry_run, lock_passed=0, next_day_games=0)
            return 0
        _require_slate_is_current_week(slate, start)

        if dry_run:
            _run_collection_only(slate)
            _print_contract(
                sink,
                dry_run=True,
                lock_passed=0,
                next_day_games=len(slate.game_ids),
            )
            return 0

        # 3. The real run.
        from pipeline.orchestrator import FridayPipeline
        from pipeline.steps import RunStatus

        pipeline = FridayPipeline(steps=build_daily_step_registry(slate))
        log = pipeline.run()
        _print_contract(
            sink, dry_run=False, lock_passed=0, next_day_games=len(slate.game_ids)
        )
        _report_skips(log.start_time)
        for game_id, reason in sorted(slate.weather_unknown.items()):
            print(f"WEATHER_UNKNOWN {game_id}: {reason}")
        print(f"RUN_STATUS= {log.status}")
        if log.status in (
            RunStatus.SUCCESS.value,
            RunStatus.FINISHED_WITH_SKIPS.value,
            "degraded",
        ):
            return 0
        return 1


def build_parser() -> argparse.ArgumentParser:
    """The CLI parser."""
    parser = argparse.ArgumentParser(
        description="Daily lock-time run: predict tomorrow's games before today's 6 PM ET lock"
    )
    parser.add_argument(
        "--date",
        type=date.fromisoformat,
        help="The run's ET calendar date, YYYY-MM-DD (default: today in ET)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the schedule refresh and the collection stage, writing nothing",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    start = datetime.now(UTC)
    run_date_et = args.date or start.astimezone(ET).date()
    try:
        return run_daily(run_date_et, start=start, dry_run=args.dry_run)
    except KeyboardInterrupt:
        logger.warning("Daily run interrupted")
        return 130
    except Exception as exc:  # noqa: BLE001 -- top-level CLI must catch all
        logger.error("Daily run failed", error=str(exc))
        print(f"RUN_STATUS= failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
