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
   capture, so results, moved games and each new playoff round arrive with no manual step. The
   season is read from the recorded schedule, and rolls to the next one on its own once the
   last is over and the calendar season has turned (step 27b) -- August needs no manual step.
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
is written. Its odds step runs on a fixture board and makes NO paid Odds API request, and its
output says so (``DRY_RUN_ODDS=``). It stops before the build, because a build cannot read captures that were never
saved -- the build and predictions are proved by a real run.

Usage:
    uv run python -m scripts.daily_lock_pipeline                    # today's run
    uv run python -m scripts.daily_lock_pipeline --date 2026-09-26  # today, or a day this week
    uv run python -m scripts.daily_lock_pipeline --date 2026-10-03 --dry-run  # rehearse any future day
    uv run python -m scripts.daily_lock_pipeline --dry-run          # collection only, no writes
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from backtest.weekly_bet_list import _with_gameday, select_games_for_decision_instant
from data.write_sink import (
    ProductionSink,
    RecordingSink,
    active_sink,
    append_jsonl,
)
from pipeline import daily_steps
from pipeline.daily_steps import (
    COLLECTION_STEP_NAMES,
    DECISION_TIME_BRANCH,
    DailySlate,
    build_daily_step_registry,
    slate_lock,
)
from utils.date_utils import ET
from utils.logging_config import get_logger

if TYPE_CHECKING:
    import pandas as pd

logger = get_logger(__name__)

#: One line per run that predicted nothing (no games tomorrow, the lock had passed, or the run
#: failed -- ``run_failed`` names the slate games left unpredicted).
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


def _record_no_prediction(
    run_date_et: date, outcome: str, game_ids: list[str], *, error: str | None = None
) -> None:
    """Append the day's no-prediction record through the write sink."""
    record: dict[str, object] = {
        "run_date_et": run_date_et.isoformat(),
        "outcome": outcome,
        "game_ids": game_ids,
        "recorded_at": datetime.now(UTC).isoformat(),
    }
    if error is not None:
        record["error"] = error
    append_jsonl(DAILY_RUN_RECORDS, record, kind="daily_run_record")


def _unpredicted_slate_games(slate: DailySlate) -> list[str]:
    """The slate's games with no prediction row made for THIS slate's lock, sorted.

    A row made for the slate carries the slate's lock as ``information_cutoff_utc``; a game
    with none was not predicted today, whatever other day's row it may have.
    """
    import pandas as pd

    from pipeline.steps import _predictions_output_dir

    path = (
        _predictions_output_dir() / f"predictions_{slate.season}_week{slate.week}.csv"
    )
    predicted: set[str] = set()
    if path.exists():
        rows = pd.read_csv(path)
        if {"game_id", "information_cutoff_utc"} <= set(rows.columns):
            cutoff = pd.to_datetime(rows["information_cutoff_utc"], utc=True)
            predicted = set(
                rows.loc[cutoff == pd.Timestamp(slate.lock), "game_id"].astype(str)
            )
    return sorted(slate.game_ids - predicted)


def _record_run_failure(
    run_date_et: date, slate: DailySlate | None, exc: BaseException
) -> None:
    """Record tomorrow's games as getting NO predictions from a run that could not finish.

    33.2 review C1 CR-05 = B WR-05: a run refused by the staleness gate, stopped by a critical
    step, or interrupted used to leave no record naming the games it failed -- unlike the
    lock-passed and no-games days, which each write one. The games are the slate's still
    unpredicted ones when the slate is known, else tomorrow's games per the recorded schedule.
    Never raises: the failure being recorded is the one the caller re-raises.
    """
    try:
        game_ids = (
            _unpredicted_slate_games(slate)
            if slate is not None
            else _recorded_tomorrow_ids(run_date_et)
        )
        detail = f": {exc}" if str(exc) else ""
        _record_no_prediction(
            run_date_et,
            "run_failed",
            game_ids,
            error=f"{type(exc).__name__}{detail}",
        )
        for game_id in game_ids:
            print(f"NOT_PREDICTED {game_id}: the run failed ({type(exc).__name__})")
    except Exception as record_error:  # noqa: BLE001 - never mask the run's own failure
        logger.error("Could not record the failed run", error=str(record_error))


class ScheduleNotPublishedError(RuntimeError):
    """nflverse serves no schedule yet for the season the run has to refresh."""


def _write_census() -> dict[str, tuple[int, int]]:
    """Every file under the data root and ``config/``, with its size and modification time.

    Byte-code caches are left out: importing the ``data`` package legitimately writes them.
    """
    from conf.settings import get_settings

    census: dict[str, tuple[int, int]] = {}
    for root in (Path(get_settings().config.data.root_path), Path("config")):
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if "__pycache__" in path.parts or not path.is_file():
                continue
            stat = path.stat()
            census[path.as_posix()] = (stat.st_size, stat.st_mtime_ns)
    return census


@dataclass
class _MeasuredRecordingSink(RecordingSink):
    """The dry run's sink, which also MEASURES writes instead of asserting there were none.

    33.2 review C1 IN-01: ``RecordingSink.production_write_count`` is zero by construction, so
    the dry run's ``WRITES=`` line could never report a write that bypassed the sink. It now
    counts the files under the data root and ``config/`` that appeared, vanished or changed
    between the start of the run and the contract line.
    """

    baseline: dict[str, tuple[int, int]] = field(default_factory=_write_census)

    def measured_writes(self) -> int:
        """Files added, removed or modified since the sink was created."""
        now = _write_census()
        changed = {path for path in now if self.baseline.get(path) != now[path]}
        removed = set(self.baseline) - set(now)
        return len(changed | removed)


def _final_game_awaits_result(season: int, instant: datetime) -> bool:
    """Whether *season*'s last recorded game has kicked off with no result stored in silver.

    The one game whose result can land AFTER its season has gone to the offseason is the last
    one -- the Super Bowl: every earlier game's result arrives while the season is still being
    refreshed daily. So the offseason refreshes the completed season until that one result is
    in, and never again.
    """
    import pandas as pd

    from data.storage import load_dataframe

    games = load_dataframe("games", layer="silver")
    played = games.loc[
        (games["season"] == season)
        & (pd.to_datetime(games["kickoff_et"], utc=True) <= instant)
    ]
    if played.empty:
        return False
    last = played.sort_values("kickoff_et").iloc[-1]
    return bool(pd.isna(last["home_score"]) or pd.isna(last["away_score"]))


def _offseason_target(instant: datetime) -> tuple[int, int] | None:
    """What an offseason day refreshes: the completed season while its final result is due."""
    from utils.current_slate import resolve_current_slate

    completed = resolve_current_slate(instant)
    if _final_game_awaits_result(completed.season, instant):
        return completed.season, completed.week
    return None


def _refresh_schedule(run_date_et: date, *, dry_run: bool) -> int | None:
    """Capture what nflverse serves now, ingest the schedule from it, and return the season.

    Returns ``None`` on an offseason day with nothing to refresh (33.2 review B WR-03 = C1
    WR-03): no capture, no probe line, no ingest -- a clean no-op, not a daily capture of a
    season that cannot change. The next season's schedule is still picked up: from August it is
    either not recorded yet (refreshing it is how it gets recorded) or recorded and refreshed
    daily until its opener. The completed season is refreshed only while its final game's
    result is due (:func:`_final_game_awaits_result`).

    The schedule ingest reads the LIVE-ZONE capture, never the network directly, so the capture
    must come first -- the Friday registry's order. Daily, this is what picks up results, moved
    games and the next playoff round with no manual step. A no-write run cannot capture (see
    :data:`DRY_RUN_UNSUPPORTED_STEPS`), so it ingests from the latest existing capture.

    THE SEASON COMES FROM THE SCHEDULE (step 27b), resolved once at the run date's lock through
    ``utils.current_slate.refresh_target``: the current slate's season, or -- once the calendar
    season has turned past a completed season -- the NEXT season, whose capture is how its
    schedule gets recorded. So a new season needs no manual switch. Until one of that season's
    games has kicked off there is no result to ingest, so results are left out.

    PLAY-BY-PLAY BEFORE THE FIRST KICKOFF (step 27c, applying D32-03): nflverse refuses a new
    season's plays until the Thursday after Labor Day, but the opener locks before that. Once
    the season's schedule is recorded, the capture records that refusal as an explicit empty
    capture (proved by the recorded first kickoff), so the opener's lock-day build reads an
    empty season and the prior season's plays. Only while the schedule is NOT recorded yet --
    the first run of a new season, which records it -- is play-by-play left out, by name: there
    is nothing yet to prove the season has not begun, and no game of it to predict.

    Raises:
        ScheduleNotPublishedError: nflverse serves no schedule for that season yet.
    """
    from data import upstream_pin
    from pipeline.steps import step_capture_live_season
    from scripts.ingest_games import GameDataIngester
    from utils.current_slate import (
        first_recorded_kickoff,
        refresh_target,
        season_has_kicked_off,
    )

    instant = slate_lock(run_date_et)
    target = refresh_target(instant) or _offseason_target(instant)
    if target is None:
        print(
            "REFRESH_SKIPPED offseason: no season is being played or due, and the completed "
            "season's results are all recorded; nothing is captured or ingested"
        )
        return None
    season, week = target
    started = season_has_kicked_off(season, instant)
    datasets = sorted(upstream_pin.DATASET_COLUMNS)
    if first_recorded_kickoff(season) is None:
        datasets.remove("pbp")
        print(
            f"CAPTURE_SKIPPED pbp: the {season} schedule is not recorded yet, so nothing "
            f"proves no {season} game has kicked off"
        )

    if dry_run:
        print(
            "DRY_RUN_SKIPPED capture_live_season: "
            f"{DRY_RUN_UNSUPPORTED_STEPS['capture_live_season']}"
        )
    else:
        step_capture_live_season(season=season, week=week, datasets=datasets)

    if upstream_pin.load_schedules([season]).empty:
        msg = (
            f"nflverse serves no {season} schedule yet, so tomorrow's games cannot be "
            f"selected. The run refreshes {season} because the recorded schedule says it is "
            "due; it will pick the schedule up on the first day nflverse publishes it."
        )
        raise ScheduleNotPublishedError(msg)
    GameDataIngester().ingest_games(seasons=[season], include_results=started)
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


def _require_slate_is_current_week(slate: DailySlate) -> None:
    """The schedule must name the slate's week as the current slate AT THE SLATE'S LOCK.

    Resolved at the run date's own lock (33.2 review C1 WR-09), never at the wall clock: every
    game's lock resolves to that game's own week (``utils.current_slate``, asserted over every
    2018-2026 game), so this holds for any ``--date`` whose slate the store records correctly.
    Resolving at the clock made a ``--date`` in any other week refuse for no reason. If it ever
    fired, the resolver and the lock-instant selector would disagree about tomorrow -- refused.
    """
    from utils.current_slate import resolve_current_slate

    current = resolve_current_slate(slate.lock)
    if (current.season, current.week) != (slate.season, slate.week):
        msg = (
            f"tomorrow's games are {slate.season} week {slate.week}, but the schedule names "
            f"{current.season} week {current.week} as the current slate at the lock "
            f"{slate.lock.isoformat()}; the resolver and the slate selector disagree."
        )
        raise RuntimeError(msg)


class RunDateRefusedError(ValueError):
    """A ``--date`` the run refuses BEFORE anything is requested or written."""


def _refuse_an_unrunnable_date(
    run_date_et: date, start: datetime, *, dry_run: bool
) -> None:
    """Refuse a ``--date`` whose run could only write false records (33.2 review C1 WR-09).

    * A PAST date: its lock has passed and its games have been played or have locked. Running
      it could only append a ``lock_passed`` record for a day that already has its own real
      history, through the PRODUCTION sink -- polluting the run record. Refused for dry runs
      too: there is nothing to rehearse about a day that is over.
    * A FUTURE date in ANOTHER week, on a real run: the capture and build steps resolve the
      week they work on from the clock, so they would capture and build a different week from
      the one predicted. Checked against the RECORDED schedule before any refresh. A dry run
      (collection only, nothing written) may rehearse any future day.

    Raises:
        RunDateRefusedError: naming the date and the reason; nothing has been written.
    """
    from utils.current_slate import SlateResolutionError, resolve_current_slate

    today_et = start.astimezone(ET).date()
    if run_date_et < today_et:
        msg = (
            f"--date {run_date_et.isoformat()} is before today ({today_et.isoformat()} ET). "
            "Its games' lock has passed, so the run could only write a record for a day that "
            "is over; nothing was requested or written."
        )
        raise RunDateRefusedError(msg)
    if dry_run or run_date_et == today_et:
        return
    try:
        now_week = resolve_current_slate(start).as_tuple()
        date_week = resolve_current_slate(slate_lock(run_date_et)).as_tuple()
    except SlateResolutionError as exc:
        msg = (
            f"--date {run_date_et.isoformat()} is a future day and the recorded schedule "
            f"cannot confirm it is in the current week ({exc}); run it on the day, or rehearse "
            "it with --dry-run. Nothing was requested or written."
        )
        raise RunDateRefusedError(msg) from exc
    if now_week != date_week:
        msg = (
            f"--date {run_date_et.isoformat()} is in {date_week[0]} week {date_week[1]}, but "
            f"the steps resolve the current week, {now_week[0]} week {now_week[1]}, from the "
            "clock; a real run would capture and build a different week from the one it "
            "predicts. Run it on the day, or rehearse it with --dry-run. Nothing was "
            "requested or written."
        )
        raise RunDateRefusedError(msg)


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
    writes = (
        sink.measured_writes() if isinstance(sink, _MeasuredRecordingSink) else "n/a"
    )
    print(f"WRITES= {writes}")
    print(f"INTENDED_WRITES= {intended}")
    print(f"LOCK_PASSED_BEFORE_COLLECTION= {lock_passed}")
    print(f"DECISION_TIME_BRANCH= {DECISION_TIME_BRANCH}")
    print(f"NEXT_DAY_GAMES= {next_day_games}")
    print("BUILD_SCOPE= full-history (owner ruling 2026-09-23)")


def _run_collection_only(slate: DailySlate) -> None:
    """The dry run: the collection stage's real code, in order, under the caller's sink.

    The odds step runs the real ingest on a FIXTURE board, never the paid Odds API (33.2 review
    C1 WR-02): a no-write rehearsal that spends credits is not a no-cost one. The run's output
    says so on its ``DRY_RUN_ODDS=`` line.
    """
    for step in build_daily_step_registry(slate):
        if step.name not in COLLECTION_STEP_NAMES:
            break
        logger.info("Dry-run collection step", step=step.name)
        if step.name == "ingest_odds":
            print(f"DRY_RUN_ODDS= {daily_steps.DRY_RUN_ODDS_SOURCE}")
            daily_steps.ingest_slate_odds(slate, fixture=True)
            continue
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
    """One daily run as of *run_date_et*, started at *start*. Returns the exit code.

    Raises:
        RunDateRefusedError: *run_date_et* cannot be run at *start*; nothing was written.
    """
    _refuse_an_unrunnable_date(run_date_et, start, dry_run=dry_run)
    lock = slate_lock(run_date_et)
    sink: ProductionSink | RecordingSink = (
        _MeasuredRecordingSink() if dry_run else ProductionSink()
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

        # Everything after the lock check: a run that cannot finish records tomorrow's games
        # as getting no predictions, then re-raises (33.2 review C1 CR-05 = B WR-05).
        progress: dict[str, DailySlate] = {}
        try:
            return _run_the_day(
                run_date_et, start=start, dry_run=dry_run, sink=sink, progress=progress
            )
        except BaseException as exc:
            _record_run_failure(run_date_et, progress.get("slate"), exc)
            raise


def _run_the_day(
    run_date_et: date,
    *,
    start: datetime,
    dry_run: bool,
    sink: ProductionSink | RecordingSink,
    progress: dict[str, DailySlate],
) -> int:
    """Steps 1-3 of one daily run. The selected slate is left in *progress* for the caller."""
    # 1-2. Refresh the schedule, then select tomorrow's games.
    season = _refresh_schedule(run_date_et, dry_run=dry_run)
    if season is None:
        # The offseason: no season has a slate to predict, and nothing was refreshed.
        _record_no_prediction(run_date_et, "offseason", [])
        _print_contract(sink, dry_run=dry_run, lock_passed=0, next_day_games=0)
        return 0
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
    progress["slate"] = slate
    _require_slate_is_current_week(slate)

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

    # The deadline bounds live-skip re-runs by the slate's lock (33.2 review C1 WR-05).
    pipeline = FridayPipeline(
        steps=build_daily_step_registry(slate), deadline=slate.lock
    )
    log = pipeline.run()
    _print_contract(
        sink, dry_run=False, lock_passed=0, next_day_games=len(slate.game_ids)
    )
    _report_skips(log.start_time)
    for game_id, reason in sorted(slate.weather_unknown.items()):
        print(f"WEATHER_UNKNOWN {game_id}: {reason}")
    for game_id, reason in sorted(slate.odds_missing.items()):
        print(f"ODDS_MISSING {game_id}: {reason}; predicted, market side blank, no bet")
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
    except RunDateRefusedError as refusal:
        logger.error("Run date refused", error=str(refusal))
        print(f"RUN_REFUSED= {refusal}")
        return 2
    except KeyboardInterrupt:
        logger.warning("Daily run interrupted")
        return 130
    except Exception as exc:  # noqa: BLE001 -- top-level CLI must catch all
        logger.error("Daily run failed", error=str(exc))
        print(f"RUN_STATUS= failed: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
