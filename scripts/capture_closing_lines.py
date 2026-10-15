#!/usr/bin/env python3
"""Take one near-kickoff closing reading of the odds board (Phase 34, LDGR-07, D-07 / D-08).

What the ``NFL_Predict_Closing`` wake task runs (``deployment/windows_closing_scheduler.xml``:
``uv run python -m scripts.capture_closing_lines``). It takes NO required arguments and is safe to
run at any time: every run decides for itself whether there is anything to read.

ONE RUN, IN ORDER
-----------------
1. The targets: the silver ``games`` rows with no recorded result whose kickoff is in
   ``(now, now + 60:00]`` (``forward_ledger.closing.CLOSING_WINDOW``). A game that has already
   kicked off is never requested. No target -> event ``no_games``, no client is built, no request
   is made and no credit is spent.
2. The credit rule (D-08): the remaining credits are read from the FREE ``/v4/sports`` endpoint
   (``OddsAPIClient.remaining_credits``, 0 credits) and handed to
   ``forward_ledger.credits.closing_capture_allowed`` with the season's schedule. Refused -> event
   ``skipped`` with the reason (``credit_reserve`` / ``credit_header_unreadable``) and the target
   game ids; no odds request. The daily DECISION capture never consults this rule.
3. The reading: one board request (3 credits) whose kickoff window is the targets' kickoffs
   widened by ``COMMENCE_TIME_TOLERANCE`` (one hour) on each side -- the same tolerance the
   transform's matcher already allows between a listing and the schedule (34-REVIEW WR-07). The
   provider bills per market per region, not per event, so the wider window costs nothing extra;
   any non-target event it returns matches no target schedule row and is never written. The
   capture instant is observed when the response returns.
   The board is transformed with ``capture_kind="closing"`` against the targets and their locks --
   ``snapshot_ts = lock``, ``created_at`` = the true capture instant, any game already kicked off
   at that instant refused -- validated against ``OddsSchema`` and appended to
   ``data/silver/odds_snapshot.parquet`` through ``data.storage.append_odds_captures``, under the
   odds store's lock shared with the daily run. Event ``captured`` with the game ids, the rows
   written and the capture instant, plus the evidence that explains an empty or partial board
   (34-REVIEW CR-02): ``board_events`` and ``board_games`` (every event the API returned, with its
   listed commence time), ``rows_by_game`` (rows written per target game), ``empty_game_ids``
   (targets that got no row) and the transform's ``unmatched`` / ``in_play`` / ``no_kickoff``
   lists.
4. Any exception -> event ``failed`` with reason ``capture_failed`` and the exception's CLASS NAME
   only (never its message, which could carry a request URL), then one normal-priority failed
   push (``notify.closing_task``, Phase 35 D-13), exit 1. The push does NOT depend on the record:
   an event append that raises prints ``CLOSING_CAPTURE_LOG= failed <class>`` and the push still
   goes out, saying the failure is not in the run log -- the outside watcher does not monitor this
   task, so the push is its only external signal. A notifier failure prints
   ``NOTIFY= failed <class>`` and never changes the exit code. This task never checks in with the
   watcher; only the daily run does.

Each outcome is exactly ONE ``closing_capture`` event in ``logs/ledger_runs.jsonl``
(``forward_ledger.run_log``). The settle pass reads the ``skipped`` and ``failed`` events' reasons
and game ids, and the ``captured`` events' ``empty_game_ids`` (``capture_empty``: the reading ran
and wrote nothing for the game -- never ``capture_missed``), when it finalizes a game's closing
columns as NULL (``forward_ledger.closing.finalize_closing``).

These closing rows can never reach a decision: every decision reader judges a line's admissibility
on ``created_at`` against the game's lock (``tests/unit/test_closing_rows_never_priced.py``).

THE ODDS API KEY IS NEVER PRINTED OR LOGGED. Every request logs its parameters through
``scripts.ingest_odds._redact_api_key``; this script records counts, ids, instants and exception
class names only.

``--now`` exists for tests only: it fixes the instant the targets are chosen at (it must carry a UTC
offset). The capture instant is always observed, never supplied.

Exit codes: 0 no games / skipped / captured; 1 failed; 2 bad ``--now``.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any, cast

import pandas as pd

import utils.game_lock as lock_rule
from data.storage import append_odds_captures, load_dataframe
from forward_ledger.closing import CLOSING_WINDOW
from forward_ledger.closing_windows import kickoff_instant
from forward_ledger.credits import closing_capture_allowed
from forward_ledger.run_log import record_event
from scripts.ingest_historical_odds import (
    NaiveTimestampError,
    require_aware_snapshot_ts,
)
from scripts.ingest_odds import (
    COMMENCE_TIME_TOLERANCE,
    SCHEDULE_COLUMNS,
    OddsDataIngester,
)

CAPTURE_EVENT = "closing_capture"
REASON_CAPTURE_FAILED = "capture_failed"

# The board request's kickoff window reaches this far past the targets' kickoffs on each side. It
# was one minute, 60 times stricter than the matcher it feeds, and two 8:15 PM single-game readings
# came back empty with nothing kept to say why (34-REVIEW WR-07). The board costs the same 3
# credits however many events it lists (one credit per market per region, forward_ledger.credits),
# and only the targets can be written: the transform matches against the targets' schedule rows
# alone and refuses any capture at or after a target's scheduled kickoff.
REQUEST_MARGIN: timedelta = COMMENCE_TIME_TOLERANCE


def _load_games() -> pd.DataFrame:
    """Every silver ``games`` row: the schedule the targets and the credit rule are read from."""
    return load_dataframe("games", layer="silver")


def _clock() -> datetime:
    """The current instant, tz-aware UTC."""
    return datetime.now(UTC)


def _notify_closing_failure(
    game_ids: list[str], error_type: str, *, log_written: bool
) -> object:
    """Send the failed reading's normal-priority push (``notify.closing_task``).

    The import is LAZY, so a broken notify module can never break this script's import or a
    reading that succeeds.
    """
    from notify.closing_task import notify_closing_failure

    return notify_closing_failure(game_ids, error_type, log_written=log_written)


# Seams: tests replace these so no real client is built, no clock is read, nothing is appended
# to logs/ and nothing is sent. The ingester owns the Odds API client (``ingester.api_client``)
# and the transform. The test suite keeps NOTIFY_CLOSING_FAILURE inert (tests/conftest.py); a
# wiring test restores this default.
LOAD_GAMES: Callable[[], pd.DataFrame] = _load_games
CLOCK: Callable[[], datetime] = _clock
MAKE_INGESTER: Callable[[], OddsDataIngester] = OddsDataIngester
LOG: Callable[..., Any] = record_event
NOTIFY_CLOSING_FAILURE: Callable[..., Any] = _notify_closing_failure


def _aware_instant(text: str) -> datetime:
    """An argparse type: an ISO instant that MUST carry a UTC offset."""
    try:
        return require_aware_snapshot_ts(text)
    except (NaiveTimestampError, ValueError) as error:
        msg = f"{text!r} is not an ISO instant with a UTC offset ({error})"
        raise argparse.ArgumentTypeError(msg) from error


def build_parser() -> argparse.ArgumentParser:
    """The CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Take one near-kickoff closing reading of the odds board for every unplayed game "
            "kicking off in the next 60 minutes. Takes no required arguments: run it at any time; "
            "with no game near it makes no request and spends no credit."
        )
    )
    parser.add_argument(
        "--now",
        type=_aware_instant,
        default=None,
        help="TESTS ONLY: the instant the targets are chosen at, ISO 8601 WITH a UTC offset",
    )
    return parser


def closing_targets(games: pd.DataFrame, now: datetime) -> pd.DataFrame:
    """The unplayed games whose kickoff is in ``(now, now + CLOSING_WINDOW]``, earliest first.

    A game with a recorded result, no kickoff, or a kickoff at or before *now* is never a target.
    """
    unplayed = games.loc[games["home_score"].isna() & games["kickoff_et"].notna()]
    kickoffs = pd.Series(
        [kickoff_instant(kickoff) for kickoff in unplayed["kickoff_et"]],
        index=unplayed.index,
        dtype=object,
    )
    near = (kickoffs > now) & (kickoffs <= now + CLOSING_WINDOW)
    targets = unplayed.loc[near.astype(bool)].copy()
    targets["_kickoff_utc"] = kickoffs.loc[near.astype(bool)]
    return targets.sort_values(["_kickoff_utc", "game_id"]).reset_index(drop=True)


def _record(outcome: str, game_ids: list[str], **fields: Any) -> None:
    """The run's one ``closing_capture`` event, and one owner-readable output line."""
    LOG(CAPTURE_EVENT, outcome=outcome, game_ids=game_ids, **fields)
    reason = fields.get("reason")
    suffix = f" reason={reason}" if reason else ""
    empty = fields.get("empty_game_ids")
    if empty:
        suffix += f" empty={len(empty)}"
    print(f"CLOSING_CAPTURE= {outcome}{suffix} games={len(game_ids)}")


def _board_games(raw_odds: list[dict[str, Any]]) -> list[str]:
    """Every event the board returned, as ``away @ home (commence_time ...)``, sorted."""
    return sorted(
        f"{event.get('away_team')} @ {event.get('home_team')} "
        f"(commence_time {event.get('commence_time')})"
        for event in raw_odds
    )


def _capture(
    targets: pd.DataFrame, games: pd.DataFrame, now: datetime, game_ids: list[str]
) -> int:
    """Steps 2 and 3 for a non-empty *targets*; returns the exit code."""
    ingester = MAKE_INGESTER()
    try:
        remaining = ingester.api_client.remaining_credits()
        season = int(targets["season"].iloc[0])
        season_games = games.loc[games["season"] == season]
        today_et = now.astimezone(lock_rule.ET).date()
        allowed, reason = closing_capture_allowed(remaining, season_games, today_et)
        if not allowed:
            _record("skipped", game_ids, reason=reason, remaining_credits=remaining)
            return 0

        kickoffs = list(targets["_kickoff_utc"])
        raw_odds = ingester.api_client.get_nfl_odds(
            date_from=min(kickoffs) - REQUEST_MARGIN,
            date_to=max(kickoffs) + REQUEST_MARGIN,
        )
        # Read ONCE, when the response returns: the instant every row carries as created_at.
        captured_at = CLOCK()

        schedule = cast("pd.DataFrame", targets[list(SCHEDULE_COLUMNS)]).reset_index(
            drop=True
        )
        locks = {
            str(game_id): lock.to_pydatetime()
            for game_id, lock in lock_rule.lock_frame(schedule).items()
        }
        odds = ingester.transform_odds_data(
            raw_odds,
            schedule=schedule,
            locks=locks,
            captured_at=captured_at,
            capture_kind="closing",
        )
        validated = pd.DataFrame()
        if not odds.empty:
            validated = ingester.validate_odds_data(odds)
            if not validated.empty:
                append_odds_captures(validated)
        rows = len(validated)
        written = (
            validated["game_id"].astype(str).value_counts()
            if rows
            else pd.Series(dtype="int64")
        )
        rows_by_game = {game_id: int(written.get(game_id, 0)) for game_id in game_ids}
        report = ingester.last_match_report
        _record(
            "captured",
            game_ids,
            rows=rows,
            captured_at=captured_at.isoformat(),
            remaining_credits=remaining,
            board_events=len(raw_odds),
            board_games=_board_games(raw_odds),
            rows_by_game=rows_by_game,
            empty_game_ids=[g for g in game_ids if rows_by_game[g] == 0],
            unmatched=list(report.unmatched_games),
            in_play=list(report.in_play_games),
            no_kickoff=list(report.no_kickoff_games),
        )
        return 0
    finally:
        ingester.close()


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    game_ids: list[str] = []
    try:
        now = args.now if args.now is not None else CLOCK()
        games = LOAD_GAMES()
        targets = closing_targets(games, now)
        if targets.empty:
            _record("no_games", game_ids, run_at_utc=now.isoformat())
            return 0
        game_ids = [str(game_id) for game_id in targets["game_id"]]
        return _capture(targets, games, now, game_ids)
    except Exception as error:  # noqa: BLE001 - every failure is recorded as one named outcome
        # The class name only: an exception message can carry a request URL with its parameters.
        error_type = type(error).__name__
        # The record and the push are INDEPENDENT: a run log that cannot be appended must not
        # silence this task's only external signal (the watcher does not monitor it).
        log_written = True
        try:
            _record(
                "failed",
                game_ids,
                reason=REASON_CAPTURE_FAILED,
                error_type=error_type,
            )
        except Exception as log_error:  # noqa: BLE001 - the push still goes out
            log_written = False
            print(f"CLOSING_CAPTURE_LOG= failed {type(log_error).__name__}")
        try:
            NOTIFY_CLOSING_FAILURE(game_ids, error_type, log_written=log_written)
        except Exception as notify_error:  # noqa: BLE001 - sending never alters the exit code
            print(f"NOTIFY= failed {type(notify_error).__name__}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
