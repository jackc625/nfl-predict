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
3. The reading: one board request (3 credits) whose kickoff window is bounded by the targets'
   kickoffs, one minute either side. The capture instant is observed when the response returns.
   The board is transformed with ``capture_kind="closing"`` against the targets and their locks --
   ``snapshot_ts = lock``, ``created_at`` = the true capture instant, any game already kicked off
   at that instant refused -- validated against ``OddsSchema`` and appended to
   ``data/silver/odds_snapshot.parquet`` through ``data.storage.append_odds_captures``, under the
   odds store's lock shared with the daily run. Event ``captured`` with the game ids, the rows
   written and the capture instant.
4. Any exception -> event ``failed`` with reason ``capture_failed`` and the exception's CLASS NAME
   only (never its message, which could carry a request URL), exit 1.

Each outcome is exactly ONE ``closing_capture`` event in ``logs/ledger_runs.jsonl``
(``forward_ledger.run_log``). The settle pass reads the ``skipped`` and ``failed`` events' reasons
and game ids when it finalizes a game's closing columns as NULL
(``forward_ledger.closing.finalize_closing``).

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
from scripts.ingest_odds import SCHEDULE_COLUMNS, OddsDataIngester

CAPTURE_EVENT = "closing_capture"
REASON_CAPTURE_FAILED = "capture_failed"

# The board request's kickoff window reaches one minute past the targets' kickoffs on each side,
# so a kickoff on the window's edge is never filtered out by the provider.
REQUEST_MARGIN = timedelta(minutes=1)


def _load_games() -> pd.DataFrame:
    """Every silver ``games`` row: the schedule the targets and the credit rule are read from."""
    return load_dataframe("games", layer="silver")


def _clock() -> datetime:
    """The current instant, tz-aware UTC."""
    return datetime.now(UTC)


# Seams: tests replace these so no real client is built, no clock is read and nothing is appended
# to logs/. The ingester owns the Odds API client (``ingester.api_client``) and the transform.
LOAD_GAMES: Callable[[], pd.DataFrame] = _load_games
CLOCK: Callable[[], datetime] = _clock
MAKE_INGESTER: Callable[[], OddsDataIngester] = OddsDataIngester
LOG: Callable[..., Any] = record_event


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
    print(f"CLOSING_CAPTURE= {outcome}{suffix} games={len(game_ids)}")


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
        rows = 0
        if not odds.empty:
            validated = ingester.validate_odds_data(odds)
            if not validated.empty:
                append_odds_captures(validated)
                rows = len(validated)
        _record(
            "captured",
            game_ids,
            rows=rows,
            captured_at=captured_at.isoformat(),
            remaining_credits=remaining,
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
        _record(
            "failed",
            game_ids,
            reason=REASON_CAPTURE_FAILED,
            error_type=type(error).__name__,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
