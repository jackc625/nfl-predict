#!/usr/bin/env python3
"""Record the whole upcoming odds board once a day, for the early-week betting study.

Owner ruling 2026-11-05: the 2026 audit found the system bets the day before kickoff, when the line
is already about as sharp as the close, and the research found most of a line's correction happens
earlier in the week. Betting earlier is planned for the next build, and testing it needs real
early-week prices from several books. Free archives carry opening line numbers but not the
per-book prices, so they are recorded here, starting now.

WHAT ONE CAPTURE DOES
---------------------
1. Reads the remaining Odds API credits from the FREE ``/v4/sports`` endpoint (0 credits).
2. Spends one board only when the credits left after the request's MAXIMUM spend (the ingest step
   may retry a lost response, so ``CREDITS_PER_BOARD`` x ``DECISION_CAPTURE_SAFETY`` = 9 credits)
   still cover every remaining decision capture of the month at its retry bound AND every remaining
   closing reading window of the month (:func:`early_capture_allowed`). The decision and closing
   captures always come first; an unreadable header refuses the capture (fail closed).
3. Requests the board with NO kickoff window, so every game the books list is returned: next
   week's games as well as this week's.
4. Appends ONE JSON line to :data:`EARLY_LINES_PATH` through the write sink: the capture instant
   (observed when the response returns), the credits read before it, and the raw board exactly
   as the API returned it (every book's ``last_update`` and its moneyline, spread and total prices).

WHO READS THIS FILE. Only the early-week run does (``pipeline.early_week``, through
``backtest.early_board``), and only to price the early arm. It is never a day-before decision input:
those rows stay in ``data/silver/odds_snapshot.parquet``, written only by the decision and closing
captures. A board can list games already in progress; the reader drops every game whose lock had
passed when the board was read.

The daily lock-time run calls :func:`capture_early_lines_now` once per real run that refreshed the
schedule (``scripts/daily_lock_pipeline.py``); it can be run by hand too:

    uv run python -m scripts.capture_early_lines

THE ODDS API KEY IS NEVER PRINTED OR LOGGED: requests log their parameters through
``scripts.ingest_odds._redact_api_key``, and this script prints counts and reasons only.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from conf.settings import get_settings
from data.storage import load_dataframe
from data.write_sink import append_jsonl
from forward_ledger.closing_windows import reading_windows
from forward_ledger.credits import (
    CREDITS_PER_BOARD,
    DECISION_CAPTURE_SAFETY,
    REASON_CREDIT_HEADER_UNREADABLE,
    REASON_CREDIT_RESERVE,
    remaining_decision_days_in_month,
)
from scripts.ingest_odds import OddsAPIClient
from utils.date_utils import ET, kickoff_wall_clock_et
from utils.logging_config import get_logger

logger = get_logger(__name__)

#: The append-only record of every early capture: one JSON line per board.
EARLY_LINES_PATH = (
    Path(get_settings().config.data.root_path)
    / "bronze"
    / "early_lines"
    / "early_lines.jsonl"
)

OUTCOME_CAPTURED = "captured"
OUTCOME_SKIPPED = "skipped"


class BoardClient(Protocol):
    """What a capture needs from the Odds API client (``scripts.ingest_odds.OddsAPIClient``)."""

    def remaining_credits(self) -> int | None: ...

    def get_nfl_odds(self) -> list[dict[str, Any]]: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class EarlyLinesOutcome:
    """What one capture did.

    Attributes:
        outcome: ``captured`` or ``skipped``.
        reason: Why a capture was skipped; None when it captured.
        events: Games on the captured board; 0 when skipped.
        remaining_credits: The credits read before the capture; None when unreadable.
    """

    outcome: str
    reason: str | None
    events: int
    remaining_credits: int | None


def remaining_closing_windows_in_month(games: pd.DataFrame, today_et: date) -> int:
    """Closing reading windows starting from *today_et* (inclusive) to the end of its month.

    Counted the way ``forward_ledger.credits.monthly_budget`` counts them: a window belongs to the
    ET calendar day its reading starts. Today's windows count even if one has already run, which
    only over-reserves.
    """
    scheduled = games.loc[games["kickoff_et"].notna()]
    windows = reading_windows(
        dict(zip(scheduled["game_id"], scheduled["kickoff_et"], strict=True))
    )
    starts = (kickoff_wall_clock_et(window.start_utc).date() for window in windows)
    return sum(
        1
        for start in starts
        if start >= today_et
        and (start.year, start.month) == (today_et.year, today_et.month)
    )


def early_capture_allowed(
    remaining_credits: int | None, games: pd.DataFrame, today_et: date
) -> tuple[bool, str | None]:
    """Whether an early capture may spend one board now, and the skip reason when it may not.

    Args:
        remaining_credits: The ``x-requests-remaining`` header just read, or None when it was
            absent or unparseable.
        games: The recorded schedule (silver ``games`` rows).
        today_et: Today's ET calendar date.

    Returns:
        ``(True, None)`` when the credits left after this request's maximum spend (one board per
        retry attempt, ``DECISION_CAPTURE_SAFETY`` attempts) still cover the month's remaining
        decision captures at their retry bound plus its remaining closing readings;
        ``(False, "credit_reserve")`` when they do not; ``(False, "credit_header_unreadable")``
        when the header could not be read.
    """
    if remaining_credits is None:
        return False, REASON_CREDIT_HEADER_UNREADABLE
    reserve = CREDITS_PER_BOARD * (
        DECISION_CAPTURE_SAFETY * remaining_decision_days_in_month(games, today_et)
        + remaining_closing_windows_in_month(games, today_et)
    )
    max_spend = CREDITS_PER_BOARD * DECISION_CAPTURE_SAFETY
    if remaining_credits - max_spend < reserve:
        return False, REASON_CREDIT_RESERVE
    return True, None


def capture_early_lines(
    games: pd.DataFrame,
    *,
    make_client: Callable[[], BoardClient] = OddsAPIClient,
    clock: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> EarlyLinesOutcome:
    """Run one capture (steps 1-4 of the module docstring) and print its outcome line.

    Args:
        games: The recorded schedule, for the credit reserve.
        make_client: Builds the Odds API client (a test hands in a stand-in).
        clock: The current instant, tz-aware UTC (a test hands in a fixed one).

    Returns:
        What the capture did. A failed request raises; the caller decides what a failure means.
    """
    client = make_client()
    try:
        remaining = client.remaining_credits()
        today_et = clock().astimezone(ET).date()
        allowed, reason = early_capture_allowed(remaining, games, today_et)
        if not allowed:
            outcome = EarlyLinesOutcome(OUTCOME_SKIPPED, reason, 0, remaining)
        else:
            board = client.get_nfl_odds()
            # Read ONCE, when the response returns: the instant the whole board was seen at.
            captured_at = clock()
            append_jsonl(
                EARLY_LINES_PATH,
                {
                    "captured_at_utc": captured_at.isoformat(),
                    "remaining_credits_before": remaining,
                    "events": len(board),
                    "board": board,
                },
                kind="early_lines_capture",
            )
            outcome = EarlyLinesOutcome(OUTCOME_CAPTURED, None, len(board), remaining)
    finally:
        client.close()

    reason_text = f" reason={outcome.reason}" if outcome.reason else ""
    print(f"EARLY_LINES= {outcome.outcome}{reason_text} events={outcome.events}")
    logger.info(
        "Early lines capture",
        outcome=outcome.outcome,
        reason=outcome.reason,
        events=outcome.events,
        remaining_credits=outcome.remaining_credits,
    )
    return outcome


def capture_early_lines_now() -> EarlyLinesOutcome:
    """One capture against the recorded schedule and the real Odds API client."""
    return capture_early_lines(load_dataframe("games", layer="silver"))


def main() -> int:
    """CLI entry point. Exit 0 on a capture or a skip, 1 on a failure (class name only)."""
    try:
        capture_early_lines_now()
    except Exception as exc:  # noqa: BLE001 - the message can carry a request URL
        print(f"EARLY_LINES= failed {type(exc).__name__}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
