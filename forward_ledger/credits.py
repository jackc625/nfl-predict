"""The free Odds API quota: when a closing reading may spend credits, and the season budget.

Phase 34, LDGR-07 / D-08. The owner's key is the free Starter plan: 500 credits a month. One board
request (``h2h``, ``spreads``, ``totals`` in one region) costs ``CREDITS_PER_BOARD`` = 3. The daily
decision capture spends one board on every lock day; each closing reading window spends one more.

THE DECISION CAPTURE ALWAYS COMES FIRST
---------------------------------------
:func:`closing_capture_allowed` lets a closing reading spend a board only when the credits left
after it still cover every remaining decision capture of the calendar month at the ingest step's
retry bound (``DECISION_CAPTURE_SAFETY`` = 3 boards per decision day). Otherwise the closing
reading is skipped with ``credit_reserve``; an unreadable remaining-credits header skips it with
``credit_header_unreadable`` (fail closed). The decision capture never calls this module.

THE HEADER IS READ, NEVER COMPUTED
----------------------------------
The provider resets the free quota "on the first of every month" without stating the hour or time
zone, so a remaining count computed from our own spend log could be wrong for hours around each
reset. The caller always passes the ``x-requests-remaining`` header it just read from the free
``/v4/sports`` endpoint (0 credits).

THE WRITTEN BUDGET
------------------
:func:`monthly_budget` computes, per calendar month (ET), the decision days and closing windows of
a recorded schedule, their credits, the total and the worst case with every decision day spending
its retry bound. ``tests/unit/test_credit_budget.py`` asserts both stay under the quota on the
recorded 2026 schedule and prints the table; ``docs/guides/AUTOMATION.md`` publishes it (Plan
34-21).

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from collections import Counter
from datetime import date

import pandas as pd

from forward_ledger.closing import CLOSING_SKIP_REASONS
from forward_ledger.closing_windows import kickoff_instant, reading_windows
from utils.date_utils import kickoff_wall_clock_et
from utils.game_lock import game_lock

__all__ = [
    "CREDITS_PER_BOARD",
    "DECISION_CAPTURE_SAFETY",
    "MONTHLY_FREE_QUOTA",
    "REASON_CREDIT_HEADER_UNREADABLE",
    "REASON_CREDIT_RESERVE",
    "closing_capture_allowed",
    "decision_days",
    "monthly_budget",
    "remaining_decision_days_in_month",
]

# One board request: three markets x one region ("the usage quota cost is 1 per region per market").
CREDITS_PER_BOARD = 3
# The free Starter plan's monthly quota.
MONTHLY_FREE_QUOTA = 500
# The ingest step's retry bound: a decision day may spend up to three boards.
DECISION_CAPTURE_SAFETY = 3

REASON_CREDIT_RESERVE = "credit_reserve"
REASON_CREDIT_HEADER_UNREADABLE = "credit_header_unreadable"
if not {REASON_CREDIT_RESERVE, REASON_CREDIT_HEADER_UNREADABLE} <= set(
    CLOSING_SKIP_REASONS
):  # pragma: no cover
    msg = "forward_ledger.credits' skip reasons are not forward_ledger.closing's skip reasons"
    raise RuntimeError(msg)


def decision_days(games: pd.DataFrame) -> list[date]:
    """The ET dates a decision capture runs: every game's lock day, sorted and distinct.

    A game is decided at its lock, 18:00 ET on the ET day before its kickoff
    (``utils.game_lock``), and the daily run captures that day's board just before it -- so a
    decision day is an ET date whose next ET day has a game. A game with no kickoff has no lock.
    """
    kickoffs = games.loc[games["kickoff_et"].notna(), "kickoff_et"]
    return sorted({game_lock(kickoff_instant(kickoff)).date() for kickoff in kickoffs})


def remaining_decision_days_in_month(games: pd.DataFrame, today_et: date) -> int:
    """Decision days from *today_et* (inclusive) to the end of its calendar month.

    Today counts: a closing reading taken before today's 17:00 decision run must leave room for it.
    """
    return sum(
        1
        for day in decision_days(games)
        if day >= today_et and (day.year, day.month) == (today_et.year, today_et.month)
    )


def closing_capture_allowed(
    remaining_credits: int | None, games: pd.DataFrame, today_et: date
) -> tuple[bool, str | None]:
    """Whether a closing reading may spend one board now, and the skip reason when it may not.

    Args:
        remaining_credits: The ``x-requests-remaining`` header just read, or None when it was
            absent or unparseable.
        games: The season's schedule (silver ``games`` rows).
        today_et: Today's ET calendar date.

    Returns:
        ``(True, None)`` when the credits left after this board still cover the month's
        remaining decision captures at the safety factor; ``(False, "credit_reserve")`` when they
        do not; ``(False, "credit_header_unreadable")`` when the header could not be read.
    """
    if remaining_credits is None:
        return False, REASON_CREDIT_HEADER_UNREADABLE
    reserve = (
        CREDITS_PER_BOARD
        * DECISION_CAPTURE_SAFETY
        * remaining_decision_days_in_month(games, today_et)
    )
    if remaining_credits - CREDITS_PER_BOARD < reserve:
        return False, REASON_CREDIT_RESERVE
    return True, None


def monthly_budget(games: pd.DataFrame) -> dict[str, dict[str, int]]:
    """The written credit budget of a schedule, per ET calendar month (``"YYYY-MM"``), in order.

    Each month carries ``decision_days``, ``decision_credits``, ``closing_windows`` (from
    :func:`forward_ledger.closing_windows.reading_windows`, counted in the month their reading
    starts), ``closing_credits``, ``total`` and ``worst_case_total`` (every decision day spending
    ``DECISION_CAPTURE_SAFETY`` boards, plus the closing boards).
    """
    decisions = Counter(f"{day:%Y-%m}" for day in decision_days(games))
    scheduled = games.loc[games["kickoff_et"].notna()]
    windows = reading_windows(
        dict(zip(scheduled["game_id"], scheduled["kickoff_et"], strict=True))
    )
    closings = Counter(
        f"{kickoff_wall_clock_et(window.start_utc):%Y-%m}" for window in windows
    )

    budget: dict[str, dict[str, int]] = {}
    for month in sorted(set(decisions) | set(closings)):
        decision_credits = CREDITS_PER_BOARD * decisions[month]
        closing_credits = CREDITS_PER_BOARD * closings[month]
        budget[month] = {
            "decision_days": decisions[month],
            "decision_credits": decision_credits,
            "closing_windows": closings[month],
            "closing_credits": closing_credits,
            "total": decision_credits + closing_credits,
            "worst_case_total": DECISION_CAPTURE_SAFETY * decision_credits
            + closing_credits,
        }
    return budget
