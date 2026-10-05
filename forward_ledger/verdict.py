"""The 2026 verdict's row set and its start week W (Phase 34, Plan 34-09; LDGR-10, D-06, D-15).

THE VERDICT READER
------------------
:func:`verdict_rows` is the one way the rows a verdict counts are read. It REFUSES when no
declaration exists (:class:`~forward_ledger.declarations.VerdictScopeUndeclaredError`) rather than
falling back to a default scope -- a verdict over a scope nobody declared is the after-the-fact
choice the pre-registration exists to forbid. Under a declaration it keeps only the declared
season and counted arm, rows labelled ``verdict``, and weeks ``start_week..end_week`` inclusive
(week 22, the Super Bowl, is in; no week 23 exists in it). Every pre-verdict row and every shadow
row is therefore outside every verdict figure.

Each kept row counts the outcome IN FORCE (D-06): the latest correction entry for its key, else
its own grade (``forward_ledger.corrections.in_force_outcomes``). The original grade stays on the
frame as ``original_grading_status``, and ``corrected`` marks the rows a correction changed.

THE START WEEK (D-15, 34-RESEARCH K and Pitfall 13)
---------------------------------------------------
W is computed from the ledger's own contents at go-live, not guessed: the first week after every
week that already has a row. Every such week is pre-verdict, so a week holding any old-writer
(NULL-stamp) row can never be W, and a go-live in the middle of a week makes W the next one. The
declaration must be on disk before W's first DECISION RUN (17:00 ET, one hour before the 18:00
lock of ``utils.game_lock``), so a week whose first run is not safely in the future -- less than
``DECLARATION_SAFETY_MARGIN`` away -- is skipped. Past week 22 there is no W to declare.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import pandas as pd

import utils.game_lock as lock_rule
from forward_ledger.canonical import ENTRY_KIND_ROW
from forward_ledger.corrections import in_force_outcomes
from forward_ledger.declarations import VerdictScope, VerdictScopeUndeclaredError
from forward_ledger.schema import LEDGER_ROW_KEY, VERDICT_SCOPE_VERDICT
from forward_ledger.store import LEDGER_SEASON, LedgerEntry, entries_to_frame
from utils.date_utils import NFL_TOTAL_WEEKS

__all__ = [
    "DECISION_RUN_LEAD",
    "DECLARATION_SAFETY_MARGIN",
    "StartWeek",
    "StartWeekUnavailableError",
    "compute_start_week",
    "verdict_rows",
]

# The daily decision run starts at 17:00 ET, one hour before the 18:00 ET lock it decides for.
DECISION_RUN_LEAD: timedelta = timedelta(hours=1)

# How far in the future W's first decision run must still be when W is computed, so the
# declaration can be committed and pushed before that run stamps the first week-W row.
DECLARATION_SAFETY_MARGIN: timedelta = timedelta(hours=3)

# The in-force columns overlaid on each verdict row (the rest of the grading half is unchanged).
_IN_FORCE_COLUMNS: tuple[str, ...] = (
    "grading_status",
    "outcome",
    "payout_flat",
    "realized_units",
)


class StartWeekUnavailableError(Exception):
    """No week can be W: the candidate passed week 22, or its schedule has no games yet."""


@dataclass(frozen=True)
class StartWeek:
    """W, its first lock and the decision run before it, both as UTC instants."""

    week: int
    first_lock_utc: datetime
    first_decision_run_utc: datetime


def verdict_rows(
    entries: Sequence[LedgerEntry], scope: VerdictScope | None
) -> pd.DataFrame:
    """The rows the declared verdict counts, each carrying its IN-FORCE outcome.

    Args:
        entries: The ledger, in file order.
        scope: The committed declaration (``forward_ledger.declarations.load_verdict_scope``).

    Returns:
        The bet-list frame of the kept rows in file order, with ``grading_status``, ``outcome``,
        ``payout_flat`` and ``realized_units`` set to the in-force values, plus
        ``original_grading_status`` (the row's own grade) and ``corrected`` (bool).

    Raises:
        VerdictScopeUndeclaredError: *scope* is None. Never a default scope.
    """
    if scope is None:
        msg = (
            "no verdict-scope declaration was given; a verdict is never computed against a "
            "default scope (LDGR-10, D-15)."
        )
        raise VerdictScopeUndeclaredError(msg)

    frame = entries_to_frame(entries)
    kept = frame[
        (frame["season"] == scope.season)
        & (frame["arm"] == scope.counted_arm)
        & (frame["verdict_scope"] == VERDICT_SCOPE_VERDICT)
        & (frame["week"] >= scope.start_week)
        & (frame["week"] <= scope.end_week)
    ].copy()

    in_force = in_force_outcomes(entries)
    states = [
        in_force[tuple(record)]
        for record in kept[list(LEDGER_ROW_KEY)].itertuples(index=False)
    ]
    kept["original_grading_status"] = kept["grading_status"]
    for column in _IN_FORCE_COLUMNS:
        kept[column] = pd.Series(
            [getattr(state, column) for state in states],
            index=kept.index,
            dtype=kept[column].dtype,
        )
    kept["corrected"] = pd.Series(
        [state.corrected for state in states], index=kept.index, dtype=bool
    )
    return kept.reset_index(drop=True)


def _first_lock(games: pd.DataFrame, week: int) -> datetime:
    """The earliest ``utils.game_lock`` lock among *week*'s scheduled games."""
    week_games = games[games["week"] == week]
    if week_games.empty:
        msg = (
            f"week {week} has no scheduled games in the schedule given, so it has no first "
            "lock and cannot be declared as W"
        )
        raise StartWeekUnavailableError(msg)
    return min(
        lock_rule.game_lock(kickoff, game_id=str(game_id))
        for game_id, kickoff in zip(
            week_games["game_id"], week_games["kickoff_et"], strict=True
        )
    )


def compute_start_week(
    entries: Sequence[LedgerEntry], games: pd.DataFrame, now: datetime
) -> StartWeek:
    """W from the ledger's contents and the schedule (D-15).

    Args:
        entries: The ledger, in file order.
        games: A silver-``games``-shaped schedule (``game_id``, ``season``, ``week``,
            ``kickoff_et``); only the ledger season's games are read.
        now: The instant W is being computed at, tz-aware.

    Returns:
        The first week after every week that already has a ledger row whose first decision run
        is at least ``DECLARATION_SAFETY_MARGIN`` after *now*.

    Raises:
        StartWeekUnavailableError: the candidate passes week 22, or a candidate week has no
            scheduled games.
        ValueError: *now* is naive.
    """
    if now.tzinfo is None:
        msg = f"now {now!r} is naive; W is computed against a tz-aware instant"
        raise ValueError(msg)

    weeks_with_rows = [
        int(entry.immutable["week"])
        for entry in entries
        if entry.kind == ENTRY_KIND_ROW and entry.immutable["season"] == LEDGER_SEASON
    ]
    candidate = max(weeks_with_rows, default=0) + 1
    season_games = games[games["season"] == LEDGER_SEASON]
    deadline = now + DECLARATION_SAFETY_MARGIN

    while candidate <= NFL_TOTAL_WEEKS:
        first_lock = _first_lock(season_games, candidate)
        first_run = first_lock - DECISION_RUN_LEAD
        if first_run >= deadline:
            return StartWeek(
                week=candidate,
                first_lock_utc=first_lock.astimezone(UTC),
                first_decision_run_utc=first_run.astimezone(UTC),
            )
        candidate += 1

    msg = (
        f"no start week remains: the candidate passed week {NFL_TOTAL_WEEKS}, the last week of "
        f"the {LEDGER_SEASON} verdict scope"
    )
    raise StartWeekUnavailableError(msg)
