"""The per-game coverage rule of the 2026 season-close readout (Phase 38; SPEC R7).

Every scheduled game of the verdict scope (season 2026, weeks ``start_week..end_week`` of the
committed declaration, read through ``forward_ledger.declarations.load_verdict_scope`` and never
restated) lands in exactly ONE state:

* ``bet``    -- it has counted live-arm rows and at least one of them has status ``live``;
* ``no bet`` -- it has counted rows and none of them is a bet (every row ``suppressed``);
* ``missed`` -- it has no counted row at all.

A partly missed week must never read as covered: a week with any row looks attempted, so the
rule is applied per GAME and then rolled up per week, with a week that has no scheduled game
reading all zeros.

Pure functions: nothing here reads or writes a store. Nothing on the daily run and no module under
``api/`` imports this module.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from api.cache import BET_STATUS_LIVE
from forward_ledger.declarations import VerdictScope

__all__ = [
    "STATE_BET",
    "STATE_MISSED",
    "STATE_NO_BET",
    "CoverageReport",
    "GameCoverage",
    "WeekCoverage",
    "classify_coverage",
]

STATE_BET: str = "bet"
STATE_NO_BET: str = "no bet"
STATE_MISSED: str = "missed"


@dataclass(frozen=True)
class GameCoverage:
    """One scheduled game of the scope and the one state it landed in."""

    game_id: str
    week: int
    state: str
    targets: tuple[str, ...]


@dataclass(frozen=True)
class WeekCoverage:
    """One scope week's per-state game counts."""

    week: int
    scheduled: int
    bet: int
    no_bet: int
    missed: int

    @property
    def decided(self) -> int:
        """Games the live arm decided: a bet or a considered no-bet."""
        return self.bet + self.no_bet


@dataclass(frozen=True)
class CoverageReport:
    """Every scope game's state, and the per-week roll-up in week order."""

    games: tuple[GameCoverage, ...]
    weeks: tuple[WeekCoverage, ...]


def _scope_games(schedule: pd.DataFrame, scope: VerdictScope) -> pd.DataFrame:
    """The schedule's games of the scope: the declared season, weeks start..end inclusive."""
    in_scope = (
        (schedule["season"] == scope.season)
        & (schedule["week"] >= scope.start_week)
        & (schedule["week"] <= scope.end_week)
    )
    return schedule.loc[in_scope].sort_values(["week", "game_id"])


def _rows_by_game(counted: pd.DataFrame) -> dict[str, tuple[set[str], bool]]:
    """``{game_id -> (its counted rows' targets, whether any row is a bet)}``."""
    by_game: dict[str, tuple[set[str], bool]] = {}
    if counted.empty:  # an empty ledger's frame may carry no columns at all
        return by_game
    for game_id, target, status in zip(
        counted["game_id"], counted["target"], counted["status"], strict=True
    ):
        targets, has_bet = by_game.get(str(game_id), (set(), False))
        targets.add(str(target))
        by_game[str(game_id)] = (targets, has_bet or status == BET_STATUS_LIVE)
    return by_game


def classify_coverage(
    schedule: pd.DataFrame,
    scores: pd.DataFrame,
    counted: pd.DataFrame,
    scope: VerdictScope,
) -> CoverageReport:
    """Every scope game's coverage state, rolled up per week.

    Args:
        schedule: The recorded schedule (``utils.current_slate.load_recorded_schedule``).
        scores: The silver ``games`` table (``forward_ledger.settle.load_silver_games``).
        counted: The counted live-arm rows (``forward_ledger.verdict.verdict_rows``).
        scope: The committed verdict scope.

    Returns:
        One :class:`GameCoverage` per scope game (week, then game id order) and one
        :class:`WeekCoverage` per week from ``scope.start_week`` to ``scope.end_week``.
    """
    del scores  # the tracer reads no score yet; the not-played rule does
    rows_by_game = _rows_by_game(counted)

    universe = _scope_games(schedule, scope)
    games: list[GameCoverage] = []
    for game_id, week in zip(
        universe["game_id"].astype(str), universe["week"].astype(int), strict=True
    ):
        if game_id in rows_by_game:
            targets, has_bet = rows_by_game[game_id]
            state = STATE_BET if has_bet else STATE_NO_BET
            games.append(GameCoverage(game_id, week, state, tuple(sorted(targets))))
        else:
            games.append(GameCoverage(game_id, week, STATE_MISSED, ()))

    weeks: list[WeekCoverage] = []
    for week in range(scope.start_week, scope.end_week + 1):
        states = [game.state for game in games if game.week == week]
        weeks.append(
            WeekCoverage(
                week=week,
                scheduled=len(states),
                bet=states.count(STATE_BET),
                no_bet=states.count(STATE_NO_BET),
                missed=states.count(STATE_MISSED),
            )
        )
    return CoverageReport(games=tuple(games), weeks=tuple(weeks))
