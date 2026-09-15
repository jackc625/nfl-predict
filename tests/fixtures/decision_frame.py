"""The shared, READ-ONLY selector fixtures the Plan 33-07 decision-frame tests drive.

WHY ONE MODULE AND NOT ONE PER TEST FILE
-----------------------------------------
Two modules need the same thing -- a week's candidates, a registered strategy per CANONICAL
target, and a frozen fit -- and they need it to be the SAME thing. ``test_empty_week_refusals``
asserts that a no-edge week trips neither refusal; ``test_weekly_decision_frame`` asserts that
the pure seam and the persisting path produce identical statuses. If those two files built
their own candidates, "identical" would be a claim about two fixtures rather than about one
decision path.

THIS MODULE READS AND COMPUTES. IT NEVER WRITES.
-------------------------------------------------
Every value here is built in memory. Nothing opens a file, creates a directory, or calls a
storage helper, so no test importing it can reach a production store through it.

THE TARGETS ARE THE CANONICAL THREE, DELIBERATELY
---------------------------------------------------
``backtest.weekly_bet_list`` maps each record onto the locked 29-column schema through
``_MODEL_COLUMN`` / ``_LINE_COLUMN``, both keyed by ``'wp'`` / ``'ats'`` / ``'ou'``. A
test-local target code (the shape ``tests/integration/test_bet_list_completeness.py`` uses,
correctly, for a selector-only claim) would raise a KeyError the moment the mapping ran, so
these strategies wear the canonical codes and the canonical market column names. They are
still test-local and make NO claim about how a real target prices a bet -- their only job is
to put a complete, Protocol-conforming registry in front of the real ``BetSelector``.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from backtest.selector_strategies import TargetStrategy
from backtest.weekly_bet_list import CANONICAL_TARGETS, WeeklyChainFit

SEASON = 2026
WEEK = 2
N_GAMES = 4

# A Sunday kickoff and its OWN preceding Friday 6 PM Eastern freeze. SPEC R6 defines
# at-freeze as fresh, so every fixture row is fresh and no row is suppressed for staleness
# by accident -- which would silently turn every "no edge" assertion into a staleness one.
SUNDAY_GAMEDAY = "2026-09-20"
FREEZE_AT_SUNDAY = "2026-09-18T18:00:00-04:00"

# The market and model columns each canonical target's candidate row must carry. Spelled the
# way ``backtest.weekly_bet_list._MODEL_COLUMN`` and ``_MARKET_COLUMNS`` spell them, because
# the mapping onto the locked schema reads those exact names off the decision record.
TARGET_MARKET_FIELDS: dict[str, tuple[str, ...]] = {
    "wp": ("model_prob", "ml_home", "ml_away"),
    "ats": ("model_spread", "closing_spread"),
    "ou": ("model_total", "closing_total"),
}

TARGET_MARKET_VALUES: dict[str, dict[str, Any]] = {
    "wp": {"model_prob": 0.61, "ml_home": -130.0, "ml_away": 110.0},
    "ats": {"model_spread": -3.5, "closing_spread": -2.5},
    "ou": {"model_total": 41.0, "closing_total": 45.0},
}


class MiniStrategy:
    """A minimal, fully implemented ``TargetStrategy`` for one canonical target.

    Plan 31-06 forbids a registerable PRODUCTION strategy whose methods are unimplemented.
    This one is test-local, complete, and checked against the Protocol by ``isinstance``
    before it is handed to the selector.
    """

    def __init__(self, target: str, market_fields: tuple[str, ...]) -> None:
        self.target = target
        self.required_market_fields = market_fields

    def resolve_bet_side(self, row: dict[str, Any]) -> str | None:
        return row.get(f"{self.target}_side", "home")

    def eligibility(self, row: dict[str, Any], bet_side: str | None) -> str | None:
        return None if bet_side is not None else "not_subpop"

    def eligibility_label(self, row: dict[str, Any], bet_side: str | None) -> str:
        return "all"

    def side_probability(
        self, row: dict[str, Any], bet_side: str
    ) -> tuple[float, float]:
        return float(row.get(f"{self.target}_p", 0.60)), float(
            row[self.required_market_fields[-1]]
        )

    def decision_extras(
        self, row: dict[str, Any], bet_side: str | None
    ) -> dict[str, Any]:
        return {}

    def grade(self, record: dict[str, Any]) -> bool | None:
        return None


def mini_strategies() -> list[Any]:
    """One conforming strategy per canonical target, in registry order."""
    strategies = [
        MiniStrategy(target, TARGET_MARKET_FIELDS[target])
        for target in CANONICAL_TARGETS
    ]
    for strategy in strategies:
        assert isinstance(strategy, TargetStrategy), strategy.target
    return strategies


def fits_with_floor(ev_floor_t: float) -> dict[str, WeeklyChainFit]:
    """A frozen fit per canonical target, sharing one EV floor.

    The floor is the dial these tests turn: a floor of 0.0 admits the fixture's candidates
    and a floor above anything it can price rejects every one of them, which is what makes a
    genuine no-edge week constructible without weakening any real threshold.
    """
    return {
        target: WeeklyChainFit(
            target=target,
            ev_floor_t=ev_floor_t,
            frozen_sd=13.0,
            season_bias_by_season={SEASON: -1.0},
        )
        for target in CANONICAL_TARGETS
    }


def chain_fit_record(ev_floor_t: float) -> dict[str, Any]:
    """The on-disk shape ``load_frozen_chain_fit`` reads, as a plain dict.

    Returned rather than written: a fixture module that wrote its own inputs would be the
    COLD-05 violation class, and the caller already has a ``tmp_path`` to write it into.

    IT CARRIES NO BIAS FOR ``SEASON`` (Plan 33-17, D33-21), and that absence is the point. The
    loader now OVERLAYS the committed Phase-33 bias for exactly this season, so a value written
    here would be a SECOND source for it -- and since the fixture's ``-1.0`` is a round number
    chosen for legibility rather than a measurement, the two would disagree and the load would
    refuse by name, which is precisely what the disagreement guard is for. Letting the overlay
    supply it keeps this fixture a record of what the RUN RECORD holds, which is what it is for.

    ``fits_with_floor`` above is unaffected: it builds ``WeeklyChainFit`` objects directly, never
    passing through the loader, so its ``-1.0`` stays the dial the pricing tests turn.
    """
    return {
        "tune_fit": {
            target: {
                "ev_floor_t": ev_floor_t,
                "frozen_sd": 13.0,
                "season_bias_by_season": {},
            }
            for target in CANONICAL_TARGETS
        }
    }


def week_schedule() -> pd.DataFrame:
    """The week's spine, in the shape ``build_weekly_candidates`` returns it."""
    return pd.DataFrame(
        {
            "game_id": [_game_id(index) for index in range(N_GAMES)],
            "season": [SEASON] * N_GAMES,
            "week": [WEEK] * N_GAMES,
            "gameday": [SUNDAY_GAMEDAY] * N_GAMES,
        }
    )


def week_candidates(priced_games: int = N_GAMES) -> pd.DataFrame:
    """One candidate row per (game, target) for the FIRST *priced_games* games.

    Leaving some games unpriced is the realistic mix: the selector builds a skeleton for
    them and suppresses the pair as ``missing_snapshot``, which is what makes the partition
    a claim about the SCHEDULE rather than about the odds join.
    """
    rows: list[dict[str, Any]] = []
    for index in range(priced_games):
        for target in CANONICAL_TARGETS:
            rows.append(
                {
                    "game_id": _game_id(index),
                    "season": SEASON,
                    "week": WEEK,
                    "target": target,
                    "gameday": SUNDAY_GAMEDAY,
                    "snapshot_ts": FREEZE_AT_SUNDAY,
                    "sportsbook": "consensus",
                    "is_live": False,
                    **TARGET_MARKET_VALUES[target],
                }
            )
    return pd.DataFrame(rows)


def builder_stub_frames(
    priced_games: int = N_GAMES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The ``(candidates, schedule)`` 2-tuple ``build_weekly_candidates`` returns."""
    return week_candidates(priced_games), week_schedule()


def _game_id(index: int) -> str:
    return f"{SEASON}_{WEEK:02d}_A{index:02d}@H{index:02d}"
