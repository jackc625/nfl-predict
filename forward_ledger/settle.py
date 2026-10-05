"""Grading forward ledger rows from refreshed silver scores (Phase 34, Plan 34-09; LDGR-01).

WHY SILVER SCORES AND NOT GOLD LABELS (34-RESEARCH Q2, Pitfall 11)
-----------------------------------------------------------------
The old writer grades inside ``generate_weekly_bet_list``, which only runs on a day that has a game
tomorrow and reads the realized values from the GOLD labels the same build just produced. So the
Monday game waited until Wednesday, and the Super Bowl (week 22, in verdict scope) was never graded
at all. The daily schedule refresh already ingests silver ``games`` with results every day it
returns a season, so grading from those scores needs no gold build and no Odds API credit.

The realized values are computed with the gold label formulas EXACTLY
(``scripts/build_features.py`` ``create_target_variables``): ``home_win`` is strict binary with a
tie as 0, ``home_margin`` is home minus away, ``total_points`` is home plus away. Plan 34-09
measured them equal to the gold labels on every gold row that has one.

ONE GRADER, ONE RE-GRADE PATH, ONE SCORE LOADER
-----------------------------------------------
The outcome comes from the row's OWN target strategy (``strategy.grade``) and the payout from the
existing one-way grader ``backtest.weekly_bet_list.grade_row`` -- neither is re-implemented here.
:func:`regrade_row` is the ONE place that calls ``grade_row`` for the ledger: the settle pass below,
the correction pass (:mod:`forward_ledger.corrections`) and the verify CLI's settled-result check
(D-19) all go through it, and :func:`load_silver_games` is the one score loader they share.

Pure functions: nothing here writes. The settle pass (Plan 34-15) applies :func:`grading_updates`
through ``forward_ledger.store.commit_changes``.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from api.cache import BET_STATUS_LIVE, GRADING_STATUS_PENDING
from backtest.weekly_bet_list import (
    DEFAULT_CHAIN_FIT_PATH,
    build_strategies,
    grade_row,
    load_frozen_chain_fit,
)
from forward_ledger.canonical import ENTRY_KIND_ROW
from forward_ledger.schema import BET_LIST_GRADING_COLUMNS, LEDGER_ROW_KEY
from forward_ledger.store import LedgerEntry

__all__ = [
    "DEFAULT_SILVER_DIR",
    "SILVER_GAMES_FILENAME",
    "grading_updates",
    "live_strategies",
    "load_silver_games",
    "realized_values_from_scores",
    "regrade_row",
]

DEFAULT_SILVER_DIR: Path = Path("data/silver")
SILVER_GAMES_FILENAME: str = "games.parquet"

# The grading half reset in the in-memory copy :func:`regrade_row` grades. ``clv`` is NOT reset:
# it is the decision-time CLV, which ``grade_row`` carries through unchanged.
_RESET_GRADING: dict[str, Any] = {
    "grading_status": GRADING_STATUS_PENDING,
    "outcome": None,
    "payout_flat": None,
    "realized_units": None,
    "graded_at": None,
}


def load_silver_games(silver_dir: Path | str = DEFAULT_SILVER_DIR) -> pd.DataFrame:
    """The silver ``games`` table, READ ONLY -- the one score loader the ledger's passes share.

    Raises:
        FileNotFoundError: the store is absent, named by its path. Never an empty frame: "no
            scores" read as "no games" would let a pass report that nothing needed grading.
    """
    path = Path(silver_dir) / SILVER_GAMES_FILENAME
    if not path.exists():
        msg = f"the silver games store {path.as_posix()} does not exist; no score can be read"
        raise FileNotFoundError(msg)
    return pd.read_parquet(path)


def realized_values_from_scores(games: pd.DataFrame) -> dict[str, dict[str, float]]:
    """``{target -> {game_id -> realized value}}`` for every game with both scores recorded.

    The gold label formulas exactly (``create_target_variables``): WP the strict-binary home win
    (a tie is 0), ATS the home margin, O/U the total points. A game with either score missing is
    absent from every target, so its rows stay ``pending``.
    """
    home = pd.to_numeric(games["home_score"], errors="coerce")
    away = pd.to_numeric(games["away_score"], errors="coerce")
    played = home.notna() & away.notna()
    game_ids = games.loc[played, "game_id"].astype(str)
    home, away = home[played], away[played]

    labels = {
        "wp": (home > away).astype(int),
        "ats": home - away,
        "ou": home + away,
    }
    return {
        target: {
            game_id: float(value)
            for game_id, value in zip(game_ids, values, strict=True)
        }
        for target, values in labels.items()
    }


def live_strategies(
    chain_fit_path: Path | str = DEFAULT_CHAIN_FIT_PATH,
) -> dict[str, Any]:
    """``{target -> strategy}``: the same registry ``generate_weekly_bet_list`` builds and grades with."""
    strategies = build_strategies(load_frozen_chain_fit(chain_fit_path))
    return {strategy.target: strategy for strategy in strategies}


def regrade_row(
    row: Mapping[str, Any],
    strategy: Any,
    realized_value: float,
    *,
    graded_at: datetime,
) -> dict[str, Any]:
    """The six grading values ``grade_row`` gives *row* against *realized_value* -- THE re-grade path.

    The outcome is the row's own target strategy's; the payout is ``grade_row``'s, applied to an
    in-memory COPY of *row* with its grading half reset to ``pending``. So a SETTLED row can be
    re-graded (a correction, the verify CLI's check) without ``AlreadyGradedError`` and without
    anything being written back: *row* itself is never mutated.

    Returns:
        ``{grading_status, outcome, clv, payout_flat, realized_units, graded_at}`` with
        ``graded_at`` the datetime given.
    """
    outcome = strategy.grade(
        {
            "bet_side": row.get("bet_side"),
            "slipped_line": row.get("slipped_line"),
            "_actual_total": realized_value,
        }
    )
    graded = grade_row({**row, **_RESET_GRADING}, outcome, graded_at=graded_at)
    return {name: graded[name] for name in BET_LIST_GRADING_COLUMNS}


def _entry_row(entry: LedgerEntry) -> dict[str, Any]:
    """A row entry as one flat bet-list row: its immutable half and its grading half."""
    return {**entry.immutable, **(entry.grading or {})}


def _utc_text(instant: datetime) -> str:
    if instant.tzinfo is None:
        msg = f"graded_at {instant!r} is naive; a settlement instant must be tz-aware"
        raise ValueError(msg)
    return instant.astimezone(UTC).isoformat()


def grading_updates(
    entries: Sequence[LedgerEntry],
    strategies: Mapping[str, Any],
    realized: Mapping[str, Mapping[str, float]],
    graded_at: datetime,
) -> dict[tuple[Any, ...], dict[str, Any]]:
    """The grading half to set on every live, pending row whose game now has a result.

    Suppressed rows, rows already settled and rows whose game has no result yet produce nothing,
    so a re-run after the updates were applied proposes nothing.

    Returns:
        ``LEDGER_ROW_KEY`` values -> the :func:`regrade_row` values, ``graded_at`` rendered as an
        ISO 8601 UTC string for the JSON store.

    Raises:
        ValueError: a row's target has no registered strategy (never guessed), or *graded_at* is
            naive.
    """
    settled_at = _utc_text(graded_at)
    updates: dict[tuple[Any, ...], dict[str, Any]] = {}
    for entry in entries:
        if entry.kind != ENTRY_KIND_ROW:
            continue
        row = _entry_row(entry)
        pending = (
            row.get("grading_status") or GRADING_STATUS_PENDING
        ) == GRADING_STATUS_PENDING
        if not pending or row.get("status") != BET_STATUS_LIVE:
            continue
        target = str(row["target"])
        value = realized.get(target, {}).get(str(row["game_id"]))
        if value is None:
            continue
        strategy = strategies.get(target)
        if strategy is None:
            msg = (
                f"no strategy registered for target {target!r}; a ledger row cannot be graded by "
                "a rule that is not present, and guessing one would book a result under the "
                "wrong target's convention."
            )
            raise ValueError(msg)
        values = regrade_row(row, strategy, value, graded_at=graded_at)
        values["graded_at"] = settled_at
        updates[tuple(row[name] for name in LEDGER_ROW_KEY)] = values
    return updates
