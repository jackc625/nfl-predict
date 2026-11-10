"""One NFL week's early-arm decision: every game and target, as a bet or a named refusal.

WHAT IT DOES. Given the model's numbers for a week and the early-week board, this module prices
each game's spread and over/under (``backtest/early_pricing.py``), keeps a priced side as a bet only
when its expected value is above the floor, sizes the week's bets together, and maps every outcome
onto the ledger's columns through the one mapping every arm uses,
``backtest.weekly_bet_list.records_to_bet_list_frame``. No game is dropped: whatever is not bet is
declined with a reason from the selector's own list.

A ROW NAMES ITS BOOK OR NAMES NO SIDE. A declined row that was priced keeps its side, line, book and
price, so a later reader can see the bet that was turned down. A row that was never priced (no
prediction, no market, or a model exactly on the consensus) carries no side and no book, because
the closing step logs an error for an early row with a side and no ``book:`` label.

THE CAPS COVER THE WEEK. ``apply_caps`` runs once over every positive-EV record of the week, both
targets together, because the per-week cap limits the week's total exposure and not each target's.

THE CANDIDATES TABLE is the working behind the rows: one row per game and target, stored in the
decision snapshot, holding what the pricing saw. An unpriced row is empty there too.

Every number comes from ``backtest/early_arm_2026.py``. Pure: no file, clock or network is touched.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd

from api.cache import RUN_MODE_FORWARD
from backtest.bet_selector import SelectionResult
from backtest.early_arm_2026 import EARLY_TARGETS, EV_FLOOR
from backtest.early_pricing import (
    PricedSide,
    Quote,
    apply_caps,
    assert_no_contradiction,
    book_label,
    price_target,
    quarter_kelly_fraction,
)
from backtest.weekly_bet_list import (
    DEFAULT_BANKROLL,
    WeeklyChainFit,
    records_to_bet_list_frame,
)
from models.key_numbers import KeyNumberTable

__all__ = ["CANDIDATE_COLUMNS", "EarlyDecision", "decide_early_week"]

# The record fields each target's model number and line go under, as the frame mapping reads them.
_MODEL_FIELD: Mapping[str, str] = {"ats": "model_spread", "ou": "model_total"}
_LINE_FIELD: Mapping[str, str] = {"ats": "closing_spread", "ou": "closing_total"}

# The candidates table, one row per game and target.
CANDIDATE_COLUMNS: tuple[str, ...] = (
    "game_id",
    "target",
    "consensus",
    "expected_close",
    "model_value",
    "bet_side",
    "sportsbook",
    "line",
    "price",
    "p_win",
    "p_push",
    "p_lose",
    "ev",
)


@dataclass(frozen=True)
class EarlyDecision:
    """One week's decision.

    Attributes:
        frame: The ``BET_LIST_COLUMNS`` rows, one per game and target, live or suppressed.
        candidates: How each game and target was priced, in ``CANDIDATE_COLUMNS``.
    """

    frame: pd.DataFrame
    candidates: pd.DataFrame


def _usable_model_value(value: float | None) -> float | None:
    """The model's number, or None when it has none a price can be read from."""
    if value is None or not math.isfinite(value):
        return None
    return float(value)


def _priced_fields(target: str, priced: PricedSide | None) -> dict[str, Any]:
    """The record fields only a priced side can fill. All empty for a row never priced."""
    return {
        "bet_side": None if priced is None else priced.side,
        _LINE_FIELD[target]: None if priced is None else priced.line,
        # The price is a real quote with no slippage, so the slipped line is the line itself.
        "slipped_line": None if priced is None else priced.line,
        "calibrated_p_side": None if priced is None else priced.p_win,
        "per_bet_ev": None if priced is None else priced.ev,
        "subpop_label": None if priced is None else book_label(priced.sportsbook),
        "selected_odds": None if priced is None else priced.price,
    }


def _candidate_row(
    game_id: str, target: str, model_value: float | None, priced: PricedSide | None
) -> dict[str, Any]:
    """One row of the candidates table. A field left out of the dict reads as empty there."""
    row: dict[str, Any] = {
        "game_id": game_id,
        "target": target,
        "model_value": model_value,
    }
    if priced is not None:
        row.update(
            {
                "consensus": priced.consensus,
                "expected_close": priced.expected_close,
                "bet_side": priced.side,
                "sportsbook": priced.sportsbook,
                "line": priced.line,
                "price": priced.price,
                "p_win": priced.p_win,
                "p_push": priced.p_push,
                "p_lose": priced.p_lose,
                "ev": priced.ev,
            }
        )
    return row


def _size_the_week(
    positive_ev: Sequence[tuple[dict[str, Any], float]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Cap the week's stakes together and split the records into bets and zero stakes.

    Args:
        positive_ev: Each positive-EV record with its uncapped fraction of the bankroll.

    Returns:
        ``(selected, zero_staked)``. The selected records carry their stake; the others carry
        their rejection reason.
    """
    capped = apply_caps(
        [(record["game_id"], fraction) for record, fraction in positive_ev]
    )
    selected: list[dict[str, Any]] = []
    zero_staked: list[dict[str, Any]] = []
    for (record, _), fraction in zip(positive_ev, capped, strict=True):
        if fraction > 0.0:
            selected.append({**record, "kelly_stake": fraction * DEFAULT_BANKROLL})
        else:
            zero_staked.append({**record, "rejection_reason": "zero_kelly_stake"})
    return selected, zero_staked


def decide_early_week(
    week_games: pd.DataFrame,
    *,
    predictions: Mapping[str, Mapping[str, float]],
    quotes: Mapping[str, Mapping[str, Sequence[Quote]]],
    tables: Mapping[str, KeyNumberTable],
    locks: Mapping[str, datetime],
    captured_at: datetime,
    decided_at: datetime,
) -> EarlyDecision:
    """Decide one week: one row per game and early target, plus the candidates table.

    Args:
        week_games: The week's games, with ``game_id``, ``season`` and ``week``.
        predictions: The model's number, by target and then game id. A game absent, or whose
            number is not finite, is declined as ``missing_prediction``.
        quotes: The board, by target and then game id. A game absent has an empty list of quotes.
        tables: The key-number table of each target.
        locks: Each game's lock.
        captured_at: The instant the board was read.
        decided_at: The instant of this decision.

    Returns:
        The stamped rows and the candidates table.

    Raises:
        KeyError: When a target is absent from ``predictions``, ``quotes`` or ``tables``, or a
            game from ``locks``.
        ValueError: When the selected bets contradict each other.
    """
    rejected: list[dict[str, Any]] = []
    positive_ev: list[tuple[dict[str, Any], float]] = []
    candidate_rows: list[dict[str, Any]] = []

    for game in week_games.to_dict("records"):
        game_id = str(game["game_id"])
        for target in EARLY_TARGETS:
            model_value = _usable_model_value(predictions[target].get(game_id))
            outcome = (
                "missing_prediction"
                if model_value is None
                else price_target(
                    target,
                    model_value,
                    quotes[target].get(game_id, []),
                    tables[target],
                    captured_at=captured_at,
                )
            )
            priced = None if isinstance(outcome, str) else outcome
            candidate_rows.append(_candidate_row(game_id, target, model_value, priced))

            record: dict[str, Any] = {
                "game_id": game_id,
                "season": int(game["season"]),
                "week": int(game["week"]),
                "target": target,
                _MODEL_FIELD[target]: model_value,
                **_priced_fields(target, priced),
                "kelly_stake": 0.0,
                "snapshot_ts": captured_at,
                "freeze_ts": locks[game_id],
                "clv": None,
            }
            if priced is not None and priced.ev > EV_FLOOR:
                fraction = quarter_kelly_fraction(
                    priced.p_win, priced.p_lose, priced.price
                )
                positive_ev.append((record, fraction))
            else:
                reason = outcome if isinstance(outcome, str) else "ev_below_floor"
                rejected.append({**record, "rejection_reason": reason})

    selected, zero_staked = _size_the_week(positive_ev)
    assert_no_contradiction(selected)

    # records_to_bet_list_frame reads only each fit's EV floor.
    fits = cast(
        "dict[str, WeeklyChainFit]",
        {target: SimpleNamespace(ev_floor_t=EV_FLOOR) for target in EARLY_TARGETS},
    )
    frame = records_to_bet_list_frame(
        SelectionResult(selected=selected, rejected=[*rejected, *zero_staked]),
        fits,
        run_mode=RUN_MODE_FORWARD,
        decided_at=decided_at,
    )
    candidates = pd.DataFrame(candidate_rows, columns=pd.Index(CANDIDATE_COLUMNS))
    return EarlyDecision(frame=frame, candidates=candidates)
