"""Early-week pricing: which quote to take, what it is worth, how much to stake.

WHAT IT DOES. Given the model's number for a game and the quotes on the early-week board, this
module finds what the market thinks (the median line of the usable quotes), decides which side the
model is on, and values that side at every usable quote. The quote worth the most is the price.
Then it sizes the bet with quarter-Kelly and holds the week's stakes inside the two caps.

WHICH QUOTES MAY BE THE PRICE. Picking the best of several quotes actively picks outliers, so a
quote has to earn its place first: a regulated book, both prices present, a bookmaker time no more
than ``QUOTE_MAX_AGE_HOURS`` before the board was read, and a line the key-number table can price.
A quote with no bookmaker time cannot be shown to be fresh, so it is not usable. A time later than
the read is the bookmaker's clock running ahead, not staleness, so it is. Finally, a quote further
than ``MAX_QUOTE_DISTANCE_POINTS`` from the median of those is dropped, so one book's wrong number
can neither move the consensus nor be chosen as a large fake edge.

THE SCALE. ``up`` means over, or the home side covering. A spread ``line`` is on the ledger's
scale, positive when the home team is favoured, which is the same scale as the model's margin and
the game's final margin. So on both targets, finishing above the line is what ``up`` wins on.

Every number comes from ``backtest/early_arm_2026.py``. Pure: no file, clock or network is touched.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from statistics import median
from typing import Any, cast

import pandas as pd

from backtest.ats_ev_chain import ATS_SIDES
from backtest.early_arm_2026 import (
    BOOK_LABEL_PREFIX,
    EXPECTED_MOVE_SLOPE,
    KELLY_FRACTION,
    MAX_QUOTE_DISTANCE_POINTS,
    MIN_BOOKS,
    PER_GAME_CAP,
    PER_WEEK_CAP,
    QUOTE_MAX_AGE_HOURS,
    REGULATED_BOOKS,
    SLIPPAGE_POINTS,
)
from backtest.ou_ev_chain import american_to_payout
from backtest.selector_strategies import OU_SIDES
from backtest.wp_ev_chain import WP_SIDES
from models.key_numbers import KeyNumberTable, outcome_chance

__all__ = [
    "PricedSide",
    "Quote",
    "apply_caps",
    "assert_no_contradiction",
    "book_label",
    "price_target",
    "quarter_kelly_fraction",
    "usable_quotes",
]

# Each early target's (up, down) side names.
_SIDES_BY_TARGET: Mapping[str, tuple[str, str]] = {"ats": ATS_SIDES, "ou": OU_SIDES}

# The team each moneyline or spread side backs. A total backs neither team.
_TEAM_BACKED: Mapping[str, str] = dict(
    zip(WP_SIDES + ATS_SIDES, ("home", "away", "home", "away"), strict=True)
)


@dataclass(frozen=True)
class Quote:
    """One book's market for one game and target, as read from the board.

    Attributes:
        sportsbook: The book's board key.
        line: The total, or the spread on the ledger's scale (positive when home is favoured).
        price_up: The American price of the over or the home side; None when the book has none.
        price_down: The American price of the under or the away side; None when the book has none.
        last_update: The bookmaker's own time for this quote; None when the board gave none.
    """

    sportsbook: str
    line: float
    price_up: int | None
    price_down: int | None
    last_update: datetime | None


@dataclass(frozen=True)
class PricedSide:
    """The side the model is on, valued at the best usable quote.

    Attributes:
        side: ``over``, ``under``, ``home_cover`` or ``away_cover``.
        sportsbook: The book the price was taken from.
        line: That book's line.
        price: That book's American price for this side.
        consensus: The median line of the usable quotes.
        expected_close: Where the line is expected to close.
        p_win: The chance this side wins.
        p_push: The chance the bet is refunded.
        p_lose: The chance this side loses.
        ev: Expected profit per unit staked.
    """

    side: str
    sportsbook: str
    line: float
    price: int
    consensus: float
    expected_close: float
    p_win: float
    p_push: float
    p_lose: float
    ev: float


def _is_price(value: int | None) -> bool:
    """Whether a price is present. A price read from a DataFrame arrives missing as NaN or NA."""
    return value is not None and not pd.isna(value)


def _is_fresh(quote: Quote, captured_at: datetime) -> bool:
    """Whether the bookmaker's own time shows the quote is recent enough to be a price.

    A missing time read from a DataFrame arrives as NaT, which compares false below.
    """
    if quote.last_update is None:
        return False
    return captured_at - quote.last_update <= timedelta(hours=QUOTE_MAX_AGE_HOURS)


def usable_quotes(quotes: Sequence[Quote], captured_at: datetime) -> list[Quote]:
    """The quotes from regulated books with both prices and a fresh bookmaker time.

    Args:
        quotes: One game and target's quotes.
        captured_at: The instant the board was read.

    Returns:
        The usable quotes, in their original order. ``price_target`` also drops any whose line
        the key-number table cannot price, and any far from the others.
    """
    return [
        quote
        for quote in quotes
        if quote.sportsbook in REGULATED_BOOKS
        and _is_price(quote.price_up)
        and _is_price(quote.price_down)
        and _is_fresh(quote, captured_at)
    ]


def _line_inside_table(line: float, table: KeyNumberTable) -> bool:
    """Whether the table can price this line. False for NaN, and for a fabricated 0.0 total."""
    return table.low < line < table.low + len(table.factors) - 1


def _quotes_near_the_median(quotes: Sequence[Quote]) -> list[Quote]:
    """The quotes within ``MAX_QUOTE_DISTANCE_POINTS`` of these quotes' median line."""
    if not quotes:
        return []
    center = median(quote.line for quote in quotes)
    return [
        quote
        for quote in quotes
        if abs(quote.line - center) <= MAX_QUOTE_DISTANCE_POINTS
    ]


def _value_quote(
    quote: Quote,
    *,
    sides: tuple[str, str],
    going_up: bool,
    consensus: float,
    expected_close: float,
    table: KeyNumberTable,
) -> PricedSide:
    """One quote's value for the side being bet."""
    threshold = (
        quote.line + SLIPPAGE_POINTS if going_up else quote.line - SLIPPAGE_POINTS
    )
    chance = outcome_chance(
        table, market_line=consensus, center=expected_close, threshold=threshold
    )
    p_win, p_lose = (
        (chance.above, chance.below) if going_up else (chance.below, chance.above)
    )
    # usable_quotes has already dropped every quote missing either price.
    price = cast("int", quote.price_up if going_up else quote.price_down)
    return PricedSide(
        side=sides[0] if going_up else sides[1],
        sportsbook=quote.sportsbook,
        line=quote.line,
        price=price,
        consensus=consensus,
        expected_close=expected_close,
        p_win=p_win,
        p_push=chance.at,
        p_lose=p_lose,
        ev=p_win * american_to_payout(price) - p_lose,
    )


def price_target(
    target: str,
    model_value: float,
    quotes: Sequence[Quote],
    table: KeyNumberTable,
    *,
    captured_at: datetime,
) -> PricedSide | str:
    """Value the model's side of one game and target at the best usable quote.

    Args:
        target: ``ats`` or ``ou``.
        model_value: The model's margin or total, on the same scale as the quotes' lines.
        quotes: The game's quotes for this target.
        table: The target's key-number table.
        captured_at: The instant the board was read.

    Returns:
        The priced side, or a rejection code: ``missing_snapshot`` with fewer than ``MIN_BOOKS``
        quotes left once the unusable and the far ones are dropped, ``no_bet_side`` when the
        model equals the consensus.

    Raises:
        KeyError: When ``target`` has no early arm.
        ValueError: When ``model_value`` is not a finite number. NaN compares false both ways,
            so it would otherwise be priced as a real ``down`` side with NaN chances.
    """
    sides = _SIDES_BY_TARGET[target]
    if not math.isfinite(model_value):
        msg = f"model_value must be a finite number, got {model_value!r}"
        raise ValueError(msg)

    inside_table = [
        quote
        for quote in usable_quotes(quotes, captured_at)
        if _line_inside_table(quote.line, table)
    ]
    priceable = _quotes_near_the_median(inside_table)
    if len(priceable) < MIN_BOOKS:
        return "missing_snapshot"

    consensus = median(quote.line for quote in priceable)
    if model_value == consensus:
        return "no_bet_side"

    expected_close = consensus + EXPECTED_MOVE_SLOPE[target] * (model_value - consensus)
    valued = [
        _value_quote(
            quote,
            sides=sides,
            going_up=model_value > consensus,
            consensus=consensus,
            expected_close=expected_close,
            table=table,
        )
        for quote in priceable
    ]
    return min(valued, key=lambda priced: (-priced.ev, priced.sportsbook))


def quarter_kelly_fraction(p_win: float, p_lose: float, american_odds: int) -> float:
    """The fraction of the bankroll to stake: quarter-Kelly, with no other multiplier.

    Dividing by ``p_win + p_lose`` measures the edge only on the outcomes where money moves, so a
    chance of a push is not counted as a chance of losing.
    """
    payout = american_to_payout(american_odds)
    full_kelly = (payout * p_win - p_lose) / (payout * (p_win + p_lose))
    return KELLY_FRACTION * max(0.0, full_kelly)


def _cap_scale(total: float, cap: float) -> float:
    """The multiplier that brings ``total`` down to ``cap``; 1.0 when it is already within it."""
    return cap / total if total > cap else 1.0


def apply_caps(stakes: Sequence[tuple[str, float]]) -> list[float]:
    """Hold a week's stakes inside the per-game cap, then the per-week cap.

    Args:
        stakes: One ``(game_id, fraction of bankroll)`` pair per bet, for one NFL week.

    Returns:
        The capped fractions, in the same order. Each cap scales the stakes it covers
        proportionally.
    """
    capped = [fraction for _, fraction in stakes]
    positions_by_game: defaultdict[str, list[int]] = defaultdict(list)
    for position, (game_id, _) in enumerate(stakes):
        positions_by_game[game_id].append(position)

    for positions in positions_by_game.values():
        game_scale = _cap_scale(sum(capped[p] for p in positions), PER_GAME_CAP)
        for position in positions:
            capped[position] *= game_scale

    week_scale = _cap_scale(sum(capped), PER_WEEK_CAP)
    return [fraction * week_scale for fraction in capped]


def assert_no_contradiction(bets: Sequence[Mapping[str, Any]]) -> None:
    """Refuse a set of bets that argue with each other on one game.

    Two bets contradict when they take both sides of one target, or when they back opposite teams
    across the moneyline and the spread. A total backs no team, so a spread and a total never
    contradict; with moneyline out of the early arm this check cannot fire on its bets.

    Args:
        bets: The selected bets, each with ``game_id``, ``target`` and ``bet_side``.

    Raises:
        ValueError: Naming the game and the two sides.
    """
    sides_by_target: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    teams_by_game: defaultdict[str, set[str]] = defaultdict(set)
    for bet in bets:
        game_id, target, side = bet["game_id"], bet["target"], bet["bet_side"]
        sides_by_target[(game_id, target)].add(side)
        if side in _TEAM_BACKED:
            teams_by_game[game_id].add(_TEAM_BACKED[side])

    for (game_id, target), sides in sides_by_target.items():
        if len(sides) > 1:
            msg = f"both sides of {target} on {game_id}: {sorted(sides)}"
            raise ValueError(msg)
    for game_id, teams in teams_by_game.items():
        if len(teams) > 1:
            msg = f"bets on {game_id} back opposite teams: {sorted(teams)}"
            raise ValueError(msg)


def book_label(sportsbook: str) -> str:
    """The value a row's ``eligibility_label`` holds to name the book it was priced at."""
    return f"{BOOK_LABEL_PREFIX}{sportsbook}"
