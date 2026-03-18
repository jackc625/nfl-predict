"""Betting utilities for Expected Value calculations and bet sizing."""

import math
from enum import Enum

import numpy as np

from utils.probability_utils import (
    devig_probabilities,
    edge_calculation,
    moneyline_to_probability,
)


class BetType(Enum):
    """Enumeration of supported bet types."""

    MONEYLINE = "moneyline"
    SPREAD = "spread"
    TOTAL = "total"


class BettingResult:
    """Container for betting analysis results."""

    def __init__(
        self,
        bet_type: BetType,
        model_prob: float,
        market_prob: float,
        edge: float,
        expected_value: float,
        kelly_size: float | None = None,
        recommended_units: float | None = None,
    ):
        self.bet_type = bet_type
        self.model_prob = model_prob
        self.market_prob = market_prob
        self.edge = edge
        self.expected_value = expected_value
        self.kelly_size = kelly_size
        self.recommended_units = recommended_units

    def __repr__(self) -> str:
        return (
            f"BettingResult(type={self.bet_type.value}, "
            f"edge={self.edge:.3f}, ev={self.expected_value:.3f})"
        )


def calculate_moneyline_ev(
    model_prob: float,
    market_odds: int,
    stake: float = 100.0,
    devig: bool = True,
    opposite_odds: int = -110,
) -> BettingResult:
    """
    Calculate Expected Value for a moneyline bet.

    Args:
        model_prob: Model's win probability estimate
        market_odds: Market moneyline odds (American format)
        stake: Bet stake amount
        devig: Whether to remove vig from market odds
        opposite_odds: Opposite side odds for devig calculation

    Returns:
        BettingResult with EV analysis
    """
    # Convert market odds to probability
    market_prob_raw = moneyline_to_probability(market_odds)

    if devig and opposite_odds:
        opposite_prob = moneyline_to_probability(opposite_odds)
        market_prob, _ = devig_probabilities(market_prob_raw, opposite_prob)
    else:
        market_prob = market_prob_raw

    # Calculate edge
    edge = edge_calculation(model_prob, market_prob, method="additive")

    # Calculate Expected Value
    # EV = (Win Prob × Win Amount) - (Lose Prob × Lose Amount)
    win_amount = (
        stake * (abs(market_odds) / 100)
        if market_odds > 0
        else stake * (100 / abs(market_odds))
    )
    lose_amount = stake

    expected_value = (model_prob * win_amount) - ((1 - model_prob) * lose_amount)

    return BettingResult(
        bet_type=BetType.MONEYLINE,
        model_prob=model_prob,
        market_prob=market_prob,
        edge=edge,
        expected_value=expected_value,
    )


def calculate_spread_ev(
    model_margin: float,
    model_margin_std: float,
    market_spread: float,
    spread_juice: int = -110,
    stake: float = 100.0,
    side: str = "home",
) -> BettingResult:
    """
    Calculate Expected Value for a spread bet.

    Args:
        model_margin: Model's predicted margin (home team perspective)
        model_margin_std: Standard deviation of margin predictions
        market_spread: Market spread (positive = home favored)
        spread_juice: Juice on the spread bet
        stake: Bet stake amount
        side: Which side to bet ("home" or "away")

    Returns:
        BettingResult with EV analysis
    """
    # Calculate cover probability using normal distribution
    # market_spread is from home team perspective (negative = home favored)
    # model_margin is from home team perspective (positive = home wins by that much)

    if side.lower() == "home":
        # Home team covers if: actual_margin > |market_spread|
        # Example: home -3.5 (market_spread), actual margin +7 → covers
        # We want P(actual_margin > -market_spread) = P(actual_margin > 3.5)
        threshold = -market_spread  # Convert -3.5 to +3.5
        z_score = (threshold - model_margin) / model_margin_std
        model_prob = 1 - _normal_cdf(z_score)  # P(X > threshold) = 1 - CDF(threshold)
    else:
        # Away team covers if: actual_margin < market_spread
        # Example: away +3.5, actual margin -2 → covers (loses by less than 3.5)
        # We want P(actual_margin < -market_spread) = P(actual_margin < 3.5)
        threshold = -market_spread  # Convert -3.5 to +3.5
        z_score = (threshold - model_margin) / model_margin_std
        model_prob = _normal_cdf(z_score)  # P(X < threshold) = CDF(threshold)

    # Market probability from juice
    market_prob = moneyline_to_probability(spread_juice)

    # Devig assuming both sides have same juice
    market_prob_devig, _ = devig_probabilities(market_prob, market_prob)

    # Calculate edge
    edge = edge_calculation(model_prob, market_prob_devig, method="additive")

    # Calculate Expected Value for spread bet
    win_amount = stake * (100 / abs(spread_juice))
    lose_amount = stake

    expected_value = (model_prob * win_amount) - ((1 - model_prob) * lose_amount)

    return BettingResult(
        bet_type=BetType.SPREAD,
        model_prob=model_prob,
        market_prob=market_prob_devig,
        edge=edge,
        expected_value=expected_value,
    )


def calculate_total_ev(
    model_total: float,
    model_total_std: float,
    market_total: float,
    total_juice: int = -110,
    stake: float = 100.0,
    side: str = "over",
) -> BettingResult:
    """
    Calculate Expected Value for an over/under total bet.

    Args:
        model_total: Model's predicted game total
        model_total_std: Standard deviation of total predictions
        market_total: Market total line
        total_juice: Juice on the total bet
        stake: Bet stake amount
        side: Which side to bet ("over" or "under")

    Returns:
        BettingResult with EV analysis
    """
    # Calculate probability based on side
    if side.lower() == "over":
        # P(over) = P(actual_total > market_total)
        z_score = (market_total - model_total) / model_total_std
        model_prob = 1 - _normal_cdf(z_score)  # P(X > threshold) = 1 - CDF(threshold)
    else:
        # P(under) = P(actual_total < market_total)
        z_score = (market_total - model_total) / model_total_std
        model_prob = _normal_cdf(z_score)  # P(X < threshold) = CDF(threshold)

    # Market probability from juice
    market_prob = moneyline_to_probability(total_juice)

    # Devig assuming both sides have same juice
    market_prob_devig, _ = devig_probabilities(market_prob, market_prob)

    # Calculate edge
    edge = edge_calculation(model_prob, market_prob_devig, method="additive")

    # Calculate Expected Value
    win_amount = stake * (100 / abs(total_juice))
    lose_amount = stake

    expected_value = (model_prob * win_amount) - ((1 - model_prob) * lose_amount)

    return BettingResult(
        bet_type=BetType.TOTAL,
        model_prob=model_prob,
        market_prob=market_prob_devig,
        edge=edge,
        expected_value=expected_value,
    )


def calculate_kelly_sizing(
    betting_result: BettingResult,
    bankroll: float,
    odds: int,
    fraction: float = 0.25,
    max_bet_pct: float = 0.05,
) -> float:
    """
    Calculate Kelly criterion bet sizing for a betting result.

    Args:
        betting_result: BettingResult object
        bankroll: Total bankroll
        odds: American odds for the bet
        fraction: Fractional Kelly (e.g., 0.25 for quarter Kelly)
        max_bet_pct: Maximum bet as percentage of bankroll

    Returns:
        Recommended bet size in dollars
    """
    if betting_result.edge <= 0:
        return 0.0

    # Convert American odds to decimal odds
    decimal_odds = odds / 100 + 1 if odds > 0 else 100 / abs(odds) + 1

    # Kelly formula: f* = (bp - q) / b
    # where b = decimal_odds - 1, p = win probability, q = lose probability
    p = betting_result.model_prob
    q = 1 - p
    b = decimal_odds - 1

    kelly_fraction = (b * p - q) / b

    # Apply fractional Kelly and constraints
    bet_fraction = min(kelly_fraction * fraction, max_bet_pct)
    bet_fraction = max(bet_fraction, 0)  # No negative bets

    bet_size = bet_fraction * bankroll

    # Update the betting result
    betting_result.kelly_size = bet_size
    betting_result.recommended_units = bet_size / 100  # Assuming $100 units

    return bet_size


def analyze_game_betting_opportunities(
    game_data: dict,
    model_predictions: dict,
    bankroll: float = 10000.0,
    min_edge_threshold: float = 0.02,
    kelly_fraction: float = 0.25,
) -> list[BettingResult]:
    """
    Analyze all betting opportunities for a single game.

    Args:
        game_data: Dictionary with market odds and game info
        model_predictions: Dictionary with model predictions
        bankroll: Available bankroll
        min_edge_threshold: Minimum edge required to consider bet
        kelly_fraction: Fractional Kelly sizing

    Returns:
        List of viable betting opportunities
    """
    opportunities = []

    # Moneyline bets
    if "ml_home" in game_data and "wp_home" in model_predictions:
        # Home moneyline
        ml_result = calculate_moneyline_ev(
            model_prob=model_predictions["wp_home"],
            market_odds=game_data["ml_home"],
            opposite_odds=game_data.get("ml_away", -110),
        )

        if ml_result.edge >= min_edge_threshold:
            calculate_kelly_sizing(
                ml_result, bankroll, game_data["ml_home"], kelly_fraction
            )
            opportunities.append(ml_result)

        # Away moneyline
        ml_away_result = calculate_moneyline_ev(
            model_prob=1 - model_predictions["wp_home"],
            market_odds=game_data.get("ml_away", -110),
            opposite_odds=game_data["ml_home"],
        )

        if ml_away_result.edge >= min_edge_threshold:
            calculate_kelly_sizing(
                ml_away_result, bankroll, game_data.get("ml_away", -110), kelly_fraction
            )
            opportunities.append(ml_away_result)

    # Spread bets
    if "spread" in game_data and "predicted_margin" in model_predictions:
        margin_std = model_predictions.get("margin_std", 14.0)  # Default NFL margin std

        # Home spread
        spread_home_result = calculate_spread_ev(
            model_margin=model_predictions["predicted_margin"],
            model_margin_std=margin_std,
            market_spread=game_data["spread"],
            spread_juice=game_data.get("spread_juice_home", -110),
            side="home",
        )

        if spread_home_result.edge >= min_edge_threshold:
            calculate_kelly_sizing(
                spread_home_result,
                bankroll,
                game_data.get("spread_juice_home", -110),
                kelly_fraction,
            )
            opportunities.append(spread_home_result)

        # Away spread
        spread_away_result = calculate_spread_ev(
            model_margin=model_predictions["predicted_margin"],
            model_margin_std=margin_std,
            market_spread=game_data["spread"],
            spread_juice=game_data.get("spread_juice_away", -110),
            side="away",
        )

        if spread_away_result.edge >= min_edge_threshold:
            calculate_kelly_sizing(
                spread_away_result,
                bankroll,
                game_data.get("spread_juice_away", -110),
                kelly_fraction,
            )
            opportunities.append(spread_away_result)

    # Total bets
    if "total" in game_data and "predicted_total" in model_predictions:
        total_std = model_predictions.get("total_std", 10.5)  # Default NFL total std

        # Over bet
        over_result = calculate_total_ev(
            model_total=model_predictions["predicted_total"],
            model_total_std=total_std,
            market_total=game_data["total"],
            total_juice=game_data.get("total_over_juice", -110),
            side="over",
        )

        if over_result.edge >= min_edge_threshold:
            calculate_kelly_sizing(
                over_result,
                bankroll,
                game_data.get("total_over_juice", -110),
                kelly_fraction,
            )
            opportunities.append(over_result)

        # Under bet
        under_result = calculate_total_ev(
            model_total=model_predictions["predicted_total"],
            model_total_std=total_std,
            market_total=game_data["total"],
            total_juice=game_data.get("total_under_juice", -110),
            side="under",
        )

        if under_result.edge >= min_edge_threshold:
            calculate_kelly_sizing(
                under_result,
                bankroll,
                game_data.get("total_under_juice", -110),
                kelly_fraction,
            )
            opportunities.append(under_result)

    # Sort by expected value (highest first)
    opportunities.sort(key=lambda x: x.expected_value, reverse=True)

    return opportunities


def _normal_cdf(x: float) -> float:
    """
    Approximate cumulative distribution function for standard normal distribution.

    Args:
        x: Input value

    Returns:
        CDF value
    """
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def portfolio_kelly_sizing(
    betting_results: list[BettingResult],
    correlation_matrix: np.ndarray | None = None,
    bankroll: float = 10000.0,
    fraction: float = 0.25,
) -> list[float]:
    """
    Calculate optimal bet sizes for a portfolio of bets using Kelly criterion.

    Args:
        betting_results: List of BettingResult objects
        correlation_matrix: Correlation matrix between bets (optional)
        bankroll: Total bankroll
        fraction: Fractional Kelly multiplier

    Returns:
        List of optimal bet sizes
    """
    n_bets = len(betting_results)

    if n_bets == 0:
        return []

    if n_bets == 1:
        # Single bet case
        result = betting_results[0]
        # Estimate odds from edge and probability
        implied_odds = 1 / result.market_prob
        kelly_size = calculate_kelly_sizing(
            result, bankroll, int(implied_odds * 100), fraction
        )
        return [kelly_size]

    # Multi-bet portfolio optimization (simplified)
    # In practice, this would require solving a quadratic programming problem
    # For now, we'll use a simplified approach

    individual_sizes = []
    for result in betting_results:
        # Estimate American odds from market probability
        if result.market_prob >= 0.5:
            odds = int(-100 * result.market_prob / (1 - result.market_prob))
        else:
            odds = int(100 * (1 - result.market_prob) / result.market_prob)

        size = calculate_kelly_sizing(result, bankroll, odds, fraction)
        individual_sizes.append(size)

    # Scale down if total allocation exceeds reasonable limit
    total_allocation = sum(individual_sizes)
    max_allocation = bankroll * 0.25  # Max 25% of bankroll across all bets

    if total_allocation > max_allocation:
        scale_factor = max_allocation / total_allocation
        individual_sizes = [size * scale_factor for size in individual_sizes]

    return individual_sizes


def summarize_betting_session(
    betting_results: list[BettingResult], actual_outcomes: list[bool] | None = None
) -> dict:
    """
    Summarize a betting session's opportunities and results.

    Args:
        betting_results: List of betting opportunities
        actual_outcomes: List of actual bet outcomes (optional)

    Returns:
        Summary dictionary
    """
    if not betting_results:
        return {"total_bets": 0, "total_ev": 0.0, "avg_edge": 0.0}

    total_ev = sum(result.expected_value for result in betting_results)
    total_kelly_size = sum(result.kelly_size or 0 for result in betting_results)
    avg_edge = np.mean([result.edge for result in betting_results])

    summary = {
        "total_bets": len(betting_results),
        "total_ev": total_ev,
        "total_kelly_size": total_kelly_size,
        "avg_edge": avg_edge,
        "bet_type_breakdown": {},
        "edge_distribution": {
            "min": min(result.edge for result in betting_results),
            "max": max(result.edge for result in betting_results),
            "mean": avg_edge,
            "std": np.std([result.edge for result in betting_results]),
        },
    }

    # Bet type breakdown
    for bet_type in BetType:
        type_results = [r for r in betting_results if r.bet_type == bet_type]
        summary["bet_type_breakdown"][bet_type.value] = {
            "count": len(type_results),
            "total_ev": sum(r.expected_value for r in type_results),
            "avg_edge": np.mean([r.edge for r in type_results])
            if type_results
            else 0.0,
        }

    # If actual outcomes provided, calculate realized returns
    if actual_outcomes and len(actual_outcomes) == len(betting_results):
        realized_returns = []
        for _i, (result, outcome) in enumerate(
            zip(betting_results, actual_outcomes, strict=False)
        ):
            if outcome:
                # Win: return stake + winnings
                realized_returns.append(result.expected_value)
            else:
                # Loss: lose stake
                realized_returns.append(-100.0)  # Assuming $100 stake

        summary["actual_returns"] = {
            "total_return": sum(realized_returns),
            "hit_rate": sum(actual_outcomes) / len(actual_outcomes),
            "roi": sum(realized_returns) / (len(betting_results) * 100),
        }

    return summary
