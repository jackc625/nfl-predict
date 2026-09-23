"""Probability and odds conversion utilities.

TWO CONVERTERS WERE DELETED HERE (Plan 33.2-21, D33.2-09). Recorded rather than silently
dropped, because a reader who remembers them should be able to find out where they went.

``convert_spread_to_moneyline`` turned a spread into a win probability through a hardcoded
``1 / (1 + exp(-spread * 0.25))``. That 0.25 is roughly 65% steeper than the owned pre-lock
spreads support: MEASURED over 1,342 graded 2020-2024 games the no-intercept slope is
0.1512, so at a +7 spread the deleted function claimed 0.85 where the data says 0.73. A
converter that steep manufactures an apparent edge at EVERY spread -- the exact failure
D33.2-11's two-test rule exists to catch. It had zero call sites.

``convert_total_to_over_under_ml`` (a "2% per point" linear rule, equally unfitted) had zero
call sites too and went with it.

The replacement is ``models.market_probability`` -- FITTED, walk-forward, on lines owned
before each game's lock, and carrying a versioned artifact with its slope, its training
seasons, its input digest and its fit time. It lives under ``models/`` and not here on
purpose: a fitted model input sitting in ``utils/`` as a free function looks like
arithmetic, which is precisely how the 0.25 constant came to sit unquestioned in this file.
One answer on disk (D30-02).
"""

import math

import numpy as np


def moneyline_to_probability(moneyline: int) -> float:
    """
    Convert American moneyline odds to implied probability.

    Args:
        moneyline: American moneyline (e.g., -110, +150)

    Returns:
        Implied probability (0.0 to 1.0)
    """
    if moneyline < 0:
        # Negative moneyline (favorite)
        return abs(moneyline) / (abs(moneyline) + 100)
    # Positive moneyline (underdog)
    return 100 / (moneyline + 100)


def probability_to_moneyline(probability: float) -> int:
    """
    Convert probability to American moneyline odds.

    Args:
        probability: Probability (0.0 to 1.0)

    Returns:
        American moneyline odds
    """
    if probability <= 0 or probability >= 1:
        raise ValueError("Probability must be between 0 and 1")

    if probability >= 0.5:
        # Favorite (negative moneyline)
        return int(-100 * probability / (1 - probability))
    # Underdog (positive moneyline)
    return int(100 * (1 - probability) / probability)


def devig_probabilities(
    prob_a: float, prob_b: float, method: str = "proportional"
) -> tuple[float, float]:
    """
    Remove vig (overround) from two-way betting probabilities.

    Args:
        prob_a: Probability of outcome A
        prob_b: Probability of outcome B
        method: Devig method ('proportional', 'worst_case', 'power')

    Returns:
        Tuple of devigged probabilities (prob_a_fair, prob_b_fair)
    """
    total = prob_a + prob_b

    if total <= 1.0:
        # No vig to remove
        return prob_a, prob_b

    if method == "proportional":
        # Proportional method (most common)
        prob_a_fair = prob_a / total
        prob_b_fair = prob_b / total

    elif method == "worst_case":
        # Worst-case scenario (removes vig from the favorite)
        if prob_a > prob_b:
            prob_a_fair = 1 - prob_b
            prob_b_fair = prob_b
        else:
            prob_a_fair = prob_a
            prob_b_fair = 1 - prob_a

    elif method == "power":
        # Power method (often used for sharp bettors)
        k = math.log(2) / math.log(total)
        prob_a_fair = prob_a**k
        prob_b_fair = prob_b**k

        # Normalize to ensure they sum to 1
        total_fair = prob_a_fair + prob_b_fair
        prob_a_fair /= total_fair
        prob_b_fair /= total_fair

    else:
        raise ValueError(f"Unknown devig method: {method}")

    return prob_a_fair, prob_b_fair


def implied_probability(
    moneyline: int, juice: int = -110, remove_vig: bool = True
) -> float:
    """
    Calculate implied probability from moneyline with optional vig removal.

    Args:
        moneyline: American moneyline odds
        juice: Standard juice/vig (e.g., -110)
        remove_vig: Whether to remove vig

    Returns:
        Implied probability
    """
    prob = moneyline_to_probability(moneyline)

    if remove_vig:
        # Estimate the other side's probability using standard juice
        other_prob = moneyline_to_probability(juice)
        prob, _ = devig_probabilities(prob, other_prob)

    return prob


def edge_calculation(
    model_prob: float, market_prob: float, method: str = "additive"
) -> float:
    """
    Calculate betting edge between model and market probabilities.

    Args:
        model_prob: Model's probability estimate
        market_prob: Market's implied probability
        method: Edge calculation method ('additive', 'multiplicative', 'kelly')

    Returns:
        Edge (positive = favorable bet)
    """
    if method == "additive":
        return model_prob - market_prob

    if method == "multiplicative":
        return (model_prob / market_prob) - 1

    if method == "kelly":
        # Kelly criterion edge calculation
        if market_prob >= 1.0:
            return 0.0
        odds = (1 / market_prob) - 1
        return (model_prob * odds - (1 - model_prob)) / odds

    raise ValueError(f"Unknown edge calculation method: {method}")


def kelly_bet_size(
    edge: float,
    odds: float,
    bankroll: float,
    fraction: float = 1.0,
    max_bet_pct: float = 0.05,
) -> float:
    """
    Calculate Kelly criterion bet size.

    Args:
        edge: Betting edge (probability advantage)
        odds: Decimal odds (e.g., 1.91 for -110)
        bankroll: Total bankroll
        fraction: Fractional Kelly (e.g., 0.25 for quarter Kelly)
        max_bet_pct: Maximum bet as percentage of bankroll

    Returns:
        Recommended bet size
    """
    if edge <= 0 or odds <= 1:
        return 0.0

    # Kelly formula: f = (bp - q) / b
    # where b = odds - 1, p = win probability, q = 1 - p
    p = edge + (1 / odds)  # Win probability
    q = 1 - p  # Lose probability
    b = odds - 1  # Net odds

    kelly_fraction = (b * p - q) / b

    # Apply fractional Kelly and maximum bet constraints
    bet_fraction = min(kelly_fraction * fraction, max_bet_pct)
    bet_fraction = max(bet_fraction, 0)  # No negative bets

    return bet_fraction * bankroll


def american_to_decimal(american_odds: int) -> float:
    """Convert American odds to decimal odds."""
    if american_odds > 0:
        return (american_odds / 100) + 1
    return (100 / abs(american_odds)) + 1


def decimal_to_american(decimal_odds: float) -> int:
    """Convert decimal odds to American odds."""
    if decimal_odds >= 2.0:
        return int((decimal_odds - 1) * 100)
    return int(-100 / (decimal_odds - 1))


def normalize_odds(
    odds: int | float, odds_format: str = "american"
) -> tuple[float, float, int]:
    """Normalize odds to all formats consistently.

    Args:
        odds: Odds in the specified format
        odds_format: "american", "decimal", or "probability"

    Returns:
        Tuple of (probability, decimal_odds, american_odds)
    """
    if odds_format == "american":
        odds = int(odds)
        probability = moneyline_to_probability(odds)
        decimal_odds = american_to_decimal(odds)
        american_odds = odds
    elif odds_format == "decimal":
        decimal_odds = float(odds)
        probability = 1 / decimal_odds
        american_odds = decimal_to_american(decimal_odds)
    elif odds_format == "probability":
        probability = float(odds)
        american_odds = probability_to_moneyline(probability)
        decimal_odds = american_to_decimal(american_odds)
    else:
        raise ValueError(f"Unknown odds format: {odds_format}")

    return probability, decimal_odds, american_odds


def calculate_roi(
    wins: int, losses: int, avg_odds: float = 1.91, stake_per_bet: float = 100
) -> dict[str, float]:
    """
    Calculate betting ROI and related metrics.

    Args:
        wins: Number of winning bets
        losses: Number of losing bets
        avg_odds: Average decimal odds
        stake_per_bet: Average stake per bet

    Returns:
        Dictionary with ROI metrics
    """
    total_bets = wins + losses
    if total_bets == 0:
        return {"roi": 0.0, "profit": 0.0, "hit_rate": 0.0}

    total_staked = total_bets * stake_per_bet
    total_returned = wins * stake_per_bet * avg_odds
    profit = total_returned - total_staked
    roi = profit / total_staked
    hit_rate = wins / total_bets

    return {
        "roi": roi,
        "profit": profit,
        "hit_rate": hit_rate,
        "total_bets": total_bets,
        "total_staked": total_staked,
        "breakeven_rate": 1 / avg_odds,
    }


def calculate_roi_with_pushes(
    wins: int,
    losses: int,
    pushes: int = 0,
    avg_odds: float = 1.91,
    stake_per_bet: float = 100,
) -> dict[str, float]:
    """Calculate betting ROI including pushes.

    Args:
        wins: Number of winning bets
        losses: Number of losing bets
        pushes: Number of push bets (stake returned)
        avg_odds: Average decimal odds
        stake_per_bet: Average stake per bet

    Returns:
        Dictionary with ROI metrics including push handling
    """
    total_bets = wins + losses + pushes
    if total_bets == 0:
        return {"roi": 0.0, "profit": 0.0, "hit_rate": 0.0, "push_rate": 0.0}

    total_staked = total_bets * stake_per_bet
    total_returned = (wins * stake_per_bet * avg_odds) + (
        pushes * stake_per_bet
    )  # Pushes return stake
    profit = total_returned - total_staked
    roi = profit / total_staked
    hit_rate = wins / total_bets if total_bets > 0 else 0.0
    push_rate = pushes / total_bets if total_bets > 0 else 0.0

    return {
        "roi": roi,
        "profit": profit,
        "hit_rate": hit_rate,
        "push_rate": push_rate,
        "total_bets": total_bets,
        "total_staked": total_staked,
        "breakeven_rate": 1 / avg_odds,
    }


def sharpe_ratio(returns: list[float], risk_free_rate: float = 0.0) -> float:
    """
    Calculate Sharpe ratio for betting returns.

    Args:
        returns: List of bet returns (profit/loss per bet)
        risk_free_rate: Risk-free rate of return

    Returns:
        Sharpe ratio
    """
    if not returns or len(returns) < 2:
        return 0.0

    returns_array = np.array(returns)
    excess_returns = returns_array - risk_free_rate

    if np.std(excess_returns) == 0:
        return 0.0

    return np.mean(excess_returns) / np.std(excess_returns)


def max_drawdown(cumulative_returns: list[float]) -> float:
    """
    Calculate maximum drawdown from cumulative returns.

    Args:
        cumulative_returns: List of cumulative returns

    Returns:
        Maximum drawdown (positive value)
    """
    if not cumulative_returns:
        return 0.0

    returns_array = np.array(cumulative_returns)
    peak = np.maximum.accumulate(returns_array)
    drawdown = (peak - returns_array) / peak

    return np.max(drawdown) if len(drawdown) > 0 else 0.0


def probability_calibration_stats(
    predicted_probs: list[float], actual_outcomes: list[bool], n_bins: int = 10
) -> dict[str, float]:
    """
    Calculate probability calibration statistics.

    Args:
        predicted_probs: List of predicted probabilities
        actual_outcomes: List of actual outcomes (True/False)
        n_bins: Number of bins for calibration curve

    Returns:
        Dictionary with calibration metrics
    """
    if len(predicted_probs) != len(actual_outcomes):
        raise ValueError("Predicted probabilities and outcomes must have same length")

    probs = np.array(predicted_probs)
    outcomes = np.array(actual_outcomes, dtype=float)

    # Brier score
    brier_score = np.mean((probs - outcomes) ** 2)

    # Expected Calibration Error (ECE)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers, strict=False):
        in_bin = (probs > bin_lower) & (probs <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = outcomes[in_bin].mean()
            avg_confidence_in_bin = probs[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return {
        "brier_score": brier_score,
        "expected_calibration_error": ece,
        "mean_predicted_prob": np.mean(probs),
        "mean_actual_rate": np.mean(outcomes),
    }
