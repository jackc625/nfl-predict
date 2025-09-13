"""Probability and odds conversion utilities."""

import math
from typing import Dict, List, Tuple, Union, Optional
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
    else:
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
    else:
        # Underdog (positive moneyline)
        return int(100 * (1 - probability) / probability)


def devig_probabilities(
    prob_a: float, 
    prob_b: float, 
    method: str = "proportional"
) -> Tuple[float, float]:
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
        prob_a_fair = prob_a ** k
        prob_b_fair = prob_b ** k
        
        # Normalize to ensure they sum to 1
        total_fair = prob_a_fair + prob_b_fair
        prob_a_fair /= total_fair
        prob_b_fair /= total_fair
        
    else:
        raise ValueError(f"Unknown devig method: {method}")
    
    return prob_a_fair, prob_b_fair


def implied_probability(
    moneyline: int, 
    juice: int = -110,
    remove_vig: bool = True
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
    model_prob: float, 
    market_prob: float,
    method: str = "additive"
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
        
    elif method == "multiplicative":
        return (model_prob / market_prob) - 1
        
    elif method == "kelly":
        # Kelly criterion edge calculation
        if market_prob >= 1.0:
            return 0.0
        odds = (1 / market_prob) - 1
        return (model_prob * odds - (1 - model_prob)) / odds
        
    else:
        raise ValueError(f"Unknown edge calculation method: {method}")


def kelly_bet_size(
    edge: float, 
    odds: float, 
    bankroll: float,
    fraction: float = 1.0,
    max_bet_pct: float = 0.05
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


def convert_spread_to_moneyline(spread: float, total: float = 45.0) -> Tuple[int, int]:
    """
    Convert point spread to approximate moneyline odds.
    
    Args:
        spread: Point spread (positive = home favorite)
        total: Game total (for context)
        
    Returns:
        Tuple of (home_ml, away_ml)
    """
    # Empirical relationship between spread and moneyline
    # This is a rough approximation
    if abs(spread) < 0.5:
        return -110, -110
    
    # Use a sigmoid-like function to convert spread to probability
    home_prob = 1 / (1 + math.exp(-spread * 0.25))
    away_prob = 1 - home_prob
    
    try:
        home_ml = probability_to_moneyline(home_prob)
        away_ml = probability_to_moneyline(away_prob)
        return home_ml, away_ml
    except ValueError:
        # Fallback for extreme cases
        return -110, -110


def convert_total_to_over_under_ml(total: float, market_total: float) -> Tuple[int, int]:
    """
    Convert predicted total to over/under moneyline odds.
    
    Args:
        total: Predicted game total
        market_total: Market's total line
        
    Returns:
        Tuple of (over_ml, under_ml)
    """
    if abs(total - market_total) < 0.5:
        return -110, -110
    
    # Simple linear relationship
    diff = total - market_total
    over_prob = 0.5 + (diff * 0.02)  # 2% per point difference
    over_prob = max(0.1, min(0.9, over_prob))  # Clamp to reasonable range
    
    under_prob = 1 - over_prob
    
    try:
        over_ml = probability_to_moneyline(over_prob)
        under_ml = probability_to_moneyline(under_prob)
        return over_ml, under_ml
    except ValueError:
        return -110, -110


def calculate_roi(
    wins: int, 
    losses: int, 
    avg_odds: float = 1.91,
    stake_per_bet: float = 100
) -> Dict[str, float]:
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
        "breakeven_rate": 1 / avg_odds
    }


def sharpe_ratio(returns: List[float], risk_free_rate: float = 0.0) -> float:
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


def max_drawdown(cumulative_returns: List[float]) -> float:
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
    predicted_probs: List[float], 
    actual_outcomes: List[bool],
    n_bins: int = 10
) -> Dict[str, float]:
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
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
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
        "mean_actual_rate": np.mean(outcomes)
    }