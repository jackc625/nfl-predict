#!/usr/bin/env python3
"""
Test script for betting utilities and Expected Value calculations.

This script validates the betting utilities implementation with
comprehensive test scenarios covering all bet types and edge cases.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.betting_utils import (
    BetType,
    BettingResult,
    calculate_moneyline_ev,
    calculate_spread_ev,
    calculate_total_ev,
    calculate_kelly_sizing,
    analyze_game_betting_opportunities,
    portfolio_kelly_sizing,
    summarize_betting_session,
    _normal_cdf
)
from utils.probability_utils import moneyline_to_probability
from utils import get_logger

logger = get_logger(__name__)


def test_betting_result_creation():
    """Test BettingResult object creation and properties."""
    logger.info("Testing BettingResult creation")

    result = BettingResult(
        bet_type=BetType.MONEYLINE,
        model_prob=0.6,
        market_prob=0.55,
        edge=0.05,
        expected_value=10.0
    )

    assert result.bet_type == BetType.MONEYLINE
    assert result.model_prob == 0.6
    assert result.market_prob == 0.55
    assert result.edge == 0.05
    assert result.expected_value == 10.0
    assert result.kelly_size is None
    assert result.recommended_units is None

    logger.info("BettingResult creation test passed")


def test_moneyline_ev_positive_odds():
    """Test EV calculation for positive moneyline odds."""
    logger.info("Testing moneyline EV with positive odds")

    result = calculate_moneyline_ev(
        model_prob=0.6,
        market_odds=150,  # +150 underdog
        stake=100.0,
        devig=False
    )

    assert result.bet_type == BetType.MONEYLINE
    assert result.model_prob == 0.6

    # Market prob for +150 should be 100/(150+100) = 0.4
    expected_market_prob = 100 / (150 + 100)
    assert abs(result.market_prob - expected_market_prob) < 0.01

    # Edge should be 0.6 - 0.4 = 0.2
    assert abs(result.edge - 0.2) < 0.01

    # EV = (0.6 × 150) - (0.4 × 100) = 90 - 40 = 50
    assert abs(result.expected_value - 50.0) < 0.01

    logger.info("Positive moneyline EV test passed")


def test_moneyline_ev_negative_odds():
    """Test EV calculation for negative moneyline odds."""
    logger.info("Testing moneyline EV with negative odds")

    result = calculate_moneyline_ev(
        model_prob=0.7,
        market_odds=-150,  # -150 favorite
        stake=100.0,
        devig=False
    )

    # Market prob for -150 should be 150/(150+100) = 0.6
    expected_market_prob = 150 / (150 + 100)
    assert abs(result.market_prob - expected_market_prob) < 0.01

    # Edge should be 0.7 - 0.6 = 0.1
    assert abs(result.edge - 0.1) < 0.01

    # Win amount for -150 is 100 * (100/150) = 66.67
    # EV = (0.7 × 66.67) - (0.3 × 100) = 46.67 - 30 = 16.67
    assert abs(result.expected_value - 16.67) < 0.1

    logger.info("Negative moneyline EV test passed")


def test_moneyline_ev_with_devig():
    """Test EV calculation with vig removal."""
    logger.info("Testing moneyline EV with devig")

    result = calculate_moneyline_ev(
        model_prob=0.55,
        market_odds=-110,
        stake=100.0,
        devig=True,
        opposite_odds=-110
    )

    # With devig, both sides at -110 should become 0.5 each
    assert abs(result.market_prob - 0.5) < 0.01

    # Edge should be 0.55 - 0.5 = 0.05
    assert abs(result.edge - 0.05) < 0.01

    logger.info("Devig moneyline EV test passed")


def test_moneyline_no_edge():
    """Test moneyline bet with no edge."""
    logger.info("Testing moneyline bet with no edge")

    market_prob = moneyline_to_probability(-110)

    result = calculate_moneyline_ev(
        model_prob=market_prob,
        market_odds=-110,
        stake=100.0,
        devig=False
    )

    # Should have minimal edge and essentially break-even EV before vig
    assert abs(result.edge) < 0.01
    # EV should be close to zero (might be slightly positive or negative due to rounding)
    assert abs(result.expected_value) < 1.0

    logger.info("No edge moneyline test passed")


def test_spread_ev_home_favorite():
    """Test spread EV for home favorite."""
    logger.info("Testing spread EV for home favorite")

    result = calculate_spread_ev(
        model_margin=7.0,  # Model predicts home wins by 7
        model_margin_std=14.0,
        market_spread=-3.5,  # Market has home favored by 3.5
        spread_juice=-110,
        side="home"
    )

    assert result.bet_type == BetType.SPREAD

    # Model predicts home covers -3.5 easily (7 > 3.5)
    assert result.model_prob > 0.5
    assert result.edge > 0
    assert result.expected_value > 0

    logger.info("Home favorite spread EV test passed")


def test_spread_ev_away_underdog():
    """Test spread EV for away underdog."""
    logger.info("Testing spread EV for away underdog")

    result = calculate_spread_ev(
        model_margin=-2.0,  # Model predicts home loses by 2
        model_margin_std=14.0,
        market_spread=-7.0,  # Market has home favored by 7
        spread_juice=-110,
        side="away"
    )

    # Away gets +7 points, model has home losing by 2
    # So away covers by 9 points (2 + 7)
    assert result.model_prob > 0.5
    assert result.edge > 0

    logger.info("Away underdog spread EV test passed")


def test_spread_ev_no_edge():
    """Test spread bet with no edge."""
    logger.info("Testing spread bet with no edge")

    result = calculate_spread_ev(
        model_margin=3.5,  # Exactly the spread
        model_margin_std=14.0,
        market_spread=-3.5,
        spread_juice=-110,
        side="home"
    )

    # Should be close to 50/50
    assert abs(result.model_prob - 0.5) < 0.1
    assert abs(result.edge) < 0.1

    logger.info("No edge spread test passed")


def test_total_ev_over():
    """Test total EV for over bet."""
    logger.info("Testing total EV for over bet")

    result = calculate_total_ev(
        model_total=52.0,  # Model predicts 52 points
        model_total_std=10.5,
        market_total=45.0,  # Market total is 45
        total_juice=-110,
        side="over"
    )

    assert result.bet_type == BetType.TOTAL

    # Model predicts significantly more than market
    assert result.model_prob > 0.5
    assert result.edge > 0
    assert result.expected_value > 0

    logger.info("Over total EV test passed")


def test_total_ev_under():
    """Test total EV for under bet."""
    logger.info("Testing total EV for under bet")

    result = calculate_total_ev(
        model_total=40.0,  # Model predicts 40 points
        model_total_std=10.5,
        market_total=47.0,  # Market total is 47
        total_juice=-110,
        side="under"
    )

    # Model predicts significantly less than market
    assert result.model_prob > 0.5
    assert result.edge > 0
    assert result.expected_value > 0

    logger.info("Under total EV test passed")


def test_total_ev_no_edge():
    """Test total bet with no edge."""
    logger.info("Testing total bet with no edge")

    result = calculate_total_ev(
        model_total=45.0,  # Exactly the market total
        model_total_std=10.5,
        market_total=45.0,
        total_juice=-110,
        side="over"
    )

    # Should be close to 50/50
    assert abs(result.model_prob - 0.5) < 0.1
    assert abs(result.edge) < 0.1

    logger.info("No edge total test passed")


def test_kelly_sizing_positive_edge():
    """Test Kelly sizing for positive edge bet."""
    logger.info("Testing Kelly sizing for positive edge")

    result = BettingResult(
        bet_type=BetType.MONEYLINE,
        model_prob=0.6,
        market_prob=0.5,
        edge=0.1,
        expected_value=10.0
    )

    bet_size = calculate_kelly_sizing(
        betting_result=result,
        bankroll=10000.0,
        odds=-110,
        fraction=0.25,
        max_bet_pct=0.05
    )

    assert bet_size > 0
    assert bet_size <= 500  # Max 5% of $10,000 bankroll
    assert result.kelly_size == bet_size
    assert result.recommended_units is not None

    logger.info("Kelly sizing positive edge test passed")


def test_kelly_sizing_no_edge():
    """Test Kelly sizing for no edge bet."""
    logger.info("Testing Kelly sizing for no edge")

    result = BettingResult(
        bet_type=BetType.SPREAD,
        model_prob=0.5,
        market_prob=0.5,
        edge=0.0,
        expected_value=0.0
    )

    bet_size = calculate_kelly_sizing(
        betting_result=result,
        bankroll=10000.0,
        odds=-110
    )

    assert bet_size == 0.0

    logger.info("Kelly sizing no edge test passed")


def test_kelly_sizing_negative_edge():
    """Test Kelly sizing for negative edge bet."""
    logger.info("Testing Kelly sizing for negative edge")

    result = BettingResult(
        bet_type=BetType.TOTAL,
        model_prob=0.45,
        market_prob=0.5,
        edge=-0.05,
        expected_value=-5.0
    )

    bet_size = calculate_kelly_sizing(
        betting_result=result,
        bankroll=10000.0,
        odds=-110
    )

    assert bet_size == 0.0

    logger.info("Kelly sizing negative edge test passed")


def create_sample_game_data():
    """Create sample game data for testing."""
    return {
        "ml_home": -150,
        "ml_away": 130,
        "spread": -3.5,
        "spread_juice_home": -110,
        "spread_juice_away": -110,
        "total": 45.0,
        "total_over_juice": -110,
        "total_under_juice": -110
    }


def create_sample_predictions():
    """Create sample model predictions for testing."""
    return {
        "wp_home": 0.65,
        "predicted_margin": 5.0,
        "margin_std": 14.0,
        "predicted_total": 48.0,
        "total_std": 10.5
    }


def test_analyze_game_opportunities():
    """Test comprehensive game analysis."""
    logger.info("Testing game opportunity analysis")

    game_data = create_sample_game_data()
    predictions = create_sample_predictions()

    opportunities = analyze_game_betting_opportunities(
        game_data=game_data,
        model_predictions=predictions,
        bankroll=10000.0,
        min_edge_threshold=0.01
    )

    # Should find some opportunities given our setup
    assert len(opportunities) > 0

    # All opportunities should meet minimum edge threshold
    for opp in opportunities:
        assert opp.edge >= 0.01
        assert opp.kelly_size is not None

    # Should be sorted by expected value (descending)
    for i in range(len(opportunities) - 1):
        assert opportunities[i].expected_value >= opportunities[i + 1].expected_value

    logger.info("Game opportunity analysis test passed")


def test_analyze_game_no_opportunities():
    """Test game analysis with no viable opportunities."""
    logger.info("Testing game analysis with no opportunities")

    game_data = create_sample_game_data()

    # Predictions that match market exactly
    no_edge_predictions = {
        "wp_home": moneyline_to_probability(game_data["ml_home"]),
        "predicted_margin": 3.5,  # Exactly the spread
        "margin_std": 14.0,
        "predicted_total": 45.0,  # Exactly the total
        "total_std": 10.5
    }

    opportunities = analyze_game_betting_opportunities(
        game_data=game_data,
        model_predictions=no_edge_predictions,
        min_edge_threshold=0.05  # Higher threshold
    )

    # Should find few or no opportunities
    assert len(opportunities) <= 2  # Maybe some small edges from rounding

    logger.info("No opportunities analysis test passed")


def test_portfolio_single_bet():
    """Test portfolio sizing with single bet."""
    logger.info("Testing single bet portfolio")

    results = [BettingResult(
        bet_type=BetType.MONEYLINE,
        model_prob=0.6,
        market_prob=0.5,
        edge=0.1,
        expected_value=10.0
    )]

    sizes = portfolio_kelly_sizing(
        betting_results=results,
        bankroll=10000.0,
        fraction=0.25
    )

    assert len(sizes) == 1
    assert sizes[0] > 0

    logger.info("Single bet portfolio test passed")


def test_portfolio_multi_bet():
    """Test portfolio sizing with multiple bets."""
    logger.info("Testing multi-bet portfolio")

    results = [
        BettingResult(BetType.MONEYLINE, 0.6, 0.5, 0.1, 10.0),
        BettingResult(BetType.SPREAD, 0.55, 0.5, 0.05, 5.0),
        BettingResult(BetType.TOTAL, 0.58, 0.5, 0.08, 8.0)
    ]

    sizes = portfolio_kelly_sizing(
        betting_results=results,
        bankroll=10000.0,
        fraction=0.25
    )

    assert len(sizes) == 3
    assert all(size >= 0 for size in sizes)

    # Total allocation should be reasonable
    total_allocation = sum(sizes)
    assert total_allocation <= 2500  # Max 25% of bankroll

    logger.info("Multi-bet portfolio test passed")


def test_portfolio_empty():
    """Test portfolio sizing with no bets."""
    logger.info("Testing empty portfolio")

    sizes = portfolio_kelly_sizing(
        betting_results=[],
        bankroll=10000.0
    )

    assert sizes == []

    logger.info("Empty portfolio test passed")


def test_summarize_empty_session():
    """Test summary of empty betting session."""
    logger.info("Testing empty session summary")

    summary = summarize_betting_session([])

    assert summary["total_bets"] == 0
    assert summary["total_ev"] == 0.0
    assert summary["avg_edge"] == 0.0

    logger.info("Empty session summary test passed")


def test_summarize_betting_session():
    """Test comprehensive betting session summary."""
    logger.info("Testing betting session summary")

    results = [
        BettingResult(BetType.MONEYLINE, 0.6, 0.5, 0.1, 10.0),
        BettingResult(BetType.SPREAD, 0.55, 0.5, 0.05, 5.0),
        BettingResult(BetType.TOTAL, 0.58, 0.5, 0.08, 8.0)
    ]

    # Add Kelly sizes
    for result in results:
        result.kelly_size = 100.0

    summary = summarize_betting_session(results)

    assert summary["total_bets"] == 3
    assert summary["total_ev"] == 23.0  # 10 + 5 + 8
    assert summary["total_kelly_size"] == 300.0
    assert abs(summary["avg_edge"] - 0.077) < 0.01  # (0.1 + 0.05 + 0.08) / 3

    # Check bet type breakdown
    breakdown = summary["bet_type_breakdown"]
    assert breakdown["moneyline"]["count"] == 1
    assert breakdown["spread"]["count"] == 1
    assert breakdown["total"]["count"] == 1

    # Check edge distribution
    edge_dist = summary["edge_distribution"]
    assert edge_dist["min"] == 0.05
    assert edge_dist["max"] == 0.1
    assert abs(edge_dist["mean"] - 0.077) < 0.01

    logger.info("Betting session summary test passed")


def test_summarize_with_outcomes():
    """Test summary with actual bet outcomes."""
    logger.info("Testing session summary with outcomes")

    results = [
        BettingResult(BetType.MONEYLINE, 0.6, 0.5, 0.1, 10.0),
        BettingResult(BetType.SPREAD, 0.55, 0.5, 0.05, 5.0)
    ]

    outcomes = [True, False]  # First bet wins, second loses

    summary = summarize_betting_session(results, outcomes)

    assert "actual_returns" in summary
    actual_returns = summary["actual_returns"]
    assert actual_returns["hit_rate"] == 0.5  # 1 win out of 2 bets
    assert actual_returns["total_return"] == -90.0  # 10 - 100 = -90

    logger.info("Session summary with outcomes test passed")


def test_normal_cdf():
    """Test normal CDF approximation."""
    logger.info("Testing normal CDF function")

    # Test known values
    assert abs(_normal_cdf(0) - 0.5) < 0.001
    assert abs(_normal_cdf(-1.96) - 0.025) < 0.01
    assert abs(_normal_cdf(1.96) - 0.975) < 0.01
    assert abs(_normal_cdf(1.0) - 0.841) < 0.01

    logger.info("Normal CDF test passed")


def test_normal_cdf_extreme_values():
    """Test normal CDF with extreme values."""
    logger.info("Testing normal CDF extreme values")

    assert _normal_cdf(-10) < 0.001
    assert _normal_cdf(10) > 0.999

    logger.info("Normal CDF extreme values test passed")


def run_all_tests():
    """Run all betting utilities tests."""
    logger.info("Starting betting utilities test suite")

    test_functions = [
        test_betting_result_creation,
        test_moneyline_ev_positive_odds,
        test_moneyline_ev_negative_odds,
        test_moneyline_ev_with_devig,
        test_moneyline_no_edge,
        test_spread_ev_home_favorite,
        test_spread_ev_away_underdog,
        test_spread_ev_no_edge,
        test_total_ev_over,
        test_total_ev_under,
        test_total_ev_no_edge,
        test_kelly_sizing_positive_edge,
        test_kelly_sizing_no_edge,
        test_kelly_sizing_negative_edge,
        test_analyze_game_opportunities,
        test_analyze_game_no_opportunities,
        test_portfolio_single_bet,
        test_portfolio_multi_bet,
        test_portfolio_empty,
        test_summarize_empty_session,
        test_summarize_betting_session,
        test_summarize_with_outcomes,
        test_normal_cdf,
        test_normal_cdf_extreme_values
    ]

    passed = 0
    failed = 0

    for test_func in test_functions:
        try:
            test_func()
            passed += 1
            logger.info(f"PASSED: {test_func.__name__}")
        except Exception as e:
            failed += 1
            logger.error(f"FAILED: {test_func.__name__} - {str(e)}")

    logger.info(f"\nTest Results: {passed} passed, {failed} failed")

    if failed == 0:
        logger.info("All betting utilities tests passed!")
        return True
    else:
        logger.error("Some tests failed. Check the logs above for details.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)