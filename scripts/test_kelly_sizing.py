#!/usr/bin/env python3
"""
Test script for Kelly Criterion and advanced bankroll management.

This script validates the Kelly sizing implementation with comprehensive
test scenarios covering all risk management features and edge cases.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.kelly_criterion import (
    KellyCalculator, KellyMode, KellyResult, BankrollState,
    calculate_simultaneous_kelly, compare_kelly_modes
)
from utils.bankroll_manager import (
    BankrollManager, RiskLevel, AlertType, BankrollAlert, BettingSession
)
from utils.unit_sizing import (
    UnitSizer, ConfidenceMethod, UnitScale, ConfidenceMetrics, UnitRecommendation
)
from utils import get_logger

logger = get_logger(__name__)


def test_kelly_calculator_initialization():
    """Test KellyCalculator initialization and basic properties."""
    logger.info("Testing KellyCalculator initialization")

    calculator = KellyCalculator(
        starting_bankroll=10000.0,
        max_bet_pct=0.05,
        max_drawdown_pct=0.20,
        base_unit_size=100.0,
        default_kelly_fraction=0.25,
        confidence_threshold=0.02
    )

    assert calculator.starting_bankroll == 10000.0
    assert calculator.max_bet_pct == 0.05
    assert calculator.bankroll_state.current_balance == 10000.0
    assert calculator.bankroll_state.starting_balance == 10000.0
    assert calculator.bankroll_state.peak_balance == 10000.0

    logger.info("KellyCalculator initialization test passed")


def test_kelly_fraction_calculation():
    """Test Kelly fraction calculations for different modes."""
    logger.info("Testing Kelly fraction calculations")

    calculator = KellyCalculator(starting_bankroll=10000.0)

    # Test with +150 odds (decimal 2.5) and 60% win probability
    # Kelly = (2.5 * 0.6 - 0.4) / 1.5 = (1.5 - 0.4) / 1.5 = 0.733

    full_kelly = calculator.calculate_kelly_fraction(0.6, 150, KellyMode.FULL)
    fractional_kelly = calculator.calculate_kelly_fraction(0.6, 150, KellyMode.FRACTIONAL)
    confidence_kelly = calculator.calculate_kelly_fraction(0.6, 150, KellyMode.CONFIDENCE_ADJUSTED)
    conservative_kelly = calculator.calculate_kelly_fraction(0.6, 150, KellyMode.CONSERVATIVE)

    assert full_kelly > fractional_kelly
    assert fractional_kelly > conservative_kelly
    assert confidence_kelly > conservative_kelly

    # Test with negative odds
    neg_kelly = calculator.calculate_kelly_fraction(0.6, -110, KellyMode.FRACTIONAL)
    assert neg_kelly > 0

    # Test with no edge (50% probability)
    no_edge_kelly = calculator.calculate_kelly_fraction(0.5, -110, KellyMode.FRACTIONAL)
    assert abs(no_edge_kelly) < 0.01  # Should be close to zero

    logger.info("Kelly fraction calculation test passed")


def test_confidence_adjustments():
    """Test confidence adjustment calculations."""
    logger.info("Testing confidence adjustments")

    calculator = KellyCalculator()

    # Test high confidence scenario
    high_conf = calculator.calculate_confidence_adjustment(0.65, 0.50, sample_size=1000)
    assert high_conf > 0.7

    # Test low confidence scenario
    low_conf = calculator.calculate_confidence_adjustment(0.52, 0.50, sample_size=50)
    assert low_conf < 0.5

    # Test with no sample size
    no_sample_conf = calculator.calculate_confidence_adjustment(0.60, 0.50)
    assert 0.3 < no_sample_conf < 0.9

    logger.info("Confidence adjustment test passed")


def test_bankroll_updates():
    """Test bankroll state updates after bets."""
    logger.info("Testing bankroll updates")

    calculator = KellyCalculator(starting_bankroll=10000.0)

    # Record a winning bet
    calculator.update_bankroll(bet_amount=100.0, outcome=True, odds=150)
    assert calculator.bankroll_state.current_balance > 10000.0
    assert calculator.bankroll_state.winning_bets == 1
    assert calculator.bankroll_state.total_bets == 1

    # Record a losing bet
    calculator.update_bankroll(bet_amount=100.0, outcome=False, odds=-110)
    assert calculator.bankroll_state.losing_bets == 1
    assert calculator.bankroll_state.total_bets == 2

    # Check drawdown calculation
    if calculator.bankroll_state.current_balance < calculator.bankroll_state.peak_balance:
        assert calculator.bankroll_state.current_drawdown > 0

    logger.info("Bankroll update test passed")


def test_optimal_bet_sizing():
    """Test optimal bet size calculation with all adjustments."""
    logger.info("Testing optimal bet sizing")

    calculator = KellyCalculator(starting_bankroll=10000.0)

    # Test with good edge and confidence
    result = calculator.calculate_optimal_bet_size(
        model_prob=0.60,
        market_odds=150,
        mode=KellyMode.FRACTIONAL
    )

    assert isinstance(result, KellyResult)
    assert result.recommended_bet >= 0
    assert result.kelly_fraction >= 0
    assert result.confidence_adjustment > 0
    assert len(result.reasoning) > 0

    # Test with insufficient edge
    no_bet_result = calculator.calculate_optimal_bet_size(
        model_prob=0.51,
        market_odds=-110,
        mode=KellyMode.FRACTIONAL
    )

    assert no_bet_result.recommended_bet == 0.0

    logger.info("Optimal bet sizing test passed")


def test_bankroll_manager_initialization():
    """Test BankrollManager initialization."""
    logger.info("Testing BankrollManager initialization")

    manager = BankrollManager(
        starting_bankroll=10000.0,
        risk_level=RiskLevel.MODERATE,
        max_drawdown_pct=0.20,
        stop_loss_pct=0.15
    )

    assert manager.starting_bankroll == 10000.0
    assert manager.risk_level == RiskLevel.MODERATE
    assert manager.kelly_calculator.bankroll_state.current_balance == 10000.0
    assert not manager.is_stopped

    logger.info("BankrollManager initialization test passed")


def test_session_management():
    """Test betting session management."""
    logger.info("Testing session management")

    manager = BankrollManager(starting_bankroll=10000.0)

    # Start a session
    session_id = manager.start_session("test_session")
    assert session_id == "test_session"
    assert manager.current_session is not None
    assert manager.current_session.session_id == session_id

    # Record some bets
    manager.record_bet_result(100.0, True, 150)  # Winning bet
    manager.record_bet_result(100.0, False, -110)  # Losing bet

    assert manager.current_session.total_bets == 2
    assert manager.current_session.winning_bets == 1

    # End session
    completed_session = manager.end_session()
    assert completed_session is not None
    assert completed_session.total_bets == 2
    assert manager.current_session is None

    logger.info("Session management test passed")


def test_risk_limits_and_alerts():
    """Test risk limit checking and alert generation."""
    logger.info("Testing risk limits and alerts")

    manager = BankrollManager(
        starting_bankroll=1000.0,
        max_drawdown_pct=0.20,
        stop_loss_pct=0.15
    )

    manager.start_session("risk_test")

    # Simulate significant losses to trigger alerts
    for _ in range(10):
        manager.record_bet_result(100.0, False, -110)

    # Check if alerts were generated
    assert len(manager.alerts) > 0

    # Check if betting might be stopped
    current_dd = manager.kelly_calculator.bankroll_state.drawdown_pct
    if current_dd >= manager.stop_loss_pct:
        assert manager.is_stopped

    logger.info("Risk limits and alerts test passed")


def test_unit_sizer_initialization():
    """Test UnitSizer initialization."""
    logger.info("Testing UnitSizer initialization")

    sizer = UnitSizer(
        base_unit_dollar=100.0,
        max_units=10.0,
        min_units=0.5,
        confidence_scaling=2.0
    )

    assert sizer.base_unit_dollar == 100.0
    assert sizer.max_units == 10.0
    assert sizer.min_units == 0.5

    logger.info("UnitSizer initialization test passed")


def test_confidence_calculations():
    """Test various confidence calculation methods."""
    logger.info("Testing confidence calculations")

    sizer = UnitSizer()

    # Test edge confidence
    edge_conf = sizer.calculate_edge_confidence(0.65, 0.50, sample_size=500)
    assert 0 <= edge_conf <= 1

    # Test model confidence
    model_conf = sizer.calculate_model_confidence(
        model_prob=0.65,
        prediction_interval=(0.55, 0.75),
        model_accuracy=0.58
    )
    assert 0 <= model_conf <= 1

    # Test market confidence
    market_conf = sizer.calculate_market_confidence(
        market_prob=0.50,
        line_movement=0.03,
        volume_indicator=0.4,
        time_to_game=12.0
    )
    assert 0 <= market_conf <= 1

    # Test combined confidence
    combined_conf, factors = sizer.calculate_combined_confidence(
        edge_conf, model_conf, market_conf, 0.7
    )
    assert 0 <= combined_conf <= 1
    assert isinstance(factors, dict)

    logger.info("Confidence calculations test passed")


def test_unit_scaling_methods():
    """Test different unit scaling methods."""
    logger.info("Testing unit scaling methods")

    sizer = UnitSizer()
    edge = 0.05  # 5% edge

    linear_units = sizer.calculate_base_units_from_edge(edge, UnitScale.LINEAR)
    log_units = sizer.calculate_base_units_from_edge(edge, UnitScale.LOGARITHMIC)
    exp_units = sizer.calculate_base_units_from_edge(edge, UnitScale.EXPONENTIAL)
    threshold_units = sizer.calculate_base_units_from_edge(edge, UnitScale.THRESHOLD)

    assert all(units >= 0 for units in [linear_units, log_units, exp_units, threshold_units])
    assert all(units <= sizer.max_units for units in [linear_units, log_units, exp_units, threshold_units])

    logger.info("Unit scaling methods test passed")


def test_unit_recommendation():
    """Test comprehensive unit recommendation."""
    logger.info("Testing unit recommendation")

    sizer = UnitSizer()

    # Test with good edge and confidence
    recommendation = sizer.calculate_unit_recommendation(
        model_prob=0.60,
        market_odds=150,
        bankroll=10000.0,
        sample_size=500,
        model_accuracy=0.58,
        line_movement=0.02
    )

    assert isinstance(recommendation, UnitRecommendation)
    assert recommendation.recommended_units >= 0
    assert recommendation.dollar_amount >= 0
    assert isinstance(recommendation.confidence_metrics, ConfidenceMetrics)
    assert len(recommendation.reasoning) > 0

    # Test with insufficient edge
    no_bet_rec = sizer.calculate_unit_recommendation(
        model_prob=0.505,
        market_odds=-110,
        bankroll=10000.0
    )

    assert no_bet_rec.recommended_units == 0.0

    logger.info("Unit recommendation test passed")


def test_simultaneous_kelly():
    """Test simultaneous Kelly sizing for multiple bets."""
    logger.info("Testing simultaneous Kelly sizing")

    opportunities = [
        {'model_prob': 0.60, 'odds': 150, 'edge': 0.10},
        {'model_prob': 0.55, 'odds': -110, 'edge': 0.05},
        {'model_prob': 0.58, 'odds': 120, 'edge': 0.08}
    ]

    bet_sizes = calculate_simultaneous_kelly(
        opportunities=opportunities,
        total_bankroll=10000.0,
        max_total_allocation=0.25
    )

    assert len(bet_sizes) == len(opportunities)
    assert all(size >= 0 for size in bet_sizes)
    assert sum(bet_sizes) <= 2500  # 25% of bankroll

    logger.info("Simultaneous Kelly sizing test passed")


def test_kelly_mode_comparison():
    """Test comparison of different Kelly modes."""
    logger.info("Testing Kelly mode comparison")

    results = compare_kelly_modes(
        model_prob=0.60,
        market_odds=150,
        bankroll=10000.0
    )

    assert len(results) == len(KellyMode)
    assert all(isinstance(result, KellyResult) for result in results.values())

    # Full Kelly should recommend largest bet
    assert results[KellyMode.FULL].recommended_bet >= results[KellyMode.FRACTIONAL].recommended_bet
    assert results[KellyMode.FRACTIONAL].recommended_bet >= results[KellyMode.CONSERVATIVE].recommended_bet

    logger.info("Kelly mode comparison test passed")


def test_performance_simulation():
    """Test Kelly performance simulation."""
    logger.info("Testing performance simulation")

    calculator = KellyCalculator(starting_bankroll=10000.0)

    # Create simulation scenarios
    scenarios = []
    np.random.seed(42)  # For reproducible results

    for _ in range(100):
        model_prob = 0.55 + np.random.normal(0, 0.05)  # Around 55% with noise
        model_prob = max(0.1, min(0.9, model_prob))  # Clamp to reasonable range

        odds = np.random.choice([-110, -105, 100, 110, 120, 150])

        # Simulate outcome based on model probability
        outcome = np.random.random() < model_prob

        scenarios.append({
            'model_prob': model_prob,
            'odds': odds,
            'outcome': outcome
        })

    performance_metrics = calculator.simulate_kelly_performance(
        scenarios=scenarios,
        mode=KellyMode.FRACTIONAL
    )

    assert 'total_return' in performance_metrics
    assert 'volatility' in performance_metrics
    assert 'max_drawdown' in performance_metrics
    assert 'final_balance' in performance_metrics

    logger.info("Performance simulation test passed")


def test_bankroll_manager_integration():
    """Test integration between all components."""
    logger.info("Testing bankroll manager integration")

    manager = BankrollManager(starting_bankroll=10000.0, risk_level=RiskLevel.MODERATE)
    manager.start_session("integration_test")

    # Test bet sizing with all features
    kelly_result = manager.calculate_bet_size(
        model_prob=0.60,
        market_odds=150,
        mode=KellyMode.FRACTIONAL
    )

    assert isinstance(kelly_result, KellyResult)

    # Record the bet result
    if kelly_result.recommended_bet > 0:
        manager.record_bet_result(
            bet_amount=kelly_result.recommended_bet,
            outcome=True,  # Assume win for test
            odds=150
        )

    # Get performance summary
    summary = manager.get_performance_summary()
    assert isinstance(summary, dict)
    assert 'current_balance' in summary
    assert 'win_rate' in summary

    manager.end_session()

    logger.info("Bankroll manager integration test passed")


def test_historical_tracking():
    """Test historical performance tracking."""
    logger.info("Testing historical performance tracking")

    sizer = UnitSizer()

    # Record some bet outcomes
    sizer.record_bet_outcome(
        model_prob=0.60,
        market_prob=0.50,
        edge=0.10,
        confidence=0.8,
        units_bet=2.0,
        outcome=True,
        profit_loss=150.0
    )

    sizer.record_bet_outcome(
        model_prob=0.55,
        market_prob=0.52,
        edge=0.03,
        confidence=0.6,
        units_bet=1.0,
        outcome=False,
        profit_loss=-100.0
    )

    # Get performance analysis
    analysis = sizer.get_performance_analysis()

    assert 'total_bets' in analysis
    assert 'overall_win_rate' in analysis
    assert 'performance_by_confidence' in analysis
    assert 'performance_by_edge' in analysis

    logger.info("Historical tracking test passed")


def test_edge_cases():
    """Test various edge cases and error conditions."""
    logger.info("Testing edge cases")

    calculator = KellyCalculator(starting_bankroll=10000.0)

    # Test with extreme probabilities
    extreme_result = calculator.calculate_optimal_bet_size(0.95, 150)
    assert extreme_result.recommended_bet >= 0

    # Test with zero probability
    zero_result = calculator.calculate_optimal_bet_size(0.0, 150)
    assert zero_result.recommended_bet == 0.0

    # Test with negative odds edge case
    neg_result = calculator.calculate_optimal_bet_size(0.60, -10000)
    assert neg_result.recommended_bet >= 0

    # Test bankroll manager with very small bankroll
    small_manager = BankrollManager(starting_bankroll=100.0)
    small_result = small_manager.calculate_bet_size(0.60, 150)
    assert small_result.recommended_bet <= 100.0

    logger.info("Edge cases test passed")


def run_all_tests():
    """Run all Kelly sizing and bankroll management tests."""
    logger.info("Starting Kelly sizing and bankroll management test suite")

    test_functions = [
        test_kelly_calculator_initialization,
        test_kelly_fraction_calculation,
        test_confidence_adjustments,
        test_bankroll_updates,
        test_optimal_bet_sizing,
        test_bankroll_manager_initialization,
        test_session_management,
        test_risk_limits_and_alerts,
        test_unit_sizer_initialization,
        test_confidence_calculations,
        test_unit_scaling_methods,
        test_unit_recommendation,
        test_simultaneous_kelly,
        test_kelly_mode_comparison,
        test_performance_simulation,
        test_bankroll_manager_integration,
        test_historical_tracking,
        test_edge_cases
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
        logger.info("All Kelly sizing and bankroll management tests passed!")
        return True
    else:
        logger.error("Some tests failed. Check the logs above for details.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)