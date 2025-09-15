"""
Test suite for bet selection and filtering system.

This script validates the BetSelector functionality including edge thresholds,
confidence requirements, position limits, and market disagreement detection.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from datetime import datetime, timedelta
from typing import List, Dict

from utils.betting_utils import BetType, BettingResult
from utils.bet_selector import (
    BetSelector, FilterCriteria, BetCandidate, BetSelectionResult,
    FilterReason, create_default_criteria, create_aggressive_criteria
)
from utils.logging_config import get_logger

logger = get_logger(__name__)


def create_test_betting_results() -> List[BettingResult]:
    """Create test betting results with various edges and probabilities."""
    results = []

    # High edge, high confidence moneyline
    results.append(BettingResult(
        bet_type=BetType.MONEYLINE,
        model_prob=0.65,  # 65% confidence
        market_prob=0.55,  # 55% implied
        edge=0.10,  # 10% edge
        expected_value=0.091
    ))

    # Medium edge spread bet
    results.append(BettingResult(
        bet_type=BetType.SPREAD,
        model_prob=0.58,  # 58% confidence
        market_prob=0.52,  # 52% implied
        edge=0.06,  # 6% edge
        expected_value=0.058
    ))

    # Low edge total bet (should be filtered)
    results.append(BettingResult(
        bet_type=BetType.TOTAL,
        model_prob=0.51,  # Very low confidence
        market_prob=0.50,
        edge=0.01,  # Only 1% edge
        expected_value=0.010
    ))

    # High edge but extreme probability (should be filtered)
    results.append(BettingResult(
        bet_type=BetType.MONEYLINE,
        model_prob=0.97,  # Too extreme
        market_prob=0.85,
        edge=0.12,
        expected_value=0.140
    ))

    # Good spread bet
    results.append(BettingResult(
        bet_type=BetType.SPREAD,
        model_prob=0.62,
        market_prob=0.54,
        edge=0.08,
        expected_value=0.074
    ))

    # Medium total bet
    results.append(BettingResult(
        bet_type=BetType.TOTAL,
        model_prob=0.59,
        market_prob=0.52,
        edge=0.07,
        expected_value=0.067
    ))

    # Low confidence moneyline (should be filtered)
    results.append(BettingResult(
        bet_type=BetType.MONEYLINE,
        model_prob=0.52,  # Too close to 50%
        market_prob=0.50,
        edge=0.02,
        expected_value=0.020
    ))

    # Another good spread bet for same game (test per-game limits)
    results.append(BettingResult(
        bet_type=BetType.SPREAD,
        model_prob=0.61,
        market_prob=0.53,
        edge=0.08,
        expected_value=0.075
    ))

    return results


def create_test_game_context() -> Dict[str, Dict]:
    """Create test game context with timing and team info."""
    future_time = datetime.now() + timedelta(hours=24)  # 24 hours from now
    soon_time = datetime.now() + timedelta(hours=1)     # 1 hour from now (too soon)

    return {
        "game_1": {
            "kickoff_time": future_time,
            "home_team": "KC",
            "away_team": "BUF",
            "week": 10,
            "season": 2024
        },
        "game_2": {
            "kickoff_time": future_time,
            "home_team": "SF",
            "away_team": "DAL",
            "week": 10,
            "season": 2024
        },
        "game_3": {
            "kickoff_time": soon_time,  # Too soon
            "home_team": "GB",
            "away_team": "MIN",
            "week": 10,
            "season": 2024
        }
    }


def create_test_closing_lines() -> Dict[str, Dict]:
    """Create test closing line data for disagreement detection."""
    return {
        "game_1": {
            "closing_odds": -120,
            "closing_probability": 0.545,
            "line_movement": 0.02
        },
        "game_2": {
            "closing_odds": +105,
            "closing_probability": 0.488,
            "line_movement": -0.01
        }
    }


def test_basic_bet_selection():
    """Test basic bet selection functionality."""
    logger.info("Running test_basic_bet_selection")

    # Create selector with default criteria
    selector = BetSelector()

    # Create test data
    betting_results = create_test_betting_results()

    # Run selection
    result = selector.select_bets(betting_results)

    # Verify basic results
    assert isinstance(result, BetSelectionResult), "Should return BetSelectionResult"
    assert result.total_candidates == len(betting_results), "Should track total candidates"
    assert len(result.selected_bets) > 0, "Should select some bets"
    assert len(result.selected_bets) <= 8, "Should respect max_bets_per_week limit"

    # Verify selected bets are ranked
    for i, bet in enumerate(result.selected_bets):
        assert bet.rank == i + 1, f"Bet {i} should have rank {i+1}"
        assert bet.selected == True, f"Selected bet {i} should be marked as selected"

    logger.info(f"Basic selection: {len(result.selected_bets)} bets selected from {result.total_candidates} candidates")
    return True


def test_edge_filtering():
    """Test edge threshold filtering."""
    logger.info("Running test_edge_filtering")

    # Create strict criteria with high edge requirements
    criteria = FilterCriteria(
        min_edge_threshold=0.05,  # 5% minimum
        min_edge_moneyline=0.05,
        min_edge_spread=0.05,
        min_edge_total=0.05
    )
    selector = BetSelector(criteria)

    betting_results = create_test_betting_results()
    result = selector.select_bets(betting_results)

    # Should filter out low-edge bets
    low_edge_filtered = result.filters_applied.get(FilterReason.EDGE_TOO_LOW.value, 0)
    assert low_edge_filtered > 0, "Should filter some bets for low edge"

    # All selected bets should have sufficient edge
    for bet in result.selected_bets:
        assert bet.edge >= 0.05, f"Selected bet has edge {bet.edge}, below threshold"

    logger.info(f"Edge filtering: {low_edge_filtered} bets filtered for low edge")
    return True


def test_confidence_filtering():
    """Test confidence threshold filtering."""
    logger.info("Running test_confidence_filtering")

    # Create criteria with high confidence requirements
    criteria = FilterCriteria(
        min_confidence_threshold=0.08,  # 58%+ probability required
        min_edge_threshold=0.01  # Low edge to focus on confidence
    )
    selector = BetSelector(criteria)

    betting_results = create_test_betting_results()
    result = selector.select_bets(betting_results)

    # Should filter out low-confidence bets
    low_conf_filtered = result.filters_applied.get(FilterReason.CONFIDENCE_TOO_LOW.value, 0)
    assert low_conf_filtered > 0, "Should filter some bets for low confidence"

    # All selected bets should have sufficient confidence
    for bet in result.selected_bets:
        assert bet.confidence >= 0.08, f"Selected bet has confidence {bet.confidence}, below threshold"

    logger.info(f"Confidence filtering: {low_conf_filtered} bets filtered for low confidence")
    return True


def test_position_limits():
    """Test position limit enforcement."""
    logger.info("Running test_position_limits")

    # Create criteria with tight position limits
    criteria = FilterCriteria(
        max_bets_per_week=3,  # Very restrictive
        max_bets_per_game=1,
        max_bets_per_bet_type={
            BetType.MONEYLINE: 1,
            BetType.SPREAD: 2,
            BetType.TOTAL: 1
        },
        min_edge_threshold=0.01  # Low threshold to test limits
    )
    selector = BetSelector(criteria)

    # Create many good betting opportunities
    betting_results = []
    for i in range(10):
        betting_results.append(BettingResult(
            bet_type=BetType.SPREAD,
            model_prob=0.6,
            market_prob=0.52,
            edge=0.08 - i*0.001,  # Decreasing edge
            expected_value=0.07 - i*0.001
        ))

    # Add game context to test per-game limits
    game_context = {}
    for i in range(5):  # 5 games, 2 bets each
        for j in range(2):
            idx = i * 2 + j
            if idx < len(betting_results):
                game_id = f"game_{i}"
                if hasattr(betting_results[idx], '__dict__'):
                    betting_results[idx].__dict__['game_id'] = game_id
                else:
                    betting_results[idx].game_id = game_id

                game_context[game_id] = {
                    "kickoff_time": datetime.now() + timedelta(hours=24),
                    "home_team": f"HOME_{i}",
                    "away_team": f"AWAY_{i}"
                }

    result = selector.select_bets(betting_results, game_context)

    # Should respect weekly limit
    assert len(result.selected_bets) <= 3, "Should respect max_bets_per_week"

    # Check position limit filtering
    position_filtered = result.filters_applied.get(FilterReason.POSITION_LIMIT_REACHED.value, 0)
    expected_filtered = len(betting_results) - len(result.selected_bets)

    logger.info(f"Position limits: {len(result.selected_bets)} selected, {position_filtered} filtered for position limits")
    return True


def test_timing_filters():
    """Test timing-based filtering."""
    logger.info("Running test_timing_filters")

    criteria = FilterCriteria(
        min_hours_before_kickoff=2.0,  # 2 hours minimum
        min_edge_threshold=0.01  # Low threshold to focus on timing
    )
    selector = BetSelector(criteria)

    betting_results = create_test_betting_results()
    game_context = create_test_game_context()

    # Assign games to betting results
    for i, result in enumerate(betting_results):
        game_key = f"game_{(i % 3) + 1}"
        if hasattr(result, '__dict__'):
            result.__dict__['game_id'] = game_key
        else:
            result.game_id = game_key

    selection_result = selector.select_bets(betting_results, game_context)

    # Should filter games that are too soon
    timing_filtered = selection_result.filters_applied.get(FilterReason.GAME_TOO_SOON.value, 0)
    assert timing_filtered > 0, "Should filter some games for timing"

    # Verify no selected bets are for games too soon
    for bet in selection_result.selected_bets:
        if bet.kickoff_time:
            hours_until = (bet.kickoff_time - datetime.now()).total_seconds() / 3600
            assert hours_until >= 2.0, f"Selected bet has kickoff in {hours_until} hours, too soon"

    logger.info(f"Timing filters: {timing_filtered} bets filtered for timing")
    return True


def test_quality_filters():
    """Test probability quality filtering."""
    logger.info("Running test_quality_filters")

    criteria = FilterCriteria(
        max_probability_extreme=0.90,  # Don't allow >90% probability
        min_probability_extreme=0.10,  # Don't allow <10% probability
        min_edge_threshold=0.01
    )
    selector = BetSelector(criteria)

    betting_results = create_test_betting_results()
    result = selector.select_bets(betting_results)

    # Should filter extreme probabilities
    extreme_filtered = result.filters_applied.get(FilterReason.PROBABILITY_EXTREME.value, 0)
    assert extreme_filtered > 0, "Should filter some bets for extreme probabilities"

    # Verify no selected bets have extreme probabilities
    for bet in result.selected_bets:
        assert 0.10 <= bet.model_prob <= 0.90, f"Selected bet has extreme probability {bet.model_prob}"

    logger.info(f"Quality filters: {extreme_filtered} bets filtered for extreme probabilities")
    return True


def test_market_disagreement():
    """Test market disagreement filtering."""
    logger.info("Running test_market_disagreement")

    criteria = FilterCriteria(
        min_market_disagreement=0.05,  # 5% disagreement required
        min_edge_threshold=0.01
    )
    selector = BetSelector(criteria)

    betting_results = create_test_betting_results()[:4]  # Limit to games with closing data
    game_context = create_test_game_context()
    closing_lines = create_test_closing_lines()

    # Assign games to betting results
    for i, result in enumerate(betting_results):
        game_key = f"game_{(i % 2) + 1}"  # Only games 1 and 2 have closing data
        if hasattr(result, '__dict__'):
            result.__dict__['game_id'] = game_key
        else:
            result.game_id = game_key

    selection_result = selector.select_bets(betting_results, game_context, closing_lines)

    # May filter some bets for low market disagreement
    disagreement_filtered = selection_result.filters_applied.get(FilterReason.MARKET_DISAGREEMENT_LOW.value, 0)

    logger.info(f"Market disagreement: {disagreement_filtered} bets filtered for low disagreement")
    return True


def test_bet_type_distribution():
    """Test bet type distribution in selections."""
    logger.info("Running test_bet_type_distribution")

    selector = BetSelector()
    betting_results = create_test_betting_results()
    result = selector.select_bets(betting_results)

    # Verify bet type distribution is tracked
    assert isinstance(result.bet_type_distribution, dict), "Should track bet type distribution"

    total_selected = sum(result.bet_type_distribution.values())
    assert total_selected == len(result.selected_bets), "Bet type counts should match total selected"

    logger.info(f"Bet type distribution: {result.bet_type_distribution}")
    return True


def test_selection_summary():
    """Test selection result summary generation."""
    logger.info("Running test_selection_summary")

    selector = BetSelector()
    betting_results = create_test_betting_results()
    result = selector.select_bets(betting_results)

    # Generate summary
    summary = selector.get_selection_summary(result)

    # Verify summary structure
    assert "selection_overview" in summary, "Summary should have selection overview"
    assert "portfolio_metrics" in summary, "Summary should have portfolio metrics"
    assert "bet_distribution" in summary, "Summary should have bet distribution"
    assert "filter_breakdown" in summary, "Summary should have filter breakdown"
    assert "top_selections" in summary, "Summary should have top selections"

    # Verify metrics are reasonable
    overview = summary["selection_overview"]
    assert overview["total_candidates"] == len(betting_results), "Should track total candidates"
    assert overview["selected_bets"] == len(result.selected_bets), "Should track selected bets"

    logger.info("Selection summary generated successfully")
    return True


def test_criteria_presets():
    """Test default and aggressive criteria presets."""
    logger.info("Running test_criteria_presets")

    # Test default criteria
    default_criteria = create_default_criteria()
    assert default_criteria.min_edge_threshold == 0.02, "Default should have 2% edge threshold"
    assert default_criteria.max_bets_per_week == 6, "Default should be conservative"

    # Test aggressive criteria
    aggressive_criteria = create_aggressive_criteria()
    assert aggressive_criteria.min_edge_threshold == 0.015, "Aggressive should have lower edge threshold"
    assert aggressive_criteria.max_bets_per_week == 12, "Aggressive should allow more bets"

    # Test both work
    betting_results = create_test_betting_results()

    default_selector = BetSelector(default_criteria)
    default_result = default_selector.select_bets(betting_results)

    aggressive_selector = BetSelector(aggressive_criteria)
    aggressive_result = aggressive_selector.select_bets(betting_results)

    # Aggressive should generally select more bets (if enough quality candidates)
    logger.info(f"Default selected: {len(default_result.selected_bets)}, Aggressive selected: {len(aggressive_result.selected_bets)}")

    logger.info("Criteria presets work correctly")
    return True


def test_portfolio_metrics():
    """Test portfolio-level metrics calculation."""
    logger.info("Running test_portfolio_metrics")

    selector = BetSelector()
    betting_results = create_test_betting_results()
    result = selector.select_bets(betting_results)

    if result.selected_bets:
        # Verify portfolio metrics
        expected_total_ev = sum(bet.expected_value for bet in result.selected_bets)
        assert abs(result.total_expected_value - expected_total_ev) < 1e-6, "Total EV should be sum of individual EVs"

        expected_avg_edge = sum(bet.edge for bet in result.selected_bets) / len(result.selected_bets)
        assert abs(result.average_edge - expected_avg_edge) < 1e-6, "Average edge should be correct"

        expected_avg_confidence = sum(bet.confidence for bet in result.selected_bets) / len(result.selected_bets)
        assert abs(result.average_confidence - expected_avg_confidence) < 1e-6, "Average confidence should be correct"

    logger.info("Portfolio metrics calculated correctly")
    return True


def run_all_tests():
    """Run all bet selector tests."""
    logger.info("Starting bet selector test suite")

    tests = [
        test_basic_bet_selection,
        test_edge_filtering,
        test_confidence_filtering,
        test_position_limits,
        test_timing_filters,
        test_quality_filters,
        test_market_disagreement,
        test_bet_type_distribution,
        test_selection_summary,
        test_criteria_presets,
        test_portfolio_metrics
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                logger.error(f"Test {test_func.__name__} failed")
        except Exception as e:
            failed += 1
            logger.error(f"Test {test_func.__name__} failed with exception: {e}")
            import traceback
            logger.error(traceback.format_exc())

    logger.info(f"Bet selector tests completed: {passed} passed, {failed} failed")

    if failed == 0:
        logger.info("All bet selector tests passed!")
    else:
        logger.error(f"{failed} tests failed")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)