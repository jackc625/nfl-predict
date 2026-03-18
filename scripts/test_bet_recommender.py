"""
Test suite for bet recommendation engine.

This script validates the BetRecommender functionality including ranking,
unit sizing, portfolio optimization, and structured output generation.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import json
import tempfile
from datetime import datetime, timedelta

from utils.bankroll_manager import BankrollManager, RiskLevel
from utils.bet_recommender import (
    BetRecommendation,
    BetRecommender,
    RecommendationAction,
    RecommendationPortfolio,
    RecommendationTier,
)
from utils.bet_selector import BetSelector, FilterCriteria
from utils.betting_utils import BettingResult, BetType
from utils.kelly_criterion import KellyCalculator, KellyMode
from utils.logging_config import get_logger
from utils.unit_sizing import UnitSizer

logger = get_logger(__name__)


def create_test_betting_results() -> list[BettingResult]:
    """Create comprehensive test betting results."""
    results = []

    # Premium tier: High edge, high confidence
    results.append(
        BettingResult(
            bet_type=BetType.MONEYLINE,
            model_prob=0.68,  # High confidence
            market_prob=0.58,  # Good edge
            edge=0.10,
            expected_value=0.093,
            market_odds=-150,
        )
    )

    # Strong tier: Good edge and confidence
    results.append(
        BettingResult(
            bet_type=BetType.SPREAD,
            model_prob=0.62,  # Good confidence
            market_prob=0.54,  # Decent edge
            edge=0.08,
            expected_value=0.074,
            market_odds=-110,
            line_value=-3.5,
        )
    )

    # Value tier: Medium edge and confidence
    results.append(
        BettingResult(
            bet_type=BetType.TOTAL,
            model_prob=0.56,  # Medium confidence
            market_prob=0.52,  # Small edge
            edge=0.04,
            expected_value=0.038,
            market_odds=+105,
            line_value=44.5,
        )
    )

    # Speculative tier: Lower confidence but edge
    results.append(
        BettingResult(
            bet_type=BetType.MONEYLINE,
            model_prob=0.53,  # Low confidence
            market_prob=0.50,  # Small edge
            edge=0.03,
            expected_value=0.030,
            market_odds=+110,
        )
    )

    # Another strong bet for portfolio testing
    results.append(
        BettingResult(
            bet_type=BetType.SPREAD,
            model_prob=0.65,  # High confidence
            market_prob=0.56,  # Good edge
            edge=0.09,
            expected_value=0.085,
            market_odds=-105,
            line_value=+7.0,
        )
    )

    return results


def create_test_game_context() -> dict[str, dict]:
    """Create test game context."""
    future_time = datetime.now() + timedelta(hours=48)  # 48 hours from now
    soon_time = datetime.now() + timedelta(minutes=30)  # 30 minutes (too soon)

    return {
        "game_1": {
            "kickoff_time": future_time,
            "home_team": "KC",
            "away_team": "BUF",
            "week": 10,
            "season": 2024,
            "line_movement": 0.015,
        },
        "game_2": {
            "kickoff_time": future_time,
            "home_team": "SF",
            "away_team": "DAL",
            "week": 10,
            "season": 2024,
            "line_movement": -0.005,
        },
        "game_3": {
            "kickoff_time": soon_time,  # Too soon for betting
            "home_team": "GB",
            "away_team": "MIN",
            "week": 10,
            "season": 2024,
            "line_movement": 0.02,
        },
    }


def create_test_risk_factors() -> dict[str, dict]:
    """Create test risk factor data."""
    return {
        "game_1": {
            "injuries": ["QB1 questionable", "WR1 out"],
            "weather_impact": "Heavy wind expected",
        },
        "game_2": {"injuries": [], "weather_impact": "Dome game"},
    }


def create_test_components():
    """Create test recommendation engine components."""
    # Create bankroll manager with $10,000 bankroll
    bankroll_manager = BankrollManager(
        starting_bankroll=10000.0, risk_level=RiskLevel.MODERATE, max_drawdown_pct=0.15
    )

    # Create bet selector with moderate criteria
    criteria = FilterCriteria(
        min_edge_threshold=0.02, max_bets_per_week=6, max_bets_per_game=2
    )
    bet_selector = BetSelector(criteria)

    # Create Kelly calculator
    kelly_calculator = KellyCalculator(mode=KellyMode.FRACTIONAL, fraction=0.25)

    # Create unit sizer
    unit_sizer = UnitSizer()

    return bankroll_manager, bet_selector, kelly_calculator, unit_sizer


def test_basic_recommendation_generation():
    """Test basic recommendation generation functionality."""
    logger.info("Running test_basic_recommendation_generation")

    # Create components
    bankroll_manager, bet_selector, kelly_calculator, unit_sizer = (
        create_test_components()
    )

    # Create recommender
    recommender = BetRecommender(
        bankroll_manager=bankroll_manager,
        bet_selector=bet_selector,
        kelly_calculator=kelly_calculator,
        unit_sizer=unit_sizer,
    )

    # Create test data
    betting_results = create_test_betting_results()
    game_context = create_test_game_context()

    # Add game IDs to betting results
    for i, result in enumerate(betting_results):
        game_id = f"game_{(i % 3) + 1}"
        if hasattr(result, "__dict__"):
            result.__dict__["game_id"] = game_id
        else:
            result.game_id = game_id

    # Generate recommendations
    portfolio = recommender.generate_recommendations(
        betting_results=betting_results, game_context=game_context
    )

    # Verify basic structure
    assert isinstance(portfolio, RecommendationPortfolio), (
        "Should return RecommendationPortfolio"
    )
    assert len(portfolio.recommendations) > 0, "Should generate some recommendations"
    assert portfolio.total_units_recommended >= 0, "Should have non-negative units"
    assert portfolio.total_expected_value >= 0, (
        "Should have non-negative expected value"
    )

    # Verify recommendations have required fields
    for rec in portfolio.recommendations:
        assert isinstance(rec, BetRecommendation), "Should be BetRecommendation objects"
        assert rec.recommendation_id is not None, "Should have recommendation ID"
        assert rec.description is not None, "Should have description"
        assert isinstance(rec.recommendation_tier, RecommendationTier), (
            "Should have tier"
        )
        assert isinstance(rec.action, RecommendationAction), "Should have action"

    logger.info(
        f"Generated {len(portfolio.recommendations)} recommendations successfully"
    )
    return True


def test_recommendation_tiers():
    """Test recommendation tier assignment."""
    logger.info("Running test_recommendation_tiers")

    bankroll_manager, bet_selector, kelly_calculator, unit_sizer = (
        create_test_components()
    )
    recommender = BetRecommender(
        bankroll_manager, bet_selector, kelly_calculator, unit_sizer
    )

    betting_results = create_test_betting_results()
    game_context = create_test_game_context()

    # Add game IDs
    for i, result in enumerate(betting_results):
        if hasattr(result, "__dict__"):
            result.__dict__["game_id"] = f"game_{i + 1}"
        else:
            result.game_id = f"game_{i + 1}"

    portfolio = recommender.generate_recommendations(betting_results, game_context)

    # Verify tier distribution
    assert isinstance(portfolio.tier_distribution, dict), (
        "Should have tier distribution"
    )

    # Check that high edge/confidence bets get premium/strong tiers
    premium_count = portfolio.tier_distribution.get(RecommendationTier.PREMIUM, 0)
    strong_count = portfolio.tier_distribution.get(RecommendationTier.STRONG, 0)

    assert premium_count + strong_count > 0, (
        "Should have some premium or strong recommendations"
    )

    # Verify tier assignments make sense
    for rec in portfolio.recommendations:
        if rec.edge >= 0.06 and rec.confidence >= 0.08:
            assert rec.recommendation_tier == RecommendationTier.PREMIUM, (
                f"High edge/confidence should be PREMIUM: edge={rec.edge}, conf={rec.confidence}"
            )

    logger.info(f"Tier distribution: {portfolio.tier_distribution}")
    return True


def test_unit_sizing_integration():
    """Test integration with unit sizing and Kelly criterion."""
    logger.info("Running test_unit_sizing_integration")

    bankroll_manager, bet_selector, kelly_calculator, unit_sizer = (
        create_test_components()
    )
    recommender = BetRecommender(
        bankroll_manager, bet_selector, kelly_calculator, unit_sizer
    )

    betting_results = create_test_betting_results()
    game_context = create_test_game_context()

    # Add game IDs
    for i, result in enumerate(betting_results):
        if hasattr(result, "__dict__"):
            result.__dict__["game_id"] = f"game_{i + 1}"
        else:
            result.game_id = f"game_{i + 1}"

    portfolio = recommender.generate_recommendations(betting_results, game_context)

    # Verify unit recommendations
    for rec in portfolio.recommendations:
        if rec.action == RecommendationAction.BET:
            assert rec.unit_recommendation.units > 0, (
                "Betting recommendations should have positive units"
            )
            assert rec.unit_recommendation.kelly_size > 0, "Should have Kelly size"
            assert rec.unit_recommendation.reasoning is not None, (
                "Should have reasoning"
            )

        # Check that units are reasonable relative to bankroll
        max_reasonable_units = (
            bankroll_manager.current_bankroll * 0.1
        )  # Max 10% of bankroll
        assert rec.unit_recommendation.units <= max_reasonable_units, (
            f"Units too large: {rec.unit_recommendation.units}"
        )

    logger.info(
        f"Unit sizing verified for {len(portfolio.recommendations)} recommendations"
    )
    return True


def test_portfolio_optimization():
    """Test portfolio-level optimization and constraints."""
    logger.info("Running test_portfolio_optimization")

    # Create stricter bankroll constraints
    bankroll_manager = BankrollManager(
        starting_bankroll=1000.0,  # Smaller bankroll
        risk_level=RiskLevel.CONSERVATIVE,
        max_drawdown_pct=0.10,
    )

    criteria = FilterCriteria(max_bets_per_week=3)  # Tight limits
    bet_selector = BetSelector(criteria)
    kelly_calculator = KellyCalculator(mode=KellyMode.FRACTIONAL)
    unit_sizer = UnitSizer()

    recommender = BetRecommender(
        bankroll_manager, bet_selector, kelly_calculator, unit_sizer
    )

    # Create many good betting opportunities (more than limits allow)
    betting_results = []
    for i in range(10):  # 10 good bets
        betting_results.append(
            BettingResult(
                bet_type=BetType.SPREAD,
                model_prob=0.60 + i * 0.005,  # Varying confidence
                market_prob=0.52,
                edge=0.08 - i * 0.003,  # Decreasing edge
                expected_value=0.07 - i * 0.002,
            )
        )

    game_context = {}
    for i, result in enumerate(betting_results):
        game_id = f"game_{i + 1}"
        if hasattr(result, "__dict__"):
            result.__dict__["game_id"] = game_id
        else:
            result.game_id = game_id

        game_context[game_id] = {
            "kickoff_time": datetime.now() + timedelta(hours=24),
            "home_team": f"HOME_{i}",
            "away_team": f"AWAY_{i}",
        }

    portfolio = recommender.generate_recommendations(betting_results, game_context)

    # Should respect position limits
    betting_recs = [
        rec
        for rec in portfolio.recommendations
        if rec.action == RecommendationAction.BET
    ]
    assert len(betting_recs) <= 3, "Should respect max_bets_per_week limit"

    # Should respect bankroll limits
    available_units = bankroll_manager.get_available_units()
    assert portfolio.total_units_recommended <= available_units, (
        "Should respect bankroll limits"
    )

    logger.info(
        f"Portfolio optimization: {len(betting_recs)} bets selected from {len(betting_results)} candidates"
    )
    return True


def test_priority_scoring():
    """Test priority scoring and ranking system."""
    logger.info("Running test_priority_scoring")

    bankroll_manager, bet_selector, kelly_calculator, unit_sizer = (
        create_test_components()
    )
    recommender = BetRecommender(
        bankroll_manager, bet_selector, kelly_calculator, unit_sizer
    )

    betting_results = create_test_betting_results()
    game_context = create_test_game_context()

    # Add game IDs
    for i, result in enumerate(betting_results):
        if hasattr(result, "__dict__"):
            result.__dict__["game_id"] = f"game_{i + 1}"
        else:
            result.game_id = f"game_{i + 1}"

    portfolio = recommender.generate_recommendations(betting_results, game_context)

    # Verify priority scores
    for rec in portfolio.recommendations:
        assert rec.priority_score > 0, "Should have positive priority score"

    # Verify ranking by priority
    sorted_recs = sorted(
        portfolio.recommendations, key=lambda x: x.priority_score, reverse=True
    )

    # Higher priority should generally correlate with better metrics
    if len(sorted_recs) >= 2:
        top_rec = sorted_recs[0]
        second_rec = sorted_recs[1]

        # Top recommendation should have better or equal metrics
        assert top_rec.priority_score >= second_rec.priority_score, (
            "Priority scores should be ordered"
        )

    logger.info(
        f"Priority scoring verified: scores range from {min(r.priority_score for r in portfolio.recommendations):.2f} to {max(r.priority_score for r in portfolio.recommendations):.2f}"
    )
    return True


def test_recommendation_actions():
    """Test recommendation action determination."""
    logger.info("Running test_recommendation_actions")

    bankroll_manager, bet_selector, kelly_calculator, unit_sizer = (
        create_test_components()
    )
    recommender = BetRecommender(
        bankroll_manager, bet_selector, kelly_calculator, unit_sizer
    )

    betting_results = create_test_betting_results()
    game_context = create_test_game_context()

    # Add game IDs including one with timing issue
    for i, result in enumerate(betting_results):
        if hasattr(result, "__dict__"):
            result.__dict__["game_id"] = f"game_{(i % 3) + 1}"
        else:
            result.game_id = f"game_{(i % 3) + 1}"

    portfolio = recommender.generate_recommendations(betting_results, game_context)

    # Verify action types
    action_counts = {}
    for rec in portfolio.recommendations:
        action = rec.action
        action_counts[action] = action_counts.get(action, 0) + 1

    # Should have some betting actions for good opportunities
    assert action_counts.get(RecommendationAction.BET, 0) > 0, (
        "Should have some BET actions"
    )

    # Check for appropriate action assignment
    for rec in portfolio.recommendations:
        if rec.hours_until_kickoff and rec.hours_until_kickoff < 1.0:
            assert rec.action == RecommendationAction.ALERT, (
                "Games starting soon should be ALERT"
            )

    logger.info(f"Action distribution: {action_counts}")
    return True


def test_json_export():
    """Test JSON export functionality."""
    logger.info("Running test_json_export")

    bankroll_manager, bet_selector, kelly_calculator, unit_sizer = (
        create_test_components()
    )
    recommender = BetRecommender(
        bankroll_manager, bet_selector, kelly_calculator, unit_sizer
    )

    betting_results = create_test_betting_results()
    game_context = create_test_game_context()
    risk_factors = create_test_risk_factors()

    # Add game IDs
    for i, result in enumerate(betting_results):
        if hasattr(result, "__dict__"):
            result.__dict__["game_id"] = f"game_{(i % 3) + 1}"
        else:
            result.game_id = f"game_{(i % 3) + 1}"

    portfolio = recommender.generate_recommendations(
        betting_results, game_context, risk_factors=risk_factors
    )

    # Export to temporary file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False
    ) as tmp_file:
        temp_path = tmp_file.name

    try:
        recommender.export_recommendations_json(portfolio, temp_path)

        # Verify file was created and contains valid JSON
        with open(temp_path) as f:
            exported_data = json.load(f)

        # Verify structure
        assert "portfolio_summary" in exported_data, "Should have portfolio summary"
        assert "recommendations" in exported_data, "Should have recommendations list"

        # Verify content
        assert len(exported_data["recommendations"]) == len(
            portfolio.recommendations
        ), "Should export all recommendations"

        # Verify required fields in first recommendation
        if exported_data["recommendations"]:
            first_rec = exported_data["recommendations"][0]
            assert "recommendation_id" in first_rec, "Should have recommendation ID"
            assert "game_info" in first_rec, "Should have game info"
            assert "bet_details" in first_rec, "Should have bet details"
            assert "analysis" in first_rec, "Should have analysis"
            assert "recommendation" in first_rec, "Should have recommendation"

    finally:
        # Clean up temp file
        os.unlink(temp_path)

    logger.info(
        f"JSON export successful: {len(portfolio.recommendations)} recommendations"
    )
    return True


def test_recommendation_summary():
    """Test recommendation summary generation."""
    logger.info("Running test_recommendation_summary")

    bankroll_manager, bet_selector, kelly_calculator, unit_sizer = (
        create_test_components()
    )
    recommender = BetRecommender(
        bankroll_manager, bet_selector, kelly_calculator, unit_sizer
    )

    betting_results = create_test_betting_results()
    game_context = create_test_game_context()

    # Add game IDs
    for i, result in enumerate(betting_results):
        if hasattr(result, "__dict__"):
            result.__dict__["game_id"] = f"game_{i + 1}"
        else:
            result.game_id = f"game_{i + 1}"

    portfolio = recommender.generate_recommendations(betting_results, game_context)

    # Generate summary
    summary = recommender.get_recommendation_summary(portfolio)

    # Verify summary structure
    assert "portfolio_overview" in summary, "Should have portfolio overview"
    assert "quality_metrics" in summary, "Should have quality metrics"
    assert "tier_breakdown" in summary, "Should have tier breakdown"
    assert "top_recommendations" in summary, "Should have top recommendations"

    # Verify content
    overview = summary["portfolio_overview"]
    assert overview["total_recommendations"] == len(portfolio.recommendations), (
        "Should match recommendation count"
    )
    assert overview["total_units"] >= 0, "Should have non-negative units"

    # Verify top recommendations
    top_recs = summary["top_recommendations"]
    assert len(top_recs) <= 5, "Should have at most 5 top recommendations"
    assert len(top_recs) <= len(portfolio.recommendations), (
        "Should not exceed total recommendations"
    )

    logger.info("Recommendation summary generated successfully")
    return True


def test_risk_factor_integration():
    """Test integration of risk factors in recommendations."""
    logger.info("Running test_risk_factor_integration")

    bankroll_manager, bet_selector, kelly_calculator, unit_sizer = (
        create_test_components()
    )
    recommender = BetRecommender(
        bankroll_manager, bet_selector, kelly_calculator, unit_sizer
    )

    betting_results = create_test_betting_results()
    game_context = create_test_game_context()
    risk_factors = create_test_risk_factors()

    # Add game IDs
    for i, result in enumerate(betting_results):
        if hasattr(result, "__dict__"):
            result.__dict__["game_id"] = (
                f"game_{(i % 2) + 1}"  # Only games 1 and 2 have risk factors
            )
        else:
            result.game_id = f"game_{(i % 2) + 1}"

    portfolio = recommender.generate_recommendations(
        betting_results, game_context, risk_factors=risk_factors
    )

    # Verify risk factors are included
    for rec in portfolio.recommendations:
        if rec.game_id in risk_factors:
            expected_injuries = risk_factors[rec.game_id].get("injuries")
            expected_weather = risk_factors[rec.game_id].get("weather_impact")

            assert rec.injury_concerns == expected_injuries, (
                "Should include injury concerns"
            )
            assert rec.weather_impact == expected_weather, (
                "Should include weather impact"
            )

    logger.info("Risk factor integration verified")
    return True


def test_portfolio_metrics():
    """Test portfolio-level metrics calculation."""
    logger.info("Running test_portfolio_metrics")

    bankroll_manager, bet_selector, kelly_calculator, unit_sizer = (
        create_test_components()
    )
    recommender = BetRecommender(
        bankroll_manager, bet_selector, kelly_calculator, unit_sizer
    )

    betting_results = create_test_betting_results()
    game_context = create_test_game_context()

    # Add game IDs
    for i, result in enumerate(betting_results):
        if hasattr(result, "__dict__"):
            result.__dict__["game_id"] = f"game_{i + 1}"
        else:
            result.game_id = f"game_{i + 1}"

    portfolio = recommender.generate_recommendations(betting_results, game_context)

    # Verify portfolio metrics
    betting_recs = [
        rec
        for rec in portfolio.recommendations
        if rec.action == RecommendationAction.BET
    ]

    if betting_recs:
        # Calculate expected metrics
        expected_total_units = sum(
            rec.unit_recommendation.units for rec in betting_recs
        )
        expected_avg_edge = sum(rec.edge for rec in betting_recs) / len(betting_recs)
        expected_avg_confidence = sum(rec.confidence for rec in betting_recs) / len(
            betting_recs
        )

        # Verify calculations
        assert abs(portfolio.total_units_recommended - expected_total_units) < 1e-6, (
            "Total units should match"
        )
        assert abs(portfolio.average_edge - expected_avg_edge) < 1e-6, (
            "Average edge should match"
        )
        assert abs(portfolio.average_confidence - expected_avg_confidence) < 1e-6, (
            "Average confidence should match"
        )

    # Verify risk metrics are reasonable
    assert 0 <= portfolio.diversification_score <= 1, (
        "Diversification score should be between 0 and 1"
    )
    assert portfolio.correlation_risk_score >= 0, (
        "Correlation risk should be non-negative"
    )
    assert portfolio.max_drawdown_risk >= 0, "Max drawdown risk should be non-negative"

    logger.info("Portfolio metrics calculation verified")
    return True


def run_all_tests():
    """Run all bet recommender tests."""
    logger.info("Starting bet recommender test suite")

    tests = [
        test_basic_recommendation_generation,
        test_recommendation_tiers,
        test_unit_sizing_integration,
        test_portfolio_optimization,
        test_priority_scoring,
        test_recommendation_actions,
        test_json_export,
        test_recommendation_summary,
        test_risk_factor_integration,
        test_portfolio_metrics,
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

    logger.info(f"Bet recommender tests completed: {passed} passed, {failed} failed")

    if failed == 0:
        logger.info("All bet recommender tests passed!")
    else:
        logger.error(f"{failed} tests failed")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
