"""
Test suite for CLV tracking functionality.

This script validates the CLV tracking system including bet recording,
closing line updates, CLV calculations, and performance analysis.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import tempfile
from datetime import datetime, timedelta
from typing import List, Dict
import numpy as np

from backtest.clv_tracking import CLVTracker, CLVSummary, BetRecord
from utils.logging_config import get_logger

logger = get_logger(__name__)


def test_bet_recording():
    """Test basic bet recording functionality."""
    logger.info("Running test_bet_recording")

    tracker = CLVTracker()

    # Record some bets
    bet_id_1 = tracker.record_bet(
        game_id="game_1",
        bet_type="moneyline",
        bet_side="home",
        our_odds=-120,
        bet_amount=100.0
    )

    bet_id_2 = tracker.record_bet(
        game_id="game_2",
        bet_type="spread",
        bet_side="away",
        our_odds=-110,
        bet_amount=50.0
    )

    # Verify bets were recorded
    assert len(tracker.bet_records) == 2, "Should have 2 bet records"

    bet_1 = tracker.bet_records[0]
    assert bet_1.game_id == "game_1", "First bet should be game_1"
    assert bet_1.bet_type == "moneyline", "Should be moneyline bet"
    assert bet_1.our_odds == -120, "Should have correct odds"
    assert bet_1.bet_amount == 100.0, "Should have correct amount"
    assert bet_1.our_probability > 0, "Should calculate implied probability"

    logger.info(f"Bet recording successful: {len(tracker.bet_records)} bets recorded")
    return True


def test_closing_line_updates():
    """Test closing line updates and CLV calculation."""
    logger.info("Running test_closing_line_updates")

    tracker = CLVTracker()

    # Record a bet
    tracker.record_bet(
        game_id="game_1",
        bet_type="moneyline",
        bet_side="home",
        our_odds=-120,  # We bet at -120
        bet_amount=100.0
    )

    # Update closing lines
    closing_data = {
        "game_1": {
            "moneyline_home": -110,  # Closing line was -110 (better for us)
            "moneyline_away": -110,
            "closing_timestamp": datetime.now()
        }
    }

    tracker.update_closing_lines(closing_data)

    # Verify CLV calculation
    bet_record = tracker.bet_records[0]
    assert bet_record.closing_odds == -110, "Should have closing odds"
    assert bet_record.absolute_clv is not None, "Should have calculated absolute CLV"
    assert bet_record.percentage_clv is not None, "Should have calculated percentage CLV"
    assert bet_record.ev_clv is not None, "Should have calculated EV CLV"

    # We bet at -120, closing was -110, so we got a worse line (negative CLV)
    assert bet_record.absolute_clv < 0, "Should have negative CLV (worse line than closing)"

    logger.info(f"CLV calculation: Absolute={bet_record.absolute_clv:.4f}, "
                f"Percentage={bet_record.percentage_clv:.4f}")
    return True


def test_bet_outcomes():
    """Test recording bet outcomes."""
    logger.info("Running test_bet_outcomes")

    tracker = CLVTracker()

    # Record a bet
    tracker.record_bet(
        game_id="game_1",
        bet_type="spread",
        bet_side="home",
        our_odds=-110,
        bet_amount=100.0
    )

    # Record winning outcome
    tracker.record_bet_outcome("game_1", "spread", "home", won=True)

    # Verify outcome was recorded
    bet_record = tracker.bet_records[0]
    assert bet_record.won == True, "Should record win"
    assert bet_record.payout is not None, "Should calculate payout"
    assert bet_record.payout > 0, "Winning bet should have positive payout"

    # Test losing outcome
    tracker.record_bet(
        game_id="game_2",
        bet_type="moneyline",
        bet_side="away",
        our_odds=+150,
        bet_amount=50.0
    )

    tracker.record_bet_outcome("game_2", "moneyline", "away", won=False)

    bet_record_2 = tracker.bet_records[1]
    assert bet_record_2.won == False, "Should record loss"
    assert bet_record_2.payout == -50.0, "Losing bet should have negative payout"

    logger.info("Bet outcome recording successful")
    return True


def test_clv_summary_calculation():
    """Test comprehensive CLV summary calculation."""
    logger.info("Running test_clv_summary_calculation")

    tracker = CLVTracker()

    # Create a series of bets with known CLV outcomes
    bet_scenarios = [
        {"our_odds": -120, "closing_odds": -110, "won": True},   # Negative CLV, won
        {"our_odds": -110, "closing_odds": -120, "won": False},  # Positive CLV, lost
        {"our_odds": -105, "closing_odds": -110, "won": True},   # Slight negative CLV, won
        {"our_odds": +110, "closing_odds": +120, "won": False},  # Positive CLV, lost
        {"our_odds": -110, "closing_odds": -110, "won": True},   # No CLV, won
    ]

    for i, scenario in enumerate(bet_scenarios):
        game_id = f"game_{i+1}"

        # Record bet
        tracker.record_bet(
            game_id=game_id,
            bet_type="moneyline",
            bet_side="home",
            our_odds=scenario["our_odds"],
            bet_amount=100.0
        )

        # Record outcome
        tracker.record_bet_outcome(game_id, "moneyline", "home", scenario["won"])

    # Update closing lines
    closing_data = {}
    for i, scenario in enumerate(bet_scenarios):
        game_id = f"game_{i+1}"
        closing_data[game_id] = {
            "moneyline_home": scenario["closing_odds"],
            "moneyline_away": -110,
            "closing_timestamp": datetime.now()
        }

    tracker.update_closing_lines(closing_data)

    # Calculate summary
    summary = tracker.calculate_clv_summary()

    # Verify summary
    assert isinstance(summary, CLVSummary), "Should return CLVSummary"
    assert summary.total_bets == 5, "Should have 5 total bets"
    assert summary.bets_with_clv_data == 5, "Should have CLV data for all bets"
    assert summary.average_absolute_clv is not None, "Should calculate average CLV"
    assert 0 <= summary.positive_clv_rate <= 1, "Positive CLV rate should be between 0 and 1"

    # Check bucket performance
    assert isinstance(summary.clv_bucket_performance, dict), "Should have bucket performance"

    logger.info(f"CLV Summary: Avg CLV={summary.average_absolute_clv:.4f}, "
                f"Positive Rate={summary.positive_clv_rate:.1%}")
    return True


def test_clv_bucket_analysis():
    """Test CLV bucket performance analysis."""
    logger.info("Running test_clv_bucket_analysis")

    tracker = CLVTracker()

    # Create bets in different CLV buckets
    scenarios = [
        # Strong Negative CLV bucket
        {"our_odds": -150, "closing_odds": -110, "won": False},
        {"our_odds": -140, "closing_odds": -110, "won": False},

        # Positive CLV bucket
        {"our_odds": -110, "closing_odds": -130, "won": True},
        {"our_odds": -105, "closing_odds": -120, "won": True},

        # Neutral CLV bucket
        {"our_odds": -110, "closing_odds": -109, "won": True},
        {"our_odds": -110, "closing_odds": -111, "won": False},
    ]

    for i, scenario in enumerate(scenarios):
        game_id = f"game_{i+1}"

        tracker.record_bet(
            game_id=game_id,
            bet_type="spread",
            bet_side="home",
            our_odds=scenario["our_odds"],
            bet_amount=100.0
        )

        tracker.record_bet_outcome(game_id, "spread", "home", scenario["won"])

    # Update closing lines
    closing_data = {}
    for i, scenario in enumerate(scenarios):
        game_id = f"game_{i+1}"
        closing_data[game_id] = {
            "spread_home": scenario["closing_odds"],
            "spread_away": -110,
            "closing_timestamp": datetime.now()
        }

    tracker.update_closing_lines(closing_data)

    # Calculate summary with bucket analysis
    summary = tracker.calculate_clv_summary()

    # Verify bucket analysis
    assert len(summary.clv_bucket_performance) > 0, "Should have bucket performance data"

    # Check that buckets have expected structure
    for bucket_name, metrics in summary.clv_bucket_performance.items():
        assert "count" in metrics, f"Bucket {bucket_name} should have count"
        assert "win_rate" in metrics, f"Bucket {bucket_name} should have win_rate"
        assert "roi" in metrics, f"Bucket {bucket_name} should have roi"
        assert "average_clv" in metrics, f"Bucket {bucket_name} should have average_clv"

    logger.info(f"Bucket analysis: {len(summary.clv_bucket_performance)} buckets with data")
    return True


def test_clv_insights():
    """Test CLV insights generation."""
    logger.info("Running test_clv_insights")

    tracker = CLVTracker()

    # Create a scenario with good CLV
    good_clv_scenarios = [
        {"our_odds": -110, "closing_odds": -120, "won": True},
        {"our_odds": -105, "closing_odds": -115, "won": True},
        {"our_odds": +110, "closing_odds": +100, "won": False},
    ]

    for i, scenario in enumerate(good_clv_scenarios):
        game_id = f"game_{i+1}"

        tracker.record_bet(
            game_id=game_id,
            bet_type="moneyline",
            bet_side="home",
            our_odds=scenario["our_odds"],
            bet_amount=100.0
        )

        tracker.record_bet_outcome(game_id, "moneyline", "home", scenario["won"])

    # Update closing lines
    closing_data = {}
    for i, scenario in enumerate(good_clv_scenarios):
        game_id = f"game_{i+1}"
        closing_data[game_id] = {
            "moneyline_home": scenario["closing_odds"],
            "moneyline_away": -110,
            "closing_timestamp": datetime.now()
        }

    tracker.update_closing_lines(closing_data)

    # Generate insights
    summary = tracker.calculate_clv_summary()
    insights = tracker.get_clv_insights(summary)

    # Verify insights structure
    assert "overall_assessment" in insights, "Should have overall assessment"
    assert "detailed_insights" in insights, "Should have detailed insights"
    assert "recommendation" in insights, "Should have recommendation"

    # Should be positive assessment for good CLV
    assert "🟢" in insights["overall_assessment"] or "🟡" in insights["overall_assessment"], \
           "Should have positive assessment for good CLV"

    logger.info(f"CLV Insights: {insights['overall_assessment']}")
    return True


def test_clv_export():
    """Test CLV analysis export functionality."""
    logger.info("Running test_clv_export")

    tracker = CLVTracker()

    # Add some test data
    tracker.record_bet("game_1", "spread", "home", -110, 100.0)
    tracker.record_bet_outcome("game_1", "spread", "home", True)

    closing_data = {
        "game_1": {
            "spread_home": -120,
            "closing_timestamp": datetime.now()
        }
    }
    tracker.update_closing_lines(closing_data)

    # Export to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
        temp_path = tmp_file.name

    try:
        tracker.export_clv_analysis(temp_path)

        # Verify file was created and contains valid JSON
        assert os.path.exists(temp_path), "Export file should exist"

        with open(temp_path, 'r') as f:
            import json
            export_data = json.load(f)

        # Verify export structure
        assert "clv_summary" in export_data, "Should have CLV summary"
        assert "bucket_performance" in export_data, "Should have bucket performance"
        assert "bet_details" in export_data, "Should have bet details"

        # Verify bet details
        assert len(export_data["bet_details"]) == 1, "Should have 1 bet with CLV data"
        bet_detail = export_data["bet_details"][0]
        assert "absolute_clv" in bet_detail, "Bet detail should have CLV data"

    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    logger.info("CLV export functionality verified")
    return True


def test_different_bet_types():
    """Test CLV calculation for different bet types."""
    logger.info("Running test_different_bet_types")

    tracker = CLVTracker()

    # Test different bet types and sides
    bet_types = [
        ("moneyline", "home", "moneyline_home"),
        ("moneyline", "away", "moneyline_away"),
        ("spread", "home", "spread_home"),
        ("spread", "away", "spread_away"),
        ("total", "over", "total_over"),
        ("total", "under", "total_under"),
    ]

    for i, (bet_type, bet_side, closing_key) in enumerate(bet_types):
        game_id = f"game_{i+1}"

        tracker.record_bet(
            game_id=game_id,
            bet_type=bet_type,
            bet_side=bet_side,
            our_odds=-110,
            bet_amount=100.0
        )

    # Update closing lines for all bet types
    closing_data = {}
    for i, (bet_type, bet_side, closing_key) in enumerate(bet_types):
        game_id = f"game_{i+1}"
        closing_data[game_id] = {
            closing_key: -120,  # Worse closing line
            "closing_timestamp": datetime.now()
        }

    tracker.update_closing_lines(closing_data)

    # Verify CLV was calculated for all bet types
    for bet_record in tracker.bet_records:
        assert bet_record.absolute_clv is not None, f"Should have CLV for {bet_record.bet_type} {bet_record.bet_side}"
        assert bet_record.absolute_clv < 0, "Should have negative CLV (worse line)"

    logger.info(f"Different bet types tested: {len(bet_types)} types successful")
    return True


def test_date_filtering():
    """Test date-based filtering in CLV summary."""
    logger.info("Running test_date_filtering")

    tracker = CLVTracker()

    # Create bets at different times
    old_date = datetime.now() - timedelta(days=30)
    recent_date = datetime.now() - timedelta(days=1)

    # Old bet
    tracker.record_bet("game_old", "moneyline", "home", -110, 100.0, old_date)

    # Recent bet
    tracker.record_bet("game_recent", "moneyline", "home", -110, 100.0, recent_date)

    # Update closing lines
    closing_data = {
        "game_old": {"moneyline_home": -120, "closing_timestamp": old_date},
        "game_recent": {"moneyline_home": -105, "closing_timestamp": recent_date}
    }
    tracker.update_closing_lines(closing_data)

    # Test date filtering
    cutoff_date = datetime.now() - timedelta(days=15)
    recent_summary = tracker.calculate_clv_summary(start_date=cutoff_date)

    assert recent_summary.total_bets == 1, "Should only include recent bet"
    assert recent_summary.bets_with_clv_data == 1, "Should have CLV data for recent bet"

    # Test full range
    full_summary = tracker.calculate_clv_summary()
    assert full_summary.total_bets == 2, "Should include both bets"

    logger.info("Date filtering functionality verified")
    return True


def run_all_tests():
    """Run all CLV tracking tests."""
    logger.info("Starting CLV tracking test suite")

    tests = [
        test_bet_recording,
        test_closing_line_updates,
        test_bet_outcomes,
        test_clv_summary_calculation,
        test_clv_bucket_analysis,
        test_clv_insights,
        test_clv_export,
        test_different_bet_types,
        test_date_filtering
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

    logger.info(f"CLV tracking tests completed: {passed} passed, {failed} failed")

    if failed == 0:
        logger.info("All CLV tracking tests passed!")
    else:
        logger.error(f"{failed} tests failed")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)