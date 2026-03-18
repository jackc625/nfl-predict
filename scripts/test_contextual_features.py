#!/usr/bin/env python3
"""
Test script for contextual features calculator.

This script validates the contextual features implementation with sample data.
"""

import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import pytz

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from features.contextual import ContextualFeaturesCalculator
from utils import get_logger

logger = get_logger(__name__)


def create_sample_games_data() -> pd.DataFrame:
    """Create sample games data for testing."""

    # Create sample games
    et_tz = pytz.timezone("America/New_York")

    sample_games = [
        {
            "game_id": "TEST_2024_01_BUF_MIA",
            "season": 2024,
            "week": 1,
            "home_team": "MIA",
            "away_team": "BUF",
            "venue_id": "hard_rock_stadium",
            "kickoff_et": et_tz.localize(datetime(2024, 9, 8, 13, 0)),  # Sunday 1 PM
        },
        {
            "game_id": "TEST_2024_01_LAR_SEA",
            "season": 2024,
            "week": 1,
            "home_team": "SEA",
            "away_team": "LAR",
            "venue_id": "lumen_field",
            "kickoff_et": et_tz.localize(
                datetime(2024, 9, 8, 16, 25)
            ),  # Sunday 4:25 PM
        },
        {
            "game_id": "TEST_2024_02_KC_BUF",
            "season": 2024,
            "week": 2,
            "home_team": "BUF",
            "away_team": "KC",
            "venue_id": "highmark_stadium",
            "kickoff_et": et_tz.localize(
                datetime(2024, 9, 12, 20, 20)
            ),  # Thursday Night Football
        },
        {
            "game_id": "TEST_2024_02_DEN_NE",
            "season": 2024,
            "week": 2,
            "home_team": "NE",
            "away_team": "DEN",
            "venue_id": "gillette_stadium",
            "kickoff_et": et_tz.localize(datetime(2024, 9, 15, 13, 0)),  # Sunday 1 PM
        },
    ]

    return pd.DataFrame(sample_games)


def test_travel_metrics():
    """Test travel metrics calculation."""
    logger.info("Testing travel metrics calculation...")

    calculator = ContextualFeaturesCalculator()

    # Test cross-country travel (LAR to SEA)
    et_tz = pytz.timezone("America/New_York")
    kickoff_dt = et_tz.localize(datetime(2024, 9, 8, 16, 25))

    travel_metrics = calculator.calculate_travel_metrics(
        away_team="LAR",
        home_team="SEA",
        game_venue_id="lumen_field",
        kickoff_datetime=kickoff_dt,
    )

    print("LAR @ SEA Travel Metrics:")
    for key, value in travel_metrics.items():
        print(f"  {key}: {value}")

    # Test East Coast travel (BUF to MIA)
    travel_metrics_2 = calculator.calculate_travel_metrics(
        away_team="BUF",
        home_team="MIA",
        game_venue_id="hard_rock_stadium",
        kickoff_datetime=kickoff_dt,
    )

    print("\nBUF @ MIA Travel Metrics:")
    for key, value in travel_metrics_2.items():
        print(f"  {key}: {value}")

    logger.info("Travel metrics test completed")


def test_short_week_detection():
    """Test short week detection."""
    logger.info("Testing short week detection...")

    calculator = ContextualFeaturesCalculator()
    et_tz = pytz.timezone("America/New_York")

    # Test Thursday Night Football
    thursday_dt = et_tz.localize(datetime(2024, 9, 12, 20, 20))
    short_week_features = calculator.detect_short_week(thursday_dt, 2024, 2)

    print("Thursday Night Football Features:")
    for key, value in short_week_features.items():
        print(f"  {key}: {value}")

    # Test Sunday game
    sunday_dt = et_tz.localize(datetime(2024, 9, 8, 13, 0))
    sunday_features = calculator.detect_short_week(sunday_dt, 2024, 1)

    print("\nSunday Game Features:")
    for key, value in sunday_features.items():
        print(f"  {key}: {value}")

    logger.info("Short week detection test completed")


def test_venue_features():
    """Test venue features encoding."""
    logger.info("Testing venue features encoding...")

    calculator = ContextualFeaturesCalculator()

    # Test outdoor stadium (Buffalo)
    buffalo_features = calculator.encode_venue_features("highmark_stadium")
    print("Buffalo (Outdoor) Venue Features:")
    for key, value in buffalo_features.items():
        print(f"  {key}: {value}")

    # Test indoor stadium (Detroit)
    detroit_features = calculator.encode_venue_features("ford_field")
    print("\nDetroit (Indoor) Venue Features:")
    for key, value in detroit_features.items():
        print(f"  {key}: {value}")

    # Test high altitude (Denver)
    denver_features = calculator.encode_venue_features("empower_field")
    print("\nDenver (High Altitude) Venue Features:")
    for key, value in denver_features.items():
        print(f"  {key}: {value}")

    logger.info("Venue features test completed")


def test_contextual_features_pipeline():
    """Test end-to-end contextual features pipeline."""
    logger.info("Testing contextual features pipeline...")

    # Create sample games data
    games_df = create_sample_games_data()

    # Initialize calculator
    calculator = ContextualFeaturesCalculator()

    # Build contextual features
    features_df = calculator.build_contextual_features(games_df)

    print(f"\nGenerated contextual features for {len(features_df)} games")
    print(f"Features columns: {len(features_df.columns)}")
    print("\nFeatures preview:")
    print(features_df.head())

    print("\nFeature columns:")
    for col in sorted(features_df.columns):
        if col not in ["game_id", "season", "week", "home_team", "away_team"]:
            sample_value = features_df[col].iloc[0] if len(features_df) > 0 else "N/A"
            print(f"  {col}: {sample_value}")

    # Validate features
    is_valid = calculator.validate_contextual_features(features_df)
    print(f"\nValidation result: {'PASS' if is_valid else 'FAIL'}")

    logger.info("Contextual features pipeline test completed")


def main():
    """Run all contextual features tests."""
    print("=" * 60)
    print("CONTEXTUAL FEATURES CALCULATOR TEST")
    print("=" * 60)

    try:
        test_travel_metrics()
        print("\n" + "-" * 40)

        test_short_week_detection()
        print("\n" + "-" * 40)

        test_venue_features()
        print("\n" + "-" * 40)

        test_contextual_features_pipeline()

        print("\n" + "=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 60)

    except Exception as e:
        logger.error("Test failed", error=str(e))
        print(f"\nTEST FAILED: {e}")
        raise


if __name__ == "__main__":
    main()
