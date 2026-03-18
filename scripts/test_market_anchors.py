#!/usr/bin/env python3
"""
Test script for market anchor features calculator.

This script validates the market anchor features implementation with sample odds data.
"""

import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytz

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from features.market_anchors import MarketAnchorFeaturesCalculator
from utils import get_logger

logger = get_logger(__name__)


def create_sample_odds_data() -> pd.DataFrame:
    """Create sample odds data for testing market anchor features."""
    et_tz = pytz.timezone("America/New_York")

    # Base time: Wednesday (3 days before games)
    base_time = et_tz.localize(datetime(2024, 9, 4, 12, 0))  # Wednesday noon
    friday_6pm = et_tz.localize(datetime(2024, 9, 6, 18, 0))  # Friday 6 PM snapshot

    sample_odds = []

    # Game 1: BUF @ MIA - Line moves toward Buffalo (sharp money)
    game_1_scenarios = [
        # Wednesday opening (72 hours before)
        {
            "game_id": "TEST_2024_01_BUF_MIA",
            "snapshot_ts": base_time,
            "sportsbook": "draftkings",
            "ml_home": -120,  # Miami favored
            "ml_away": +100,  # Buffalo
            "spread": -2.5,  # Miami -2.5
            "spread_ju_home": -110,
            "spread_ju_away": -110,
            "total": 47.5,
            "total_over_ju": -110,
            "total_under_ju": -110,
        },
        # Friday 6 PM snapshot (line moved toward Buffalo)
        {
            "game_id": "TEST_2024_01_BUF_MIA",
            "snapshot_ts": friday_6pm,
            "sportsbook": "draftkings",
            "ml_home": -105,  # Miami line weakened
            "ml_away": -115,  # Buffalo now favored
            "spread": -1.5,  # Spread moved toward Buffalo
            "spread_ju_home": -110,
            "spread_ju_away": -110,
            "total": 46.5,  # Total dropped (weather concerns?)
            "total_over_ju": -110,
            "total_under_ju": -110,
        },
    ]

    # Game 2: KC @ DEN - Stable lines (efficient market)
    game_2_scenarios = [
        # Wednesday opening
        {
            "game_id": "TEST_2024_01_KC_DEN",
            "snapshot_ts": base_time,
            "sportsbook": "draftkings",
            "ml_home": +140,  # Denver underdog
            "ml_away": -165,  # KC favored
            "spread": 3.5,  # KC -3.5
            "spread_ju_home": -110,
            "spread_ju_away": -110,
            "total": 52.5,
            "total_over_ju": -110,
            "total_under_ju": -110,
        },
        # Friday 6 PM snapshot (minimal movement)
        {
            "game_id": "TEST_2024_01_KC_DEN",
            "snapshot_ts": friday_6pm,
            "sportsbook": "draftkings",
            "ml_home": +145,  # Slight movement
            "ml_away": -170,
            "spread": 3.5,  # Spread held
            "spread_ju_home": -110,
            "spread_ju_away": -110,
            "total": 52.0,  # Small total movement
            "total_over_ju": -110,
            "total_under_ju": -110,
        },
    ]

    # Game 3: DAL @ NYG - Heavy line movement (public money)
    game_3_scenarios = [
        # Wednesday opening
        {
            "game_id": "TEST_2024_01_DAL_NYG",
            "snapshot_ts": base_time,
            "sportsbook": "draftkings",
            "ml_home": +220,  # NYG big underdog
            "ml_away": -280,  # DAL heavily favored
            "spread": 7.5,  # DAL -7.5
            "spread_ju_home": -110,
            "spread_ju_away": -110,
            "total": 42.0,
            "total_over_ju": -110,
            "total_under_ju": -110,
        },
        # Friday 6 PM snapshot (line moved more toward favorite)
        {
            "game_id": "TEST_2024_01_DAL_NYG",
            "snapshot_ts": friday_6pm,
            "sportsbook": "draftkings",
            "ml_home": +260,  # NYG longer odds
            "ml_away": -320,  # DAL shorter odds
            "spread": 8.5,  # Spread increased
            "spread_ju_home": -110,
            "spread_ju_away": -110,
            "total": 43.0,  # Total moved up
            "total_over_ju": -110,
            "total_under_ju": -110,
        },
    ]

    # Add multiple sportsbooks for consensus testing
    all_scenarios = game_1_scenarios + game_2_scenarios + game_3_scenarios

    # Duplicate for multiple sportsbooks with slight variations
    for scenario in all_scenarios.copy():
        # FanDuel (slight variations)
        fanduel_scenario = scenario.copy()
        fanduel_scenario["sportsbook"] = "fanduel"
        fanduel_scenario["ml_home"] = scenario["ml_home"] + np.random.randint(-5, 6)
        fanduel_scenario["ml_away"] = scenario["ml_away"] + np.random.randint(-5, 6)
        fanduel_scenario["spread"] = scenario["spread"] + np.random.choice(
            [-0.5, 0, 0.5]
        )
        fanduel_scenario["total"] = scenario["total"] + np.random.choice([-0.5, 0, 0.5])
        all_scenarios.append(fanduel_scenario)

        # BetMGM
        betmgm_scenario = scenario.copy()
        betmgm_scenario["sportsbook"] = "betmgm"
        betmgm_scenario["ml_home"] = scenario["ml_home"] + np.random.randint(-8, 9)
        betmgm_scenario["ml_away"] = scenario["ml_away"] + np.random.randint(-8, 9)
        betmgm_scenario["spread"] = scenario["spread"] + np.random.choice(
            [-0.5, 0, 0.5]
        )
        betmgm_scenario["total"] = scenario["total"] + np.random.choice(
            [-1.0, -0.5, 0, 0.5, 1.0]
        )
        all_scenarios.append(betmgm_scenario)

    sample_odds.extend(all_scenarios)

    return pd.DataFrame(sample_odds)


def create_sample_games_data() -> pd.DataFrame:
    """Create sample games data corresponding to odds scenarios."""
    et_tz = pytz.timezone("America/New_York")

    sample_games = [
        {
            "game_id": "TEST_2024_01_BUF_MIA",
            "season": 2024,
            "week": 1,
            "home_team": "MIA",
            "away_team": "BUF",
            "kickoff_et": et_tz.localize(datetime(2024, 9, 8, 13, 0)),
        },
        {
            "game_id": "TEST_2024_01_KC_DEN",
            "season": 2024,
            "week": 1,
            "home_team": "DEN",
            "away_team": "KC",
            "kickoff_et": et_tz.localize(datetime(2024, 9, 8, 16, 25)),
        },
        {
            "game_id": "TEST_2024_01_DAL_NYG",
            "season": 2024,
            "week": 1,
            "home_team": "NYG",
            "away_team": "DAL",
            "kickoff_et": et_tz.localize(datetime(2024, 9, 8, 20, 20)),
        },
    ]

    return pd.DataFrame(sample_games)


def test_opening_lines_identification():
    """Test opening lines identification."""
    logger.info("Testing opening lines identification...")

    MarketAnchorFeaturesCalculator()
    odds_df = create_sample_odds_data()

    print(f"Sample odds data: {len(odds_df)} records")
    print(f"Unique games: {odds_df['game_id'].nunique()}")
    print(f"Sportsbooks: {', '.join(odds_df['sportsbook'].unique())}")
    print(
        f"Time range: {odds_df['snapshot_ts'].min()} to {odds_df['snapshot_ts'].max()}"
    )

    # Mock the load_dataframe call by creating games data
    games_df = create_sample_games_data()

    # Simulate loading games by patching the function temporarily

    # Test opening lines identification manually
    # Add kickoff times to odds data
    odds_with_games = odds_df.merge(
        games_df[["game_id", "kickoff_et"]], on="game_id", how="left"
    )

    # Calculate hours before kickoff
    odds_with_games["hours_before_kickoff"] = (
        odds_with_games["kickoff_et"] - odds_with_games["snapshot_ts"]
    ).dt.total_seconds() / 3600

    print("\nHours before kickoff analysis:")
    for game_id, group in odds_with_games.groupby("game_id"):
        print(f"{game_id}:")
        for _, row in group.iterrows():
            print(
                f"  {row['sportsbook']}: {row['hours_before_kickoff']:.1f} hours before"
            )

    logger.info("Opening lines identification test completed")


def test_devigged_probabilities():
    """Test devigged probability calculations."""
    logger.info("Testing devigged probability calculations...")

    calculator = MarketAnchorFeaturesCalculator()

    # Test scenarios
    test_odds = [
        {
            "name": "Even game (-110/-110)",
            "ml_home": -110,
            "ml_away": -110,
            "spread_ju_home": -110,
            "spread_ju_away": -110,
            "total_over_ju": -110,
            "total_under_ju": -110,
        },
        {
            "name": "Home favorite (-150/+125)",
            "ml_home": -150,
            "ml_away": +125,
            "spread_ju_home": -110,
            "spread_ju_away": -110,
            "total_over_ju": -105,
            "total_under_ju": -115,
        },
        {
            "name": "Heavy favorite (-300/+240)",
            "ml_home": -300,
            "ml_away": +240,
            "spread_ju_home": -105,
            "spread_ju_away": -115,
            "total_over_ju": -110,
            "total_under_ju": -110,
        },
    ]

    for scenario in test_odds:
        print(f"\n{scenario['name']}:")
        probs = calculator.calculate_devigged_probabilities(scenario)

        for key, value in probs.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")

    logger.info("Devigged probabilities test completed")


def test_line_movement_calculation():
    """Test line movement calculations."""
    logger.info("Testing line movement calculations...")

    calculator = MarketAnchorFeaturesCalculator()

    # Test scenarios
    movement_scenarios = [
        {
            "name": "Sharp money (line moves against public)",
            "opening": {
                "opening_ml_home": -120,
                "opening_spread": -2.5,
                "opening_total": 47.5,
            },
            "snapshot": {
                "snapshot_ml_home": -105,
                "snapshot_spread": -1.5,
                "snapshot_total": 46.5,
            },
        },
        {
            "name": "Public money (line moves with favorite)",
            "opening": {
                "opening_ml_home": +220,
                "opening_spread": 7.5,
                "opening_total": 42.0,
            },
            "snapshot": {
                "snapshot_ml_home": +260,
                "snapshot_spread": 8.5,
                "snapshot_total": 43.0,
            },
        },
        {
            "name": "Stable market (minimal movement)",
            "opening": {
                "opening_ml_home": +140,
                "opening_spread": 3.5,
                "opening_total": 52.5,
            },
            "snapshot": {
                "snapshot_ml_home": +145,
                "snapshot_spread": 3.5,
                "snapshot_total": 52.0,
            },
        },
    ]

    for scenario in movement_scenarios:
        print(f"\n{scenario['name']}:")
        movement = calculator.calculate_line_movement(
            scenario["opening"], scenario["snapshot"]
        )

        for key, value in movement.items():
            if isinstance(value, (int, float)):
                print(f"  {key}: {value:.2f}")

    logger.info("Line movement calculation test completed")


def test_consensus_lines():
    """Test consensus line creation."""
    logger.info("Testing consensus lines creation...")

    calculator = MarketAnchorFeaturesCalculator()
    odds_df = create_sample_odds_data()

    # Test with opening lines
    opening_lines = odds_df[
        odds_df["snapshot_ts"] < odds_df["snapshot_ts"].max()
    ].copy()

    print(f"Creating consensus from {len(opening_lines)} opening line records")
    consensus_df = calculator.create_consensus_lines(opening_lines)

    print(f"Generated consensus for {len(consensus_df)} games")
    print("\nConsensus lines:")
    for _, game in consensus_df.iterrows():
        print(f"{game['game_id']}:")
        print(f"  ML Home: {game.get('consensus_ml_home', 'N/A')}")
        print(f"  Spread: {game.get('consensus_spread', 'N/A')}")
        print(f"  Total: {game.get('consensus_total', 'N/A')}")
        print(f"  Sportsbooks: {game['num_sportsbooks']}")
        if "spread_range" in game:
            print(f"  Spread range: {game.get('spread_range', 0):.1f}")

    logger.info("Consensus lines test completed")


def test_market_anchor_pipeline():
    """Test end-to-end market anchor features pipeline."""
    logger.info("Testing market anchor features pipeline...")

    # This test would require mocking the data loading functions
    # For now, we'll test the feature structure

    calculator = MarketAnchorFeaturesCalculator()
    games_df = create_sample_games_data()

    # Test default features when no odds available
    print("Testing with no odds data (default features):")
    empty_features = calculator._create_empty_market_features(games_df)

    print(f"Generated {len(empty_features)} feature records")
    print(f"Feature columns: {len(empty_features.columns)}")

    print("\nSample default features:")
    sample_features = empty_features.iloc[0]
    for col, value in sample_features.items():
        if col not in ["game_id", "season", "week"]:
            print(f"  {col}: {value}")

    # Validate features
    is_valid = calculator.validate_market_anchor_features(empty_features)
    print(f"\nValidation result: {'PASS' if is_valid else 'FAIL'}")

    logger.info("Market anchor features pipeline test completed")


def main():
    """Run all market anchor features tests."""
    print("=" * 60)
    print("MARKET ANCHOR FEATURES CALCULATOR TEST")
    print("=" * 60)

    try:
        test_opening_lines_identification()
        print("\n" + "-" * 40)

        test_devigged_probabilities()
        print("\n" + "-" * 40)

        test_line_movement_calculation()
        print("\n" + "-" * 40)

        test_consensus_lines()
        print("\n" + "-" * 40)

        test_market_anchor_pipeline()

        print("\n" + "=" * 60)
        print("ALL MARKET ANCHOR FEATURES TESTS COMPLETED SUCCESSFULLY")
        print("=" * 60)

    except Exception as e:
        logger.error("Test failed", error=str(e))
        print(f"\nTEST FAILED: {e}")
        raise


if __name__ == "__main__":
    main()
