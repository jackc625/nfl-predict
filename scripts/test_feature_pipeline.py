#!/usr/bin/env python3
"""
Test script for the unified feature building pipeline.

This script validates the feature pipeline with mock data from all feature sources.
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

from scripts.build_features import FeatureMatrixBuilder
from utils import get_logger

logger = get_logger(__name__)


def create_mock_games_data() -> pd.DataFrame:
    """Create mock games data for testing."""
    et_tz = pytz.timezone("America/New_York")

    mock_games = []
    for week in range(1, 3):  # 2 weeks
        week_games = [
            {
                "game_id": f"MOCK_2024_W{week:02d}_BUF@MIA",
                "season": 2024,
                "week": week,
                "home_team": "MIA",
                "away_team": "BUF",
                "kickoff_et": et_tz.localize(
                    datetime(2024, 9, 7 + (week - 1) * 7, 13, 0)
                ),
                "home_score": 24 if week == 1 else None,  # Only week 1 has results
                "away_score": 21 if week == 1 else None,
            },
            {
                "game_id": f"MOCK_2024_W{week:02d}_KC@DEN",
                "season": 2024,
                "week": week,
                "home_team": "DEN",
                "away_team": "KC",
                "kickoff_et": et_tz.localize(
                    datetime(2024, 9, 7 + (week - 1) * 7, 16, 25)
                ),
                "home_score": 17 if week == 1 else None,
                "away_score": 28 if week == 1 else None,
            },
            {
                "game_id": f"MOCK_2024_W{week:02d}_DAL@NYG",
                "season": 2024,
                "week": week,
                "home_team": "NYG",
                "away_team": "DAL",
                "kickoff_et": et_tz.localize(
                    datetime(2024, 9, 7 + (week - 1) * 7, 20, 20)
                ),
                "home_score": 13 if week == 1 else None,
                "away_score": 35 if week == 1 else None,
            },
        ]
        mock_games.extend(week_games)

    return pd.DataFrame(mock_games)


def create_mock_team_form() -> pd.DataFrame:
    """Create mock team form features."""
    teams = ["BUF", "MIA", "KC", "DEN", "DAL", "NYG"]
    weeks = [1, 2]
    sides = ["offense", "defense"]

    mock_form = []
    for team in teams:
        for week in weeks:
            for side in sides:
                # Generate realistic EPA values
                if side == "offense":
                    base_epa = np.random.normal(
                        0.05, 0.15
                    )  # Slight positive for offense
                else:
                    base_epa = np.random.normal(
                        -0.05, 0.15
                    )  # Slight negative for defense

                mock_form.append(
                    {
                        "team": team,
                        "target_season": 2024,
                        "target_week": week,
                        "side": side,
                        "rolling_epa_per_play": base_epa,
                        "rolling_pass_epa_per_play": base_epa
                        + np.random.normal(0, 0.05),
                        "rolling_rush_epa_per_play": base_epa
                        + np.random.normal(0, 0.05),
                        "rolling_success_rate": np.random.uniform(0.35, 0.55),
                        "rolling_neutral_pass_rate": np.random.uniform(0.55, 0.75)
                        if side == "offense"
                        else np.nan,
                    }
                )

    return pd.DataFrame(mock_form)


def create_mock_elo() -> pd.DataFrame:
    """Create mock Elo ratings."""
    teams = ["BUF", "MIA", "KC", "DEN", "DAL", "NYG"]
    weeks = [1, 2]

    mock_elo = []
    for team in teams:
        base_elo = 1500 + np.random.normal(0, 100)  # Teams around 1500
        for week in weeks:
            mock_elo.append(
                {
                    "team": team,
                    "season": 2024,
                    "week": week,
                    "elo_rating": base_elo + np.random.normal(0, 20),
                    "elo_uncertainty": np.random.uniform(50, 150),
                    "elo_games_played": week
                    + 15,  # Simulate games from previous season
                    "elo_form_rating": base_elo + np.random.normal(0, 30),
                }
            )

    return pd.DataFrame(mock_elo)


def create_mock_contextual() -> pd.DataFrame:
    """Create mock contextual features."""
    games = create_mock_games_data()

    mock_contextual = []
    for _, game in games.iterrows():
        mock_contextual.append(
            {
                "game_id": game["game_id"],
                "season": game["season"],
                "week": game["week"],
                "home_team": game["home_team"],
                "away_team": game["away_team"],
                "away_travel_distance_miles": np.random.uniform(200, 2500),
                "away_timezone_diff_hours": np.random.choice([-3, -2, -1, 0, 1, 2, 3]),
                "home_rest_days": np.random.choice([6, 7, 8, 9, 10]),
                "away_rest_days": np.random.choice([6, 7, 8, 9, 10]),
                "thursday_game": 1.0 if game["game_id"].endswith("THU") else 0.0,
                "venue_outdoor": np.random.choice([0.0, 1.0]),
                "venue_elevation_ft": np.random.uniform(0, 5300),
            }
        )

    return pd.DataFrame(mock_contextual)


def create_mock_weather() -> pd.DataFrame:
    """Create mock weather features."""
    games = create_mock_games_data()

    mock_weather = []
    for _, game in games.iterrows():
        outdoor = np.random.choice([True, False])
        mock_weather.append(
            {
                "game_id": game["game_id"],
                "season": game["season"],
                "week": game["week"],
                "weather_affects_game": 1.0 if outdoor else 0.0,
                "temp_f": np.random.uniform(20, 90) if outdoor else 72.0,
                "wind_mph": np.random.uniform(0, 25) if outdoor else 0.0,
                "precip_prob": np.random.uniform(0, 0.8) if outdoor else 0.0,
                "weather_severity_score": np.random.uniform(0, 0.8) if outdoor else 0.0,
                "wind_impact_score": np.random.uniform(0, 0.6) if outdoor else 0.0,
            }
        )

    return pd.DataFrame(mock_weather)


def create_mock_market() -> pd.DataFrame:
    """Create mock market anchor features."""
    games = create_mock_games_data()

    mock_market = []
    for _, game in games.iterrows():
        has_lines = np.random.choice([True, False], p=[0.8, 0.2])  # 80% have lines

        if has_lines:
            # Generate realistic spreads and totals
            spread = np.random.uniform(-14, 14)
            total = np.random.uniform(35, 65)

            mock_market.append(
                {
                    "game_id": game["game_id"],
                    "season": game["season"],
                    "week": game["week"],
                    "has_snapshot_lines": 1.0,
                    "has_line_movement": np.random.choice([0.0, 1.0]),
                    "snapshot_ml_prob_home_fair": np.random.uniform(0.2, 0.8),
                    "snapshot_ml_prob_away_fair": np.random.uniform(0.2, 0.8),
                    "snapshot_spread_prob_home_fair": np.random.uniform(0.4, 0.6),
                    "snapshot_spread_prob_away_fair": np.random.uniform(0.4, 0.6),
                    "snapshot_total_prob_over_fair": np.random.uniform(0.4, 0.6),
                    "snapshot_total_prob_under_fair": np.random.uniform(0.4, 0.6),
                    "snapshot_spread": spread,
                    "snapshot_total": total,
                    "ml_home_movement": np.random.uniform(-20, 20),
                    "spread_movement": np.random.uniform(-2, 2),
                    "total_movement": np.random.uniform(-3, 3),
                    "snapshot_ml_vig": np.random.uniform(0.03, 0.08),
                    "snapshot_spread_vig": np.random.uniform(0.03, 0.08),
                    "snapshot_total_vig": np.random.uniform(0.03, 0.08),
                }
            )
        else:
            # Default features for games without lines
            mock_market.append(
                {
                    "game_id": game["game_id"],
                    "season": game["season"],
                    "week": game["week"],
                    "has_snapshot_lines": 0.0,
                    "has_line_movement": 0.0,
                    "snapshot_ml_prob_home_fair": 0.5,
                    "snapshot_ml_prob_away_fair": 0.5,
                    "snapshot_spread_prob_home_fair": 0.5,
                    "snapshot_spread_prob_away_fair": 0.5,
                    "snapshot_total_prob_over_fair": 0.5,
                    "snapshot_total_prob_under_fair": 0.5,
                    "snapshot_spread": np.nan,
                    "snapshot_total": np.nan,
                    "ml_home_movement": 0.0,
                    "spread_movement": 0.0,
                    "total_movement": 0.0,
                    "snapshot_ml_vig": 0.05,
                    "snapshot_spread_vig": 0.05,
                    "snapshot_total_vig": 0.05,
                }
            )

    return pd.DataFrame(mock_market)


def test_feature_combination():
    """Test combining features from all sources."""
    logger.info("Testing feature combination...")

    builder = FeatureMatrixBuilder()

    # Create mock data
    feature_sources = {
        "games": create_mock_games_data(),
        "team_form": create_mock_team_form(),
        "elo": create_mock_elo(),
        "contextual": create_mock_contextual(),
        "weather": create_mock_weather(),
        "market": create_mock_market(),
    }

    print("Mock data summary:")
    for source, df in feature_sources.items():
        print(f"  {source}: {len(df)} records, {len(df.columns)} columns")

    # Combine features
    combined_features = builder.combine_features(feature_sources)

    print(
        f"\nCombined features: {len(combined_features)} games, {len(combined_features.columns)} total columns"
    )

    # Show feature breakdown
    feature_cols = [
        col
        for col in combined_features.columns
        if col
        not in [
            "game_id",
            "season",
            "week",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "feature_timestamp",
        ]
    ]

    feature_types = {}
    for col in feature_cols:
        if any(x in col for x in ["home_off_", "away_off_", "home_def_", "away_def_"]):
            feature_types.setdefault("Team Form", []).append(col)
        elif any(x in col for x in ["home_elo_", "away_elo_"]):
            feature_types.setdefault("Elo", []).append(col)
        elif any(x in col for x in ["travel_", "rest_", "venue_", "thursday_"]):
            feature_types.setdefault("Contextual", []).append(col)
        elif any(x in col for x in ["weather_", "temp_", "wind_", "precip_"]):
            feature_types.setdefault("Weather", []).append(col)
        elif any(x in col for x in ["snapshot_", "ml_", "spread_", "total_", "has_"]):
            feature_types.setdefault("Market", []).append(col)
        else:
            feature_types.setdefault("Other", []).append(col)

    print("\nFeature breakdown:")
    for ftype, fcols in feature_types.items():
        print(f"  {ftype}: {len(fcols)} features")
        # Show first few feature names
        print(f"    Examples: {', '.join(fcols[:3])}")

    logger.info("Feature combination test completed")
    return combined_features


def test_missing_data_handling():
    """Test missing data and outlier handling."""
    logger.info("Testing missing data and outlier handling...")

    # Create features with missing data and outliers
    combined_features = test_feature_combination()

    # Introduce missing data
    feature_cols = [
        col
        for col in combined_features.columns
        if col
        not in [
            "game_id",
            "season",
            "week",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "feature_timestamp",
        ]
    ]

    # Randomly set some values to NaN
    np.random.seed(42)
    for col in feature_cols[:5]:  # Test on first 5 features
        mask = np.random.random(len(combined_features)) < 0.3  # 30% missing
        combined_features.loc[mask, col] = np.nan

    # Introduce outliers
    for col in feature_cols[:3]:  # Test on first 3 features
        if combined_features[col].dtype in ["float64", "int64"]:
            outlier_mask = (
                np.random.random(len(combined_features)) < 0.1
            )  # 10% outliers
            outlier_values = combined_features[col].std() * 5  # 5 std deviations
            combined_features.loc[outlier_mask, col] = outlier_values

    print("Before processing:")
    missing_counts = combined_features[feature_cols].isnull().sum()
    print(f"  Features with missing data: {(missing_counts > 0).sum()}")
    print(f"  Total missing values: {missing_counts.sum()}")

    # Process missing data and outliers
    builder = FeatureMatrixBuilder()
    processed_features = builder.handle_missing_data_and_outliers(combined_features)

    print("After processing:")
    missing_counts_after = processed_features[feature_cols].isnull().sum()
    print(f"  Features with missing data: {(missing_counts_after > 0).sum()}")
    print(f"  Total missing values: {missing_counts_after.sum()}")

    logger.info("Missing data handling test completed")
    return processed_features


def test_normalization():
    """Test within-season normalization."""
    logger.info("Testing within-season normalization...")

    processed_features = test_missing_data_handling()

    # Add some data from different seasons to test normalization
    season_2023_data = processed_features.copy()
    season_2023_data["season"] = 2023
    season_2023_data["game_id"] = season_2023_data["game_id"].str.replace(
        "2024", "2023"
    )

    # Combine seasons
    multi_season_data = pd.concat(
        [processed_features, season_2023_data], ignore_index=True
    )

    builder = FeatureMatrixBuilder()

    # Get a few numeric features to check
    feature_cols = [
        col
        for col in multi_season_data.columns
        if col
        not in [
            "game_id",
            "season",
            "week",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "feature_timestamp",
        ]
    ]
    numeric_cols = (
        multi_season_data[feature_cols]
        .select_dtypes(include=[np.number])
        .columns.tolist()[:5]
    )

    print("Before normalization (sample features):")
    for col in numeric_cols:
        for season in [2023, 2024]:
            season_data = multi_season_data[multi_season_data["season"] == season][col]
            print(
                f"  {col} (season {season}): mean={season_data.mean():.3f}, std={season_data.std():.3f}"
            )

    # Normalize features
    normalized_features = builder.normalize_features_within_seasons(multi_season_data)

    print("\nAfter normalization (should be ~0 mean, ~1 std within season):")
    for col in numeric_cols:
        for season in [2023, 2024]:
            season_data = normalized_features[normalized_features["season"] == season][
                col
            ]
            print(
                f"  {col} (season {season}): mean={season_data.mean():.3f}, std={season_data.std():.3f}"
            )

    logger.info("Normalization test completed")
    return normalized_features


def test_target_creation():
    """Test target variable creation."""
    logger.info("Testing target variable creation...")

    normalized_features = test_normalization()

    builder = FeatureMatrixBuilder()
    features_with_targets = builder.create_target_variables(normalized_features)

    print("Target variables created:")

    # WP targets
    if "target_wp" in features_with_targets.columns:
        wp_values = features_with_targets["target_wp"].value_counts()
        print(f"  Win Probability: {wp_values.to_dict()}")

    # ATS targets (if spread data available)
    if "target_ats" in features_with_targets.columns:
        ats_count = features_with_targets["target_ats"].notna().sum()
        if ats_count > 0:
            ats_mean = features_with_targets["target_ats"].mean()
            print(f"  ATS: {ats_count} games, mean={ats_mean:.2f}")

    # O/U targets (if total data available)
    if "target_ou" in features_with_targets.columns:
        ou_count = features_with_targets["target_ou"].notna().sum()
        if ou_count > 0:
            ou_mean = features_with_targets["target_ou"].mean()
            print(f"  O/U: {ou_count} games, mean={ou_mean:.2f}")

    logger.info("Target creation test completed")
    return features_with_targets


def test_feature_matrices_generation():
    """Test generation of separate feature matrices."""
    logger.info("Testing feature matrices generation...")

    # Mock the load functions by providing data directly
    builder = FeatureMatrixBuilder()

    # Override the load method temporarily for testing
    original_load = builder.load_all_feature_sources

    def mock_load_all_feature_sources(target_season=None, target_week=None):
        return {
            "games": create_mock_games_data(),
            "team_form": create_mock_team_form(),
            "elo": create_mock_elo(),
            "contextual": create_mock_contextual(),
            "weather": create_mock_weather(),
            "market": create_mock_market(),
        }

    builder.load_all_feature_sources = mock_load_all_feature_sources

    # Generate feature matrices
    feature_matrices = builder.generate_feature_matrices()

    print("Generated feature matrices:")
    for target, matrix in feature_matrices.items():
        print(f"  {target.upper()}: {len(matrix)} games, {len(matrix.columns)} columns")

        # Show target distribution
        target_col = f"target_{target}"
        if target_col in matrix.columns:
            target_data = matrix[target_col].dropna()
            if len(target_data) > 0:
                print(
                    f"    Target range: {target_data.min():.3f} to {target_data.max():.3f}"
                )
                print(f"    Target mean: {target_data.mean():.3f}")

    # Restore original method
    builder.load_all_feature_sources = original_load

    logger.info("Feature matrices generation test completed")
    return feature_matrices


def main():
    """Run all feature pipeline tests."""
    print("=" * 60)
    print("FEATURE BUILDING PIPELINE TEST")
    print("=" * 60)

    try:
        # Test individual components
        print("\n1. Testing feature combination...")
        test_feature_combination()

        print("\n" + "-" * 40)
        print("2. Testing missing data handling...")
        test_missing_data_handling()

        print("\n" + "-" * 40)
        print("3. Testing normalization...")
        test_normalization()

        print("\n" + "-" * 40)
        print("4. Testing target creation...")
        test_target_creation()

        print("\n" + "-" * 40)
        print("5. Testing end-to-end pipeline...")
        feature_matrices = test_feature_matrices_generation()

        print("\n" + "=" * 60)
        print("ALL FEATURE PIPELINE TESTS COMPLETED SUCCESSFULLY")
        print("=" * 60)

        # Final summary
        if feature_matrices:
            total_features = (
                len(next(iter(feature_matrices.values())).columns) - 5
            )  # Exclude metadata
            print("\nFinal pipeline output:")
            print(f"  Total features per matrix: {total_features}")
            print(f"  WP matrix: {len(feature_matrices.get('wp', []))} games")
            print(f"  ATS matrix: {len(feature_matrices.get('ats', []))} games")
            print(f"  O/U matrix: {len(feature_matrices.get('ou', []))} games")

    except Exception as e:
        logger.error("Test failed", error=str(e))
        print(f"\nTEST FAILED: {e}")
        raise


if __name__ == "__main__":
    main()
