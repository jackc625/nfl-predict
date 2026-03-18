"""Pytest configuration and fixtures for NFL Prediction System tests."""

import shutil

# Add project root to path
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture(scope="session")
def project_root_path():
    """Return the project root path."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def temp_data_dir():
    """Create a temporary data directory for tests."""
    temp_dir = tempfile.mkdtemp()
    data_dir = Path(temp_dir) / "data"

    # Create data structure
    (data_dir / "bronze").mkdir(parents=True)
    (data_dir / "silver").mkdir(parents=True)
    (data_dir / "gold").mkdir(parents=True)

    yield data_dir

    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_games_data():
    """Create mock games data for testing."""
    et_tz = ZoneInfo("America/New_York")

    mock_games = []
    for week in range(1, 4):  # 3 weeks
        week_games = [
            {
                "game_id": f"MOCK_2024_W{week:02d}_BUF@MIA",
                "season": 2024,
                "week": week,
                "home_team": "MIA",
                "away_team": "BUF",
                "kickoff_et": datetime(
                    2024, 9, 7 + (week - 1) * 7, 13, 0, tzinfo=et_tz
                ),
                "home_score": 24 if week <= 2 else None,  # First 2 weeks have results
                "away_score": 21 if week <= 2 else None,
            },
            {
                "game_id": f"MOCK_2024_W{week:02d}_KC@DEN",
                "season": 2024,
                "week": week,
                "home_team": "DEN",
                "away_team": "KC",
                "kickoff_et": datetime(
                    2024, 9, 7 + (week - 1) * 7, 16, 25, tzinfo=et_tz
                ),
                "home_score": 17 if week <= 2 else None,
                "away_score": 28 if week <= 2 else None,
            },
        ]
        mock_games.extend(week_games)

    return pd.DataFrame(mock_games)


@pytest.fixture
def mock_team_form_data():
    """Create mock team form data."""
    teams = ["BUF", "MIA", "KC", "DEN"]
    weeks = [1, 2, 3]
    sides = ["offense", "defense"]

    mock_form = []
    for team in teams:
        for week in weeks:
            for side in sides:
                base_epa = np.random.normal(0.05 if side == "offense" else -0.05, 0.15)
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


@pytest.fixture
def mock_elo_data():
    """Create mock Elo ratings data."""
    teams = ["BUF", "MIA", "KC", "DEN"]
    weeks = [1, 2, 3]

    mock_elo = []
    for team in teams:
        base_elo = 1500 + np.random.normal(0, 100)
        for week in weeks:
            mock_elo.append(
                {
                    "team": team,
                    "season": 2024,
                    "week": week,
                    "elo_rating": base_elo + np.random.normal(0, 20),
                    "elo_uncertainty": np.random.uniform(50, 150),
                    "elo_games_played": week + 15,
                    "elo_form_rating": base_elo + np.random.normal(0, 30),
                }
            )

    return pd.DataFrame(mock_elo)


@pytest.fixture
def mock_odds_data():
    """Create mock odds data."""
    games = [
        "MOCK_2024_W01_BUF@MIA",
        "MOCK_2024_W01_KC@DEN",
        "MOCK_2024_W02_BUF@MIA",
        "MOCK_2024_W02_KC@DEN",
        "MOCK_2024_W03_BUF@MIA",
        "MOCK_2024_W03_KC@DEN",
    ]

    mock_odds = []
    for game_id in games:
        mock_odds.append(
            {
                "game_id": game_id,
                "snapshot_spread": np.random.uniform(-14, 14),
                "snapshot_total": np.random.uniform(35, 65),
                "snapshot_ml_home": np.random.randint(-300, 300),
                "snapshot_ml_away": np.random.randint(-300, 300),
                "has_snapshot_lines": 1.0,
            }
        )

    return pd.DataFrame(mock_odds)


@pytest.fixture
def mock_weather_data():
    """Create mock weather data."""
    games = [
        "MOCK_2024_W01_BUF@MIA",
        "MOCK_2024_W01_KC@DEN",
        "MOCK_2024_W02_BUF@MIA",
        "MOCK_2024_W02_KC@DEN",
        "MOCK_2024_W03_BUF@MIA",
        "MOCK_2024_W03_KC@DEN",
    ]

    mock_weather = []
    for game_id in games:
        outdoor = np.random.choice([True, False])
        mock_weather.append(
            {
                "game_id": game_id,
                "weather_affects_game": 1.0 if outdoor else 0.0,
                "temp_f": np.random.uniform(20, 90) if outdoor else 72.0,
                "wind_mph": np.random.uniform(0, 25) if outdoor else 0.0,
                "precip_prob": np.random.uniform(0, 0.8) if outdoor else 0.0,
                "weather_severity_score": np.random.uniform(0, 0.8) if outdoor else 0.0,
            }
        )

    return pd.DataFrame(mock_weather)


@pytest.fixture
def sample_feature_matrix():
    """Create a sample feature matrix with realistic data."""
    np.random.seed(42)  # For reproducible tests

    # Create base games
    games = [
        {
            "game_id": "TEST_2024_W01_BUF@MIA",
            "season": 2024,
            "week": 1,
            "home_team": "MIA",
            "away_team": "BUF",
        },
        {
            "game_id": "TEST_2024_W01_KC@DEN",
            "season": 2024,
            "week": 1,
            "home_team": "DEN",
            "away_team": "KC",
        },
        {
            "game_id": "TEST_2024_W02_DAL@NYG",
            "season": 2024,
            "week": 2,
            "home_team": "NYG",
            "away_team": "DAL",
        },
    ]

    features = []
    for game in games:
        feature_row = game.copy()

        # Add Elo features
        feature_row.update(
            {
                "home_elo_rating": np.random.uniform(1400, 1600),
                "away_elo_rating": np.random.uniform(1400, 1600),
                "elo_diff": np.random.uniform(-200, 200),
            }
        )

        # Add form features
        for side in ["home_off", "away_off", "home_def", "away_def"]:
            feature_row.update(
                {
                    f"{side}_rolling_epa_per_play": np.random.normal(0, 0.2),
                    f"{side}_rolling_success_rate": np.random.uniform(0.3, 0.6),
                }
            )

        # Add contextual features
        feature_row.update(
            {
                "venue_outdoor": np.random.choice([0.0, 1.0]),
                "away_travel_distance_miles": np.random.uniform(200, 2500),
                "home_rest_days": np.random.choice([6, 7, 8, 9, 10]),
                "away_rest_days": np.random.choice([6, 7, 8, 9, 10]),
            }
        )

        # Add weather features
        feature_row.update(
            {
                "temp_f": np.random.uniform(20, 90),
                "wind_mph": np.random.uniform(0, 25),
                "weather_severity_score": np.random.uniform(0, 1),
            }
        )

        # Add market features
        feature_row.update(
            {
                "snapshot_spread": np.random.uniform(-14, 14),
                "snapshot_total": np.random.uniform(35, 65),
                "has_snapshot_lines": 1.0,
            }
        )

        # Add targets
        feature_row.update(
            {
                "target_wp": np.random.choice([0, 1]),
                "target_ats": np.random.uniform(-30, 30),
                "target_ou": np.random.uniform(-15, 15),
            }
        )

        features.append(feature_row)

    return pd.DataFrame(features)


@pytest.fixture
def api_client():
    """Create a test client for the FastAPI application."""
    from api.main import app

    return TestClient(app)


@pytest.fixture
def mock_predictions_data():
    """Create mock predictions data for API tests."""
    return {
        "current_week": 3,
        "current_season": 2024,
        "predictions_available": True,
        "last_updated": "2024-09-15T18:00:00-04:00",
        "games": [
            {
                "game_id": "TEST_2024_W03_BUF@MIA",
                "season": 2024,
                "week": 3,
                "home_team": "MIA",
                "away_team": "BUF",
                "kickoff_et": "2024-09-15T13:00:00-04:00",
                "wp_prob_home": 0.58,
                "wp_prob_away": 0.42,
                "wp_confidence": 0.73,
                "ats_prob_home": 0.52,
                "ats_prob_away": 0.48,
                "ats_confidence": 0.64,
                "ou_prob_over": 0.49,
                "ou_prob_under": 0.51,
                "ou_confidence": 0.56,
                "spread_line": -3.5,
                "total_line": 45.5,
                "wp_edge": 0.08,
                "ats_edge": 0.02,
                "ou_edge": 0.01,
            }
        ],
    }


@pytest.fixture
def mock_backtest_data():
    """Create mock backtest results for API tests."""
    return {
        "summary_metrics": {
            "wp_log_loss": 0.642,
            "wp_brier_score": 0.241,
            "wp_accuracy": 0.652,
            "ats_accuracy": 0.534,
            "ou_accuracy": 0.518,
            "betting_roi": 0.067,
            "total_games": 1024,
            "seasons_tested": ["2020", "2021", "2022", "2023", "2024"],
        },
        "season_breakdown": [
            {
                "season": 2024,
                "wp_accuracy": 0.658,
                "ats_accuracy": 0.541,
                "ou_accuracy": 0.523,
                "betting_roi": 0.071,
                "games_count": 256,
            }
        ],
    }


@pytest.fixture
def mock_calibration_data():
    """Create mock calibration data for API tests."""
    return {
        "wp_calibration": {
            "bin_centers": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "observed_frequencies": [
                0.09,
                0.18,
                0.31,
                0.39,
                0.52,
                0.61,
                0.72,
                0.81,
                0.91,
            ],
            "bin_counts": [45, 78, 92, 123, 156, 134, 89, 67, 32],
        },
        "ats_calibration": {
            "bin_centers": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            "observed_frequencies": [
                0.11,
                0.19,
                0.28,
                0.42,
                0.48,
                0.59,
                0.69,
                0.78,
                0.88,
            ],
            "bin_counts": [67, 89, 134, 156, 178, 123, 92, 78, 45],
        },
    }


@pytest.fixture(autouse=True)
def setup_test_environment(temp_data_dir, monkeypatch):
    """Set up test environment with temporary directories."""
    # Patch data directory paths
    monkeypatch.setenv("NFL_DATA_DIR", str(temp_data_dir))

    # Set up logging for tests
    import logging

    logging.basicConfig(level=logging.WARNING)  # Reduce log noise in tests


# Utility functions for tests
def assert_dataframe_structure(df, expected_columns=None, min_rows=0):
    """Assert DataFrame has expected structure."""
    assert isinstance(df, pd.DataFrame), "Expected pandas DataFrame"
    assert len(df) >= min_rows, f"Expected at least {min_rows} rows, got {len(df)}"

    if expected_columns:
        missing_cols = set(expected_columns) - set(df.columns)
        assert not missing_cols, f"Missing columns: {missing_cols}"


def assert_probability_range(values, tolerance=1e-6):
    """Assert values are valid probabilities (0-1 range)."""
    values = pd.Series(values).dropna()
    assert values.min() >= -tolerance, f"Found probability < 0: {values.min()}"
    assert values.max() <= 1 + tolerance, f"Found probability > 1: {values.max()}"


def assert_no_data_leakage(df, prediction_date=None):
    """Assert no future data is used in features."""
    if prediction_date and "feature_timestamp" in df.columns:
        future_data = df[df["feature_timestamp"] > prediction_date]
        assert len(future_data) == 0, f"Found {len(future_data)} rows with future data"
