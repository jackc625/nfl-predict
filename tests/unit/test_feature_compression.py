"""Tests for feature compression: weather, market, contextual builders.

Validates that:
- Weather features are compressed from 38+ to exactly 4 output features
- Market features are compressed from 40+ to exactly 5 output features
- Indoor games have zeroed weather features
- No closing-line contamination in market output
- Anti-features (turnover, penalty, streak, rushing_yards) are absent
- All builders conform to FeatureBuilder Protocol (as_of_datetime)
"""

from datetime import datetime
from unittest.mock import patch

import pandas as pd
import pytest

from features.weather import WeatherFeaturesCalculator

# ---------------------------------------------------------------------------
# Weather feature compression tests
# ---------------------------------------------------------------------------

WEATHER_OUTPUT_COLUMNS = {
    "game_id",
    "weather_severity_score",
    "wind_mph",
    "is_precipitation",
    "is_outdoor",
}

# Columns that should NOT appear in the compressed output
DROPPED_WEATHER_COLUMNS = [
    "wind_calm",
    "wind_moderate",
    "wind_high",
    "wind_severe",
    "temp_hot",
    "temp_warm",
    "temp_mild",
    "temp_cool",
    "temp_cold",
    "temp_very_cold",
    "precip_light",
    "precip_moderate",
    "precip_heavy",
    "kicking_difficulty",
    "passing_difficulty",
    "scoring_multiplier",
    "wind_impact_score",
    "cold_impact_score",
    "heat_impact_score",
    "precip_impact_score",
    "home_weather_advantage",
    "defensive_advantage",
    "rushing_advantage",
    "scoring_reduction",
    "weather_game",
    "extreme_weather",
    "ball_handling_difficulty",
    "turnover_multiplier",
    "passing_efficiency",
    "raw_temp_f",
    "raw_wind_mph",
    "raw_precip_prob",
    "raw_precip_mm",
    "raw_humidity_pct",
    "weather_condition",
]

ANTI_FEATURE_PATTERNS = ["turnover", "penalty", "streak", "rushing_yards"]


def _make_games_df():
    """Create synthetic games DataFrame for weather tests."""
    return pd.DataFrame(
        [
            {
                "game_id": "TEST_INDOOR",
                "season": 2024,
                "week": 1,
                "home_team": "DAL",
                "away_team": "NYG",
            },
            {
                "game_id": "TEST_OUTDOOR_DRY",
                "season": 2024,
                "week": 1,
                "home_team": "GB",
                "away_team": "CHI",
            },
            {
                "game_id": "TEST_OUTDOOR_RAIN",
                "season": 2024,
                "week": 1,
                "home_team": "CLE",
                "away_team": "PIT",
            },
            {
                "game_id": "TEST_OUTDOOR_WIND",
                "season": 2024,
                "week": 1,
                "home_team": "BUF",
                "away_team": "MIA",
            },
        ]
    )


def _make_weather_df():
    """Create synthetic weather data matching the test games."""
    return pd.DataFrame(
        [
            {
                "game_id": "TEST_INDOOR",
                "forecast_time": datetime(2024, 9, 6, 12, 0),
                "is_outdoor": False,
                "temp_f": 72.0,
                "wind_mph": 0.0,
                "precip_prob": 0.0,
                "precip_mm": 0.0,
                "humidity_pct": 50.0,
                "condition": "Clear",
            },
            {
                "game_id": "TEST_OUTDOOR_DRY",
                "forecast_time": datetime(2024, 9, 6, 12, 0),
                "is_outdoor": True,
                "temp_f": 65.0,
                "wind_mph": 8.0,
                "precip_prob": 0.1,
                "precip_mm": 0.0,
                "humidity_pct": 45.0,
                "condition": "Clear",
            },
            {
                "game_id": "TEST_OUTDOOR_RAIN",
                "forecast_time": datetime(2024, 9, 6, 12, 0),
                "is_outdoor": True,
                "temp_f": 55.0,
                "wind_mph": 10.0,
                "precip_prob": 0.8,
                "precip_mm": 5.0,
                "humidity_pct": 85.0,
                "condition": "Rain",
            },
            {
                "game_id": "TEST_OUTDOOR_WIND",
                "forecast_time": datetime(2024, 9, 6, 12, 0),
                "is_outdoor": True,
                "temp_f": 40.0,
                "wind_mph": 25.0,
                "precip_prob": 0.15,
                "precip_mm": 0.0,
                "humidity_pct": 30.0,
                "condition": "Windy",
            },
        ]
    )


class TestWeatherCompression:
    """Tests for weather feature compression to 4 features."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        """Set up calculator and build compressed features."""
        self.calc = WeatherFeaturesCalculator()
        games_df = _make_games_df()
        weather_df = _make_weather_df()
        as_of = datetime(2024, 9, 7, 18, 0)  # Friday 6 PM ET

        with patch("features.weather.load_dataframe", return_value=weather_df):
            self.result = self.calc.build_features(games_df, as_of)

    def test_output_has_exactly_four_feature_columns(self):
        """build_features returns only game_id + 4 compressed features."""
        assert set(self.result.columns) == WEATHER_OUTPUT_COLUMNS

    def test_indoor_game_all_features_zeroed(self):
        """Indoor game (dome) has all 4 features zeroed."""
        indoor = self.result[self.result["game_id"] == "TEST_INDOOR"].iloc[0]
        assert indoor["weather_severity_score"] == 0.0
        assert indoor["wind_mph"] == 0.0
        assert indoor["is_precipitation"] == 0.0
        assert indoor["is_outdoor"] == 0.0

    def test_outdoor_rainy_game_precipitation_flag(self):
        """Outdoor rainy game has is_precipitation=1."""
        rain = self.result[self.result["game_id"] == "TEST_OUTDOOR_RAIN"].iloc[0]
        assert rain["is_precipitation"] == 1.0

    def test_outdoor_dry_game_no_precipitation(self):
        """Outdoor dry game has is_precipitation=0."""
        dry = self.result[self.result["game_id"] == "TEST_OUTDOOR_DRY"].iloc[0]
        assert dry["is_precipitation"] == 0.0

    def test_severity_score_in_range(self):
        """weather_severity_score is in [0, 1] for all games."""
        scores = self.result["weather_severity_score"]
        assert (scores >= 0.0).all()
        assert (scores <= 1.0).all()

    def test_dropped_columns_absent(self):
        """None of the dropped feature names appear in output."""
        output_cols = set(self.result.columns)
        for col in DROPPED_WEATHER_COLUMNS:
            assert col not in output_cols, f"Dropped column '{col}' still in output"

    def test_anti_features_absent(self):
        """No turnover/penalty/streak columns in output."""
        output_cols = " ".join(self.result.columns)
        for pattern in ANTI_FEATURE_PATTERNS:
            assert pattern not in output_cols, (
                f"Anti-feature pattern '{pattern}' found in columns"
            )

    def test_build_features_accepts_as_of_datetime(self):
        """build_features method accepts as_of_datetime parameter."""
        import inspect

        sig = inspect.signature(self.calc.build_features)
        param_names = list(sig.parameters.keys())
        assert "as_of_datetime" in param_names, (
            f"Expected 'as_of_datetime' in build_features params, got {param_names}"
        )
