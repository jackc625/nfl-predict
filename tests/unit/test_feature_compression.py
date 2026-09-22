"""Tests for feature compression: weather, market, contextual builders.

Validates that:
- Weather features are compressed from 38+ to exactly 4 output features
- Market features are compressed from 40+ to exactly 5 output features
- Indoor games have zeroed weather features
- No closing-line contamination in market output
- Anti-features (turnover, penalty, streak, rushing_yards) are absent
- All builders conform to FeatureBuilder Protocol (as_of_datetime)
"""

from datetime import UTC, datetime
from unittest.mock import patch

import pandas as pd
import pytest

from features.contextual import ContextualFeaturesCalculator
from features.market_anchors import MarketAnchorFeaturesCalculator
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


# THE ONE WEATHER FENCE (Plan 33.2-12) admits a live forecast row when its fetch instant is at
# or before min(the game's lock, the build instant), so the synthetic rows are LIVE forecast
# rows fetched on the Friday, the games kick off on Sunday 13:00 ET, and every instant is
# timezone-aware (a naive instant is refused, never relabelled).
LIVE_FETCH_UTC = pd.Timestamp("2024-09-06 12:00", tz="UTC")
KICKOFF_UTC = pd.Timestamp("2024-09-08 17:00", tz="UTC")


def _make_games_df():
    """Create synthetic games DataFrame for weather tests."""
    return pd.DataFrame(
        [
            {
                "game_id": "TEST_INDOOR",
                "season": 2024,
                "week": 1,
                "kickoff_et": KICKOFF_UTC,
                "home_team": "DAL",
                "away_team": "NYG",
            },
            {
                "game_id": "TEST_OUTDOOR_DRY",
                "season": 2024,
                "week": 1,
                "kickoff_et": KICKOFF_UTC,
                "home_team": "GB",
                "away_team": "CHI",
            },
            {
                "game_id": "TEST_OUTDOOR_RAIN",
                "season": 2024,
                "week": 1,
                "kickoff_et": KICKOFF_UTC,
                "home_team": "CLE",
                "away_team": "PIT",
            },
            {
                "game_id": "TEST_OUTDOOR_WIND",
                "season": 2024,
                "week": 1,
                "kickoff_et": KICKOFF_UTC,
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
                "forecast_time": LIVE_FETCH_UTC,
                "weather_source": "forecast",
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
                "forecast_time": LIVE_FETCH_UTC,
                "weather_source": "forecast",
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
                "forecast_time": LIVE_FETCH_UTC,
                "weather_source": "forecast",
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
                "forecast_time": LIVE_FETCH_UTC,
                "weather_source": "forecast",
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
        as_of = datetime(2024, 9, 7, 22, 0, tzinfo=UTC)  # 18:00 ET, the lock

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


# ---------------------------------------------------------------------------
# Market feature compression tests
# ---------------------------------------------------------------------------

MARKET_OUTPUT_COLUMNS = {
    "game_id",
    "snapshot_spread",
    "snapshot_total",
    "snapshot_ml_prob_home_fair",
    "spread_movement",
    "total_movement",
}

# Columns that should NOT appear in the compressed market output
DROPPED_MARKET_COLUMNS = [
    "has_opening_lines",
    "has_snapshot_lines",
    "has_line_movement",
    "reverse_line_movement",
    "num_sportsbooks",
    "spread_range",
    "total_range",
    "opening_num_sportsbooks",
    "snapshot_num_sportsbooks",
    "ml_home_movement",
    "ml_away_movement",
    "spread_moved_toward_home",
    "spread_moved_toward_away",
    "spread_moved",
    "total_moved_up",
    "total_moved_down",
    "total_moved",
    "significant_line_movement",
    "ml_vig_change",
    "spread_vig_change",
    "total_vig_change",
    "ml_vig_increased",
    "spread_vig_increased",
    "total_vig_increased",
    "ml_prob_movement",
    "spread_prob_movement",
    "total_prob_movement",
]


def _make_market_games_df():
    """Create synthetic games DataFrame for market tests."""
    from zoneinfo import ZoneInfo

    et_tz = ZoneInfo("America/New_York")
    return pd.DataFrame(
        [
            {
                "game_id": "MKT_GAME_1",
                "season": 2024,
                "week": 1,
                "home_team": "KC",
                "away_team": "DET",
                "kickoff_et": datetime(2024, 9, 8, 13, 0, tzinfo=et_tz),
            },
            {
                "game_id": "MKT_GAME_2",
                "season": 2024,
                "week": 1,
                "home_team": "BUF",
                "away_team": "MIA",
                "kickoff_et": datetime(2024, 9, 8, 16, 25, tzinfo=et_tz),
            },
        ]
    )


def _make_odds_df():
    """Create synthetic odds data with opening and snapshot values."""
    return pd.DataFrame(
        [
            # MKT_GAME_1 opening (Monday before)
            {
                "game_id": "MKT_GAME_1",
                "sportsbook": "pinnacle",
                "snapshot_ts": datetime(2024, 9, 2, 10, 0),
                "ml_home": -200,
                "ml_away": 170,
                "spread": -4.5,
                "spread_ju_home": -110,
                "spread_ju_away": -110,
                "total": 47.5,
                "total_over_ju": -110,
                "total_under_ju": -110,
            },
            # MKT_GAME_1 snapshot (Thursday)
            {
                "game_id": "MKT_GAME_1",
                "sportsbook": "pinnacle",
                "snapshot_ts": datetime(2024, 9, 5, 12, 0),
                "ml_home": -220,
                "ml_away": 185,
                "spread": -5.5,
                "spread_ju_home": -110,
                "spread_ju_away": -110,
                "total": 48.0,
                "total_over_ju": -110,
                "total_under_ju": -110,
            },
            # MKT_GAME_2 opening (Tuesday)
            {
                "game_id": "MKT_GAME_2",
                "sportsbook": "pinnacle",
                "snapshot_ts": datetime(2024, 9, 3, 10, 0),
                "ml_home": -150,
                "ml_away": 130,
                "spread": -3.0,
                "spread_ju_home": -110,
                "spread_ju_away": -110,
                "total": 44.0,
                "total_over_ju": -110,
                "total_under_ju": -110,
            },
            # MKT_GAME_2 snapshot (Friday before cutoff)
            {
                "game_id": "MKT_GAME_2",
                "sportsbook": "pinnacle",
                "snapshot_ts": datetime(2024, 9, 6, 17, 0),
                "ml_home": -160,
                "ml_away": 140,
                "spread": -3.5,
                "spread_ju_home": -110,
                "spread_ju_away": -110,
                "total": 44.5,
                "total_over_ju": -110,
                "total_under_ju": -110,
            },
        ]
    )


class TestMarketCompression:
    """Tests for market feature compression to 5 features."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        """Set up calculator and build compressed features."""
        self.calc = MarketAnchorFeaturesCalculator()
        games_df = _make_market_games_df()
        odds_df = _make_odds_df()
        as_of = datetime(2024, 9, 6, 18, 0)  # Friday 6 PM ET

        with patch(
            "features.market_anchors.load_dataframe",
            return_value=odds_df,
        ):
            self.result = self.calc.build_features(games_df, as_of)

    def test_output_has_exactly_five_feature_columns(self):
        """build_features returns only game_id + 5 compressed market features."""
        assert set(self.result.columns) == MARKET_OUTPUT_COLUMNS

    def test_ml_prob_home_fair_in_probability_range(self):
        """snapshot_ml_prob_home_fair is a devigged probability in [0, 1]."""
        probs = self.result["snapshot_ml_prob_home_fair"]
        assert (probs >= 0.0).all()
        assert (probs <= 1.0).all()

    def test_spread_movement_is_signed_difference(self):
        """spread_movement = snapshot_spread - opening_spread."""
        game1 = self.result[self.result["game_id"] == "MKT_GAME_1"].iloc[0]
        # Opening=-4.5, Snapshot=-5.5 => movement = -5.5 - (-4.5) = -1.0
        assert game1["spread_movement"] == pytest.approx(-1.0, abs=0.01)

    def test_total_movement_is_signed_difference(self):
        """total_movement = snapshot_total - opening_total."""
        game1 = self.result[self.result["game_id"] == "MKT_GAME_1"].iloc[0]
        # Opening=47.5, Snapshot=48.0 => movement = 48.0 - 47.5 = 0.5
        assert game1["total_movement"] == pytest.approx(0.5, abs=0.01)

    def test_no_closing_columns_in_output(self):
        """No column names containing 'closing' in the output."""
        closing_cols = [col for col in self.result.columns if "closing" in col.lower()]
        assert closing_cols == [], f"Found closing columns: {closing_cols}"

    def test_dropped_market_columns_absent(self):
        """Dropped market columns do not appear in output."""
        output_cols = set(self.result.columns)
        for col in DROPPED_MARKET_COLUMNS:
            assert col not in output_cols, f"Dropped column '{col}' still in output"

    def test_anti_features_absent_from_market(self):
        """Anti-features (turnover, penalty, streak, rushing_yards) absent."""
        output_cols = " ".join(self.result.columns)
        for pattern in ANTI_FEATURE_PATTERNS:
            assert pattern not in output_cols, (
                f"Anti-feature pattern '{pattern}' found in market columns"
            )

    def test_build_features_accepts_as_of_datetime(self):
        """build_features method accepts as_of_datetime parameter."""
        import inspect

        sig = inspect.signature(self.calc.build_features)
        param_names = list(sig.parameters.keys())
        assert "as_of_datetime" in param_names, (
            f"Expected 'as_of_datetime' in build_features params, got {param_names}"
        )

    def test_as_of_datetime_filters_odds(self):
        """build_features only uses odds data with snapshot_ts <= as_of_datetime."""
        games_df = _make_market_games_df()
        odds_df = _make_odds_df()
        # Use early cutoff -- only the opening lines should be available
        early_cutoff = datetime(2024, 9, 3, 11, 0)

        with patch(
            "features.market_anchors.load_dataframe",
            return_value=odds_df,
        ):
            result = self.calc.build_features(games_df, early_cutoff)

        # MKT_GAME_2 has opening at Sep 3 10:00 (before cutoff), so snapshot = opening
        # Therefore spread_movement should be 0.0 (no movement possible)
        game2 = result[result["game_id"] == "MKT_GAME_2"].iloc[0]
        assert game2["spread_movement"] == pytest.approx(0.0, abs=0.01)


# ---------------------------------------------------------------------------
# Contextual builder Protocol conformance tests
# ---------------------------------------------------------------------------


class TestContextualProtocol:
    """Tests for ContextualFeaturesCalculator Protocol conformance."""

    def test_has_build_features_with_as_of_datetime(self):
        """ContextualFeaturesCalculator has build_features with as_of_datetime."""
        import inspect

        calc = ContextualFeaturesCalculator()
        sig = inspect.signature(calc.build_features)
        param_names = list(sig.parameters.keys())
        assert "as_of_datetime" in param_names, (
            f"Expected 'as_of_datetime' in build_features params, got {param_names}"
        )

    def test_has_get_features_for_game_with_as_of_datetime(self):
        """ContextualFeaturesCalculator has get_features_for_game with as_of_datetime."""
        import inspect

        calc = ContextualFeaturesCalculator()
        sig = inspect.signature(calc.get_features_for_game)
        param_names = list(sig.parameters.keys())
        assert "as_of_datetime" in param_names, (
            f"Expected 'as_of_datetime' in get_features_for_game params, got {param_names}"
        )
