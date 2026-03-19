"""Unit tests for Pydantic v2 schemas and hard-fail quality gates.

Tests data/schemas.py (Pydantic v2 migration) and data/quality_gates.py
(Bronze-to-Silver and Silver-to-Gold validation boundaries).
"""

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest
from pydantic import ValidationError

from data.quality_gates import validate_bronze_to_silver, validate_silver_to_gold
from data.schemas import GameSchema, OddsSchema, VenueRoof, WeatherSchema
from utils.exceptions import DataValidationError

ET = ZoneInfo("America/New_York")


# --- GameSchema validation tests ---


class TestGameSchemaValidation:
    """Tests for GameSchema Pydantic v2 model_validator behavior."""

    def _make_game(self, **overrides):
        """Helper to create a valid game dict with sensible defaults."""
        defaults = {
            "game_id": "2024_W06_KC@BUF",
            "season": 2024,
            "week": 6,
            "kickoff_et": datetime(2024, 10, 13, 13, 0, tzinfo=ET),
            "home_team": "BUF",
            "away_team": "KC",
            "venue": "Highmark Stadium",
            "venue_roof": VenueRoof.OUTDOOR,
            "home_score": None,
            "away_score": None,
            "game_type": "REG",
        }
        defaults.update(overrides)
        return defaults

    def test_completed_game_with_both_scores(self):
        """Completed game with both scores should validate."""
        game = GameSchema(**self._make_game(home_score=21, away_score=14))
        assert game.home_score == 21
        assert game.away_score == 14

    def test_future_game_with_no_scores(self):
        """Future game with both scores None should validate."""
        game = GameSchema(**self._make_game(home_score=None, away_score=None))
        assert game.home_score is None
        assert game.away_score is None

    def test_partial_scores_home_only_raises(self):
        """Partial scores (home present, away None) should raise ValidationError."""
        with pytest.raises(ValidationError, match="partial scores"):
            GameSchema(**self._make_game(home_score=21, away_score=None))

    def test_partial_scores_away_only_raises(self):
        """Partial scores (away present, home None) should raise ValidationError."""
        with pytest.raises(ValidationError, match="partial scores"):
            GameSchema(**self._make_game(home_score=None, away_score=14))

    def test_zero_score_is_valid(self):
        """A score of 0 is valid (shutout), not None."""
        game = GameSchema(**self._make_game(home_score=0, away_score=0))
        assert game.home_score == 0
        assert game.away_score == 0


# --- OddsSchema validation tests ---


class TestOddsSchemaValidation:
    """Tests for OddsSchema validation."""

    def _make_odds(self, **overrides):
        defaults = {
            "game_id": "2024_W06_KC@BUF",
            "snapshot_ts": datetime(2024, 10, 11, 18, 0, tzinfo=ET),
            "sportsbook": "DraftKings",
            "ml_home": -150,
            "ml_away": 130,
            "spread": -3.0,
            "total": 47.5,
        }
        defaults.update(overrides)
        return defaults

    def test_valid_odds(self):
        odds = OddsSchema(**self._make_odds())
        assert odds.spread == -3.0
        assert odds.total == 47.5
        assert odds.sportsbook == "DraftKings"

    def test_odds_with_all_fields(self):
        odds = OddsSchema(
            **self._make_odds(
                spread_ju_home=-110,
                spread_ju_away=-110,
                total_over_ju=-105,
                total_under_ju=-115,
            )
        )
        assert odds.total_over_ju == -105


# --- WeatherSchema validation tests ---


class TestWeatherSchemaValidation:
    """Tests for WeatherSchema validation."""

    def _make_weather(self, **overrides):
        defaults = {
            "game_id": "2024_W06_KC@BUF",
            "forecast_time": datetime(2024, 10, 11, 12, 0, tzinfo=UTC),
            "game_time": datetime(2024, 10, 13, 17, 0, tzinfo=UTC),
            "is_outdoor": True,
            "temp_f": 55.0,
            "wind_mph": 8.0,
            "humidity_pct": 65.0,
        }
        defaults.update(overrides)
        return defaults

    def test_valid_outdoor_weather(self):
        weather = WeatherSchema(**self._make_weather())
        assert weather.is_outdoor is True
        assert weather.temp_f == 55.0

    def test_indoor_weather(self):
        weather = WeatherSchema(
            **self._make_weather(is_outdoor=False, temp_f=None, wind_mph=None)
        )
        assert weather.is_outdoor is False


# --- Bronze-to-Silver quality gate tests ---


class TestBronzeToSilverGate:
    """Tests for validate_bronze_to_silver hard-fail validation."""

    def _make_game_row(self, **overrides):
        defaults = {
            "game_id": "2024_W06_KC@BUF",
            "season": 2024,
            "week": 6,
            "kickoff_et": datetime(2024, 10, 13, 13, 0, tzinfo=ET),
            "home_team": "BUF",
            "away_team": "KC",
            "venue": "Highmark Stadium",
            "venue_roof": "outdoor",
            "home_score": 21,
            "away_score": 14,
            "game_type": "REG",
        }
        defaults.update(overrides)
        return defaults

    def test_valid_rows_pass(self):
        """All valid rows should return validated DataFrame."""
        df = pd.DataFrame([self._make_game_row()])
        result = validate_bronze_to_silver(df, GameSchema)
        assert len(result) == 1
        assert isinstance(result, pd.DataFrame)

    def test_missing_required_field_raises(self):
        """Missing game_id should raise DataValidationError."""
        row = self._make_game_row()
        del row["game_id"]
        df = pd.DataFrame([row])
        with pytest.raises(DataValidationError, match="validation failed"):
            validate_bronze_to_silver(df, GameSchema)

    def test_completed_game_null_score_raises(self):
        """Completed game with one null score should raise DataValidationError."""
        row = self._make_game_row(home_score=21, away_score=None)
        df = pd.DataFrame([row])
        with pytest.raises(DataValidationError, match="validation failed"):
            validate_bronze_to_silver(df, GameSchema)

    def test_multiple_errors_reported(self):
        """Multiple bad rows should all be reported."""
        rows = [
            self._make_game_row(
                game_id="2024_W06_KC@BUF", home_score=21, away_score=None
            ),
            self._make_game_row(
                game_id="2024_W07_KC@BUF", home_score=None, away_score=14
            ),
        ]
        df = pd.DataFrame(rows)
        with pytest.raises(DataValidationError, match="2 error"):
            validate_bronze_to_silver(df, GameSchema)

    def test_returns_validated_dataframe(self):
        """Returned DataFrame should have model_dump'd data."""
        row = self._make_game_row()
        df = pd.DataFrame([row])
        result = validate_bronze_to_silver(df, GameSchema)
        assert "game_id" in result.columns
        assert result.iloc[0]["game_id"] == "2024_W06_KC@BUF"


# --- Silver-to-Gold quality gate tests ---


class TestSilverToGoldGate:
    """Tests for validate_silver_to_gold hard-fail range validation."""

    def test_valid_features_pass(self):
        """All in-range values should pass."""
        df = pd.DataFrame(
            {
                "elo_rating": [1500.0, 1600.0],
                "epa_per_play": [0.1, -0.2],
                "spread": [-3.0, 7.0],
                "total": [45.0, 52.0],
            }
        )
        result = validate_silver_to_gold(df)
        assert len(result) == 2

    def test_elo_above_max_raises(self):
        """Elo rating above 2200 should raise DataValidationError."""
        df = pd.DataFrame({"elo_rating": [1500.0, 3000.0]})
        with pytest.raises(DataValidationError, match="elo_rating"):
            validate_silver_to_gold(df)

    def test_epa_below_min_raises(self):
        """EPA/play below -1.0 should raise DataValidationError."""
        df = pd.DataFrame({"epa_per_play": [0.1, -2.0]})
        with pytest.raises(DataValidationError, match="epa_per_play"):
            validate_silver_to_gold(df)

    def test_error_includes_column_and_count(self):
        """Error message should include column name and violation count."""
        df = pd.DataFrame({"elo_rating": [3000.0, 2500.0, 1500.0]})
        with pytest.raises(DataValidationError, match=r"elo_rating.*2 value"):
            validate_silver_to_gold(df)

    def test_nan_values_ignored(self):
        """NaN values should be skipped during range checks."""
        df = pd.DataFrame({"elo_rating": [1500.0, float("nan"), 1600.0]})
        result = validate_silver_to_gold(df)
        assert len(result) == 3

    def test_columns_not_in_ranges_ignored(self):
        """Columns not in FEATURE_RANGES should be silently skipped."""
        df = pd.DataFrame({"some_other_col": [999999.0]})
        result = validate_silver_to_gold(df)
        assert len(result) == 1

    def test_custom_ranges(self):
        """Custom range dict should override defaults."""
        df = pd.DataFrame({"custom_col": [5.0, 15.0]})
        custom_ranges = {"custom_col": (0.0, 10.0)}
        with pytest.raises(DataValidationError, match="custom_col"):
            validate_silver_to_gold(df, feature_ranges=custom_ranges)
