"""
Tests for extended contextual features: season-week position,
surface mismatch, and divisional game indicator.

These test the additions from FEAT-17, FEAT-18, FEAT-19.
"""

from datetime import datetime
from unittest.mock import patch

import pandas as pd
import pytest

from features.contextual import ContextualFeaturesCalculator, GRASS_SURFACES
from features.protocol import FeatureBuilder


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MOCK_VENUES = {
    "venues": [
        {
            "venue_id": "grass_stadium",
            "venue_name": "Grass Stadium",
            "city": "Miami Gardens",
            "state": "FL",
            "country": "USA",
            "latitude": 25.958,
            "longitude": -80.239,
            "elevation_ft": 7,
            "roof_type": "outdoor",
            "surface": "Bermuda Grass",
            "capacity": 65326,
            "climate_zone": "tropical",
            "timezone": "America/New_York",
            "home_teams": ["MIA"],
        },
        {
            "venue_id": "turf_stadium",
            "venue_name": "Turf Stadium",
            "city": "East Rutherford",
            "state": "NJ",
            "country": "USA",
            "latitude": 40.814,
            "longitude": -74.075,
            "elevation_ft": 7,
            "roof_type": "outdoor",
            "surface": "FieldTurf",
            "capacity": 82500,
            "climate_zone": "humid_continental",
            "timezone": "America/New_York",
            "home_teams": ["NYJ"],
        },
        {
            "venue_id": "matrix_stadium",
            "venue_name": "Matrix Stadium",
            "city": "Los Angeles",
            "state": "CA",
            "country": "USA",
            "latitude": 33.954,
            "longitude": -118.339,
            "elevation_ft": 107,
            "roof_type": "indoor",
            "surface": "Matrix Turf",
            "capacity": 70240,
            "climate_zone": "mediterranean",
            "timezone": "America/Los_Angeles",
            "home_teams": ["LA", "LAC"],
        },
        {
            "venue_id": "bluegrass_stadium",
            "venue_name": "Bluegrass Stadium",
            "city": "Green Bay",
            "state": "WI",
            "country": "USA",
            "latitude": 44.501,
            "longitude": -88.062,
            "elevation_ft": 640,
            "roof_type": "outdoor",
            "surface": "Kentucky Bluegrass",
            "capacity": 81441,
            "climate_zone": "humid_continental",
            "timezone": "America/Chicago",
            "home_teams": ["GB"],
        },
        {
            "venue_id": "kc_stadium",
            "venue_name": "KC Stadium",
            "city": "Kansas City",
            "state": "MO",
            "country": "USA",
            "latitude": 39.049,
            "longitude": -94.484,
            "elevation_ft": 909,
            "roof_type": "outdoor",
            "surface": "Bermuda Grass",
            "capacity": 76416,
            "climate_zone": "humid_continental",
            "timezone": "America/Chicago",
            "home_teams": ["KC"],
        },
    ]
}


def _make_games_df(
    game_id: str = "2024_01_KC_MIA",
    season: int = 2024,
    week: int = 1,
    home_team: str = "KC",
    away_team: str = "MIA",
    venue_name: str = "KC Stadium",
) -> pd.DataFrame:
    """Create a minimal games DataFrame for testing."""
    return pd.DataFrame(
        [
            {
                "game_id": game_id,
                "season": season,
                "week": week,
                "home_team": home_team,
                "away_team": away_team,
                "venue": venue_name,
                "kickoff_et": datetime(2024, 9, 5, 20, 0),
            }
        ]
    )


@pytest.fixture
def calc():
    """Create a ContextualFeaturesCalculator with mock venues data."""
    with patch.object(
        ContextualFeaturesCalculator, "_load_venues_data", return_value=MOCK_VENUES
    ):
        return ContextualFeaturesCalculator()


# ---------------------------------------------------------------------------
# FEAT-17: Season-Week Position Features
# ---------------------------------------------------------------------------


class TestSeasonProgress:
    """Tests for season_progress feature."""

    def test_season_progress_week_1(self, calc):
        """Week 1 should produce season_progress = 1/18 ~ 0.0556."""
        df = _make_games_df(week=1)
        result = calc.build_features(df, datetime(2024, 9, 5, 12, 0))
        assert "season_progress" in result.columns
        assert abs(result.iloc[0]["season_progress"] - (1.0 / 18.0)) < 1e-6

    def test_season_progress_week_9(self, calc):
        """Week 9 should produce season_progress = 9/18 = 0.5."""
        df = _make_games_df(week=9)
        result = calc.build_features(df, datetime(2024, 10, 31, 12, 0))
        assert abs(result.iloc[0]["season_progress"] - 0.5) < 1e-6

    def test_season_progress_week_18(self, calc):
        """Week 18 should produce season_progress = 18/18 = 1.0."""
        df = _make_games_df(week=18)
        result = calc.build_features(df, datetime(2025, 1, 5, 12, 0))
        assert abs(result.iloc[0]["season_progress"] - 1.0) < 1e-6

    def test_season_progress_is_float(self, calc):
        """season_progress should be a float, not int or bool."""
        df = _make_games_df(week=5)
        result = calc.build_features(df, datetime(2024, 10, 1, 12, 0))
        assert isinstance(result.iloc[0]["season_progress"], float)


class TestLateSeason:
    """Tests for late_season feature."""

    def test_late_season_week_13(self, calc):
        """Week 13 (< 14) should produce late_season = 0.0."""
        df = _make_games_df(week=13)
        result = calc.build_features(df, datetime(2024, 12, 1, 12, 0))
        assert result.iloc[0]["late_season"] == 0.0

    def test_late_season_week_14(self, calc):
        """Week 14 (>= 14) should produce late_season = 1.0."""
        df = _make_games_df(week=14)
        result = calc.build_features(df, datetime(2024, 12, 5, 12, 0))
        assert result.iloc[0]["late_season"] == 1.0

    def test_late_season_week_18(self, calc):
        """Week 18 should produce late_season = 1.0."""
        df = _make_games_df(week=18)
        result = calc.build_features(df, datetime(2025, 1, 5, 12, 0))
        assert result.iloc[0]["late_season"] == 1.0

    def test_late_season_is_float(self, calc):
        """late_season should be a float, not int or bool."""
        df = _make_games_df(week=14)
        result = calc.build_features(df, datetime(2024, 12, 5, 12, 0))
        assert isinstance(result.iloc[0]["late_season"], float)


# ---------------------------------------------------------------------------
# FEAT-18: Surface Type Mismatch
# ---------------------------------------------------------------------------


class TestSurfaceMismatch:
    """Tests for surface_mismatch feature."""

    def test_mismatch_grass_away_on_turf(self, calc):
        """Grass-home team playing on turf venue = mismatch 1.0.

        MIA (Bermuda Grass home) plays at NYJ (FieldTurf).
        """
        df = _make_games_df(
            home_team="NYJ", away_team="MIA", venue_name="Turf Stadium"
        )
        result = calc.build_features(df, datetime(2024, 9, 5, 12, 0))
        assert result.iloc[0]["surface_mismatch"] == 1.0

    def test_no_mismatch_turf_away_on_turf(self, calc):
        """Turf-home team playing on turf venue = no mismatch 0.0.

        NYJ (FieldTurf home) plays at LA (Matrix Turf) -- both synthetic.
        """
        df = _make_games_df(
            home_team="LA", away_team="NYJ", venue_name="Matrix Stadium"
        )
        result = calc.build_features(df, datetime(2024, 9, 5, 12, 0))
        assert result.iloc[0]["surface_mismatch"] == 0.0

    def test_no_mismatch_grass_away_on_grass(self, calc):
        """Grass-home team playing on grass venue = no mismatch 0.0.

        MIA (Bermuda Grass home) plays at GB (Kentucky Bluegrass).
        Both are grass.
        """
        df = _make_games_df(
            home_team="GB", away_team="MIA", venue_name="Bluegrass Stadium"
        )
        result = calc.build_features(df, datetime(2024, 9, 5, 12, 0))
        assert result.iloc[0]["surface_mismatch"] == 0.0

    def test_no_mismatch_for_home_team(self, calc):
        """Home team always plays on own surface -- surface_mismatch = 0.0.

        This tests that even when home surface differs from what we would
        compute for an away team, the home team perspective is always 0.0.
        (The feature is for the away team's perspective in the game row.)
        """
        # KC home (Bermuda Grass), venue is KC Stadium (Bermuda Grass)
        # Away team NYJ (FieldTurf) -- this IS a mismatch for away,
        # so surface_mismatch should reflect the away team's mismatch = 1.0
        df = _make_games_df(
            home_team="KC", away_team="NYJ", venue_name="KC Stadium"
        )
        result = calc.build_features(df, datetime(2024, 9, 5, 12, 0))
        # The feature captures away team mismatch
        assert result.iloc[0]["surface_mismatch"] == 1.0

    def test_no_mismatch_home_on_own_surface(self, calc):
        """When away team surface matches game venue, mismatch = 0.0."""
        # KC home (Bermuda Grass), away MIA (Bermuda Grass) at KC Stadium
        df = _make_games_df(
            home_team="KC", away_team="MIA", venue_name="KC Stadium"
        )
        result = calc.build_features(df, datetime(2024, 9, 5, 12, 0))
        assert result.iloc[0]["surface_mismatch"] == 0.0

    def test_surface_mismatch_is_float(self, calc):
        """surface_mismatch should be a float."""
        df = _make_games_df()
        result = calc.build_features(df, datetime(2024, 9, 5, 12, 0))
        assert isinstance(result.iloc[0]["surface_mismatch"], float)


class TestGrassSurfacesConstant:
    """Test that GRASS_SURFACES constant is correctly defined."""

    def test_contains_bermuda_grass(self):
        assert "Bermuda Grass" in GRASS_SURFACES

    def test_contains_kentucky_bluegrass(self):
        assert "Kentucky Bluegrass" in GRASS_SURFACES

    def test_no_synthetic_surfaces(self):
        assert "FieldTurf" not in GRASS_SURFACES
        assert "Matrix Turf" not in GRASS_SURFACES
        assert "NexTurf" not in GRASS_SURFACES


# ---------------------------------------------------------------------------
# FEAT-19: Divisional Game Indicator
# ---------------------------------------------------------------------------


class TestDivisionalIndicator:
    """Tests for is_divisional feature."""

    def test_divisional_game(self, calc):
        """KC vs LAC (both AFC West) should be is_divisional = 1.0."""
        df = _make_games_df(
            home_team="KC", away_team="LAC", venue_name="KC Stadium"
        )
        result = calc.build_features(df, datetime(2024, 9, 5, 12, 0))
        assert result.iloc[0]["is_divisional"] == 1.0

    def test_non_divisional_game(self, calc):
        """KC vs MIA (different divisions) should be is_divisional = 0.0."""
        df = _make_games_df(
            home_team="KC", away_team="MIA", venue_name="KC Stadium"
        )
        result = calc.build_features(df, datetime(2024, 9, 5, 12, 0))
        assert result.iloc[0]["is_divisional"] == 0.0

    def test_is_divisional_is_float(self, calc):
        """is_divisional should be a float."""
        df = _make_games_df()
        result = calc.build_features(df, datetime(2024, 9, 5, 12, 0))
        assert isinstance(result.iloc[0]["is_divisional"], float)


# ---------------------------------------------------------------------------
# Integration: build_features output
# ---------------------------------------------------------------------------


class TestBuildFeaturesOutput:
    """Test that build_features output contains all new columns."""

    def test_output_contains_all_new_columns(self, calc):
        """build_features output must include all 4 new feature columns."""
        df = _make_games_df()
        result = calc.build_features(df, datetime(2024, 9, 5, 12, 0))
        for col in ["season_progress", "late_season", "surface_mismatch", "is_divisional"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_existing_features_preserved(self, calc):
        """Existing contextual features should still be present."""
        df = _make_games_df()
        result = calc.build_features(df, datetime(2024, 9, 5, 12, 0))
        existing_cols = [
            "game_id", "season", "week", "home_team", "away_team",
            "is_home_game", "is_away_game",
        ]
        for col in existing_cols:
            assert col in result.columns, f"Existing column missing: {col}"

    def test_all_new_features_are_floats(self, calc):
        """All 4 new features must be float type."""
        df = _make_games_df(week=15)
        result = calc.build_features(df, datetime(2024, 12, 10, 12, 0))
        for col in ["season_progress", "late_season", "surface_mismatch", "is_divisional"]:
            val = result.iloc[0][col]
            assert isinstance(val, float), f"{col} is {type(val)}, expected float"


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    """Verify ContextualFeaturesCalculator still satisfies FeatureBuilder Protocol."""

    def test_is_feature_builder(self, calc):
        """Calculator must be an instance of FeatureBuilder Protocol."""
        assert isinstance(calc, FeatureBuilder)
