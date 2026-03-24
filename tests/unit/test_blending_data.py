"""Unit tests for pre-2018 odds data ingestion from nflverse.

Tests use mocked nflreadpy.load_schedules to avoid network dependency.
The mock returns a polars DataFrame simulating nflverse schedule data.

Tests cover:
- Column mapping (spread_line -> spread, total_line -> total, etc.)
- Regular season filter (game_type == "REG")
- NaN dropping for required odds columns
- Default seasons = 2010-2017
- Custom seasons parameter
- Team abbreviation normalization
- Return type is pandas DataFrame
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import polars as pl
import pytest


def _make_mock_schedule(
    seasons: list[int] | None = None,
    include_playoffs: bool = False,
    include_nans: bool = False,
) -> pl.DataFrame:
    """Build a mock polars DataFrame simulating nflreadpy.load_schedules output.

    Mimics the column names and structure that nflverse returns.
    """
    if seasons is None:
        seasons = [2015, 2016]

    rows = []
    for season in seasons:
        for week in range(1, 4):  # 3 weeks per season
            rows.append(
                {
                    "game_id": f"{season}_0{week}_KC_DEN",
                    "season": season,
                    "week": week,
                    "game_type": "REG",
                    "home_team": "KC",
                    "away_team": "DEN",
                    "home_score": 24 + week,
                    "away_score": 17 + week,
                    "spread_line": -3.0 - week,
                    "total_line": 44.0 + week,
                    "home_moneyline": -150 - week * 10,
                    "away_moneyline": 130 + week * 10,
                    "home_spread_odds": -110,
                    "away_spread_odds": -110,
                    "over_odds": -110,
                    "under_odds": -110,
                }
            )

    if include_playoffs:
        for season in seasons:
            rows.append(
                {
                    "game_id": f"{season}_18_KC_BUF",
                    "season": season,
                    "week": 18,
                    "game_type": "WC",
                    "home_team": "KC",
                    "away_team": "BUF",
                    "home_score": 42,
                    "away_score": 36,
                    "spread_line": -1.0,
                    "total_line": 54.0,
                    "home_moneyline": -120,
                    "away_moneyline": 100,
                    "home_spread_odds": -110,
                    "away_spread_odds": -110,
                    "over_odds": -110,
                    "under_odds": -110,
                }
            )

    if include_nans:
        # Add a row with NaN in spread (missing odds data)
        rows.append(
            {
                "game_id": f"{seasons[0]}_04_SF_SEA",
                "season": seasons[0],
                "week": 4,
                "game_type": "REG",
                "home_team": "SF",
                "away_team": "SEA",
                "home_score": 20,
                "away_score": 17,
                "spread_line": None,
                "total_line": 42.0,
                "home_moneyline": -130,
                "away_moneyline": 110,
                "home_spread_odds": -110,
                "away_spread_odds": -110,
                "over_odds": -110,
                "under_odds": -110,
            }
        )

    return pl.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLoadTuningPeriodData:
    """Tests for load_tuning_period_data function."""

    @patch("models.blending_data.nflreadpy")
    def test_returns_expected_columns(self, mock_nflreadpy: object) -> None:
        """load_tuning_period_data returns DataFrame with the expected columns."""
        mock_nflreadpy.load_schedules.return_value = _make_mock_schedule([2015])
        from models.blending_data import load_tuning_period_data

        result = load_tuning_period_data(seasons=[2015])

        expected_cols = {
            "game_id",
            "season",
            "week",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "spread",
            "total",
            "ml_home",
            "ml_away",
        }
        assert expected_cols.issubset(set(result.columns))

    @patch("models.blending_data.nflreadpy")
    def test_custom_seasons(self, mock_nflreadpy: object) -> None:
        """load_tuning_period_data with custom seasons loads only those seasons."""
        mock_nflreadpy.load_schedules.return_value = _make_mock_schedule([2015, 2016])
        from models.blending_data import load_tuning_period_data

        result = load_tuning_period_data(seasons=[2015, 2016])
        seasons_in_result = result["season"].unique()
        assert set(seasons_in_result) == {2015, 2016}

    @patch("models.blending_data.nflreadpy")
    def test_regular_season_only(self, mock_nflreadpy: object) -> None:
        """All returned rows have game_type == 'REG' (regular season only)."""
        mock_nflreadpy.load_schedules.return_value = _make_mock_schedule(
            [2015], include_playoffs=True
        )
        from models.blending_data import load_tuning_period_data

        result = load_tuning_period_data(seasons=[2015])
        # No playoff games should appear
        assert len(result) == 3  # Only 3 REG games from the mock

    @patch("models.blending_data.nflreadpy")
    def test_nan_dropping(self, mock_nflreadpy: object) -> None:
        """Rows with NaN in spread/total/ml_home/ml_away are dropped."""
        mock_nflreadpy.load_schedules.return_value = _make_mock_schedule(
            [2015], include_nans=True
        )
        from models.blending_data import load_tuning_period_data

        result = load_tuning_period_data(seasons=[2015])
        # Should have 3 REG rows (not the NaN one)
        assert len(result) == 3
        assert result["spread"].isna().sum() == 0
        assert result["total"].isna().sum() == 0
        assert result["ml_home"].isna().sum() == 0
        assert result["ml_away"].isna().sum() == 0

    @patch("models.blending_data.nflreadpy")
    def test_column_mapping_spread(self, mock_nflreadpy: object) -> None:
        """spread column maps from nflverse spread_line."""
        mock_nflreadpy.load_schedules.return_value = _make_mock_schedule([2015])
        from models.blending_data import load_tuning_period_data

        result = load_tuning_period_data(seasons=[2015])
        # First row: spread_line was -4.0 (week 1: -3.0 - 1)
        assert result.iloc[0]["spread"] == pytest.approx(-4.0)

    @patch("models.blending_data.nflreadpy")
    def test_column_mapping_total(self, mock_nflreadpy: object) -> None:
        """total column maps from nflverse total_line."""
        mock_nflreadpy.load_schedules.return_value = _make_mock_schedule([2015])
        from models.blending_data import load_tuning_period_data

        result = load_tuning_period_data(seasons=[2015])
        # First row: total_line was 45.0 (week 1: 44.0 + 1)
        assert result.iloc[0]["total"] == pytest.approx(45.0)

    @patch("models.blending_data.nflreadpy")
    def test_column_mapping_moneylines(self, mock_nflreadpy: object) -> None:
        """ml_home/ml_away columns map from nflverse home_moneyline/away_moneyline."""
        mock_nflreadpy.load_schedules.return_value = _make_mock_schedule([2015])
        from models.blending_data import load_tuning_period_data

        result = load_tuning_period_data(seasons=[2015])
        # First row: home_moneyline was -160 (week 1: -150 - 10)
        assert result.iloc[0]["ml_home"] == pytest.approx(-160)
        # First row: away_moneyline was 140 (week 1: 130 + 10)
        assert result.iloc[0]["ml_away"] == pytest.approx(140)

    @patch("models.blending_data.nflreadpy")
    def test_team_abbreviation_normalization(self, mock_nflreadpy: object) -> None:
        """Team abbreviations are normalized through normalize_team_abbreviation."""
        # Use a schedule with teams that have alias abbreviations
        rows = [
            {
                "game_id": "2015_01_OAK_STL",
                "season": 2015,
                "week": 1,
                "game_type": "REG",
                "home_team": "OAK",  # Should normalize to LV
                "away_team": "STL",  # Should normalize to LA
                "home_score": 20,
                "away_score": 17,
                "spread_line": -3.0,
                "total_line": 44.0,
                "home_moneyline": -150,
                "away_moneyline": 130,
                "home_spread_odds": -110,
                "away_spread_odds": -110,
                "over_odds": -110,
                "under_odds": -110,
            }
        ]
        mock_nflreadpy.load_schedules.return_value = pl.DataFrame(rows)
        from models.blending_data import load_tuning_period_data

        result = load_tuning_period_data(seasons=[2015])
        # OAK -> LV, STL -> LA
        assert result.iloc[0]["home_team"] == "LV"
        assert result.iloc[0]["away_team"] == "LA"

    @patch("models.blending_data.nflreadpy")
    def test_returns_pandas_dataframe(self, mock_nflreadpy: object) -> None:
        """load_tuning_period_data returns a pandas DataFrame (not polars)."""
        mock_nflreadpy.load_schedules.return_value = _make_mock_schedule([2015])
        from models.blending_data import load_tuning_period_data

        result = load_tuning_period_data(seasons=[2015])
        assert isinstance(result, pd.DataFrame)

    def test_default_seasons_constant(self) -> None:
        """TUNING_SEASONS defaults to 2010-2017 (8 seasons)."""
        from models.blending_data import TUNING_SEASONS

        assert TUNING_SEASONS == list(range(2010, 2018))
        assert len(TUNING_SEASONS) == 8


class TestGetTuningPeriodGames:
    """Tests for get_tuning_period_games function."""

    @patch("models.blending_data.nflreadpy")
    def test_returns_game_info_columns_only(self, mock_nflreadpy: object) -> None:
        """get_tuning_period_games returns only game info columns."""
        mock_nflreadpy.load_schedules.return_value = _make_mock_schedule([2015])
        from models.blending_data import get_tuning_period_games

        result = get_tuning_period_games(seasons=[2015])
        expected_cols = {"game_id", "season", "week", "home_team", "away_team", "home_score", "away_score"}
        assert set(result.columns) == expected_cols

    @patch("models.blending_data.nflreadpy")
    def test_no_odds_columns(self, mock_nflreadpy: object) -> None:
        """get_tuning_period_games does not include odds columns."""
        mock_nflreadpy.load_schedules.return_value = _make_mock_schedule([2015])
        from models.blending_data import get_tuning_period_games

        result = get_tuning_period_games(seasons=[2015])
        for col in ["spread", "total", "ml_home", "ml_away"]:
            assert col not in result.columns
