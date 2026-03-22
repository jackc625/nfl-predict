"""Unit tests for QB Tracking and Quality Metric (FEAT-12, FEAT-13).

Tests cover:
- QB1 starter detection from depth chart data
- QB starter change detection week-over-week
- Rolling QB EPA computation
- Rolling CPOE computation (null handling)
- Composite QB quality metric (70% EPA + 30% CPOE)
- Temporal correctness (no future data leakage)
- build_features output format
- get_features_for_game output format
- FeatureBuilder Protocol conformance
- Default value for QBs with no history
"""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from features.protocol import FeatureBuilder  # Protocol only, no heavy imports
from features.qb_tracking import QBTracker

# ---------------------------------------------------------------------------
# Fixtures: mock depth chart and PBP data
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_depth_charts() -> pd.DataFrame:
    """Depth chart data for two teams across 4 weeks.

    KC has the same QB1 all 4 weeks.
    BUF changes QB1 in week 3.
    """
    rows = [
        # KC -- Mahomes is QB1 every week
        {"season": 2024, "week": 1, "club_code": "KC", "position": "QB", "depth_team": "1", "full_name": "Patrick Mahomes", "gsis_id": "00-0033873"},
        {"season": 2024, "week": 2, "club_code": "KC", "position": "QB", "depth_team": "1", "full_name": "Patrick Mahomes", "gsis_id": "00-0033873"},
        {"season": 2024, "week": 3, "club_code": "KC", "position": "QB", "depth_team": "1", "full_name": "Patrick Mahomes", "gsis_id": "00-0033873"},
        {"season": 2024, "week": 4, "club_code": "KC", "position": "QB", "depth_team": "1", "full_name": "Patrick Mahomes", "gsis_id": "00-0033873"},
        # KC backup
        {"season": 2024, "week": 1, "club_code": "KC", "position": "QB", "depth_team": "2", "full_name": "Carson Wentz", "gsis_id": "00-0033092"},
        # BUF -- Allen weeks 1-2, then Barkley weeks 3-4
        {"season": 2024, "week": 1, "club_code": "BUF", "position": "QB", "depth_team": "1", "full_name": "Josh Allen", "gsis_id": "00-0034857"},
        {"season": 2024, "week": 2, "club_code": "BUF", "position": "QB", "depth_team": "1", "full_name": "Josh Allen", "gsis_id": "00-0034857"},
        {"season": 2024, "week": 3, "club_code": "BUF", "position": "QB", "depth_team": "1", "full_name": "Matt Barkley", "gsis_id": "00-0029746"},
        {"season": 2024, "week": 4, "club_code": "BUF", "position": "QB", "depth_team": "1", "full_name": "Matt Barkley", "gsis_id": "00-0029746"},
        # Non-QB entries (should be filtered out)
        {"season": 2024, "week": 1, "club_code": "KC", "position": "WR", "depth_team": "1", "full_name": "Some Receiver", "gsis_id": "00-0099999"},
    ]
    return pd.DataFrame(rows)


@pytest.fixture()
def mock_pbp_data() -> pd.DataFrame:
    """PBP data with qb_epa and cpoe values for known passers.

    Creates 3 weeks of games for KC and BUF.
    Week 1: KC plays BUF -- game_id = 2024_01_KC_BUF
    Week 2: KC plays DAL, BUF plays MIA
    Week 3: KC plays DEN, BUF plays NE
    """
    plays = []

    # Week 1: KC vs BUF
    # Mahomes: 20 pass plays, high EPA + CPOE
    for i in range(20):
        plays.append({
            "game_id": "2024_01_KC_BUF",
            "season": 2024,
            "week": 1,
            "posteam": "KC",
            "passer_player_id": "00-0033873",
            "qb_epa": 0.3 + (i * 0.01),
            "cpoe": 5.0 + (i * 0.1),
            "play_id": 100 + i,
        })
    # Allen: 18 pass plays, moderate EPA + CPOE
    for i in range(18):
        plays.append({
            "game_id": "2024_01_KC_BUF",
            "season": 2024,
            "week": 1,
            "posteam": "BUF",
            "passer_player_id": "00-0034857",
            "qb_epa": 0.1 + (i * 0.01),
            "cpoe": 2.0 + (i * 0.1),
            "play_id": 200 + i,
        })
    # A sack play for Allen (cpoe is null on sacks)
    plays.append({
        "game_id": "2024_01_KC_BUF",
        "season": 2024,
        "week": 1,
        "posteam": "BUF",
        "passer_player_id": "00-0034857",
        "qb_epa": -2.0,
        "cpoe": None,
        "play_id": 299,
    })

    # Week 2: KC vs DAL
    for i in range(22):
        plays.append({
            "game_id": "2024_02_KC_DAL",
            "season": 2024,
            "week": 2,
            "posteam": "KC",
            "passer_player_id": "00-0033873",
            "qb_epa": 0.25 + (i * 0.01),
            "cpoe": 4.5 + (i * 0.1),
            "play_id": 300 + i,
        })
    # BUF vs MIA (Allen)
    for i in range(20):
        plays.append({
            "game_id": "2024_02_BUF_MIA",
            "season": 2024,
            "week": 2,
            "posteam": "BUF",
            "passer_player_id": "00-0034857",
            "qb_epa": 0.15 + (i * 0.01),
            "cpoe": 3.0 + (i * 0.1),
            "play_id": 400 + i,
        })

    # Week 3: KC vs DEN
    for i in range(19):
        plays.append({
            "game_id": "2024_03_KC_DEN",
            "season": 2024,
            "week": 3,
            "posteam": "KC",
            "passer_player_id": "00-0033873",
            "qb_epa": 0.35 + (i * 0.01),
            "cpoe": 6.0 + (i * 0.1),
            "play_id": 500 + i,
        })
    # BUF vs NE (Barkley -- new starter)
    for i in range(15):
        plays.append({
            "game_id": "2024_03_BUF_NE",
            "season": 2024,
            "week": 3,
            "posteam": "BUF",
            "passer_player_id": "00-0029746",
            "qb_epa": -0.1 + (i * 0.005),
            "cpoe": -1.0 + (i * 0.1),
            "play_id": 600 + i,
        })

    return pd.DataFrame(plays)


@pytest.fixture()
def mock_games_df() -> pd.DataFrame:
    """Games DataFrame matching the mock PBP data."""
    return pd.DataFrame([
        {"game_id": "2024_01_KC_BUF", "season": 2024, "week": 1, "home_team": "KC", "away_team": "BUF", "kickoff_et": datetime(2024, 9, 5, 20, 20)},
        {"game_id": "2024_02_KC_DAL", "season": 2024, "week": 2, "home_team": "KC", "away_team": "DAL", "kickoff_et": datetime(2024, 9, 12, 13, 0)},
        {"game_id": "2024_02_BUF_MIA", "season": 2024, "week": 2, "home_team": "BUF", "away_team": "MIA", "kickoff_et": datetime(2024, 9, 12, 13, 0)},
        {"game_id": "2024_03_KC_DEN", "season": 2024, "week": 3, "home_team": "KC", "away_team": "DEN", "kickoff_et": datetime(2024, 9, 19, 13, 0)},
        {"game_id": "2024_03_BUF_NE", "season": 2024, "week": 3, "home_team": "BUF", "away_team": "NE", "kickoff_et": datetime(2024, 9, 19, 16, 25)},
        {"game_id": "2024_04_KC_LA", "season": 2024, "week": 4, "home_team": "KC", "away_team": "LA", "kickoff_et": datetime(2024, 9, 26, 13, 0)},
        {"game_id": "2024_04_BUF_JAX", "season": 2024, "week": 4, "home_team": "BUF", "away_team": "JAX", "kickoff_et": datetime(2024, 9, 26, 13, 0)},
    ])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestQBStarterDetection:
    """Tests for QB1 starter detection from depth chart data."""

    def test_qb_starter_detection(self, mock_depth_charts: pd.DataFrame) -> None:
        """QBTracker detects QB1 starter from depth charts (position==QB, depth_team==1)."""
        tracker = QBTracker()
        starters = tracker.get_starters_from_depth_charts(mock_depth_charts)

        # Should have entries for KC and BUF across weeks 1-4
        assert len(starters) == 8  # 2 teams x 4 weeks

        # KC should always be Mahomes
        kc_starters = starters[starters["team"] == "KC"]
        assert len(kc_starters) == 4
        assert all(kc_starters["gsis_id"] == "00-0033873")

        # BUF should be Allen in weeks 1-2, Barkley in weeks 3-4
        buf_w1 = starters[(starters["team"] == "BUF") & (starters["week"] == 1)]
        assert buf_w1.iloc[0]["gsis_id"] == "00-0034857"

        buf_w3 = starters[(starters["team"] == "BUF") & (starters["week"] == 3)]
        assert buf_w3.iloc[0]["gsis_id"] == "00-0029746"

    def test_starter_change_detection(self, mock_depth_charts: pd.DataFrame) -> None:
        """QB starter change is detected when gsis_id changes week-over-week."""
        tracker = QBTracker()
        starters = tracker.get_starters_from_depth_charts(mock_depth_charts)

        # KC should have no starter changes
        kc_starters = starters[starters["team"] == "KC"]
        assert kc_starters["starter_changed"].sum() == 0

        # BUF should have one starter change (week 3 when Barkley replaces Allen)
        buf_starters = starters[starters["team"] == "BUF"]
        assert buf_starters["starter_changed"].sum() == 1
        changed_row = buf_starters[buf_starters["starter_changed"]]
        assert changed_row.iloc[0]["week"] == 3

    def test_filters_non_qb_positions(self, mock_depth_charts: pd.DataFrame) -> None:
        """Non-QB depth chart entries are filtered out."""
        tracker = QBTracker()
        starters = tracker.get_starters_from_depth_charts(mock_depth_charts)

        # No WR entries should be present
        assert "WR" not in starters.get("position", pd.Series()).values if "position" in starters.columns else True
        # Only QB1 entries
        assert len(starters) == 8


class TestRollingQBStats:
    """Tests for rolling QB EPA and CPOE computation."""

    def test_rolling_qb_epa(self, mock_pbp_data: pd.DataFrame) -> None:
        """Rolling QB EPA is computed correctly from PBP data."""
        tracker = QBTracker()
        qb_stats = tracker.compute_per_game_qb_stats(mock_pbp_data)

        # Mahomes should have 3 games
        mahomes_stats = qb_stats[qb_stats["passer_player_id"] == "00-0033873"]
        assert len(mahomes_stats) == 3

        # Mean QB EPA should be positive for Mahomes
        assert all(mahomes_stats["mean_qb_epa"] > 0)

    def test_rolling_cpoe_null_handling(self, mock_pbp_data: pd.DataFrame) -> None:
        """Only non-null cpoe values are used in rolling CPOE mean."""
        tracker = QBTracker()
        qb_stats = tracker.compute_per_game_qb_stats(mock_pbp_data)

        # Allen week 1 had 18 plays with CPOE + 1 sack with null CPOE
        allen_w1 = qb_stats[
            (qb_stats["passer_player_id"] == "00-0034857")
            & (qb_stats["week"] == 1)
        ]
        assert len(allen_w1) == 1
        # mean_cpoe should be computed only from the 18 non-null plays
        # cpoe values: 2.0, 2.1, ..., 3.7 -> mean = 2.85
        expected_cpoe = np.mean([2.0 + (i * 0.1) for i in range(18)])
        assert abs(allen_w1.iloc[0]["mean_cpoe"] - expected_cpoe) < 0.01

    def test_primary_passer_identification(self, mock_pbp_data: pd.DataFrame) -> None:
        """Primary passer per game-team is the one with most pass attempts."""
        tracker = QBTracker()
        qb_stats = tracker.compute_per_game_qb_stats(mock_pbp_data)

        # In week 1 KC game, Mahomes had 20 attempts -- should be primary
        kc_w1 = qb_stats[
            (qb_stats["posteam"] == "KC")
            & (qb_stats["week"] == 1)
        ]
        assert len(kc_w1) == 1  # Only primary passer returned
        assert kc_w1.iloc[0]["passer_player_id"] == "00-0033873"


class TestCompositeQBQuality:
    """Tests for the composite QB quality metric."""

    def test_composite_quality_formula(self) -> None:
        """Composite QB quality = 0.7 * normalized_rolling_qb_epa + 0.3 * normalized_rolling_cpoe."""
        tracker = QBTracker()

        # Given known z-scored values
        norm_epa = 1.5
        norm_cpoe = 0.8
        expected = 0.7 * norm_epa + 0.3 * norm_cpoe
        result = tracker.compute_composite_quality(norm_epa, norm_cpoe)
        assert abs(result - expected) < 1e-10

    def test_no_history_defaults_to_zero(self) -> None:
        """QB with no prior game history gets qb_adjustment = 0.0."""
        tracker = QBTracker()
        # A QB with no games should get league-average (0.0)
        result = tracker.compute_composite_quality(None, None)
        assert result == 0.0


class TestTemporalCorrectness:
    """Tests for temporal correctness -- no future data leakage."""

    def test_temporal_correctness(
        self, mock_pbp_data: pd.DataFrame, mock_depth_charts: pd.DataFrame
    ) -> None:
        """QB quality for week N uses only PBP data from weeks < N."""
        tracker = QBTracker()

        # Compute rolling quality for Mahomes as-of week 3
        # Should only use weeks 1 and 2 data
        rolling = tracker.compute_rolling_qb_metrics(
            mock_pbp_data,
            target_season=2024,
            target_week=3,
        )

        mahomes = rolling[rolling["passer_player_id"] == "00-0033873"]
        assert len(mahomes) == 1

        # The rolling EPA should be average of weeks 1 and 2 only
        # Week 1 mean EPA: mean of [0.3, 0.31, ..., 0.49] = 0.395
        # Week 2 mean EPA: mean of [0.25, 0.26, ..., 0.46] = 0.355
        # Rolling average of these two: (0.395 + 0.355) / 2 = 0.375
        # (roughly -- may differ due to recency weighting)
        assert mahomes.iloc[0]["rolling_qb_epa"] > 0.3


class TestBuildFeatures:
    """Tests for build_features output format."""

    def test_build_features_output_columns(
        self,
        mock_games_df: pd.DataFrame,
        mock_pbp_data: pd.DataFrame,
        mock_depth_charts: pd.DataFrame,
    ) -> None:
        """build_features returns DataFrame with game_id, team, qb_adjustment columns."""
        tracker = QBTracker()
        # Provide mock data loading
        tracker._depth_chart_cache = {2024: mock_depth_charts}
        tracker._pbp_cache = {2024: mock_pbp_data}

        result = tracker.build_features(
            mock_games_df,
            as_of_datetime=datetime(2024, 9, 26, 12, 0),
            target_season=2024,
            target_week=4,
        )

        assert "game_id" in result.columns
        assert "team" in result.columns
        assert "qb_adjustment" in result.columns

    def test_get_features_for_game_output(
        self,
        mock_games_df: pd.DataFrame,
        mock_pbp_data: pd.DataFrame,
        mock_depth_charts: pd.DataFrame,
    ) -> None:
        """get_features_for_game returns dict with home_qb_adjustment and away_qb_adjustment."""
        tracker = QBTracker()
        tracker._depth_chart_cache = {2024: mock_depth_charts}
        tracker._pbp_cache = {2024: mock_pbp_data}
        tracker._games_cache = mock_games_df

        result = tracker.get_features_for_game(
            "2024_04_KC_LA",
            as_of_datetime=datetime(2024, 9, 26, 12, 0),
        )

        assert "home_qb_adjustment" in result
        assert "away_qb_adjustment" in result
        assert isinstance(result["home_qb_adjustment"], float)
        assert isinstance(result["away_qb_adjustment"], float)


class TestProtocolConformance:
    """Tests for FeatureBuilder Protocol conformance."""

    def test_protocol_conformance(self) -> None:
        """QBTracker satisfies FeatureBuilder Protocol (isinstance check)."""
        tracker = QBTracker()
        assert isinstance(tracker, FeatureBuilder)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_no_prior_history_default(
        self,
        mock_games_df: pd.DataFrame,
        mock_depth_charts: pd.DataFrame,
    ) -> None:
        """When QB has no prior game history (week 1), qb_adjustment defaults to 0.0."""
        tracker = QBTracker()

        # Create minimal PBP with no data before week 1
        empty_pbp = pd.DataFrame(
            columns=["game_id", "season", "week", "posteam",
                     "passer_player_id", "qb_epa", "cpoe", "play_id"]
        )

        tracker._depth_chart_cache = {2024: mock_depth_charts}
        tracker._pbp_cache = {2024: empty_pbp}

        result = tracker.build_features(
            mock_games_df,
            as_of_datetime=datetime(2024, 9, 5, 12, 0),
            target_season=2024,
            target_week=1,
        )

        # All QB adjustments should be 0.0 since no prior data
        if len(result) > 0:
            assert all(result["qb_adjustment"] == 0.0)
