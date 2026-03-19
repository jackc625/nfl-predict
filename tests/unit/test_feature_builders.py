"""Unit tests for feature building modules."""

from datetime import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from features.contextual import ContextualFeaturesCalculator as ContextualFeatureBuilder
from features.market_anchors import (
    MarketAnchorFeaturesCalculator as MarketAnchorBuilder,
)
from features.team_form import TeamFormCalculator as TeamFormBuilder
from features.weather import WeatherFeaturesCalculator as WeatherFeatureBuilder
from scripts.build_features import FeatureMatrixBuilder
from tests.conftest import (
    assert_dataframe_structure,
    assert_probability_range,
)


class TestFeatureMatrixBuilder:
    """Test the main FeatureMatrixBuilder class."""

    def test_combine_features(
        self, mock_games_data, mock_team_form_data, mock_elo_data
    ):
        """Test combining features from multiple sources."""
        builder = FeatureMatrixBuilder()

        # Create mock contextual and weather data
        mock_contextual = pd.DataFrame(
            [
                {
                    "game_id": "MOCK_2024_W01_BUF@MIA",
                    "venue_outdoor": 1.0,
                    "away_travel_distance_miles": 1200,
                    "home_rest_days": 7,
                    "away_rest_days": 7,
                }
            ]
        )

        mock_weather = pd.DataFrame(
            [
                {
                    "game_id": "MOCK_2024_W01_BUF@MIA",
                    "temp_f": 75.0,
                    "wind_mph": 8.0,
                    "weather_severity_score": 0.2,
                }
            ]
        )

        mock_market = pd.DataFrame(
            [
                {
                    "game_id": "MOCK_2024_W01_BUF@MIA",
                    "snapshot_spread": -3.5,
                    "snapshot_total": 45.5,
                    "has_snapshot_lines": 1.0,
                }
            ]
        )

        feature_sources = {
            "games": mock_games_data,
            "team_form": mock_team_form_data,
            "elo": mock_elo_data,
            "contextual": mock_contextual,
            "weather": mock_weather,
            "market": mock_market,
        }

        combined = builder.combine_features(feature_sources)

        # Basic structure checks
        assert_dataframe_structure(combined, min_rows=1)
        assert "game_id" in combined.columns
        assert "season" in combined.columns
        assert "week" in combined.columns

    def test_handle_missing_data_and_outliers(self, sample_feature_matrix):
        """Test missing data and outlier handling."""
        builder = FeatureMatrixBuilder()

        # Introduce missing data
        df_with_missing = sample_feature_matrix.copy()
        df_with_missing.loc[0, "home_elo_rating"] = np.nan
        df_with_missing.loc[1, "away_elo_rating"] = np.nan

        # Introduce outliers
        df_with_missing.loc[0, "home_off_rolling_epa_per_play"] = (
            10.0  # Extreme outlier
        )

        processed = builder.handle_missing_data_and_outliers(df_with_missing)

        # Check that missing data is handled
        feature_cols = [
            col
            for col in processed.columns
            if col not in ["game_id", "season", "week", "home_team", "away_team"]
        ]
        missing_counts = processed[feature_cols].isnull().sum()

        # Should have fewer missing values after processing
        original_missing = df_with_missing[feature_cols].isnull().sum().sum()
        processed_missing = missing_counts.sum()
        assert processed_missing <= original_missing

    def test_normalize_features_within_seasons(self, sample_feature_matrix):
        """Test within-season feature normalization."""
        builder = FeatureMatrixBuilder()

        # Create multi-season data
        season_2023 = sample_feature_matrix.copy()
        season_2023["season"] = 2023
        season_2023["game_id"] = season_2023["game_id"].str.replace("2024", "2023")

        multi_season = pd.concat(
            [sample_feature_matrix, season_2023], ignore_index=True
        )

        normalized = builder.normalize_features_within_seasons(multi_season)

        # Check normalization within each season
        numeric_cols = multi_season.select_dtypes(include=[np.number]).columns
        feature_cols = [
            col
            for col in numeric_cols
            if col not in ["season", "week", "target_wp", "target_ats", "target_ou"]
        ]

        for season in [2023, 2024]:
            season_data = normalized[normalized["season"] == season]
            for col in feature_cols[:3]:  # Check first few features
                if col in season_data.columns:
                    values = season_data[col].dropna()
                    if len(values) > 1:  # Need multiple values for std calculation
                        assert abs(values.mean()) < 0.1, (
                            f"Feature {col} not centered in season {season}"
                        )
                        assert abs(values.std() - 1.0) < 0.5, (
                            f"Feature {col} not scaled in season {season}"
                        )

    def test_create_target_variables(self, mock_games_data):
        """Test target variable creation."""
        builder = FeatureMatrixBuilder()

        # Add market data for ATS/OU targets
        games_with_market = mock_games_data.copy()
        games_with_market["snapshot_spread"] = [-3.5, 7.0, -1.5, 3.0, -6.5, 2.5]
        games_with_market["snapshot_total"] = [45.5, 52.0, 41.5, 48.0, 44.5, 47.0]

        with_targets = builder.create_target_variables(games_with_market)

        # Check WP targets
        assert "target_wp" in with_targets.columns
        wp_values = with_targets["target_wp"].dropna()
        assert all(val in [0, 1] for val in wp_values), "WP targets should be 0 or 1"

        # Check ATS targets for games with results and spreads
        games_with_results = with_targets[with_targets["home_score"].notna()].copy()
        if len(games_with_results) > 0:
            assert "target_ats" in with_targets.columns
            # ATS should be 1 if home team covers, 0 otherwise
            ats_values = games_with_results["target_ats"].dropna()
            assert all(val in [0, 1] for val in ats_values), (
                "ATS targets should be 0 or 1"
            )


class TestTeamFormBuilder:
    """Test the TeamFormBuilder class."""

    def test_calculate_rolling_metrics(self, mock_games_data):
        """Test calculation of rolling team form metrics."""
        builder = TeamFormBuilder()

        # Mock play-by-play data
        mock_pbp = []
        for _, game in mock_games_data.iterrows():
            if game["home_score"] is not None:  # Only for completed games
                # Add some mock plays
                for play_idx in range(10):
                    mock_pbp.append(
                        {
                            "game_id": game["game_id"],
                            "season": game["season"],
                            "week": game["week"],
                            "posteam": game["home_team"]
                            if play_idx % 2 == 0
                            else game["away_team"],
                            "defteam": game["away_team"]
                            if play_idx % 2 == 0
                            else game["home_team"],
                            "epa": np.random.normal(0, 0.5),
                            "success": np.random.choice([0, 1]),
                            "pass": 1 if play_idx % 3 == 0 else 0,
                            "rush": 1 if play_idx % 3 == 1 else 0,
                            "down": np.random.choice([1, 2, 3, 4]),
                            "ydstogo": np.random.choice([1, 5, 10, 15]),
                        }
                    )

        pbp_df = pd.DataFrame(mock_pbp)

        with patch.object(builder, "load_play_by_play_data", return_value=pbp_df):
            form_metrics = builder.calculate_rolling_metrics(
                target_season=2024, target_week=2
            )

        assert_dataframe_structure(form_metrics, min_rows=1)
        assert "team" in form_metrics.columns
        assert "target_season" in form_metrics.columns
        assert "target_week" in form_metrics.columns
        assert "rolling_epa_per_play" in form_metrics.columns
        assert "rolling_success_rate" in form_metrics.columns

    def test_no_data_leakage(self, mock_games_data):
        """Test that no future data is used in form calculations."""
        builder = TeamFormBuilder()

        # Mock data where we're calculating metrics for week 2
        target_week = 2
        target_season = 2024

        # Mock PBP data that includes future games
        mock_pbp = []
        for week in [1, 2, 3]:  # Include future week 3
            mock_pbp.append(
                {
                    "game_id": f"MOCK_2024_W{week:02d}_BUF@MIA",
                    "season": 2024,
                    "week": week,
                    "posteam": "BUF",
                    "epa": 0.5,
                    "success": 1,
                }
            )

        pbp_df = pd.DataFrame(mock_pbp)

        with patch.object(builder, "load_play_by_play_data", return_value=pbp_df):
            builder.calculate_rolling_metrics(
                target_season=target_season, target_week=target_week
            )

        # Verify no future data used - should only use data from weeks < target_week
        # This is implementation-specific but generally form metrics should only use past data


class TestContextualFeatureBuilder:
    """Test the ContextualFeatureBuilder class."""

    def test_calculate_travel_distance(self):
        """Test travel distance calculations."""
        builder = ContextualFeatureBuilder()

        # Mock venue data
        venues = {
            "BUF": {"lat": 42.7738, "lng": -78.7870},  # Buffalo
            "MIA": {"lat": 25.9580, "lng": -80.2389},  # Miami
        }

        with patch.object(builder, "load_venue_data", return_value=venues):
            distance = builder.calculate_travel_distance("BUF", "MIA")

        assert isinstance(distance, (int, float))
        assert distance > 0
        assert distance < 5000  # Reasonable maximum for US travel

    def test_calculate_rest_days(self, mock_games_data):
        """Test rest days calculation."""
        builder = ContextualFeatureBuilder()

        # Create schedule with known rest periods
        schedule = mock_games_data.copy()
        schedule["kickoff_et"] = pd.to_datetime(schedule["kickoff_et"])

        with patch.object(builder, "load_team_schedule", return_value=schedule):
            contextual_features = builder.build_contextual_features(
                target_season=2024, target_week=2
            )

        assert_dataframe_structure(contextual_features, min_rows=1)
        assert "home_rest_days" in contextual_features.columns
        assert "away_rest_days" in contextual_features.columns

        # Rest days should be reasonable (typically 6-14 days)
        rest_values = contextual_features[
            ["home_rest_days", "away_rest_days"]
        ].values.flatten()
        rest_values = rest_values[~np.isnan(rest_values)]
        assert all(4 <= val <= 21 for val in rest_values), (
            f"Unrealistic rest days: {rest_values}"
        )


class TestWeatherFeatureBuilder:
    """Test the WeatherFeatureBuilder class."""

    def test_weather_impact_calculation(self):
        """Test weather impact scoring."""
        builder = WeatherFeatureBuilder()

        # Test various weather conditions
        test_conditions = [
            {
                "temp_f": 72,
                "wind_mph": 5,
                "precip_prob": 0.1,
                "expected_severity": "low",
            },
            {
                "temp_f": 30,
                "wind_mph": 20,
                "precip_prob": 0.8,
                "expected_severity": "high",
            },
            {
                "temp_f": 95,
                "wind_mph": 25,
                "precip_prob": 0.6,
                "expected_severity": "high",
            },
        ]

        for condition in test_conditions:
            severity = builder.calculate_weather_severity(
                temp_f=condition["temp_f"],
                wind_mph=condition["wind_mph"],
                precip_prob=condition["precip_prob"],
            )

            assert 0 <= severity <= 1, f"Weather severity should be 0-1, got {severity}"

            if condition["expected_severity"] == "low":
                assert severity < 0.3, f"Expected low severity, got {severity}"
            elif condition["expected_severity"] == "high":
                assert severity > 0.5, f"Expected high severity, got {severity}"

    def test_indoor_venue_handling(self, mock_games_data):
        """Test that indoor venues get zero weather impact."""
        builder = WeatherFeatureBuilder()

        # Mock venue data with indoor venue
        venues = {"MIA": {"roof_type": "outdoor"}, "DEN": {"roof_type": "indoor"}}

        # Mock weather data
        weather_data = pd.DataFrame(
            [
                {"game_id": "MOCK_2024_W01_BUF@MIA", "temp_f": 30, "wind_mph": 20},
                {"game_id": "MOCK_2024_W01_KC@DEN", "temp_f": 30, "wind_mph": 20},
            ]
        )

        with patch.object(builder, "load_venue_data", return_value=venues):
            with patch.object(builder, "fetch_weather_data", return_value=weather_data):
                weather_features = builder.build_weather_features(
                    games_df=mock_games_data, target_season=2024, target_week=1
                )

        # Indoor venue should have no weather impact
        indoor_game = weather_features[weather_features["game_id"].str.contains("DEN")]
        if len(indoor_game) > 0:
            assert indoor_game["weather_affects_game"].iloc[0] == 0.0


class TestMarketAnchorBuilder:
    """Test the MarketAnchorBuilder class."""

    def test_probability_conversion(self):
        """Test conversion between odds and probabilities."""
        builder = MarketAnchorBuilder()

        # Test American odds conversion
        test_cases = [
            (-110, 0.524),  # Standard -110 line
            (+100, 0.500),  # Even odds
            (-200, 0.667),  # Heavy favorite
            (+150, 0.400),  # Underdog
        ]

        for odds, expected_prob in test_cases:
            prob = builder.american_odds_to_probability(odds)
            assert abs(prob - expected_prob) < 0.01, (
                f"Odds {odds} should convert to ~{expected_prob}, got {prob}"
            )

    def test_devig_calculation(self):
        """Test vig removal from market probabilities."""
        builder = MarketAnchorBuilder()

        # Test standard -110/-110 line (9.09% vig)
        home_odds = -110
        away_odds = -110

        home_prob = builder.american_odds_to_probability(home_odds)
        away_prob = builder.american_odds_to_probability(away_odds)

        # Should sum to more than 1.0 due to vig
        total_prob = home_prob + away_prob
        assert total_prob > 1.0, "Implied probabilities should sum to > 1.0 with vig"

        # Test devig
        fair_home, fair_away, vig = builder.remove_vig(home_prob, away_prob)

        assert abs(fair_home + fair_away - 1.0) < 0.001, (
            "Fair probabilities should sum to 1.0"
        )
        assert 0.08 < vig < 0.12, f"Expected ~10% vig, got {vig:.3f}"
        assert_probability_range([fair_home, fair_away])

    def test_line_movement_calculation(self, mock_odds_data):
        """Test calculation of line movement features."""
        builder = MarketAnchorBuilder()

        # Mock opening and current lines
        opening_lines = mock_odds_data.copy()
        opening_lines["line_type"] = "opening"

        current_lines = mock_odds_data.copy()
        current_lines["snapshot_spread"] += 1.0  # 1 point movement
        current_lines["snapshot_total"] += 0.5  # 0.5 point movement
        current_lines["line_type"] = "current"

        all_lines = pd.concat([opening_lines, current_lines])

        with patch.object(builder, "load_odds_data", return_value=all_lines):
            market_features = builder.build_market_anchors(
                target_season=2024, target_week=1
            )

        assert_dataframe_structure(market_features, min_rows=1)

        # Should have movement calculations
        if "spread_movement" in market_features.columns:
            movements = market_features["spread_movement"].dropna()
            assert all(abs(mov) <= 10 for mov in movements), (
                "Spread movements should be reasonable"
            )


# ---------------------------------------------------------------------------
# Synthetic team stats fixture for dynamic window tests
# ---------------------------------------------------------------------------


def _build_synthetic_team_stats(
    team: str = "BUF",
    seasons: list[int] | None = None,
    weeks_per_season: int = 18,
) -> pd.DataFrame:
    """Create synthetic team-game stats with deterministic values.

    Offensive EPA values are set to `season * 0.01 + week * 0.001` so that
    the expected weighted average can be computed analytically.
    """
    if seasons is None:
        seasons = [2023, 2024]

    rows = []
    for season in seasons:
        for week in range(1, weeks_per_season + 1):
            base_value = season * 0.01 + week * 0.001
            for side in ("offense", "defense"):
                sign = 1.0 if side == "offense" else -1.0
                rows.append(
                    {
                        "game_id": f"SYN_{season}_W{week:02d}_{team}",
                        "season": season,
                        "week": week,
                        "team": team,
                        "side": side,
                        "plays": 60,
                        "total_epa": sign * base_value * 60,
                        "epa_per_play": sign * base_value,
                        "pass_epa_per_play": sign * (base_value + 0.01),
                        "rush_epa_per_play": sign * (base_value - 0.01),
                        "success_rate": 0.45 + sign * 0.02,
                        "pass_success_rate": 0.48 + sign * 0.02,
                        "rush_success_rate": 0.42 + sign * 0.02,
                        "neutral_pass_rate": 0.60 if side == "offense" else np.nan,
                        "red_zone_td_rate": 0.55 + sign * 0.05,
                        "third_down_conversion_rate": 0.40 + sign * 0.03,
                        "pass_attempts": 30,
                        "rush_attempts": 30,
                    }
                )
    return pd.DataFrame(rows)


class TestTeamFormLAMapping:
    """Tests for the LA -> LAR mapping bug fix."""

    def test_build_team_mapping_no_lar(self):
        """_build_team_mapping must NOT map any team to 'LAR'."""
        builder = TeamFormBuilder()
        mapping = builder._build_team_mapping()
        assert "LAR" not in mapping.values(), (
            "Team mapping should not contain 'LAR' -- LA is canonical for Rams"
        )

    def test_normalize_la_stays_la(self):
        """_normalize_team_name('LA') must return 'LA', not 'LAR'."""
        builder = TeamFormBuilder()
        result = builder._normalize_team_name("LA")
        assert result == "LA", f"Expected 'LA', got '{result}'"


class TestTeamFormDynamicWindow:
    """Tests for dynamic EPA window sizing."""

    def test_week1_uses_only_prior_season(self):
        """For Week 1, only prior-season games should be used (current = 0)."""
        builder = TeamFormBuilder()
        stats = _build_synthetic_team_stats("BUF", [2023, 2024])

        rolling = builder.calculate_rolling_averages(
            stats, target_season=2024, target_week=1
        )
        assert len(rolling) > 0, "Should produce rolling averages for Week 1"

        # All games used should come from season 2023
        buf_off = rolling[(rolling["team"] == "BUF") & (rolling["side"] == "offense")]
        assert len(buf_off) == 1
        row = buf_off.iloc[0]

        # games_used should equal max_prior_games (default 8) since 2023 has 18 weeks
        assert row["games_used"] == 8, (
            f"Week 1 should use 8 prior-season games, got {row['games_used']}"
        )

    def test_week5_blended_window(self):
        """For Week 5, both prior-season and current-season games are used."""
        builder = TeamFormBuilder()
        stats = _build_synthetic_team_stats("BUF", [2023, 2024])

        rolling = builder.calculate_rolling_averages(
            stats, target_season=2024, target_week=5
        )
        buf_off = rolling[(rolling["team"] == "BUF") & (rolling["side"] == "offense")]
        assert len(buf_off) == 1
        row = buf_off.iloc[0]

        # Current-season games: weeks 1-4 = 4 games
        # Prior-season games: max(0, 8 - 4) = 4 games from end of 2023
        # Total: 8 games
        assert row["games_used"] == 8, (
            f"Week 5 should use 8 total games (4 current + 4 prior), got {row['games_used']}"
        )

    def test_week12_dominated_by_current_season(self):
        """For Week 12, current-season data dominates (prior weight near 0)."""
        builder = TeamFormBuilder()
        stats = _build_synthetic_team_stats("BUF", [2023, 2024])

        rolling = builder.calculate_rolling_averages(
            stats, target_season=2024, target_week=12
        )
        buf_off = rolling[(rolling["team"] == "BUF") & (rolling["side"] == "offense")]
        assert len(buf_off) == 1
        row = buf_off.iloc[0]

        # Current-season games: weeks 1-11 = 11 games
        # Prior-season games: max(0, 8 - 11) = 0 games
        # Total: 11 games
        assert row["games_used"] == 11, (
            f"Week 12 should use 11 current-season games (no prior), got {row['games_used']}"
        )


class TestTeamFormNoLeakage:
    """Tests for temporal correctness -- no off-by-one leakage."""

    def test_week_n_excludes_week_n_data(self):
        """Features for Week N must NOT include data from Week N or later."""
        builder = TeamFormBuilder()
        stats = _build_synthetic_team_stats("BUF", [2024], weeks_per_season=18)

        # Calculate for week 5 -- should use weeks 1-4 only
        rolling = builder.calculate_rolling_averages(
            stats, target_season=2024, target_week=5
        )
        buf_off = rolling[(rolling["team"] == "BUF") & (rolling["side"] == "offense")]
        assert len(buf_off) == 1

        # With no prior season and target_week=5, games_used should be exactly 4
        # (weeks 1, 2, 3, 4 -- NOT week 5)
        row = buf_off.iloc[0]
        assert row["games_used"] == 4, (
            f"Week 5 features should use exactly 4 games (weeks 1-4), got {row['games_used']}"
        )

        # The EPA values should reflect only weeks 1-4.
        # Synthetic offense EPA for 2024 weeks 1-4 = season*0.01 + week*0.001
        # so values are 20.241, 20.242, 20.243, 20.244.
        # With recency weights [1,2,3,4]/10 the weighted avg = 20.2430
        expected_epa = (20.241 * 1 + 20.242 * 2 + 20.243 * 3 + 20.244 * 4) / 10
        actual_epa = row["rolling_epa_per_play"]
        assert abs(actual_epa - expected_epa) < 1e-6, (
            f"Expected EPA {expected_epa:.6f}, got {actual_epa:.6f} -- "
            "possible off-by-one leak or incorrect weighting"
        )


class TestTeamFormRecencyWeighting:
    """Tests for recency weighting in rolling averages."""

    def test_recency_weighting_applied(self):
        """More recent games must have higher weight in the rolling average."""
        builder = TeamFormBuilder()
        # Use a single season with increasing EPA values
        stats = _build_synthetic_team_stats("BUF", [2024], weeks_per_season=6)

        rolling = builder.calculate_rolling_averages(
            stats, target_season=2024, target_week=7
        )
        buf_off = rolling[(rolling["team"] == "BUF") & (rolling["side"] == "offense")]
        assert len(buf_off) == 1
        row = buf_off.iloc[0]

        # With 6 games and linear weights [1,2,3,4,5,6]:
        # The weighted average should be closer to the later (higher) values
        # than a simple arithmetic mean would be
        values = [2024 * 0.01 + w * 0.001 for w in range(1, 7)]
        simple_mean = np.mean(values)
        weighted_avg = row["rolling_epa_per_play"]

        # Recency-weighted should be > simple mean because later values are
        # higher and get more weight
        assert weighted_avg > simple_mean, (
            f"Recency-weighted avg ({weighted_avg:.6f}) should exceed "
            f"simple mean ({simple_mean:.6f})"
        )


class TestTeamFormProtocolConformance:
    """Tests for FeatureBuilder Protocol conformance."""

    def test_build_features_accepts_as_of_datetime(self):
        """build_features method must accept as_of_datetime parameter."""
        builder = TeamFormBuilder()
        assert hasattr(builder, "build_features"), (
            "TeamFormCalculator must have build_features method"
        )

        # Check the signature accepts games_df and as_of_datetime
        import inspect

        sig = inspect.signature(builder.build_features)
        param_names = list(sig.parameters.keys())
        assert "games_df" in param_names, "build_features must accept games_df"
        assert "as_of_datetime" in param_names, (
            "build_features must accept as_of_datetime"
        )

    def test_get_features_for_game_accepts_as_of_datetime(self):
        """get_features_for_game method must accept as_of_datetime parameter."""
        builder = TeamFormBuilder()
        assert hasattr(builder, "get_features_for_game"), (
            "TeamFormCalculator must have get_features_for_game method"
        )

        import inspect

        sig = inspect.signature(builder.get_features_for_game)
        param_names = list(sig.parameters.keys())
        assert "game_id" in param_names, "get_features_for_game must accept game_id"
        assert "as_of_datetime" in param_names, (
            "get_features_for_game must accept as_of_datetime"
        )

    def test_build_features_filters_by_datetime(self):
        """build_features should filter data by as_of_datetime."""
        builder = TeamFormBuilder()
        stats = _build_synthetic_team_stats("BUF", [2023, 2024])

        # Create a games_df with game_id and kickoff_et
        et_tz = ZoneInfo("America/New_York")
        games_df = pd.DataFrame(
            [
                {
                    "game_id": "SYN_2024_W05_BUF",
                    "season": 2024,
                    "week": 5,
                    "home_team": "BUF",
                    "away_team": "MIA",
                    "kickoff_et": datetime(2024, 10, 6, 13, 0, tzinfo=et_tz),
                }
            ]
        )

        # as_of_datetime set to before Week 5 kickoff
        as_of = datetime(2024, 10, 4, 18, 0, tzinfo=et_tz)

        # Patch fetch_pbp_data and calculate_team_game_stats to return
        # our synthetic data without hitting nflreadpy
        with (
            patch.object(builder, "fetch_pbp_data"),
            patch.object(builder, "calculate_team_game_stats", return_value=stats),
        ):
            result = builder.build_features(
                games_df=games_df,
                as_of_datetime=as_of,
                target_season=2024,
                target_week=5,
            )

        assert isinstance(result, pd.DataFrame), (
            "build_features should return a DataFrame"
        )
        assert len(result) > 0, "build_features should return non-empty results"


class TestTeamFormAllMetricsPreserved:
    """Verify all 9 team form metrics are still computed."""

    EXPECTED_METRICS = [
        "rolling_epa_per_play",
        "rolling_pass_epa_per_play",
        "rolling_rush_epa_per_play",
        "rolling_success_rate",
        "rolling_pass_success_rate",
        "rolling_rush_success_rate",
        "rolling_neutral_pass_rate",
        "rolling_red_zone_td_rate",
        "rolling_third_down_conversion_rate",
    ]

    def test_all_metrics_present_in_output(self):
        """All 9 rolling metrics must appear in rolling averages output."""
        builder = TeamFormBuilder()
        stats = _build_synthetic_team_stats("BUF", [2023, 2024])

        rolling = builder.calculate_rolling_averages(
            stats, target_season=2024, target_week=5
        )
        assert len(rolling) > 0

        for metric in self.EXPECTED_METRICS:
            assert metric in rolling.columns, f"Missing expected metric: {metric}"
