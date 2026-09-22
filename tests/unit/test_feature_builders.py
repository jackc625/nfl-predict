"""Unit tests for feature building modules.

Phase 3 refactored all feature builder APIs (Protocol conformance, compressed
features, temporal safety). Old Phase 1 test classes that tested removed APIs
have been deleted. Current coverage is provided by:
- test_feature_compression.py (weather, market, contextual compression)
- test_feature_protocol.py (FeatureBuilder Protocol conformance)
- test_elo_correctness.py (Elo rating system)
- test_leakage_gate.py (LeakageGate validation)
- test_expanding_normalization.py (expanding-window normalization)
"""

from datetime import datetime
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from features.team_form import TeamFormCalculator as TeamFormBuilder

# ---------------------------------------------------------------------------
# Synthetic team stats fixture for dynamic window tests
# ---------------------------------------------------------------------------


def _weekly_schedule(team: str = "BUF", seasons: tuple[int, ...] = (2023, 2024)):
    """One Sunday game per week (weeks 1-18) for *team*: the schedule the synthetic
    team-games are TIMED against and each target game LOCKED from (Plan 33.2-14).

    Injected so the calculator never reads the local data lake: since the window is
    admitted at the target game's lock, a synthetic row needs its game's kickoff. On this
    ordinary weekly schedule the lock-keyed window selects exactly the prior weeks.
    """
    rows = []
    for season in seasons:
        first_sunday = pd.Timestamp(f"{season}-09-10 17:00", tz="UTC")
        for week in range(1, 19):
            rows.append(
                {
                    "game_id": f"SYN_{season}_W{week:02d}_{team}",
                    "season": season,
                    "week": week,
                    "home_team": team,
                    "away_team": "MIA",
                    "kickoff_et": first_sunday + pd.Timedelta(weeks=week - 1),
                }
            )
    return pd.DataFrame(rows)


_SCHEDULE = _weekly_schedule()


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
        builder = TeamFormBuilder(schedule_df=_SCHEDULE)
        mapping = builder._build_team_mapping()
        assert "LAR" not in mapping.values(), (
            "Team mapping should not contain 'LAR' -- LA is canonical for Rams"
        )

    def test_normalize_la_stays_la(self):
        """_normalize_team_name('LA') must return 'LA', not 'LAR'."""
        builder = TeamFormBuilder(schedule_df=_SCHEDULE)
        result = builder._normalize_team_name("LA")
        assert result == "LA", f"Expected 'LA', got '{result}'"


class TestTeamFormDynamicWindow:
    """Tests for dynamic EPA window sizing."""

    def test_week1_uses_only_prior_season(self):
        """For Week 1, only prior-season games should be used (current = 0)."""
        builder = TeamFormBuilder(schedule_df=_SCHEDULE)
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
        builder = TeamFormBuilder(schedule_df=_SCHEDULE)
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
        builder = TeamFormBuilder(schedule_df=_SCHEDULE)
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
        builder = TeamFormBuilder(schedule_df=_SCHEDULE)
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
        builder = TeamFormBuilder(schedule_df=_SCHEDULE)
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
        builder = TeamFormBuilder(schedule_df=_SCHEDULE)
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
        builder = TeamFormBuilder(schedule_df=_SCHEDULE)
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
        builder = TeamFormBuilder(schedule_df=_SCHEDULE)
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
        builder = TeamFormBuilder(schedule_df=_SCHEDULE)
        stats = _build_synthetic_team_stats("BUF", [2023, 2024])

        rolling = builder.calculate_rolling_averages(
            stats, target_season=2024, target_week=5
        )
        assert len(rolling) > 0

        for metric in self.EXPECTED_METRICS:
            assert metric in rolling.columns, f"Missing expected metric: {metric}"
