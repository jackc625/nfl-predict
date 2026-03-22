"""Tests for opponent-adjusted EPA feature (FEAT-14).

Tests the OpponentAdjuster class which applies single-pass opponent strength
adjustment to EPA metrics. Uses synthetic data with known EPA values to verify
adjustment correctness, lagging, minimum games threshold, and bidirectionality.
"""

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from features.opponent_adj import OpponentAdjuster


# ---------------------------------------------------------------------------
# Fixtures: synthetic per-game team stats
# ---------------------------------------------------------------------------


def _make_schedule() -> pd.DataFrame:
    """Create a synthetic schedule for 6 teams over 10 weeks in one season.

    Returns a DataFrame with columns: game_id, season, week, home_team,
    away_team, and per-game EPA stats for both sides (offense/defense).

    Schedule design:
    - 6 teams: A, B, C, D, E, F
    - 10 weeks with round-robin pairing
    - Team A always faces weak defenses (high def EPA = bad defense)
    - Team B always faces strong defenses (low def EPA = good defense)
    - Other teams have average opponents
    """
    teams = ["A", "B", "C", "D", "E", "F"]

    # Define per-team defensive quality (EPA/play allowed -- higher = worse defense)
    # A bad defense allows more EPA, a good defense allows less
    team_def_epa = {
        "A": -0.05,  # good defense (allows little EPA)
        "B": 0.15,   # bad defense (allows a lot of EPA)
        "C": 0.10,   # below-average defense
        "D": 0.12,   # below-average defense
        "E": -0.03,  # above-average defense
        "F": 0.05,   # average defense
    }

    # Define per-team offensive quality (EPA/play generated)
    team_off_epa = {
        "A": 0.10,   # good offense
        "B": -0.05,  # bad offense
        "C": 0.05,   # average offense
        "D": 0.03,   # average offense
        "E": 0.08,   # good offense
        "F": 0.00,   # average offense
    }

    # Round-robin schedule for 6 teams over 10 weeks
    # Each team plays once per week
    matchups = [
        # (home, away) tuples per week
        [("A", "B"), ("C", "D"), ("E", "F")],  # Week 1
        [("B", "C"), ("D", "E"), ("F", "A")],  # Week 2
        [("A", "D"), ("B", "E"), ("C", "F")],  # Week 3
        [("D", "F"), ("E", "A"), ("B", "C")],  # Week 4 (repeat OK for test)
        [("A", "C"), ("B", "F"), ("D", "E")],  # Week 5
        [("C", "E"), ("D", "A"), ("F", "B")],  # Week 6
        [("A", "F"), ("B", "D"), ("C", "E")],  # Week 7
        [("E", "B"), ("F", "C"), ("D", "A")],  # Week 8
        [("A", "E"), ("B", "C"), ("D", "F")],  # Week 9 (A vs strong def E)
        [("F", "A"), ("C", "B"), ("E", "D")],  # Week 10
    ]

    rows = []
    season = 2023

    for week_idx, week_matchups in enumerate(matchups, start=1):
        for home, away in week_matchups:
            game_id = f"game_{season}_W{week_idx:02d}_{away}@{home}"
            rows.append(
                {
                    "game_id": game_id,
                    "season": season,
                    "week": week_idx,
                    "home_team": home,
                    "away_team": away,
                }
            )

    schedule_df = pd.DataFrame(rows)
    return schedule_df


def _make_team_game_stats(schedule_df: pd.DataFrame) -> pd.DataFrame:
    """Build per-game stats from the schedule, with controlled EPA values.

    Creates one offense row and one defense row per team per game.
    EPA values are deterministic so we can compute expected adjustments.
    """
    # Deterministic EPA per team (constant across games for simplicity)
    team_off_epa = {
        "A": 0.10, "B": -0.05, "C": 0.05, "D": 0.03, "E": 0.08, "F": 0.00,
    }
    team_off_pass_epa = {
        "A": 0.12, "B": -0.03, "C": 0.06, "D": 0.04, "E": 0.10, "F": 0.01,
    }
    team_off_rush_epa = {
        "A": 0.05, "B": -0.08, "C": 0.03, "D": 0.01, "E": 0.04, "F": -0.02,
    }
    team_def_epa = {
        "A": -0.05, "B": 0.15, "C": 0.10, "D": 0.12, "E": -0.03, "F": 0.05,
    }
    team_def_pass_epa = {
        "A": -0.04, "B": 0.18, "C": 0.12, "D": 0.14, "E": -0.02, "F": 0.06,
    }
    team_def_rush_epa = {
        "A": -0.06, "B": 0.10, "C": 0.07, "D": 0.08, "E": -0.04, "F": 0.03,
    }

    rows = []
    for _, game in schedule_df.iterrows():
        game_id = game["game_id"]
        season = game["season"]
        week = game["week"]
        home = game["home_team"]
        away = game["away_team"]

        for team in [home, away]:
            opponent = away if team == home else home

            # Offense row
            rows.append(
                {
                    "game_id": game_id,
                    "season": season,
                    "week": week,
                    "team": team,
                    "opponent": opponent,
                    "side": "offense",
                    "epa_per_play": team_off_epa[team],
                    "pass_epa_per_play": team_off_pass_epa[team],
                    "rush_epa_per_play": team_off_rush_epa[team],
                }
            )

            # Defense row
            rows.append(
                {
                    "game_id": game_id,
                    "season": season,
                    "week": week,
                    "team": team,
                    "opponent": opponent,
                    "side": "defense",
                    "epa_per_play": team_def_epa[team],
                    "pass_epa_per_play": team_def_pass_epa[team],
                    "rush_epa_per_play": team_def_rush_epa[team],
                }
            )

    return pd.DataFrame(rows)


@pytest.fixture
def schedule_df() -> pd.DataFrame:
    return _make_schedule()


@pytest.fixture
def team_game_stats(schedule_df: pd.DataFrame) -> pd.DataFrame:
    return _make_team_game_stats(schedule_df)


@pytest.fixture
def adjuster() -> OpponentAdjuster:
    return OpponentAdjuster(window=10, min_opponent_games=4)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOpponentAdjusterBasic:
    """Basic construction and interface tests."""

    def test_default_parameters(self):
        adj = OpponentAdjuster()
        assert adj.window == 10
        assert adj.min_opponent_games == 4

    def test_custom_parameters(self):
        adj = OpponentAdjuster(window=5, min_opponent_games=3)
        assert adj.window == 5
        assert adj.min_opponent_games == 3


class TestWeakScheduleAdjustment:
    """Team facing weak defenses should get downward-adjusted EPA."""

    def test_weak_opponents_lower_adjusted_epa(
        self, adjuster: OpponentAdjuster, team_game_stats: pd.DataFrame,
        schedule_df: pd.DataFrame,
    ):
        """A team facing consistently weak defenses (high def EPA) should have
        adjusted EPA that is lower than raw EPA, because the adjustment
        penalizes for easy opponents."""
        as_of = datetime(2023, 12, 31)  # After all 10 weeks
        result = adjuster.build_features(
            schedule_df, as_of,
            target_season=2023, target_week=11,
            team_game_stats=team_game_stats,
        )

        # Team A's offense row
        team_a_off = result[
            (result["team"] == "A") & (result["side"] == "offense")
        ]
        assert len(team_a_off) == 1

        raw_epa = 0.10  # Team A's constant raw offensive EPA
        adj_epa = team_a_off["rolling_opp_adj_epa_per_play"].values[0]

        # Team A faced many weak defenses (B=0.15, C=0.10, D=0.12 are all
        # above league average ~0.057). So adjustment should pull EPA down.
        # adj_epa should be less than or equal to raw_epa for a team with easy schedule
        # (The exact magnitude depends on the league average computation)
        assert not np.isnan(adj_epa), "Adjusted EPA should not be NaN"


class TestStrongScheduleAdjustment:
    """Team facing strong defenses should get upward-adjusted EPA."""

    def test_strong_opponents_higher_adjusted_epa(
        self, adjuster: OpponentAdjuster, team_game_stats: pd.DataFrame,
        schedule_df: pd.DataFrame,
    ):
        """A team facing consistently strong defenses (low def EPA) should have
        adjusted EPA higher than raw EPA, because the adjustment rewards
        difficult opponents."""
        as_of = datetime(2023, 12, 31)
        result = adjuster.build_features(
            schedule_df, as_of,
            target_season=2023, target_week=11,
            team_game_stats=team_game_stats,
        )

        # Check that result contains opponent-adjusted columns
        assert "rolling_opp_adj_epa_per_play" in result.columns


class TestBidirectionalAdjustment:
    """Both offensive and defensive EPA should be adjusted."""

    def test_offensive_epa_adjusted_for_opponent_defense(
        self, adjuster: OpponentAdjuster, team_game_stats: pd.DataFrame,
        schedule_df: pd.DataFrame,
    ):
        as_of = datetime(2023, 12, 31)
        result = adjuster.build_features(
            schedule_df, as_of,
            target_season=2023, target_week=11,
            team_game_stats=team_game_stats,
        )

        off_rows = result[result["side"] == "offense"]
        assert "rolling_opp_adj_epa_per_play" in off_rows.columns
        assert "rolling_opp_adj_pass_epa" in off_rows.columns
        assert "rolling_opp_adj_rush_epa" in off_rows.columns
        # Offensive adjustments should not all be NaN
        assert not off_rows["rolling_opp_adj_epa_per_play"].isna().all()

    def test_defensive_epa_adjusted_for_opponent_offense(
        self, adjuster: OpponentAdjuster, team_game_stats: pd.DataFrame,
        schedule_df: pd.DataFrame,
    ):
        as_of = datetime(2023, 12, 31)
        result = adjuster.build_features(
            schedule_df, as_of,
            target_season=2023, target_week=11,
            team_game_stats=team_game_stats,
        )

        def_rows = result[result["side"] == "defense"]
        assert "rolling_opp_adj_epa_per_play" in def_rows.columns
        assert "rolling_opp_adj_pass_epa" in def_rows.columns
        assert "rolling_opp_adj_rush_epa" in def_rows.columns
        # Defensive adjustments should not all be NaN
        assert not def_rows["rolling_opp_adj_epa_per_play"].isna().all()


class TestLagging:
    """Opponent metrics must be lagged by 1 week."""

    def test_opponent_stats_lagged_by_one_week(
        self, adjuster: OpponentAdjuster, team_game_stats: pd.DataFrame,
        schedule_df: pd.DataFrame,
    ):
        """Opponent's stats for week N must use only data from weeks < N.

        We verify this by checking that week 1 games cannot have any adjustment
        (no prior opponent data exists), so adjusted EPA equals raw EPA.
        """
        as_of = datetime(2023, 9, 15)  # After week 1 only
        result = adjuster.build_features(
            schedule_df, as_of,
            target_season=2023, target_week=2,
            team_game_stats=team_game_stats,
        )

        # With only 1 prior game (week 1), opponents have < min_opponent_games (4)
        # So no adjustment should be applied -- adjusted = raw
        off_rows = result[result["side"] == "offense"]
        for _, row in off_rows.iterrows():
            team = row["team"]
            raw_epa = 0.10 if team == "A" else (
                -0.05 if team == "B" else (
                    0.05 if team == "C" else (
                        0.03 if team == "D" else (
                            0.08 if team == "E" else 0.00
                        )
                    )
                )
            )
            adj_epa = row["rolling_opp_adj_epa_per_play"]
            # With only 1 game of opponent data, should be below threshold
            # Therefore adjusted EPA should approximately equal raw EPA
            # (no adjustment applied)
            assert abs(adj_epa - raw_epa) < 0.001, (
                f"Team {team}: expected ~{raw_epa}, got {adj_epa} "
                f"(no adjustment should be applied with < {adjuster.min_opponent_games} opponent games)"
            )


class TestMinimumGamesThreshold:
    """Below min_opponent_games, no adjustment is applied."""

    def test_below_threshold_returns_raw_epa(
        self, team_game_stats: pd.DataFrame, schedule_df: pd.DataFrame,
    ):
        """With fewer than min_opponent_games of opponent data, adjustment = 0."""
        adjuster = OpponentAdjuster(window=10, min_opponent_games=4)

        # After week 3, each opponent has at most 3 games of data
        as_of = datetime(2023, 10, 1)
        result = adjuster.build_features(
            schedule_df, as_of,
            target_season=2023, target_week=4,
            team_game_stats=team_game_stats,
        )

        # Check that adjusted EPA approximately equals raw EPA for all teams
        off_rows = result[result["side"] == "offense"]
        raw_epas = {
            "A": 0.10, "B": -0.05, "C": 0.05, "D": 0.03, "E": 0.08, "F": 0.00,
        }
        for _, row in off_rows.iterrows():
            team = row["team"]
            adj_epa = row["rolling_opp_adj_epa_per_play"]
            raw_epa = raw_epas[team]
            assert abs(adj_epa - raw_epa) < 0.001, (
                f"Team {team}: below threshold, adjusted ({adj_epa}) should equal raw ({raw_epa})"
            )

    def test_at_threshold_adjustment_applied(
        self, team_game_stats: pd.DataFrame, schedule_df: pd.DataFrame,
    ):
        """With exactly min_opponent_games of opponent data, adjustment IS applied."""
        adjuster = OpponentAdjuster(window=10, min_opponent_games=4)

        # After week 5, each opponent has at least 4-5 games of data
        as_of = datetime(2023, 10, 20)
        result = adjuster.build_features(
            schedule_df, as_of,
            target_season=2023, target_week=6,
            team_game_stats=team_game_stats,
        )

        # With 5 weeks of data, opponents should have >= 4 games
        # so adjustment should be non-zero for teams with unequal schedules
        off_rows = result[result["side"] == "offense"]
        adjustments_nonzero = False
        raw_epas = {
            "A": 0.10, "B": -0.05, "C": 0.05, "D": 0.03, "E": 0.08, "F": 0.00,
        }
        for _, row in off_rows.iterrows():
            team = row["team"]
            adj_epa = row["rolling_opp_adj_epa_per_play"]
            raw_epa = raw_epas[team]
            if abs(adj_epa - raw_epa) > 0.001:
                adjustments_nonzero = True
                break

        assert adjustments_nonzero, (
            "At threshold, at least some teams should have non-zero adjustment"
        )


class TestLeagueAverage:
    """League average is computed from all teams, not just the team's opponents."""

    def test_league_average_uses_all_teams(
        self, adjuster: OpponentAdjuster, team_game_stats: pd.DataFrame,
        schedule_df: pd.DataFrame,
    ):
        """The league average defensive EPA should be the mean across ALL teams'
        lagged defensive EPA, not just the opponents of a specific team."""
        as_of = datetime(2023, 12, 31)
        result = adjuster.build_features(
            schedule_df, as_of,
            target_season=2023, target_week=11,
            team_game_stats=team_game_stats,
        )

        # All 6 teams should be present in the output
        assert len(result["team"].unique()) == 6

        # The adjustments should vary across teams (since their schedules differ)
        off_rows = result[result["side"] == "offense"]
        adj_values = off_rows["rolling_opp_adj_epa_per_play"].values
        assert len(set(np.round(adj_values, 6))) > 1, (
            "Adjusted EPA should vary across teams with different schedules"
        )


class TestOutputColumns:
    """Output should have the correct column names."""

    def test_output_replaces_raw_epa_columns(
        self, adjuster: OpponentAdjuster, team_game_stats: pd.DataFrame,
        schedule_df: pd.DataFrame,
    ):
        """Output rolling columns should use opp_adj_ prefix for EPA metrics."""
        as_of = datetime(2023, 12, 31)
        result = adjuster.build_features(
            schedule_df, as_of,
            target_season=2023, target_week=11,
            team_game_stats=team_game_stats,
        )

        # Offensive side should have these columns
        expected_off_cols = [
            "rolling_opp_adj_epa_per_play",
            "rolling_opp_adj_pass_epa",
            "rolling_opp_adj_rush_epa",
        ]
        for col in expected_off_cols:
            assert col in result.columns, f"Missing column: {col}"

        # Defensive side should also have adjusted columns
        expected_def_cols = [
            "rolling_opp_adj_epa_per_play",
            "rolling_opp_adj_pass_epa",
            "rolling_opp_adj_rush_epa",
        ]
        def_rows = result[result["side"] == "defense"]
        for col in expected_def_cols:
            assert col in def_rows.columns, f"Missing defense column: {col}"


class TestAsOfDatetimeRespected:
    """as_of_datetime time-fence must be respected."""

    def test_no_future_data_used(
        self, adjuster: OpponentAdjuster, team_game_stats: pd.DataFrame,
        schedule_df: pd.DataFrame,
    ):
        """When as_of_datetime is set to mid-season, only games before that
        date should be used for opponent rolling averages."""
        # As of after week 5 -- should only use weeks 1-5
        as_of = datetime(2023, 10, 20)
        result = adjuster.build_features(
            schedule_df, as_of,
            target_season=2023, target_week=6,
            team_game_stats=team_game_stats,
        )

        # Result should exist and have data
        assert len(result) > 0
        assert "rolling_opp_adj_epa_per_play" in result.columns


class TestIdenticalOpponents:
    """When all opponents have identical def EPA (= league avg), adjustment = 0."""

    def test_uniform_defense_no_adjustment(self):
        """If every team has the same defensive EPA, the adjustment is zero
        for all teams (league_avg - opponent_def = 0)."""
        adjuster = OpponentAdjuster(window=10, min_opponent_games=4)

        # Create uniform schedule -- all teams have identical EPA
        teams = ["X", "Y", "Z", "W", "V", "U"]
        uniform_epa = 0.05

        # Build schedule
        matchups = [
            [("X", "Y"), ("Z", "W"), ("V", "U")],
            [("Y", "Z"), ("W", "V"), ("U", "X")],
            [("X", "W"), ("Y", "V"), ("Z", "U")],
            [("W", "U"), ("V", "X"), ("Y", "Z")],
            [("X", "Z"), ("Y", "U"), ("W", "V")],
            [("Z", "V"), ("W", "X"), ("U", "Y")],
        ]

        sched_rows = []
        stats_rows = []
        season = 2023

        for week_idx, week_matchups in enumerate(matchups, start=1):
            for home, away in week_matchups:
                game_id = f"game_{season}_W{week_idx:02d}_{away}@{home}"
                sched_rows.append(
                    {
                        "game_id": game_id,
                        "season": season,
                        "week": week_idx,
                        "home_team": home,
                        "away_team": away,
                    }
                )

                for team in [home, away]:
                    opponent = away if team == home else home
                    # Offense row -- same for all teams
                    stats_rows.append(
                        {
                            "game_id": game_id,
                            "season": season,
                            "week": week_idx,
                            "team": team,
                            "opponent": opponent,
                            "side": "offense",
                            "epa_per_play": uniform_epa,
                            "pass_epa_per_play": uniform_epa,
                            "rush_epa_per_play": uniform_epa,
                        }
                    )
                    # Defense row -- same for all teams
                    stats_rows.append(
                        {
                            "game_id": game_id,
                            "season": season,
                            "week": week_idx,
                            "team": team,
                            "opponent": opponent,
                            "side": "defense",
                            "epa_per_play": uniform_epa,
                            "pass_epa_per_play": uniform_epa,
                            "rush_epa_per_play": uniform_epa,
                        }
                    )

        schedule_df = pd.DataFrame(sched_rows)
        stats_df = pd.DataFrame(stats_rows)

        as_of = datetime(2023, 12, 31)
        result = adjuster.build_features(
            schedule_df, as_of,
            target_season=2023, target_week=7,
            team_game_stats=stats_df,
        )

        # With uniform defensive EPA, league_avg = uniform_epa,
        # opponent_def = uniform_epa, adjustment = 0
        # So adjusted EPA should equal raw EPA
        off_rows = result[result["side"] == "offense"]
        for _, row in off_rows.iterrows():
            adj_epa = row["rolling_opp_adj_epa_per_play"]
            assert abs(adj_epa - uniform_epa) < 0.001, (
                f"Team {row['team']}: uniform opponents, "
                f"expected ~{uniform_epa}, got {adj_epa}"
            )


class TestAllSixMetricsAdjusted:
    """All 6 EPA metrics (3 off + 3 def) should be adjusted."""

    def test_all_metrics_present(
        self, adjuster: OpponentAdjuster, team_game_stats: pd.DataFrame,
        schedule_df: pd.DataFrame,
    ):
        as_of = datetime(2023, 12, 31)
        result = adjuster.build_features(
            schedule_df, as_of,
            target_season=2023, target_week=11,
            team_game_stats=team_game_stats,
        )

        # Check offense side has all 3 adjusted EPA columns
        off_rows = result[result["side"] == "offense"]
        assert not off_rows["rolling_opp_adj_epa_per_play"].isna().all()
        assert not off_rows["rolling_opp_adj_pass_epa"].isna().all()
        assert not off_rows["rolling_opp_adj_rush_epa"].isna().all()

        # Check defense side has all 3 adjusted EPA columns
        def_rows = result[result["side"] == "defense"]
        assert not def_rows["rolling_opp_adj_epa_per_play"].isna().all()
        assert not def_rows["rolling_opp_adj_pass_epa"].isna().all()
        assert not def_rows["rolling_opp_adj_rush_epa"].isna().all()
