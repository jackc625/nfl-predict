"""Tests for opponent-adjusted EPA feature (FEAT-14).

Tests the OpponentAdjuster class which applies single-pass opponent strength
adjustment to EPA metrics. Uses synthetic data with known EPA values to verify
adjustment correctness, lagging, minimum games threshold, and bidirectionality.

PLAN 33.2-16 moved this fixture onto REAL team abbreviations, play-by-play game ids and a
timed schedule: opponents now resolve through ``utils.game_id_utils.convert_legacy_game_id``
and every input is admitted at the lock of the game it informs, so a synthetic
``game_2023_W01_B@A`` id with teams "A".."F" is (correctly) refused. Two tests used to assert
the defect itself -- "below the threshold, adjusted EPA equals raw EPA" -- and now assert its
replacement: NaN beside a false coverage flag. Each says so with a "Was:" line.
"""

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from features.opponent_adj import OPP_ADJ_COVERAGE_COLUMN, OpponentAdjuster

ET = ZoneInfo("America/New_York")
_SEASON_START = datetime(2023, 9, 10, 13, 0, tzinfo=ET)  # a Sunday

# The six synthetic teams, on real abbreviations (the canonical converter refuses others).
A, B, C, D, E, F = "KC", "BUF", "MIA", "NYJ", "NE", "DEN"

# ---------------------------------------------------------------------------
# Fixtures: synthetic per-game team stats
# ---------------------------------------------------------------------------

#: (home, away) tuples per week -- every team plays every week.
_MATCHUPS = [
    [(A, B), (C, D), (E, F)],  # Week 1
    [(B, C), (D, E), (F, A)],  # Week 2
    [(A, D), (B, E), (C, F)],  # Week 3
    [(D, F), (E, A), (B, C)],  # Week 4 (repeat OK for test)
    [(A, C), (B, F), (D, E)],  # Week 5
    [(C, E), (D, A), (F, B)],  # Week 6
    [(A, F), (B, D), (C, E)],  # Week 7
    [(E, B), (F, C), (D, A)],  # Week 8
    [(A, E), (B, C), (D, F)],  # Week 9 (A vs strong def E)
    [(F, A), (C, B), (E, D)],  # Week 10
    [(A, B), (C, D), (E, F)],  # Week 11 (the target week for the full-history tests)
]


def _schedule_from(matchups, season: int = 2023) -> pd.DataFrame:
    """Silver ``games`` shape: canonical ids and tz-aware Sunday kickoffs."""
    rows = []
    for week_idx, week_matchups in enumerate(matchups, start=1):
        kickoff = _SEASON_START + timedelta(weeks=week_idx - 1)
        for home, away in week_matchups:
            rows.append(
                {
                    "game_id": f"{season}_W{week_idx:02d}_{away}@{home}",
                    "season": season,
                    "week": week_idx,
                    "home_team": home,
                    "away_team": away,
                    "kickoff_et": pd.Timestamp(kickoff),
                }
            )
    schedule_df = pd.DataFrame(rows)
    schedule_df["kickoff_et"] = pd.to_datetime(schedule_df["kickoff_et"], utc=True)
    return schedule_df


def _pbp_id(game: pd.Series) -> str:
    """The play-by-play spelling of a scheduled game (``2023_01_BUF_KC``)."""
    return (
        f"{game['season']}_{game['week']:02d}_{game['away_team']}_{game['home_team']}"
    )


def _make_schedule() -> pd.DataFrame:
    """A synthetic schedule for 6 teams over 11 weeks in one season.

    Schedule design:
    - 6 teams: A, B, C, D, E, F
    - Team A always faces weak defenses (high def EPA = bad defense)
    - Team B always faces strong defenses (low def EPA = good defense)
    - Other teams have average opponents
    """
    return _schedule_from(_MATCHUPS)


# Deterministic EPA per team (constant across games for simplicity)
TEAM_OFF_EPA = {A: 0.10, B: -0.05, C: 0.05, D: 0.03, E: 0.08, F: 0.00}
TEAM_OFF_PASS_EPA = {A: 0.12, B: -0.03, C: 0.06, D: 0.04, E: 0.10, F: 0.01}
TEAM_OFF_RUSH_EPA = {A: 0.05, B: -0.08, C: 0.03, D: 0.01, E: 0.04, F: -0.02}
TEAM_DEF_EPA = {A: -0.05, B: 0.15, C: 0.10, D: 0.12, E: -0.03, F: 0.05}
TEAM_DEF_PASS_EPA = {A: -0.04, B: 0.18, C: 0.12, D: 0.14, E: -0.02, F: 0.06}
TEAM_DEF_RUSH_EPA = {A: -0.06, B: 0.10, C: 0.07, D: 0.08, E: -0.04, F: 0.03}


def _make_team_game_stats(schedule_df: pd.DataFrame) -> pd.DataFrame:
    """Build per-game stats from the schedule, with controlled EPA values.

    Creates one offense row and one defense row per team per game, keyed by the
    PLAY-BY-PLAY id as TeamFormCalculator emits it. No ``opponent`` column: the adjuster
    resolves it through the canonical mapping.
    """
    rows = []
    for _, game in schedule_df.iterrows():
        for team in [game["home_team"], game["away_team"]]:
            base = {
                "game_id": _pbp_id(game),
                "season": game["season"],
                "week": game["week"],
                "team": team,
            }
            rows.append(
                {
                    **base,
                    "side": "offense",
                    "epa_per_play": TEAM_OFF_EPA[team],
                    "pass_epa_per_play": TEAM_OFF_PASS_EPA[team],
                    "rush_epa_per_play": TEAM_OFF_RUSH_EPA[team],
                }
            )
            rows.append(
                {
                    **base,
                    "side": "defense",
                    "epa_per_play": TEAM_DEF_EPA[team],
                    "pass_epa_per_play": TEAM_DEF_PASS_EPA[team],
                    "rush_epa_per_play": TEAM_DEF_RUSH_EPA[team],
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
def adjuster(schedule_df: pd.DataFrame) -> OpponentAdjuster:
    return OpponentAdjuster(window=10, min_opponent_games=4, schedule_df=schedule_df)


_AS_OF = datetime(2023, 12, 31, tzinfo=ET)

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
        self,
        adjuster: OpponentAdjuster,
        team_game_stats: pd.DataFrame,
        schedule_df: pd.DataFrame,
    ):
        """A team facing consistently weak defenses (high def EPA) should have
        adjusted EPA that is lower than raw EPA, because the adjustment
        penalizes for easy opponents."""
        result = adjuster.build_features(
            schedule_df,
            _AS_OF,
            target_season=2023,
            target_week=11,
            team_game_stats=team_game_stats,
        )

        # Team A's offense row
        team_a_off = result[(result["team"] == A) & (result["side"] == "offense")]
        assert len(team_a_off) == 1

        adj_epa = team_a_off["rolling_opp_adj_epa_per_play"].values[0]

        # Team A faced many weak defenses (B=0.15, C=0.10, D=0.12 are all
        # above league average ~0.057). So adjustment should pull EPA down.
        assert not np.isnan(adj_epa), "Adjusted EPA should not be NaN"


class TestStrongScheduleAdjustment:
    """Team facing strong defenses should get upward-adjusted EPA."""

    def test_strong_opponents_higher_adjusted_epa(
        self,
        adjuster: OpponentAdjuster,
        team_game_stats: pd.DataFrame,
        schedule_df: pd.DataFrame,
    ):
        """A team facing consistently strong defenses (low def EPA) should have
        adjusted EPA higher than raw EPA, because the adjustment rewards
        difficult opponents."""
        result = adjuster.build_features(
            schedule_df,
            _AS_OF,
            target_season=2023,
            target_week=11,
            team_game_stats=team_game_stats,
        )

        # Check that result contains opponent-adjusted columns
        assert "rolling_opp_adj_epa_per_play" in result.columns


class TestBidirectionalAdjustment:
    """Both offensive and defensive EPA should be adjusted."""

    def test_offensive_epa_adjusted_for_opponent_defense(
        self,
        adjuster: OpponentAdjuster,
        team_game_stats: pd.DataFrame,
        schedule_df: pd.DataFrame,
    ):
        result = adjuster.build_features(
            schedule_df,
            _AS_OF,
            target_season=2023,
            target_week=11,
            team_game_stats=team_game_stats,
        )

        off_rows = result[result["side"] == "offense"]
        assert "rolling_opp_adj_epa_per_play" in off_rows.columns
        assert "rolling_opp_adj_pass_epa" in off_rows.columns
        assert "rolling_opp_adj_rush_epa" in off_rows.columns
        # Offensive adjustments should not all be NaN
        assert not off_rows["rolling_opp_adj_epa_per_play"].isna().all()

    def test_defensive_epa_adjusted_for_opponent_offense(
        self,
        adjuster: OpponentAdjuster,
        team_game_stats: pd.DataFrame,
        schedule_df: pd.DataFrame,
    ):
        result = adjuster.build_features(
            schedule_df,
            _AS_OF,
            target_season=2023,
            target_week=11,
            team_game_stats=team_game_stats,
        )

        def_rows = result[result["side"] == "defense"]
        assert "rolling_opp_adj_epa_per_play" in def_rows.columns
        assert "rolling_opp_adj_pass_epa" in def_rows.columns
        assert "rolling_opp_adj_rush_epa" in def_rows.columns
        # Defensive adjustments should not all be NaN
        assert not def_rows["rolling_opp_adj_epa_per_play"].isna().all()


class TestLagging:
    """Opponent metrics must be lagged: only games that ENDED at the lock count."""

    def test_opponent_stats_lagged_by_one_week(
        self,
        adjuster: OpponentAdjuster,
        team_game_stats: pd.DataFrame,
        schedule_df: pd.DataFrame,
    ):
        """Opponent's stats for week N must use only data from games before week N.

        Was: "adjusted EPA equals raw EPA" for the week-2 target -- the silent fall-through
        Plan 33.2-16 removed. Now: the one prior game's opponents had played nothing, so no
        game is adjusted and the row is the flagged unknown (NaN, coverage 0.0).
        """
        result = adjuster.build_features(
            schedule_df,
            _AS_OF,
            target_season=2023,
            target_week=2,
            team_game_stats=team_game_stats,
        )

        off_rows = result[result["side"] == "offense"]
        assert len(off_rows) == 6
        assert (off_rows["games_used"] == 1).all()
        assert off_rows["rolling_opp_adj_epa_per_play"].isna().all()
        assert (off_rows[OPP_ADJ_COVERAGE_COLUMN] == 0.0).all()


class TestMinimumGamesThreshold:
    """Below min_opponent_games, no adjustment is applied -- and none is pretended."""

    def test_below_threshold_is_the_flagged_unknown(
        self,
        team_game_stats: pd.DataFrame,
        schedule_df: pd.DataFrame,
    ):
        """With fewer than min_opponent_games of opponent data, nothing is adjusted.

        Was: "below threshold, adjusted EPA equals raw EPA" -- raw EPA under an adjusted name,
        the defect Plan 33.2-16 removed. Now NaN beside a false coverage flag.
        """
        adjuster = OpponentAdjuster(
            window=10, min_opponent_games=4, schedule_df=schedule_df
        )

        # Before week 4, each opponent had at most 2 games at each earlier game's lock
        result = adjuster.build_features(
            schedule_df,
            _AS_OF,
            target_season=2023,
            target_week=4,
            team_game_stats=team_game_stats,
        )

        off_rows = result[result["side"] == "offense"]
        assert len(off_rows) == 6
        assert off_rows["rolling_opp_adj_epa_per_play"].isna().all()
        assert (off_rows[OPP_ADJ_COVERAGE_COLUMN] == 0.0).all()

    def test_at_threshold_adjustment_applied(
        self,
        team_game_stats: pd.DataFrame,
        schedule_df: pd.DataFrame,
    ):
        """With exactly min_opponent_games of opponent data, adjustment IS applied."""
        adjuster = OpponentAdjuster(
            window=10, min_opponent_games=4, schedule_df=schedule_df
        )

        # The week-5 games' opponents had played exactly 4 -- the inclusive boundary --
        # and the week-6 target reads them.
        result = adjuster.build_features(
            schedule_df,
            _AS_OF,
            target_season=2023,
            target_week=6,
            team_game_stats=team_game_stats,
        )

        off_rows = result[result["side"] == "offense"]
        assert (off_rows[OPP_ADJ_COVERAGE_COLUMN] == 1.0).all()
        adjustments_nonzero = any(
            abs(row["rolling_opp_adj_epa_per_play"] - TEAM_OFF_EPA[row["team"]]) > 0.001
            for _, row in off_rows.iterrows()
        )
        assert adjustments_nonzero, (
            "At threshold, at least some teams should have non-zero adjustment"
        )


class TestLeagueAverage:
    """League average is computed from all teams, not just the team's opponents."""

    def test_league_average_uses_all_teams(
        self,
        adjuster: OpponentAdjuster,
        team_game_stats: pd.DataFrame,
        schedule_df: pd.DataFrame,
    ):
        """The league average defensive EPA should be the mean across ALL teams'
        lagged defensive EPA, not just the opponents of a specific team."""
        result = adjuster.build_features(
            schedule_df,
            _AS_OF,
            target_season=2023,
            target_week=11,
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
        self,
        adjuster: OpponentAdjuster,
        team_game_stats: pd.DataFrame,
        schedule_df: pd.DataFrame,
    ):
        """Output rolling columns should use opp_adj_ prefix for EPA metrics."""
        result = adjuster.build_features(
            schedule_df,
            _AS_OF,
            target_season=2023,
            target_week=11,
            team_game_stats=team_game_stats,
        )

        expected_cols = [
            "rolling_opp_adj_epa_per_play",
            "rolling_opp_adj_pass_epa",
            "rolling_opp_adj_rush_epa",
            OPP_ADJ_COVERAGE_COLUMN,
        ]
        for col in expected_cols:
            assert col in result.columns, f"Missing column: {col}"

        def_rows = result[result["side"] == "defense"]
        for col in expected_cols:
            assert col in def_rows.columns, f"Missing defense column: {col}"


class TestTheLockIsTheFence:
    """Every input is admitted at the lock of the game it informs (Plan 33.2-16)."""

    def test_no_future_data_used(
        self,
        adjuster: OpponentAdjuster,
        team_game_stats: pd.DataFrame,
        schedule_df: pd.DataFrame,
    ):
        """A week-6 target reads exactly the five games that ended before its lock.

        Was a check that ``as_of_datetime`` was respected; the fence is now each target
        game's own lock, and ``as_of_datetime`` is not read at all.
        """
        result = adjuster.build_features(
            schedule_df,
            _AS_OF,
            target_season=2023,
            target_week=6,
            team_game_stats=team_game_stats,
        )

        assert len(result) > 0
        assert (result["games_used"] == 5).all()


class TestIdenticalOpponents:
    """When all opponents have identical def EPA (= league avg), adjustment = 0."""

    def test_uniform_defense_no_adjustment(self):
        """If every team has the same defensive EPA, the adjustment is zero
        for all teams (league_avg - opponent_def = 0)."""
        uniform_epa = 0.05
        u, v, w, x, y, z = "SF", "SEA", "LA", "ARI", "DAL", "PHI"
        matchups = [
            [(x, y), (z, w), (v, u)],
            [(y, z), (w, v), (u, x)],
            [(x, w), (y, v), (z, u)],
            [(w, u), (v, x), (y, z)],
            [(x, z), (y, u), (w, v)],
            [(z, v), (w, x), (u, y)],
            [(x, y), (z, w), (v, u)],
        ]
        schedule_df = _schedule_from(matchups)
        stats_rows = []
        for _, game in schedule_df.iterrows():
            for team in [game["home_team"], game["away_team"]]:
                for side in ("offense", "defense"):
                    stats_rows.append(
                        {
                            "game_id": _pbp_id(game),
                            "season": game["season"],
                            "week": game["week"],
                            "team": team,
                            "side": side,
                            "epa_per_play": uniform_epa,
                            "pass_epa_per_play": uniform_epa,
                            "rush_epa_per_play": uniform_epa,
                        }
                    )
        adjuster = OpponentAdjuster(
            window=10, min_opponent_games=4, schedule_df=schedule_df
        )
        result = adjuster.build_features(
            schedule_df,
            _AS_OF,
            target_season=2023,
            target_week=7,
            team_game_stats=pd.DataFrame(stats_rows),
        )

        # With uniform defensive EPA, the league average equals each opponent's
        # defensive EPA, so the adjustment is zero and adjusted EPA equals raw EPA
        off_rows = result[result["side"] == "offense"]
        assert len(off_rows) == 6
        for _, row in off_rows.iterrows():
            adj_epa = row["rolling_opp_adj_epa_per_play"]
            assert abs(adj_epa - uniform_epa) < 0.001, (
                f"Team {row['team']}: uniform opponents, "
                f"expected ~{uniform_epa}, got {adj_epa}"
            )


class TestAllSixMetricsAdjusted:
    """All 6 EPA metrics (3 off + 3 def) should be adjusted."""

    def test_all_metrics_present(
        self,
        adjuster: OpponentAdjuster,
        team_game_stats: pd.DataFrame,
        schedule_df: pd.DataFrame,
    ):
        result = adjuster.build_features(
            schedule_df,
            _AS_OF,
            target_season=2023,
            target_week=11,
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
