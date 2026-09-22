"""Unit tests for TeamFormCalculator PBP-derived extensions.

Tests for three new metrics added to team_form.py:
- FEAT-15: Team-level rolling CPOE (using only non-null cpoe values)
- FEAT-20: Average drive starting field position (from yardline_100 of first play per fixed_drive)
- FEAT-21: Neutral-situation pace of play (count of neutral-situation plays per game)
- FEAT-16: Win totals prior -- documented as SKIPPED (no data source)

All new metrics use the same dynamic expanding window as existing EPA metrics.
"""

import numpy as np
import pandas as pd

from features.team_form import TeamFormCalculator

# ---------------------------------------------------------------------------
# Synthetic PBP data fixture for extension tests
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


def _build_synthetic_pbp(
    game_id: str = "SYN_2024_W05_BUF@MIA",
    season: int = 2024,
    week: int = 5,
    team: str = "BUF",
    defteam: str = "MIA",
    num_pass_plays: int = 30,
    num_rush_plays: int = 30,
    cpoe_values: list[float | None] | None = None,
    drive_starts_yardline: list[int] | None = None,
    num_drives: int = 5,
) -> pd.DataFrame:
    """Create synthetic PBP data with CPOE, drive info, and neutral situation markers.

    Args:
        game_id: Game identifier.
        season: Season year.
        week: Week number.
        team: Offensive team abbreviation.
        defteam: Defensive team abbreviation.
        num_pass_plays: Number of pass plays.
        num_rush_plays: Number of rush plays.
        cpoe_values: Optional list of CPOE values for pass plays (None = null).
            If shorter than num_pass_plays, remaining are filled with None.
        drive_starts_yardline: Yardline_100 values for the first play of each drive.
            Length should match num_drives.
        num_drives: Number of offensive drives.
    """
    total_plays = num_pass_plays + num_rush_plays
    plays_per_drive = total_plays // max(num_drives, 1)

    rows = []
    play_id = 1

    # Default CPOE: 8 valid values and 2 nulls for a 10-pass scenario
    if cpoe_values is None:
        cpoe_values = [2.5, -1.0, 3.0, 0.5, -2.0, 1.5, 4.0, -0.5, None, None]

    # Default drive starts
    if drive_starts_yardline is None:
        drive_starts_yardline = [75, 80, 65, 70, 60]

    # Generate plays organized by drives
    cpoe_idx = 0
    for drive_num in range(1, num_drives + 1):
        drive_play_count = plays_per_drive
        # Last drive gets remaining plays
        if drive_num == num_drives:
            drive_play_count = total_plays - (num_drives - 1) * plays_per_drive

        yardline = drive_starts_yardline[
            min(drive_num - 1, len(drive_starts_yardline) - 1)
        ]

        for play_in_drive in range(drive_play_count):
            is_first_play = play_in_drive == 0
            is_pass = play_id <= num_pass_plays
            play_type = "pass" if is_pass else "run"

            # Determine CPOE for pass plays
            cpoe_val = np.nan
            if is_pass and cpoe_idx < len(cpoe_values):
                cpoe_val = (
                    cpoe_values[cpoe_idx]
                    if cpoe_values[cpoe_idx] is not None
                    else np.nan
                )
                cpoe_idx += 1

            # Determine neutral situation: roughly 60% of plays
            down = 1 if play_in_drive % 3 == 0 else (2 if play_in_drive % 3 == 1 else 3)
            ydstogo = 10 if down == 1 else (7 if down == 2 else 5)
            score_diff = 3  # within 14 points
            half_seconds = 600  # more than 120

            rows.append(
                {
                    "play_id": play_id,
                    "game_id": game_id,
                    "season": season,
                    "week": week,
                    "posteam": team,
                    "defteam": defteam,
                    "play_type": play_type,
                    "epa": 0.1 if is_pass else -0.05,
                    "success": 1 if play_id % 2 == 0 else 0,
                    "cpoe": cpoe_val,
                    "yardline_100": yardline
                    if is_first_play
                    else max(yardline - play_in_drive * 5, 1),
                    "fixed_drive": drive_num,
                    "down": down,
                    "ydstogo": ydstogo,
                    "score_differential": score_diff,
                    "half_seconds_remaining": half_seconds,
                    "touchdown": 0,
                    "first_down": 1 if play_id % 3 == 0 else 0,
                }
            )
            play_id += 1

    return pd.DataFrame(rows)


def _build_synthetic_team_stats_with_new_metrics(
    team: str = "BUF",
    seasons: list[int] | None = None,
    weeks_per_season: int = 18,
) -> pd.DataFrame:
    """Create synthetic team-game stats with the three new per-game metrics.

    Includes team_cpoe, avg_drive_start_yardline, and neutral_pace alongside
    existing metrics.
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
                        # New per-game metrics
                        "team_cpoe": 1.5 + sign * 0.5 if side == "offense" else np.nan,
                        "avg_drive_start_yardline": 70.0 + week * 0.5
                        if side == "offense"
                        else np.nan,
                        "neutral_pace": 35 + week if side == "offense" else np.nan,
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Test Class: CPOE computation
# ---------------------------------------------------------------------------


class TestRollingCPOE:
    """Tests for team-level rolling CPOE metric (FEAT-15)."""

    def test_per_game_cpoe_filters_null_values(self):
        """Per-game CPOE must use mean of cpoe where cpoe is not null.

        Given PBP with 10 passes (8 with cpoe values, 2 null/sacks),
        mean uses only the 8 valid values.
        """
        cpoe_values = [2.5, -1.0, 3.0, 0.5, -2.0, 1.5, 4.0, -0.5, None, None]
        expected_mean = np.mean([2.5, -1.0, 3.0, 0.5, -2.0, 1.5, 4.0, -0.5])  # = 1.0

        pbp = _build_synthetic_pbp(
            num_pass_plays=10,
            num_rush_plays=20,
            cpoe_values=cpoe_values,
        )

        builder = TeamFormCalculator(schedule_df=_SCHEDULE)
        # The neutral_situation column needs to exist for calculate_team_game_stats
        stats = builder.calculate_team_game_stats(pbp)

        buf_offense = stats[(stats["team"] == "BUF") & (stats["side"] == "offense")]
        assert len(buf_offense) == 1
        actual_cpoe = buf_offense.iloc[0]["team_cpoe"]
        assert abs(actual_cpoe - expected_mean) < 1e-6, (
            f"Expected CPOE {expected_mean:.4f}, got {actual_cpoe:.4f}"
        )

    def test_per_game_cpoe_all_null_returns_nan(self):
        """When all cpoe values are null (e.g., all sacks), per-game CPOE should be NaN."""
        cpoe_values = [None, None, None, None, None]
        pbp = _build_synthetic_pbp(
            num_pass_plays=5,
            num_rush_plays=25,
            cpoe_values=cpoe_values,
        )

        builder = TeamFormCalculator(schedule_df=_SCHEDULE)
        stats = builder.calculate_team_game_stats(pbp)

        buf_offense = stats[(stats["team"] == "BUF") & (stats["side"] == "offense")]
        assert len(buf_offense) == 1
        assert np.isnan(buf_offense.iloc[0]["team_cpoe"]), (
            "CPOE should be NaN when all cpoe values are null"
        )

    def test_rolling_cpoe_uses_same_window_as_epa(self):
        """Rolling CPOE must use the same dynamic expanding window as existing EPA."""
        builder = TeamFormCalculator(schedule_df=_SCHEDULE)
        stats = _build_synthetic_team_stats_with_new_metrics("BUF", [2023, 2024])

        rolling = builder.calculate_rolling_averages(
            stats, target_season=2024, target_week=5
        )
        buf_off = rolling[(rolling["team"] == "BUF") & (rolling["side"] == "offense")]
        assert len(buf_off) == 1

        # rolling_cpoe should exist in output
        assert "rolling_cpoe" in rolling.columns, (
            "rolling_cpoe must be in rolling averages output"
        )
        assert not np.isnan(buf_off.iloc[0]["rolling_cpoe"]), (
            "rolling_cpoe should not be NaN when valid data exists"
        )

    def test_rolling_cpoe_appears_in_output(self):
        """rolling_cpoe must appear in build_features output DataFrame."""
        builder = TeamFormCalculator(schedule_df=_SCHEDULE)
        stats = _build_synthetic_team_stats_with_new_metrics("BUF", [2023, 2024])

        rolling = builder.calculate_rolling_averages(
            stats, target_season=2024, target_week=5
        )
        assert "rolling_cpoe" in rolling.columns


# ---------------------------------------------------------------------------
# Test Class: Drive starting field position
# ---------------------------------------------------------------------------


class TestDriveStartPosition:
    """Tests for average drive starting field position (FEAT-20)."""

    def test_avg_drive_start_from_yardline_100(self):
        """Drive start yard line computed as mean yardline_100 from first play of each fixed_drive.

        Given 5 drives with yardline_100 of [75, 80, 65, 70, 60], mean is 70.0.
        """
        drive_starts = [75, 80, 65, 70, 60]
        expected_mean = 70.0

        pbp = _build_synthetic_pbp(
            drive_starts_yardline=drive_starts,
            num_drives=5,
        )

        builder = TeamFormCalculator(schedule_df=_SCHEDULE)
        stats = builder.calculate_team_game_stats(pbp)

        buf_offense = stats[(stats["team"] == "BUF") & (stats["side"] == "offense")]
        assert len(buf_offense) == 1
        actual = buf_offense.iloc[0]["avg_drive_start_yardline"]
        assert abs(actual - expected_mean) < 1e-6, (
            f"Expected avg drive start {expected_mean:.1f}, got {actual:.4f}"
        )

    def test_drive_start_groups_by_fixed_drive(self):
        """First play of fixed_drive is identified by grouping on fixed_drive."""
        pbp = _build_synthetic_pbp(
            num_drives=3,
            drive_starts_yardline=[80, 70, 60],
        )

        builder = TeamFormCalculator(schedule_df=_SCHEDULE)
        stats = builder.calculate_team_game_stats(pbp)

        buf_offense = stats[(stats["team"] == "BUF") & (stats["side"] == "offense")]
        assert len(buf_offense) == 1
        actual = buf_offense.iloc[0]["avg_drive_start_yardline"]
        expected = np.mean([80, 70, 60])
        assert abs(actual - expected) < 1e-6

    def test_rolling_avg_drive_start_yardline_in_output(self):
        """rolling_avg_drive_start_yardline must appear in rolling averages output."""
        builder = TeamFormCalculator(schedule_df=_SCHEDULE)
        stats = _build_synthetic_team_stats_with_new_metrics("BUF", [2023, 2024])

        rolling = builder.calculate_rolling_averages(
            stats, target_season=2024, target_week=5
        )
        assert "rolling_avg_drive_start_yardline" in rolling.columns


# ---------------------------------------------------------------------------
# Test Class: Neutral-situation pace
# ---------------------------------------------------------------------------


class TestNeutralPace:
    """Tests for neutral-situation pace of play metric (FEAT-21)."""

    def test_neutral_pace_counts_neutral_plays(self):
        """Neutral-situation pace = count of plays where neutral_situation is True.

        Given synthetic PBP where neutral plays are deterministic based on down/ydstogo.
        """
        pbp = _build_synthetic_pbp(num_pass_plays=30, num_rush_plays=30)

        builder = TeamFormCalculator(schedule_df=_SCHEDULE)
        stats = builder.calculate_team_game_stats(pbp)

        buf_offense = stats[(stats["team"] == "BUF") & (stats["side"] == "offense")]
        assert len(buf_offense) == 1
        neutral_pace = buf_offense.iloc[0]["neutral_pace"]

        # Neutral pace should be a non-negative count
        assert neutral_pace >= 0, (
            f"Neutral pace should be non-negative, got {neutral_pace}"
        )
        # With our synthetic data, we should have some neutral plays
        assert neutral_pace > 0, "Expected at least some neutral-situation plays"

    def test_neutral_pace_zero_when_no_neutral_plays(self):
        """When no plays meet neutral situation criteria, neutral_pace should be 0."""
        # Create PBP where all plays are non-neutral (down=3, which is not in [1,2])
        pbp = _build_synthetic_pbp(num_pass_plays=10, num_rush_plays=10)
        # Override down to 3 for all plays (not neutral)
        pbp["down"] = 3
        # Also need to recalculate neutral_situation since we changed down

        builder = TeamFormCalculator(schedule_df=_SCHEDULE)
        stats = builder.calculate_team_game_stats(pbp)

        buf_offense = stats[(stats["team"] == "BUF") & (stats["side"] == "offense")]
        assert len(buf_offense) == 1
        neutral_pace = buf_offense.iloc[0]["neutral_pace"]
        assert neutral_pace == 0, f"Expected 0 neutral plays, got {neutral_pace}"

    def test_rolling_neutral_pace_in_output(self):
        """rolling_neutral_pace must appear in rolling averages output."""
        builder = TeamFormCalculator(schedule_df=_SCHEDULE)
        stats = _build_synthetic_team_stats_with_new_metrics("BUF", [2023, 2024])

        rolling = builder.calculate_rolling_averages(
            stats, target_season=2024, target_week=5
        )
        assert "rolling_neutral_pace" in rolling.columns


# ---------------------------------------------------------------------------
# Test Class: All three metrics in build_features output
# ---------------------------------------------------------------------------


class TestNewMetricsInOutput:
    """Verify all three new rolling metrics appear in build_features output."""

    EXPECTED_NEW_METRICS = [
        "rolling_cpoe",
        "rolling_avg_drive_start_yardline",
        "rolling_neutral_pace",
    ]

    def test_all_new_metrics_in_rolling_output(self):
        """All three new metrics must appear in calculate_rolling_averages output."""
        builder = TeamFormCalculator(schedule_df=_SCHEDULE)
        stats = _build_synthetic_team_stats_with_new_metrics("BUF", [2023, 2024])

        rolling = builder.calculate_rolling_averages(
            stats, target_season=2024, target_week=5
        )
        assert len(rolling) > 0

        for metric in self.EXPECTED_NEW_METRICS:
            assert metric in rolling.columns, f"Missing expected new metric: {metric}"

    def test_existing_metrics_still_present(self):
        """Adding new metrics must not remove existing metrics."""
        existing_metrics = [
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

        builder = TeamFormCalculator(schedule_df=_SCHEDULE)
        stats = _build_synthetic_team_stats_with_new_metrics("BUF", [2023, 2024])

        rolling = builder.calculate_rolling_averages(
            stats, target_season=2024, target_week=5
        )
        assert len(rolling) > 0

        for metric in existing_metrics:
            assert metric in rolling.columns, f"Existing metric missing: {metric}"


# ---------------------------------------------------------------------------
# Test Class: Temporal correctness for new metrics
# ---------------------------------------------------------------------------


class TestNewMetricsTemporalCorrectness:
    """Verify new metrics use only games before as_of_datetime."""

    def test_rolling_cpoe_uses_only_prior_games(self):
        """Rolling CPOE for target week N must not include week N data."""
        builder = TeamFormCalculator(schedule_df=_SCHEDULE)
        stats = _build_synthetic_team_stats_with_new_metrics("BUF", [2024])

        rolling = builder.calculate_rolling_averages(
            stats, target_season=2024, target_week=5
        )
        buf_off = rolling[(rolling["team"] == "BUF") & (rolling["side"] == "offense")]
        assert len(buf_off) == 1

        # With no prior season, games_used should be exactly 4 (weeks 1-4)
        assert buf_off.iloc[0]["games_used"] == 4, (
            f"Expected 4 games for week 5 (no prior season), got {buf_off.iloc[0]['games_used']}"
        )


# ---------------------------------------------------------------------------
# Test Class: Early-season handling
# ---------------------------------------------------------------------------


class TestNewMetricsEarlySeason:
    """Verify early-season handling with prior-season bootstrap."""

    def test_week1_new_metrics_use_prior_season(self):
        """When fewer than min_periods games exist, metrics use prior-season bootstrap."""
        builder = TeamFormCalculator(schedule_df=_SCHEDULE)
        stats = _build_synthetic_team_stats_with_new_metrics("BUF", [2023, 2024])

        rolling = builder.calculate_rolling_averages(
            stats, target_season=2024, target_week=1
        )
        buf_off = rolling[(rolling["team"] == "BUF") & (rolling["side"] == "offense")]
        assert len(buf_off) == 1

        # Week 1 should use 8 prior-season games
        assert buf_off.iloc[0]["games_used"] == 8

        # New metrics should have valid values from prior season
        for metric in [
            "rolling_cpoe",
            "rolling_avg_drive_start_yardline",
            "rolling_neutral_pace",
        ]:
            assert metric in rolling.columns, f"Missing {metric} in Week 1 output"
            value = buf_off.iloc[0][metric]
            assert not np.isnan(value), (
                f"{metric} should not be NaN at Week 1 (prior-season data exists)"
            )


# ---------------------------------------------------------------------------
# Test Class: FEAT-16 skip documentation
# ---------------------------------------------------------------------------


class TestFeat16SkipDocumented:
    """Verify FEAT-16 (Win Totals Prior) skip is documented."""

    def test_feat16_comment_in_source(self):
        """features/team_form.py must contain FEAT-16 skip documentation."""
        import inspect

        source = inspect.getsource(TeamFormCalculator)
        assert "FEAT-16" in source, (
            "TeamFormCalculator source must contain FEAT-16 skip documentation"
        )
