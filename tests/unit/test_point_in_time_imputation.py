"""A mid-season gap is filled from EARLIER games only (p332_ extra step 7b, owner ruling 2026-09-22).

THE LEAK. ``FeatureMatrixBuilder._impute_team_features`` filled a NaN in any ``home_*`` /
``away_*`` column with that team's mean over the WHOLE season -- weeks after the gap included --
then with the whole-season league mean, and ``_impute_game_level_features`` fell back to a
median of the gap's own whole season when no prior season could be fitted. Under D33.2-01 a
game's inputs are what was known at its lock (18:00 ET the day before kickoff), so a week-3 value
filled from weeks 4-17 is post-lock information.

THE RULE this module pins (the owner's choice, "Fill from earlier games"):

* a gap is filled with the team's mean over its games that had ENDED by the gap's lock (kickoff
  plus ``features.provenance.DECLARED_GAME_DURATION``, admitted at or before the lock -- the
  timing every lock-keyed builder uses);
* if the team has no such game, the league's mean over games ended by that lock that season;
* then the strictly-prior-seasons median (never a self-fit on the gap's own season);
* and where nothing honest exists the value stays blank, all the way through normalization --
  never a number standing in for it. The flag every model reads for it is the one D33.2-08
  item 2 names: WP's fold-fitted ``_was_missing`` indicator, and XGBoost's native missing branch.

The tests are REVEAL tests: a planted later value must not move an earlier fill, and a planted
earlier value must.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import utils.game_lock as lock_rule
from features.provenance import DECLARED_GAME_DURATION
from scripts.build_features import FeatureMatrixBuilder

ET = "America/New_York"


# ---------------------------------------------------------------------------
# Fixtures: a small, time-coherent schedule (week order = time order).
# ---------------------------------------------------------------------------


def _schedule(
    season: int = 2024,
    weeks: int = 6,
    *,
    kc_from_week: int = 1,
) -> pd.DataFrame:
    """KC hosts DET and MIA hosts BUF every Sunday at 13:00 ET (KC only from *kc_from_week*)."""
    first_sunday = pd.Timestamp(f"{season}-09-08 13:00", tz=ET)
    rows = []
    for week in range(1, weeks + 1):
        kickoff = first_sunday + pd.Timedelta(days=7 * (week - 1))
        pairs = [("MIA", "BUF")]
        if week >= kc_from_week:
            pairs.append(("KC", "DET"))
        for home, away in pairs:
            rows.append(
                {
                    "game_id": f"{season}_W{week:02d}_{away}@{home}",
                    "season": season,
                    "week": week,
                    "home_team": home,
                    "away_team": away,
                    "kickoff_et": kickoff,
                }
            )
    return pd.DataFrame(rows)


def _frame(games: pd.DataFrame, column: str = "home_metric") -> pd.DataFrame:
    """The combined-matrix shape: identifiers plus one feature column whose value is 10*week (+1 for MIA)."""
    frame = games[["game_id", "season", "week", "home_team", "away_team"]].copy()
    frame[column] = 10.0 * frame["week"] + (frame["home_team"] == "MIA").astype(float)
    return frame


def _timing(games: pd.DataFrame) -> pd.DataFrame:
    """The expected timing, derived here from the one lock rule (independent of the module)."""
    ends = (games["kickoff_et"] + DECLARED_GAME_DURATION).dt.tz_convert("UTC")
    locks = lock_rule.lock_frame(games).dt.tz_convert("UTC")
    return pd.DataFrame(
        {
            "end_ns": ends.astype("int64").to_numpy(),
            "lock_ns": locks.astype("int64").to_numpy(),
        },
        index=pd.Index(games["game_id"].astype(str), name="game_id"),
    )


def _builder(games: pd.DataFrame) -> FeatureMatrixBuilder:
    builder = FeatureMatrixBuilder()
    builder.imputation_timing = _timing(games)
    return builder


def _row(frame: pd.DataFrame, game_id: str) -> int:
    return int(frame.index[frame["game_id"] == game_id][0])


def _fill(builder: FeatureMatrixBuilder, frame: pd.DataFrame, column: str, row: int):
    return builder.handle_missing_data_and_outliers(frame).loc[row, column]


# ---------------------------------------------------------------------------
# The reveal tests: the team's own earlier games.
# ---------------------------------------------------------------------------


class TestAWithinSeasonGapReadsOnlyEarlierGames:
    def _setup(self):
        games = _schedule()
        frame = _frame(games)
        gap = _row(frame, "2024_W03_DET@KC")
        frame.loc[gap, "home_metric"] = np.nan
        return games, frame, gap

    def test_the_fill_is_the_teams_mean_over_games_ended_by_the_lock(self) -> None:
        games, frame, gap = self._setup()
        filled = _fill(_builder(games), frame, "home_metric", gap)
        # KC's weeks 1 and 2 (10.0 and 20.0) ended before the week-3 lock; weeks 4-6 did not.
        assert filled == pytest.approx(15.0)

    def test_a_later_week_value_cannot_change_the_fill(self) -> None:
        games, frame, gap = self._setup()
        before = _fill(_builder(games), frame, "home_metric", gap)
        moved = frame.copy()
        moved.loc[_row(moved, "2024_W05_DET@KC"), "home_metric"] = 9_999.0
        after = _fill(_builder(games), moved, "home_metric", gap)
        assert after == before  # byte-identical, not approximately equal

    def test_an_earlier_week_value_does_change_the_fill(self) -> None:
        games, frame, gap = self._setup()
        before = _fill(_builder(games), frame, "home_metric", gap)
        moved = frame.copy()
        moved.loc[_row(moved, "2024_W02_DET@KC"), "home_metric"] = 9_999.0
        after = _fill(_builder(games), moved, "home_metric", gap)
        assert after != before

    def test_no_value_from_a_game_after_the_lock_reaches_any_fill(self) -> None:
        # The general property: for EVERY gap, perturbing every game that ended after that gap's
        # lock leaves its fill byte-identical.
        games = _schedule()
        frame = _frame(games)
        gaps = [_row(frame, g) for g in ("2024_W02_DET@KC", "2024_W04_DET@KC")]
        frame.loc[gaps, "home_metric"] = np.nan
        timing = _timing(games)
        base = _builder(games).handle_missing_data_and_outliers(frame)
        for gap in gaps:
            lock = timing.loc[frame.loc[gap, "game_id"], "lock_ns"]
            later = frame["game_id"].map(timing["end_ns"]) > lock
            assert later.any(), "non-vacuity: some game ended after this lock"
            planted = frame.copy()
            planted.loc[later & planted["home_metric"].notna(), "home_metric"] = -1e6
            out = _builder(games).handle_missing_data_and_outliers(planted)
            assert out.loc[gap, "home_metric"] == base.loc[gap, "home_metric"]


# ---------------------------------------------------------------------------
# The fallbacks, each point-in-time.
# ---------------------------------------------------------------------------


class TestATeamWithNoEarlierGame:
    def test_it_takes_the_leagues_mean_over_games_ended_by_its_lock(self) -> None:
        games = _schedule(kc_from_week=3)
        frame = _frame(games)
        gap = _row(frame, "2024_W03_DET@KC")
        frame.loc[gap, "home_metric"] = np.nan
        filled = _fill(_builder(games), frame, "home_metric", gap)
        # Only MIA's weeks 1-2 had ended: 11.0 and 21.0.
        assert filled == pytest.approx(16.0)

    def test_a_later_league_value_cannot_change_that_fill(self) -> None:
        games = _schedule(kc_from_week=3)
        frame = _frame(games)
        gap = _row(frame, "2024_W03_DET@KC")
        frame.loc[gap, "home_metric"] = np.nan
        before = _fill(_builder(games), frame, "home_metric", gap)
        frame.loc[_row(frame, "2024_W06_BUF@MIA"), "home_metric"] = 9_999.0
        assert _fill(_builder(games), frame, "home_metric", gap) == before

    def test_with_nothing_earlier_this_season_it_takes_the_strictly_prior_median(
        self,
    ) -> None:
        prior = _schedule(2023, weeks=8)
        target = _schedule(2024)
        games = pd.concat([prior, target], ignore_index=True)
        frame = _frame(games)
        gap = _row(frame, "2024_W01_DET@KC")
        frame.loc[gap, "home_metric"] = np.nan
        filled = _fill(_builder(games), frame, "home_metric", gap)
        expected = float(frame.loc[frame["season"] == 2023, "home_metric"].median())
        assert filled == pytest.approx(expected)

    def test_the_prior_median_never_self_fits_on_the_gaps_own_season(self) -> None:
        # The pre-7b last resort fell back to a median over the gap's OWN season when no prior
        # season could be fitted -- a read of every later game that season.
        games = _schedule()
        frame = _frame(games)
        gap = _row(frame, "2024_W01_DET@KC")
        frame.loc[gap, "home_metric"] = np.nan
        out = _builder(games).handle_missing_data_and_outliers(frame)
        assert np.isnan(out.loc[gap, "home_metric"])


class TestAtLockIsAdmissible:
    """The ONE rule's operator is ``<=``: a result that existed AT the lock counts."""

    @staticmethod
    def _games(kickoff: str) -> pd.DataFrame:
        games = pd.DataFrame(
            [
                {
                    "game_id": "2024_W01_BUF@MIA",
                    "season": 2024,
                    "week": 1,
                    "home_team": "MIA",
                    "away_team": "BUF",
                    # Ends at kickoff + 4h; the KC game's lock is 18:00 ET Saturday.
                    "kickoff_et": pd.Timestamp(kickoff, tz=ET),
                },
                {
                    "game_id": "2024_W01_DET@KC",
                    "season": 2024,
                    "week": 1,
                    "home_team": "KC",
                    "away_team": "DET",
                    "kickoff_et": pd.Timestamp("2024-09-08 13:00", tz=ET),
                },
            ]
        )
        return games

    def _filled(self, kickoff: str) -> float:
        games = self._games(kickoff)
        frame = _frame(games)
        gap = _row(frame, "2024_W01_DET@KC")
        frame.loc[gap, "home_metric"] = np.nan
        return _fill(_builder(games), frame, "home_metric", gap)

    def test_a_game_that_ended_exactly_at_the_lock_is_admitted(self) -> None:
        assert self._filled("2024-09-07 14:00") == pytest.approx(11.0)

    def test_a_game_that_ended_one_minute_after_the_lock_is_not(self) -> None:
        assert np.isnan(self._filled("2024-09-07 14:01"))


class TestAGameLevelGapFollowsTheSameRule:
    def test_its_earliest_season_gap_reads_only_games_ended_by_the_lock(self) -> None:
        games = _schedule()
        frame = _frame(games, column="tempo")
        gap = _row(frame, "2024_W03_DET@KC")
        frame.loc[gap, "tempo"] = np.nan
        before = _fill(_builder(games), frame, "tempo", gap)
        # Weeks 1-2, both games: 10, 11, 20, 21.
        assert before == pytest.approx(15.5)
        frame.loc[_row(frame, "2024_W05_BUF@MIA"), "tempo"] = 9_999.0
        assert _fill(_builder(games), frame, "tempo", gap) == before


class TestAnUntimedFrameReadsNothingWithinTheSeason:
    def test_with_no_timing_a_gap_never_takes_a_within_season_statistic(self) -> None:
        # A frame whose games cannot be timed admits no within-season value (a team-game that
        # cannot be timed is never admitted -- the builders' rule). Only a strictly-prior season
        # could fill it, and there is none here.
        games = _schedule()
        frame = _frame(games)
        gap = _row(frame, "2024_W03_DET@KC")
        frame.loc[gap, "home_metric"] = np.nan
        out = FeatureMatrixBuilder().handle_missing_data_and_outliers(frame)
        assert np.isnan(out.loc[gap, "home_metric"])


# ---------------------------------------------------------------------------
# Blank means blank: through normalization, and only for this step's cells.
# ---------------------------------------------------------------------------


def _normalized(
    builder: FeatureMatrixBuilder, games: pd.DataFrame, frame: pd.DataFrame
):
    # Normalization refuses to run without a merged weather frame (the fail-closed coverage-flag
    # rule), so a minimal full-builder weather frame rides along, as in the Plan 33.2-15 tests.
    weather = pd.DataFrame(
        {"game_id": games["game_id"], "weather_coverage": 1.0, "temp_f": 60.0}
    )
    builder.record_missing_preserving_columns(weather)
    processed = builder.handle_missing_data_and_outliers(frame)
    return builder.normalize_combined_features(processed)


class TestBlankMeansBlank:
    def test_a_gap_with_no_honest_fill_stays_nan_through_normalization(self) -> None:
        games = _schedule()
        frame = _frame(games)
        frame["weather_coverage"] = 1.0
        frame["temp_f"] = 60.0
        gap = _row(frame, "2024_W01_DET@KC")
        frame.loc[gap, "home_metric"] = np.nan
        builder = _builder(games)
        normalized = _normalized(builder, games, frame)
        assert np.isnan(normalized.loc[gap, "home_metric"]), (
            "a value nobody could know at the lock came out of normalization as a number"
        )
        assert normalized["home_metric"].notna().sum() == len(frame) - 1

    def test_the_blank_register_names_exactly_that_cell(self) -> None:
        games = _schedule()
        frame = _frame(games)
        gap = _row(frame, "2024_W01_DET@KC")
        frame.loc[gap, "home_metric"] = np.nan
        builder = _builder(games)
        builder.handle_missing_data_and_outliers(frame)
        assert list(builder.imputation_left_blank["home_metric"]) == [gap]

    def test_a_whole_season_with_no_value_is_not_this_steps_cell(self) -> None:
        # Scope control: a column with NO value in a season is a coverage floor (Plan 33.2-17
        # Task 2 / rung 8's cause), not a within-season gap. Step 7b does not register it.
        games = pd.concat([_schedule(2023), _schedule(2024)], ignore_index=True)
        frame = _frame(games)
        frame.loc[frame["season"] == 2023, "home_metric"] = np.nan
        builder = _builder(games)
        builder.handle_missing_data_and_outliers(frame)
        assert "home_metric" not in builder.imputation_left_blank


class TestTheBuildRecordsTheTiming:
    def test_combine_features_times_every_game_from_its_kickoff(self) -> None:
        games = _schedule()
        builder = FeatureMatrixBuilder()
        builder.combine_features({"games": games})
        timing = builder.imputation_timing
        assert timing is not None
        assert set(timing.index) == set(games["game_id"])
        pd.testing.assert_frame_equal(timing.sort_index(), _timing(games).sort_index())

    def test_the_end_is_kickoff_plus_the_declared_duration_and_the_lock_the_one_rule(
        self,
    ) -> None:
        from features.point_in_time_fill import imputation_game_timing

        from utils.game_lock import game_lock

        games = _schedule(weeks=1)
        timing = imputation_game_timing(games)
        game = games.iloc[0]
        end = (game["kickoff_et"] + DECLARED_GAME_DURATION).tz_convert("UTC")
        lock = pd.Timestamp(game_lock(game["kickoff_et"])).tz_convert("UTC")
        assert timing.loc[game["game_id"], "end_ns"] == end.value
        assert timing.loc[game["game_id"], "lock_ns"] == lock.value
