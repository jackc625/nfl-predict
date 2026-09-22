"""p332_ extra step 8b: winsorization stops fitting on the season it clips.

OWNER RULING 2026-09-22 ("Skip trim, first season", deferred-items.md). A season's q01/q99
clip bounds are fitted ONLY on strictly-prior seasons. A season with no strictly-prior fit --
the earliest season, or a column's first populated season -- is left UNCLIPPED, exactly as a
degenerate bound already is; clipping starts the following season, from past seasons only.

WHY. ``handle_missing_data_and_outliers`` used to fall back to the season's OWN rows whenever
no strictly-prior slice could be fitted (``_season_fit_source``'s self-fit branch). The whole
season goes into that bound, so a week-1 value was clipped by a bound that had seen week 18.
Under the day-before lock (D33.2-01) that is post-lock information -- the residual D30-16
accepted before the lock rule existed, and the last statistic inside this method that read it.

WHAT MUST NOT CHANGE, and is asserted here beside the fix:

* a season WITH a usable strictly-prior fit clips exactly as it did before;
* the minimum-fit-points rule (``_MIN_FIT_POINTS``) is unchanged;
* the degenerate-bound skip (CR-02 generalized) is unchanged;
* the discrete-indicator exemption (CR-02) is unchanged;
* the PRE-CLIP snapshot semantics are unchanged (a season's bound never reads an
  already-clipped earlier season).

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.build_features import FeatureMatrixBuilder


@pytest.fixture
def builder() -> FeatureMatrixBuilder:
    return FeatureMatrixBuilder()


def _season_block(season: int, n: int, **columns) -> pd.DataFrame:
    """One season's rows, shaped like the combined matrix.

    Weeks run in ORDERED BLOCKS (week 1 first, week 18 last) rather than cycling, so
    "a late-season value" is literally the tail of the block and "an early week" is
    literally its head. A cycling week column would scatter week 18 through the frame
    and make the plant below unreadable.
    """
    frame = pd.DataFrame(
        {
            "game_id": [f"{season}_{i:04d}" for i in range(n)],
            "season": season,
            "week": (np.arange(n) * 18 // n) + 1,
            "home_team": "KC",
            "away_team": "DET",
        }
    )
    for name, values in columns.items():
        frame[name] = values
    return frame


def _stack(*blocks: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(blocks, ignore_index=True)


class TestASelfFittingSeasonIsNotClipped:
    """The leak itself: a late-season value must not be able to move an earlier week."""

    _N = 200
    #: Week-18 rows, at the tail of the ordered block above.
    _PLANTED = 10

    def _frame(self, planted: bool) -> pd.DataFrame:
        """One season only, so it has no strictly-prior slice at all.

        The values rise smoothly from 0 to 100 across the season. When *planted*, the
        last ten rows -- week 18 -- carry an extreme LOW value, which drags a self-fitted
        q01 from about 1.0 down to -1e6 and so stops week 1's 0.0 being clipped up to it.
        The difference is a whole unit on a column whose early values are near zero, not
        a rounding wobble.
        """
        values = np.linspace(0.0, 100.0, self._N)
        if planted:
            values[-self._PLANTED :] = -1.0e6
        return _stack(_season_block(2002, self._N, metric=values))

    def _early_weeks(self, out: pd.DataFrame) -> pd.Series:
        early = out.loc[out["week"] <= 4, "metric"]
        return early.reset_index(drop=True)

    def test_a_planted_late_season_extreme_moves_no_earlier_week(self, builder) -> None:
        """The target assertion.

        Under the self-fit branch 2002's q01/q99 came from 2002's OWN 200 rows, so
        planting ten extreme values in week 18 moved the lower bound from about 1.0 to
        -1e6 and stopped week 1's values being clipped up to it -- a week-18 value
        deciding a week-1 value. With bounds fitted on strictly-prior seasons only, 2002
        has no bound at all and is left alone, so the plant cannot reach any earlier week.
        """
        calm = builder.handle_missing_data_and_outliers(self._frame(planted=False))
        planted = builder.handle_missing_data_and_outliers(self._frame(planted=True))

        pd.testing.assert_series_equal(
            self._early_weeks(calm), self._early_weeks(planted)
        )

    def test_the_self_fitting_season_keeps_its_own_values(self, builder) -> None:
        """Left UNCLIPPED means exactly that: the input values come back out."""
        frame = self._frame(planted=True)
        out = builder.handle_missing_data_and_outliers(frame)

        pd.testing.assert_series_equal(
            out["metric"].reset_index(drop=True),
            frame["metric"].reset_index(drop=True),
        )

    def test_the_unclipped_season_is_recorded(self, builder) -> None:
        """The machine-readable log now records the season left UNCLIPPED, not a self-fit."""
        builder.handle_missing_data_and_outliers(self._frame(planted=True))

        assert builder.unclipped_seasons.get("metric") == [2002]
        assert not hasattr(builder, "self_fit_seasons"), (
            "the self-fit log survived the rule that removed self-fitting"
        )


class TestALateArrivingColumnIsNotClippedInItsFirstPopulatedSeason:
    """A column whose upstream source starts mid-history has the same empty prior slice."""

    _SEASONS = tuple(range(2002, 2006))
    _N = 60

    def _frame(self, tail_value: float) -> pd.DataFrame:
        blocks = []
        for season in self._SEASONS:
            if season < 2004:
                values = np.full(self._N, np.nan)
            else:
                values = np.linspace(10.0, 20.0, self._N)
                if season == 2004:
                    values[-1] = tail_value
            blocks.append(_season_block(season, self._N, late_metric=values))
        return _stack(*blocks)

    def test_its_first_populated_season_is_left_unclipped(self, builder) -> None:
        frame = self._frame(1.0e6)
        out = builder.handle_missing_data_and_outliers(frame)

        first = out.loc[out["season"] == 2004, "late_metric"].to_numpy()
        expected = frame.loc[frame["season"] == 2004, "late_metric"].to_numpy()
        np.testing.assert_array_equal(first, expected)
        # 2002 and 2003 are all-NaN for this column, so they have no prior fit either
        # and are recorded with 2004. The load-bearing entry is 2004: the column's FIRST
        # POPULATED season, which used to fit its bound on its own rows.
        assert builder.unclipped_seasons.get("late_metric") == [2002, 2003, 2004]

    def test_the_next_season_still_clips_against_it(self, builder) -> None:
        """2005 has a real strictly-prior slice, so the prior-only rule still bites."""
        frame = self._frame(20.0)
        out = builder.handle_missing_data_and_outliers(frame)

        later = out.loc[out["season"] == 2005, "late_metric"]
        assert later.max() <= 20.0
        assert 2005 not in builder.unclipped_seasons.get("late_metric", [])


class TestASeasonWithAPriorFitClipsExactlyAsBefore:
    """The half of the behaviour the ruling leaves alone."""

    _N = 200

    def _frame(self) -> pd.DataFrame:
        rng = np.random.default_rng(8102)
        later = rng.uniform(10.0, 20.0, self._N)
        later[0] = 900.0
        later[1] = -900.0
        return _stack(
            _season_block(2001, self._N, metric=rng.uniform(10.0, 20.0, self._N)),
            _season_block(2002, self._N, metric=later),
        )

    def test_the_second_seasons_outliers_are_clipped_to_the_first_seasons_bounds(
        self, builder
    ) -> None:
        out = builder.handle_missing_data_and_outliers(self._frame())
        later = out.loc[out["season"] == 2002, "metric"]

        assert later.max() <= 20.0, f"observed max {later.max()}"
        assert later.min() >= 10.0, f"observed min {later.min()}"

    def test_moving_the_second_seasons_own_values_does_not_move_its_bound(
        self, builder
    ) -> None:
        base = self._frame()
        shifted = base.copy()
        mask = shifted["season"] == 2002
        shifted.loc[mask, "metric"] = shifted.loc[mask, "metric"] + 0.0
        shifted.loc[mask & (shifted["week"] == 18), "metric"] = 5.0e5

        out_base = builder.handle_missing_data_and_outliers(base)
        out_shifted = builder.handle_missing_data_and_outliers(shifted)

        def early(frame: pd.DataFrame) -> pd.Series:
            mask = (frame["season"] == 2002) & (frame["week"] <= 4)
            return frame.loc[mask, "metric"].reset_index(drop=True)

        pd.testing.assert_series_equal(early(out_base), early(out_shifted))


class TestTheSkipsThisRuleDoesNotTouch:
    """The minimum-fit-points rule, the degenerate bound and CR-02, all unchanged."""

    def test_a_prior_slice_below_the_minimum_fit_points_is_still_not_fitted(
        self, builder
    ) -> None:
        """2002 carries 8 rows -- fewer than _MIN_FIT_POINTS -- so 2003 gets no bound."""
        assert FeatureMatrixBuilder._MIN_FIT_POINTS == 10
        later = np.linspace(10.0, 20.0, 60)
        later[0] = 900.0
        frame = _stack(
            _season_block(2002, 8, metric=np.linspace(0.0, 1.0, 8)),
            _season_block(2003, 60, metric=later),
        )

        out = builder.handle_missing_data_and_outliers(frame)

        assert out.loc[out["season"] == 2003, "metric"].max() == pytest.approx(900.0)
        assert builder.unclipped_seasons.get("metric") == [2002, 2003]

    def test_a_degenerate_prior_bound_is_still_skipped(self, builder) -> None:
        """A constant-dominated prehistory must not erase the first real season."""
        blocks = []
        rng = np.random.default_rng(2806)
        for season in range(2002, 2016):
            values = np.ones(120) if season < 2013 else rng.uniform(0.55, 1.0, 120)
            blocks.append(_season_block(season, 120, availability=values))

        out = builder.handle_missing_data_and_outliers(_stack(*blocks))

        assert out.loc[out["season"] == 2013, "availability"].nunique() > 1
        assert (out.loc[out["season"] < 2013, "availability"] == 1.0).all()

    def test_a_discrete_indicator_is_still_exempt_in_every_season(
        self, builder
    ) -> None:
        early = np.ones(200)
        early[:2] = 0.0
        frame = _stack(
            _season_block(2001, 200, saturday_game=early),
            _season_block(2002, 200, saturday_game=np.ones(200)),
        )

        out = builder.handle_missing_data_and_outliers(frame)

        assert (out.loc[out["season"] == 2001, "saturday_game"] == 0.0).sum() == 2
        assert (out.loc[out["season"] == 2002, "saturday_game"] == 1.0).all()
        assert "saturday_game" not in builder.unclipped_seasons

    def test_the_bound_is_still_fitted_on_the_pre_clip_snapshot(self, builder) -> None:
        """Season 2003's bound reads 2002's INPUT values, not 2002's clipped ones.

        2001 is narrow, so 2002 is clipped hard; if 2003's bound were fitted on the
        LIVE frame it would inherit 2002's clipped ceiling instead of its real spread.
        """
        rng = np.random.default_rng(55)
        frame = _stack(
            _season_block(2001, 200, metric=rng.uniform(0.0, 1.0, 200)),
            _season_block(2002, 200, metric=rng.uniform(100.0, 200.0, 200)),
            _season_block(2003, 200, metric=rng.uniform(100.0, 200.0, 200)),
        )

        out = builder.handle_missing_data_and_outliers(frame)
        third = out.loc[out["season"] == 2003, "metric"]

        assert third.max() > 50.0, (
            "2003's bound collapsed onto 2002's clipped values -- the pre-clip "
            f"snapshot was lost (observed max {third.max()})"
        )
