"""CR-02 regression: winsorization must never clip a discrete indicator column.

``handle_missing_data_and_outliers`` winsorizes every numeric column at the 1st
and 99th percentiles. For a continuous measurement that removes implausible
extremes. For an INDICATOR -- a coverage flag, a sign, a boolean game-context
marker -- it does something else entirely: when the minority level is rarer than
1%, q01 and q99 are both the modal level, so the clip overwrites every minority
row with the majority level and leaves a constant.

The concrete defect the Phase-29 review found: on a single covered season the
uncovered fraction is far below 1% (2023 is 271/272 covered per
``tests/integration/test_gold_line_movement_columns.py``), so
``python -m scripts.build_features --season 2023`` produced gold in which
``line_movement_coverage`` was a constant 1.0 -- every unmeasurable game stamped
MEASURED. That is precisely the fabrication the WR-10 neutral-fill guard was
written to prevent, arriving through a different door, and it additionally
destroyed the column's variance before ``expanding_normalize`` saw it.

It was invisible because the guard sat inside ``if original_missing > 0:`` and
ended in ``continue``: in the normal case the builder emits a row for every game,
so the guard never ran. The full-history rebuild that produced the published gold
happened to be safe (q01 = 0.0 at ~13% uncovered), which is why every committed
Phase-29 number is unaffected.

These tests pin the corrected contract, including the positive controls that stop
it being satisfied by simply disabling winsorization.
"""

import numpy as np
import pandas as pd
import pytest

from scripts.build_features import FeatureMatrixBuilder

# Roughly one NFL season including playoffs. The uncovered count is chosen to sit
# BELOW the 1st percentile, which is the whole trigger condition.
_SEASON_ROWS = 285
_UNCOVERED = 2


@pytest.fixture(scope="module")
def builder() -> FeatureMatrixBuilder:
    return FeatureMatrixBuilder()


def _single_season_frame(**extra_columns) -> pd.DataFrame:
    """One covered season: 285 games, 2 of them without a trajectory.

    2 / 285 == 0.7%, i.e. strictly inside the 1% winsorization tail -- the exact
    shape of a real ``--season 2023`` build.
    """
    coverage = np.ones(_SEASON_ROWS)
    coverage[:_UNCOVERED] = 0.0

    frame = pd.DataFrame(
        {
            "game_id": [f"2023_{i:03d}" for i in range(_SEASON_ROWS)],
            "season": 2023,
            "week": (np.arange(_SEASON_ROWS) % 18) + 1,
            "home_team": "KC",
            "away_team": "DET",
            "line_movement_coverage": coverage,
        }
    )
    for name, values in extra_columns.items():
        frame[name] = values
    return frame


class TestTheCoverageFlagSurvivesASingleSeasonBuild:
    """The named CR-02 defect, on the invocation WR-10 deliberately enabled."""

    def test_a_rare_uncovered_game_is_not_stamped_covered(self, builder) -> None:
        """``--season 2023`` must not collapse line_movement_coverage to 1.0.

        Pre-fix this asserted frame came back with q01 == q99 == 1.0 and every
        zero clipped away, so ``nunique()`` was 1 and the flag lied about 2 games.
        """
        out = builder.handle_missing_data_and_outliers(_single_season_frame())

        assert out["line_movement_coverage"].nunique() > 1, (
            "line_movement_coverage was winsorized to a constant -- every "
            "uncovered game is now stamped COVERED (CR-02)"
        )
        assert (out["line_movement_coverage"] == 0.0).sum() == _UNCOVERED
        assert (out["line_movement_coverage"] == 1.0).sum() == _SEASON_ROWS - _UNCOVERED

    def test_the_full_history_shape_is_also_preserved(self, builder) -> None:
        """The safe case must stay safe: ~13% uncovered, nothing clipped.

        This is the configuration that produced the published gold, so it doubles
        as a guard that the fix did not move any committed Phase-29 number.
        """
        coverage = np.ones(2000)
        coverage[:260] = 0.0  # 13%, as in the real 2018-2024 matrix
        frame = _single_season_frame()
        frame = pd.concat([frame] * 8, ignore_index=True).head(2000)
        frame["game_id"] = [f"G{i:05d}" for i in range(len(frame))]
        frame["line_movement_coverage"] = coverage

        out = builder.handle_missing_data_and_outliers(frame)

        assert (out["line_movement_coverage"] == 0.0).sum() == 260

    def test_a_rare_boolean_game_context_flag_is_also_protected(self, builder) -> None:
        """The defect is general to rare binary flags, not specific to coverage.

        ``saturday_game`` is true for a handful of late-season games; under the
        old code every one of them was clipped to 0.0 and the feature became
        uniformly false.
        """
        saturday = np.zeros(_SEASON_ROWS)
        saturday[:2] = 1.0

        out = builder.handle_missing_data_and_outliers(
            _single_season_frame(saturday_game=saturday)
        )

        assert (out["saturday_game"] == 1.0).sum() == 2

    def test_a_ternary_direction_column_keeps_all_three_levels(self, builder) -> None:
        """``total_drift_dir`` is a SIGN in {-1, 0, 1}, not a measurement."""
        direction = np.zeros(_SEASON_ROWS)
        direction[:1] = -1.0
        direction[1:2] = 1.0

        out = builder.handle_missing_data_and_outliers(
            _single_season_frame(total_drift_dir=direction)
        )

        assert set(out["total_drift_dir"].unique()) == {-1.0, 0.0, 1.0}


class TestContinuousColumnsAreStillWinsorized:
    """Positive controls: the fix must not amount to disabling winsorization."""

    def test_a_continuous_outlier_is_still_clipped(self, builder) -> None:
        """A four-sigma value in a genuine measurement must still be pulled in."""
        rng = np.random.default_rng(29)
        values = rng.normal(44.0, 2.0, _SEASON_ROWS)
        values[0] = 500.0

        out = builder.handle_missing_data_and_outliers(
            _single_season_frame(opening_total=values)
        )

        assert out["opening_total"].max() < 100.0

    def test_the_continuous_line_movement_family_is_not_exempted(self, builder) -> None:
        """``spread_drift`` is a continuous measurement and stays winsorized.

        The published readout's argument about what the model saw rests on the
        line-movement drift columns being clipped, so exempting the whole family
        by name -- rather than exempting indicators by value -- would silently
        falsify a committed claim.
        """
        rng = np.random.default_rng(4)
        values = rng.normal(0.0, 1.5, _SEASON_ROWS)
        values[0] = 60.0

        out = builder.handle_missing_data_and_outliers(
            _single_season_frame(spread_drift=values)
        )

        assert out["spread_drift"].max() < 20.0


class TestMissingHandlingAndWinsorizationAreIndependent:
    """The old ``continue`` coupled two unrelated decisions."""

    def test_a_continuous_column_is_winsorized_whether_or_not_it_had_gaps(
        self, builder
    ) -> None:
        """Same column, same outlier; the presence of a NaN must not change it.

        Pre-fix, a line-movement column with ANY missing value took the WR-10
        branch and hit ``continue``, skipping winsorization entirely -- so the
        same column was winsorized or not depending on whether a gap happened to
        exist upstream. That made the treatment of the published gold an accident
        of data completeness.
        """
        rng = np.random.default_rng(29)
        clean = rng.normal(0.0, 1.5, _SEASON_ROWS)
        clean[0] = 60.0
        gapped = clean.copy()
        gapped[5] = np.nan

        without_gap = builder.handle_missing_data_and_outliers(
            _single_season_frame(spread_drift=clean)
        )
        with_gap = builder.handle_missing_data_and_outliers(
            _single_season_frame(spread_drift=gapped)
        )

        assert without_gap["spread_drift"].max() == pytest.approx(
            with_gap["spread_drift"].max()
        )

    def test_a_gap_in_the_coverage_flag_still_reads_as_not_covered(
        self, builder
    ) -> None:
        """WR-10's contract is unchanged: a NaN fills to 0.0, never to the median.

        The column median of ``line_movement_coverage`` is 1.0, so a median fill
        would fabricate coverage. The neutral fill must still win, and the row
        must then survive winsorization as a zero.
        """
        coverage = np.ones(_SEASON_ROWS)
        coverage[:_UNCOVERED] = np.nan
        frame = _single_season_frame()
        frame["line_movement_coverage"] = coverage

        out = builder.handle_missing_data_and_outliers(frame)

        assert (out["line_movement_coverage"] == 0.0).sum() == _UNCOVERED
        assert out["line_movement_coverage"].isna().sum() == 0


class TestTheIndicatorPredicateItself:
    """``_is_discrete_indicator`` is a VALUE test, not a name test."""

    @pytest.mark.parametrize(
        "values,expected",
        [
            pytest.param([0.0, 1.0, 1.0, 1.0], True, id="binary-flag"),
            pytest.param([-1.0, 0.0, 1.0], True, id="ternary-sign"),
            pytest.param([1.0, 1.0, 1.0], True, id="constant-flag"),
            pytest.param([0.0, 1.0, np.nan], True, id="binary-with-nan"),
            pytest.param([0.0, 0.5, 1.0], False, id="continuous-in-unit-range"),
            pytest.param([44.0, 44.5, 45.0], False, id="a-totals-line"),
            pytest.param([-2.0, 0.0, 1.0], False, id="outside-indicator-levels"),
            pytest.param([np.nan, np.nan], False, id="all-null"),
        ],
    )
    def test_predicate(self, values, expected) -> None:
        assert (
            FeatureMatrixBuilder._is_discrete_indicator(pd.Series(values)) is expected
        )
