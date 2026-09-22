"""p332_ extra step 8d: a cell with NO statistic is BLANK, never the neutral zero.

OWNER RULING 2026-09-22 (deferred-items.md, "Normalization maps an early-season MEASURED
value to the neutral 0.0 when a column has no prior-season statistics", option (a) --
"LEAVE THE CELL BLANK"). Where the expanding statistic could not be formed at all, the
cell is left BLANK (NaN) instead of the neutral 0.0 z-score, beside whatever flag it
already carries. This applies to EVERY such cell, not only those whose family happens to
declare a coverage flag: whether a family declares coverage is not a reason to treat its
unscorable cells differently.

WHAT WAS WRONG. ``expanding_normalize`` ends with a fill of 0.0 for a position whose
statistic could not be formed -- fewer than ``min_periods`` admitted rows AND no usable
prior-season bootstrap. A model reads a centred 0.0 as "exactly average", so a value that
WAS measured came out of gold asserting it was perfectly ordinary. Measured on step 8c's
own gold: 307 such cells in 124 columns, 46 of them in 26 columns whose coverage flag
reads TRUE (``tests.phase33_state.P332_19_STEP8C_NEUTRAL_ZERO_*``).

THE TWO CASES MUST STAY DISTINGUISHABLE, which is what most of this module is about:

* a cell with NO statistic reads 0.0 WHATEVER its input value, so the 0.0 says nothing;
* a cell whose statistic WAS formed reads 0.0 only when its value happens to equal the
  window mean -- a real reading, and one that moves when the values move.

The second must survive untouched. It is asserted here with the same double-normalize
technique the step-8c measurement used: normalize the frame twice, the second time with a
per-row distinct offset added, and only a cell that is exactly 0.0 in BOTH runs had no
statistic.

WHAT MUST NOT CHANGE, asserted beside the fix: the prior-season bootstrap path; the
treatment of a cell whose INPUT VALUE was absent (this ruling is about a value that EXISTS
and has no statistic to be scored against, and a cell that was never measured keeps
whatever treatment it already had); ``preserve_missing_cols``; ``preserve_level_cols``;
the row set and the returned row order.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from features.normalization import expanding_normalize
from tests.normalization_locks import row_order_locks

MIN_PERIODS = 4


def _frame(values: list[float], season: int = 2024) -> pd.DataFrame:
    """One season, one metric, one row per week, in the order given."""
    return pd.DataFrame(
        {
            "game_id": [f"{season}_W{i + 1:02d}_AA@BB" for i in range(len(values))],
            "season": season,
            "week": list(range(1, len(values) + 1)),
            "metric": values,
        }
    )


def _normalize(frame: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return expanding_normalize(
        frame,
        feature_cols=["metric"],
        min_periods=MIN_PERIODS,
        row_locks=row_order_locks(frame),
        **kwargs,
    )


class TestACellWithNoStatisticIsBlank:
    """The ruling itself: no statistic and no bootstrap means BLANK, not 0.0."""

    def test_a_short_season_with_no_bootstrap_comes_back_blank(self) -> None:
        # Three rows against min_periods=4: the expanding standard deviation is
        # undefined for every one of them, and no prior_season_stats is supplied,
        # so no cell here has a statistic to be scored against.
        frame = _frame([11.0, 13.0, 17.0])

        result = _normalize(frame)

        assert result["metric"].isna().all(), (
            "every cell of a season with no statistic and no bootstrap must be "
            f"blank; got {result['metric'].tolist()}"
        )

    def test_the_early_rows_of_a_long_season_come_back_blank(self) -> None:
        # The first three rows have no statistic; the fourth is the first the
        # expanding window can score. Both halves are asserted in ONE frame, so a
        # fix that blanked the whole column would fail here.
        frame = _frame([1.0, 3.0, 5.0, 7.0, 4.0])

        result = _normalize(frame)

        assert result["metric"].iloc[:3].isna().all()
        assert result["metric"].iloc[3:].notna().all()

    def test_it_is_blank_whatever_the_value_was(self) -> None:
        # The defect this closes: the 0.0 was written WHATEVER the input value, so
        # two very different measurements read identically. Blank is the only
        # reading that does not make a claim about them.
        low = _normalize(_frame([0.5, 0.6, 0.7]))["metric"]
        high = _normalize(_frame([500.0, 600.0, 700.0]))["metric"]

        assert low.isna().all()
        assert high.isna().all()


class TestAFormedStatisticThatHappensToBeZeroSurvives:
    """The negative half: a real 0.0 reading is NOT swept up by the fix."""

    def test_a_value_equal_to_its_window_mean_still_reads_zero(self) -> None:
        # Row 5 sees rows 1-5 (mean 4.0) and its own value IS 4.0, so its honest
        # z-score is exactly 0.0. min_periods is met, so the statistic was formed.
        frame = _frame([1.0, 3.0, 5.0, 7.0, 4.0])

        result = _normalize(frame)

        assert result["metric"].iloc[4] == 0.0

    def test_the_two_kinds_of_zero_are_told_apart_by_a_shifted_rerun(self) -> None:
        # The step-8c measurement technique, run as a control: normalize twice, the
        # second time with a per-row distinct offset. A cell whose statistic was
        # FORMED moves when the values move; a cell with no statistic would read
        # 0.0 in both. After this fix no cell reads 0.0 in both runs, because the
        # second kind is blank.
        values = [1.0, 3.0, 5.0, 7.0, 4.0]
        frame = _frame(values)
        offset = [0.0, 10.0, 20.0, 30.0, 40.0]
        shifted = _frame([v + o for v, o in zip(values, offset, strict=True)])

        first = _normalize(frame)["metric"].to_numpy()
        second = _normalize(shifted)["metric"].to_numpy()

        zero_in_both = (first == 0.0) & (second == 0.0)
        assert not zero_in_both.any(), (
            "a cell reading exactly 0.0 in both runs has no statistic behind it "
            "and must be blank"
        )
        assert first[4] == 0.0, "the formed-statistic zero must survive"


class TestTheBootstrapPathIsUnchanged:
    """A prior-season bootstrap IS a usable statistic, so it still scores the cell."""

    def test_a_short_season_with_a_bootstrap_is_scored_not_blanked(self) -> None:
        frame = _frame([11.0, 13.0, 17.0])

        result = _normalize(frame, prior_season_stats={"metric": (13.0, 2.0)})

        assert result["metric"].notna().all()
        assert result["metric"].tolist() == [-1.0, 0.0, 2.0]

    def test_the_bootstrapped_zero_is_a_real_reading(self) -> None:
        # 13.0 is exactly the prior season's mean, so 0.0 here MEANS "average for
        # last season" -- a formed statistic, and the one shape of 0.0 that the
        # neutral fill was forever confused with.
        frame = _frame([11.0, 13.0, 17.0])

        result = _normalize(frame, prior_season_stats={"metric": (13.0, 2.0)})

        assert result["metric"].iloc[1] == 0.0


class TestACellWhoseValueWasAbsentKeepsItsTreatment:
    """The ruling is about a value that EXISTS. An absent one is not re-handled."""

    def test_an_absent_value_outside_preserve_missing_still_reads_zero(self) -> None:
        # Unchanged behaviour, asserted so the fix cannot quietly widen into it:
        # a column nobody named in preserve_missing_cols still gets the neutral
        # 0.0 for an input NaN, exactly as it did before step 8d.
        frame = _frame([1.0, 3.0, 5.0, 7.0, np.nan, 9.0])

        result = _normalize(frame)

        assert result["metric"].iloc[4] == 0.0

    def test_an_absent_value_inside_preserve_missing_stays_blank(self) -> None:
        frame = _frame([1.0, 3.0, 5.0, 7.0, np.nan, 9.0])

        result = _normalize(frame, preserve_missing_cols=("metric",))

        assert pd.isna(result["metric"].iloc[4])
        assert result["metric"].iloc[3] == result["metric"].iloc[3]  # not NaN

    def test_a_level_preserved_column_is_untouched(self) -> None:
        frame = _frame([1.0, 1.0, 1.0])

        result = _normalize(frame, preserve_level_cols=("metric",))

        assert result["metric"].tolist() == [1.0, 1.0, 1.0]


class TestTheShapeAndOrderAreUnchanged:
    """Blanking a value changes no row, no column and no order."""

    def test_the_frame_keeps_its_rows_columns_and_order(self) -> None:
        frame = _frame([11.0, 13.0, 17.0])

        result = _normalize(frame)

        assert list(result.columns) == list(frame.columns)
        assert result["game_id"].tolist() == frame["game_id"].tolist()
        assert len(result) == len(frame)

    def test_a_fully_scored_column_gains_no_blank(self) -> None:
        # Non-vacuity in the other direction: the fix must not introduce a NaN
        # anywhere a statistic exists.
        frame = _frame([1.0, 3.0, 5.0, 7.0, 4.0, 6.0, 2.0, 8.0])

        result = _normalize(frame, prior_season_stats={"metric": (4.0, 2.0)})

        assert result["metric"].notna().all()
