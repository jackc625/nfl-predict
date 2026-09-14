"""The coverage flag must still say what it was written to say once it reaches gold.

WHAT IS BEING PINNED
--------------------
SPEC R5: a NULL observation is NaN in gold ALONGSIDE an explicit
``weather_coverage`` column. The column exists so a reader can tell "there was
no weather record" apart from "the weather was mild" -- the distinction that was
unrepresentable before Plan 33.1-02, when a missing record was filled with
``temp_f: 65.0`` AND ``is_outdoor: False`` and became indistinguishable from a
dome BY CONSTRUCTION.

THE DEFECT THIS MODULE EXISTS TO PROVE AND THEN CLOSE
-----------------------------------------------------
MEASURED on production gold after the Plan 33.1-07 rung:
``weather_coverage`` read **0.0 on all 6,499 rows**, while the silver table it
is built from read **1.0 on all 6,499 rows**.

The flag was not mis-written. It was DESTROYED IN TRANSIT, and by the one stage
nobody thought to exempt it from:

* ``features/weather.py:1207`` sets it to 1.0 for a covered row; silver
  ``weather_features.parquet`` carries 1.0 for every one of the 6,499 games.
* ``scripts/build_features.normalize_combined_features`` then z-scores it with
  every other numeric feature. For a CONSTANT column the expanding standard
  deviation is zero, so ``expanding_normalize`` clips it to ``1e-8``, and
  ``(1.0 - 1.0) / 1e-8`` is ``0.0``. Positions with no statistic at all fall
  back to the neutral ``0.0`` z-score and land on the same number.

So gold recorded, for 6,499 games that ALL have a real ERA5 observation behind
them, the exact value ``_absent_observation_features`` writes to mean NO
OBSERVATION. A constant 0.0 is not merely uninformative here; it is the wrong
answer to the question the column was added to answer.

WHY THE EXEMPTION IS FROM NORMALIZATION AND NOT FROM THE MODEL SET
-------------------------------------------------------------------
CR-02 already exempts a discrete indicator from WINSORIZATION, on exactly this
argument: "clipping one destroys the distinction it exists to encode". The same
argument reaches one stage further for a coverage flag whose levels ARE its
meaning. The narrow fix is a new ``preserve_level_cols`` parameter beside the
existing ``preserve_missing_cols`` -- the column is returned at its recorded
level -- rather than adding the flag to ``DISPLAY_ONLY_COLUMNS``, which would
also remove it from the MODEL feature set and is a decision this plan was not
asked to make.

FOUR CONTROLS
-------------
1. NON-VACUITY: the assertions run through the REAL
   ``normalize_combined_features`` call, not through a hand-called normalizer,
   so a wiring that never passes the parameter fails here.
2. THE ASSERTION: a constant 1.0 comes back as 1.0, and a VARYING flag keeps
   both of its levels -- because a fix that only rescued the constant case
   would leave the varying case silently z-scored into a continuum.
3. A PLANTED VIOLATION: a NaN coverage cell still comes back NaN, so preserving
   the LEVEL did not quietly also fabricate one.
4. NO FALSE POSITIVE: a constant NON-flag weather column is STILL z-scored to
   0.0. The exemption is one named column wide, and this control is what proves
   the tests above are asserting the exemption rather than a global change.

Nothing here reads or writes a data store. Every frame is built in memory.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from features.normalization import expanding_normalize
from features.weather import (
    WEATHER_COVERAGE_COLUMN,
    WEATHER_FEATURE_COLUMNS_BY_BUILDER,
    WEATHER_FLAG_COLUMNS,
)
from scripts.build_features import FeatureMatrixBuilder

# A constant weather column that is NOT the coverage flag. Control 4 rides on
# it: it travels the identical path and must still be z-scored flat, or the
# exemption is not one column wide.
CONTROL_COLUMN = "weather_severity_score"

SEASON = 2019
WEEKS = list(range(1, 9))


def _weather_only_frame() -> pd.DataFrame:
    """The merged weather frame the full builder writes, for the seam recorder."""
    columns = WEATHER_FEATURE_COLUMNS_BY_BUILDER["full"]
    return pd.DataFrame(
        [
            {"game_id": f"{SEASON}_W0{week}_AAA@BBB", **dict.fromkeys(columns, 0.0)}
            for week in WEEKS
        ]
    )


def _builder_with_merged_weather() -> FeatureMatrixBuilder:
    builder = FeatureMatrixBuilder()
    resolved = builder.record_missing_preserving_columns(_weather_only_frame())
    assert resolved == "full", (
        "the builder key is RESOLVED from the merged frame's own columns; a "
        "resolution to the compressed builder would make the rest of this "
        "module assert about the wrong set"
    )
    return builder


def _combined_frame(coverage: list[float]) -> pd.DataFrame:
    """One season, eight weeks, the flag under test and one control column."""
    return pd.DataFrame(
        {
            "season": [SEASON] * len(WEEKS),
            "week": WEEKS,
            WEATHER_COVERAGE_COLUMN: coverage,
            # Constant, exactly like the flag, so the ONLY difference between
            # the two outcomes is the exemption itself.
            CONTROL_COLUMN: [0.4] * len(WEEKS),
        }
    )


class TestTheFlagIsTheOneColumnItsOwnLevelsAreTheMeaningOf:
    """KIND: unit, through the REAL normalize_combined_features call site."""

    def test_a_constant_covered_flag_does_not_come_back_as_the_absent_value(
        self,
    ) -> None:
        """THE DEFECT. Silver said 1.0 on every row; gold recorded 0.0 on every row.

        This is the assertion that fails on the pre-fix tree. 0.0 is the value
        ``_absent_observation_features`` writes to mean NO OBSERVATION, so the
        z-score of a fully-covered corpus produced a column asserting the exact
        opposite of the truth for all 6,499 games.
        """
        builder = _builder_with_merged_weather()
        result = builder.normalize_combined_features(_combined_frame([1.0] * 8))

        levels = sorted(set(result[WEATHER_COVERAGE_COLUMN].tolist()))
        assert levels == [1.0], (
            "a fully-covered corpus must read COVERED in gold. Measured before "
            f"the fix: 0.0 on all 6,499 rows. Read here: {levels}"
        )

    def test_a_varying_flag_keeps_both_of_its_levels_distinguishable(self) -> None:
        """The case a constant-only fix would have left broken.

        A rescue that special-cased the degenerate column would still have
        z-scored a flag that DOES vary into a continuum whose values drift with
        the expanding window -- so a reader could no longer recover which rows
        carried an observation.
        """
        builder = _builder_with_merged_weather()
        coverage = [1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1.0, 1.0]
        result = builder.normalize_combined_features(_combined_frame(coverage))

        assert result[WEATHER_COVERAGE_COLUMN].tolist() == coverage, (
            "the flag's LEVELS are its meaning. A z-score maps them onto "
            "window-dependent numbers, which is the same destruction CR-02 "
            "already refuses for winsorization"
        )

    def test_an_absent_flag_cell_is_still_absent(self) -> None:
        """PLANTED VIOLATION: preserving the level must not fabricate one.

        ``weather_coverage`` is in the missing-preserving set already. The new
        level exemption has to compose with that rather than overwrite it.
        """
        builder = _builder_with_merged_weather()
        coverage = [1.0, 1.0, np.nan, 1.0, 1.0, 1.0, 1.0, 1.0]
        result = builder.normalize_combined_features(_combined_frame(coverage))

        assert np.isnan(result[WEATHER_COVERAGE_COLUMN].iloc[2])

    def test_a_constant_non_flag_weather_column_is_still_z_scored_flat(self) -> None:
        """CONTROL 4. The exemption is ONE named column wide.

        Without this the tests above would pass equally well against a change
        that stopped normalizing every weather column, which is a different and
        much larger decision than the one this plan makes.
        """
        builder = _builder_with_merged_weather()
        result = builder.normalize_combined_features(_combined_frame([1.0] * 8))

        assert set(result[CONTROL_COLUMN].tolist()) == {0.0}, (
            "a constant column that is NOT the coverage flag must keep today's "
            "behaviour exactly"
        )


class TestTheNormalizerIsWhatDestroysTheFlag:
    """NON-VACUITY at the lower layer: the mechanism is named and demonstrated.

    The class above asserts the OUTCOME through the real call site. This one
    shows WHERE the outcome comes from, so the diagnosis in this module's
    docstring is a demonstration rather than a claim -- and so a future reader
    who sees the exemption cannot mistake it for defensive decoration.
    """

    def test_z_scoring_a_constant_column_yields_exactly_the_absent_value(
        self,
    ) -> None:
        """A constant 1.0 in, a constant 0.0 out. No fix involved.

        This test describes the normalizer's GENERAL behaviour for any constant
        column and stays true after the fix, because the fix does not change
        the normalizer's treatment of columns outside the named exemption.
        """
        frame = _combined_frame([1.0] * 8)
        result = expanding_normalize(frame.copy(), feature_cols=[CONTROL_COLUMN])
        assert set(result[CONTROL_COLUMN].tolist()) == {0.0}, (
            "the expanding std of a constant column is zero, so safe_std clips "
            "to 1e-8 and (v - v) / 1e-8 is 0.0. That is the whole mechanism"
        )

    def test_without_the_parameter_the_flag_itself_still_flattens_to_zero(
        self,
    ) -> None:
        """CONTROL 4 at the lower layer: the new behaviour is OPT-IN.

        Called directly, with no ``preserve_level_cols``, the normalizer treats
        the coverage flag exactly as it always did -- so every existing caller
        is byte-preserved and only the gold build's named opt-in changes.
        """
        frame = _combined_frame([1.0] * 8)
        result = expanding_normalize(
            frame.copy(), feature_cols=[WEATHER_COVERAGE_COLUMN]
        )
        assert set(result[WEATHER_COVERAGE_COLUMN].tolist()) == {0.0}

    def test_with_the_parameter_the_recorded_levels_are_returned_unchanged(
        self,
    ) -> None:
        """And a column outside the named set in the SAME call still z-scores."""
        coverage = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]
        frame = _combined_frame(coverage)
        result = expanding_normalize(
            frame.copy(),
            feature_cols=[WEATHER_COVERAGE_COLUMN, CONTROL_COLUMN],
            preserve_level_cols=[WEATHER_COVERAGE_COLUMN],
        )
        assert result[WEATHER_COVERAGE_COLUMN].tolist() == coverage
        assert set(result[CONTROL_COLUMN].tolist()) == {0.0}


class TestTheColumnIsNamedOnce:
    """The single-source check, so the name cannot drift between two spellings."""

    def test_the_coverage_column_constant_is_a_member_of_the_flag_family(self) -> None:
        assert WEATHER_COVERAGE_COLUMN in WEATHER_FLAG_COLUMNS
        assert WEATHER_COVERAGE_COLUMN in WEATHER_FEATURE_COLUMNS_BY_BUILDER["full"]

    def test_the_compressed_builder_does_not_emit_the_flag(self) -> None:
        """So the wiring must derive the exemption from the MERGED frame.

        A build whose weather frame came from the compressed builder carries no
        coverage flag at all, and an exemption naming a column that is not
        present would be an assertion about a set nothing consumed.
        """
        assert (
            WEATHER_COVERAGE_COLUMN
            not in WEATHER_FEATURE_COLUMNS_BY_BUILDER["compressed"]
        )


@pytest.mark.parametrize("coverage", [[1.0] * 8, [0.0] * 8])
def test_a_constant_flag_at_either_level_is_returned_at_that_level(
    coverage: list[float],
) -> None:
    """Both degenerate cases, because both are real corpus states.

    All-covered is what the corrected corpus is. All-absent is what a build
    over a season the ingest never reached would be, and it must not be
    indistinguishable from all-covered -- which, before the fix, it was: both
    z-scored to 0.0.
    """
    builder = _builder_with_merged_weather()
    result = builder.normalize_combined_features(_combined_frame(coverage))
    assert result[WEATHER_COVERAGE_COLUMN].tolist() == coverage
