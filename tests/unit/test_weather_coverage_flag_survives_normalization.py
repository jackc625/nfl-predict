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
from features.point_in_time_fill import imputation_game_timing
from features.weather import (
    WEATHER_COVERAGE_COLUMN,
    WEATHER_FEATURE_COLUMNS_BY_BUILDER,
    WEATHER_FLAG_COLUMNS,
)
from scripts.build_features import FeatureMatrixBuilder
from tests.normalization_locks import row_order_locks
from tests.phase33_state import GOLD_LEVEL_PRESERVED_COLUMNS_33_14


def _normalize(frame, **kwargs):
    """``expanding_normalize`` with the per-row lock p332_ extra step 8c made mandatory.

    These tests are not about lock ORDERING -- ``tests/unit/test_normalization_lock_order.py``
    is -- so they are given a lock that REPRODUCES the window they already assumed: one
    distinct instant per row, in the order the function sorts the frame into. Every
    assertion below is therefore unchanged by step 8c.
    """
    kwargs.setdefault("row_locks", row_order_locks(frame, kwargs.get("sort_cols")))
    return expanding_normalize(frame, **kwargs)


# A constant weather column that is NOT the coverage flag. Control 4 rides on
# it: it travels the identical path and must still be z-scored flat, or the
# exemption is not one column wide.
CONTROL_COLUMN = "weather_severity_score"

SEASON = 2019
WEEKS = list(range(1, 9))

#: Rows of these eight-row, one-season frames that the expanding window cannot score:
#: the first ``min_periods - 1``, which have fewer than four earlier rows and no
#: prior-season bootstrap.
#:
#: P332_ EXTRA STEP 8d (owner ruling 2026-09-22) returns those rows BLANK. Was: every
#: assertion below read ``set(result[column].tolist()) == {0.0}``, which was true only
#: while an unscorable cell was written as the neutral 0.0 -- the reading this step
#: replaced, because a 0.0 there said "exactly average" about a value nothing could
#: place. The SUBJECT of each assertion is untouched: a column outside the level
#: exemption is Z-SCORED, so it never comes back at its recorded level, and every row
#: that CAN be scored is the flat 0.0 a constant column z-scores to.
UNSCORABLE_ROWS = 3


def _scored_levels(series: pd.Series) -> set[float]:
    """The distinct values of the rows that HAVE a statistic (blanks excluded)."""
    return set(series.dropna().tolist())


def _blank_rows(series: pd.Series) -> list[int]:
    """The positions returned blank, so the step-8d boundary is pinned, not waved."""
    return [position for position, blank in enumerate(series.isna()) if blank]


def _weather_only_frame() -> pd.DataFrame:
    """The merged weather frame the full builder writes, for the seam recorder."""
    columns = WEATHER_FEATURE_COLUMNS_BY_BUILDER["full"]
    return pd.DataFrame(
        [
            {"game_id": f"{SEASON}_W0{week}_AAA@BBB", **dict.fromkeys(columns, 0.0)}
            for week in WEEKS
        ]
    )


def _games_frame() -> pd.DataFrame:
    """The eight games these frames describe, one kickoff a week apart.

    p332_ EXTRA STEP 8c: ``normalize_combined_features`` orders every expanding
    statistic by each game's LOCK and refuses a frame it cannot time, so the fixture
    now carries the ``game_id`` and ``kickoff_et`` the one lock rule reads. One game a
    week, so the lock order IS the week order these assertions already assumed.
    """
    return pd.DataFrame(
        {
            "game_id": [f"{SEASON}_W0{week}_AAA@BBB" for week in WEEKS],
            "kickoff_et": pd.to_datetime(
                [f"{SEASON}-09-{week + 7:02d} 17:00:00+00:00" for week in WEEKS],
                utc=True,
            ),
        }
    )


def _builder_with_merged_weather() -> FeatureMatrixBuilder:
    builder = FeatureMatrixBuilder()
    resolved = builder.record_missing_preserving_columns(_weather_only_frame())
    assert resolved == "full", (
        "the builder key is RESOLVED from the merged frame's own columns; a "
        "resolution to the compressed builder would make the rest of this "
        "module assert about the wrong set"
    )
    builder.imputation_timing = imputation_game_timing(_games_frame())
    return builder


def _combined_frame(coverage: list[float]) -> pd.DataFrame:
    """One season, eight weeks, the flag under test and one control column."""
    return pd.DataFrame(
        {
            "game_id": [f"{SEASON}_W0{week}_AAA@BBB" for week in WEEKS],
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

        assert _scored_levels(result[CONTROL_COLUMN]) == {0.0}, (
            "a constant column that is NOT the coverage flag must keep today's "
            "behaviour exactly"
        )
        assert _blank_rows(result[CONTROL_COLUMN]) == list(range(UNSCORABLE_ROWS))


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
        result = _normalize(frame.copy(), feature_cols=[CONTROL_COLUMN])
        assert _scored_levels(result[CONTROL_COLUMN]) == {0.0}, (
            "the expanding std of a constant column is zero, so safe_std clips "
            "to 1e-8 and (v - v) / 1e-8 is 0.0. That is the whole mechanism"
        )
        assert _blank_rows(result[CONTROL_COLUMN]) == list(range(UNSCORABLE_ROWS))

    def test_without_the_parameter_the_flag_itself_still_flattens_to_zero(
        self,
    ) -> None:
        """CONTROL 4 at the lower layer: the new behaviour is OPT-IN.

        Called directly, with no ``preserve_level_cols``, the normalizer treats
        the coverage flag exactly as it always did -- so every existing caller
        is byte-preserved and only the gold build's named opt-in changes.
        """
        frame = _combined_frame([1.0] * 8)
        result = _normalize(frame.copy(), feature_cols=[WEATHER_COVERAGE_COLUMN])
        assert _scored_levels(result[WEATHER_COVERAGE_COLUMN]) == {0.0}
        assert 1.0 not in result[WEATHER_COVERAGE_COLUMN].tolist(), (
            "the recorded level must not survive without the opt-in"
        )

    def test_with_the_parameter_the_recorded_levels_are_returned_unchanged(
        self,
    ) -> None:
        """And a column outside the named set in the SAME call still z-scores."""
        coverage = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]
        frame = _combined_frame(coverage)
        result = _normalize(
            frame.copy(),
            feature_cols=[WEATHER_COVERAGE_COLUMN, CONTROL_COLUMN],
            preserve_level_cols=[WEATHER_COVERAGE_COLUMN],
        )
        assert result[WEATHER_COVERAGE_COLUMN].tolist() == coverage
        assert _scored_levels(result[CONTROL_COLUMN]) == {0.0}
        assert _blank_rows(result[CONTROL_COLUMN]) == list(range(UNSCORABLE_ROWS))


class TestTheExemptionCannotNARROWToNothingInSilence:
    """Code review WR-10: the mirror failure of "it cannot widen".

    The exemption is computed as a filter over the active builder's preserving set,
    keeping only ``weather_coverage``. That is correctly non-WIDEABLE. But if the
    column ever LEAVES that set -- a builder change, a renamed family tuple -- the
    generator yields an EMPTY tuple, ``expanding_normalize`` z-scores the flag back
    to 0.0 on every row, and nothing raises. That is the defect this phase spent a
    rung fixing, restored by omission, and 0.0 is the code's own word for NO
    OBSERVATION.

    ``_preserved_weather_columns`` already refuses BY NAME when its entry is missing.
    These pin the same discipline at the site that consumes it.
    """

    def test_a_full_build_whose_preserving_set_lost_the_flag_REFUSES(self) -> None:
        builder = _builder_with_merged_weather()
        assert builder.active_builder_key == "full"
        # Drop exactly the coverage flag from the recorded preserving set.
        builder.missing_preserving_columns = {
            "full": tuple(
                column
                for column in builder.missing_preserving_columns["full"]
                if column != WEATHER_COVERAGE_COLUMN
            )
        }

        with pytest.raises(ValueError) as excinfo:
            builder.normalize_combined_features(_combined_frame([1.0] * len(WEEKS)))

        message = str(excinfo.value)
        assert WEATHER_COVERAGE_COLUMN in message, message
        assert "NO OBSERVATION" in message, (
            "the refusal must say what the silent alternative asserts about every "
            f"game, not merely that a column is missing. Got: {message}"
        )

    def test_the_control_is_that_the_intact_set_does_NOT_refuse(self) -> None:
        """A refusal that fires on the normal case is worse than no refusal."""
        builder = _builder_with_merged_weather()

        out = builder.normalize_combined_features(_combined_frame([1.0] * len(WEEKS)))

        assert (out[WEATHER_COVERAGE_COLUMN] == 1.0).all()


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


class TestTheExemptionIsAPredicateWithAPinnedResolvedSet:
    """The one-column filter became a PREDICATE (Plan 33-14 Task 3, D33-34(a)).

    KIND: unit, through the REAL ``normalize_combined_features`` call site. No
    I/O -- the live-gold both-directions check is a Task-3 ``<verify>`` command,
    because this module reads no data store and that constraint is worth keeping.

    WHAT CHANGED AND WHY. The owner ruled on 2026-09-14 that
    ``.planning/WINDOWS.md`` rows 33 and 40 -- both marked "OWNER DECISION DUE
    BEFORE ANY RE-FIT" -- are TAKEN before Wave 15's re-fit rather than carried
    past it. Row 33: nineteen weather yes/no flags reach gold z-scored into many
    distinct decimals, ``is_snow`` into 274 of them and ``wind_moderate`` into
    5,667, so the same snowy game reads differently in week 3 than in week 15.
    Row 40: six sibling ``*_coverage`` flags carry the mirror defect, where an
    uncovered row reads far CLOSER to the covered level than to the uncovered
    one.

    THE TWO ARMS CARRY DIFFERENT RULES, AND THAT IS THE POINT.
    A predicate that widens a REFUSAL fails safe. This one widens an EXEMPTION,
    so a column it wrongly selects silently stops being normalized and reaches
    the re-fit as a raw level -- threat T-33-114. So:

    * the NAME arm (``*_coverage``) has NO varying-levels requirement, because a
      coverage flag's canonical state is CONSTANT and that constant being
      z-scored to 0.0 was the entire Phase-33.1 defect;
    * the VALUE arm DOES require more than one level, because without it the
      predicate selects ``precip_prob``, ``raw_precip_prob`` and
      ``extreme_weather`` -- three columns nobody declared, two of them
      continuous PROBABILITIES that are "discrete" only by accident of today's
      corpus.
    """

    def test_the_pinned_set_is_twenty_six_names_with_no_duplicate(self) -> None:
        """The declaration's own shape, checked before anything rests on it."""
        pinned = list(GOLD_LEVEL_PRESERVED_COLUMNS_33_14)
        assert len(pinned) == 26, f"expected 26 pinned names, got {len(pinned)}"
        assert len(set(pinned)) == len(pinned), "the pinned set has a duplicate"

        coverage = [name for name in pinned if name.endswith("_coverage")]
        assert len(coverage) == 7, (
            "weather_coverage plus the six siblings WINDOWS row 40 names. Got: "
            f"{sorted(coverage)}"
        )
        assert WEATHER_COVERAGE_COLUMN in coverage
        assert len(pinned) - len(coverage) == 19, (
            "the nineteen weather indicator flags WINDOWS row 33 names"
        )

    def test_a_varying_weather_flag_comes_back_at_its_recorded_levels(self) -> None:
        """ROW 33, the defect itself: a two-level flag must not become a decimal.

        ``is_snow`` is the measured worst case in the ledger -- 274 distinct
        values in gold for a column whose only honest answers are yes and no.
        """
        builder = _builder_with_merged_weather()
        levels = [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]
        frame = _combined_frame([1.0] * len(WEEKS))
        frame["is_snow"] = levels

        result = builder.normalize_combined_features(frame)

        assert result["is_snow"].tolist() == levels, (
            "a weather indicator flag's LEVELS are its meaning. Measured in gold "
            "before this fix: is_snow carried 274 distinct values"
        )

    def test_a_sibling_coverage_column_comes_back_at_its_recorded_levels(self) -> None:
        """ROW 40: the six siblings are caught by the NAME arm, not by their values.

        ``home_injury_coverage`` read a constant 0.0 across 2002-2008 -- which
        are genuinely UNCOVERED -- while 2009-2024 ranged -15.97 to +0.207, so an
        uncovered row sat far closer to the covered level than to the uncovered
        one.
        """
        builder = _builder_with_merged_weather()
        levels = [1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]
        frame = _combined_frame([1.0] * len(WEEKS))
        frame["home_injury_coverage"] = levels

        result = builder.normalize_combined_features(frame)

        assert result["home_injury_coverage"].tolist() == levels, (
            "a *_coverage column is caught by the NAME arm. Its levels are its "
            "meaning whatever its values happen to be"
        )

    def test_a_continuous_weather_measurement_is_still_z_scored(self) -> None:
        """THE FALSE-POSITIVE CONTROL. The predicate must not swallow a measurement.

        ``temp_f`` is a real continuous reading. A widening exemption that took
        it would send an un-normalized temperature into the re-fit, which is the
        direction that fails UNSAFE.
        """
        builder = _builder_with_merged_weather()
        frame = _combined_frame([1.0] * len(WEEKS))
        frame["temp_f"] = [31.0, 44.5, 58.2, 62.0, 70.1, 48.8, 35.4, 55.0]

        result = builder.normalize_combined_features(frame)

        assert result["temp_f"].tolist() != frame["temp_f"].tolist(), (
            "a continuous weather measurement must still be normalized; the "
            "exemption is for columns whose LEVELS are their meaning"
        )

    def test_a_single_level_indicator_is_still_z_scored(self) -> None:
        """THE T-33-114 GUARD, and the clause that makes the predicate resolve to 26.

        Without the varying-levels clause the value arm selects three columns
        nobody declared. On today's corpus ``precip_prob`` and
        ``raw_precip_prob`` each carry exactly ONE non-null value, so they
        satisfy ``_is_discrete_indicator`` while being continuous PROBABILITIES
        -- and the exemption would FLICKER between generations, because a 2026
        live forecast supplying real probabilities makes them continuous again
        and silently drops them back out.

        ``extreme_weather`` is used here because it makes the clause OBSERVABLE:
        a constant 1.0 that came back 1.0 would prove the column was preserved,
        and a constant 1.0 that comes back 0.0 proves it was normalized.

        THE COST IS RECORDED RATHER THAN HIDDEN: a genuinely level-bearing flag
        that happens to be CONSTANT is not preserved by the value arm. On today's
        corpus this is a no-op -- ``extreme_weather`` is a constant 0.0 in silver
        and a constant 0.0 in gold -- and the coverage columns, where constancy
        is the canonical case, are caught by the NAME arm instead. If such a flag
        ever gained a second level the predicate would select it, the pinned-set
        equality would fail loudly, and it would need a declaration. That is the
        safe direction for a widening exemption.
        """
        builder = _builder_with_merged_weather()
        frame = _combined_frame([1.0] * len(WEEKS))
        frame["extreme_weather"] = [1.0] * len(WEEKS)

        result = builder.normalize_combined_features(frame)

        assert _scored_levels(result["extreme_weather"]) == {0.0}, (
            "a SINGLE-level indicator is not level-bearing; it is discrete only "
            "by accident of the data, and exempting it is T-33-114"
        )
        assert 1.0 not in result["extreme_weather"].tolist(), (
            "a constant 1.0 that came back 1.0 would prove the column was "
            "preserved; it must be normalized"
        )

    def test_the_resolved_set_is_the_union_of_the_two_arms(self) -> None:
        """The predicate is the two arms and nothing else, asserted directly."""
        frame = _combined_frame([1.0] * len(WEEKS))
        frame["is_snow"] = [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0]
        frame["home_injury_coverage"] = [1.0] * len(WEEKS)
        frame["temp_f"] = [31.0, 44.5, 58.2, 62.0, 70.1, 48.8, 35.4, 55.0]
        frame["extreme_weather"] = [1.0] * len(WEEKS)
        preserved = WEATHER_FEATURE_COLUMNS_BY_BUILDER["full"]

        resolved = FeatureMatrixBuilder._level_preserved_columns(frame, preserved)

        assert set(resolved) == {
            WEATHER_COVERAGE_COLUMN,
            "home_injury_coverage",
            "is_snow",
        }, (
            "the NAME arm takes both coverage columns whatever their values; the "
            "VALUE arm takes the varying flag and refuses the constant one and "
            f"the continuous measurement. Got: {sorted(resolved)}"
        )

    def test_the_narrow_to_nothing_refusal_still_fires_under_the_predicate(
        self,
    ) -> None:
        """The widening must not have disarmed the refusal it sits beneath.

        ``TestTheExemptionCannotNARROWToNothingInSilence`` pins this for the
        pre-widening shape. It is re-asserted HERE because a predicate that can
        find ``weather_coverage`` by NAME could plausibly have been read as
        making the refusal redundant -- it is not. The refusal fires on the
        PRESERVING SET losing the flag, which is a real defect in its own right,
        and it fires BEFORE the predicate runs.
        """
        builder = _builder_with_merged_weather()
        builder.missing_preserving_columns = {
            "full": tuple(
                column
                for column in builder.missing_preserving_columns["full"]
                if column != WEATHER_COVERAGE_COLUMN
            )
        }

        with pytest.raises(ValueError) as excinfo:
            builder.normalize_combined_features(_combined_frame([1.0] * len(WEEKS)))

        assert "NO OBSERVATION" in str(excinfo.value)


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
