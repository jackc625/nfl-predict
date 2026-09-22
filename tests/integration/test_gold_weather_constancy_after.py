"""R5's before/after pair: the weather columns in gold now vary, and say what they measure.

WHAT IS BEING PINNED
--------------------
SPEC R5, both halves:

* "The O/U model's 17 weather features are no longer 17-of-17 constant in the
  2018-2019 window nor in the 2021-2024 holdout, stated as a MEASURED
  before/after PAIR against tests/phase33_state.GOLD_WEATHER_CONSTANCY_MEASUREMENT."
* "A NULL observation is NaN in gold alongside an explicit weather_coverage
  column, and no weather column in any gold matrix contains an imputed numeric
  stand-in."

IN ORDINARY WORDS. The weather columns in gold used to be the same number for
every game in every window that mattered -- one fabricated 65F temperature
repeated on 6,485 of 6,499 rows, with the whole band-and-score family derived
from it. They now carry real ERA5 measurements, so they vary.

THE PAIR IS STATED, NEVER REPLACED
-----------------------------------
``GOLD_WEATHER_CONSTANCY_MEASUREMENT`` is the BEFORE half and is byte-unchanged;
``GOLD_WEATHER_CONSTANCY_AFTER`` is the AFTER half and carries a ``supersedes``
key naming it. Every assertion below prints BOTH figures in its failure message,
because an acceptance criterion that is a pair cannot be read from one number.

WHY THE NO-STAND-IN PROPERTY IS ASSERTED POSITIVELY
-----------------------------------------------------
A test that only asserted "no cell equals 65.0" would pass against a column
median-filled with a NEW number -- which is SPEC prohibition 1 exactly: "a
seasonal average or a venue mean is the same defect wearing a better label". So
the assertions here are per-column NaN COUNTS, and the two populations Ruling J
distinguishes are asserted SEPARATELY:

* the 13 temperature-derived columns carry NaN for at least the 1,652 indoor
  games -- there is no outdoor temperature inside a dome, and a band with
  nothing to band has no answer;
* the 7 composite columns carry ZERO NaN among those same indoor games -- a
  genuine 0.0, because "weather reduced scoring by nothing" is a TRUE statement
  about a covered indoor game rather than a stand-in.

A test that asserted both groups the same way would prove neither.

FOUR CONTROLS
-------------
1. NON-VACUITY: every assertion reads LIVE gold and skips with a remediation
   message naming the build command when it is absent (``data/`` is gitignored).
2. THE ASSERTIONS: the pair, in both windows, with both halves printed.
3. A PLANTED VIOLATION: the four columns that are STILL constant are named and
   asserted to be exactly those four, so a regression that flattened a fifth
   fails here rather than hiding inside "mostly varying".
4. NO FALSE POSITIVE: ``raw_temp_f`` is REPORTED at the old default rather than
   asserted to be zero there -- a handful of games genuinely were 65F, and a
   test demanding zero would be asserting something untrue.

THE THIRD HALF: p332_ RUNG 4 (Plan 33.2-12). Rung 4 replaced the ERA5 observations with the
archived day-before forecasts, so the LIVE assertions below now read
``tests.phase33_state.P332_12_GOLD_WEATHER_CONSTANCY_AFTER_RUNG4`` -- recorded beside the Phase
33.1 pair, which is left byte-unchanged and still asserted as the record it is. Three facts
changed and each is stated where it bites: the forecast probability now EXISTS (so nothing is
constant any more), the coverage flag takes BOTH levels (the 56 games abroad have no
forecast), and the bulletins report WHOLE DEGREES (so a surviving 65 F default is detected as a
spike beside its neighbours rather than by a raw count).

WHAT THIS MODULE DOES NOT CLAIM (SPEC R8). That any model is more accurate. No
model was re-fit, no gate was run, and ``artifacts/latest.json`` is unchanged.
This is a statement about what the columns CONTAIN.

ASCII only, no emoji (CLAUDE.md hard constraint). Read-only: nothing here writes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import tests.phase33_state as state

REPO_ROOT = Path(__file__).resolve().parents[2]
GOLD_DIR = REPO_ROOT / "data" / "gold"
SILVER_WEATHER_FEATURES = REPO_ROOT / "data" / "silver" / "weather_features.parquet"
OU_ARTIFACT = REPO_ROOT / "artifacts" / "ou_20260326_163930" / "feature_list.json"

MATRICES = ("features_wp", "features_ats", "features_ou")

POPULATIONS = {
    "ats_train_2015_2019": tuple(range(2015, 2020)),
    "wp_ou_train_2018_2019": (2018, 2019),
    "gate_holdout_2021_2024": (2021, 2022, 2023, 2024),
    "all_2002_2025": tuple(range(2002, 2026)),
}

BUILD_COMMAND = "uv run python scripts/build_features.py --all-seasons"

#: The latest AFTER record: p332_ rung 4's, stated beside the Phase 33.1 pair.
RUNG4_AFTER = state.P332_12_GOLD_WEATHER_CONSTANCY_AFTER_RUNG4


def _gold(matrix: str) -> pd.DataFrame:
    path = GOLD_DIR / f"{matrix}.parquet"
    if not path.exists():
        pytest.skip(
            f"live gold is not present at {path} -- run `{BUILD_COMMAND}`, or "
            "ignore on a fresh checkout where data/ is legitimately empty"
        )
    return pd.read_parquet(path)


def _weather_columns() -> list[str]:
    """The weather family as it reaches gold, DERIVED from the silver table.

    The same rule the before slot used: everything ``weather_features.parquet``
    contributes except ``weather_condition`` (dropped at build time), plus
    ``venue_cold_climate``, which Ruling H puts in the venue family for
    ATTRIBUTION but which the constancy measurement has always counted here.
    Derived rather than listed, so the family and its description cannot drift.
    """
    if not SILVER_WEATHER_FEATURES.exists():
        pytest.skip(
            f"silver weather features are not present at {SILVER_WEATHER_FEATURES} "
            "-- run `uv run python scripts/build_weather.py --all-seasons`"
        )
    frame = pd.read_parquet(SILVER_WEATHER_FEATURES)
    columns = [
        column
        for column in frame.columns
        if column not in ("game_id", "season", "week", "weather_condition")
    ]
    columns.append("venue_cold_climate")
    return columns


def _ou_weather_features() -> list[str]:
    """The weather features the DEPLOYED O/U artifact consumes, read from it."""
    if not OU_ARTIFACT.exists():
        pytest.skip(f"the deployed O/U feature list is not present at {OU_ARTIFACT}")
    payload = json.loads(OU_ARTIFACT.read_text(encoding="utf-8"))
    features = (
        payload
        if isinstance(payload, list)
        else payload.get("features") or payload.get("feature_names")
    )
    weather = set(_weather_columns())
    return sorted(name for name in features if name in weather)


def _constant_columns(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    """The members of *columns* taking at most one distinct non-null value."""
    return sorted(
        column
        for column in columns
        if column in frame.columns and frame[column].nunique(dropna=True) <= 1
    )


@pytest.mark.integration
class TestTheOuSeventeenNoLongerSitAtOneValue:
    """R5's headline pair. TEST CLASS: integration / slow, reads live gold."""

    @pytest.mark.parametrize(
        ("population", "before_key"),
        [
            ("wp_ou_train_2018_2019", "ou_train_2018_2019"),
            ("gate_holdout_2021_2024", "gate_holdout_2021_2024"),
        ],
    )
    def test_fewer_than_seventeen_are_constant(
        self, population: str, before_key: str
    ) -> None:
        """BOTH figures are printed, because the acceptance is a PAIR."""
        frame = _gold("features_ou")
        ou_weather = _ou_weather_features()
        seasons = POPULATIONS[population]
        window = frame[frame["season"].isin(seasons)]

        constant = _constant_columns(window, ou_weather)
        before = state.GOLD_WEATHER_CONSTANCY_MEASUREMENT[
            "ou_weather_features_constant"
        ][before_key]

        assert len(constant) < len(ou_weather), (
            f"R5's pair for {population}: RECORDED BEFORE "
            f"{before[0]} of {before[1]} constant; MEASURED AFTER "
            f"{len(constant)} of {len(ou_weather)} constant. The whole point of "
            "the phase is that this figure moved, and it has not. Still "
            f"constant: {constant}"
        )
        assert before[0] == len(ou_weather), (
            "the recorded BEFORE half no longer reads 17-of-17, so the pair "
            "this test states is not the pair R5 names"
        )

    def test_the_only_one_still_constant_is_the_forecast_probability(self) -> None:
        """PLANTED-VIOLATION CONTROL: the residual is NAMED, not just counted.

        At Phase 33.1 the residual was ``raw_precip_prob`` alone: a FORECAST
        probability, which ERA5 reanalysis never reports. Since p332_ rung 4 the
        history IS a forecast, so the probability exists and the residual is EMPTY --
        recorded as ``(0, 17)`` in the rung-4 slot. Any constant column among the
        seventeen now would be a flattening nobody intended.
        """
        frame = _gold("features_ou")
        ou_weather = _ou_weather_features()
        window = frame[frame["season"].isin(POPULATIONS["wp_ou_train_2018_2019"])]

        constant = _constant_columns(window, ou_weather)
        recorded = RUNG4_AFTER["ou_weather_features_constant"]["ou_train_2018_2019"]
        assert (len(constant), len(ou_weather)) == recorded, (
            f"the O/U seventeen: RECORDED at rung 4 {recorded}, MEASURED "
            f"{(len(constant), len(ou_weather))}; constant now: {constant}"
        )


@pytest.mark.integration
class TestTheWholeWeatherFamilyVariesInEveryWindow:
    """The wider statement the before slot made at 45 of 46."""

    @pytest.mark.parametrize("population", sorted(POPULATIONS))
    @pytest.mark.parametrize("matrix", MATRICES)
    def test_only_the_four_named_columns_are_constant(
        self, matrix: str, population: str
    ) -> None:
        """Four at Phase 33.1, each for a reason on the record; none since rung 4.

        BEFORE: 45 of 46 constant, only ``venue_cold_climate`` varying.
        AFTER (Phase 33.1, ERA5): 4 of 47 constant, 43 varying.
        AFTER (p332_ rung 4, the day-before forecast): none constant. The expected
        set is read from the rung-4 slot; the Phase 33.1 figures stay recorded.
        """
        frame = _gold(matrix)
        columns = _weather_columns()
        window = frame[frame["season"].isin(POPULATIONS[population])]

        constant = _constant_columns(window, columns)
        expected = list(RUNG4_AFTER["populations"][population]["constant_columns"])
        before = state.GOLD_WEATHER_CONSTANCY_MEASUREMENT["populations"].get(population)
        before_text = (
            f"RECORDED BEFORE {before['constant']} of "
            f"{state.GOLD_WEATHER_CONSTANCY_MEASUREMENT['weather_columns_counted']} "
            "constant"
            if before
            else "no BEFORE figure recorded for this population"
        )

        assert constant == expected, (
            f"{matrix} / {population}: {before_text}; MEASURED AFTER "
            f"{len(constant)} of {len(columns)} constant. Expected exactly "
            f"{expected}, the rung-4 record (Phase 33.1's four are named in "
            "GOLD_WEATHER_CONSTANCY_AFTER['why_the_four_are_still_constant']). "
            f"Measured: {constant}"
        )


@pytest.mark.integration
class TestNoWeatherColumnCarriesAnImputedStandIn:
    """Asserted POSITIVELY, by NaN counts, never by the absence of one literal."""

    @pytest.mark.parametrize("matrix", MATRICES)
    def test_the_thirteen_temperature_derived_columns_are_null_for_every_dome(
        self, matrix: str
    ) -> None:
        """There is no outdoor temperature inside a dome (Ruling J)."""
        frame = _gold(matrix)
        groups = state.WEATHER_NULL_STATE_MATRIX["column_groups"]
        temperature_derived = [
            *groups["temperature"],
            *groups["temperature_impact"],
        ]
        indoor = int(state.WEATHER_NULL_STATE_MATRIX["indoor_games_gaining_nan"])

        assert len(temperature_derived) == 13
        for column in temperature_derived:
            nan_count = int(frame[column].isna().sum())
            assert nan_count >= indoor, (
                f"{matrix}.{column} carries {nan_count} NaN, fewer than the "
                f"{indoor} indoor games. A dome has no outdoor temperature, so "
                "a number here is a stand-in for a measurement that does not "
                "exist -- which a test checking only 'no cell equals 65.0' "
                "would have missed entirely"
            )

    @pytest.mark.parametrize("matrix", MATRICES)
    def test_the_seven_composites_are_a_genuine_zero_for_a_dome_not_a_null(
        self, matrix: str
    ) -> None:
        """The OTHER half of Ruling J, asserted separately so it is proven.

        'Weather reduced scoring by nothing' is a TRUE statement about a covered
        indoor game. Asserting both groups the same way would prove neither.
        """
        frame = _gold(matrix)
        composites = state.WEATHER_NULL_STATE_MATRIX["column_groups"]["composite"]
        # Since rung 4 a NULL temperature is no longer only a dome: the 56 games abroad
        # have no forecast either, and THEIR composites are rightly NULL. So the indoor
        # population is read from the applicability flag itself.
        indoor_mask = frame["weather_affects_game"] == 0.0

        assert int(indoor_mask.sum()) == int(RUNG4_AFTER["dome_or_closed_roof_rows"])
        for column in composites:
            nan_among_indoor = int(frame.loc[indoor_mask, column].isna().sum())
            assert nan_among_indoor == 0, (
                f"{matrix}.{column} is NaN for {nan_among_indoor} of the indoor "
                "games. Ruling J keeps the composites at a genuine 0.0 indoors"
            )

    def test_the_old_default_is_reported_rather_than_asserted_to_be_absent(
        self,
    ) -> None:
        """NO-FALSE-POSITIVE CONTROL. Plenty of games genuinely were 65F.

        Demanding zero rows at the old default would be asserting something
        untrue about the weather. The honest form is a bound with the before
        figure beside it.

        SINCE RUNG 4 THE BOUND IS A SPIKE TEST. The bulletins forecast WHOLE
        degrees, so 65 F is simply one of about 80 values an outdoor game takes and
        a raw count at it is normal (128 on rung-4 gold). A surviving default shows
        as a SPIKE: thousands of rows at 65 beside ordinary counts at 64 and 66. The
        bound is therefore relative to the two neighbouring degrees, and the
        distinct-value floor is a whole-degree one.
        """
        frame = _gold("features_ou")
        temps = frame["raw_temp_f"]
        at_default = int((temps == 65.0).sum())
        neighbours = max(int((temps == 64.0).sum()), int((temps == 66.0).sum()))
        distinct = int(temps.nunique())
        before = state.GOLD_WEATHER_CONSTANCY_MEASUREMENT["raw_temp_f_imputed"]

        assert at_default <= 2 * neighbours, (
            f"RECORDED BEFORE {before['rows_at_default']} of "
            f"{before['rows_total']} rows at the 65.0 default "
            f"({before['share']:.4%}); MEASURED AFTER {at_default} at 65 F against "
            f"{neighbours} at the busier neighbouring degree. A spike means the "
            "default survived"
        )
        assert distinct >= 60, (
            f"raw_temp_f takes only {distinct} distinct values. Whole-degree "
            "forecasts over 4,793 outdoor games still take dozens"
        )


@pytest.mark.integration
class TestTheCoverageFlagSaysWhatItMeasures:
    """The second half of R5: a NULL observation beside a MEANINGFUL flag."""

    @pytest.mark.parametrize("matrix", MATRICES)
    def test_the_flag_is_present_and_does_not_read_as_an_absence(
        self, matrix: str
    ) -> None:
        """Measured 0.0 on all 6,499 rows before rung 3; 1.0 on all 6,499 after.

        0.0 is the value ``_absent_observation_features`` writes to mean NO
        OBSERVATION, so the before state asserted the exact opposite of the
        truth for every game in the corpus.

        THE FLAG DOES NOT VARY, and that is recorded rather than glossed: every
        one of the 6,499 games has a real ERA5 observation behind it, so a
        coverage flag over this corpus is constant by construction. That the two
        LEVELS survive to gold distinguishable is proven where it can be --
        tests/unit/test_weather_coverage_flag_survives_normalization.py -- over
        a frame that actually contains both.
        """
        frame = _gold(matrix)
        column = state.WEATHER_COVERAGE_GOLD_COLUMN

        assert column in frame.columns
        levels = sorted(set(frame[column].dropna().tolist()))
        # SINCE RUNG 4 THE FLAG VARIES, and that is its point: the 56 games abroad have
        # no forecast, so they read 0.0 -- the value that means NO FORECAST -- and every
        # other row reads 1.0. Both levels are asserted with the absence count.
        assert levels == list(RUNG4_AFTER["coverage_levels"]), (
            f"{matrix}.{column} reads {levels}. Every game in this corpus has a "
            "real observation, so the honest value is 1.0 on every row; 0.0 is "
            "what an ABSENT observation reads, and it is what this column "
            "carried on all 6,499 rows before the rung-3 rebuild"
        )
        assert int((frame[column] == 0.0).sum()) == int(RUNG4_AFTER["absence_rows"])


@pytest.mark.integration
class TestTheAfterSlotIsThePairAndNotAReplacement:
    """CONTROL: the before half survives, and the after half says so."""

    def test_the_after_slot_names_the_before_slot(self) -> None:
        after = state.GOLD_WEATHER_CONSTANCY_AFTER
        assert after["supersedes"] == "GOLD_WEATHER_CONSTANCY_MEASUREMENT"
        assert "never 'replaces'" in after["supersedes_note"]

    def test_the_before_slot_still_carries_its_original_figures(self) -> None:
        """Byte-unchanged in the ways that matter to the pair."""
        before = state.GOLD_WEATHER_CONSTANCY_MEASUREMENT
        assert before["weather_columns_counted"] == 46
        assert before["ou_weather_features_constant"]["ou_train_2018_2019"] == (17, 17)
        assert before["ou_weather_features_constant"]["gate_holdout_2021_2024"] == (
            17,
            17,
        )
        assert before["raw_temp_f_imputed"]["rows_at_default"] == 6485
        for population in (
            "ats_train_2015_2019",
            "wp_ou_train_2018_2019",
            "gate_holdout_2021_2024",
        ):
            assert before["populations"][population]["constant"] == 45
            assert before["populations"][population]["varying"] == 1

    def test_the_denominator_divergence_is_recorded_as_both_numbers(self) -> None:
        """46 against 47, and the one column is NAMED rather than absorbed."""
        after = state.GOLD_WEATHER_CONSTANCY_AFTER
        divergence = after["divergence_from_recorded_claims"]["weather_columns_counted"]

        assert after["weather_columns_counted"] == 47
        assert "46" in divergence and "47" in divergence
        assert "weather_coverage" in divergence

    def test_the_fourteenth_null_column_is_recorded_rather_than_overwriting_the_thirteen(
        self,
    ) -> None:
        """MEASURED: raw_humidity_pct is NaN for the 1,652 domes too.

        ``WEATHER_NULL_STATE_MATRIX`` records 13, counting the temperature and
        temperature-impact groups; Ruling J's own table also marks the separate
        'humidity' group null for a covered indoor game. Both figures are
        recorded and neither is overwritten -- the same idiom the before slot
        established for its 32-against-45 denominator.
        """
        after = state.GOLD_WEATHER_CONSTANCY_AFTER
        assert after["divergence_from_recorded_claims"]["raw_humidity_pct_nan"] == 1652
        assert state.WEATHER_NULL_STATE_MATRIX["indoor_columns_gaining_nan"] == 13, (
            "the recorded figure must not be edited; the divergence is recorded beside it"
        )

        frame = _gold("features_ou")
        # Live gold since rung 4: the 1,650 domes and closed roofs plus the 56 games
        # abroad. The recorded 1,652 above is Phase 33.1's figure and stays as it is.
        assert int(frame["raw_humidity_pct"].isna().sum()) == int(
            RUNG4_AFTER["raw_humidity_pct_null_rows"]
        )

    def test_the_after_slot_does_not_claim_an_accuracy_improvement(self) -> None:
        """SPEC R8. This phase corrects a record; it does not improve a model."""
        after = state.GOLD_WEATHER_CONSTANCY_AFTER
        assert after["is_not_an_accuracy_claim"] is True
        assert after["no_model_refit"] is True
        assert after["no_gate_run"] is True
