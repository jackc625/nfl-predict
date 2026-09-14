"""Rain that FELL is a measurement, and it must not be discarded for want of a forecast.

WHAT IS BEING PINNED
--------------------
SPEC R5 and Plan 33.1-04's Ruling J: a weather column for a covered OUTDOOR game
is MEASURED, and only what genuinely cannot be answered is NULL.

THE DEFECT THIS MODULE EXISTS TO PROVE AND THEN CLOSE
-----------------------------------------------------
``WeatherFeaturesCalculator.calculate_precipitation_features`` opened with an
all-or-nothing gate::

    if _is_missing(raw_prob) or _is_missing(raw_mm):
        return dict.fromkeys(PRECIPITATION_FEATURE_COLUMNS, NAN)

``raw_prob`` is ``precip_prob``, a FORECAST probability. The corpus this phase
built is ERA5 REANALYSIS, and a reanalysis states what happened rather than what
was forecast to happen -- so the archive has no probability to give and never
will. ``COVERAGE.md`` records that as a deliberate opt-out, and
``scripts/ingest_weather.fetch_game_weather`` writes ``precip_prob: None`` for
exactly that reason.

So the gate fired on EVERY outdoor game in the corrected corpus, and threw away
``precip_mm`` -- the rainfall that ACTUALLY FELL, present on all 6,499 rows -- on
the grounds that a probability nobody can ever supply was absent.

MEASURED on production gold before the fix (``data/gold/features_ou.parquet``,
6,499 rows): ``raw_precip_mm`` non-null on 6,499 rows; ``precip_mm`` non-null on
1,652 -- exactly the indoor games, which take the dome branch and never reach
this gate. Every one of the 4,847 OUTDOOR games -- the games where rain is the
entire point -- carried NULL across the twelve precipitation columns, and the
seven composite severity columns inherited it because
``calculate_weather_severity`` reads ``precip_impact_score``.

WHAT THE FIX IS, AND WHAT IT IS NOT
------------------------------------
The fix NARROWS the gate; it does not remove it. Three properties are asserted
here in both directions, because each one is a way the fix could go wrong:

1. NO INVENTED PROBABILITY. ``precip_prob`` stays NaN when the archive gave
   none. Deriving a probability from the measurement would be the fabrication
   this phase exists to delete, wearing a statistician's hat.
2. NO RETURN TO ``or 0.0``. The old pre-33.1 code wrote
   ``weather_data.get("precip_mm", 0.0) or 0.0`` and reported an ABSENT reading
   as a dry day. With the MEASUREMENT itself absent the whole family is still
   NULL -- test ``test_an_absent_measurement_is_still_the_whole_family_null``.
3. ONLY THE GENUINELY UNANSWERABLE STAYS NULL. Rain-versus-snow needs a
   temperature; without one the three type one-hots and the impact score that
   reads them stay NaN, while the bands and the two multipliers -- which consult
   no temperature -- stay real.

A FOURTH CONTROL: THE DUAL-READING PATH IS BYTE-PRESERVED
----------------------------------------------------------
``test_a_payload_carrying_both_readings_is_byte_preserved`` pins the exact
numbers the pre-fix code produced when BOTH readings were present. Narrowing a
gate is the kind of change that quietly rewrites the branch it was not aiming
at, and the live forecast path (COLD-06) still supplies a probability.

ASCII only, no emoji (CLAUDE.md hard constraint). This module writes nothing and
needs no ``writes_production_store`` marker.
"""

from __future__ import annotations

import math

import pytest

from features.weather import (
    PRECIPITATION_FEATURE_COLUMNS,
    SEVERITY_FEATURE_COLUMNS,
    WeatherFeaturesCalculator,
)

# The shape `scripts/ingest_weather.fetch_game_weather` writes for a real ERA5
# archive row: a measured rainfall, a measured temperature, and `precip_prob`
# explicitly None because the reanalysis has no probability to give.
ERA5_OUTDOOR_PAYLOAD: dict[str, object] = {
    "game_id": "2019_W10_BUF@CLE",
    "temp_f": 45.0,
    "wind_mph": 8.0,
    "humidity_pct": 70.0,
    "precip_prob": None,
    "precip_mm": 3.2,
    "condition": "Rain",
    "is_outdoor": True,
    "weather_coverage": True,
}


def _payload(**overrides: object) -> dict[str, object]:
    """A copy of the ERA5 payload with *overrides* applied."""
    return {**ERA5_OUTDOOR_PAYLOAD, **overrides}


@pytest.fixture
def calculator() -> WeatherFeaturesCalculator:
    return WeatherFeaturesCalculator()


class TestTheMeasuredRainfallReachesTheFeatureFrame:
    """KIND: unit, over the calculator's own public method. No I/O, no data/."""

    def test_the_whole_precipitation_family_is_not_discarded_for_a_missing_forecast(
        self, calculator: WeatherFeaturesCalculator
    ) -> None:
        """THE DEFECT. An ERA5 row measures 3.2 mm of rain; gold recorded NULL.

        This is the assertion that fails on the pre-fix tree: the all-or-nothing
        gate returned every one of the twelve columns as NaN because a forecast
        probability the archive cannot supply was absent.
        """
        features = calculator.calculate_precipitation_features(ERA5_OUTDOOR_PAYLOAD)

        nulled = sorted(
            column
            for column in PRECIPITATION_FEATURE_COLUMNS
            if math.isnan(float(features[column]))
        )
        assert nulled == ["precip_prob"], (
            "the ONLY precipitation column a real ERA5 observation cannot answer "
            "is the forecast probability, which a reanalysis never reports. "
            f"NULL here: {nulled}. Measured on production gold before the fix: "
            "raw_precip_mm was non-null on all 6,499 rows while precip_mm was "
            "non-null on 1,652 -- the indoor games alone -- so the measured "
            "rainfall was discarded for all 4,847 outdoor games."
        )
        assert features["precip_mm"] == pytest.approx(3.2)

    def test_the_bands_come_from_the_millimetres_that_actually_fell(
        self, calculator: WeatherFeaturesCalculator
    ) -> None:
        """3.2 mm is the MODERATE band under the module's own mm thresholds.

        The thresholds are not new. ``0.5`` and ``2.0`` and
        ``self.heavy_precip_threshold`` are the mm cut points the dual-reading
        branch already used; the mm-only branch applies the same ones without
        the probability disjunct.
        """
        features = calculator.calculate_precipitation_features(ERA5_OUTDOOR_PAYLOAD)

        assert features["precip_none"] == 0.0
        assert features["precip_light"] == 0.0
        assert features["precip_moderate"] == 1.0
        assert features["precip_heavy"] == 0.0

    @pytest.mark.parametrize(
        ("millimetres", "expected_band"),
        [
            (0.0, "precip_none"),
            (0.4, "precip_none"),
            (1.5, "precip_light"),
            (3.2, "precip_moderate"),
            (9.0, "precip_heavy"),
        ],
    )
    def test_the_mm_only_bands_are_mutually_exclusive_across_the_range(
        self,
        calculator: WeatherFeaturesCalculator,
        millimetres: float,
        expected_band: str,
    ) -> None:
        """Exactly ONE band fires at every point of the range.

        The dual-reading branch composes its bands with ``or`` over two
        readings, so two of them can fire at once. With one reading the bands
        are an ordered partition, and a partition that overlaps would report a
        game as both light and moderate rain.
        """
        features = calculator.calculate_precipitation_features(
            _payload(precip_mm=millimetres)
        )
        bands = ("precip_none", "precip_light", "precip_moderate", "precip_heavy")
        fired = [band for band in bands if features[band] == 1.0]
        assert fired == [expected_band], (
            f"{millimetres} mm fired {fired}; exactly one band must fire"
        )

    def test_rain_versus_snow_is_decided_by_the_measured_temperature(
        self, calculator: WeatherFeaturesCalculator
    ) -> None:
        """45F with rainfall is RAIN; 30F with the same rainfall is SNOW."""
        rain = calculator.calculate_precipitation_features(ERA5_OUTDOOR_PAYLOAD)
        assert rain["is_rain"] == 1.0
        assert rain["is_snow"] == 0.0
        assert rain["is_dry"] == 0.0

        snow = calculator.calculate_precipitation_features(_payload(temp_f=30.0))
        assert snow["is_snow"] == 1.0
        assert snow["is_rain"] == 0.0

    def test_the_composite_severity_family_stops_inheriting_a_null_it_no_longer_has(
        self, calculator: WeatherFeaturesCalculator
    ) -> None:
        """The seven composites read ``precip_impact_score``.

        While the precipitation family was NULL for every outdoor game, so was
        every composite -- which is how `scoring_reduction`, `weather_game` and
        `weather_severity_score` came to be NULL on 4,847 of 6,499 gold rows.
        """
        wind = calculator.calculate_wind_features(ERA5_OUTDOOR_PAYLOAD)
        temperature = calculator.calculate_temperature_features(ERA5_OUTDOOR_PAYLOAD)
        precipitation = calculator.calculate_precipitation_features(
            ERA5_OUTDOOR_PAYLOAD
        )
        severity = calculator.calculate_weather_severity(
            wind, temperature, precipitation
        )

        nulled = sorted(
            column
            for column in SEVERITY_FEATURE_COLUMNS
            if math.isnan(float(severity[column]))
        )
        assert nulled == [], (
            "a covered outdoor game with a measured wind, temperature and "
            f"rainfall can answer every composite. NULL here: {nulled}"
        )
        assert severity["weather_severity_score"] > 0.0


class TestTheGateNarrowsRatherThanDisappears:
    """KIND: unit. The three ways the fix could go wrong, each asserted."""

    def test_no_probability_is_invented_for_a_reanalysis_that_has_none(
        self, calculator: WeatherFeaturesCalculator
    ) -> None:
        """``precip_prob`` is NaN, not a number derived from the millimetres.

        SPEC prohibition 1: the deleted default must not be replaced by any
        other number under any name. A probability inferred from an observed
        rainfall is still a fabricated forecast.
        """
        features = calculator.calculate_precipitation_features(ERA5_OUTDOOR_PAYLOAD)
        assert math.isnan(float(features["precip_prob"]))

    def test_an_absent_measurement_is_still_the_whole_family_null(
        self, calculator: WeatherFeaturesCalculator
    ) -> None:
        """With NO rainfall reading the family is NULL, exactly as before.

        This is the control that separates a NARROWED gate from a REMOVED one.
        The pre-33.1 code wrote ``or 0.0`` here and reported an absent reading
        as a dry day; the honest fix keeps refusing when the measurement itself
        is gone.
        """
        features = calculator.calculate_precipitation_features(_payload(precip_mm=None))
        assert all(
            math.isnan(float(features[column]))
            for column in PRECIPITATION_FEATURE_COLUMNS
        ), "an absent MEASUREMENT leaves the whole family NULL"

    def test_a_measured_zero_is_a_dry_day_and_is_not_an_absence(
        self, calculator: WeatherFeaturesCalculator
    ) -> None:
        """0.0 mm is a measurement. It must not read like the absent case.

        ``_is_missing`` exists precisely because ``not value`` collapses a
        measured calm onto an absent reading. The same distinction has to hold
        one level up, in the bands.

        The condition string is set to ``Clear`` here rather than inherited.
        The module has ALWAYS treated the condition text as an independent
        signal -- ``"rain" in condition`` sets ``is_rain`` on its own -- so a
        payload measuring 0.0 mm while its condition still said ``Rain`` would
        be contradictory input, and would be testing that contradiction rather
        than the measured-zero rule this test is about.
        """
        features = calculator.calculate_precipitation_features(
            _payload(precip_mm=0.0, condition="Clear")
        )
        assert features["precip_mm"] == 0.0
        assert features["precip_none"] == 1.0
        assert features["is_dry"] == 1.0
        assert features["precip_impact_score"] == 0.0

    def test_without_a_temperature_only_the_type_one_hots_and_their_score_are_null(
        self, calculator: WeatherFeaturesCalculator
    ) -> None:
        """Rain-versus-snow genuinely cannot be answered without a temperature.

        The bands and the two multipliers consult no temperature, so they stay
        real. This is the "leave null ONLY what cannot be answered" rule, made
        checkable rather than described.
        """
        features = calculator.calculate_precipitation_features(_payload(temp_f=None))

        nulled = sorted(
            column
            for column in PRECIPITATION_FEATURE_COLUMNS
            if math.isnan(float(features[column]))
        )
        assert nulled == [
            "is_dry",
            "is_rain",
            "is_snow",
            "precip_impact_score",
            "precip_prob",
        ], f"NULL here: {nulled}"
        assert features["precip_mm"] == pytest.approx(3.2)
        assert features["precip_moderate"] == 1.0
        assert features["turnover_multiplier"] > 1.0
        assert features["passing_efficiency"] < 1.0

    def test_a_payload_carrying_both_readings_is_byte_preserved(
        self, calculator: WeatherFeaturesCalculator
    ) -> None:
        """THE REGRESSION CONTROL. The dual-reading branch is untouched.

        The numbers below were computed from the PRE-FIX formulas for
        ``precip_prob=0.6, precip_mm=3.2, temp_f=45.0``. Narrowing a gate is
        exactly the kind of change that rewrites the branch it was not aiming
        at, and the live forecast path still supplies a probability.
        """
        features = calculator.calculate_precipitation_features(
            _payload(precip_prob=0.6)
        )

        assert features["precip_prob"] == pytest.approx(0.6)
        assert features["precip_mm"] == pytest.approx(3.2)
        assert features["precip_none"] == 0.0
        assert features["precip_light"] == 0.0
        assert features["precip_moderate"] == 1.0
        assert features["precip_heavy"] == 0.0
        assert features["is_rain"] == 1.0
        assert features["is_snow"] == 0.0
        assert features["is_dry"] == 0.0
        assert features["precip_impact_score"] == pytest.approx(0.6)
        assert features["turnover_multiplier"] == pytest.approx(1.384)
        assert features["passing_efficiency"] == pytest.approx(0.85)
