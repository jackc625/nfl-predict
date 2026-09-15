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
    PRECIPITATION_BAND_COLUMNS,
    PRECIPITATION_FEATURE_COLUMNS,
    PRECIPITATION_PARTITION_RULE,
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


# One representative value inside each of the four FORECAST-PROBABILITY bands
# (cut points 0.2, 0.5, 0.8) and each of the four RAINFALL bands (cut points
# 0.5, 2.0, and the calculator's heavy_precip_threshold of 5.0 mm). Paired with
# the ordinal index the module's rule assigns them, so the grid below is a
# statement about BANDS rather than about four arbitrary decimals.
PROBABILITY_BY_BAND_INDEX: tuple[tuple[float, int], ...] = (
    (0.1, 0),
    (0.4, 1),
    (0.6, 2),
    (0.9, 3),
)

RAINFALL_BY_BAND_INDEX: tuple[tuple[float, int], ...] = (
    (0.2, 0),
    (1.0, 1),
    (3.0, 2),
    (9.0, 3),
)

# The full sixteen-point cross product, flattened for parametrisation.
PROBABILITY_BY_RAINFALL_GRID: tuple[tuple[float, int, float, int], ...] = tuple(
    (probability, probability_index, millimetres, rainfall_index)
    for probability, probability_index in PROBABILITY_BY_BAND_INDEX
    for millimetres, rainfall_index in RAINFALL_BY_BAND_INDEX
)


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


class TestTheForecastBandsAreAPartitionToo:
    """The LIVE-FORECAST branch is a partition, not an overlapping disjunction.

    KIND: unit, driven through the calculator's public
    ``calculate_precipitation_features``. No I/O, no ``data/``.

    WHAT THIS CLASS ADDS, AND WHY IT IS A DIFFERENT DEFECT FROM THE ONE ABOVE.
    ``TestTheMeasuredRainfallReachesTheFeatureFrame`` is about the ARCHIVE
    branch, which Phase 33.1 repaired: it asserts that measured rainfall is not
    discarded, and its
    ``test_the_mm_only_bands_are_mutually_exclusive_across_the_range`` already
    pins the partition property THERE. This class asserts the same property on
    the branch 33.1 deliberately left byte-unchanged -- the one a live
    Open-Meteo forecast takes, and therefore the one serving 2026 predictions.

    THE DEFECT, as ``.planning/WINDOWS.md`` row 39 registers it: the
    dual-reading branch composed each band as a DISJUNCTION over two readings,
    so ``precip_light`` fired on ``0.2 < prob <= 0.5 OR 0.5 < mm <= 2.0`` and
    ``precip_moderate`` on ``0.5 < prob <= 0.8 OR 2.0 < mm <= 5.0``. A forecast
    of probability 0.4 with 3.0 mm of rain therefore set BOTH to 1.0 -- one game
    reported as two intensities at once, in a family whose whole contract is
    that exactly one member is hot.

    ROW 39 IS MARKED "OWNER DECISION DUE BEFORE ANY RE-FIT". This class is the
    assertion half of taking that decision (D33-34(b)); Plan 33-14 Task 1's
    two-half premise measurement is the other half.
    """

    def test_the_band_order_and_the_rule_are_stated_in_committed_source(self) -> None:
        """The ORDER is a declaration, not an accident of dict literal order.

        ``max(probability_index, rainfall_index)`` is meaningless unless the
        four bands are ordered, so the order has to be something a reader can
        quote and a test can check -- not something inferred from the sequence
        in which a dict happens to be written.
        """
        assert PRECIPITATION_BAND_COLUMNS == (
            "precip_none",
            "precip_light",
            "precip_moderate",
            "precip_heavy",
        ), (
            "the four bands must be declared in ASCENDING intensity order; the "
            "maximum-band-index rule reads this tuple. Got: "
            f"{PRECIPITATION_BAND_COLUMNS}"
        )
        assert all(
            band in PRECIPITATION_FEATURE_COLUMNS for band in PRECIPITATION_BAND_COLUMNS
        ), "every declared band must be a member of the emitted family"
        assert "LARGER" in PRECIPITATION_PARTITION_RULE, (
            "the rule must state the maximum-index convention in words, because "
            "a one-hot family that is not a partition reads as a modelling "
            "choice unless the intended rule is written down. Got: "
            f"{PRECIPITATION_PARTITION_RULE!r}"
        )

    @pytest.mark.parametrize(
        ("probability", "probability_index", "millimetres", "rainfall_index"),
        PROBABILITY_BY_RAINFALL_GRID,
    )
    def test_exactly_one_band_fires_at_every_point_of_the_forecast_grid(
        self,
        calculator: WeatherFeaturesCalculator,
        probability: float,
        probability_index: int,
        millimetres: float,
        rainfall_index: int,
    ) -> None:
        """Sixteen combinations, one band each, and it is the LARGER index.

        Both halves matter. "Exactly one fires" alone would be satisfied by a
        rule that silently DOWNGRADES a heavy forecast to a light one; naming
        WHICH band must fire is what pins the widening direction.
        """
        features = calculator.calculate_precipitation_features(
            _payload(precip_prob=probability, precip_mm=millimetres)
        )
        fired = [band for band in PRECIPITATION_BAND_COLUMNS if features[band] == 1.0]
        expected = PRECIPITATION_BAND_COLUMNS[max(probability_index, rainfall_index)]

        assert fired == [expected], (
            f"probability {probability} (band index {probability_index}) with "
            f"{millimetres} mm (band index {rainfall_index}) fired {fired}; "
            f"exactly [{expected!r}] must fire. A disjunction over the two "
            "readings fires one band per reading and reports a single game as "
            "two intensities at once."
        )

    def test_the_windows_row_39_example_fires_exactly_one_band(
        self, calculator: WeatherFeaturesCalculator
    ) -> None:
        """The measured example from the ledger row: probability 0.4, 3.0 mm.

        Under the old disjunction this set ``precip_light`` AND
        ``precip_moderate``. Under the maximum-band-index rule the rainfall's
        index (2, moderate) wins over the probability's (1, light), so the band
        is MODERATE -- the honest reading, because 3.0 mm of rain actually fell
        and a 40 per cent chance of rain cannot make it lighter than that.
        """
        features = calculator.calculate_precipitation_features(
            _payload(precip_prob=0.4, precip_mm=3.0)
        )
        fired = [band for band in PRECIPITATION_BAND_COLUMNS if features[band] == 1.0]

        assert fired == ["precip_moderate"], (
            "WINDOWS.md row 39's own example must resolve to exactly one band. "
            f"Fired: {fired}"
        )

    @pytest.mark.parametrize(
        ("probability", "probability_index", "millimetres", "rainfall_index"),
        PROBABILITY_BY_RAINFALL_GRID,
    )
    def test_the_probability_widens_a_band_and_never_adds_a_level(
        self,
        calculator: WeatherFeaturesCalculator,
        probability: float,
        probability_index: int,
        millimetres: float,
        rainfall_index: int,
    ) -> None:
        """The module's own docstring claim, made checkable.

        ``_precipitation_impact`` states the rule both branches were written to
        follow: "the probability only ever widened the LIGHT and MODERATE bands,
        never added a level". Widening means the forecast band index is never
        BELOW the band the rainfall alone would have produced -- so adding a
        probability can only ever move the answer UP the intensity order.
        """
        with_probability = calculator.calculate_precipitation_features(
            _payload(precip_prob=probability, precip_mm=millimetres)
        )
        without_probability = calculator.calculate_precipitation_features(
            _payload(precip_prob=None, precip_mm=millimetres)
        )

        fired_with = [
            position
            for position, band in enumerate(PRECIPITATION_BAND_COLUMNS)
            if with_probability[band] == 1.0
        ]
        fired_without = [
            position
            for position, band in enumerate(PRECIPITATION_BAND_COLUMNS)
            if without_probability[band] == 1.0
        ]

        assert len(fired_with) == 1, (
            f"the forecast branch fired {fired_with} at probability "
            f"{probability} with {millimetres} mm; exactly one band must fire"
        )
        assert len(fired_without) == 1, (
            f"the mm-only branch fired {fired_without} at {millimetres} mm; "
            "exactly one band must fire"
        )
        assert fired_with[0] >= fired_without[0], (
            f"probability {probability} DOWNGRADED {millimetres} mm from band "
            f"{fired_without[0]} to band {fired_with[0]}. The probability widens "
            "a band; it never lowers one."
        )
        assert fired_without[0] == rainfall_index, (
            f"{millimetres} mm must sit in rainfall band {rainfall_index} on the "
            f"mm-only branch; it fired band {fired_without[0]}"
        )
        assert fired_with[0] == max(probability_index, rainfall_index), (
            f"the fired band must be the LARGER of the two indices; expected "
            f"{max(probability_index, rainfall_index)}, got {fired_with[0]}"
        )

    @pytest.mark.parametrize(
        ("millimetres", "rainfall_index"),
        RAINFALL_BY_BAND_INDEX,
    )
    def test_the_mm_only_branch_is_unchanged_across_the_same_grid(
        self,
        calculator: WeatherFeaturesCalculator,
        millimetres: float,
        rainfall_index: int,
    ) -> None:
        """THE CONTROL. The archive branch is the one 33.1 repaired; leave it be.

        Pinned as an explicit four-value mapping rather than as "one band
        fires", because the failure this guards against is the fix reaching
        across into the branch it was not aiming at -- and a laxer assertion
        would not notice a band that MOVED as long as the result stayed a
        partition.

        The cut points restated in the maximum-index helpers are the SAME cut
        points the untouched mm-only literal carries. This test is what keeps
        the two spellings honest: if either drifts, the expected one-hot below
        stops matching.
        """
        features = calculator.calculate_precipitation_features(
            _payload(precip_prob=None, precip_mm=millimetres)
        )
        observed = {band: features[band] for band in PRECIPITATION_BAND_COLUMNS}
        expected = {
            band: (1.0 if position == rainfall_index else 0.0)
            for position, band in enumerate(PRECIPITATION_BAND_COLUMNS)
        }

        assert observed == expected, (
            f"the mm-only branch moved at {millimetres} mm. Expected "
            f"{expected}, got {observed}. That branch is byte-unchanged source; "
            "a move here means the forecast fix reached into it."
        )
