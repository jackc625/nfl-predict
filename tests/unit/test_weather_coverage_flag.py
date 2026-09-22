"""The coverage flag tells "no forecast" from "no weather applies"; an absent mm is not rain.

Plan 33.2-12 Task 1 (SPEC R6), p332_ rung 4.

THE FLAG. An outdoor game with no bulletin is NULL weather with ``weather_coverage`` 0.0; a
dome or closed-roof game is NOT flagged missing (it has no weather by design); an open-roof
game with a bulletin is populated and unflagged.

THE ABSENT-MILLIMETRE ARM. Every archived bulletin carries a probability and NO millimetre
amount. Before the arm, such a row reached ``_rainfall_band_index`` with a NaN, which is
False at every cut point and returns the HEAVY band -- every outdoor game in 24 seasons
would have read ``precip_heavy``. With the arm the band comes from the readings that exist:
the probability's band and the QPF level.

THE PLAN'S WORDING, AND WHERE IT DISAGREES WITH THE MODULE. The plan asks for a probability of
0.9 to resolve to "the band the probability alone selects -- NEITHER resolves to
precip_heavy". Those two halves cannot both hold: ``_probability_band_index`` puts any
probability above 0.8 in the heavy band. The defect the arm exists for is heavy FROM A NaN,
so the not-heavy assertions use probabilities whose own band is below heavy (0.1 and 0.6),
and 0.9 is asserted to take exactly the probability's own band -- which is how the module
has always banded a forecast probability.

The regression partner is
``tests/unit/test_precipitation_from_measurement.py::TestTheGateNarrowsRatherThanDisappears::test_a_payload_carrying_both_readings_is_byte_preserved``,
run unchanged in the same verification.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import math
from datetime import UTC, datetime

import pandas as pd
import pytest

from features.weather import PRECIPITATION_BAND_COLUMNS, WeatherFeaturesCalculator

KICKOFF = pd.Timestamp("2016-09-11 17:00", tz="UTC")
BULLETIN = pd.Timestamp("2016-09-10 12:00", tz="UTC")
BUILD = datetime(2016, 9, 20, tzinfo=UTC)

OUTDOOR_NO_BULLETIN = "2016_W01_ABS@ABS"
DOME = "2016_W01_DOM@DOM"
OPEN_ROOF_WITH_BULLETIN = "2016_W01_OPN@OPN"


def _games() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"game_id": g, "season": 2016, "week": 1, "kickoff_et": KICKOFF}
            for g in (OUTDOOR_NO_BULLETIN, DOME, OPEN_ROOF_WITH_BULLETIN)
        ]
    )


def _silver() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": OUTDOOR_NO_BULLETIN,
                "forecast_time": BULLETIN,
                "forecast_issue_time": None,
                "weather_source": "historical_forecast",
                "is_outdoor": True,
                "weather_coverage": False,
            },
            {
                "game_id": DOME,
                "forecast_time": BULLETIN,
                "forecast_issue_time": None,
                "weather_source": "historical_forecast",
                "is_outdoor": False,
                "weather_coverage": True,
            },
            {
                "game_id": OPEN_ROOF_WITH_BULLETIN,
                "forecast_time": BULLETIN,
                "forecast_issue_time": BULLETIN,
                "weather_source": "historical_forecast",
                "is_outdoor": True,
                "weather_coverage": True,
                "temp_f": 61.0,
                "wind_mph": 9.2,
                "humidity_pct": 64.0,
                "precip_prob": 0.25,
                "precip_mm": None,
                "mos_precip_level": 1,
            },
        ]
    )


@pytest.fixture(scope="module")
def built() -> pd.DataFrame:
    calculator = WeatherFeaturesCalculator()
    with pytest.warns(DeprecationWarning):
        frame = calculator.build_weather_features(
            _games(), weather_df=_silver(), build_instant=BUILD
        )
    return frame.set_index("game_id")


class TestTheCoverageFlag:
    def test_an_outdoor_game_with_no_bulletin_is_null_plus_flag(self, built):
        row = built.loc[OUTDOOR_NO_BULLETIN]
        assert row["weather_affects_game"] == 1.0
        assert row["weather_coverage"] == 0.0
        for column in ("temp_f", "wind_mph", "precip_prob", "weather_severity_score"):
            assert math.isnan(row[column]), column

    def test_a_dome_is_not_flagged_missing(self, built):
        row = built.loc[DOME]
        assert row["weather_coverage"] == 1.0
        assert row["weather_affects_game"] == 0.0

    def test_an_open_roof_game_with_a_bulletin_is_populated_and_unflagged(self, built):
        row = built.loc[OPEN_ROOF_WITH_BULLETIN]
        assert row["weather_coverage"] == 1.0
        assert row["weather_affects_game"] == 1.0
        assert row["temp_f"] == 61.0 and row["wind_mph"] == 9.2
        assert math.isnan(row["precip_mm"]), "no millimetre amount is invented"
        assert row["precip_light"] == 1.0, "QPF level 1 and probability 0.25 are light"


def _band(payload: dict) -> str:
    features = WeatherFeaturesCalculator().calculate_precipitation_features(payload)
    hot = [column for column in PRECIPITATION_BAND_COLUMNS if features[column] == 1.0]
    assert len(hot) == 1, f"the bands are not a partition: {hot}"
    return hot[0]


def _payload(**readings: object) -> dict:
    return {"game_id": "G", "temp_f": 55.0, "condition": None, **readings}


class TestAnAbsentMillimetreIsNotRain:
    def test_a_null_mm_with_a_low_probability_is_none(self):
        assert _band(_payload(precip_mm=None, precip_prob=0.1)) == "precip_none"

    def test_a_null_mm_takes_the_probabilitys_own_band_not_heavy(self):
        assert _band(_payload(precip_mm=None, precip_prob=0.6)) == "precip_moderate"

    def test_a_nan_mm_raises_nothing_and_is_not_heavy(self):
        assert _band(_payload(precip_mm=float("nan"), precip_prob=0.1)) == "precip_none"
        assert (
            _band(_payload(precip_mm=float("nan"), precip_prob=0.6))
            == "precip_moderate"
        )

    def test_a_high_probability_takes_exactly_its_own_band(self):
        calculator = WeatherFeaturesCalculator()
        expected = PRECIPITATION_BAND_COLUMNS[calculator._probability_band_index(0.9)]
        assert _band(_payload(precip_mm=None, precip_prob=0.9)) == expected

    def test_the_qpf_level_is_the_rainfall_side_of_the_max_rule(self):
        assert (
            _band(_payload(precip_mm=None, precip_prob=0.1, mos_precip_level=2))
            == "precip_moderate"
        )
        assert (
            _band(_payload(precip_mm=None, precip_prob=None, mos_precip_level=3))
            == "precip_heavy"
        )

    def test_the_arm_is_load_bearing(self):
        """Non-vacuity: the rainfall helper itself still reads a NaN as HEAVY.

        NaN is False at all three cut points, so the helper falls through to band 3. The
        not-heavy results above therefore come from the absent-millimetre arm routing
        around it, not from the helper having become safe.
        """
        assert WeatherFeaturesCalculator()._rainfall_band_index(float("nan")) == 3

    def test_no_reading_at_all_is_the_null_family(self):
        features = WeatherFeaturesCalculator().calculate_precipitation_features(
            _payload(precip_mm=None, precip_prob=None, mos_precip_level=None)
        )
        assert all(math.isnan(value) for value in features.values())

    def test_the_impact_score_and_the_band_agree_without_a_millimetre(self):
        features = WeatherFeaturesCalculator().calculate_precipitation_features(
            _payload(precip_mm=None, precip_prob=0.6)
        )
        assert features["precip_impact_score"] == 0.6
