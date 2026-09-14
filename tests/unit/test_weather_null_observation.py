"""An absent observation is recorded as ABSENT, and never as a number.

WHAT IS BEING PINNED
--------------------
SPEC R4, second half: "a game whose venue resolves but whose ERA5 observation is
absent writes explicit NULL weather with a coverage flag and a `weather_source`".

D33.1-07 names three states. The coverage flag is what makes them recoverable,
because it composes with `is_outdoor`::

    (weather_coverage=1, is_outdoor=0)  a dome -- entitled to no observation
    (weather_coverage=1, is_outdoor=1)  a real observation
    (weather_coverage=0, is_outdoor=1)  an ABSENCE -- every weather column NULL

Before the flag those three collapsed into one: `features/weather.py` filled a
missing record with `temp_f: 65.0` AND `is_outdoor: False`, so a missing record
was indistinguishable from a dome BY CONSTRUCTION. That default was reached for
6,485 of 6,499 gold rows.

WHY A FACTORY-ONLY TEST WOULD PROVE NOTHING (Codex 33.1-02 HIGH)
-----------------------------------------------------------------
D33.1-07 says what the three states ARE but not how the middle one is DETECTED,
and before Plan 33.1-02 the code could not detect it at all: `fetch_game_weather`
read `hourly["temperature_2m"][idx]` and returned a full record even when that
value and every sibling was null. So an all-null ERA5 day produced a row that
LOOKED like an observation.

A unit test of a new factory says nothing about that, because nothing would have
called the factory. So test 10 below drives the WHOLE path --
`fetch_game_weather -> the record factory -> the bronze row` -- against a stub
whose arrays are PRESENT but all-null, and asserts on the bronze row.

Ruling D3's predicate is FIVE series wide, and the width is the decision. One
series would call an ordinary day absent whenever ERA5 happened to have
temperature but no cloud cover. All twelve would be unreachable, because
`snowfall` and `wind_gusts_10m` are legitimately null on ordinary days. Test 11 is
the PARTIAL-null control that proves "absent" is narrower than "any null".

FOUR CONTROLS, copied from `tests/unit/test_weather_archive_quarantined.py:31-42`
---------------------------------------------------------------------------------
1. NON-VACUITY: the absent path is REACHED through the real fetch, asserted by
   the stub's own request count, not assumed.
2. THE ASSERTIONS: NULL everywhere, coverage False, and it survives the schema.
3. A PLANTED VIOLATION: an undeclared sibling column IS dropped by
   `validate_bronze_to_silver`, proving the round-trip assertion is asserting
   something; and a schema without the field makes the promotion refuse.
4. NO FALSE POSITIVE: a partial null is an observation, and a dome round-trips
   with coverage True.

Every write in this module goes to `sandbox_data_root`. It needs no
`writes_production_store` marker.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from pydantic import BaseModel, ConfigDict, Field

from data.quality_gates import validate_bronze_to_silver
from data.schemas import WeatherSchema
from scripts.backfill_historical_weather import (
    ABSENT_OBSERVATION,
    ABSENT_OBSERVATION_MEASUREMENTS,
    WEATHER_SOURCE_ARCHIVE,
    fetch_game_weather,
    observation_is_absent,
    promote_weather_bronze_to_silver,
)
from utils.exceptions import DataIngestionError, WeatherDataError

OPEN_GAME_ID = "2016_W01_DET@IND"

# Every nullable weather MEASUREMENT on the silver schema. An absent-observation
# row must be NULL in all of them, with no numeric value anywhere.
WEATHER_MEASUREMENT_COLUMNS: tuple[str, ...] = (
    "temp_f",
    "temp_c",
    "wind_mph",
    "wind_direction",
    "humidity_pct",
    "precip_prob",
    "precip_mm",
    "dew_point_f",
    "apparent_temp_f",
    "snowfall_cm",
    "wind_gusts_mph",
    "cloud_cover_pct",
)


@pytest.fixture
def backfiller():
    with patch("scripts.ingest_weather.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock()
        from scripts.backfill_historical_weather import HistoricalWeatherBackfiller

        return HistoricalWeatherBackfiller()


@pytest.fixture
def venues_df():
    return pd.DataFrame(json.loads(Path("data/venues.json").read_text())["venues"])


def _is_null(value) -> bool:
    return value is None or pd.isna(value)


def _drive_one_game(backfiller, venues_df, games_df, client):
    """Drive ONE game through `fetch_game_weather` and return the bronze frame.

    The real `fetch_game_weather` runs -- only the HTTP client is replaced -- so
    the predicate, the sentinel and the factory branch are all exercised. This is
    what makes the assertions below statements about the PATH rather than about a
    factory nothing calls.
    """

    async def _fetch(
        latitude,
        longitude,
        game_date,
        game_hour,
        venue_timezone,
        *,
        budget=None,
        throttle=None,
    ):
        # `budget` and `throttle` are THREADED THROUGH rather than dropped: Plan
        # 33.1-05's Ruling L2 puts the debit and the wait inside the retried
        # `fetch_game_weather`, so a seam that swallowed them would exercise a
        # different function from the one production calls.
        return await fetch_game_weather(
            client,
            latitude,
            longitude,
            game_date,
            game_hour,
            venue_timezone,
            budget=budget,
            throttle=throttle,
        )

    with patch.object(backfiller, "_fetch_openmeteo_weather", _fetch):
        return backfiller.fetch_weather_for_games(games_df, venues_df)


def _open_game_only(per_game_roof_games):
    """Just the roof-OPEN game, so exactly one fetch is issued."""
    frame = per_game_roof_games()
    return frame[frame["game_id"] == OPEN_GAME_ID].reset_index(drop=True)


class TestThePredicateItself:
    """The five-name rule, before anything is concluded from it."""

    def test_the_predicate_is_exactly_five_series_wide(self):
        assert len(ABSENT_OBSERVATION_MEASUREMENTS) == 5
        assert set(ABSENT_OBSERVATION_MEASUREMENTS) == {
            "temperature_2m",
            "wind_speed_10m",
            "precipitation",
            "relative_humidity_2m",
            "cloud_cover",
        }

    def test_all_five_null_is_absent(self):
        hourly = {name: [None] for name in ABSENT_OBSERVATION_MEASUREMENTS}
        assert observation_is_absent(hourly, 0) is True

    @pytest.mark.parametrize("present", ABSENT_OBSERVATION_MEASUREMENTS)
    def test_any_single_series_present_makes_it_an_observation(self, present):
        """PARTIAL IS NEITHER STATE, proven one series at a time.

        This is what stops a later reader widening "absent" into "any null".
        """
        hourly = {name: [None] for name in ABSENT_OBSERVATION_MEASUREMENTS}
        hourly[present] = [1.0]
        assert observation_is_absent(hourly, 0) is False


class TestTheAbsentPathIsReachedThroughTheRealFetch:
    """CONTROL 1 (non-vacuity) and CONTROL 2 (the assertion) -- test 10.

    The stub returns an archive payload whose five series are PRESENT AS ARRAYS
    and null at every index. Nothing here calls the record factory directly.
    """

    def test_an_all_null_payload_produces_an_absent_bronze_row(
        self, backfiller, venues_df, per_game_roof_games, stub_archive_client
    ):
        client = stub_archive_client("absent")
        frame = _drive_one_game(
            backfiller, venues_df, _open_game_only(per_game_roof_games), client
        )

        assert client.calls == 1, (
            "the absent branch must be reached BY A REAL RESPONSE. A zero count "
            "would mean this test proves nothing about the fetch path."
        )
        assert len(frame) == 1
        row = frame.iloc[0]

        assert bool(row["weather_coverage"]) is False
        assert bool(row["is_outdoor"]) is True, (
            "an absence is only reachable for a game that needed a fetch. "
            "is_outdoor=False here would make the row a dome, which is the "
            "collapse the coverage flag exists to undo."
        )
        assert row["weather_source"] == WEATHER_SOURCE_ARCHIVE
        assert row["condition"] is None

        for column in WEATHER_MEASUREMENT_COLUMNS:
            assert _is_null(row[column]), (
                f"{column} carries {row[column]!r} on a row with no observation. "
                "A missing observation silently becoming a number is the "
                "fabricated-data class this phase exists to remove."
            )

    def test_the_sentinel_is_what_the_fetch_returns(self, stub_archive_client):
        """The branch the loop takes is decided by a value, not by an exception."""
        client = stub_archive_client("absent")
        result = asyncio.run(
            fetch_game_weather(
                client, 39.76, -86.16, "2016-09-11", 13, "America/New_York"
            )
        )
        assert result == ABSENT_OBSERVATION


class TestThePartialNullControl:
    """CONTROL 4 -- test 11. "Absent" is narrower than "any null"."""

    def test_a_partially_null_payload_is_an_observation(
        self, backfiller, venues_df, per_game_roof_games, stub_archive_client
    ):
        client = stub_archive_client("partial")
        frame = _drive_one_game(
            backfiller, venues_df, _open_game_only(per_game_roof_games), client
        )

        assert client.calls == 1
        row = frame.iloc[0]

        assert bool(row["weather_coverage"]) is True, (
            "a day with a real temperature and no cloud-cover reading is an "
            "observation with a missing field, not an absence."
        )
        assert not _is_null(row["temp_f"]), "the present value must be carried"
        assert _is_null(row["cloud_cover_pct"]), "the absent field must stay NULL"


class TestTheRoundTripThroughTheSchema:
    """CONTROL 2 and CONTROL 3 -- tests 7, 8 and 9 (RESEARCH P-6)."""

    def test_an_absent_row_keeps_its_flag_through_validate_bronze_to_silver(
        self, backfiller, venues_df, per_game_roof_games, stub_archive_client
    ):
        frame = _drive_one_game(
            backfiller,
            venues_df,
            _open_game_only(per_game_roof_games),
            stub_archive_client("absent"),
        )
        validated = validate_bronze_to_silver(frame, WeatherSchema)

        assert "weather_coverage" in validated.columns
        assert bool(validated.iloc[0]["weather_coverage"]) is False
        for column in WEATHER_MEASUREMENT_COLUMNS:
            assert _is_null(validated.iloc[0][column])

    def test_an_undeclared_sibling_column_IS_dropped(
        self, backfiller, venues_df, per_game_roof_games, stub_archive_client
    ):
        """THE PLANTED VIOLATION. Without it the test above asserts nothing.

        Pydantic v2 defaults to extra="ignore" and `validate_bronze_to_silver`
        rebuilds every row as `schema_class(**row).model_dump()`, so a column the
        schema does not name disappears with NO error. This proves the drop is
        real -- and therefore that `weather_coverage` surviving is a fact about
        the schema rather than about the gate being lenient.
        """
        frame = _drive_one_game(
            backfiller,
            venues_df,
            _open_game_only(per_game_roof_games),
            stub_archive_client("absent"),
        )
        frame["weather_coverag"] = True  # one character off, as a typo would be

        validated = validate_bronze_to_silver(frame, WeatherSchema)

        assert "weather_coverag" not in validated.columns, (
            "the undeclared column survived, so extra='ignore' is not in force "
            "and this control is not reproducing the drop it exists to show."
        )
        assert "weather_coverage" in validated.columns

    def test_a_schema_without_the_field_makes_the_promotion_refuse(
        self, backfiller, venues_df, per_game_roof_games, stub_archive_client
    ):
        """The PRODUCTION-side assertion, driven by a schema that drops the field.

        A passing round-trip proves the column survives TODAY. This proves the
        promotion path would REFUSE rather than write a silver frame silently one
        column short.
        """

        class WeatherSchemaWithoutCoverage(BaseModel):
            model_config = ConfigDict(extra="ignore")

            game_id: str = Field(...)
            forecast_time: datetime = Field(...)
            game_time: datetime = Field(...)
            is_outdoor: bool = Field(...)

        frame = _drive_one_game(
            backfiller,
            venues_df,
            _open_game_only(per_game_roof_games),
            stub_archive_client("absent"),
        )

        with pytest.raises(DataIngestionError) as excinfo:
            promote_weather_bronze_to_silver(frame, WeatherSchemaWithoutCoverage)

        message = str(excinfo.value)
        assert "weather_coverage" in message
        assert 'extra="ignore"' in message, (
            "the refusal must name the MECHANISM, not only the symptom: a reader "
            "who learns a column is missing still does not know why."
        )

    def test_the_real_schema_promotion_does_not_refuse(
        self, backfiller, venues_df, per_game_roof_games, stub_archive_client
    ):
        """CONTROL 4 -- the assertion is reachable, not universal."""
        frame = _drive_one_game(
            backfiller,
            venues_df,
            _open_game_only(per_game_roof_games),
            stub_archive_client("absent"),
        )
        validated = promote_weather_bronze_to_silver(frame)
        assert len(validated) == 1


class TestTheDomeStateIsDistinguished:
    """CONTROL 4 -- test 9. Covered-and-indoor is not covered-and-absent."""

    def test_a_dome_row_round_trips_with_coverage_true_and_temp_null(
        self, backfiller, venues_df, per_game_roof_games, stub_archive_client
    ):
        client = stub_archive_client("good")
        frame = per_game_roof_games()
        result = _drive_one_game(backfiller, venues_df, frame, client)

        closed = result[result["game_id"] != OPEN_GAME_ID].iloc[0]
        validated = promote_weather_bronze_to_silver(result)
        row = validated[validated["game_id"] == closed["game_id"]].iloc[0]

        assert bool(row["weather_coverage"]) is True
        assert bool(row["is_outdoor"]) is False
        assert _is_null(row["temp_f"])
        assert row["wind_mph"] == 0.0, (
            "wind indoors is genuinely 0.0, not unknown -- which is why it is a "
            "number here and NULL on the absent row."
        )
        assert row["condition"] == "indoor"

    def test_the_three_states_are_pairwise_distinguishable(
        self, backfiller, venues_df, per_game_roof_games, stub_archive_client
    ):
        """The whole point of the flag, stated as one assertion."""
        observed = _drive_one_game(
            backfiller,
            venues_df,
            per_game_roof_games(),
            stub_archive_client("good"),
        )
        absent = _drive_one_game(
            backfiller,
            venues_df,
            _open_game_only(per_game_roof_games),
            stub_archive_client("absent"),
        )

        states = set()
        for frame in (observed, absent):
            for _, row in frame.iterrows():
                states.add((bool(row["weather_coverage"]), bool(row["is_outdoor"])))

        assert states == {(True, False), (True, True), (False, True)}, (
            "the dome, the observation and the absence must occupy three "
            f"different (coverage, outdoor) cells. Got {sorted(states)}."
        )


class TestTheRefusalsTheFetchStillMakes:
    """The two named refusals that replaced the clamp, and the 400 `reason`."""

    def test_a_payload_that_does_not_reach_the_hour_is_refused_by_name(
        self, stub_archive_client
    ):
        client = stub_archive_client("good", hours=12)

        with pytest.raises(WeatherDataError) as excinfo:
            asyncio.run(
                fetch_game_weather(
                    client, 39.76, -86.16, "2016-09-11", 19, "America/New_York"
                )
            )

        message = str(excinfo.value)
        # The message is the forecast sibling's, verbatim in shape: "The hour is
        # NOT clamped to the last available one". Matched case-insensitively so
        # the emphasis capital cannot be mistaken for a different message.
        assert "not clamped" in message.lower()
        assert "12" in message and "19" in message, (
            "the refusal reports the returned hour count AND the requested hour, "
            "so the reader can see which side is wrong."
        )

    def test_a_400_surfaces_the_apis_own_reason(self, stub_archive_client):
        client = stub_archive_client("error_400")

        with pytest.raises(WeatherDataError) as excinfo:
            asyncio.run(
                fetch_game_weather(
                    client, 39.76, -86.16, "1939-01-01", 13, "America/New_York"
                )
            )

        message = str(excinfo.value)
        assert "out of allowed range" in message, (
            "the archive explains itself in the `reason` field on every 400. A "
            "refusal that reports only a status code throws away the only "
            "diagnostic there is, and the operator re-runs into the same wall."
        )
        assert "400" in message, "the status code is kept as well, not replaced"

    def test_the_request_is_made_in_the_venues_own_zone(self, stub_archive_client):
        """CONTROL 4 -- the corrected zone is observable on the request itself."""
        client = stub_archive_client("good")
        asyncio.run(
            fetch_game_weather(
                client, 51.4561, -0.3417, "2016-10-23", 14, "Europe/London"
            )
        )

        assert client.params_seen[0]["timezone"] == "Europe/London", (
            "a fixed Eastern zone is the wrong zone for 50 of the 55 venues in "
            "the corpus, and for a 09:30 London kickoff it is also the wrong DAY."
        )


class TestNothingHereTouchesAProductionStore:
    """The sandbox is a DIRECTORY, not a patched name."""

    def test_the_sandbox_root_is_outside_the_repository_data_tree(
        self, sandbox_data_root
    ):
        assert sandbox_data_root.is_dir()
        for layer in ("bronze", "silver", "gold"):
            assert (sandbox_data_root / layer).is_dir()

        production = Path("data").resolve()
        assert production not in sandbox_data_root.resolve().parents
        assert sandbox_data_root.resolve() != production

    def test_a_promoted_frame_can_be_written_into_the_sandbox_only(
        self,
        backfiller,
        venues_df,
        per_game_roof_games,
        stub_archive_client,
        sandbox_data_root,
        data_boundary_guard,
    ):
        """The write happens, and `data_boundary_guard` proves where it did not.

        `data_boundary_guard` digests every tracked file under `data/` before and
        after this test and fails if any byte moved -- which is a stronger claim
        than "the code was told to write elsewhere".
        """
        from data.storage import upsert_silver

        frame = _drive_one_game(
            backfiller,
            venues_df,
            _open_game_only(per_game_roof_games),
            stub_archive_client("absent"),
        )
        validated = promote_weather_bronze_to_silver(frame)
        written = upsert_silver(validated, "weather", base_path=sandbox_data_root)

        assert Path(written).resolve().is_relative_to(sandbox_data_root.resolve())
        reread = pd.read_parquet(written)
        assert bool(reread.iloc[0]["weather_coverage"]) is False


# ===========================================================================
# PLAN 33.1-04 TASK 1 -- RULING J, THE THREE D33.1-07 STATES IN THE FEATURE FRAME
#
# Everything above this line is Plan 33.1-02's: it pins the three states in
# BRONZE and SILVER, where the observation is written. Everything below pins the
# same three states one layer down, in the FEATURE FRAME `features/weather.py`
# builds -- which is where the mild-temperature default actually reached 6,485
# of 6,499 gold rows.
#
# The two halves are deliberately in one module rather than two. D33.1-07 is ONE
# ruling about three states, and splitting it across modules is how the silver
# half and the feature half drift into disagreeing about what "absent" means.
#
# The builds below all drive the INJECTED `weather_df` seam. No storage layer is
# read and nothing is monkeypatched, so no test here can reach a production
# store (Phase 33 Wave 6: a `monkeypatch` is not a sandbox).
# ===========================================================================

import ast
import inspect

import numpy as np

import features.weather as weather_module
from features.weather import (
    WEATHER_FEATURE_COLUMNS,
    WEATHER_FEATURE_COLUMNS_BY_BUILDER,
    WEATHER_NON_NUMERIC_FEATURE_COLUMNS,
    WeatherFeaturesCalculator,
    WeatherObservationError,
)

DOME_GAME_ID = "2016_W01_DME@DME"
ABSENT_GAME_ID = "2016_W01_ABS@ABS"
OBSERVED_GAME_ID = "2016_W01_OBS@OBS"

FORECAST_TIME = datetime(2016, 9, 11, 13, 0)

# Ruling J's row groups, as data. The committed copy lives in
# tests.phase33_state.WEATHER_NULL_STATE_MATRIX; these are the names the
# assertions below iterate, and the matrix is cross-checked against them.
TEMPERATURE_GROUP: tuple[str, ...] = (
    "raw_temp_f",
    "temp_f",
    "apparent_temp_f",
    "temp_hot",
    "temp_warm",
    "temp_mild",
    "temp_cool",
    "temp_cold",
    "temp_very_cold",
)
TEMPERATURE_IMPACT_GROUP: tuple[str, ...] = (
    "cold_impact_score",
    "heat_impact_score",
    "scoring_multiplier",
    "ball_handling_difficulty",
)
WIND_GROUP: tuple[str, ...] = (
    "raw_wind_mph",
    "wind_mph",
    "wind_calm",
    "wind_moderate",
    "wind_high",
    "wind_severe",
    "wind_impact_score",
    "kicking_difficulty",
    "passing_difficulty",
)
PRECIPITATION_GROUP: tuple[str, ...] = (
    "raw_precip_mm",
    "raw_precip_prob",
    "precip_mm",
    "precip_prob",
    "precip_none",
    "precip_light",
    "precip_moderate",
    "precip_heavy",
    "is_snow",
    "is_rain",
    "is_dry",
    "precip_impact_score",
    "turnover_multiplier",
    "passing_efficiency",
)
COMPOSITE_GROUP: tuple[str, ...] = (
    "weather_severity_score",
    "home_weather_advantage",
    "defensive_advantage",
    "rushing_advantage",
    "scoring_reduction",
    "weather_game",
    "extreme_weather",
)


def _ruling_j_games() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"game_id": DOME_GAME_ID, "season": 2016, "week": 1},
            {"game_id": ABSENT_GAME_ID, "season": 2016, "week": 1},
            {"game_id": OBSERVED_GAME_ID, "season": 2016, "week": 1},
        ]
    )


def _ruling_j_weather() -> pd.DataFrame:
    """One silver row per Ruling J state, written exactly as 33.1-02 writes them."""
    return pd.DataFrame(
        [
            {
                "game_id": DOME_GAME_ID,
                "forecast_time": FORECAST_TIME,
                "is_outdoor": False,
                "weather_coverage": True,
                "temp_f": None,
                "wind_mph": 0.0,
                "precip_prob": 0.0,
                "precip_mm": 0.0,
                "humidity_pct": None,
                "condition": "indoor",
            },
            {
                "game_id": ABSENT_GAME_ID,
                "forecast_time": FORECAST_TIME,
                "is_outdoor": True,
                "weather_coverage": False,
                "temp_f": None,
                "wind_mph": None,
                "precip_prob": None,
                "precip_mm": None,
                "humidity_pct": None,
                "condition": None,
            },
            {
                "game_id": OBSERVED_GAME_ID,
                "forecast_time": FORECAST_TIME,
                "is_outdoor": True,
                "weather_coverage": True,
                "temp_f": 41.0,
                "wind_mph": 14.0,
                "precip_prob": 0.6,
                "precip_mm": 3.0,
                "humidity_pct": 70.0,
                "condition": "Rain",
            },
        ]
    )


@pytest.fixture
def ruling_j_frame() -> pd.DataFrame:
    """The full (uncompressed) weather feature frame over all three states."""
    calculator = WeatherFeaturesCalculator()
    with pytest.warns(DeprecationWarning):
        return calculator.build_weather_features(
            _ruling_j_games(), weather_df=_ruling_j_weather()
        )


def _one_row(frame: pd.DataFrame, game_id: str) -> pd.Series:
    matched = frame[frame["game_id"] == game_id]
    assert len(matched) == 1, f"expected exactly one row for {game_id}"
    return matched.iloc[0]


class TestRulingJRowByRow:
    """One assertion block per row of Ruling J's table."""

    def test_a_dome_row_is_null_in_temperature_and_genuine_elsewhere(
        self, ruling_j_frame
    ):
        row = _one_row(ruling_j_frame, DOME_GAME_ID)

        for column in (*TEMPERATURE_GROUP, *TEMPERATURE_IMPACT_GROUP):
            assert _is_null(row[column]), (
                f"{column} reads {row[column]!r} on a DOME row. There is no outdoor "
                "temperature indoors, and a band with no temperature to band has no "
                "answer -- 1.0 in temp_warm is the mild default in one-hot clothing."
            )

        assert row["wind_calm"] == 1.0, "indoors the wind genuinely IS calm"
        assert row["is_dry"] == 1.0, "indoors it genuinely IS dry"
        assert row["scoring_reduction"] == 0.0, (
            "'weather reduced scoring by nothing' is a TRUE statement about an "
            "indoor game, not a stand-in for an absent measurement"
        )
        assert row["weather_coverage"] == 1.0, "a dome is entitled to no observation"
        assert row["weather_affects_game"] == 0.0

    def test_an_absent_observation_row_is_null_in_every_weather_column(
        self, ruling_j_frame
    ):
        row = _one_row(ruling_j_frame, ABSENT_GAME_ID)

        assert row["weather_affects_game"] == 1.0, (
            "weather DOES apply to this game; what is absent is the observation, "
            "which weather_coverage records"
        )
        assert row["weather_coverage"] == 0.0

        exempt = {"weather_affects_game", "weather_coverage"}
        for column in WEATHER_FEATURE_COLUMNS_BY_BUILDER["full"]:
            if column in exempt:
                continue
            assert _is_null(row[column]), (
                f"{column} carries {row[column]!r} on a row with NO observation. "
                "A missing observation silently becoming a number is the "
                "fabricated-data class this phase exists to remove."
            )
            if column not in WEATHER_NON_NUMERIC_FEATURE_COLUMNS:
                assert not np.isfinite(row[column]), (
                    f"{column} is a finite number on an absent-observation row"
                )

    def test_an_observed_outdoor_row_is_finite_in_every_group(self, ruling_j_frame):
        """The NEGATIVE CONTROL. Without it the NaN assertions above would pass
        equally against a builder that returned an empty frame."""
        row = _one_row(ruling_j_frame, OBSERVED_GAME_ID)

        for group in (
            TEMPERATURE_GROUP,
            TEMPERATURE_IMPACT_GROUP,
            WIND_GROUP,
            PRECIPITATION_GROUP,
            COMPOSITE_GROUP,
        ):
            for column in group:
                assert np.isfinite(row[column]), (
                    f"{column} is not finite on an OBSERVED outdoor row, so the "
                    "NaN assertions above are describing an empty frame rather "
                    "than distinguishing a state"
                )

        assert row["weather_coverage"] == 1.0
        assert row["weather_affects_game"] == 1.0
        assert row["raw_temp_f"] == 41.0


class TestACaughtExceptionRaisesForEveryFamily:
    """D33.1-07's third state, applied to the CALCULATION and not to temperature.

    Three families, three tests. Removing only the temperature handler would
    leave a malformed wind payload inventing a perfectly calm day -- in the
    family the deployed O/U model reads most heavily.
    """

    def test_a_malformed_temperature_payload_raises_and_names_the_family(self):
        calculator = WeatherFeaturesCalculator()
        with pytest.raises(WeatherObservationError) as excinfo:
            calculator.calculate_temperature_features(
                {"game_id": OBSERVED_GAME_ID, "temp_f": "warm", "wind_mph": 3.0}
            )
        message = str(excinfo.value)
        assert OBSERVED_GAME_ID in message
        assert "temperature" in message

    def test_a_malformed_wind_payload_raises_and_names_the_family(self):
        calculator = WeatherFeaturesCalculator()
        with pytest.raises(WeatherObservationError) as excinfo:
            calculator.calculate_wind_features(
                {"game_id": OBSERVED_GAME_ID, "wind_mph": "breezy"}
            )
        message = str(excinfo.value)
        assert OBSERVED_GAME_ID in message
        assert "wind" in message

    def test_a_malformed_precipitation_payload_raises_and_names_the_family(self):
        calculator = WeatherFeaturesCalculator()
        with pytest.raises(WeatherObservationError) as excinfo:
            calculator.calculate_precipitation_features(
                {
                    "game_id": OBSERVED_GAME_ID,
                    "precip_prob": 0.5,
                    "precip_mm": "heavy",
                    "temp_f": 40.0,
                }
            )
        message = str(excinfo.value)
        assert OBSERVED_GAME_ID in message
        assert "precipitation" in message

    def test_a_caught_exception_is_named_a_bug_rather_than_a_state(self):
        calculator = WeatherFeaturesCalculator()
        with pytest.raises(WeatherObservationError) as excinfo:
            calculator.calculate_wind_features(
                {"game_id": OBSERVED_GAME_ID, "wind_mph": "breezy"}
            )
        assert "bug" in str(excinfo.value).lower(), (
            "the message must SAY a caught exception is a bug rather than a "
            "state -- recording it as missing data is precisely how the mild "
            "default survived unnoticed for years"
        )


class TestTheIndoorFactoriesAreIndoorOnly:
    """The Ruling J amendment: the rename is about REACHABILITY, not naming."""

    def test_the_default_factories_are_gone_and_the_indoor_ones_exist(self):
        for gone in (
            "_default_temperature_features",
            "_default_wind_features",
            "_default_precipitation_features",
        ):
            assert not hasattr(WeatherFeaturesCalculator, gone), (
                f"{gone} still exists. While it is called _default_ the next "
                "reader will wire the next `except` branch to it by analogy, and "
                "the name is the only thing that stops that."
            )
        for present in (
            "_indoor_temperature_features",
            "_indoor_wind_features",
            "_indoor_precipitation_features",
            "_absent_observation_features",
        ):
            assert hasattr(WeatherFeaturesCalculator, present)

    @pytest.mark.parametrize(
        "factory", ["_indoor_wind_features", "_indoor_precipitation_features"]
    )
    def test_each_renamed_factory_has_exactly_one_call_site(self, factory):
        tree = ast.parse(inspect.getsource(weather_module))
        call_sites = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and (
                getattr(node.func, "attr", None) == factory
                or getattr(node.func, "id", None) == factory
            )
        ]
        assert len(call_sites) == 1, (
            f"{factory} is called from {len(call_sites)} places ({call_sites}). "
            "The indoor branch is the ONLY legitimate caller; a second call site "
            "is a failure rather than a review question."
        )

    def test_no_except_handler_reaches_an_indoor_factory(self):
        tree = ast.parse(inspect.getsource(weather_module))
        handlers = [n for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler)]
        reached = [
            (handler.lineno, name)
            for handler in handlers
            for node in ast.walk(handler)
            if isinstance(node, ast.Call)
            for name in [
                getattr(node.func, "attr", None) or getattr(node.func, "id", None)
            ]
            if name
            in {
                "_indoor_wind_features",
                "_indoor_precipitation_features",
                "_indoor_temperature_features",
                "_absent_observation_features",
            }
        ]
        assert reached == [], (
            f"an indoor factory is reachable from an `except` branch: {reached}. "
            "That is the fabrication path the Ruling J amendment removes."
        )

    def test_every_except_handler_in_the_module_raises(self):
        tree = ast.parse(inspect.getsource(weather_module))
        handlers = [n for n in ast.walk(tree) if isinstance(n, ast.ExceptHandler)]
        assert handlers, "no handlers found -- this scan would assert nothing"
        non_raising = [
            handler.lineno
            for handler in handlers
            if not any(isinstance(n, ast.Raise) for n in ast.walk(handler))
        ]
        assert non_raising == [], (
            f"handlers at lines {non_raising} swallow an exception. In this "
            "module an `except` branch that returns a value at all is the "
            "fabrication pattern D33.1-07 forbids."
        )


class TestAMissingWeatherRowIsAnAnomalyRatherThanADefault:
    def test_a_game_absent_from_the_weather_frame_raises_by_name(self):
        calculator = WeatherFeaturesCalculator()
        games = pd.concat(
            [
                _ruling_j_games(),
                pd.DataFrame(
                    [{"game_id": "2016_W01_GAP@GAP", "season": 2016, "week": 1}]
                ),
            ],
            ignore_index=True,
        )
        with pytest.raises(WeatherObservationError) as excinfo:
            with pytest.warns(DeprecationWarning):
                calculator.build_weather_features(games, weather_df=_ruling_j_weather())
        message = str(excinfo.value)
        assert "2016_W01_GAP@GAP" in message
        assert "weather" in message, "the raise must name the silver table to backfill"

    def test_the_compressed_builder_also_raises_on_a_missing_row(self):
        calculator = WeatherFeaturesCalculator()
        games = pd.DataFrame(
            [{"game_id": "2016_W01_GAP@GAP", "season": 2016, "week": 1}]
        )
        with pytest.raises(WeatherObservationError) as excinfo:
            calculator.build_features(
                games,
                datetime(2016, 9, 12, 0, 0),
                weather_df=_ruling_j_weather(),
            )
        assert "2016_W01_GAP@GAP" in str(excinfo.value)


class TestTheColumnDeclarationCannotDriftFromEitherBuilder:
    def test_the_full_builder_emits_exactly_its_declared_entry(self, ruling_j_frame):
        emitted = set(ruling_j_frame.columns) - {"game_id", "season", "week"}
        assert emitted == set(WEATHER_FEATURE_COLUMNS_BY_BUILDER["full"])
        assert "weather_coverage" in WEATHER_FEATURE_COLUMNS

    def test_the_compressed_builder_emits_exactly_its_declared_entry(self):
        calculator = WeatherFeaturesCalculator()
        frame = calculator.build_features(
            _ruling_j_games(),
            datetime(2016, 9, 12, 0, 0),
            weather_df=_ruling_j_weather(),
        )
        emitted = set(frame.columns) - {"game_id"}
        assert emitted == set(WEATHER_FEATURE_COLUMNS_BY_BUILDER["compressed"])

    def test_the_two_builder_entries_are_a_real_distinction(self):
        by_builder = WEATHER_FEATURE_COLUMNS_BY_BUILDER
        assert sorted(by_builder) == ["compressed", "full"]
        assert by_builder["full"] and by_builder["compressed"]
        assert set(by_builder["full"]) != set(by_builder["compressed"]), (
            "if the two entries were equal the per-builder split would be "
            "describing a distinction that does not exist"
        )
        assert set(by_builder["full"]) | set(by_builder["compressed"]) == set(
            WEATHER_FEATURE_COLUMNS
        )


class TestTheCompressedBuilderFollowsTheSameRuling:
    def test_the_compressed_absent_row_is_null_and_the_dome_row_is_zero(self):
        calculator = WeatherFeaturesCalculator()
        frame = calculator.build_features(
            _ruling_j_games(),
            datetime(2016, 9, 12, 0, 0),
            weather_df=_ruling_j_weather(),
        )

        absent = _one_row(frame, ABSENT_GAME_ID)
        assert absent["is_outdoor"] == 1.0
        for column in ("weather_severity_score", "wind_mph", "is_precipitation"):
            assert _is_null(absent[column])

        dome = _one_row(frame, DOME_GAME_ID)
        assert dome["is_outdoor"] == 0.0
        assert dome["weather_severity_score"] == 0.0
        assert dome["wind_mph"] == 0.0

        observed = _one_row(frame, OBSERVED_GAME_ID)
        assert observed["is_outdoor"] == 1.0
        assert np.isfinite(observed["weather_severity_score"])


class TestANullMeasurementInsideAnObservationStaysNull:
    """A PARTIAL null is an observation (33.1-02), and its derived family is
    still NULL: the `or 0.0` and `or 50` fallbacks were the same stand-in.
    """

    def test_a_null_wind_on_an_observed_row_does_not_become_calm(self):
        calculator = WeatherFeaturesCalculator()
        features = calculator.calculate_wind_features(
            {"game_id": OBSERVED_GAME_ID, "wind_mph": None}
        )
        assert _is_null(features["wind_mph"])
        assert _is_null(features["wind_calm"]), (
            "an absent wind reading became `wind_calm: 1.0` -- a perfectly calm "
            "day invented from a missing measurement"
        )

    def test_a_null_precipitation_on_an_observed_row_does_not_become_dry(self):
        calculator = WeatherFeaturesCalculator()
        features = calculator.calculate_precipitation_features(
            {
                "game_id": OBSERVED_GAME_ID,
                "precip_mm": None,
                "precip_prob": None,
                "temp_f": 40.0,
            }
        )
        assert _is_null(features["is_dry"])
        assert _is_null(features["precip_impact_score"])

    def test_a_null_temperature_on_an_observed_row_does_not_become_mild(self):
        calculator = WeatherFeaturesCalculator()
        features = calculator.calculate_temperature_features(
            {"game_id": OBSERVED_GAME_ID, "temp_f": None, "wind_mph": 5.0}
        )
        assert _is_null(features["temp_f"])
        assert _is_null(features["temp_warm"])
        assert _is_null(features["scoring_multiplier"])

    def test_an_unknown_severity_does_not_become_a_calm_game(self):
        calculator = WeatherFeaturesCalculator()
        severity = calculator.calculate_weather_severity(
            wind_features={"wind_impact_score": float("nan")},
            temp_features={"cold_impact_score": 0.0, "heat_impact_score": 0.0},
            precip_features={"precip_impact_score": 0.0},
        )
        assert _is_null(severity["weather_severity_score"])
        assert _is_null(severity["weather_game"]), (
            "`weather_game: 0.0` from an unknown severity is a fabricated "
            "statement that this was not a weather game"
        )


class TestNoFabricatedReadingReachesTheApparentTemperatureFormula:
    """Code review WR-08: the last two numeric stand-ins in this module.

    ``_apparent_temperature_or_null`` passed ``50.0`` for an absent humidity in the
    wind-chill branch and ``0.0`` for an absent wind in the heat-index branch. Both
    were INERT, because ``_calculate_apparent_temperature``'s wind-chill formula
    never reads humidity and its heat-index formula never reads wind -- but that is
    a coupling to ANOTHER function's internal thresholds, not a property of this
    one. The module's own docstring names ``or 50`` as the defect being removed.

    These tests pin the argument that is PASSED, not just the value that comes back,
    because the returned value cannot distinguish the two versions today. That is
    exactly why a behavioural test alone would not have caught it.
    """

    @staticmethod
    def _captured_call(monkeypatch, temp_f, raw_wind, raw_humidity):
        calculator = WeatherFeaturesCalculator()
        seen: dict[str, float] = {}

        def _spy(temp, wind, humidity):
            seen.update(temp=temp, wind=wind, humidity=humidity)
            return temp

        monkeypatch.setattr(calculator, "_calculate_apparent_temperature", _spy)
        calculator._apparent_temperature_or_null(temp_f, raw_wind, raw_humidity)
        return seen

    def test_an_absent_humidity_is_passed_as_NULL_not_as_fifty(self, monkeypatch):
        seen = self._captured_call(monkeypatch, 40.0, 10.0, None)

        assert seen["wind"] == 10.0
        assert _is_null(seen["humidity"]), (
            f"absent humidity reached the formula as {seen['humidity']!r}. A "
            "plausible number here is a fabricated measurement one branch-threshold "
            "change away from becoming a model input."
        )

    def test_an_absent_wind_is_passed_as_NULL_not_as_calm(self, monkeypatch):
        seen = self._captured_call(monkeypatch, 90.0, None, 60.0)

        assert seen["humidity"] == 60.0
        assert _is_null(seen["wind"]), (
            f"absent wind reached the formula as {seen['wind']!r} -- an invented "
            "perfectly calm day, the same stand-in as `wind_mph or 0`."
        )

    def test_a_null_reading_can_only_fall_THROUGH_a_branch(self):
        """Why NAN is safe: neither guard can be satisfied by an absent reading."""
        calculator = WeatherFeaturesCalculator()
        nan = float("nan")

        assert not (nan >= 3)
        assert not (nan >= 40)
        # Cold with real wind: the wind chill applies and humidity is never read.
        assert calculator._apparent_temperature_or_null(
            40.0, 10.0, None
        ) == calculator._calculate_apparent_temperature(40.0, 10.0, nan)
        # Hot with real humidity: the heat index applies and wind is never read.
        assert calculator._apparent_temperature_or_null(
            90.0, None, 60.0
        ) == calculator._calculate_apparent_temperature(90.0, nan, 60.0)


class TestTheStateMatrixIsCommitted:
    def test_the_matrix_records_all_three_states_and_their_column_groups(self):
        from tests.phase33_state import WEATHER_NULL_STATE_MATRIX as matrix

        assert sorted(matrix["states"]) == [
            "covered_indoor",
            "covered_outdoor_observed",
            "uncovered_outdoor_absent",
        ]
        assert matrix["indoor_games_gaining_nan"] == 1652
        assert matrix["indoor_columns_gaining_nan"] == 13
        groups = matrix["column_groups"]
        assert tuple(groups["temperature"]) == TEMPERATURE_GROUP
        assert tuple(groups["temperature_impact"]) == TEMPERATURE_IMPACT_GROUP
        assert tuple(groups["wind"]) == WIND_GROUP
        assert tuple(groups["precipitation"]) == PRECIPITATION_GROUP
        assert tuple(groups["composite"]) == COMPOSITE_GROUP
