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

    async def _fetch(latitude, longitude, game_date, game_hour, venue_timezone):
        return await fetch_game_weather(
            client, latitude, longitude, game_date, game_hour, venue_timezone
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
