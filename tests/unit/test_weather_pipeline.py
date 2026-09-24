"""Unit tests for weather pipeline behavior.

Tests cover:
- Indoor game handling (zeroed weather fields)
- Retractable roof treated as outdoor
- Outdoor game with successful weather fetch
- Outdoor game with failed weather fetch (hard-fail)
- Venue coordinate lookup with team normalization
- Indoor weather record factory
- Bronze-to-Silver validation with WeatherSchema
- nflverse roof type mapping
"""

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from utils.exceptions import WeatherDataError

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def ingester():
    """Create a WeatherDataIngester instance with mocked settings."""
    with patch("scripts.ingest_weather.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock()
        from scripts.ingest_weather import WeatherDataIngester

        return WeatherDataIngester()


@pytest.fixture
def backfiller():
    """The QUARANTINED archive path (Plan 33-09 Task 2).

    `WeatherDataIngester` is the live FORECAST path and no longer reaches the archive
    endpoint at all; the archive fetch lives in `scripts/backfill_historical_weather.py`,
    which subclasses it. Every test below that exercises archive behaviour asks for this
    fixture, so which endpoint a test is about is visible in its signature.
    """
    with patch("scripts.ingest_weather.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock()
        from scripts.backfill_historical_weather import HistoricalWeatherBackfiller

        return HistoricalWeatherBackfiller()


@pytest.fixture
def venues_df() -> pd.DataFrame:
    """Create a minimal venues DataFrame for testing."""
    import json
    from pathlib import Path

    venues_path = Path("data/venues.json")
    with open(venues_path) as f:
        venues_data = json.load(f)

    return pd.DataFrame(venues_data["venues"])


@pytest.fixture
def sample_outdoor_weather() -> dict[str, Any]:
    """Sample weather data returned by Open-Meteo."""
    return {
        "temp_f": 45.0,
        "temp_c": 7.2,
        "wind_mph": 15.0,
        "wind_direction": 270,
        "humidity_pct": 65.0,
        "precip_prob": None,
        "precip_mm": 0.0,
        "condition": None,
        "condition_code": 3,
        "visibility_km": None,
        "dew_point_f": 32.0,
        "apparent_temp_f": 38.0,
        "snowfall_cm": 0.0,
        "wind_gusts_mph": 22.0,
        "cloud_cover_pct": 50.0,
        "weather_code": 3,
    }


# REAL GAMES, NOT INVENTED ONES (Plan 33.1-02 Task 1).
#
# These fixtures used to describe `2024_W06_KC@BUF` and `2024_W06_KC@LV`, neither
# of which was ever played. That was harmless while the backfill routed by HOME
# TEAM -- any id with a plausible home team resolved. It is not harmless now: the
# archive path joins the pinned schedules for each game's own `stadium_id` and
# `roof`, and a game the feed has never heard of is refused by name rather than
# routed on a default. A fixture that cannot describe a game the feed does not
# have is the same discipline `tests/unit/test_venues_json_historical.py` applies
# to venues.
#
# Each id below was read from `data.upstream_pin.load_schedules` on 2026-09-12,
# with its stadium and its own feed `roof` recorded beside it.
# The live weather ingest reads the pinned model's run time from meta.json once a
# forecast is fetched (Plan 33.2-27 Task 2); this module stubs the forecast, so the
# stamp is kept offline too.
pytestmark = pytest.mark.usefixtures("openmeteo_meta_offline")

_DOME_GAME_ID = "2024_W06_PIT@LV"  # VEG00 Allegiant, feed roof "dome"
_DOME_KICKOFF_ET = datetime(2024, 10, 13, 16, 5)

_OUTDOOR_GAME_ID = "2024_W06_WAS@BAL"  # BAL00 M&T Bank, feed roof "outdoors"
_OUTDOOR_KICKOFF_ET = datetime(2024, 10, 13, 13, 0)

# The ONE shape that proves R4: a retractable stadium whose roof was OPEN for this
# game. 2024 has no `open` game at all, so this comes from 2016 -- IND00 Lucas Oil
# played two of its eight 2016 home games with the roof open and six closed.
_OPEN_ROOF_GAME_ID = "2016_W01_DET@IND"  # IND00, feed roof "open"
_OPEN_ROOF_KICKOFF_ET = datetime(2016, 9, 11, 13, 0)


def _make_games_df(
    game_id: str = _OUTDOOR_GAME_ID,
    *,
    kickoff_et: datetime = _OUTDOOR_KICKOFF_ET,
) -> pd.DataFrame:
    """A one-row games frame for *game_id*, with its columns DERIVED from the id.

    Season, week and both teams are parsed out of the id rather than passed
    separately, so a fixture cannot describe a game whose id and columns disagree.
    """
    from utils.game_id_utils import parse_game_id

    parts = parse_game_id(game_id)
    return pd.DataFrame(
        [
            {
                "game_id": game_id,
                "season": parts["season"],
                "week": parts["week"],
                "home_team": parts["home_team"],
                "away_team": parts["away_team"],
                "kickoff_et": kickoff_et,
            }
        ]
    )


# ---------------------------------------------------------------------------
# Test: Indoor game produces zeroed weather
# ---------------------------------------------------------------------------


class TestIndoorWeather:
    """Indoor game (roof_type='indoor') produces zeroed weather fields."""

    def test_indoor_game_zeroed_fields(self, backfiller, venues_df):
        """Indoor game gets is_outdoor=False, temp=None, wind=0, precip=0."""
        # PIT at Allegiant Stadium, whose feed roof for this game is "dome".
        games_df = _make_games_df(_DOME_GAME_ID, kickoff_et=_DOME_KICKOFF_ET)

        weather_df = backfiller.fetch_weather_for_games(
            games_df, venues_df, forecast_time=datetime(2024, 10, 11, 22, 0)
        )

        assert len(weather_df) == 1
        row = weather_df.iloc[0]
        assert row["is_outdoor"] == False  # noqa: E712 - pandas returns numpy bool
        assert row["temp_f"] is None
        assert row["wind_mph"] == 0.0
        assert row["precip_mm"] == 0.0
        assert row["humidity_pct"] is None
        assert row["condition"] == "indoor"
        assert row["is_cold"] == False  # noqa: E712
        assert row["is_windy"] == False  # noqa: E712
        assert row["is_precipitation"] == False  # noqa: E712
        # A dome IS covered: it carries the record it is entitled to, which is no
        # observation at all (Plan 33.1-02 Ruling E).
        assert row["weather_coverage"] == True  # noqa: E712


# ---------------------------------------------------------------------------
# Test: Retractable roof treated as outdoor
# ---------------------------------------------------------------------------


class TestRetractableRoof:
    """An OPEN roof is outdoor -- and it is the GAME's roof that decides (R4).

    This class used to drive a 2024 NRG Stadium game and assert `is_outdoor=True`
    because the VENUE's `roof_type` is "retractable". That assertion passed for the
    wrong reason: every game at a retractable stadium answered the same way,
    including the ones played with the roof shut. Plan 33.1-02 branches on the
    game's own feed `roof`, so the fixture now names a game that was actually
    played with the roof OPEN.
    """

    def test_retractable_roof_is_outdoor(self, backfiller, venues_df):
        """DET at Lucas Oil, roof OPEN in the feed -- weather applies."""
        games_df = _make_games_df(_OPEN_ROOF_GAME_ID, kickoff_et=_OPEN_ROOF_KICKOFF_ET)

        mock_weather = AsyncMock(
            return_value={
                "temp_f": 82.0,
                "temp_c": 27.8,
                "wind_mph": 8.0,
                "wind_direction": 180,
                "humidity_pct": 70.0,
                "precip_prob": None,
                "precip_mm": 0.0,
                "condition": None,
                "condition_code": 1,
                "visibility_km": None,
                "dew_point_f": 72.0,
                "apparent_temp_f": 85.0,
                "snowfall_cm": 0.0,
                "wind_gusts_mph": 12.0,
                "cloud_cover_pct": 20.0,
                "weather_code": 1,
            },
        )
        with patch.object(
            backfiller,
            "_fetch_openmeteo_weather",
            mock_weather,
        ):
            weather_df = backfiller.fetch_weather_for_games(
                games_df,
                venues_df,
                forecast_time=datetime(2024, 10, 11, 22, 0),
            )

        assert len(weather_df) == 1
        row = weather_df.iloc[0]
        assert row["is_outdoor"] == True  # noqa: E712 - pandas returns numpy bool
        assert row["temp_f"] == 82.0


# ---------------------------------------------------------------------------
# Test: Outdoor game with successful fetch
# ---------------------------------------------------------------------------


class TestOutdoorWeatherFetch:
    """Outdoor game with successful weather fetch has populated fields."""

    def test_outdoor_game_populated_fields(self, backfiller, venues_df):
        """WAS at M&T Bank Stadium (outdoors) -- real weather data returned."""
        games_df = _make_games_df(_OUTDOOR_GAME_ID)

        mock_weather = AsyncMock(
            return_value={
                "temp_f": 45.0,
                "temp_c": 7.2,
                "wind_mph": 15.0,
                "wind_direction": 270,
                "humidity_pct": 65.0,
                "precip_prob": None,
                "precip_mm": 0.0,
                "condition": None,
                "condition_code": 3,
                "visibility_km": None,
                "dew_point_f": 32.0,
                "apparent_temp_f": 38.0,
                "snowfall_cm": 0.0,
                "wind_gusts_mph": 22.0,
                "cloud_cover_pct": 50.0,
                "weather_code": 3,
            },
        )
        with patch.object(
            backfiller,
            "_fetch_openmeteo_weather",
            mock_weather,
        ):
            weather_df = backfiller.fetch_weather_for_games(
                games_df,
                venues_df,
                forecast_time=datetime(2024, 10, 11, 22, 0),
            )

        assert len(weather_df) == 1
        row = weather_df.iloc[0]
        assert row["is_outdoor"] == True  # noqa: E712 - pandas returns numpy bool
        assert row["temp_f"] == 45.0
        assert row["wind_mph"] == 15.0


# ---------------------------------------------------------------------------
# Test: Outdoor game where weather fetch fails raises WeatherDataError
# ---------------------------------------------------------------------------


class TestOutdoorWeatherFetchFailure:
    """Outdoor game where weather fetch fails raises WeatherDataError."""

    def test_outdoor_fetch_failure_raises(self, backfiller, venues_df):
        """If Open-Meteo fails for an outdoor game, WeatherDataError propagates."""
        games_df = _make_games_df(_OUTDOOR_GAME_ID)

        mock_weather = AsyncMock(side_effect=WeatherDataError("API timeout"))
        with patch.object(
            backfiller,
            "_fetch_openmeteo_weather",
            mock_weather,
        ):
            with pytest.raises(WeatherDataError, match="API timeout"):
                backfiller.fetch_weather_for_games(
                    games_df,
                    venues_df,
                    forecast_time=datetime(2024, 10, 11, 22, 0),
                )


# ---------------------------------------------------------------------------
# Test: _get_venue_coordinates returns correct lat/lon for BUF
# ---------------------------------------------------------------------------


class TestVenueCoordinates:
    """Venue coordinate lookup tests."""

    def test_buf_coordinates(self, ingester, venues_df):
        """BUF -> Highmark Stadium at (42.7738, -78.787)."""
        lat, lon, roof = ingester._get_venue_coordinates("BUF", venues_df)
        assert abs(lat - 42.7738) < 0.001
        assert abs(lon - (-78.787)) < 0.001
        assert roof == "outdoor"

    def test_lar_normalizes_to_la(self, ingester, venues_df):
        """'LAR' normalizes to 'LA' and finds SoFi Stadium."""
        lat, lon, roof = ingester._get_venue_coordinates("LAR", venues_df)
        # SoFi Stadium
        assert abs(lat - 33.9535) < 0.001
        assert abs(lon - (-118.3392)) < 0.001
        assert roof == "indoor"


# ---------------------------------------------------------------------------
# Indoor weather record factory
# ---------------------------------------------------------------------------


class TestCreateIndoorWeatherRecord:
    """Indoor weather record factory creates zeroed fields."""

    def test_creates_zeroed_record(self, ingester):
        """Indoor weather record has is_outdoor=False and zeroed fields."""
        game_time = datetime(2024, 10, 13, 17, 0, tzinfo=UTC)
        forecast_time = datetime(2024, 10, 11, 22, 0, tzinfo=UTC)

        record = ingester._create_indoor_weather_record(
            game_id="2024_W06_KC@LV",
            game_time=game_time,
            forecast_time=forecast_time,
            weather_source="forecast",
        )

        assert record["game_id"] == "2024_W06_KC@LV"
        assert record["weather_source"] == "forecast"
        assert record["is_outdoor"] is False
        assert record["temp_f"] is None
        assert record["temp_c"] is None
        assert record["wind_mph"] == 0.0
        assert record["wind_direction"] is None
        assert record["humidity_pct"] is None
        assert record["precip_prob"] == 0.0
        assert record["precip_mm"] == 0.0
        assert record["condition"] == "indoor"
        assert record["condition_code"] is None
        assert record["visibility_km"] is None
        assert record["is_cold"] is False
        assert record["is_windy"] is False
        assert record["is_precipitation"] is False
        assert record["weather_coverage"] is True


# ---------------------------------------------------------------------------
# Test: Weather data validates through validate_bronze_to_silver with WeatherSchema
# ---------------------------------------------------------------------------


class TestBronzeToSilverValidation:
    """Weather data passes validate_bronze_to_silver with WeatherSchema."""

    def test_valid_indoor_weather_passes_validation(self, ingester):
        """An indoor weather record passes WeatherSchema validation."""
        from data.quality_gates import validate_bronze_to_silver
        from data.schemas import WeatherSchema

        game_time = datetime(2024, 10, 13, 17, 0, tzinfo=UTC)
        forecast_time = datetime(2024, 10, 11, 22, 0, tzinfo=UTC)

        record = ingester._create_indoor_weather_record(
            game_id="2024_W06_KC@LV",
            game_time=game_time,
            forecast_time=forecast_time,
            weather_source="forecast",
        )
        df = pd.DataFrame([record])

        validated_df = validate_bronze_to_silver(df, WeatherSchema)
        assert len(validated_df) == 1
        assert validated_df.iloc[0]["is_outdoor"] == False  # noqa: E712


# ---------------------------------------------------------------------------
# Test: nflverse roof type mapping
# ---------------------------------------------------------------------------


class TestNflverseRoofTypeMapping:
    """nflreadpy roof types map correctly to project VenueRoof values."""

    def test_dome_maps_to_indoor(self, ingester):
        assert ingester._map_nflverse_roof_type("dome") == "indoor"

    def test_closed_maps_to_indoor(self, ingester):
        assert ingester._map_nflverse_roof_type("closed") == "indoor"

    def test_outdoors_maps_to_outdoor(self, ingester):
        assert ingester._map_nflverse_roof_type("outdoors") == "outdoor"

    def test_open_maps_to_retractable(self, ingester):
        assert ingester._map_nflverse_roof_type("open") == "retractable"


# ---------------------------------------------------------------------------
# Test: NaN -> None coercion validator on WeatherSchema
# ---------------------------------------------------------------------------


class TestNanToNoneValidator:
    """WeatherSchema.nan_to_none maps pandas NaN/pd.NA/None to None uniformly
    across all nullable numeric fields, before ge/le range checks run.

    Regression guard for Phase 15 UAT bug: None -> NaN round-trip via pandas
    DataFrame was failing `precip_prob le=1` and indoor temp/humidity/condition_code
    range checks.
    """

    def _base_kwargs(self):
        return {
            "game_id": "2024_W06_KC@BUF",
            "forecast_time": datetime(2024, 10, 11, 22, 0, tzinfo=UTC),
            "game_time": datetime(2024, 10, 13, 17, 0, tzinfo=UTC),
            "is_outdoor": True,
            "weather_coverage": True,
            "weather_source": "forecast",
        }

    def test_nan_wind_direction_coerces_to_none(self):
        from data.schemas import WeatherSchema

        m = WeatherSchema(**self._base_kwargs(), wind_direction=float("nan"))
        assert m.wind_direction is None

    def test_nan_precip_prob_coerces_to_none(self):
        from data.schemas import WeatherSchema

        m = WeatherSchema(**self._base_kwargs(), precip_prob=float("nan"))
        assert m.precip_prob is None

    def test_nan_indoor_unknowns_coerce_to_none(self):
        from data.schemas import WeatherSchema

        m = WeatherSchema(
            **self._base_kwargs(),
            temp_f=float("nan"),
            humidity_pct=float("nan"),
            condition_code=float("nan"),  # pyright: ignore[reportArgumentType]
        )
        assert m.temp_f is None
        assert m.humidity_pct is None
        assert m.condition_code is None

    def test_real_float_wind_direction_survives(self):
        from data.schemas import WeatherSchema

        m = WeatherSchema(**self._base_kwargs(), wind_direction=270.5)
        assert m.wind_direction == 270.5

    def test_wind_direction_range_still_enforced(self):
        from pydantic import ValidationError

        from data.schemas import WeatherSchema

        with pytest.raises(ValidationError):
            WeatherSchema(**self._base_kwargs(), wind_direction=361.0)
        with pytest.raises(ValidationError):
            WeatherSchema(**self._base_kwargs(), wind_direction=-0.1)

    def test_real_zero_wind_mph_not_coerced(self):
        """Indoor record emits wind_mph=0.0 -- must NOT be coerced to None by
        the nan_to_none validator (pd.isna(0.0) returns False, confirmed)."""
        from data.schemas import WeatherSchema

        m = WeatherSchema(**self._base_kwargs(), wind_mph=0.0)
        assert m.wind_mph == 0.0

    def test_new_openmeteo_fields_round_trip(self):
        """The 5 new Open-Meteo fields persist through model_dump (not dropped
        by Pydantic extra='ignore' anymore)."""
        from data.schemas import WeatherSchema

        m = WeatherSchema(
            **self._base_kwargs(),
            dew_point_f=32.0,
            apparent_temp_f=38.0,
            snowfall_cm=0.0,
            wind_gusts_mph=22.0,
            cloud_cover_pct=50.0,
        )
        dumped = m.model_dump()
        assert dumped["dew_point_f"] == 32.0
        assert dumped["apparent_temp_f"] == 38.0
        assert dumped["snowfall_cm"] == 0.0
        assert dumped["wind_gusts_mph"] == 22.0
        assert dumped["cloud_cover_pct"] == 50.0


# ---------------------------------------------------------------------------
# `weather_source` PROVENANCE (D33-26, Plan 33-09 Task 2).
#
# Every weather row says where it came from. Three sources, one closed
# vocabulary, and a value outside it is rejected BY NAME at write time rather
# than stored and puzzled over later.
#
# WHY PROVENANCE AND NOT A GUESS FROM THE DATA. An archive row and a forecast row
# for the same game are the same 23 columns of plausible numbers. Nothing in the
# values distinguishes "measured after the fact" from "predicted three days out",
# and the difference is exactly what a Phase-37 weather-aware re-fit needs to
# know. Deriving it later from a forecast_time-before-game_time comparison would
# be an inference about our own pipeline, recorded nowhere, that stops being true
# the first time a backfill runs.
# ---------------------------------------------------------------------------


class TestWeatherSourceVocabulary:
    """The vocabulary is CLOSED, single-sourced, and rejects by name."""

    def test_the_vocabulary_is_the_three_declared_values(self):
        import scripts.ingest_weather as ingest

        assert tuple(ingest.WEATHER_SOURCE_VOCABULARY) == (
            "archive",
            "forecast",
            "historical_forecast",
        )

    def test_the_vocabulary_matches_the_recorded_state(self):
        """One value, one home -- the single-source discipline BET_LIST_COLUMNS uses."""
        import scripts.ingest_weather as ingest
        from tests import phase33_state

        assert tuple(ingest.WEATHER_SOURCE_VOCABULARY) == tuple(
            phase33_state.WEATHER_SOURCE_VOCABULARY
        )

    @pytest.mark.parametrize("value", ["archive", "forecast", "historical_forecast"])
    def test_each_permitted_value_is_returned_unchanged(self, value):
        import scripts.ingest_weather as ingest

        assert ingest.validate_weather_source(value) == value

    def test_an_out_of_vocabulary_value_is_rejected_naming_the_value(self):
        import scripts.ingest_weather as ingest

        with pytest.raises(ValueError) as caught:
            ingest.validate_weather_source("made_up")

        message = str(caught.value)
        assert "made_up" in message
        for permitted in ingest.WEATHER_SOURCE_VOCABULARY:
            assert permitted in message, (
                "the refusal must name the ALLOWED vocabulary as well as the "
                "rejected value; a reader who only learns their value was wrong "
                "still does not know what to write instead."
            )

    def test_none_is_rejected_rather_than_treated_as_unknown(self):
        """An unstamped row is the state this column exists to make impossible."""
        import scripts.ingest_weather as ingest

        with pytest.raises(ValueError, match="weather_source"):
            ingest.validate_weather_source(None)


class TestWeatherSourceOnTheSilverTable:
    """The stored table, read-only. Nothing in this class writes."""

    def test_the_silver_weather_table_is_the_widened_shape(self):
        """The width is WEATHER_COLUMNS_AFTER_COVERAGE now, not WEATHER_COLUMNS_AFTER.

        RE-TARGETED by Plan 33.1-06 Task 4, not weakened. Plan 33-09 widened this
        table 23 -> 24 by adding `weather_source`, and this assertion pinned that
        24. Plan 33.1-02 then declared `weather_coverage` as the 25th column and
        recorded the new width in its OWN constant,
        `WEATHER_COLUMNS_AFTER_COVERAGE`, which until now no test read. The
        promotion in Plan 33.1-06 Task 3 is the write that makes the stored table
        25 wide.

        The assertion still pins an EXACT declared width against a constant in the
        manifest; only which constant it reads has moved, and the superseded
        `WEATHER_COLUMNS_AFTER = 24` is retained in source with its own history.

        RE-TARGETED AGAIN by Plan 33.2-12 (p332_ rung 4), in the same way. Plan 33.2-11
        declared five MOS fields on WeatherSchema and recorded the schema's width, 30, as
        `P332_11_WEATHER_SCHEMA_FIELDS_AFTER_MOS`, naming Plan 33.2-12's regeneration as
        the write that widens the stored table. That write happened, so the stored table
        is now the schema's full width; `WEATHER_COLUMNS_AFTER_COVERAGE = 25` is retained
        in the manifest as history.
        """
        from tests import phase33_state

        frame = pd.read_parquet("data/silver/weather.parquet")
        assert (
            len(frame.columns) == phase33_state.P332_11_WEATHER_SCHEMA_FIELDS_AFTER_MOS
        )
        assert "weather_source" in frame.columns
        assert "weather_coverage" in frame.columns

    def test_the_pre_existing_rows_are_stamped_archive(self):
        """The fourteen rows already on disk came from the ARCHIVE endpoint, and the
        backfill says so rather than leaving them null.

        THE SELECTOR NARROWED, THE CLAIM DID NOT (Plan 33.1-06 Task 4). Those
        fourteen rows are 2024 WEEK 6, and `2024_` isolated them only while 2024
        was the sole season in the table. The full-corpus promotion put all 285
        rows of 2024 in, so the old selector now names a population this assertion
        was never about. `2024_W06_` names exactly the original fourteen -- still
        fourteen, still every one of them stamped `archive` -- so the assertion
        tests what it always tested.

        THE STAMP MOVED WITH THE SOURCE (Plan 33.2-12, p332_ rung 4). Rung 4 replaced
        every 2002-2025 ERA5 observation -- these fourteen included -- with the archived
        day-before forecast, so the same fourteen rows now read `historical_forecast`.
        What is still tested is the claim: the rows are all there and each says, from the
        closed vocabulary, where it came from.
        """
        from tests import phase33_state

        frame = pd.read_parquet("data/silver/weather.parquet")
        pre_existing = frame[frame["game_id"].str.startswith("2024_W06_")]
        assert len(pre_existing) == phase33_state.WEATHER_ROWS_BEFORE
        assert set(pre_existing["weather_source"]) == {"historical_forecast"}

    def test_no_stored_row_carries_a_value_outside_the_vocabulary(self):
        import scripts.ingest_weather as ingest

        frame = pd.read_parquet("data/silver/weather.parquet")
        stored = set(frame["weather_source"].dropna())
        assert stored <= set(ingest.WEATHER_SOURCE_VOCABULARY), (
            "stored weather_source values outside the vocabulary: "
            f"{sorted(stored - set(ingest.WEATHER_SOURCE_VOCABULARY))}"
        )


class TestTheLivePathStampsForecast:
    """A row produced by the forecast path reads `forecast`, never `archive`."""

    def test_a_forecast_row_is_stamped_forecast(self, ingester):
        from datetime import timedelta

        as_of = datetime(2026, 9, 11, 20, 0, tzinfo=UTC)
        games = pd.DataFrame(
            [
                {
                    "game_id": "2026_W02_KC@BUF",
                    "season": 2026,
                    "week": 2,
                    "home_team": "BUF",
                    "away_team": "KC",
                    "kickoff_et": as_of + timedelta(days=3),
                    "stadium_id": "BUF00",
                    "neutral_site": False,
                }
            ]
        )
        venues = pd.DataFrame(
            [
                {
                    "stadium_id": "BUF00",
                    "venue_id": "highmark_stadium",
                    "latitude": 42.7738,
                    "longitude": -78.787,
                    "roof_type": "outdoor",
                    "timezone": "America/New_York",
                    "home_teams": ["BUF"],
                }
            ]
        )
        record = {
            "temp_f": 45.0,
            "temp_c": 7.2,
            "wind_mph": 15.0,
            "wind_direction": 270.0,
            "humidity_pct": 65.0,
            "precip_prob": None,
            "precip_mm": 0.0,
            "condition": None,
            "condition_code": 3,
            "visibility_km": None,
            "dew_point_f": 32.0,
            "apparent_temp_f": 38.0,
            "snowfall_cm": 0.0,
            "wind_gusts_mph": 22.0,
            "cloud_cover_pct": 50.0,
            "weather_code": 3,
        }

        with patch.object(
            ingester, "_fetch_openmeteo_forecast", AsyncMock(return_value=record)
        ):
            frame = ingester.fetch_forecast_for_games(games, venues, as_of_utc=as_of)

        assert set(frame["weather_source"]) == {"forecast"}

    def test_an_indoor_forecast_row_is_also_stamped_forecast(self, ingester):
        """An indoor row is never fetched, but it IS produced by the forecast run.

        Leaving it unstamped would put a null in the one column whose whole purpose is
        that every row says where it came from.
        """
        from datetime import timedelta

        as_of = datetime(2026, 9, 11, 20, 0, tzinfo=UTC)
        games = pd.DataFrame(
            [
                {
                    "game_id": "2026_W02_KC@LV",
                    "season": 2026,
                    "week": 2,
                    "home_team": "LV",
                    "away_team": "KC",
                    "kickoff_et": as_of + timedelta(days=3),
                    "stadium_id": "LVS00",
                    "neutral_site": False,
                }
            ]
        )
        venues = pd.DataFrame(
            [
                {
                    "stadium_id": "LVS00",
                    "venue_id": "allegiant_stadium",
                    "latitude": 36.0909,
                    "longitude": -115.1833,
                    "roof_type": "indoor",
                    "timezone": "America/Los_Angeles",
                    "home_teams": ["LV"],
                }
            ]
        )

        frame = ingester.fetch_forecast_for_games(games, venues, as_of_utc=as_of)
        assert set(frame["weather_source"]) == {"forecast"}
        assert bool(frame.iloc[0]["is_outdoor"]) is False


class TestTheBackfillPathStampsArchive:
    """The quarantined archive path stamps `archive`, never `forecast`."""

    def test_an_archive_row_is_stamped_archive(self, backfiller, venues_df):
        games_df = _make_games_df(_OUTDOOR_GAME_ID)
        record = {
            "temp_f": 45.0,
            "temp_c": 7.2,
            "wind_mph": 15.0,
            "wind_direction": 270.0,
            "humidity_pct": 65.0,
            "precip_prob": None,
            "precip_mm": 0.0,
            "condition": None,
            "condition_code": 3,
            "visibility_km": None,
            "dew_point_f": 32.0,
            "apparent_temp_f": 38.0,
            "snowfall_cm": 0.0,
            "wind_gusts_mph": 22.0,
            "cloud_cover_pct": 50.0,
            "weather_code": 3,
        }
        with patch.object(
            backfiller, "_fetch_openmeteo_weather", AsyncMock(return_value=record)
        ):
            frame = backfiller.fetch_weather_for_games(
                games_df, venues_df, forecast_time=datetime(2024, 10, 11, 22, 0)
            )

        assert set(frame["weather_source"]) == {"archive"}
