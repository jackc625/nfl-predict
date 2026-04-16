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


def _make_games_df(
    home_team: str = "BUF",
    game_id: str = "2024_W06_KC@BUF",
) -> pd.DataFrame:
    """Create a minimal games DataFrame for testing."""
    return pd.DataFrame(
        [
            {
                "game_id": game_id,
                "season": 2024,
                "week": 6,
                "home_team": home_team,
                "away_team": "KC",
                "kickoff_et": datetime(2024, 10, 13, 13, 0),
            }
        ]
    )


# ---------------------------------------------------------------------------
# Test: Indoor game produces zeroed weather
# ---------------------------------------------------------------------------


class TestIndoorWeather:
    """Indoor game (roof_type='indoor') produces zeroed weather fields."""

    def test_indoor_game_zeroed_fields(self, ingester, venues_df):
        """Indoor game gets is_outdoor=False, temp=None, wind=0, precip=0."""
        # LV plays at Allegiant Stadium (indoor)
        games_df = _make_games_df(home_team="LV", game_id="2024_W06_KC@LV")

        weather_df = ingester.fetch_weather_for_games(
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


# ---------------------------------------------------------------------------
# Test: Retractable roof treated as outdoor
# ---------------------------------------------------------------------------


class TestRetractableRoof:
    """Retractable roof game is treated as outdoor (is_outdoor=True)."""

    def test_retractable_roof_is_outdoor(self, ingester, venues_df):
        """HOU plays at NRG Stadium (retractable) -- should be treated as outdoor."""
        games_df = _make_games_df(home_team="HOU", game_id="2024_W06_KC@HOU")

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
            ingester,
            "_fetch_openmeteo_weather",
            mock_weather,
        ):
            weather_df = ingester.fetch_weather_for_games(
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

    def test_outdoor_game_populated_fields(self, ingester, venues_df):
        """BUF plays at Highmark Stadium (outdoor) -- real weather data returned."""
        games_df = _make_games_df(home_team="BUF", game_id="2024_W06_KC@BUF")

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
            ingester,
            "_fetch_openmeteo_weather",
            mock_weather,
        ):
            weather_df = ingester.fetch_weather_for_games(
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

    def test_outdoor_fetch_failure_raises(self, ingester, venues_df):
        """If Open-Meteo fails for an outdoor game, WeatherDataError propagates."""
        games_df = _make_games_df(home_team="BUF", game_id="2024_W06_KC@BUF")

        mock_weather = AsyncMock(side_effect=WeatherDataError("API timeout"))
        with patch.object(
            ingester,
            "_fetch_openmeteo_weather",
            mock_weather,
        ):
            with pytest.raises(WeatherDataError, match="API timeout"):
                ingester.fetch_weather_for_games(
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
        )

        assert record["game_id"] == "2024_W06_KC@LV"
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
            condition_code=float("nan"),
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
