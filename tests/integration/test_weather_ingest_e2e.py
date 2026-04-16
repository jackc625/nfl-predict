"""E2E integration test: Open-Meteo fixture -> fetch_game_weather -> WeatherSchema roundtrip.

Catches schema drift (Open-Meteo adding/renaming fields, or our WeatherSchema
missing a field the ingest emits) at CI time rather than in UAT.

Covers:
- Outdoor game path (BUF @ Highmark Stadium) using a recorded Open-Meteo fixture.
- Indoor game path (LV @ Allegiant Stadium) which does NOT hit Open-Meteo and
  exercises the None -> NaN -> None round-trip through pandas that was the
  root cause of the Phase 15 UAT schema failures.

Network-free: `_fetch_openmeteo_weather` is patched with an AsyncMock returning
a dict derived from `tests/integration/fixtures/openmeteo_buf_2024_W06.json`.
The live Open-Meteo run is a separate CONTEXT.md-locked verification (Task 6).
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "openmeteo_buf_2024_W06.json"


@pytest.fixture
def openmeteo_payload() -> dict:
    """Load the recorded Open-Meteo hourly response once per test session."""
    return json.loads(FIXTURE_PATH.read_text())


@pytest.fixture
def ingester():
    """WeatherDataIngester with mocked settings (mirrors unit-test fixture)."""
    with patch("scripts.ingest_weather.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock()
        from scripts.ingest_weather import WeatherDataIngester

        return WeatherDataIngester()


@pytest.fixture
def venues_df() -> pd.DataFrame:
    """Load real venues.json (same pattern as unit tests)."""
    venues_data = json.loads(Path("data/venues.json").read_text())
    return pd.DataFrame(venues_data["venues"])


def _game_row(home_team: str, game_id: str) -> pd.DataFrame:
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


def _record_from_fixture(payload: dict, idx: int = 13) -> dict:
    """Translate hour-`idx` of the Open-Meteo hourly payload into the dict
    shape `fetch_game_weather` returns (see scripts/ingest_weather.py:144-162).

    Using idx=13 = 1 PM local ET, matching the default game_hour in ingest.
    """
    h = payload["hourly"]
    temp_f = h["temperature_2m"][idx]
    return {
        "temp_f": temp_f,
        "temp_c": round((temp_f - 32) * 5 / 9, 1) if temp_f is not None else None,
        "wind_mph": h["wind_speed_10m"][idx],
        "wind_direction": h["wind_direction_10m"][idx],
        "humidity_pct": h["relative_humidity_2m"][idx],
        "precip_mm": h["precipitation"][idx],
        "precip_prob": None,  # ERA5 reanalysis does not provide probability
        "condition": None,
        "condition_code": h["weather_code"][idx],
        "visibility_km": None,
        "dew_point_f": h["dew_point_2m"][idx],
        "apparent_temp_f": h["apparent_temperature"][idx],
        "snowfall_cm": h["snowfall"][idx],
        "wind_gusts_mph": h["wind_gusts_10m"][idx],
        "cloud_cover_pct": h["cloud_cover"][idx],
        "weather_code": h["weather_code"][idx],
    }


class TestWeatherIngestE2E:
    """Integration: ingest -> validate_bronze_to_silver -> WeatherSchema, end to end."""

    def test_outdoor_roundtrip_passes_schema(
        self, ingester, venues_df, openmeteo_payload
    ):
        """BUF outdoor game: fixture -> _fetch_openmeteo_weather (mocked)
        -> fetch_weather_for_games -> WeatherSchema validation passes with
        float wind_direction in [0, 360)."""
        from data.quality_gates import validate_bronze_to_silver
        from data.schemas import WeatherSchema

        record = _record_from_fixture(openmeteo_payload, idx=13)
        with patch.object(
            ingester,
            "_fetch_openmeteo_weather",
            AsyncMock(return_value=record),
        ):
            df = ingester.fetch_weather_for_games(
                _game_row("BUF", "2024_W06_KC@BUF"),
                venues_df,
                forecast_time=datetime(2024, 10, 11, 22, 0, tzinfo=UTC),
            )

        validated = validate_bronze_to_silver(df, WeatherSchema)
        assert len(validated) == 1
        row = validated.iloc[0]
        assert row["is_outdoor"] == True  # noqa: E712
        wd = row["wind_direction"]
        assert wd is not None and not pd.isna(wd)
        assert 0 <= float(wd) < 360, f"wind_direction out of range: {wd!r}"

    def test_indoor_roundtrip_passes_schema(self, ingester, venues_df):
        """LV indoor game: NO Open-Meteo call (indoor path short-circuits).
        Tests the None -> NaN -> None round-trip through pandas DataFrame that
        was the Phase 15 UAT failure."""
        from data.quality_gates import validate_bronze_to_silver
        from data.schemas import WeatherSchema

        df = ingester.fetch_weather_for_games(
            _game_row("LV", "2024_W06_KC@LV"),
            venues_df,
            forecast_time=datetime(2024, 10, 11, 22, 0, tzinfo=UTC),
        )

        validated = validate_bronze_to_silver(df, WeatherSchema)
        assert len(validated) == 1
        row = validated.iloc[0]
        assert row["is_outdoor"] == False  # noqa: E712
        # None -> NaN through DataFrame; after validation these are either
        # Python None (if quality_gates converts back) or pandas NaN. Either is
        # acceptable -- what matters is no validation error was raised.
        assert row["temp_f"] is None or pd.isna(row["temp_f"])
        assert row["humidity_pct"] is None or pd.isna(row["humidity_pct"])
        assert row["condition_code"] is None or pd.isna(row["condition_code"])
        # Indoor record emits real zero wind_mph + precip_mm -- must NOT be NaN.
        assert row["wind_mph"] == 0.0
        assert row["precip_mm"] == 0.0

    def test_new_openmeteo_fields_preserved_through_silver(
        self, ingester, venues_df, openmeteo_payload
    ):
        """The 5 Open-Meteo fields added to WeatherSchema persist through
        validate_bronze_to_silver (previously silently dropped by extra='ignore')."""
        from data.quality_gates import validate_bronze_to_silver
        from data.schemas import WeatherSchema

        record = _record_from_fixture(openmeteo_payload, idx=13)
        with patch.object(
            ingester,
            "_fetch_openmeteo_weather",
            AsyncMock(return_value=record),
        ):
            df = ingester.fetch_weather_for_games(
                _game_row("BUF", "2024_W06_KC@BUF"),
                venues_df,
                forecast_time=datetime(2024, 10, 11, 22, 0, tzinfo=UTC),
            )

        validated = validate_bronze_to_silver(df, WeatherSchema)
        assert len(validated) == 1
        row = validated.iloc[0]

        for col in (
            "dew_point_f",
            "apparent_temp_f",
            "snowfall_cm",
            "wind_gusts_mph",
            "cloud_cover_pct",
        ):
            assert col in validated.columns, f"{col} dropped from validated DataFrame"
            val = row[col]
            # Real float values from the fixture (not None/NaN, since record dict
            # was populated from real Open-Meteo data).
            assert val is not None and not pd.isna(val), f"{col} is None/NaN"
            assert isinstance(float(val), float)
