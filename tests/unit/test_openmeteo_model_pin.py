"""The live weather ingest pins its Open-Meteo model and stamps the model run's own time.

WHY (Plan 33.2-27 Task 2, D33.2-18)
-----------------------------------
A forecast response never names the model or the run that produced it. Without ``models=`` the
provider answers with ``best_match``, whose answering model is unnamed, so no run time can be
obtained at all. Pinning ``gfs_global`` makes the model known, and the free, quota-exempt
``meta.json`` publishes its ``last_run_availability_time``. ``gfs_global`` is served from TWO
domains (``ncep_gfs013`` and ``ncep_gfs025``), so the honest stamp is the LATER of the two.

A missing stamp is a NAMED REFUSAL, never our own clock. And no capture is ever written for a
game that has already kicked off (the ``2026_W02_DET@BUF`` row, captured two days after the
game, is the defect this closes).

NO NETWORK: every fetch here is a stand-in.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pandas as pd
import pytest

import scripts.ingest_weather as weather
from data.quality_gates import validate_bronze_to_silver
from data.schemas import WeatherSchema

AS_OF = datetime(2026, 9, 26, 20, 0, tzinfo=UTC)
KICKOFF = AS_OF + timedelta(days=1)
GFS013_STAMP = 1789515171
GFS025_STAMP = 1789497255
MODEL_STAMP = datetime(2026, 9, 26, 16, 5, tzinfo=UTC)

VENUES = pd.DataFrame(
    [
        {
            "stadium_id": "BUF00",
            "venue_id": "highmark_stadium",
            "latitude": 42.7738,
            "longitude": -78.787,
            "roof_type": "outdoor",
            "timezone": "America/New_York",
            "home_teams": ["BUF"],
        },
        {
            "stadium_id": "GNB00",
            "venue_id": "lambeau_field",
            "latitude": 44.5013,
            "longitude": -88.0622,
            "roof_type": "outdoor",
            "timezone": "America/Chicago",
            "home_teams": ["GB"],
        },
    ]
)

FORECAST_RECORD: dict[str, Any] = {
    "temp_f": 51.2,
    "temp_c": 10.7,
    "wind_mph": 11.4,
    "wind_direction": 240.0,
    "humidity_pct": 72.0,
    "precip_mm": 0.6,
    "precip_prob": None,
    "condition": None,
    "condition_code": 61,
    "visibility_km": None,
    "dew_point_f": 42.1,
    "apparent_temp_f": 47.0,
    "snowfall_cm": 0.0,
    "wind_gusts_mph": 19.3,
    "cloud_cover_pct": 88.0,
    "weather_code": 61,
}


def _game(game_id: str, stadium_id: str, home: str, kickoff: datetime) -> dict:
    return {
        "game_id": game_id,
        "season": 2026,
        "week": 4,
        "home_team": home,
        "away_team": game_id.split("_")[2].split("@", maxsplit=1)[0],
        "kickoff_et": kickoff,
        "stadium_id": stadium_id,
        "neutral_site": False,
    }


@pytest.fixture
def ingester():
    with patch("scripts.ingest_weather.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock()
        return weather.WeatherDataIngester()


class _Response:
    def __init__(self, payload: Any, status: int = 200) -> None:
        self._payload = payload
        self.status_code = status

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("GET", "https://example.invalid")
            raise httpx.HTTPStatusError(
                "server error",
                request=request,
                response=httpx.Response(self.status_code, request=request),
            )

    def json(self) -> Any:
        return self._payload


def _meta_get(stamps: dict[str, Any], calls: list[str] | None = None):
    def _get(url: str, **_kwargs):
        if calls is not None:
            calls.append(url)
        for domain, stamp in stamps.items():
            if f"/{domain}/" in url:
                if isinstance(stamp, Exception):
                    raise stamp
                return _Response(stamp)
        raise AssertionError(f"unexpected URL {url}")

    return _get


class TestTheRequestPinsTheModel:
    """The forecast request names its model, so the answering model is known."""

    def test_the_request_carries_models_gfs_global(self):
        captured: dict[str, Any] = {}

        class _Client:
            async def get(self, url, params=None):
                captured.update(params or {})
                return _Response(
                    {
                        "hourly": {
                            "temperature_2m": [50.0] * 24,
                            "wind_speed_10m": [5.0] * 24,
                            "wind_direction_10m": [180.0] * 24,
                            "relative_humidity_2m": [60.0] * 24,
                            "precipitation": [0.0] * 24,
                            "weather_code": [1] * 24,
                            "dew_point_2m": [40.0] * 24,
                            "apparent_temperature": [48.0] * 24,
                            "snowfall": [0.0] * 24,
                            "wind_gusts_10m": [9.0] * 24,
                            "cloud_cover": [20.0] * 24,
                        }
                    }
                )

        asyncio.run(
            weather.fetch_game_forecast(
                _Client(), 42.77, -78.78, "2026-09-27", 13, "America/New_York"
            )
        )
        assert captured.get("models") == "gfs_global"
        assert weather.OPENMETEO_FORECAST_MODEL == "gfs_global"


class TestTheModelRunStamp:
    """``last_run_availability_time`` from ``meta.json``, the later of two domains."""

    def test_the_stamp_is_the_later_of_the_two_domains(self):
        stamp = weather.fetch_model_run_available_at(
            get=_meta_get(
                {
                    "ncep_gfs013": {"last_run_availability_time": GFS013_STAMP},
                    "ncep_gfs025": {"last_run_availability_time": GFS025_STAMP},
                }
            )
        )
        assert stamp == datetime.fromtimestamp(max(GFS013_STAMP, GFS025_STAMP), tz=UTC)
        assert stamp.tzinfo is not None

    def test_the_order_of_the_domains_does_not_decide(self):
        """Control: the later stamp wins when it is the SECOND domain's too."""
        stamp = weather.fetch_model_run_available_at(
            get=_meta_get(
                {
                    "ncep_gfs013": {"last_run_availability_time": GFS025_STAMP},
                    "ncep_gfs025": {"last_run_availability_time": GFS013_STAMP},
                }
            )
        )
        assert stamp == datetime.fromtimestamp(GFS013_STAMP, tz=UTC)

    def test_both_meta_json_urls_are_read(self):
        calls: list[str] = []
        weather.fetch_model_run_available_at(
            get=_meta_get(
                {
                    "ncep_gfs013": {"last_run_availability_time": GFS013_STAMP},
                    "ncep_gfs025": {"last_run_availability_time": GFS025_STAMP},
                },
                calls,
            )
        )
        assert sorted(calls) == [
            "https://api.open-meteo.com/data/ncep_gfs013/static/meta.json",
            "https://api.open-meteo.com/data/ncep_gfs025/static/meta.json",
        ]

    def test_a_network_failure_is_a_named_refusal(self):
        with pytest.raises(weather.ModelRunTimeUnavailableError, match="ncep_gfs025"):
            weather.fetch_model_run_available_at(
                get=_meta_get(
                    {
                        "ncep_gfs013": {"last_run_availability_time": GFS013_STAMP},
                        "ncep_gfs025": httpx.ConnectError("offline"),
                    }
                )
            )

    def test_a_server_error_is_a_named_refusal(self):
        def _get(url: str, **_kwargs):
            return _Response({}, status=503)

        with pytest.raises(weather.ModelRunTimeUnavailableError):
            weather.fetch_model_run_available_at(get=_get)

    def test_a_response_with_no_availability_field_raises_rather_than_defaulting(self):
        with pytest.raises(
            weather.ModelRunTimeUnavailableError, match="last_run_availability_time"
        ):
            weather.fetch_model_run_available_at(
                get=_meta_get(
                    {
                        "ncep_gfs013": {"last_run_initialisation_time": GFS013_STAMP},
                        "ncep_gfs025": {"last_run_availability_time": GFS025_STAMP},
                    }
                )
            )

    def test_the_refusal_is_a_weather_data_error(self):
        """So every existing weather catch treats it as a failed fetch, not a crash."""
        from utils.exceptions import WeatherDataError

        assert issubclass(weather.ModelRunTimeUnavailableError, WeatherDataError)


class TestEveryForecastRowCarriesBothProofs:
    """The model run's own time AND our capture instant, on every forecast row."""

    def test_the_rows_carry_the_model_stamp_and_the_capture_instant(self, ingester):
        order: list[str] = []

        async def _fetch(*_args):
            order.append("forecast")
            return dict(FORECAST_RECORD)

        def _stamp():
            order.append("meta")
            return MODEL_STAMP

        games = pd.DataFrame(
            [
                _game("2026_W04_KC@BUF", "BUF00", "BUF", KICKOFF),
                _game("2026_W04_MIN@GB", "GNB00", "GB", KICKOFF),
            ]
        )
        with (
            patch.object(ingester, "_fetch_openmeteo_forecast", _fetch),
            patch.object(ingester, "_fetch_model_run_available_at", _stamp),
        ):
            frame = ingester.fetch_forecast_for_games(
                games, VENUES, forecast_time=AS_OF, as_of_utc=AS_OF
            )

        assert list(frame["model_run_available_at"]) == [MODEL_STAMP, MODEL_STAMP]
        assert list(frame["forecast_time"]) == [AS_OF, AS_OF]
        assert order == ["forecast", "forecast", "meta"], (
            "the model stamp must be read AFTER the forecasts, so it is an upper bound on "
            "the run that served them"
        )

    def test_the_schema_declares_the_stamp(self, ingester):
        """Pydantic defaults to extra='ignore': an undeclared column vanishes silently."""

        async def _fetch(*_args):
            return dict(FORECAST_RECORD)

        games = pd.DataFrame([_game("2026_W04_KC@BUF", "BUF00", "BUF", KICKOFF)])
        with (
            patch.object(ingester, "_fetch_openmeteo_forecast", _fetch),
            patch.object(
                ingester, "_fetch_model_run_available_at", lambda: MODEL_STAMP
            ),
        ):
            frame = ingester.fetch_forecast_for_games(
                games, VENUES, forecast_time=AS_OF, as_of_utc=AS_OF
            )
        validated = validate_bronze_to_silver(frame, WeatherSchema)
        assert pd.Timestamp(validated["model_run_available_at"].iloc[0]) == MODEL_STAMP

    def test_a_naive_model_stamp_is_refused(self):
        with pytest.raises(ValueError, match="model_run_available_at"):
            WeatherSchema(
                game_id="2026_W04_KC@BUF",
                forecast_time=AS_OF,
                game_time=KICKOFF,
                is_outdoor=True,
                weather_coverage=True,
                weather_source="forecast",
                model_run_available_at=datetime(2026, 9, 26, 16, 5),
            )

    def test_a_meta_refusal_writes_nothing(self, ingester, tmp_path: Path):
        async def _fetch(*_args):
            return dict(FORECAST_RECORD)

        def _refuse():
            raise weather.ModelRunTimeUnavailableError("meta.json unreachable")

        games = pd.DataFrame([_game("2026_W04_KC@BUF", "BUF00", "BUF", KICKOFF)])
        with (
            patch.object(ingester, "_fetch_openmeteo_forecast", _fetch),
            patch.object(ingester, "_fetch_model_run_available_at", _refuse),
            pytest.raises(weather.ModelRunTimeUnavailableError),
        ):
            ingester.ingest_week_forecast(
                games, VENUES, as_of_utc=AS_OF, base_path=tmp_path
            )
        assert not (tmp_path / "silver" / "weather.parquet").exists()


class TestNoCaptureAfterKickoff:
    """A game that has already kicked off is never captured, by name."""

    def test_the_fetch_refuses_a_played_game_by_name(self, ingester):
        played = pd.DataFrame(
            [_game("2026_W04_KC@BUF", "BUF00", "BUF", AS_OF - timedelta(hours=2))]
        )
        with pytest.raises(weather.CaptureAfterKickoffError, match="2026_W04_KC@BUF"):
            ingester.fetch_forecast_for_games(played, VENUES, as_of_utc=AS_OF)

    def test_the_week_ingest_skips_a_played_game_and_writes_the_rest(
        self, ingester, tmp_path: Path
    ):
        fetched: list[str] = []

        async def _fetch(latitude, *_args):
            fetched.append(str(latitude))
            return dict(FORECAST_RECORD)

        games = pd.DataFrame(
            [
                _game("2026_W04_KC@BUF", "BUF00", "BUF", AS_OF - timedelta(hours=2)),
                _game("2026_W04_MIN@GB", "GNB00", "GB", KICKOFF),
            ]
        )
        with (
            patch.object(ingester, "_fetch_openmeteo_forecast", _fetch),
            patch.object(
                ingester, "_fetch_model_run_available_at", lambda: MODEL_STAMP
            ),
        ):
            written = ingester.ingest_week_forecast(
                games, VENUES, as_of_utc=AS_OF, base_path=tmp_path
            )

        assert list(written["game_id"]) == ["2026_W04_MIN@GB"]
        assert fetched == ["44.5013"], "the played game must not even be requested"
        stored = pd.read_parquet(tmp_path / "silver" / "weather.parquet")
        assert "2026_W04_KC@BUF" not in set(stored["game_id"])


class TestNoCaptureAfterTheLock:
    """33.2 review C1 CR-03: a game whose LOCK has passed is never re-captured.

    Silver weather is latest-wins by game_id. The week ingest left out only KICKED-OFF games,
    so a Saturday-evening run replaced each Sunday game's pre-lock forecast with a post-lock
    one, which the lock fence then refused: the game lost its only admissible forecast.
    """

    def test_a_locked_but_not_kicked_off_game_is_left_out_and_its_row_survives(
        self, ingester, tmp_path: Path
    ):
        sunday = pd.DataFrame([_game("2026_W04_KC@BUF", "BUF00", "BUF", KICKOFF)])
        lock = datetime(2026, 9, 26, 22, 0, tzinfo=UTC)  # Saturday 18:00 ET
        fetched: list[str] = []

        async def _fetch(latitude, *_args):
            fetched.append(str(latitude))
            return dict(FORECAST_RECORD)

        with (
            patch.object(ingester, "_fetch_openmeteo_forecast", _fetch),
            patch.object(
                ingester, "_fetch_model_run_available_at", lambda: MODEL_STAMP
            ),
        ):
            ingester.ingest_week_forecast(
                sunday, VENUES, as_of_utc=lock, forecast_time=lock, base_path=tmp_path
            )
            after = lock + timedelta(hours=1)
            written = ingester.ingest_week_forecast(
                sunday, VENUES, as_of_utc=after, forecast_time=after, base_path=tmp_path
            )

        assert written.empty, "a capture after the lock was written"
        assert len(fetched) == 1, "the locked game must not even be requested"
        stored = pd.read_parquet(tmp_path / "silver" / "weather.parquet")
        assert list(pd.to_datetime(stored["forecast_time"], utc=True)) == [lock], (
            "the at-lock forecast must survive a later run"
        )
