"""The live weather path answers for a game that has not happened -- or refuses by name.

WHAT THIS MODULE IS FOR
-----------------------
`scripts/ingest_weather.py` pointed at the Open-Meteo ARCHIVE endpoint, which is a
reanalysis product: it can only describe weather that has already occurred. A 2026 kickoff
has not occurred, so the live path could never have answered for one, and the fourteen rows
in `data/silver/weather.parquet` -- all of them 2024 Week 6 -- are the evidence it never
tried. R8 moves that path to the FORECAST endpoint.

Moving it creates a new way to be wrong. A forecast has a HORIZON, and a request beyond it
comes back empty or partial rather than as an error. The tempting repairs -- fall back to
the archive, or fill the gap with a seasonal average -- are both the fabricated-data class
this project has already disclosed once. So the horizon is refused BY NAME, before the
request is made, and the refusal is proven here.

WHY EVERY TEST IN THIS MODULE IS OFFLINE
----------------------------------------
`FORECAST_HORIZON_DAYS` is OURS, not the provider's (D33-26), and `as_of_utc` is INJECTED
rather than read from a process clock. Those two decisions together are what make the
refusal provable with no network and no dependence on the day the suite runs. A horizon
read from the provider at run time would need a live call to test; a horizon compared
against a process clock would pass in September and fail in December.

NO TEST HERE MAKES A NETWORK CALL. `httpx.AsyncClient` is replaced by a counting factory
whose client REFUSES to be used, so a request that should never have been made surfaces as
a failure rather than as a slow test.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from tests import phase33_state

AS_OF = datetime(2026, 9, 11, 20, 0, tzinfo=UTC)

# A venue record shaped like one row of data/venues.json. Outdoor, so the forecast path
# is actually reached rather than short-circuited by the indoor branch.
OUTDOOR_VENUE: dict[str, Any] = {
    "stadium_id": "BUF00",
    "venue_id": "highmark_stadium",
    "venue_name": "Highmark Stadium",
    "latitude": 42.7738,
    "longitude": -78.787,
    "roof_type": "outdoor",
    "timezone": "America/New_York",
    "home_teams": ["BUF"],
}

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


class RefusingClientFactory:
    """Stands in for `httpx.AsyncClient` and counts every construction.

    The client it hands back raises on `get`, so a request the horizon check should have
    prevented fails loudly instead of silently reaching the network. Counting the
    CONSTRUCTION rather than the call is deliberate: the production path builds the client
    inside the async wrapper, so a construction with zero gets is still evidence the
    refusal came too late.
    """

    def __init__(self) -> None:
        self.constructions = 0
        self.gets = 0

    def __call__(self, *args: object, **kwargs: object) -> RefusingClientFactory:
        self.constructions += 1
        return self

    async def __aenter__(self) -> RefusingClientFactory:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    async def get(self, *args: object, **kwargs: object) -> None:
        self.gets += 1
        raise AssertionError(
            "the forecast path made an HTTP request. Every test in this module is "
            "offline by construction; a request here means the beyond-horizon refusal "
            "was raised AFTER the call rather than before it."
        )


@pytest.fixture
def ingester():
    """A WeatherDataIngester with settings mocked, matching the existing weather tests."""
    with patch("scripts.ingest_weather.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock()
        from scripts.ingest_weather import WeatherDataIngester

        return WeatherDataIngester()


@pytest.fixture
def venues_df() -> pd.DataFrame:
    """A one-row venue frame carrying the IANA `timezone` Plan 33-06 put on all 38."""
    return pd.DataFrame([OUTDOOR_VENUE])


def _game(kickoff_utc: datetime, game_id: str = "2026_W02_KC@BUF") -> pd.DataFrame:
    """One game whose kickoff INSTANT is exactly *kickoff_utc*.

    `kickoff_et` is written as an AWARE value on purpose: `kickoff_wall_clock_et`
    CONVERTS an aware cell rather than relabelling it, so the instant the test states is
    the instant the production path sees.
    """
    return pd.DataFrame(
        [
            {
                "game_id": game_id,
                "season": 2026,
                "week": 2,
                "home_team": "BUF",
                "away_team": "KC",
                "kickoff_et": kickoff_utc,
                "stadium_id": "BUF00",
                "neutral_site": False,
            }
        ]
    )


class TestTheHorizonConstantIsOursAndIsRecorded:
    """D33-26: the horizon is a declared constant, not a value read from the provider."""

    def test_the_module_constant_is_a_positive_integer(self) -> None:
        import scripts.ingest_weather as ingest

        assert isinstance(ingest.FORECAST_HORIZON_DAYS, int)
        assert ingest.FORECAST_HORIZON_DAYS > 0

    def test_the_module_constant_equals_the_recorded_state(self) -> None:
        """Two homes for one number is the second-list failure this phase guards against."""
        import scripts.ingest_weather as ingest

        assert ingest.FORECAST_HORIZON_DAYS == phase33_state.FORECAST_HORIZON_DAYS

    def test_our_horizon_is_inside_the_providers_measured_window(self) -> None:
        """A horizon wider than the provider's is a promise we cannot keep.

        The provider's window was PROBED on 2026-09-12 and recorded in
        `phase33_state.FORECAST_PROVIDER_HORIZON_DAYS`; this asserts the relationship
        between the two numbers rather than either number alone.
        """
        import scripts.ingest_weather as ingest

        assert (
            ingest.FORECAST_HORIZON_DAYS <= phase33_state.FORECAST_PROVIDER_HORIZON_DAYS
        )

    def test_the_forecast_endpoint_is_not_the_archive_endpoint(self) -> None:
        import scripts.ingest_weather as ingest

        archive = getattr(
            ingest, "OPEN_METEO_URL", getattr(ingest, "ARCHIVE_ENDPOINT_URL", "")
        )
        assert archive != ingest.FORECAST_ENDPOINT_URL
        assert "archive" not in ingest.FORECAST_ENDPOINT_URL


class TestTheHorizonIsMeasuredBetweenTwoInstants:
    """The comparison is instant-to-instant against an INJECTED `as_of_utc`."""

    def test_a_kickoff_inside_the_horizon_is_accepted(self) -> None:
        import scripts.ingest_weather as ingest

        kickoff = AS_OF + timedelta(days=3)
        cutoff = ingest.assert_within_forecast_horizon(
            "2026_W02_KC@BUF", kickoff, as_of_utc=AS_OF
        )
        assert cutoff == AS_OF + timedelta(days=ingest.FORECAST_HORIZON_DAYS)

    def test_a_kickoff_beyond_the_horizon_is_refused_by_name(self) -> None:
        import scripts.ingest_weather as ingest

        kickoff = AS_OF + timedelta(days=ingest.FORECAST_HORIZON_DAYS + 5)
        with pytest.raises(ingest.BeyondForecastHorizonError) as caught:
            ingest.assert_within_forecast_horizon(
                "2026_W99_AAA@BBB", kickoff, as_of_utc=AS_OF
            )

        message = str(caught.value)
        assert "2026_W99_AAA@BBB" in message
        assert kickoff.isoformat() in message
        assert str(ingest.FORECAST_HORIZON_DAYS) in message
        expected_cutoff = AS_OF + timedelta(days=ingest.FORECAST_HORIZON_DAYS)
        assert expected_cutoff.isoformat() in message

    def test_a_kickoff_exactly_at_the_boundary_is_inside(self) -> None:
        """The convention is stated in the constant's docstring and asserted here.

        Leaving the boundary to whichever comparison was typed first is how a one-day
        disagreement between the code and its own documentation survives a review.
        """
        import scripts.ingest_weather as ingest

        boundary = AS_OF + timedelta(days=ingest.FORECAST_HORIZON_DAYS)
        assert (
            ingest.assert_within_forecast_horizon(
                "2026_W02_KC@BUF", boundary, as_of_utc=AS_OF
            )
            == boundary
        )

    def test_one_microsecond_past_the_boundary_is_outside(self) -> None:
        import scripts.ingest_weather as ingest

        boundary = AS_OF + timedelta(days=ingest.FORECAST_HORIZON_DAYS)
        with pytest.raises(ingest.BeyondForecastHorizonError):
            ingest.assert_within_forecast_horizon(
                "2026_W02_KC@BUF",
                boundary + timedelta(microseconds=1),
                as_of_utc=AS_OF,
            )

    def test_a_naive_kickoff_is_refused_rather_than_assumed_to_be_utc(self) -> None:
        """A naive instant has no meaning to compare. Guessing one is how a five-hour
        error enters a fence that reads as though it passed."""
        import scripts.ingest_weather as ingest

        with pytest.raises(ValueError, match="kickoff_utc"):
            ingest.assert_within_forecast_horizon(
                "2026_W02_KC@BUF",
                datetime(2026, 9, 14, 17, 0),
                as_of_utc=AS_OF,
            )

    def test_a_naive_as_of_is_refused(self) -> None:
        import scripts.ingest_weather as ingest

        with pytest.raises(ValueError, match="as_of_utc"):
            ingest.assert_within_forecast_horizon(
                "2026_W02_KC@BUF",
                AS_OF + timedelta(days=1),
                as_of_utc=datetime(2026, 9, 11, 20, 0),
            )

    def test_the_verdict_moves_with_as_of_and_not_with_the_calendar(self) -> None:
        """The SAME kickoff is inside for one `as_of_utc` and outside for another.

        This is the property that makes the horizon testable at all: nothing here depends
        on the day the suite runs, and a process clock could not produce both answers.
        """
        import scripts.ingest_weather as ingest

        kickoff = datetime(2026, 12, 25, 18, 0, tzinfo=UTC)
        near = kickoff - timedelta(days=1)
        far = kickoff - timedelta(days=ingest.FORECAST_HORIZON_DAYS + 1)

        ingest.assert_within_forecast_horizon("2026_W17_X@Y", kickoff, as_of_utc=near)
        with pytest.raises(ingest.BeyondForecastHorizonError):
            ingest.assert_within_forecast_horizon(
                "2026_W17_X@Y", kickoff, as_of_utc=far
            )


class TestTheRefusalHappensBeforeTheRequest:
    """The out-of-horizon path makes no HTTP call and never reaches for the archive."""

    def test_a_beyond_horizon_week_makes_no_http_call(
        self, ingester, venues_df
    ) -> None:
        import scripts.ingest_weather as ingest

        factory = RefusingClientFactory()
        games = _game(AS_OF + timedelta(days=ingest.FORECAST_HORIZON_DAYS + 2))

        with patch("scripts.ingest_weather.httpx.AsyncClient", factory):
            with pytest.raises(ingest.BeyondForecastHorizonError):
                ingester.fetch_forecast_for_games(games, venues_df, as_of_utc=AS_OF)

        assert factory.constructions == 0, (
            "an HTTP client was constructed for a game beyond the forecast horizon. "
            "The refusal must be raised BEFORE the request, not after it."
        )
        assert factory.gets == 0

    def test_a_beyond_horizon_week_does_not_fall_back_to_the_archive(
        self, ingester, venues_df
    ) -> None:
        """No archive fetch, and no imputed record: the week raises and yields nothing.

        The archive fetch is patched IN THE QUARANTINE MODULE, which is the only place
        it exists. Patching it on the live ingester would be impossible now -- that
        method is not on `WeatherDataIngester` any more -- and asserting a call count
        on a function that cannot be reached would be a test that cannot fail. So the
        call counter is placed where a fallback would actually have to land.
        """
        import scripts.backfill_historical_weather as backfill
        import scripts.ingest_weather as ingest

        archive_calls: list[object] = []

        async def _record_archive(*args: object, **kwargs: object) -> dict[str, Any]:
            archive_calls.append(args)
            return dict(FORECAST_RECORD)

        games = _game(AS_OF + timedelta(days=ingest.FORECAST_HORIZON_DAYS + 2))

        with patch.object(
            backfill, "fetch_game_weather", AsyncMock(side_effect=_record_archive)
        ):
            with pytest.raises(ingest.BeyondForecastHorizonError):
                ingester.fetch_forecast_for_games(games, venues_df, as_of_utc=AS_OF)

        assert archive_calls == [], (
            "the beyond-horizon path called the ARCHIVE fetch. Filling a missing live "
            "forecast from the archive is the fabricated-data class this plan's own "
            "prohibition names."
        )

    def test_the_live_ingester_has_no_archive_fetch_to_fall_back_to(self) -> None:
        """Structural, and stronger than the call count above.

        The live ingester does not merely decline to call the archive fetch -- it does
        not have one. A fallback would have to be WRITTEN, not merely permitted.
        """
        from scripts.ingest_weather import WeatherDataIngester

        assert not hasattr(WeatherDataIngester, "_fetch_openmeteo_weather")
        assert not hasattr(WeatherDataIngester, "fetch_weather_for_games")


class TestAWeekInsideTheHorizonIsPopulated:
    """R8's headline, at the unit tier: real non-null values for an outdoor game."""

    def test_an_outdoor_game_inside_the_horizon_gets_non_null_values(
        self, ingester, venues_df
    ) -> None:
        games = _game(AS_OF + timedelta(days=4))

        with patch.object(
            ingester,
            "_fetch_openmeteo_forecast",
            AsyncMock(return_value=dict(FORECAST_RECORD)),
        ):
            frame = ingester.fetch_forecast_for_games(games, venues_df, as_of_utc=AS_OF)

        assert len(frame) == 1
        row = frame.iloc[0]
        assert bool(row["is_outdoor"]) is True
        for column in ("temp_f", "wind_mph", "precip_mm"):
            assert row[column] is not None and not pd.isna(row[column]), column
        assert row["temp_f"] == FORECAST_RECORD["temp_f"]
        assert row["wind_mph"] == FORECAST_RECORD["wind_mph"]


class TestAPartialPayloadIsRejectedBeforeAnythingIsWritten:
    """T-33-47c: a half-populated week is indistinguishable from a complete one."""

    def test_a_missing_game_is_named(self) -> None:
        import scripts.ingest_weather as ingest

        requested = ["2026_W02_A@B", "2026_W02_C@D", "2026_W02_E@F"]
        returned = pd.DataFrame([{"game_id": "2026_W02_A@B"}])

        with pytest.raises(ingest.IncompleteForecastPayloadError) as caught:
            ingest.assert_complete_forecast_coverage(requested, returned)

        message = str(caught.value)
        assert "2026_W02_C@D" in message
        assert "2026_W02_E@F" in message
        # The error carries the SHORTFALL, not the whole request: "2 of 3" is what
        # tells a reader the payload was partial rather than empty.
        assert "2 of 3" in message
        assert caught.value.absent_game_ids == ("2026_W02_C@D", "2026_W02_E@F")

    def test_complete_coverage_passes(self) -> None:
        import scripts.ingest_weather as ingest

        requested = ["2026_W02_A@B", "2026_W02_C@D"]
        returned = pd.DataFrame([{"game_id": gid} for gid in requested])
        ingest.assert_complete_forecast_coverage(requested, returned)

    def test_an_empty_payload_for_a_non_empty_request_is_rejected(self) -> None:
        import scripts.ingest_weather as ingest

        with pytest.raises(ingest.IncompleteForecastPayloadError):
            ingest.assert_complete_forecast_coverage(
                ["2026_W02_A@B"], pd.DataFrame(columns=["game_id"])
            )


class TestTheWriteHelperRoutesThroughTheStorageLayer:
    """The atomic per-week write is a STORAGE concern and lives in the storage layer."""

    def test_it_delegates_to_upsert_silver(self) -> None:
        """Reviewer-confirmed: importing the bet-list module's private replacement
        helper to serve the storage layer would be inverted coupling. `upsert_silver`
        already reaches atomic parquet replacement."""
        import inspect

        import scripts.ingest_weather as ingest

        source = inspect.getsource(ingest.write_week_weather_atomically)
        assert "upsert_silver" in source
        assert "_replace_atomically" not in source
