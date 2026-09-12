"""Weather data ingestion using Open-Meteo Historical Weather API.

Produces timestamped Bronze snapshots and upserts to Silver weather table.
Indoor games get zeroed weather fields (no API call).
Outdoor and retractable-roof games use real Open-Meteo data.
Missing weather for an outdoor game raises a hard error.
"""

import argparse
import asyncio
import sys
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import httpx
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from conf.settings import get_settings
from data.quality_gates import validate_bronze_to_silver
from data.schemas import WeatherSchema
from data.storage import load_dataframe, save_bronze_snapshot, upsert_silver
from features.contextual import (
    STADIUM_ID_ROUTING_FIRST_SEASON,
    game_is_neutral_site,
)
from utils import (
    DataIngestionError,
    get_current_nfl_week,
    get_logger,
    log_data_operation,
)
from utils.date_utils import kickoff_wall_clock_et
from utils.exceptions import WeatherDataError
from utils.game_id_utils import is_valid_game_id
from utils.team_data import normalize_team_abbreviation

logger = get_logger(__name__)

# Open-Meteo Historical Weather API endpoint
OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"

# ---------------------------------------------------------------------------
# THE FORECAST PATH (R8 / COLD-06, Plan 33-09 Task 1).
#
# The endpoint above is a REANALYSIS product: it describes weather that has
# already happened. It cannot answer for a 2026 kickoff, and the fourteen rows in
# data/silver/weather.parquet -- all 2024 Week 6 -- are the evidence this path
# never ran forward. Everything below is the path that can.
# ---------------------------------------------------------------------------

# The LIVE forecast endpoint. Distinct from the archive endpoint above; CONFIRMED
# against the provider's documentation and PROBED from this checkout on
# 2026-09-12, which returned 200 with the same twelve hourly variable names the
# archive call already uses and accepted the same start_date/end_date pair.
FORECAST_ENDPOINT_URL = "https://api.open-meteo.com/v1/forecast"

FORECAST_HORIZON_DAYS: int = 14
"""How many days ahead THIS PROJECT will ask the forecast endpoint for (D33-26).

OURS, NOT THE PROVIDER'S, AND THAT IS THE POINT. Open-Meteo's own window was
measured on 2026-09-12 by asking for a date past it, which it reports in an error
body rather than in prose::

    400 {"error":true,"reason":"Parameter 'start_date' is out of allowed range
         from 2026-06-11 to 2026-09-27"}

That upper bound was today+15 on the day it was measured, and today+14 returned
24 non-null hours. The documented parameter is ``forecast_days`` (0-16, counting
today as day one), which reaches the same place. So 14 sits INSIDE the provider's
window by a day, deliberately:

* A horizon WIDER than the provider's is a promise we cannot keep -- the request
  simply fails, or worse comes back partial.
* A horizon EQUAL to the provider's turns any future narrowing on their side into
  a silent empty response instead of a named refusal on ours.
* A horizon that is OURS makes the refusal testable OFFLINE. No test in this
  module's suite needs the network to prove a beyond-horizon kickoff is refused.

BOUNDARY CONVENTION, chosen and stated rather than left to whichever comparison
was typed first: a kickoff EXACTLY ``FORECAST_HORIZON_DAYS`` after ``as_of_utc``
is INSIDE the horizon. The comparison is a strict ``>`` between two
TIMEZONE-AWARE INSTANTS, never between two calendar dates -- a date comparison
would put a Monday 00:15 kickoff and a Monday 23:59 kickoff on the same footing.
It is asserted in tests/unit/test_weather_forecast_horizon.py.
"""


# RED-phase stubs for Plan 33-09 Task 2. Replaced in that task's GREEN commit.
WEATHER_SOURCE_VOCABULARY: tuple[str, ...] = ()


def validate_weather_source(value):
    """RED stub -- replaced in the Task 2 GREEN commit."""
    raise NotImplementedError("validate_weather_source is not implemented yet")


class BeyondForecastHorizonError(WeatherDataError):
    """A kickoff lies beyond the horizon this project declares it can forecast.

    A FOURTH member of the existing ``WeatherDataError`` family, raised BEFORE the
    request rather than after it. The two tempting repairs -- read the archive
    endpoint instead, or fill the gap with a seasonal average -- are both the
    fabricated-data class this project has already disclosed once, so the refusal
    is terminal and names the game.
    """


class IncompleteForecastPayloadError(WeatherDataError):
    """The provider answered, but not for every game that was asked about.

    A partially-populated week is indistinguishable from a complete one once it is
    on disk, which makes it the quieter form of the same fabrication. Raised
    BEFORE any write, with the absent ``game_id`` values named.
    """

    def __init__(self, message: str, absent_game_ids: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.absent_game_ids = absent_game_ids


@dataclass(frozen=True)
class ForecastHour:
    """WHICH hour of WHICH local day a game's forecast must be read at.

    A forecast is an array of hours. Reading the wrong index returns a real,
    plausible, internally-consistent value for the wrong moment, and when the venue
    is on another continent it is also the wrong DAY -- there is nothing in the
    number itself to say so. This carries the decision explicitly so it can be
    asserted rather than inferred from a request URL.
    """

    game_id: str
    timezone: str
    local_date: str
    hour: int
    kickoff_utc: datetime
    local_instant: datetime


def _require_aware(value: datetime, name: str) -> datetime:
    """Return *value* unchanged, or raise naming the argument that was naive.

    A naive instant has no meaning to compare against another instant. Guessing a
    zone for it is how a four- or five-hour error enters a fence that reads as
    though it passed.
    """
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError(
            f"{name} must be a TIMEZONE-AWARE datetime; got {value!r}. A naive "
            "value would be compared against an instant under an assumed zone, "
            "which is how a multi-hour error enters a check that still reads as "
            "passing."
        )
    return value


def assert_within_forecast_horizon(
    game_id: str,
    kickoff_utc: datetime,
    *,
    as_of_utc: datetime,
) -> datetime:
    """Refuse a kickoff beyond ``FORECAST_HORIZON_DAYS``, by name, before any request.

    ``as_of_utc`` is INJECTED and never read from a process clock. That is what
    makes this reproducible: the caller states the instant, so the same input gives
    the same verdict in September and in December, and the refusal can be proven
    with no network at all.

    Args:
        game_id: The game being asked about, named in the refusal.
        kickoff_utc: The kickoff INSTANT. Must be timezone-aware.
        as_of_utc: The instant the horizon is measured FROM. Must be aware.

    Returns:
        The cutoff instant, so a caller can report it without recomputing it.

    Raises:
        ValueError: either instant is naive.
        BeyondForecastHorizonError: the kickoff is strictly past the cutoff.
    """
    _require_aware(kickoff_utc, "kickoff_utc")
    _require_aware(as_of_utc, "as_of_utc")

    cutoff = as_of_utc + timedelta(days=FORECAST_HORIZON_DAYS)
    if kickoff_utc > cutoff:
        raise BeyondForecastHorizonError(
            f"{game_id} kicks off at {kickoff_utc.isoformat()}, which is beyond the "
            f"{FORECAST_HORIZON_DAYS}-day forecast horizon measured from "
            f"{as_of_utc.isoformat()} (cutoff {cutoff.isoformat()}). NO row is "
            "written for it and NO value is invented: the archive endpoint cannot "
            "answer for a game that has not happened, and a seasonal average is an "
            "imputed constant wearing a statistic's clothes. Re-run the ingest "
            "closer to kickoff."
        )
    return cutoff


def select_forecast_hour_for_kickoff(
    game: Any,
    venue: Any,
    *,
    as_of_utc: datetime,
) -> ForecastHour:
    """Resolve the local day and hour a game's forecast must be read at.

    THE VENUE'S OWN IANA ZONE, taken from the ``timezone`` field Plan 33-06 put on
    all 38 records in ``data/venues.json``. Never a fixed zone and never the home
    team's: eight of 2026's games are international, the Maracana game is nominally
    a Dallas home game, and the Melbourne game read in Eastern time is the wrong
    hour on the wrong DATE. This is COLD-09's defect expressed in time rather than
    in space, and it has the same fix -- resolve the venue, then use what the venue
    says.

    The horizon check runs HERE, so a caller that reaches for the hour directly
    cannot skip it.

    Args:
        game: A game row (mapping or Series) carrying ``game_id`` and ``kickoff_et``.
        venue: A venue record (mapping or Series) carrying an IANA ``timezone``.
        as_of_utc: The instant the horizon is measured from. Injected, never read
            from a process clock.

    Returns:
        The resolved :class:`ForecastHour`.

    Raises:
        WeatherDataError: the venue record carries no usable zone.
        BeyondForecastHorizonError: the kickoff is beyond the declared horizon.
    """
    game_id = str(game["game_id"])
    kickoff = kickoff_wall_clock_et(game["kickoff_et"]).astimezone(UTC)
    assert_within_forecast_horizon(game_id, kickoff, as_of_utc=as_of_utc)

    # `.get` rather than `[...]`: a venue record with no zone at all must reach the
    # named refusal below, not a KeyError that says nothing about why it matters.
    zone_name = venue.get("timezone")
    if not zone_name or (isinstance(zone_name, float) and pd.isna(zone_name)):
        raise WeatherDataError(
            f"the venue record for {game_id} carries no IANA timezone, and this "
            "function will NOT default to one. A default is exactly how the "
            "Maracana game would be read in the wrong zone -- silently, with a "
            "plausible value. Add the timezone to data/venues.json."
        )

    try:
        local = kickoff.astimezone(ZoneInfo(str(zone_name)))
    except Exception as exc:  # ZoneInfoNotFoundError and friends
        raise WeatherDataError(
            f"the venue record for {game_id} names timezone {zone_name!r}, which "
            f"is not a resolvable IANA zone: {exc}"
        ) from exc

    return ForecastHour(
        game_id=game_id,
        timezone=str(zone_name),
        local_date=local.strftime("%Y-%m-%d"),
        hour=local.hour,
        kickoff_utc=kickoff,
        local_instant=local,
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
async def fetch_game_forecast(
    client: httpx.AsyncClient,
    latitude: float,
    longitude: float,
    game_date: str,
    game_hour: int,
    venue_timezone: str,
) -> dict[str, Any]:
    """Fetch a FORECAST for a venue from Open-Meteo, in the venue's own local day.

    A SIBLING of :func:`fetch_game_weather`, reusing its retry policy and its three
    error branches verbatim. The two differences are the endpoint and the zone: the
    request is made in *venue_timezone*, so the 24 hours that come back are that
    venue's local day and ``game_hour`` indexes them directly.

    Args:
        client: httpx async client.
        latitude: Venue latitude.
        longitude: Venue longitude.
        game_date: The venue-LOCAL date, ``YYYY-MM-DD``.
        game_hour: The venue-LOCAL hour, 0-23.
        venue_timezone: The venue's IANA zone.

    Returns:
        Dict with weather data matching the system schema.

    Raises:
        WeatherDataError: the call failed or the response is malformed.
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": game_date,
        "end_date": game_date,
        "hourly": HOURLY_VARIABLES,
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": venue_timezone,
    }

    try:
        response = await client.get(FORECAST_ENDPOINT_URL, params=params)
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise WeatherDataError(
            f"Open-Meteo forecast API returned {e.response.status_code} "
            f"for ({latitude}, {longitude}) on {game_date}"
        )
    except httpx.TimeoutException as e:
        raise WeatherDataError(
            f"Open-Meteo forecast API timeout for ({latitude}, {longitude}) "
            f"on {game_date}: {e}"
        )
    except httpx.RequestError as e:
        raise WeatherDataError(
            f"Open-Meteo forecast API request failed for ({latitude}, {longitude}) "
            f"on {game_date}: {e}"
        )

    data = response.json()

    if "hourly" not in data:
        raise WeatherDataError(
            f"Open-Meteo forecast response missing 'hourly' key for "
            f"({latitude}, {longitude}) on {game_date}"
        )

    hourly = data["hourly"]
    temperatures = hourly.get("temperature_2m", [])
    if not temperatures:
        raise WeatherDataError(
            f"Open-Meteo forecast returned NO hourly data for "
            f"({latitude}, {longitude}) on {game_date}. Nothing is imputed for it."
        )
    if game_hour >= len(temperatures):
        raise WeatherDataError(
            f"Open-Meteo forecast returned {len(temperatures)} hours for "
            f"({latitude}, {longitude}) on {game_date}, which does not reach hour "
            f"{game_hour}. The hour is NOT clamped to the last available one: a "
            "clamped hour is a real reading for the wrong moment."
        )

    idx = game_hour
    temp_f = temperatures[idx]

    return {
        "temp_f": temp_f,
        "temp_c": round((temp_f - 32) * 5 / 9, 1) if temp_f is not None else None,
        "wind_mph": hourly["wind_speed_10m"][idx],
        "wind_direction": hourly["wind_direction_10m"][idx],
        "humidity_pct": hourly["relative_humidity_2m"][idx],
        "precip_mm": hourly["precipitation"][idx],
        # The FORECAST endpoint does offer `precipitation_probability`, which the
        # archive endpoint does not. It is deliberately NOT wired here: the
        # provider reports it as a PERCENTAGE (0-100) and WeatherSchema.precip_prob
        # is a FRACTION with `le=1`, so adopting it is a unit conversion and a
        # schema decision rather than a parameter addition. Out of scope for R8.
        "precip_prob": None,
        "condition": None,  # Derive from weather_code if needed downstream
        "condition_code": hourly["weather_code"][idx],
        "visibility_km": None,
        "dew_point_f": hourly["dew_point_2m"][idx],
        "apparent_temp_f": hourly["apparent_temperature"][idx],
        "snowfall_cm": hourly["snowfall"][idx],
        "wind_gusts_mph": hourly["wind_gusts_10m"][idx],
        "cloud_cover_pct": hourly["cloud_cover"][idx],
        "weather_code": hourly["weather_code"][idx],
    }


def assert_complete_forecast_coverage(
    requested_game_ids: Any,
    week_frame: pd.DataFrame,
) -> None:
    """Refuse a week whose payload does not cover every game that was asked about.

    Called BEFORE anything is written. A shortfall names the absent ids and the
    count, because "2 of 3 missing" is what tells a reader the payload was PARTIAL
    rather than empty -- an empty payload announces itself, a partial one does not.

    Raises:
        IncompleteForecastPayloadError: any requested id is absent from the frame.
    """
    requested = tuple(str(value) for value in requested_game_ids)
    covered = (
        set()
        if week_frame is None or len(week_frame) == 0
        else {str(value) for value in week_frame["game_id"]}
    )
    absent = tuple(gid for gid in requested if gid not in covered)
    if not absent:
        return

    raise IncompleteForecastPayloadError(
        f"the forecast payload is MISSING {len(absent)} of {len(requested)} "
        f"requested games: {', '.join(absent)}. NOTHING is written for this week. "
        "A partially-populated week is indistinguishable from a complete one once "
        "it is on disk, which makes it the quieter form of the fabricated-data "
        "class this project has already disclosed once.",
        absent_game_ids=absent,
    )


def write_week_weather_atomically(
    week_frame: pd.DataFrame,
    table: str = "weather",
    *,
    requested_game_ids: Any = None,
    base_path=None,
):
    """Write one week's weather all-at-once, through the STORAGE layer.

    Delegates to :func:`data.storage.upsert_silver`, which already reaches an
    atomic parquet replacement. It does NOT import the bet-list module's private
    replacement helper and does not write a second temp-file dance of its own: a
    filesystem primitive for the silver store belongs in the storage layer, and
    reaching into an unrelated module's private for it would make the weather
    ingest depend on the betting module.

    Args:
        week_frame: The validated week, complete.
        table: Silver table name.
        requested_game_ids: When given, coverage is asserted before the write, so
            an incomplete week raises instead of landing half-populated.
        base_path: Data root. Threaded so a test can redirect the write.

    Returns:
        The path written.
    """
    if requested_game_ids is not None:
        assert_complete_forecast_coverage(requested_game_ids, week_frame)
    return upsert_silver(week_frame, table, base_path=base_path)


# Full set of hourly weather variables to fetch from Open-Meteo
HOURLY_VARIABLES = ",".join(
    [
        "temperature_2m",
        "relative_humidity_2m",
        "dew_point_2m",
        "apparent_temperature",
        "precipitation",
        "rain",
        "snowfall",
        "weather_code",
        "cloud_cover",
        "wind_speed_10m",
        "wind_direction_10m",
        "wind_gusts_10m",
    ]
)

# Mapping from nflverse/nflreadpy roof type strings to project VenueRoof values.
# nflreadpy returns: "dome", "closed", "outdoors", "open"
# Project uses: "indoor", "outdoor", "retractable"
NFLVERSE_ROOF_MAP = {
    "dome": "indoor",
    "closed": "indoor",
    "outdoors": "outdoor",
    "open": "retractable",
}


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
async def fetch_game_weather(
    client: httpx.AsyncClient,
    latitude: float,
    longitude: float,
    game_date: str,
    game_hour: int = 13,
) -> dict[str, Any]:
    """
    Fetch historical weather from Open-Meteo for a game venue.

    Args:
        client: httpx async client
        latitude: Venue latitude
        longitude: Venue longitude
        game_date: Game date as ISO string (YYYY-MM-DD)
        game_hour: Hour of game in local timezone (default 1 PM ET)

    Returns:
        Dict with weather data matching the system schema

    Raises:
        WeatherDataError: If API call fails or response is malformed
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": game_date,
        "end_date": game_date,
        "hourly": HOURLY_VARIABLES,
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": "America/New_York",
    }

    try:
        response = await client.get(OPEN_METEO_URL, params=params)
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise WeatherDataError(
            f"Open-Meteo API returned {e.response.status_code} "
            f"for ({latitude}, {longitude}) on {game_date}"
        )
    except httpx.TimeoutException as e:
        raise WeatherDataError(
            f"Open-Meteo API timeout for ({latitude}, {longitude}) on {game_date}: {e}"
        )
    except httpx.RequestError as e:
        raise WeatherDataError(
            f"Open-Meteo API request failed for ({latitude}, {longitude}) "
            f"on {game_date}: {e}"
        )

    data = response.json()

    if "hourly" not in data:
        raise WeatherDataError(
            f"Open-Meteo response missing 'hourly' key for "
            f"({latitude}, {longitude}) on {game_date}"
        )

    hourly = data["hourly"]

    # Select the game_hour index from the hourly arrays (0-23 for single-day)
    idx = min(game_hour, len(hourly.get("temperature_2m", [])) - 1)
    if idx < 0:
        raise WeatherDataError(
            f"No hourly data returned for ({latitude}, {longitude}) on {game_date}"
        )

    temp_f = hourly["temperature_2m"][idx]

    return {
        "temp_f": temp_f,
        "temp_c": round((temp_f - 32) * 5 / 9, 1) if temp_f is not None else None,
        "wind_mph": hourly["wind_speed_10m"][idx],
        "wind_direction": hourly["wind_direction_10m"][idx],
        "humidity_pct": hourly["relative_humidity_2m"][idx],
        "precip_mm": hourly["precipitation"][idx],
        "precip_prob": None,  # Not available in historical reanalysis data
        "condition": None,  # Derive from weather_code if needed downstream
        "condition_code": hourly["weather_code"][idx],
        "visibility_km": None,  # Not available in ERA5
        # New fields available from Open-Meteo
        "dew_point_f": hourly["dew_point_2m"][idx],
        "apparent_temp_f": hourly["apparent_temperature"][idx],
        "snowfall_cm": hourly["snowfall"][idx],
        "wind_gusts_mph": hourly["wind_gusts_10m"][idx],
        "cloud_cover_pct": hourly["cloud_cover"][idx],
        "weather_code": hourly["weather_code"][idx],
    }


class WeatherDataIngester:
    """NFL weather data ingestion using Open-Meteo Historical Weather API."""

    def __init__(self):
        """Initialize weather data ingester."""
        self.settings = get_settings()

        # Timezone mappings using stdlib zoneinfo
        self.timezone_map = {
            "America/New_York": ZoneInfo("America/New_York"),
            "America/Chicago": ZoneInfo("America/Chicago"),
            "America/Denver": ZoneInfo("America/Denver"),
            "America/Los_Angeles": ZoneInfo("America/Los_Angeles"),
            "America/Phoenix": ZoneInfo("America/Phoenix"),
        }

    def _load_venue_data(self) -> pd.DataFrame:
        """Load venue data with coordinates."""
        try:
            import json
            from pathlib import Path

            venues_path = Path("data/venues.json")
            with open(venues_path) as f:
                venues_data = json.load(f)

            venues_df = pd.DataFrame(venues_data["venues"])
            logger.info("Loaded venue data", venues=len(venues_df))
            return venues_df
        except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
            logger.error("Failed to load venue data", error=str(e))
            raise DataIngestionError(f"Venue data load failed: {e}")

    def _load_games_data(self, season: int, week: int | None = None) -> pd.DataFrame:
        """Load games data for specified season/week."""
        try:
            games_df = load_dataframe("games", layer="silver")

            # Filter for season, and optionally week
            if week is not None:
                filtered_games = games_df[
                    (games_df["season"] == season) & (games_df["week"] == week)
                ].copy()
                logger.info(
                    "Loaded games data",
                    season=season,
                    week=week,
                    games=len(filtered_games),
                )
            else:
                # Load entire season
                filtered_games = games_df[games_df["season"] == season].copy()
                logger.info(
                    "Loaded games data",
                    season=season,
                    week="all",
                    games=len(filtered_games),
                )

            # CRITICAL: Deduplicate games data to ensure one record per game
            initial_count = len(filtered_games)
            filtered_games = filtered_games.drop_duplicates(
                subset=["game_id"], keep="first"
            )
            final_count = len(filtered_games)

            if initial_count != final_count:
                logger.info(
                    "Deduplicated games data",
                    initial_records=initial_count,
                    final_records=final_count,
                    duplicates_removed=initial_count - final_count,
                )

            return filtered_games

        except (FileNotFoundError, KeyError, ValueError) as e:
            logger.error(
                "Failed to load games data", season=season, week=week, error=str(e)
            )
            raise DataIngestionError(f"Games data load failed: {e}")

    def _get_venue_record(self, home_team: str, venues_df: pd.DataFrame) -> Any:
        """The WHOLE venue row for a team, not just its coordinates.

        The row is what the forecast path needs: R8 reads the venue's IANA
        ``timezone`` as well as its latitude and longitude, and returning a triple
        would mean two lookups against the same table with two chances to disagree
        about which row won.

        Uses normalize_team_abbreviation to handle variant abbreviations
        (e.g. 'LAR' -> 'LA') before venue lookup.

        Raises:
            WeatherDataError: If no venue found for the (normalized) team.
        """
        canonical_team = normalize_team_abbreviation(home_team)
        venue_info = venues_df[
            venues_df["home_teams"].apply(lambda teams: canonical_team in teams)
        ]

        if venue_info.empty:
            raise WeatherDataError(
                f"No venue found for team '{canonical_team}' "
                f"(original: '{home_team}'). "
                f"Check data/venues.json home_teams arrays."
            )

        return venue_info.iloc[0]

    def _get_venue_record_by_stadium_id(
        self, stadium_id: object, venues_df: pd.DataFrame
    ) -> Any:
        """The WHOLE venue row for an nflverse ``stadium_id``.

        EXACT and CASE-SENSITIVE, with no normalization and no fuzzy match (R11).
        A miss RAISES rather than falling back to the home team: for a neutral-site
        game the home team's stadium is the wrong answer by construction, and the
        wrong answer arriving silently is the defect this whole path exists to fix.

        Raises:
            WeatherDataError: ``stadium_id`` is absent from ``data/venues.json``.
        """
        if "stadium_id" in venues_df.columns:
            match = venues_df[venues_df["stadium_id"] == stadium_id]
            if not match.empty:
                return match.iloc[0]

        raise WeatherDataError(
            f"No venue found for stadium_id {stadium_id!r}. It is NOT resolved to "
            "the home team's stadium: that would give the wrong coordinates and the "
            "wrong weather for a neutral-site game, silently. Add the venue record "
            "to data/venues.json with its stadium_id, latitude, longitude and "
            "roof_type entered explicitly. Matching is exact and case-sensitive."
        )

    def _resolve_venue_record_for_game(self, game: Any, venues_df: pd.DataFrame) -> Any:
        """The venue ROW for one game, under the D33-15 routing rule.

        ONE routing rule, in one place. The coordinate accessors below delegate to
        it rather than repeating the season/neutral-site test, so the forecast path
        and the archive path can never diverge about which stadium a game is at.

        Seasons before ``STADIUM_ID_ROUTING_FIRST_SEASON`` keep the home-team
        resolution unchanged, including their neutral-site games -- those 91 rows
        are a disclosure, not a repair (see the contextual module's note).
        """
        try:
            season = int(game.get("season"))
        except (TypeError, ValueError):
            season = 0

        if season >= STADIUM_ID_ROUTING_FIRST_SEASON and game_is_neutral_site(game):
            return self._get_venue_record_by_stadium_id(
                game.get("stadium_id"), venues_df
            )

        return self._get_venue_record(game["home_team"], venues_df)

    def _get_venue_coordinates(
        self, home_team: str, venues_df: pd.DataFrame
    ) -> tuple[float, float, str]:
        """Get venue coordinates and roof type for a team."""
        venue = self._get_venue_record(home_team, venues_df)
        return venue["latitude"], venue["longitude"], venue["roof_type"]

    def _get_venue_coordinates_by_stadium_id(
        self, stadium_id: object, venues_df: pd.DataFrame
    ) -> tuple[float, float, str]:
        """Get venue coordinates and roof type by nflverse ``stadium_id``."""
        venue = self._get_venue_record_by_stadium_id(stadium_id, venues_df)
        return venue["latitude"], venue["longitude"], venue["roof_type"]

    def _resolve_venue_for_game_row(
        self, game: Any, venues_df: pd.DataFrame
    ) -> tuple[float, float, str]:
        """Coordinates and roof for one game, under the D33-15 routing rule."""
        venue = self._resolve_venue_record_for_game(game, venues_df)
        return venue["latitude"], venue["longitude"], venue["roof_type"]

    def _is_outdoor_game(self, roof_type: str) -> bool:
        """Determine if weather affects the game."""
        return roof_type.lower() in ["outdoor", "retractable"]

    def _map_nflverse_roof_type(self, nflverse_roof: str) -> str:
        """Map nflreadpy roof type string to project VenueRoof value.

        nflreadpy uses: 'dome', 'closed', 'outdoors', 'open'
        Project uses: 'indoor', 'outdoor', 'retractable'
        """
        return NFLVERSE_ROOF_MAP.get(nflverse_roof.lower(), "outdoor")

    def _convert_timezone(
        self, dt: datetime, from_tz: str, to_utc: bool = True
    ) -> datetime:
        """Convert datetime between timezones."""
        if from_tz in self.timezone_map:
            tz = self.timezone_map[from_tz]

            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=tz)

            if to_utc:
                return dt.astimezone(UTC)
            return dt

        return dt

    def _create_indoor_weather_record(
        self,
        game_id: str,
        game_time: datetime,
        forecast_time: datetime,
    ) -> dict[str, Any]:
        """Create zeroed weather record for indoor/dome games.

        Indoor games have no meaningful weather impact, so all weather
        fields are set to zero or None with is_outdoor=False.
        """
        return {
            "game_id": game_id,
            "forecast_time": forecast_time,
            "game_time": game_time,
            "temp_f": None,
            "temp_c": None,
            "wind_mph": 0.0,
            "wind_direction": None,
            "humidity_pct": None,
            "precip_prob": 0.0,
            "precip_mm": 0.0,
            "condition": "indoor",
            "condition_code": None,
            "visibility_km": None,
            "is_outdoor": False,
            "is_cold": False,
            "is_windy": False,
            "is_precipitation": False,
        }

    def _create_weather_record(
        self,
        game_id: str,
        game_time: datetime,
        forecast_time: datetime,
        weather_data: dict[str, Any],
        roof_type: str,
    ) -> dict[str, Any]:
        """Create weather record for an outdoor/retractable game."""
        is_outdoor = self._is_outdoor_game(roof_type)

        # Derived weather flags - handle None values safely
        temp_f = weather_data.get("temp_f")
        is_cold = temp_f < 32 if temp_f is not None else False

        wind_mph = weather_data.get("wind_mph")
        is_windy = wind_mph > 12 if wind_mph is not None else False

        precip_prob = weather_data.get("precip_prob", 0)
        precip_mm = weather_data.get("precip_mm", 0)
        is_precipitation = (
            precip_prob > 0.3 if precip_prob is not None else False
        ) or (precip_mm > 0 if precip_mm is not None else False)

        return {
            "game_id": game_id,
            "forecast_time": forecast_time,
            "game_time": game_time,
            "is_outdoor": is_outdoor,
            "is_cold": is_cold,
            "is_windy": is_windy,
            "is_precipitation": is_precipitation,
            **weather_data,
        }

    async def _fetch_openmeteo_weather(
        self,
        latitude: float,
        longitude: float,
        game_date: str,
        game_hour: int = 13,
    ) -> dict[str, Any]:
        """
        Fetch weather data from Open-Meteo Historical API.

        Args:
            latitude: Venue latitude
            longitude: Venue longitude
            game_date: Game date as YYYY-MM-DD string
            game_hour: Hour of game in ET (default 1 PM)

        Returns:
            Dict with weather data matching system schema

        Raises:
            WeatherDataError: If API call fails
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            return await fetch_game_weather(
                client, latitude, longitude, game_date, game_hour
            )

    def fetch_weather_for_games(
        self,
        games_df: pd.DataFrame,
        venues_df: pd.DataFrame,
        forecast_time: datetime | None = None,
    ) -> pd.DataFrame:
        """Fetch weather data for all games.

        Indoor games get zeroed weather records (no API call).
        Outdoor/retractable games get real Open-Meteo data.
        If an outdoor game's weather fetch fails, raises WeatherDataError
        (hard-fail -- no silent skipping).
        """
        if forecast_time is None:
            # tz-aware UTC -- Phase 15-04 requires storage writers to provide
            # timezone-aware datetimes. Storage will reject naive datetimes
            # at write time via _normalize_parquet_datetime_columns.
            forecast_time = datetime.now(UTC)

        logger.info("Fetching weather for games", games=len(games_df))

        weather_records = []

        for _, game in games_df.iterrows():
            # Validate game ID format before processing
            if not is_valid_game_id(game["game_id"]):
                logger.warning(
                    "Invalid game ID format, skipping game", game_id=game["game_id"]
                )
                continue

            # Get venue coordinates and roof type. D33-15: a 2026-and-later
            # neutral-site game resolves by the feed's stadium_id, so the eight
            # international games get their OWN coordinates and their own roof
            # rather than the nominal home team's.
            lat, lon, roof_type = self._resolve_venue_for_game_row(game, venues_df)

            # Convert game time to UTC through the ONE kickoff accessor (WR-06).
            #
            # Recorded honestly: this is uniformity, NOT a behaviour fix.
            # _convert_timezone attaches the named zone only to a NAIVE value and
            # otherwise calls astimezone(UTC) on the aware one, so it already behaved
            # as the true-instant contract for every aware kickoff -- which is what
            # the column carries. The delta here is zero on both an aware-UTC and a
            # naive input (pinned in tests/unit/test_date_utils_contract.py). The
            # point is that there is now exactly one place that decides what
            # kickoff_et means.
            game_time_utc = kickoff_wall_clock_et(game["kickoff_et"]).astimezone(UTC)

            if not self._is_outdoor_game(roof_type):
                # Indoor game: use zeroed weather record (no API call)
                weather_record = self._create_indoor_weather_record(
                    game["game_id"],
                    game_time_utc,
                    forecast_time,
                )
            else:
                # Outdoor/retractable game: fetch real weather data
                game_date_str = game_time_utc.strftime("%Y-%m-%d")
                game_hour = game_time_utc.hour

                # Use asyncio.run for the async fetch.
                # If this fails, WeatherDataError propagates (hard-fail).
                weather_data = asyncio.run(
                    self._fetch_openmeteo_weather(lat, lon, game_date_str, game_hour)
                )

                weather_record = self._create_weather_record(
                    game["game_id"],
                    game_time_utc,
                    forecast_time,
                    weather_data,
                    roof_type,
                )

            weather_records.append(weather_record)

            logger.debug(
                "Weather fetched for game",
                game_id=game["game_id"],
                outdoor=weather_record["is_outdoor"],
            )

        weather_df = pd.DataFrame(weather_records)

        logger.info(
            "Weather data fetched",
            games=len(games_df),
            weather_records=len(weather_df),
        )

        return weather_df

    async def _fetch_openmeteo_forecast(
        self,
        latitude: float,
        longitude: float,
        game_date: str,
        game_hour: int,
        venue_timezone: str,
    ) -> dict[str, Any]:
        """Open a client and fetch one game's FORECAST. The seam tests replace."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            return await fetch_game_forecast(
                client, latitude, longitude, game_date, game_hour, venue_timezone
            )

    def fetch_forecast_for_games(
        self,
        games_df: pd.DataFrame,
        venues_df: pd.DataFrame,
        forecast_time: datetime | None = None,
        *,
        as_of_utc: datetime,
    ) -> pd.DataFrame:
        """Fetch FORECASTS for a week of games, or raise without producing a row.

        THE FORWARD SIBLING of :meth:`fetch_weather_for_games`. Three differences,
        each of them the point:

        1. The horizon is checked for EVERY game, BEFORE any client is opened. An
           indoor game needs no forecast, but the horizon is still asserted for it:
           this method's contract is a WEEK, and a week that cannot be forecast in
           full is refused in full rather than written half-true.
        2. The forecast hour comes from the VENUE's own IANA zone.
        3. Nothing is imputed. A failure anywhere raises and the caller writes
           nothing.

        Args:
            games_df: The week's games.
            venues_df: The venue table, carrying ``timezone`` per record.
            forecast_time: When this forecast was taken. Defaults to now, in UTC.
            as_of_utc: The instant the horizon is measured from. Injected.

        Returns:
            One record per game, in the order the games arrived.

        Raises:
            BeyondForecastHorizonError: any kickoff is past the horizon.
            WeatherDataError: any venue or any fetch failed.
        """
        _require_aware(as_of_utc, "as_of_utc")
        if forecast_time is None:
            forecast_time = datetime.now(UTC)

        logger.info(
            "Fetching FORECAST weather for games",
            games=len(games_df),
            as_of=as_of_utc.isoformat(),
        )

        weather_records: list[dict[str, Any]] = []

        for _, game in games_df.iterrows():
            game_id = str(game["game_id"])
            if not is_valid_game_id(game_id):
                raise WeatherDataError(
                    f"invalid game_id {game_id!r} in the forecast week. It "
                    "is NOT skipped: a skipped game is a game the coverage check "
                    "can never notice is missing."
                )

            venue = self._resolve_venue_record_for_game(game, venues_df)

            # The horizon refusal lives inside hour selection, so it is raised
            # BEFORE any client is constructed for any game in the week.
            selected = select_forecast_hour_for_kickoff(
                game, venue, as_of_utc=as_of_utc
            )

            roof_type = venue["roof_type"]
            if not self._is_outdoor_game(roof_type):
                weather_records.append(
                    self._create_indoor_weather_record(
                        game_id, selected.kickoff_utc, forecast_time
                    )
                )
                continue

            weather_data = asyncio.run(
                self._fetch_openmeteo_forecast(
                    venue["latitude"],
                    venue["longitude"],
                    selected.local_date,
                    selected.hour,
                    selected.timezone,
                )
            )
            weather_records.append(
                self._create_weather_record(
                    game_id,
                    selected.kickoff_utc,
                    forecast_time,
                    weather_data,
                    roof_type,
                )
            )

        return pd.DataFrame(weather_records)

    def ingest_week_forecast(
        self,
        games_df: pd.DataFrame,
        venues_df: pd.DataFrame,
        *,
        as_of_utc: datetime,
        forecast_time: datetime | None = None,
        base_path=None,
        table: str = "weather",
    ) -> pd.DataFrame:
        """Fetch and write ONE week's forecast, all-or-nothing.

        The order is the guarantee: fetch every game, assert COMPLETE coverage,
        THEN write once. A failure at any point before the write leaves the silver
        table byte-identical, which is asserted by content digest in
        ``tests/unit/test_weather_atomic_week_write.py`` rather than assumed.

        Args:
            games_df: The week's games. Must be a single (season, week).
            venues_df: The venue table.
            as_of_utc: The instant the horizon is measured from. Injected.
            forecast_time: When this forecast was taken. Defaults to now, in UTC.
            base_path: Data root. Threaded so a test can redirect the write.
            table: Silver table name.

        Returns:
            The validated frame that was written.
        """
        if forecast_time is None:
            forecast_time = datetime.now(UTC)

        season, week = self._single_season_week(games_df)
        requested_game_ids = [str(value) for value in games_df["game_id"]]

        weather_df = self.fetch_forecast_for_games(
            games_df, venues_df, forecast_time=forecast_time, as_of_utc=as_of_utc
        )

        # BEFORE the bronze snapshot as well as before the silver write: an
        # incomplete week should leave no trace at all, not an orphan bronze file
        # a later reader could mistake for a captured week.
        assert_complete_forecast_coverage(requested_game_ids, weather_df)

        save_bronze_snapshot(
            weather_df, table, season=season, week=week, base_path=base_path
        )

        validated_df = validate_bronze_to_silver(weather_df, WeatherSchema)
        validated_df = validated_df.drop_duplicates(subset=["game_id"], keep="first")
        validated_df["created_at"] = datetime.now(UTC)

        assert_complete_forecast_coverage(requested_game_ids, validated_df)

        write_week_weather_atomically(
            validated_df,
            table,
            requested_game_ids=requested_game_ids,
            base_path=base_path,
        )

        log_data_operation(
            operation="ingest_forecast",
            table=table,
            rows=len(validated_df),
            season=season,
            week=week,
        )
        return validated_df

    @staticmethod
    def _single_season_week(games_df: pd.DataFrame) -> tuple[int, int]:
        """The one (season, week) this frame covers, or a refusal naming what it saw.

        The all-or-nothing write is a WEEK-level promise. A frame spanning two weeks
        would make "the week is unchanged" meaningless, so the constraint is
        enforced here rather than assumed by the caller.
        """
        pairs = {
            (int(row["season"]), int(row["week"])) for _, row in games_df.iterrows()
        }
        if len(pairs) != 1:
            raise WeatherDataError(
                "ingest_week_forecast writes ONE week atomically and was handed "
                f"{len(pairs)} (season, week) pairs: {sorted(pairs)}. Split the "
                "call; an all-or-nothing promise over a mixed frame is not one."
            )
        return next(iter(pairs))

    def ingest_weather(
        self,
        season: int | None = None,
        week: int | None = None,
        forecast_time: datetime | None = None,
    ) -> pd.DataFrame:
        """
        Full weather data ingestion pipeline.

        Uses Bronze/Silver separation:
        - Bronze: timestamped append-only snapshots via save_bronze_snapshot
        - Silver: validated, upserted via validate_bronze_to_silver + upsert_silver

        Args:
            season: Season to ingest (default: current)
            week: Week to ingest (default: current), None for entire season
            forecast_time: Time when forecast was made

        Returns:
            Ingested and validated weather data
        """
        # Handle default season
        if season is None:
            current_season, current_week = get_current_nfl_week()
            season = current_season
            # If no week specified, default to current week for current season
            if week is None:
                week = current_week

        if forecast_time is None:
            # tz-aware UTC -- Phase 15-04 requires storage writers to provide
            # timezone-aware datetimes. Storage will reject naive datetimes
            # at write time via _normalize_parquet_datetime_columns.
            forecast_time = datetime.now(UTC)

        scope = f"Week {week}" if week is not None else "Entire Season"
        logger.info(
            "Starting weather data ingestion",
            season=season,
            week=week,
            scope=scope,
            forecast_time=forecast_time.isoformat(),
        )

        try:
            # Load required data
            venues_df = self._load_venue_data()
            games_df = self._load_games_data(season, week)

            if games_df.empty:
                logger.warning("No games found", season=season, week=week)
                return pd.DataFrame()

            # Fetch weather data (hard-fails on missing outdoor weather)
            weather_df = self.fetch_weather_for_games(
                games_df, venues_df, forecast_time
            )

            if weather_df.empty:
                logger.warning("No weather data fetched")
                return weather_df

            # Save Bronze snapshot (append-only, timestamped)
            if week is not None:
                save_bronze_snapshot(weather_df, "weather", season=season, week=week)
            else:
                # For full-season ingestion, use week=0 as a sentinel
                save_bronze_snapshot(weather_df, "weather", season=season, week=0)

            # Validate through quality gate (hard-fail on bad rows)
            validated_df = validate_bronze_to_silver(weather_df, WeatherSchema)

            # Ensure no duplicate weather records per game
            initial_count = len(validated_df)
            validated_df = validated_df.drop_duplicates(
                subset=["game_id"], keep="first"
            )
            final_count = len(validated_df)

            if initial_count != final_count:
                logger.info(
                    "Deduplicated weather records",
                    initial_records=initial_count,
                    final_records=final_count,
                    duplicates_removed=initial_count - final_count,
                )

            # Add metadata timestamp
            validated_df["created_at"] = datetime.now(UTC)

            # Upsert to Silver (latest wins by game_id)
            upsert_silver(validated_df, "weather")

            log_data_operation(
                operation="ingest",
                table="weather",
                rows=len(validated_df),
                season=season,
                week=week,
            )

            logger.info(
                "Weather data ingestion completed successfully",
                total_records=len(validated_df),
                unique_games=len(validated_df),
                outdoor_games=int(validated_df["is_outdoor"].sum()),
                season=season,
                week=week,
            )

            return validated_df

        except DataIngestionError:
            raise
        except (httpx.HTTPStatusError, httpx.TimeoutException) as e:
            logger.error("Weather data ingestion failed", error=str(e))
            raise DataIngestionError(f"Weather ingestion failed: {e}")


def main():
    """CLI entry point for weather data ingestion."""
    parser = argparse.ArgumentParser(description="Ingest NFL weather data")

    # Add standardized ingestion arguments
    from utils.ingestion_args import (
        add_standard_ingestion_args,
        parse_season_week_args,
    )

    parser = add_standard_ingestion_args(parser)

    # Add weather-specific arguments
    parser.add_argument(
        "--forecast-time", type=str, help="Forecast time (ISO format, default: now)"
    )
    parser.add_argument(
        "--outdoor-only",
        action="store_true",
        help="Only fetch weather for outdoor games",
    )

    args = parser.parse_args()

    try:
        # Setup logging
        from utils import setup_logging

        setup_logging()

        # Parse standardized season/week arguments
        seasons, weeks = parse_season_week_args(args)

        # Weather ingestion currently supports single season only
        if len(seasons) > 1:
            print(
                "Warning: Weather ingestion only supports single season. "
                "Using first season."
            )
        season = seasons[0]

        # Weather can handle multiple weeks or single week
        if weeks and len(weeks) == 1:
            week = weeks[0]
            logger.info(
                f"Starting weather data ingestion for season {season}, week {week}"
            )
        elif weeks and len(weeks) > 1:
            week = None  # Will ingest multiple weeks
            logger.info(
                f"Starting weather data ingestion for season {season}, weeks {weeks}"
            )
        else:
            week = None  # Will ingest entire season
            logger.info(
                f"Starting weather data ingestion for season {season}, all weeks"
            )

        # Parse forecast time
        forecast_time = None
        if args.forecast_time:
            try:
                forecast_time = datetime.fromisoformat(args.forecast_time)
                if forecast_time.tzinfo is None:
                    forecast_time = forecast_time.replace(tzinfo=ZoneInfo("UTC"))
            except ValueError:
                print(f"Invalid forecast time format: {args.forecast_time}")
                sys.exit(1)

        # Initialize ingester
        ingester = WeatherDataIngester()

        # Run ingestion
        weather_df = ingester.ingest_weather(
            season=season, week=week, forecast_time=forecast_time
        )

        if weather_df.empty:
            print("No weather data ingested")
            return

        scope = f"Week {week}" if week is not None else "Entire Season"
        print(f"Successfully ingested {len(weather_df)} weather records")
        print(f"Season: {season}, Scope: {scope}")
        print(f"Unique games: {len(weather_df)}")
        print(f"Outdoor games: {weather_df['is_outdoor'].sum()}")
        print(f"Indoor games: {(~weather_df['is_outdoor']).sum()}")

        # Show weather summary for outdoor games
        outdoor_weather = weather_df[weather_df["is_outdoor"]]
        if not outdoor_weather.empty:
            print("\nOutdoor weather summary:")
            if "temp_f" in outdoor_weather.columns:
                print(f"  Temperature: {outdoor_weather['temp_f'].mean():.1f}F avg")
            if "wind_mph" in outdoor_weather.columns:
                print(f"  Wind: {outdoor_weather['wind_mph'].mean():.1f} mph avg")
            if "is_cold" in outdoor_weather.columns:
                print(f"  Cold games: {outdoor_weather['is_cold'].sum()}")
            if "is_windy" in outdoor_weather.columns:
                print(f"  Windy games: {outdoor_weather['is_windy'].sum()}")

    except Exception as e:
        logger.error("Weather ingestion CLI failed", error=str(e))
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
