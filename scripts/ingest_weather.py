"""LIVE weather ingestion using the Open-Meteo FORECAST API.

THIS MODULE CANNOT REACH THE ARCHIVE ENDPOINT, BY DESIGN (D33-26). The archive
endpoint is an ERA5 reanalysis product: it describes weather that has already
occurred, so it cannot answer for a game that has not been played. It lives in
`scripts/backfill_historical_weather.py` and nowhere else, and
`tests/unit/test_weather_archive_quarantined.py` scans this file structurally to
keep that true.

Produces timestamped Bronze snapshots and upserts to Silver weather table.
Indoor games get zeroed weather fields (no API call) and are still stamped with
the run's provenance. Outdoor and retractable-roof games get a real forecast,
read at the hour the kickoff falls on IN THE VENUE'S OWN TIMEZONE. A kickoff
beyond the declared horizon, or a payload missing any requested game, raises by
name and writes nothing -- never an archive fallback and never an imputed value.
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

# THE MODULE, NOT THE FUNCTION, AND THAT IS DELIBERATE (Plan 33.1-03, Ruling I2).
# `from features.contextual import venue_record_for_stadium_id` would bind a SECOND
# name to the resolver here, and monkeypatching the canonical one would then leave
# this call site pointing at the original -- so a mutation test could not tell one
# shared implementation from two that happen to agree, which is the exact thing it
# exists to distinguish. Reaching the function through the module keeps ONE binding.
# The exception class is imported directly: it is re-raised as a type, never called.
from features import contextual as venue_routing
from features.contextual import UnknownStadiumError
from features.schedule_moves import facts_at_lock
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


# ---------------------------------------------------------------------------
# `weather_source` PROVENANCE (D33-26, Plan 33-09 Task 2).
#
# An archive row and a forecast row for the same game are the same columns of
# equally plausible numbers. Nothing in the VALUES distinguishes "measured after
# the fact" from "predicted three days out" -- and that difference is exactly
# what a Phase-37 weather-aware re-fit has to know. So every row says which it is.
#
# THE VOCABULARY IS CLOSED and a value outside it is rejected BY NAME at write
# time. One tuple, one home, mirroring api/cache.BET_LIST_COLUMNS's single-source
# discipline; tests/phase33_state.WEATHER_SOURCE_VOCABULARY pins it so a silent
# edit to either is a test failure rather than a divergence nobody notices.
# ---------------------------------------------------------------------------

WEATHER_SOURCE_FORECAST = "forecast"
"""The Open-Meteo FORECAST endpoint -- predicted before kickoff.

The only thing that can answer for an unplayed game, and therefore what every
live 2026 row carries. It is the ONLY member this module names as a constant.
"""

WEATHER_SOURCE_VOCABULARY: tuple[str, ...] = (
    # The ERA5 reanalysis -- measured after the fact, the only source for
    # 2002-2020. Named as a CONSTANT in scripts/backfill_historical_weather.py
    # rather than here, deliberately: the quarantine's mechanical form is that
    # this module exposes no name containing "ARCHIVE" at all, and a constant
    # called WEATHER_SOURCE_ARCHIVE would satisfy the letter of the source scan
    # while making the runtime check false.
    "archive",
    WEATHER_SOURCE_FORECAST,
    # The Historical Forecast API -- what the forecast SAID at the time, for a
    # game that has since been played. Distinct from "archive" even for the same
    # game: one is what happened, the other is what was predicted. It reaches
    # back only to about 2021, which is why the archive endpoint cannot simply be
    # replaced by it. Also named in the backfill module, which is what produces it.
    "historical_forecast",
)
"""The CLOSED vocabulary. Pinned by tests/phase33_state.WEATHER_SOURCE_VOCABULARY,
and the two historical members are pinned against the backfill module's own
constants by tests/unit/test_weather_archive_quarantined.py, so splitting the
names across two modules cannot let the values drift apart."""


def validate_weather_source(value: object) -> str:
    """Return *value* if it is in the closed vocabulary, else raise naming both.

    The refusal names the REJECTED value AND the ALLOWED set, because a reader
    who only learns their value was wrong still does not know what to write
    instead.

    ``None`` is rejected rather than treated as "unknown". An unstamped row is
    precisely the state this column exists to make impossible: a row that does
    not say where it came from is a row a later reader has to guess about, and
    guessing is how the archive/forecast distinction gets lost.

    Raises:
        ValueError: *value* is not one of ``WEATHER_SOURCE_VOCABULARY``.
    """
    if value not in WEATHER_SOURCE_VOCABULARY:
        raise ValueError(
            f"weather_source {value!r} is outside the closed vocabulary "
            f"{WEATHER_SOURCE_VOCABULARY}. Every weather row must say where it "
            "came from: 'archive' is the ERA5 reanalysis (measured after the "
            "fact), 'forecast' is the live prediction made before kickoff, and "
            "'historical_forecast' is what the forecast said at the time for a "
            "game that has since been played."
        )
    return str(value)


def normalize_wind_direction(value: Any) -> float | None:
    """Map a meteorological bearing onto ``[0, 360)``, the range the schema states.

    FOUND BY THE TRACER, ON REAL DATA (Plan 33.1-02 Task 1). Open-Meteo reports a
    due-north wind as ``360.0``. ``WeatherSchema.wind_direction`` is declared
    ``ge=0, lt=360``, so that value is REJECTED and one bad row kills the whole
    batch by design. Season 2016 has exactly one: ``2016_W07_CHI@GB``, 360.0. Over
    6,499 games the shape recurs, and it would have surfaced as a whole-season
    hard failure partway through a paced 80-minute run.

    THE SCHEMA IS RIGHT AND IS NOT WIDENED. 0 and 360 are the same direction, so
    ``le=360`` would let one bearing have two encodings -- and then a comparison,
    a bucketing or a circular mean over the column silently depends on which one
    the provider happened to send. The half-open range is the correct invariant;
    the fetch is the right place to canonicalise onto it.

    ONE HELPER, BOTH FETCHERS. :func:`fetch_game_forecast` carries the identical
    expression and therefore the identical latent defect -- a north wind on a live
    2026 kickoff would have hard-failed the weekly ingest. Fixing it in one shared
    place rather than twice is the same discipline that keeps the archive and
    forecast paths from disagreeing about which stadium a game is at.

    Args:
        value: A bearing in degrees, or ``None``.

    Returns:
        The bearing in ``[0, 360)``, or ``None``. ``NaN`` passes through as
        ``NaN`` for ``WeatherSchema.nan_to_none`` to resolve.
    """
    if value is None:
        return None
    return float(value) % 360.0


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


def resolve_venue_local_hour(game: Any, venue: Any) -> ForecastHour:
    """Resolve WHICH hour of WHICH venue-local day a game must be read at.

    THE ZONE-AND-HOUR HALF, WITH NO FORECAST HORIZON (Plan 33.1-02 Ruling F).
    Extracted from :func:`select_forecast_hour_for_kickoff`, which now calls it
    after asserting the horizon. The body lives HERE and is not duplicated: two
    copies of a zone resolution would eventually disagree about which hour a game
    is at, which is the same class of defect COLD-09 fixed in space.

    THE ARCHIVE PATH DELIBERATELY DOES NOT PASS THROUGH THE HORIZON ASSERTION, and
    that is a decision rather than an omission. A past kickoff is never beyond a
    future cutoff, so reusing the forecast entry point would be SAFE -- and would
    leave ``assert_within_forecast_horizon`` sitting in the call graph of a path
    that reads an archive. A vacuous check in a call graph reads as a promise, and
    the next reader would have to re-derive that it can never fire. So the horizon
    stays with the forecast caller, which is the only caller it means anything to.

    ``ForecastHour``'s NAME predates this shared use. It is not renamed here: it is
    a frozen dataclass referenced by the forecast path and by its tests, and a
    rename inside a tracer slice would be churn for a word. Read it as
    "the resolved hour", not as "a forecast-only hour".

    Args:
        game: A game row (mapping or Series) carrying ``game_id`` and ``kickoff_et``.
        venue: A venue record (mapping or Series) carrying an IANA ``timezone``.

    Returns:
        The resolved :class:`ForecastHour`.

    Raises:
        WeatherDataError: the venue record carries no usable zone.
    """
    game_id = str(game["game_id"])
    kickoff = kickoff_wall_clock_et(game["kickoff_et"]).astimezone(UTC)

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
    cannot skip it. The zone-and-hour resolution itself now lives in
    :func:`resolve_venue_local_hour`, which the archive backfill calls WITHOUT this
    wrapper -- see that function's docstring for why the horizon does not travel
    with it.

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
    assert_within_forecast_horizon(
        str(game["game_id"]),
        kickoff_wall_clock_et(game["kickoff_et"]).astimezone(UTC),
        as_of_utc=as_of_utc,
    )
    return resolve_venue_local_hour(game, venue)


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
        "wind_direction": normalize_wind_direction(hourly["wind_direction_10m"][idx]),
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
        """The WHOLE venue row for a TEAM -- "where does this team play".

        NOT REACHABLE FROM ANY GAME-VENUE RESOLUTION PATH since D33.1-06. It answers
        a different question from "where was this game played", and confusing the two
        is what gave 1,153 of 6,499 games the wrong stadium. It is kept because the
        team question is still asked (the surface-mismatch comparison needs the AWAY
        team's own home venue) and because Plan 33.1-03's coordinate diff evaluates
        the real prior rule through it rather than through a paraphrase of it.

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
        self,
        stadium_id: object,
        venues_df: pd.DataFrame,
        *,
        game_id: object = None,
    ) -> Any:
        """Adapt a venues FRAME to the shared resolver. The DECISION is not here.

        THE LOOKUP DECISION LIVES IN ``features.contextual.venue_record_for_stadium_id``
        (Plan 33.1-03, Ruling I2). This function exists for ONE reason: its callers
        hand it a ``venues_df``, and the shared resolver takes a list of records. It
        adapts the frame and nothing else -- do NOT re-inline the match body or the
        refusal string for convenience. Two implementations that happen to agree are
        not one rule; they are two rules waiting for the next edit to either, and
        that is the drift COLD-09 exists to close.

        The refusal is likewise raised by the shared resolver and merely re-typed
        here. ``WeatherDataError`` is this module's contract with its callers, so the
        TYPE is preserved while the TEXT stays single-sourced -- the two consumers
        cannot hand an operator different recovery instructions for the same missing
        record.

        EXACT and CASE-SENSITIVE, with no normalization and no fuzzy match. A miss
        RAISES rather than falling back to the home team, for every game of every
        season (D33.1-06).

        Returns:
            The venue RECORD (a mapping). Callers read it by key, so a mapping and a
            frame row are interchangeable here -- and returning what the shared
            resolver returned is what makes the mutation test in
            ``tests/unit/test_venue_resolver_is_shared.py`` able to observe that this
            path really does go through it.

        Raises:
            WeatherDataError: ``stadium_id`` is absent from the given records.
        """
        try:
            return venue_routing.venue_record_for_stadium_id(
                stadium_id, venues_df.to_dict("records"), game_id=game_id
            )
        except UnknownStadiumError as exc:
            raise WeatherDataError(str(exc)) from exc

    def _resolve_venue_record_for_game(self, game: Any, venues_df: pd.DataFrame) -> Any:
        """The venue record for one game: ONE routing rule, in one place.

        The rule is the game's own ``stadium_id``, for EVERY season and EVERY game
        (D33.1-06). There is no season test and no neutral-site test. The coordinate
        accessors below delegate to this rather than repeating the lookup, and this
        delegates in turn to ``features.contextual.venue_record_for_stadium_id`` --
        so the contextual builder, the forecast path and the archive path cannot
        diverge about which stadium a game is at, mechanically rather than by
        coincidence.

        THE WEATHER LOCATION IS THE VENUE AT THE LOCK (Plan 33.2-10). The
        ``stadium_id`` comes from ``features.schedule_moves.facts_at_lock`` -- the same
        accessor the contextual builder reads -- so a game moved by an emergency
        announced after its lock takes its weather at the venue it was scheduled at
        before the move. Every other game resolves by its own ``stadium_id``, exactly
        as before.
        """
        game_id = game.get("game_id")
        stadium_id = facts_at_lock(game_id, game).weather_stadium_id
        return self._get_venue_record_by_stadium_id(
            stadium_id, venues_df, game_id=game_id
        )

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
        *,
        weather_source: str,
    ) -> dict[str, Any]:
        """Create zeroed weather record for indoor/dome games.

        Indoor games have no meaningful weather impact, so all weather
        fields are set to zero or None with is_outdoor=False.

        ``weather_source`` is REQUIRED and keyword-only, with no default. An
        indoor row is never fetched, but it is still produced BY a particular
        run, and a default here would let a caller ship an unstamped row -- the
        one state the provenance column exists to make impossible.

        ``weather_coverage`` is TRUE here (Plan 33.1-02 Ruling E). The flag means
        "this row carries the weather record it is ENTITLED to", not "a number was
        fetched". A dome game is entitled to no observation, so it is covered. The
        three D33.1-07 states are recoverable because the flag composes with
        ``is_outdoor``: ``(coverage=1, outdoor=0)`` is this row, a dome;
        ``(coverage=1, outdoor=1)`` is a real observation; ``(coverage=0,
        outdoor=1)`` is an absence.
        """
        return {
            "game_id": game_id,
            "weather_source": validate_weather_source(weather_source),
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
            "weather_coverage": True,
        }

    def _create_absent_observation_record(
        self,
        game_id: str,
        game_time: datetime,
        forecast_time: datetime,
        *,
        weather_source: str,
    ) -> dict[str, Any]:
        """The THIRD D33.1-07 state: the venue resolved, the observation did not.

        NOT a dome and NOT an observation. The game was played outdoors and the
        provider has no reading for that hour, so every weather measurement is
        NULL and ``weather_coverage`` is False. There is no numeric path through
        this function at all -- that is the point. A missing observation silently
        becoming a number is the fabricated-data class this project has already
        disclosed once, and it is what ``features/weather.py``'s 65.0 default did
        to 6,485 of 6,499 gold rows.

        WHAT REACHES HERE is decided by a PAYLOAD PREDICATE, not by a caught
        exception: :func:`scripts.backfill_historical_weather.observation_is_absent`
        fires only when every one of the five
        ``ABSENT_OBSERVATION_MEASUREMENTS`` series is null at the selected index.
        A PARTIAL null is an observation with missing fields and routes to
        :meth:`_create_weather_record` instead.

        ``is_outdoor`` is True by construction. An absent observation is only
        reachable for a game that needed a fetch, and a fetch is only issued for a
        game whose own ``roof`` says weather applies. Writing False here would make
        the row indistinguishable from a dome -- which is exactly the collapse the
        coverage flag exists to undo.

        ``weather_source`` copies :meth:`_create_indoor_weather_record`'s signature
        discipline exactly: REQUIRED, keyword-only, no default. A row with no
        observation still has a provenance -- it says which run looked and found
        nothing -- and a default here would let a caller ship an unstamped row.
        """
        return {
            "game_id": game_id,
            "weather_source": validate_weather_source(weather_source),
            "forecast_time": forecast_time,
            "game_time": game_time,
            "temp_f": None,
            "temp_c": None,
            "wind_mph": None,
            "wind_direction": None,
            "humidity_pct": None,
            "precip_prob": None,
            "precip_mm": None,
            "condition": None,
            "condition_code": None,
            "visibility_km": None,
            "dew_point_f": None,
            "apparent_temp_f": None,
            "snowfall_cm": None,
            "wind_gusts_mph": None,
            "cloud_cover_pct": None,
            "is_outdoor": True,
            # A derived flag inherits its input's NULL rather than defaulting to
            # False (Plan 33.1-04 Ruling J). `is_cold=False` would be a claim that
            # the game was not cold, which nothing here knows.
            "is_cold": None,
            "is_windy": None,
            "is_precipitation": None,
            "weather_coverage": False,
        }

    def _create_weather_record(
        self,
        game_id: str,
        game_time: datetime,
        forecast_time: datetime,
        weather_data: dict[str, Any],
        roof_type: str,
        *,
        weather_source: str,
    ) -> dict[str, Any]:
        """Create weather record for an outdoor/retractable game.

        ``weather_source`` is REQUIRED and keyword-only: an archive row and a
        forecast row are otherwise indistinguishable, so the caller has to say.

        ``weather_coverage`` is TRUE here (Plan 33.1-02 Ruling E): reaching this
        factory means a real observation was returned for the game's own hour. An
        all-null payload never arrives here -- the payload predicate in
        :func:`scripts.backfill_historical_weather.observation_is_absent` routes it
        to :meth:`_create_absent_observation_record` instead. A PARTIAL null does
        arrive here, and is covered: the present fields are carried and the absent
        ones stay NULL.
        """
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
            "weather_source": validate_weather_source(weather_source),
            "forecast_time": forecast_time,
            "game_time": game_time,
            "is_outdoor": is_outdoor,
            "is_cold": is_cold,
            "is_windy": is_windy,
            "is_precipitation": is_precipitation,
            "weather_coverage": True,
            **weather_data,
        }

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
                        game_id,
                        selected.kickoff_utc,
                        forecast_time,
                        weather_source=WEATHER_SOURCE_FORECAST,
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
                    weather_source=WEATHER_SOURCE_FORECAST,
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
        *,
        as_of_utc: datetime | None = None,
        base_path=None,
    ) -> pd.DataFrame:
        """THE LIVE WEEKLY WEATHER INGEST. Loads a week and forecasts it.

        R8: this entry point used to read the ARCHIVE endpoint, which cannot answer
        for a game that has not been played, and the fourteen 2024 rows in
        `data/silver/weather.parquet` are the evidence it never ran forward. It now
        reads the FORECAST endpoint through :meth:`ingest_week_forecast`.

        THE SCOPE IS ONE WEEK, and that is a consequence rather than a restriction.
        A whole season cannot be forecast: most of it lies beyond the declared
        horizon, so a season-wide call would correctly refuse. History is the
        historical backfill command's job.

        ``as_of_utc`` defaults to NOW here, and only here. This is the entry point
        of a live run, so "now" is the honest answer to "when is this being asked";
        the helpers underneath take it as an argument so that the RULES stay
        testable while the RUN stays live.

        Args:
            season: Season to ingest (default: current).
            week: Week to ingest (default: current).
            forecast_time: When this forecast was taken (default: now, in UTC).
            as_of_utc: The instant the horizon is measured from (default: now).
            base_path: Data root, threaded through to the read and the write.

        Returns:
            The validated frame that was written, or an empty frame when the week
            has no games in silver.
        """
        if season is None:
            season, current_week = get_current_nfl_week()
            if week is None:
                week = current_week
        if week is None:
            raise WeatherDataError(
                "the live weather ingest runs ONE week at a time and no week was "
                f"given for season {season}. A whole season cannot be forecast: "
                f"most of it lies beyond the {FORECAST_HORIZON_DAYS}-day horizon "
                "and the run would correctly refuse. Use "
                "scripts/backfill_historical_weather.py for history."
            )

        if as_of_utc is None:
            as_of_utc = datetime.now(UTC)
        if forecast_time is None:
            forecast_time = datetime.now(UTC)

        logger.info(
            "Starting live weather FORECAST ingestion",
            season=season,
            week=week,
            as_of=as_of_utc.isoformat(),
        )

        venues_df = self._load_venue_data()
        games_df = self._load_games_data(season, week)
        if games_df.empty:
            logger.warning("No games found", season=season, week=week)
            return pd.DataFrame()

        return self.ingest_week_forecast(
            games_df,
            venues_df,
            as_of_utc=as_of_utc,
            forecast_time=forecast_time,
            base_path=base_path,
        )


def main():
    """CLI entry point for the LIVE weather ingest (the FORECAST endpoint).

    ONE WEEK AT A TIME, and a season-wide request is refused rather than truncated.
    A forecast cannot reach most of a season, so a `--season 2018` style call has no
    honest answer here; `scripts/backfill_historical_weather.py` is the command for
    history. Refusing by name beats returning whichever weeks happened to fit.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Ingest LIVE NFL weather from the Open-Meteo FORECAST endpoint, one week "
            "at a time. For seasons already played use "
            "scripts/backfill_historical_weather.py, which holds the archive endpoint."
        )
    )

    from utils.ingestion_args import add_standard_ingestion_args, parse_season_week_args

    parser = add_standard_ingestion_args(parser)
    parser.add_argument(
        "--forecast-time", type=str, help="Forecast time (ISO format, default: now)"
    )

    args = parser.parse_args()

    try:
        from utils import setup_logging

        setup_logging()

        seasons, weeks = parse_season_week_args(args)
        season = seasons[0] if seasons else None
        if seasons and len(seasons) > 1:
            print(
                "The live weather ingest runs one season at a time. "
                f"Using the first: {season}."
            )

        if not weeks:
            print(
                "A --week is required. The live path reads the FORECAST endpoint, "
                f"which reaches {FORECAST_HORIZON_DAYS} days ahead at most, so a "
                "whole season cannot be forecast. Use "
                "scripts/backfill_historical_weather.py for seasons already played.",
                file=sys.stderr,
            )
            sys.exit(1)
        if len(weeks) > 1:
            print(
                "The live weather ingest writes ONE week atomically and was given "
                f"{len(weeks)} weeks: {weeks}. Run it once per week; an "
                "all-or-nothing promise over several weeks is not one.",
                file=sys.stderr,
            )
            sys.exit(1)
        week = weeks[0]

        forecast_time = None
        if args.forecast_time:
            try:
                forecast_time = datetime.fromisoformat(args.forecast_time)
                if forecast_time.tzinfo is None:
                    forecast_time = forecast_time.replace(tzinfo=ZoneInfo("Etc/UTC"))
            except ValueError:
                print(f"Invalid forecast time format: {args.forecast_time}")
                sys.exit(1)

        ingester = WeatherDataIngester()
        weather_df = ingester.ingest_weather(
            season=season, week=week, forecast_time=forecast_time
        )

        if weather_df.empty:
            print("No weather data ingested")
            return

        print(f"Successfully ingested {len(weather_df)} weather forecast records")
        print(f"Season: {season}, Week: {week}")
        print(f"Outdoor games: {weather_df['is_outdoor'].sum()}")
        print(f"Indoor games: {(~weather_df['is_outdoor']).sum()}")

        outdoor_weather = weather_df[weather_df["is_outdoor"]]
        if not outdoor_weather.empty:
            print("")
            print("Outdoor forecast summary:")
            print(f"  Temperature: {outdoor_weather['temp_f'].mean():.1f}F avg")
            print(f"  Wind: {outdoor_weather['wind_mph'].mean():.1f} mph avg")
            print(f"  Cold games: {outdoor_weather['is_cold'].sum()}")
            print(f"  Windy games: {outdoor_weather['is_windy'].sum()}")

    except (DataIngestionError, WeatherDataError, OSError, ValueError) as e:
        logger.error("Weather ingestion CLI failed", error=str(e))
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
