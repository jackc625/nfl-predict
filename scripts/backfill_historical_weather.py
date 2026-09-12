"""Historical weather backfill -- the Open-Meteo ARCHIVE endpoint's ONLY caller.

THE ENDPOINT IS QUARANTINED HERE, NOT DELETED (D33-26)
------------------------------------------------------
`https://archive-api.open-meteo.com/v1/archive` is an ERA5 REANALYSIS product. It
describes weather that has already occurred, which means it is the right answer for
history and a category error for a game that has not been played. Pointing the live
2026 path at it was R8's defect: the request succeeds, the hours come back null, and
the only "repairs" available at that point -- read the archive anyway, or substitute a
seasonal average -- are the fabricated-data class this project has already disclosed
once.

So the live path can no longer reach it. `scripts/ingest_weather.py` holds the FORECAST
endpoint and nothing else, and `tests/unit/test_weather_archive_quarantined.py` scans it
structurally to keep that true: a source scan is the only instrument that works here,
because an archive fallback RETURNS A VALUE and therefore leaves a green test suite
behind it.

WHY IT IS KEPT AT ALL, WHICH IS THE OTHER HALF OF THE DECISION
---------------------------------------------------------------
Deleting the endpoint would strand two decades of history. Open-Meteo's Historical
Forecast API -- what the forecast SAID at the time -- reaches back only to about 2021.
For 2002-2020 the archive endpoint is the only source there is, and the historical
weather backfill it powers is the NAMED flip condition for the 2026 gold-default switch
in `features/weather.py`. A quarantine that removed the capability would quietly cancel
Phase 37's input.

This module is therefore a separately-named CLI command whose name says what it does. It
is deliberately EXCLUDED from the quarantine scan's module list, and that exclusion is
asserted rather than assumed -- see
`test_the_scan_does_not_flag_the_legitimate_backfill_module`.

WHAT LIVES HERE
---------------
* `ARCHIVE_ENDPOINT_URL` and `fetch_game_weather`, moved verbatim from the ingest module.
* `HistoricalWeatherBackfiller`, a subclass of the live `WeatherDataIngester`. It
  inherits the SHARED machinery -- venue resolution under the D33-15 stadium_id routing,
  the roof map, the record factories -- and adds only the archive fetch and the archive
  ingest. Subclassing rather than copying is what stops the two paths from drifting apart
  about which stadium a game is at.
* `stamp_weather_source_on_existing_rows`, the one-shot provenance backfill for the rows
  that predate the `weather_source` column.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

import argparse
import asyncio
import sys
from datetime import UTC, datetime
from typing import Any

import httpx
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from data.quality_gates import validate_bronze_to_silver
from data.schemas import WeatherSchema
from data.storage import save_bronze_snapshot, upsert_silver
from scripts.ingest_weather import (
    HOURLY_VARIABLES,
    WEATHER_SOURCE_VOCABULARY,
    WeatherDataIngester,
    validate_weather_source,
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

logger = get_logger(__name__)

# The Open-Meteo ARCHIVE (ERA5 reanalysis) endpoint. THE ONLY DECLARATION OF IT IN
# THIS REPOSITORY. See the module docstring for why it is quarantined here rather
# than deleted: the Historical Forecast API reaches back only to about 2021, so
# this is the sole source for 2002-2020.
ARCHIVE_ENDPOINT_URL = "https://archive-api.open-meteo.com/v1/archive"

# The two HISTORICAL members of the weather_source vocabulary, named here rather
# than in the live ingest module. That split is not cosmetic: the quarantine's
# mechanical form is that `scripts/ingest_weather.py` exposes no name containing
# "ARCHIVE", and a WEATHER_SOURCE_ARCHIVE constant over there would pass the
# source scan while making that runtime check false. The values are pinned
# against the shared vocabulary immediately below, so the split cannot drift.
WEATHER_SOURCE_ARCHIVE = "archive"
WEATHER_SOURCE_HISTORICAL_FORECAST = "historical_forecast"

assert WEATHER_SOURCE_ARCHIVE in WEATHER_SOURCE_VOCABULARY
assert WEATHER_SOURCE_HISTORICAL_FORECAST in WEATHER_SOURCE_VOCABULARY


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
        response = await client.get(ARCHIVE_ENDPOINT_URL, params=params)
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


class HistoricalWeatherBackfiller(WeatherDataIngester):
    """The archive-endpoint weather path, subclassing the live forecast ingester.

    SUBCLASS RATHER THAN COPY. Venue resolution under the D33-15 stadium_id routing,
    the nflverse roof map, and the record factories are SHARED behaviour, and two
    copies of them would eventually disagree about which stadium a neutral-site game
    is at -- the exact defect COLD-09 exists to fix. What this class adds is only the
    two things that are genuinely different: the archive fetch, and the archive
    ingest.
    """

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
                    weather_source=WEATHER_SOURCE_ARCHIVE,
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
                    weather_source=WEATHER_SOURCE_ARCHIVE,
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

    def backfill_weather(
        self,
        season: int | None = None,
        week: int | None = None,
        forecast_time: datetime | None = None,
        base_path=None,
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
                save_bronze_snapshot(
                    weather_df,
                    "weather",
                    season=season,
                    week=week,
                    base_path=base_path,
                )
            else:
                # For full-season ingestion, use week=0 as a sentinel
                save_bronze_snapshot(
                    weather_df,
                    "weather",
                    season=season,
                    week=0,
                    base_path=base_path,
                )

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
            upsert_silver(validated_df, "weather", base_path=base_path)

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


def stamp_weather_source_on_existing_rows(
    table: str = "weather",
    base_path=None,
    *,
    source: str = WEATHER_SOURCE_ARCHIVE,
) -> pd.DataFrame:
    """Stamp `weather_source` onto silver weather rows that predate the column.

    A ONE-SHOT PROVENANCE BACKFILL, not an ingest. It reads the silver table, fills
    only the rows whose `weather_source` is absent or null, and writes the frame
    back through the same storage-layer path every other silver write uses. It
    fetches nothing and it changes no weather VALUE -- only the column that says
    where each value came from.

    Every pre-existing row is stamped `archive` because every pre-existing row was
    produced by the archive endpoint: those fourteen rows are 2024 Week 6, written
    by the only weather ingest this project has ever run, and that ingest hit
    `ARCHIVE_ENDPOINT_URL`. This is a statement of fact about how they were made,
    not a default.

    Args:
        table: Silver table name.
        base_path: Data root. Threaded so a test can redirect the read and write.
        source: The provenance value to stamp. Validated against the closed
            vocabulary before anything is written.

    Returns:
        The frame as written.

    Raises:
        ValueError: *source* is outside the closed vocabulary.
    """
    from pathlib import Path

    from conf.settings import get_settings

    validate_weather_source(source)

    # Read the parquet DIRECTLY rather than through `load_dataframe`, which has no
    # `base_path` and would resolve the production root regardless of what a caller
    # passed. A sandbox argument that is silently ignored is worse than none: it
    # reads as a guarantee and behaves as a comment.
    root = (
        Path(base_path)
        if base_path is not None
        else Path(get_settings().config.data.root_path)
    )
    silver_path = root / "silver" / f"{table}.parquet"
    if not silver_path.is_file():
        raise DataIngestionError(
            f"no silver weather table at {silver_path.as_posix()} to stamp."
        )
    frame = pd.read_parquet(silver_path, engine="pyarrow")
    if "weather_source" not in frame.columns:
        frame["weather_source"] = None

    unstamped = frame["weather_source"].isna()
    filled = int(unstamped.sum())
    frame.loc[unstamped, "weather_source"] = source

    upsert_silver(frame, table, base_path=base_path)
    logger.info(
        "Stamped weather_source on pre-existing rows",
        table=table,
        rows=len(frame),
        filled=filled,
        source=source,
    )
    return frame


def main():
    """CLI entry point for the HISTORICAL (archive-endpoint) weather backfill.

    Separately named from `scripts/ingest_weather.py` on purpose: the two commands
    reach different endpoints with different temporal meanings, and one command
    with a mode flag would put them one typo apart.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Backfill HISTORICAL NFL weather from the Open-Meteo ARCHIVE endpoint. "
            "For an unplayed game use scripts/ingest_weather.py, which reaches the "
            "FORECAST endpoint -- the archive cannot answer for a game that has not "
            "been played."
        )
    )
    parser.add_argument("--season", type=int, help="Season to backfill")
    parser.add_argument(
        "--week", type=int, default=None, help="Week to backfill (default: all)"
    )
    parser.add_argument(
        "--stamp-weather-source",
        action="store_true",
        help=(
            "Do not fetch anything. Stamp weather_source='archive' onto silver "
            "weather rows that predate the column, and exit."
        ),
    )

    args = parser.parse_args()

    from utils import setup_logging

    setup_logging()

    if args.stamp_weather_source:
        frame = stamp_weather_source_on_existing_rows()
        print(f"Stamped weather_source on {len(frame)} silver weather rows")
        return

    if args.season is None:
        print("--season is required unless --stamp-weather-source is given")
        sys.exit(1)

    try:
        backfiller = HistoricalWeatherBackfiller()
        weather_df = backfiller.backfill_weather(season=args.season, week=args.week)
        if weather_df.empty:
            print("No weather data backfilled")
            return
        print(f"Backfilled {len(weather_df)} weather records from the ARCHIVE endpoint")
        print(f"Season: {args.season}, Week: {args.week or 'all'}")
    except (DataIngestionError, WeatherDataError, OSError, ValueError) as exc:
        logger.error("Historical weather backfill failed", error=str(exc))
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
