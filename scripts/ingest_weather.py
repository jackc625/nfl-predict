"""Weather data ingestion using Open-Meteo Historical Weather API.

Produces timestamped Bronze snapshots and upserts to Silver weather table.
Indoor games get zeroed weather fields (no API call).
Outdoor and retractable-roof games use real Open-Meteo data.
Missing weather for an outdoor game raises a hard error.
"""

import argparse
import asyncio
import sys
from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo

import httpx
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from conf.settings import get_settings
from data.quality_gates import validate_bronze_to_silver
from data.schemas import WeatherSchema
from data.storage import load_dataframe, save_bronze_snapshot, upsert_silver
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

    def _get_venue_coordinates(
        self, home_team: str, venues_df: pd.DataFrame
    ) -> tuple[float, float, str]:
        """Get venue coordinates and roof type for a team.

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

        venue = venue_info.iloc[0]
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

            # Get venue coordinates and roof type
            lat, lon, roof_type = self._get_venue_coordinates(
                game["home_team"], venues_df
            )

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
