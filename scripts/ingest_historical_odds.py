"""Ingest historical NFL betting odds data from nflreadpy.

Transforms nflreadpy schedule data (which includes closing lines) into
standardized OddsSchema records with synthetic Friday 6 PM ET snapshot
timestamps. Stores as timestamped Bronze snapshots and upserts to Silver.
"""

import argparse
import sys
from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import nflreadpy as nfl
import pandas as pd

from data.quality_gates import validate_bronze_to_silver
from data.schemas import OddsSchema
from data.storage import save_bronze_snapshot, upsert_silver
from utils import DataIngestionError, get_logger, log_data_operation
from utils.game_id_utils import create_standard_game_id
from utils.team_data import normalize_team_abbreviation

logger = get_logger(__name__)

# Playoff game types that require odds data (like regular season)
_PLAYOFF_TYPES = {"WC", "DIV", "CON", "SB"}


def get_synthetic_snapshot_ts(gameday: str) -> datetime:
    """Generate synthetic Friday 6 PM ET snapshot time for a game.

    For historical data from nflreadpy, we don't have actual snapshot times.
    Use the gameday to find the preceding Friday at 6 PM ET.

    Args:
        gameday: Game date as string (YYYY-MM-DD)

    Returns:
        datetime: Friday 6 PM ET before the game, as UTC datetime
    """
    et = ZoneInfo("America/New_York")
    game_date = pd.to_datetime(gameday)

    # Find the preceding Friday (weekday 4 = Friday)
    days_since_friday = (game_date.weekday() - 4) % 7
    if days_since_friday == 0:
        days_since_friday = 7  # If gameday IS Friday, use prior Friday
    friday = game_date - timedelta(days=days_since_friday)

    snapshot = datetime(friday.year, friday.month, friday.day, 18, 0, tzinfo=et)
    # Convert to UTC for storage
    return snapshot.astimezone(ZoneInfo("UTC"))


def transform_nfl_odds_to_standard_format(schedules_df: pd.DataFrame) -> pd.DataFrame:
    """Transform nflreadpy schedules with betting odds to standard odds format.

    Filters out preseason games, normalizes team abbreviations through
    utils.team_data, and generates synthetic Friday 6 PM ET snapshot timestamps.

    Args:
        schedules_df: DataFrame from nflreadpy load_schedules().to_pandas()

    Returns:
        DataFrame in standard OddsSchema format
    """
    # Filter out preseason games -- odds only required for REG and playoff types
    valid_game_types = {"REG"} | _PLAYOFF_TYPES
    filtered = schedules_df[schedules_df["game_type"].isin(valid_game_types)].copy()

    if filtered.empty:
        logger.warning("No regular season or playoff games found")
        return pd.DataFrame()

    # Filter for games with any betting data
    betting_games = filtered[
        filtered["spread_line"].notna()
        | filtered["total_line"].notna()
        | filtered["away_moneyline"].notna()
    ].copy()

    if betting_games.empty:
        logger.warning("No games with betting odds found")
        return pd.DataFrame()

    # Create standardized odds records
    odds_records = []

    for _, game in betting_games.iterrows():
        try:
            # Normalize team abbreviations using canonical mapping
            home_team = normalize_team_abbreviation(game["home_team"])
            away_team = normalize_team_abbreviation(game["away_team"])

            # Create project-standard game ID
            game_id = create_standard_game_id(
                season=game["season"],
                week=game["week"],
                away_team=away_team,
                home_team=home_team,
            )

            # Generate synthetic snapshot timestamp from gameday
            snapshot_ts = get_synthetic_snapshot_ts(game["gameday"])

            # Convert moneyline values to int (may be float from pandas)
            ml_home = (
                int(game["home_moneyline"])
                if pd.notna(game.get("home_moneyline"))
                else None
            )
            ml_away = (
                int(game["away_moneyline"])
                if pd.notna(game.get("away_moneyline"))
                else None
            )

            # Convert juice/vig values to int
            spread_ju_home = (
                int(game["home_spread_odds"])
                if pd.notna(game.get("home_spread_odds"))
                else -110
            )
            spread_ju_away = (
                int(game["away_spread_odds"])
                if pd.notna(game.get("away_spread_odds"))
                else -110
            )
            total_over_ju = (
                int(game["over_odds"]) if pd.notna(game.get("over_odds")) else -110
            )
            total_under_ju = (
                int(game["under_odds"]) if pd.notna(game.get("under_odds")) else -110
            )

            # Build odds record matching OddsSchema
            odds_record = {
                "game_id": game_id,
                "snapshot_ts": snapshot_ts,
                "sportsbook": "nflverse_closing",
                "ml_home": ml_home,
                "ml_away": ml_away,
                "spread": game.get("spread_line")
                if pd.notna(game.get("spread_line"))
                else None,
                "spread_ju_home": spread_ju_home,
                "spread_ju_away": spread_ju_away,
                "total": game.get("total_line")
                if pd.notna(game.get("total_line"))
                else None,
                "total_over_ju": total_over_ju,
                "total_under_ju": total_under_ju,
                "is_live": False,
                "last_update": None,
                "created_at": datetime.now(UTC),
            }

            odds_records.append(odds_record)

        except Exception as e:
            logger.warning(
                "Failed to transform odds record",
                home_team=game.get("home_team"),
                away_team=game.get("away_team"),
                error=str(e),
            )
            continue

    odds_df = pd.DataFrame(odds_records)

    logger.info(
        "Transformed odds data",
        input_games=len(betting_games),
        output_records=len(odds_df),
    )

    return odds_df


def ingest_historical_odds_for_seasons(seasons: list[int]) -> pd.DataFrame:
    """Ingest historical odds data for specified seasons.

    Args:
        seasons: List of seasons to ingest

    Returns:
        Combined DataFrame with all odds data
    """
    all_odds = []

    for season in seasons:
        try:
            logger.info("Ingesting historical odds", season=season)

            # Load schedules with betting data from nflreadpy
            schedules = nfl.load_schedules([season]).to_pandas()

            # Transform to standard odds format (handles filtering internally)
            odds_df = transform_nfl_odds_to_standard_format(schedules)

            if not odds_df.empty:
                # Save timestamped Bronze snapshot per season
                save_bronze_snapshot(
                    odds_df,
                    "odds",
                    season=season,
                    week=0,  # 0 indicates full-season snapshot
                )

                all_odds.append(odds_df)
                logger.info(
                    "Successfully processed season",
                    season=season,
                    odds_records=len(odds_df),
                )
            else:
                logger.warning("No odds data generated", season=season)

        except (ConnectionError, TimeoutError, ValueError, KeyError) as e:
            logger.error(
                "Failed to ingest odds for season", season=season, error=str(e)
            )
            raise DataIngestionError(f"Season {season} ingestion failed: {e}")

    if not all_odds:
        raise DataIngestionError("No odds data was successfully ingested")

    # Combine all seasons
    combined_odds = pd.concat(all_odds, ignore_index=True)

    # Validate through OddsSchema quality gate
    validated_odds = validate_bronze_to_silver(combined_odds, OddsSchema)

    # Upsert to Silver odds_snapshot table
    upsert_silver(validated_odds, "odds_snapshot")

    logger.info(
        "Historical odds ingestion completed",
        seasons=seasons,
        total_records=len(validated_odds),
        unique_games=validated_odds["game_id"].nunique(),
    )

    return validated_odds


def main():
    """CLI entry point for historical odds data ingestion."""
    parser = argparse.ArgumentParser(
        description="Ingest historical NFL odds data from nflreadpy"
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=[2018, 2019, 2020, 2021, 2022, 2023, 2024],
        help="Seasons to ingest (default: 2018-2024)",
    )

    args = parser.parse_args()

    try:
        # Setup logging
        from utils import setup_logging

        setup_logging()

        logger.info("Starting historical odds ingestion", seasons=args.seasons)

        # Ingest historical odds
        odds_df = ingest_historical_odds_for_seasons(args.seasons)

        # Log operation
        log_data_operation(
            operation="ingest_historical",
            table="odds_snapshot",
            rows=len(odds_df),
            metadata={
                "seasons": args.seasons,
                "unique_games": odds_df["game_id"].nunique(),
                "data_source": "nflreadpy",
                "sportsbook": "nflverse_closing",
            },
        )

        print(f"Successfully ingested {len(odds_df)} historical odds records")
        print(f"Covering {odds_df['game_id'].nunique()} unique games")
        print(f"Seasons: {args.seasons}")

        # Show sample data
        print("\nSample historical odds data:")
        sample_cols = ["game_id", "sportsbook", "spread", "total", "ml_home", "ml_away"]
        available_cols = [col for col in sample_cols if col in odds_df.columns]
        print(odds_df[available_cols].head().to_string(index=False))

    except Exception as e:
        logger.error("Historical odds ingestion failed", error=str(e))
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
