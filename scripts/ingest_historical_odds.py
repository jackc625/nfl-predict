"""Ingest historical NFL betting odds data from nflreadpy."""

import argparse
import sys
from datetime import datetime

import nflreadpy as nfl
import pandas as pd

from data.storage import save_dataframe
from utils import DataIngestionError, get_logger, log_data_operation
from utils.game_id_utils import create_standard_game_id, is_valid_game_id

logger = get_logger(__name__)


def transform_nfl_odds_to_standard_format(schedules_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform nflreadpy schedules with betting odds to standard odds format.

    Args:
        schedules_df: DataFrame from nflreadpy load_schedules().to_pandas()

    Returns:
        DataFrame in standard odds snapshot format
    """
    # Filter for games with betting data
    betting_games = schedules_df[
        schedules_df["spread_line"].notna()
        | schedules_df["total_line"].notna()
        | schedules_df["away_moneyline"].notna()
    ].copy()

    if betting_games.empty:
        logger.warning("No games with betting odds found")
        return pd.DataFrame()

    # Create standardized odds records
    odds_records = []

    for _, game in betting_games.iterrows():
        # Create standardized game ID instead of using nflreadpy format
        game_id = create_standard_game_id(
            season=game["season"],
            week=game["week"],
            away_team=game["away_team"],
            home_team=game["home_team"],
        )

        # Validate the generated game ID
        if not is_valid_game_id(game_id):
            logger.warning(f"Generated invalid game ID: {game_id}, skipping game")
            continue

        # Create a complete record with both game metadata and odds data
        base_record = {
            # Game metadata from schedules DataFrame
            "game_id": game_id,
            "week": game.get("week"),
            "kickoff_et": game.get("gameday"),
            "home_team": game.get("home_team"),
            "away_team": game.get("away_team"),
            "venue": game.get("stadium"),
            "venue_roof": game.get("roof"),
            "home_score": game.get("home_score"),
            "away_score": game.get("away_score"),
            "result": game.get("result"),
            "game_type": game.get("game_type"),
            "season_type": game.get("season_type"),
            "neutral_site": game.get("neutral_site"),
            "created_at": game.get("season"),  # Use season as creation reference
            # Odds data (consolidated from nflreadpy sources)
            "sportsbook": "consensus",  # nflreadpy provides consensus lines
            "ml_home": game.get("home_moneyline"),
            "ml_away": game.get("away_moneyline"),
            "spread": game.get("spread_line"),
            "spread_ju_home": game.get("home_spread_odds", -110),
            "spread_ju_away": game.get("away_spread_odds", -110),
            "total": game.get("total_line"),
            "total_over_ju": game.get("over_odds", -110),
            "total_under_ju": game.get("under_odds", -110),
            "is_live": False,
            "last_update": datetime.now(),
            "snapshot_ts": f"{game['season']}-09-19T18:00:00-04:00",  # Standard snapshot time
        }

        odds_records.append(base_record)

    odds_df = pd.DataFrame(odds_records)

    logger.info(
        "Transformed odds data",
        input_games=len(betting_games),
        output_records=len(odds_df),
    )

    return odds_df


def ingest_historical_odds_for_seasons(seasons: list[int]) -> pd.DataFrame:
    """
    Ingest historical odds data for specified seasons.

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

            # Filter for regular season games only
            regular_season = schedules[schedules["game_type"] == "REG"].copy()

            if regular_season.empty:
                logger.warning("No regular season games found", season=season)
                continue

            # Transform to standard odds format
            odds_df = transform_nfl_odds_to_standard_format(regular_season)

            if not odds_df.empty:
                all_odds.append(odds_df)
                logger.info(
                    "Successfully processed season",
                    season=season,
                    games=len(regular_season),
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

    logger.info(
        "Historical odds ingestion completed",
        seasons=seasons,
        total_records=len(combined_odds),
        unique_games=combined_odds["game_id"].nunique(),
    )

    return combined_odds


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
    parser.add_argument(
        "--save-bronze", action="store_true", help="Also save raw data to bronze layer"
    )

    args = parser.parse_args()

    try:
        # Setup logging
        from utils import setup_logging

        setup_logging()

        logger.info("Starting historical odds ingestion", seasons=args.seasons)

        # Ingest historical odds
        odds_df = ingest_historical_odds_for_seasons(args.seasons)

        # Save to silver layer
        save_dataframe(
            odds_df,
            table_name="odds_historical",
            layer="silver",
            partition_cols=None,  # Don't partition historical data
            save_to_db=False,  # Skip DuckDB for now due to timestamp issues
            save_to_parquet=True,
            append_mode=False,  # Don't append to avoid loading wrong partitioned data
        )

        # Also update main odds_snapshot table
        save_dataframe(
            odds_df,
            table_name="odds_snapshot",
            layer="silver",
            partition_cols=["snapshot_ts"] if len(odds_df) > 100 else None,
            save_to_db=False,
            save_to_parquet=True,
            append_mode=True,
        )

        # Log operation
        log_data_operation(
            operation="ingest_historical",
            table="odds_historical",
            rows=len(odds_df),
            metadata={
                "seasons": args.seasons,
                "unique_games": odds_df["game_id"].nunique(),
                "data_source": "nflreadpy",
            },
        )

        logger.info(
            "Historical odds ingestion completed successfully",
            total_records=len(odds_df),
            unique_games=odds_df["game_id"].nunique(),
            seasons=args.seasons,
        )

        print(f"Successfully ingested {len(odds_df)} historical odds records")
        print(f"Covering {odds_df['game_id'].nunique()} unique games")
        print(f"Seasons: {args.seasons}")

        # Show sample data
        print("\nSample historical odds data:")
        sample_cols = ["game_id", "sportsbook", "spread", "total", "ml_home", "ml_away"]
        print(odds_df[sample_cols].head().to_string(index=False))

    except Exception as e:
        logger.error("Historical odds ingestion failed", error=str(e))
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
