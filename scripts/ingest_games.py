"""Game data ingestion using nflreadpy."""

import argparse
import sys
from datetime import UTC, datetime

import nflreadpy as nfl
import pandas as pd

from conf.settings import get_settings
from data.schemas import GameSchema
from data.storage import get_db_connection, save_dataframe
from utils import (
    DataIngestionError,
    get_current_nfl_week,
    get_logger,
    log_data_operation,
)
from utils.game_id_utils import is_valid_game_id

logger = get_logger(__name__)


class GameDataIngester:
    """NFL game data ingestion from nflreadpy."""

    def __init__(self):
        """Initialize game data ingester."""
        self.settings = get_settings()
        self.db = get_db_connection()

        # Team name mapping to canonical abbreviations
        self.team_mapping = self._build_team_mapping()

    def _build_team_mapping(self) -> dict[str, str]:
        """Build mapping from various team name formats to canonical abbreviations."""
        # This should be updated based on actual nflreadpy team formats
        return {
            # Standard abbreviations (these should pass through unchanged)
            "ARI": "ARI",
            "ATL": "ATL",
            "BAL": "BAL",
            "BUF": "BUF",
            "CAR": "CAR",
            "CHI": "CHI",
            "CIN": "CIN",
            "CLE": "CLE",
            "DAL": "DAL",
            "DEN": "DEN",
            "DET": "DET",
            "GB": "GB",
            "HOU": "HOU",
            "IND": "IND",
            "JAX": "JAX",
            "KC": "KC",
            "LV": "LV",
            "LAC": "LAC",
            "LAR": "LAR",
            "MIA": "MIA",
            "MIN": "MIN",
            "NE": "NE",
            "NO": "NO",
            "NYG": "NYG",
            "NYJ": "NYJ",
            "PHI": "PHI",
            "PIT": "PIT",
            "SF": "SF",
            "SEA": "SEA",
            "TB": "TB",
            "TEN": "TEN",
            "WAS": "WAS",
            # Alternative formats that might appear in data
            "GNB": "GB",
            "NWE": "NE",
            "NOR": "NO",
            "SFO": "SF",
            "TAM": "TB",
            "HST": "HOU",
            "CLV": "CLE",
            "LVR": "LV",
            "OAK": "LV",  # Oakland moved to Las Vegas
            "SD": "LAC",  # San Diego moved to Los Angeles
            "STL": "LAR",  # St. Louis moved to Los Angeles
        }

    def _normalize_team_name(self, team: str) -> str:
        """Normalize team name to canonical abbreviation."""
        if not team:
            return team

        team_upper = team.upper().strip()
        return self.team_mapping.get(team_upper, team_upper)

    def _get_venue_roof_type(self, venue: str) -> str:
        """Determine venue roof type from venue name."""
        # This is a simplified mapping - should be updated with comprehensive venue data
        indoor_venues = {
            "Mercedes-Benz Superdome",
            "NRG Stadium",
            "Ford Field",
            "U.S. Bank Stadium",
            "Allegiant Stadium",
            "State Farm Stadium",
            "AT&T Stadium",
            "Lucas Oil Stadium",
            "Caesars Superdome",
        }

        retractable_venues = {
            "Mercedes-Benz Stadium",
            "State Farm Stadium",
            "NRG Stadium",
            "Lucas Oil Stadium",
            "AT&T Stadium",
        }

        if venue in indoor_venues:
            return "indoor"
        if venue in retractable_venues:
            return "retractable"
        return "outdoor"

    def _create_game_id(self, row: pd.Series) -> str:
        """Create standardized game ID."""
        season = row["season"]
        week = row["week"]
        home_team = self._normalize_team_name(row["home_team"])
        away_team = self._normalize_team_name(row["away_team"])

        game_id = f"{season}_W{week:02d}_{away_team}@{home_team}"

        # Validate the generated game ID
        if not is_valid_game_id(game_id):
            raise ValueError(f"Generated invalid game ID: {game_id}")

        return game_id

    def _determine_result(self, row: pd.Series) -> int | None:
        """Determine game result from home team perspective."""
        home_score = row.get("home_score")
        away_score = row.get("away_score")

        if pd.isna(home_score) or pd.isna(away_score):
            return None

        if home_score > away_score:
            return 1  # Home win
        if home_score < away_score:
            return -1  # Away win
        return 0  # Tie

    def fetch_schedule_data(
        self, seasons: list[int], weeks: list[int] | None = None
    ) -> pd.DataFrame:
        """
        Fetch schedule data from nflreadpy.

        Args:
            seasons: List of seasons to fetch
            weeks: Optional list of specific weeks

        Returns:
            DataFrame with schedule data
        """
        try:
            logger.info("Fetching schedule data", seasons=seasons, weeks=weeks)

            # Load schedule data (nflreadpy returns Polars, convert to pandas)
            schedule_df = nfl.load_schedules(seasons).to_pandas()

            if weeks:
                schedule_df = schedule_df[schedule_df["week"].isin(weeks)]

            logger.info(
                "Fetched schedule data", rows=len(schedule_df), seasons=len(seasons)
            )

            return schedule_df

        except (ConnectionError, TimeoutError, ValueError) as e:
            logger.error(
                "Failed to fetch schedule data",
                seasons=seasons,
                weeks=weeks,
                error=str(e),
            )
            raise DataIngestionError(f"Schedule data fetch failed: {e}")

    def fetch_pbp_data(
        self, seasons: list[int], weeks: list[int] | None = None
    ) -> pd.DataFrame:
        """
        Fetch play-by-play data for game results.

        Args:
            seasons: List of seasons to fetch
            weeks: Optional list of specific weeks

        Returns:
            DataFrame with game results
        """
        try:
            logger.info(
                "Fetching play-by-play data for results", seasons=seasons, weeks=weeks
            )

            # Load play-by-play data (nflreadpy returns Polars, convert to pandas)
            pbp_df = nfl.load_pbp(seasons).to_pandas()

            if weeks:
                pbp_df = pbp_df[pbp_df["week"].isin(weeks)]

            # Extract game-level results
            game_results = (
                pbp_df.groupby(["game_id", "season", "week", "home_team", "away_team"])
                .agg({"home_score": "max", "away_score": "max"})
                .reset_index()
            )

            logger.info("Extracted game results from PBP data", rows=len(game_results))

            return game_results

        except (ConnectionError, TimeoutError, ValueError) as e:
            logger.error(
                "Failed to fetch PBP data", seasons=seasons, weeks=weeks, error=str(e)
            )
            raise DataIngestionError(f"PBP data fetch failed: {e}")

    def transform_schedule_data(self, schedule_df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform raw schedule data to our schema format.

        Args:
            schedule_df: Raw schedule DataFrame from nflreadpy

        Returns:
            Transformed DataFrame matching GameSchema
        """
        logger.info("Transforming schedule data", input_rows=len(schedule_df))

        # Create transformed DataFrame
        transformed_data = []

        for _, row in schedule_df.iterrows():
            try:
                # Normalize team names
                home_team = self._normalize_team_name(row["home_team"])
                away_team = self._normalize_team_name(row["away_team"])

                # Create game record
                game_record = {
                    "game_id": self._create_game_id(row),
                    "season": int(row["season"]),
                    "week": int(row["week"]),
                    "kickoff_et": pd.to_datetime(
                        row["gameday"] + " " + row.get("gametime", "13:00")
                    ),
                    "home_team": home_team,
                    "away_team": away_team,
                    "venue": row.get("stadium", "Unknown Stadium"),
                    "venue_roof": self._get_venue_roof_type(row.get("stadium", "")),
                    "home_score": row.get("home_score")
                    if pd.notna(row.get("home_score"))
                    else None,
                    "away_score": row.get("away_score")
                    if pd.notna(row.get("away_score"))
                    else None,
                    "result": self._determine_result(row),
                    "game_type": row.get("game_type", "REG"),
                    "season_type": row.get("season_type", "Regular"),
                    "neutral_site": row.get("neutral_site", False),
                }

                transformed_data.append(game_record)

            except Exception as e:
                logger.warning(
                    "Failed to transform game record",
                    game_data=row.to_dict(),
                    error=str(e),
                )
                continue

        transformed_df = pd.DataFrame(transformed_data)
        logger.info(
            "Transformed schedule data",
            input_rows=len(schedule_df),
            output_rows=len(transformed_df),
        )

        return transformed_df

    def validate_game_data(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate game data against schema.

        Args:
            games_df: DataFrame with game data

        Returns:
            Validated DataFrame (invalid records removed)
        """
        logger.info("Validating game data", input_rows=len(games_df))

        valid_records = []
        validation_errors = []

        for idx, row in games_df.iterrows():
            try:
                # Validate against schema
                game = GameSchema(**row.to_dict())
                valid_records.append(game.model_dump())

            except Exception as e:
                validation_errors.append(f"Row {idx}: {e!s}")
                logger.warning(
                    "Game data validation failed", row_index=idx, error=str(e)
                )

        if validation_errors:
            logger.warning(
                "Game data validation issues",
                total_errors=len(validation_errors),
                sample_errors=validation_errors[:5],
            )

        validated_df = pd.DataFrame(valid_records)
        logger.info(
            "Game data validation completed",
            input_rows=len(games_df),
            output_rows=len(validated_df),
            errors=len(validation_errors),
        )

        return validated_df

    def ingest_games(
        self,
        seasons: list[int] | None = None,
        weeks: list[int] | None = None,
        include_results: bool = True,
    ) -> pd.DataFrame:
        """
        Full game data ingestion pipeline.

        Args:
            seasons: Seasons to ingest (default: current season)
            weeks: Specific weeks to ingest (default: all)
            include_results: Whether to fetch game results from PBP data

        Returns:
            Ingested and validated game data
        """
        if seasons is None:
            current_season, _ = get_current_nfl_week()
            seasons = [current_season]

        logger.info(
            "Starting game data ingestion",
            seasons=seasons,
            weeks=weeks,
            include_results=include_results,
        )

        try:
            # Fetch schedule data
            schedule_df = self.fetch_schedule_data(seasons, weeks)

            # Transform to our schema
            games_df = self.transform_schedule_data(schedule_df)

            # Fetch results if requested and available
            if include_results:
                try:
                    results_df = self.fetch_pbp_data(seasons, weeks)

                    # Merge results into games data
                    games_df = self._merge_game_results(games_df, results_df)

                except Exception as e:
                    logger.warning("Failed to fetch game results", error=str(e))

            # Validate data
            validated_df = self.validate_game_data(games_df)

            # Add metadata timestamp

            validated_df["created_at"] = datetime.now(UTC)

            # Save to bronze layer (raw)
            save_dataframe(
                schedule_df,
                "games_raw_bronze",
                layer="bronze",
                save_to_db=False,  # Don't save raw data to DB
            )

            # Save to silver layer (processed)
            save_dataframe(
                validated_df,
                "games",
                layer="silver",
                partition_cols=["season"] if len(seasons) > 1 else None,
            )

            log_data_operation(
                operation="ingest",
                table="games",
                rows=len(validated_df),
                seasons=seasons,
                weeks=weeks,
            )

            logger.info(
                "Game data ingestion completed successfully",
                total_games=len(validated_df),
                seasons=seasons,
            )

            return validated_df

        except Exception as e:
            logger.error("Game data ingestion failed", error=str(e))
            raise DataIngestionError(f"Game ingestion failed: {e}")

    def _merge_game_results(
        self, games_df: pd.DataFrame, results_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge game results into games DataFrame."""
        logger.info(
            "Merging game results", games=len(games_df), results=len(results_df)
        )

        # Create merge keys
        games_df["merge_key"] = (
            games_df["season"].astype(str)
            + "_"
            + games_df["week"].astype(str)
            + "_"
            + games_df["home_team"]
            + "_"
            + games_df["away_team"]
        )

        results_df["merge_key"] = (
            results_df["season"].astype(str)
            + "_"
            + results_df["week"].astype(str)
            + "_"
            + results_df["home_team"]
            + "_"
            + results_df["away_team"]
        )

        # Merge results
        merged_df = games_df.merge(
            results_df[["merge_key", "home_score", "away_score"]],
            on="merge_key",
            how="left",
            suffixes=("", "_pbp"),
        )

        # Update scores where available
        merged_df["home_score"] = merged_df["home_score_pbp"].combine_first(
            merged_df["home_score"]
        )
        merged_df["away_score"] = merged_df["away_score_pbp"].combine_first(
            merged_df["away_score"]
        )

        # Recalculate results
        for idx, row in merged_df.iterrows():
            merged_df.loc[idx, "result"] = self._determine_result(row)

        # Clean up
        merged_df = merged_df.drop(
            ["merge_key", "home_score_pbp", "away_score_pbp"], axis=1
        )

        logger.info("Game results merged", final_games=len(merged_df))
        return merged_df


def main():
    """CLI entry point for game data ingestion."""
    parser = argparse.ArgumentParser(description="Ingest NFL game data")

    # Add standardized ingestion arguments
    from utils.ingestion_args import (
        add_standard_ingestion_args,
        get_ingestion_summary,
        parse_season_week_args,
    )

    parser = add_standard_ingestion_args(parser)

    # Add games-specific arguments
    parser.add_argument(
        "--no-results", action="store_true", help="Skip fetching game results"
    )
    parser.add_argument(
        "--validate-only", action="store_true", help="Only validate existing data"
    )

    args = parser.parse_args()

    try:
        # Setup logging
        from utils import setup_logging

        setup_logging()

        ingester = GameDataIngester()

        if args.validate_only:
            # Validate existing data
            from data.storage import load_dataframe

            games_df = load_dataframe("games", layer="silver")
            validated_df = ingester.validate_game_data(games_df)
            print(f"Validated {len(validated_df)}/{len(games_df)} games")
            return

        # Parse standardized season/week arguments
        seasons, weeks = parse_season_week_args(args)

        # Log what we're about to ingest
        summary = get_ingestion_summary(seasons, weeks)
        logger.info(f"Starting game data ingestion for {summary}")

        # Run ingestion
        games_df = ingester.ingest_games(
            seasons=seasons, weeks=weeks, include_results=not args.no_results
        )

        print(f"Successfully ingested {len(games_df)} games")
        print(f"Seasons: {sorted(games_df['season'].unique())}")
        print(f"Weeks: {sorted(games_df['week'].unique())}")

    except Exception as e:
        logger.error("Game ingestion CLI failed", error=str(e))
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
