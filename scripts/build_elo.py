"""
Build and update Elo ratings for NFL teams.

This script processes historical game data chronologically to build Elo ratings
for all NFL teams. It can process multiple seasons or update ratings for the
current season.

Usage:
    python scripts/build_elo.py --season 2024              # Process single season
    python scripts/build_elo.py --seasons 2020 2021 2022  # Process multiple seasons
    python scripts/build_elo.py --all-seasons              # Process all available seasons
    python scripts/build_elo.py --current                  # Process current season only
"""

import argparse
import sys

import pandas as pd

# Add project root to path
sys.path.append(".")

from conf.settings import get_settings
from data.storage import load_dataframe, save_dataframe
from ratings.elo import EloRatingSystem, is_divisional_game
from utils import get_current_nfl_week, get_logger, setup_logging

logger = get_logger(__name__)


class EloBuilder:
    """Build and manage Elo ratings for NFL teams."""

    def __init__(self):
        """Initialize Elo builder."""
        self.settings = get_settings()
        self.elo_system = EloRatingSystem()

    def load_games_data(self, seasons: list[int] | None = None) -> pd.DataFrame:
        """
        Load games data for specified seasons.

        Args:
            seasons: List of seasons to load (default: all available)

        Returns:
            DataFrame with game data
        """
        try:
            games_df = load_dataframe("games", layer="silver")

            if seasons:
                games_df = games_df[games_df["season"].isin(seasons)]

            logger.info(
                "Loaded games data",
                total_games=len(games_df),
                seasons=sorted(games_df["season"].unique())
                if len(games_df) > 0
                else [],
            )

            return games_df

        except (ValueError, KeyError, TypeError, FileNotFoundError, OSError) as e:
            logger.error("Failed to load games data", error=str(e))
            raise

    def process_seasons_chronologically(self, seasons: list[int]) -> pd.DataFrame:
        """
        Process multiple seasons chronologically to build Elo ratings.

        Args:
            seasons: List of seasons to process in order

        Returns:
            DataFrame with all processed games and rating updates
        """
        logger.info("Starting chronological Elo processing", seasons=seasons)

        all_processed_games = []

        for season in sorted(seasons):
            logger.info(f"Processing season {season}")

            # Load games for this season
            season_games = self.load_games_data([season])

            if len(season_games) == 0:
                logger.warning(f"No games found for season {season}")
                continue

            # Process season chronologically
            processed_games = self.elo_system.process_season_chronologically(
                season_games, season
            )

            all_processed_games.append(processed_games)

            # Log season summary
            current_ratings = self.elo_system.get_current_ratings(season)
            logger.info(
                f"Completed season {season}",
                games_processed=len(processed_games),
                teams_rated=len(current_ratings),
                top_team=current_ratings.iloc[0]["team"]
                if len(current_ratings) > 0
                else None,
                top_rating=current_ratings.iloc[0]["rating"]
                if len(current_ratings) > 0
                else None,
            )

        # Combine all processed games
        if all_processed_games:
            result_df = pd.concat(all_processed_games, ignore_index=True)
            logger.info(
                "Chronological processing completed",
                total_games=len(result_df),
                seasons_processed=len(seasons),
            )
            return result_df
        return pd.DataFrame()

    def update_current_season(self) -> pd.DataFrame:
        """
        Update Elo ratings for the current season only.

        Returns:
            DataFrame with current season games and updates
        """
        current_season, _ = get_current_nfl_week()

        logger.info(f"Updating Elo ratings for current season {current_season}")

        # Load existing ratings if available
        self.elo_system.load_ratings()

        # Process current season
        return self.process_seasons_chronologically([current_season])

    def build_elo_with_snapshots(self, start_season: int = 2002) -> pd.DataFrame:
        """Build Elo ratings and save per-game pre-game snapshots.

        For each game in chronological order:
        1. Capture PRE-GAME ratings for both teams
        2. Record snapshot (pre-game Elo, uncertainty, win prob, HFA)
        3. THEN process the game result (updates ratings)

        This ensures each game's snapshot contains the ratings computed
        from all prior games but NOT the current game's result (no batch
        leakage).

        Args:
            start_season: First season to process. Default 2002 for full
                burn-in (16 seasons before first backtest season 2018).

        Returns:
            DataFrame of per-game pre-game snapshots with columns:
            game_id, season, week, home_team, away_team, home_elo_pre,
            away_elo_pre, home_elo_uncertainty, away_elo_uncertainty,
            elo_prob_home, hfa_used.
        """
        logger.info(
            "Building Elo ratings with per-game snapshots",
            start_season=start_season,
        )

        # Reset Elo system for clean build
        self.elo_system = EloRatingSystem()

        # Load all games from silver layer
        all_games = self.load_games_data()
        available_seasons = sorted(all_games["season"].unique())
        seasons_to_process = [s for s in available_seasons if s >= start_season]

        if not seasons_to_process:
            logger.warning("No seasons to process", start_season=start_season)
            return pd.DataFrame()

        logger.info(
            "Processing seasons for Elo snapshots",
            seasons=seasons_to_process,
            total_seasons=len(seasons_to_process),
        )

        snapshots = []

        for season in seasons_to_process:
            # Apply season carryover
            self.elo_system.apply_season_carryover(season)

            # Learn HFA from prior season data (no lookahead per D-11)
            self.elo_system.learn_home_field_advantage(all_games, season)

            # Get this season's games sorted chronologically
            season_games = all_games[all_games["season"] == season].sort_values(
                "kickoff_et"
            )

            games_processed = 0
            for _, game in season_games.iterrows():
                # Skip games without results
                if pd.isna(game["home_score"]) or pd.isna(game["away_score"]):
                    continue

                home = game["home_team"]
                away = game["away_team"]

                # Step 1: Capture PRE-GAME ratings
                home_rating = self.elo_system.get_or_create_rating(home, season)
                away_rating = self.elo_system.get_or_create_rating(away, season)

                divisional = is_divisional_game(home, away)

                # Get prediction using current (pre-game) state
                prediction = self.elo_system.predict_game(
                    home, away, season, is_divisional=divisional
                )

                # Step 2: Record snapshot
                snapshots.append({
                    "game_id": game["game_id"],
                    "season": season,
                    "week": game["week"],
                    "home_team": home,
                    "away_team": away,
                    "home_elo_pre": home_rating.rating,
                    "away_elo_pre": away_rating.rating,
                    "home_elo_uncertainty": home_rating.uncertainty,
                    "away_elo_uncertainty": away_rating.uncertainty,
                    "elo_prob_home": prediction["home_win_prob"],
                    "hfa_used": prediction["hfa_used"],
                })

                # Step 3: THEN process game result (updates ratings)
                self.elo_system.update_ratings(
                    home_team=home,
                    away_team=away,
                    home_score=int(game["home_score"]),
                    away_score=int(game["away_score"]),
                    season=season,
                    game_date=game["kickoff_et"],
                    game_id=game["game_id"],
                    is_divisional=divisional,
                )
                games_processed += 1

            logger.info(
                f"Completed season {season} snapshots",
                games_processed=games_processed,
            )

        snapshots_df = pd.DataFrame(snapshots)
        logger.info(
            "Built all Elo snapshots",
            total_snapshots=len(snapshots_df),
            seasons=len(seasons_to_process),
        )

        return snapshots_df

    def build_all_ratings(self, start_season: int = 2002) -> pd.DataFrame:
        """Build Elo ratings from scratch with per-game snapshots.

        Uses build_elo_with_snapshots to process all seasons chronologically
        and produce per-game pre-game snapshots. Also processes seasons
        through the legacy path for backward compatibility of games_with_elo.

        Args:
            start_season: First season to include (default: 2002 for burn-in)

        Returns:
            DataFrame with all processed games (from legacy path)
        """
        # Build snapshots (the primary output)
        self._snapshots_df = self.build_elo_with_snapshots(start_season)

        # Also run the legacy processing path for games_with_elo compatibility
        # Reset Elo system for clean processing
        self.elo_system = EloRatingSystem()

        all_games = self.load_games_data()
        available_seasons = sorted(all_games["season"].unique())
        seasons_to_process = [s for s in available_seasons if s >= start_season]

        logger.info(
            "Building all Elo ratings from scratch",
            start_season=start_season,
            seasons_to_process=seasons_to_process,
        )

        return self.process_seasons_chronologically(seasons_to_process)

    def save_results(self, processed_games: pd.DataFrame) -> None:
        """Save Elo results to data storage.

        Saves:
        - elo_game_snapshots: Per-game pre-game Elo snapshots (primary artifact)
        - games_with_elo: Games with Elo rating updates (legacy)
        - elo_ratings_current: Current team ratings
        - elo_rating_history: Full rating history
        - elo_ratings.json: Elo system state

        Args:
            processed_games: DataFrame with games and rating updates
        """
        # Save per-game pre-game snapshots (primary artifact for EloFeatureBuilder)
        if hasattr(self, "_snapshots_df") and self._snapshots_df is not None:
            if len(self._snapshots_df) > 0:
                save_dataframe(
                    self._snapshots_df,
                    "elo_game_snapshots",
                    layer="silver",
                    append_mode=False,  # Always replace, full rebuild
                )
                logger.info(
                    "Saved elo_game_snapshots",
                    total_snapshots=len(self._snapshots_df),
                )

        # Save updated games with Elo ratings
        if len(processed_games) > 0:
            save_dataframe(
                processed_games,
                "games_with_elo",
                layer="silver",
                partition_cols=["season"]
                if len(processed_games["season"].unique()) > 1
                else None,
            )

        # Save current ratings
        current_ratings = self.elo_system.get_current_ratings()
        if len(current_ratings) > 0:
            save_dataframe(
                current_ratings,
                "elo_ratings_current",
                layer="silver",
                append_mode=False,  # Always replace current ratings, don't append
            )

        # Save rating history
        rating_history = self.elo_system.get_rating_history()
        if len(rating_history) > 0:
            save_dataframe(
                rating_history,
                "elo_rating_history",
                layer="silver",
                partition_cols=["season"],
            )

        # Save Elo system state to JSON
        self.elo_system.save_ratings()

        logger.info("Saved all Elo results to data storage")

    def validate_ratings(self) -> bool:
        """
        Validate Elo ratings for sanity checks.

        Returns:
            True if validation passes
        """
        current_ratings = self.elo_system.get_current_ratings()

        if len(current_ratings) == 0:
            logger.error("No ratings found - validation failed")
            return False

        # Check rating ranges
        min_rating = current_ratings["rating"].min()
        max_rating = current_ratings["rating"].max()

        if min_rating < 800 or max_rating > 2200:
            logger.warning(
                "Ratings outside expected range",
                min_rating=min_rating,
                max_rating=max_rating,
            )

        # Check for reasonable spread
        rating_std = current_ratings["rating"].std()
        if rating_std < 50 or rating_std > 300:
            logger.warning("Rating standard deviation unusual", rating_std=rating_std)

        # Check that all teams have played games
        no_games = current_ratings[current_ratings["games_played"] == 0]
        if len(no_games) > 0:
            logger.warning(
                "Teams with no games played", teams=no_games["team"].tolist()
            )

        logger.info(
            "Elo rating validation completed",
            teams=len(current_ratings),
            rating_range=(min_rating, max_rating),
            rating_std=rating_std,
        )

        return True


def main():
    """CLI entry point for Elo rating builder."""
    parser = argparse.ArgumentParser(description="Build NFL Elo ratings")
    parser.add_argument("--season", type=int, help="Process single season")
    parser.add_argument(
        "--seasons", nargs="+", type=int, help="Process multiple seasons"
    )
    parser.add_argument(
        "--all-seasons", action="store_true", help="Process all available seasons"
    )
    parser.add_argument(
        "--current", action="store_true", help="Update current season only"
    )
    parser.add_argument(
        "--start-season",
        type=int,
        default=2002,
        help="Starting season for all-seasons build (default: 2002)",
    )
    parser.add_argument(
        "--validate-only", action="store_true", help="Only validate existing ratings"
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Skip saving results (dry run)"
    )

    args = parser.parse_args()

    try:
        # Setup logging
        setup_logging()

        builder = EloBuilder()

        if args.validate_only:
            # Load existing ratings and validate
            builder.elo_system.load_ratings()
            success = builder.validate_ratings()
            if success:
                print("Elo rating validation passed")
            else:
                print("Elo rating validation failed")
                sys.exit(1)
            return

        # Determine what to process
        processed_games = pd.DataFrame()

        if args.current:
            processed_games = builder.update_current_season()
        elif args.all_seasons:
            processed_games = builder.build_all_ratings(args.start_season)
        elif args.seasons:
            processed_games = builder.process_seasons_chronologically(args.seasons)
        elif args.season:
            processed_games = builder.process_seasons_chronologically([args.season])
        else:
            # Default: update current season
            processed_games = builder.update_current_season()

        # Save results unless --no-save
        if not args.no_save and len(processed_games) > 0:
            builder.save_results(processed_games)

        # Validate results
        builder.validate_ratings()

        # Print summary
        current_ratings = builder.elo_system.get_current_ratings()
        if len(current_ratings) > 0:
            print("\nElo ratings built successfully!")
            print(f"Teams rated: {len(current_ratings)}")
            print(f"Games processed: {len(processed_games)}")
            print("\nTop 5 teams:")
            for _, team in current_ratings.head().iterrows():
                print(
                    f"  {team['team']}: {team['rating']:.1f} ({team['games_played']} games)"
                )

    except (ValueError, KeyError, TypeError, FileNotFoundError, OSError) as e:
        logger.error("Elo building failed", error=str(e))
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
