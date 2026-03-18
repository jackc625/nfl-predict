"""
Elo Feature Builder

This module creates Elo-based features for game prediction models.
"""

import pandas as pd

from ratings.elo import EloRatingSystem
from utils import get_logger

logger = get_logger(__name__)


class EloFeatureBuilder:
    """Build Elo-based features for game predictions."""

    def __init__(self):
        """Initialize Elo feature builder."""
        self.elo_system = EloRatingSystem()

    def load_elo_system(self, filepath: str | None = None) -> None:
        """Load existing Elo ratings."""
        self.elo_system.load_ratings(filepath)

    def get_elo_features_for_game(
        self,
        home_team: str,
        away_team: str,
        season: int,
        week: int,
        neutral_site: bool = False,
    ) -> dict[str, float]:
        """
        Get Elo-based features for a specific game.

        Args:
            home_team: Home team abbreviation
            away_team: Away team abbreviation
            season: Season year
            week: Week number
            neutral_site: Whether game is at neutral site

        Returns:
            Dictionary with Elo features
        """
        try:
            # Get Elo prediction
            prediction = self.elo_system.predict_game(
                home_team, away_team, season, neutral_site
            )

            return {
                "home_elo": prediction["home_rating"],
                "away_elo": prediction["away_rating"],
                "elo_diff": prediction["rating_diff"],
                "elo_prob_home": prediction["home_win_prob"],
                "elo_prob_away": prediction["away_win_prob"],
                "hfa_used": prediction["hfa_used"],
                "home_elo_uncertainty": prediction["home_uncertainty"],
                "away_elo_uncertainty": prediction["away_uncertainty"],
            }

        except Exception as e:
            logger.error(
                "Failed to get Elo features for game",
                home_team=home_team,
                away_team=away_team,
                season=season,
                week=week,
                error=str(e),
            )
            return {}

    def build_elo_features_for_games(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """
        Build Elo features for a DataFrame of games.

        Args:
            games_df: DataFrame with games

        Returns:
            DataFrame with Elo features added
        """
        logger.info("Building Elo features for games", games=len(games_df))

        games_with_elo = games_df.copy()

        # Initialize feature columns
        elo_columns = [
            "home_elo",
            "away_elo",
            "elo_diff",
            "elo_prob_home",
            "elo_prob_away",
            "hfa_used",
            "home_elo_uncertainty",
            "away_elo_uncertainty",
        ]

        for col in elo_columns:
            games_with_elo[col] = None

        # Calculate Elo features for each game
        for idx, game in games_df.iterrows():
            elo_features = self.get_elo_features_for_game(
                game["home_team"],
                game["away_team"],
                game["season"],
                game["week"],
                game.get("neutral_site", False),
            )

            # Add features to DataFrame
            for feature, value in elo_features.items():
                games_with_elo.loc[idx, feature] = value

        logger.info("Built Elo features", games=len(games_with_elo))
        return games_with_elo
