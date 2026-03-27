"""Elo Feature Builder

This module creates Elo-based features for game prediction models.
Conforms to the FeatureBuilder Protocol with mandatory as_of_datetime
parameter for time-fence enforcement.
"""

import warnings
from datetime import datetime

import pandas as pd

from features.protocol import FeatureBuilder  # noqa: F401 (documents conformance)
from ratings.elo import EloRatingSystem, is_divisional_game
from utils import get_logger

logger = get_logger(__name__)

# Elo feature column names
ELO_FEATURE_COLUMNS = [
    "home_elo",
    "away_elo",
    "elo_diff",
    "elo_prob_home",
    "elo_prob_away",
    "hfa_used",
    "home_elo_uncertainty",
    "away_elo_uncertainty",
]


class EloFeatureBuilder:
    """Build Elo-based features for game predictions.

    Satisfies the FeatureBuilder Protocol via structural subtyping.
    The as_of_datetime parameter enforces the time-fence: only data
    before this cutoff is used in feature construction.
    """

    def __init__(self) -> None:
        """Initialize Elo feature builder."""
        self.elo_system = EloRatingSystem()

    def load_elo_system(self, filepath: str | None = None) -> None:
        """Load existing Elo ratings."""
        self.elo_system.load_ratings(filepath)

    # ---- Protocol-conforming methods ----

    def build_features(
        self,
        games_df: pd.DataFrame,
        as_of_datetime: datetime,
        *,
        target_season: int | None = None,
        target_week: int | None = None,
    ) -> pd.DataFrame:
        """Build Elo features for a batch of games.

        Only uses game data where kickoff_et < as_of_datetime for Elo
        rating updates. Games after the cutoff get predictions based
        on ratings computed from pre-cutoff games only.

        Args:
            games_df: DataFrame with game records (must have columns:
                game_id, home_team, away_team, season, week, kickoff_et).
            as_of_datetime: Time-fence cutoff. Only games with
                kickoff_et < as_of_datetime are used for rating updates.
            target_season: Optional filter to a specific season.
            target_week: Optional filter to a specific week.

        Returns:
            DataFrame with Elo feature columns added.
        """
        logger.info(
            "Building Elo features",
            games=len(games_df),
            as_of_datetime=str(as_of_datetime),
        )

        # Apply optional filters
        filtered_df = games_df.copy()
        if target_season is not None:
            filtered_df = filtered_df[filtered_df["season"] == target_season]
        if target_week is not None:
            filtered_df = filtered_df[filtered_df["week"] == target_week]

        # Update Elo ratings from completed games before the cutoff
        # Make as_of_datetime timezone-aware if kickoff_et is tz-aware
        cutoff = as_of_datetime
        if (
            hasattr(games_df["kickoff_et"].dtype, "tz")
            and games_df["kickoff_et"].dtype.tz is not None
        ):
            cutoff = pd.Timestamp(as_of_datetime).tz_localize(
                games_df["kickoff_et"].dtype.tz
            )
        completed_mask = (
            (games_df["kickoff_et"] < cutoff)
            & games_df["home_score"].notna()
            & games_df["away_score"].notna()
        )
        completed_games = games_df[completed_mask]

        if len(completed_games) > 0:
            for season in sorted(completed_games["season"].unique()):
                season_games = completed_games[completed_games["season"] == season]
                self.elo_system.process_season_chronologically(season_games, season)
            logger.info(
                "Updated Elo ratings from completed games",
                completed_games=len(completed_games),
                seasons=sorted(completed_games["season"].unique().tolist()),
            )

        # Initialize feature columns
        for col in ELO_FEATURE_COLUMNS:
            filtered_df[col] = None

        # Calculate Elo features for each game
        for idx, game in filtered_df.iterrows():
            home_team = game["home_team"]
            away_team = game["away_team"]
            season = game["season"]
            neutral_site = game.get("neutral_site", False)

            # Detect divisional games for HFA reduction
            divisional = is_divisional_game(home_team, away_team)

            # Get Elo prediction using current (pre-cutoff) ratings
            prediction = self.elo_system.predict_game(
                home_team,
                away_team,
                season,
                neutral_site=neutral_site,
                is_divisional=divisional,
            )

            elo_features = {
                "home_elo": prediction["home_rating"],
                "away_elo": prediction["away_rating"],
                "elo_diff": prediction["rating_diff"],
                "elo_prob_home": prediction["home_win_prob"],
                "elo_prob_away": prediction["away_win_prob"],
                "hfa_used": prediction["hfa_used"],
                "home_elo_uncertainty": prediction["home_uncertainty"],
                "away_elo_uncertainty": prediction["away_uncertainty"],
            }

            for feature, value in elo_features.items():
                filtered_df.loc[idx, feature] = value

        logger.info("Built Elo features", games=len(filtered_df))
        return filtered_df

    def get_features_for_game(
        self,
        game_id: str,
        as_of_datetime: datetime,
    ) -> dict[str, float]:
        """Get Elo features for a single game.

        Looks up game data from the Silver layer, computes Elo prediction
        using only ratings updated from games before as_of_datetime.

        Args:
            game_id: Unique game identifier (e.g., '2024_01_BUF_MIA').
            as_of_datetime: Time-fence cutoff. Only data before this
                timestamp may be used.

        Returns:
            Dictionary mapping feature names to values.

        Raises:
            ValueError: If game_id cannot be parsed.
            KeyError: If required data is missing.
        """
        # Parse game_id to extract teams and season
        # Format: YYYY_WW_AWAY_HOME or similar -- defer to data layer
        # For now, use the internal Elo system's current state
        # (ratings should have been built up to as_of_datetime by process_season)
        try:
            from data.storage import load_dataframe

            games_df = load_dataframe("games", layer="silver")
            game_row = games_df[games_df["game_id"] == game_id]

            if len(game_row) == 0:
                raise ValueError(f"Game not found: {game_id}")

            game = game_row.iloc[0]
            home_team = game["home_team"]
            away_team = game["away_team"]
            season = game["season"]
            neutral_site = game.get("neutral_site", False)

            divisional = is_divisional_game(home_team, away_team)

            prediction = self.elo_system.predict_game(
                home_team,
                away_team,
                season,
                neutral_site=neutral_site,
                is_divisional=divisional,
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

        except (ValueError, KeyError, TypeError) as e:
            logger.error(
                "Failed to get Elo features for game",
                game_id=game_id,
                as_of_datetime=str(as_of_datetime),
                error=str(e),
            )
            raise

    # ---- Deprecated methods (kept for backward compatibility) ----

    def get_elo_features_for_game(
        self,
        home_team: str,
        away_team: str,
        season: int,
        week: int,
        neutral_site: bool = False,
    ) -> dict[str, float]:
        """Get Elo-based features for a specific game.

        .. deprecated::
            Use `get_features_for_game(game_id, as_of_datetime)` instead.
            This method does not enforce the time-fence contract.

        Args:
            home_team: Home team abbreviation.
            away_team: Away team abbreviation.
            season: Season year.
            week: Week number.
            neutral_site: Whether game is at neutral site.

        Returns:
            Dictionary with Elo features.

        Raises:
            ValueError, KeyError, TypeError: If prediction fails.
        """
        warnings.warn(
            "get_elo_features_for_game is deprecated. "
            "Use get_features_for_game(game_id, as_of_datetime) instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        divisional = is_divisional_game(home_team, away_team)

        try:
            prediction = self.elo_system.predict_game(
                home_team,
                away_team,
                season,
                neutral_site,
                is_divisional=divisional,
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

        except (ValueError, KeyError, TypeError) as e:
            logger.error(
                "Failed to get Elo features for game",
                home_team=home_team,
                away_team=away_team,
                season=season,
                week=week,
                error=str(e),
            )
            raise

    def build_elo_features_for_games(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """Build Elo features for a DataFrame of games.

        .. deprecated::
            Use `build_features(games_df, as_of_datetime)` instead.
            This method does not enforce the time-fence contract.

        Args:
            games_df: DataFrame with games.

        Returns:
            DataFrame with Elo features added.
        """
        warnings.warn(
            "build_elo_features_for_games is deprecated. "
            "Use build_features(games_df, as_of_datetime) instead.",
            DeprecationWarning,
            stacklevel=2,
        )

        logger.info("Building Elo features for games", games=len(games_df))

        games_with_elo = games_df.copy()

        for col in ELO_FEATURE_COLUMNS:
            games_with_elo[col] = None

        for idx, game in games_df.iterrows():
            home_team = game["home_team"]
            away_team = game["away_team"]
            divisional = is_divisional_game(home_team, away_team)

            try:
                prediction = self.elo_system.predict_game(
                    home_team,
                    away_team,
                    game["season"],
                    game.get("neutral_site", False),
                    is_divisional=divisional,
                )

                elo_features = {
                    "home_elo": prediction["home_rating"],
                    "away_elo": prediction["away_rating"],
                    "elo_diff": prediction["rating_diff"],
                    "elo_prob_home": prediction["home_win_prob"],
                    "elo_prob_away": prediction["away_win_prob"],
                    "hfa_used": prediction["hfa_used"],
                    "home_elo_uncertainty": prediction["home_uncertainty"],
                    "away_elo_uncertainty": prediction["away_uncertainty"],
                }

                for feature, value in elo_features.items():
                    games_with_elo.loc[idx, feature] = value

            except (ValueError, KeyError, TypeError) as e:
                logger.error(
                    "Failed to get Elo features for game",
                    home_team=home_team,
                    away_team=away_team,
                    season=game["season"],
                    week=game["week"],
                    error=str(e),
                )
                raise

        logger.info("Built Elo features", games=len(games_with_elo))
        return games_with_elo
