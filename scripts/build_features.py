#!/usr/bin/env python3
"""
Unified Feature Building Pipeline

This script combines all feature sources into complete feature matrices:
- Team form metrics (rolling 4-week EPA, success rates)
- Elo ratings (with uncertainty and home field advantage)
- Contextual features (travel, weather, venue, rest)
- Weather features (wind, temperature, precipitation - outdoor only)
- Market anchor features (opening/snapshot lines, devigged probabilities)

Feature processing includes:
- Missing data imputation and outlier handling (winsorization)
- Z-score normalization within seasons
- Separate feature matrices for WP, ATS, and O/U targets
- Storage in gold layer for model consumption

Usage:
    python scripts/build_features.py --season 2024 --week 1
    python scripts/build_features.py --season 2024  # All weeks in season
    python scripts/build_features.py  # All available data
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.storage import load_dataframe, save_dataframe
from features.contextual import ContextualFeaturesCalculator
from features.elo_features import EloFeatureBuilder
from features.market_anchors import MarketAnchorFeaturesCalculator
from features.team_form import TeamFormCalculator
from features.weather import WeatherFeaturesCalculator
from utils import get_logger

logger = get_logger(__name__)


class FeatureMatrixBuilder:
    """
    Build unified feature matrices from all feature sources.

    Combines team form, Elo, contextual, weather, and market features
    into complete feature matrices ready for model training.
    """

    def __init__(self):
        """Initialize feature matrix builder."""
        self.logger = get_logger(__name__)

        # Feature calculators
        self.team_form_calc = TeamFormCalculator()
        self.elo_calc = EloFeatureBuilder()
        self.contextual_calc = ContextualFeaturesCalculator()
        self.weather_calc = WeatherFeaturesCalculator()
        self.market_calc = MarketAnchorFeaturesCalculator()

        # Feature processing parameters
        self.outlier_percentiles = (1, 99)  # Winsorization bounds
        self.min_games_for_stats = 10  # Minimum games for normalization

    def load_all_feature_sources(
        self, target_season: int | None = None, target_week: int | None = None
    ) -> dict[str, pd.DataFrame]:
        """
        Load all feature sources from silver layer.

        Args:
            target_season: Specific season to load
            target_week: Specific week to load

        Returns:
            Dictionary with all feature DataFrames
        """
        logger.info(
            "Loading all feature sources",
            target_season=target_season,
            target_week=target_week,
        )

        feature_sources = {}

        try:
            # Core game data
            games_df = load_dataframe("games", layer="silver")
            if target_season and target_week:
                games_df = games_df[
                    (games_df["season"] == target_season)
                    & (games_df["week"] == target_week)
                ]
            feature_sources["games"] = games_df
            logger.info("Loaded games data", records=len(games_df))

            # Team form features
            try:
                team_form_df = load_dataframe("team_form_features", layer="silver")
                if target_season and target_week:
                    team_form_df = team_form_df[
                        (team_form_df["target_season"] == target_season)
                        & (team_form_df["target_week"] == target_week)
                    ]
                feature_sources["team_form"] = team_form_df
                logger.info("Loaded team form features", records=len(team_form_df))
            except Exception as e:
                logger.warning("Failed to load team form features", error=str(e))
                feature_sources["team_form"] = pd.DataFrame()

            # Elo features
            try:
                elo_df = load_dataframe("elo_ratings", layer="silver")
                if target_season and target_week:
                    elo_df = elo_df[
                        (elo_df["season"] == target_season)
                        & (elo_df["week"] == target_week)
                    ]
                feature_sources["elo"] = elo_df
                logger.info("Loaded Elo features", records=len(elo_df))
            except Exception as e:
                logger.warning("Failed to load Elo features", error=str(e))
                feature_sources["elo"] = pd.DataFrame()

            # Contextual features
            try:
                contextual_df = load_dataframe("contextual_features", layer="silver")
                if target_season and target_week:
                    contextual_df = contextual_df[
                        (contextual_df["season"] == target_season)
                        & (contextual_df["week"] == target_week)
                    ]
                feature_sources["contextual"] = contextual_df
                logger.info("Loaded contextual features", records=len(contextual_df))
            except Exception as e:
                logger.warning("Failed to load contextual features", error=str(e))
                feature_sources["contextual"] = pd.DataFrame()

            # Weather features
            try:
                weather_df = load_dataframe("weather_features", layer="silver")
                if target_season and target_week:
                    weather_df = weather_df[
                        (weather_df["season"] == target_season)
                        & (weather_df["week"] == target_week)
                    ]
                feature_sources["weather"] = weather_df
                logger.info("Loaded weather features", records=len(weather_df))
            except Exception as e:
                logger.warning("Failed to load weather features", error=str(e))
                feature_sources["weather"] = pd.DataFrame()

            # Market anchor features
            try:
                market_df = load_dataframe("market_anchor_features", layer="silver")
                if target_season and target_week:
                    market_df = market_df[
                        (market_df["season"] == target_season)
                        & (market_df["week"] == target_week)
                    ]
                feature_sources["market"] = market_df
                logger.info("Loaded market anchor features", records=len(market_df))
            except Exception as e:
                logger.warning("Failed to load market anchor features", error=str(e))
                feature_sources["market"] = pd.DataFrame()

            return feature_sources

        except Exception as e:
            logger.error("Failed to load feature sources", error=str(e))
            raise

    def combine_features(
        self, feature_sources: dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Combine all feature sources into unified feature matrix.

        Args:
            feature_sources: Dictionary with all feature DataFrames

        Returns:
            Combined feature matrix
        """
        logger.info("Combining all feature sources")

        # Start with games as base
        games_df = feature_sources["games"].copy()

        if len(games_df) == 0:
            logger.error("No games data available")
            return pd.DataFrame()

        # Initialize combined features with game identifiers
        combined_features = games_df[
            ["game_id", "season", "week", "home_team", "away_team"]
        ].copy()

        # Add game results if available (for target creation)
        if "home_score" in games_df.columns:
            combined_features["home_score"] = games_df["home_score"]
        if "away_score" in games_df.columns:
            combined_features["away_score"] = games_df["away_score"]

        # Merge each feature source
        feature_counts = {}

        # Team form features (need to handle home/away separately)
        team_form_df = feature_sources.get("team_form", pd.DataFrame())
        if len(team_form_df) > 0:
            home_form = self._get_team_features(
                team_form_df, combined_features, "home_team", "home"
            )
            away_form = self._get_team_features(
                team_form_df, combined_features, "away_team", "away"
            )
            combined_features = combined_features.merge(
                home_form, on="game_id", how="left"
            )
            combined_features = combined_features.merge(
                away_form, on="game_id", how="left"
            )
            feature_counts["team_form"] = len(
                [col for col in combined_features.columns if "form_" in col]
            )

        # Elo features (also need home/away)
        elo_df = feature_sources.get("elo", pd.DataFrame())
        if len(elo_df) > 0:
            home_elo = self._get_team_elo_features(
                elo_df, combined_features, "home_team", "home"
            )
            away_elo = self._get_team_elo_features(
                elo_df, combined_features, "away_team", "away"
            )
            combined_features = combined_features.merge(
                home_elo, on="game_id", how="left"
            )
            combined_features = combined_features.merge(
                away_elo, on="game_id", how="left"
            )
            feature_counts["elo"] = len(
                [col for col in combined_features.columns if "elo_" in col]
            )

        # Contextual features (game-level)
        contextual_df = feature_sources.get("contextual", pd.DataFrame())
        if len(contextual_df) > 0:
            merge_cols = ["game_id"]
            contextual_features = contextual_df.drop(
                columns=["season", "week", "home_team", "away_team"], errors="ignore"
            )
            combined_features = combined_features.merge(
                contextual_features, on=merge_cols, how="left"
            )
            feature_counts["contextual"] = len(
                [col for col in contextual_features.columns if col != "game_id"]
            )

        # Weather features (game-level)
        weather_df = feature_sources.get("weather", pd.DataFrame())
        if len(weather_df) > 0:
            merge_cols = ["game_id"]
            weather_features = weather_df.drop(
                columns=["season", "week"], errors="ignore"
            )
            combined_features = combined_features.merge(
                weather_features, on=merge_cols, how="left"
            )
            feature_counts["weather"] = len(
                [col for col in weather_features.columns if col != "game_id"]
            )

        # Market anchor features (game-level)
        market_df = feature_sources.get("market", pd.DataFrame())
        if len(market_df) > 0:
            merge_cols = ["game_id"]
            market_features = market_df.drop(
                columns=["season", "week"], errors="ignore"
            )
            combined_features = combined_features.merge(
                market_features, on=merge_cols, how="left"
            )
            feature_counts["market"] = len(
                [col for col in market_features.columns if col != "game_id"]
            )

        # Add feature timestamp
        combined_features["feature_timestamp"] = datetime.now()

        logger.info(
            "Combined all features",
            total_games=len(combined_features),
            total_features=len(combined_features.columns),
            feature_breakdown=feature_counts,
        )

        return combined_features

    def _get_team_features(
        self,
        team_form_df: pd.DataFrame,
        games_df: pd.DataFrame,
        team_col: str,
        prefix: str,
    ) -> pd.DataFrame:
        """Get team form features for home or away team."""
        team_features = []

        for _, game in games_df.iterrows():
            game_id = game["game_id"]
            team = game[team_col]
            season = game["season"]
            week = game["week"]

            # Get team's offensive features
            team_off = team_form_df[
                (team_form_df["team"] == team)
                & (team_form_df["target_season"] == season)
                & (team_form_df["target_week"] == week)
                & (team_form_df["side"] == "offense")
            ]

            # Get team's defensive features
            team_def = team_form_df[
                (team_form_df["team"] == team)
                & (team_form_df["target_season"] == season)
                & (team_form_df["target_week"] == week)
                & (team_form_df["side"] == "defense")
            ]

            game_features = {"game_id": game_id}

            # Add offensive features
            if len(team_off) > 0:
                off_row = team_off.iloc[0]
                for col in off_row.index:
                    if col.startswith("rolling_"):
                        feature_name = f"{prefix}_off_{col}"
                        game_features[feature_name] = off_row[col]

            # Add defensive features
            if len(team_def) > 0:
                def_row = team_def.iloc[0]
                for col in def_row.index:
                    if col.startswith("rolling_"):
                        feature_name = f"{prefix}_def_{col}"
                        game_features[feature_name] = def_row[col]

            team_features.append(game_features)

        return pd.DataFrame(team_features)

    def _get_team_elo_features(
        self, elo_df: pd.DataFrame, games_df: pd.DataFrame, team_col: str, prefix: str
    ) -> pd.DataFrame:
        """Get Elo features for home or away team."""
        team_features = []

        for _, game in games_df.iterrows():
            game_id = game["game_id"]
            team = game[team_col]
            season = game["season"]
            week = game["week"]

            # Get team's Elo rating
            team_elo = elo_df[
                (elo_df["team"] == team)
                & (elo_df["season"] == season)
                & (elo_df["week"] == week)
            ]

            game_features = {"game_id": game_id}

            if len(team_elo) > 0:
                elo_row = team_elo.iloc[0]
                for col in elo_row.index:
                    if col not in ["team", "season", "week"]:
                        feature_name = f"{prefix}_{col}"
                        game_features[feature_name] = elo_row[col]

            team_features.append(game_features)

        return pd.DataFrame(team_features)

    def handle_missing_data_and_outliers(
        self, features_df: pd.DataFrame, target_columns: list[str] | None = None
    ) -> pd.DataFrame:
        """
        Handle missing data and outliers with winsorization.

        Args:
            features_df: Feature matrix
            target_columns: Columns to exclude from processing

        Returns:
            Processed feature matrix
        """
        logger.info(
            "Handling missing data and outliers", features=len(features_df.columns)
        )

        if target_columns is None:
            target_columns = [
                "game_id",
                "season",
                "week",
                "home_team",
                "away_team",
                "home_score",
                "away_score",
                "feature_timestamp",
            ]

        processed_df = features_df.copy()

        # Get numeric feature columns
        feature_cols = [
            col for col in processed_df.columns if col not in target_columns
        ]
        numeric_cols = (
            processed_df[feature_cols]
            .select_dtypes(include=[np.number])
            .columns.tolist()
        )

        missing_stats = {}
        outlier_stats = {}

        for col in numeric_cols:
            original_missing = processed_df[col].isna().sum()

            # Handle missing data
            if original_missing > 0:
                # For team-based features, use team's season average
                if any(prefix in col for prefix in ["home_", "away_"]):
                    processed_df[col] = self._impute_team_features(processed_df, col)
                else:
                    # For game-level features, use overall median
                    median_value = processed_df[col].median()
                    processed_df[col] = processed_df[col].fillna(median_value)

                missing_stats[col] = original_missing

            # Handle outliers with winsorization
            if processed_df[col].notna().sum() > 10:  # Need minimum data points
                lower_bound = processed_df[col].quantile(
                    self.outlier_percentiles[0] / 100
                )
                upper_bound = processed_df[col].quantile(
                    self.outlier_percentiles[1] / 100
                )

                outliers_count = (
                    (processed_df[col] < lower_bound)
                    | (processed_df[col] > upper_bound)
                ).sum()

                if outliers_count > 0:
                    processed_df[col] = processed_df[col].clip(
                        lower=lower_bound, upper=upper_bound
                    )
                    outlier_stats[col] = outliers_count

        logger.info(
            "Completed missing data and outlier handling",
            missing_imputed=len(missing_stats),
            outliers_winsorized=len(outlier_stats),
            total_features_processed=len(numeric_cols),
        )

        return processed_df

    def _impute_team_features(self, df: pd.DataFrame, col: str) -> pd.Series:
        """Impute missing team features using team's season average."""
        # Extract team and prefix from column name
        if col.startswith("home_"):
            team_col = "home_team"
            col.replace("home_", "")
        elif col.startswith("away_"):
            team_col = "away_team"
            col.replace("away_", "")
        else:
            # Not a team feature, use median
            return df[col].fillna(df[col].median())

        result = df[col].copy()

        # For each team with missing data, use their season average
        for season in df["season"].unique():
            season_data = df[df["season"] == season]

            for team in season_data[team_col].unique():
                team_mask = (df["season"] == season) & (df[team_col] == team)
                team_values = df.loc[team_mask, col]

                if team_values.isna().any():
                    team_mean = team_values.mean()
                    if pd.isna(team_mean):
                        # Use season average if team has no data
                        team_mean = season_data[col].mean()
                    if pd.isna(team_mean):
                        # Use overall median as last resort
                        team_mean = df[col].median()

                    result.loc[team_mask & result.isna()] = team_mean

        return result

    def normalize_features_within_seasons(
        self, features_df: pd.DataFrame, target_columns: list[str] | None = None
    ) -> pd.DataFrame:
        """
        Apply Z-score normalization within seasons.

        Args:
            features_df: Feature matrix
            target_columns: Columns to exclude from normalization

        Returns:
            Normalized feature matrix
        """
        logger.info("Normalizing features within seasons")

        if target_columns is None:
            target_columns = [
                "game_id",
                "season",
                "week",
                "home_team",
                "away_team",
                "home_score",
                "away_score",
                "feature_timestamp",
            ]

        normalized_df = features_df.copy()

        # Get numeric feature columns to normalize
        feature_cols = [
            col for col in normalized_df.columns if col not in target_columns
        ]
        numeric_cols = (
            normalized_df[feature_cols]
            .select_dtypes(include=[np.number])
            .columns.tolist()
        )

        normalization_stats = {}

        # Normalize within each season
        for season in normalized_df["season"].unique():
            season_mask = normalized_df["season"] == season
            season_data = normalized_df.loc[season_mask]

            if len(season_data) < self.min_games_for_stats:
                logger.warning(
                    "Insufficient data for normalization",
                    season=season,
                    games=len(season_data),
                )
                continue

            season_normalized_cols = 0

            for col in numeric_cols:
                col_data = season_data[col]

                if col_data.notna().sum() >= self.min_games_for_stats:
                    mean_val = col_data.mean()
                    std_val = col_data.std()

                    if std_val > 0:  # Avoid division by zero
                        normalized_df.loc[season_mask, col] = (
                            col_data - mean_val
                        ) / std_val
                        season_normalized_cols += 1

            normalization_stats[season] = season_normalized_cols

        logger.info(
            "Completed within-season normalization",
            seasons_processed=len(normalization_stats),
            avg_features_per_season=np.mean(list(normalization_stats.values())),
        )

        return normalized_df

    def create_target_variables(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create target variables for WP, ATS, and O/U prediction.

        Args:
            features_df: Feature matrix with game scores

        Returns:
            Feature matrix with target variables added
        """
        logger.info("Creating target variables")

        target_df = features_df.copy()

        # Only create targets if we have scores
        if (
            "home_score" not in target_df.columns
            or "away_score" not in target_df.columns
        ):
            logger.warning("No score data available for target creation")
            return target_df

        # Convert score columns to numeric (they may be stored as strings)
        target_df["home_score"] = pd.to_numeric(
            target_df["home_score"], errors="coerce"
        )
        target_df["away_score"] = pd.to_numeric(
            target_df["away_score"], errors="coerce"
        )

        # Remove games with missing scores
        score_mask = target_df["home_score"].notna() & target_df["away_score"].notna()
        target_df = target_df[score_mask].copy()

        # Win Probability target (1 = home win, 0 = away win)
        target_df["target_wp"] = (
            target_df["home_score"] > target_df["away_score"]
        ).astype(int)

        # Handle ties (rare in NFL)
        ties = target_df["home_score"] == target_df["away_score"]
        if ties.sum() > 0:
            logger.info("Found tied games", count=ties.sum())
            # For WP, treat ties as 0.5 (convert to regression target)
            target_df.loc[ties, "target_wp"] = 0.5

        # Point differential (for ATS calculation if spreads available)
        target_df["point_differential"] = (
            target_df["home_score"] - target_df["away_score"]
        )

        # Total points (for O/U calculation if totals available)
        target_df["total_points"] = target_df["home_score"] + target_df["away_score"]

        # ATS target (requires market data)
        if "snapshot_spread" in target_df.columns:
            # ATS = actual margin - spread (positive = home team covered)
            target_df["target_ats"] = (
                target_df["point_differential"] - target_df["snapshot_spread"]
            )
            target_df["home_covered_spread"] = (target_df["target_ats"] > 0).astype(int)

        # O/U target (requires market data)
        if "snapshot_total" in target_df.columns:
            # O/U = actual total - market total (positive = over)
            target_df["target_ou"] = (
                target_df["total_points"] - target_df["snapshot_total"]
            )
            target_df["game_went_over"] = (target_df["target_ou"] > 0).astype(int)

        logger.info(
            "Created target variables",
            wp_targets=target_df["target_wp"].notna().sum(),
            ats_targets=target_df.get("target_ats", pd.Series()).notna().sum(),
            ou_targets=target_df.get("target_ou", pd.Series()).notna().sum(),
        )

        return target_df

    def generate_feature_matrices(
        self, target_season: int | None = None, target_week: int | None = None
    ) -> dict[str, pd.DataFrame]:
        """
        Generate complete feature matrices for all prediction targets.

        Args:
            target_season: Specific season to process
            target_week: Specific week to process

        Returns:
            Dictionary with feature matrices for each target
        """
        logger.info(
            "Generating feature matrices",
            target_season=target_season,
            target_week=target_week,
        )

        try:
            # Load all feature sources
            feature_sources = self.load_all_feature_sources(target_season, target_week)

            # Combine features
            combined_features = self.combine_features(feature_sources)

            if len(combined_features) == 0:
                logger.error("No features to process")
                return {}

            # Handle missing data and outliers
            processed_features = self.handle_missing_data_and_outliers(
                combined_features
            )

            # Normalize features within seasons
            normalized_features = self.normalize_features_within_seasons(
                processed_features
            )

            # Create target variables
            final_features = self.create_target_variables(normalized_features)

            # Generate separate matrices for each prediction target
            feature_matrices = {}

            # Base feature columns (exclude identifiers and targets)
            exclude_cols = [
                "game_id",
                "season",
                "week",
                "home_team",
                "away_team",
                "home_score",
                "away_score",
                "feature_timestamp",
                "target_wp",
                "target_ats",
                "target_ou",
                "point_differential",
                "total_points",
                "home_covered_spread",
                "game_went_over",
            ]

            feature_cols = [
                col for col in final_features.columns if col not in exclude_cols
            ]

            # Win Probability matrix
            wp_matrix = final_features[
                [
                    "game_id",
                    "season",
                    "week",
                    "feature_timestamp",
                    "target_wp",
                    *feature_cols,
                ]
            ].copy()
            feature_matrices["wp"] = wp_matrix

            # ATS matrix (only include games with spread data)
            if "target_ats" in final_features.columns:
                ats_games = final_features["target_ats"].notna()
                ats_matrix = final_features.loc[
                    ats_games,
                    [
                        "game_id",
                        "season",
                        "week",
                        "feature_timestamp",
                        "target_ats",
                        *feature_cols,
                    ],
                ].copy()
                feature_matrices["ats"] = ats_matrix

            # O/U matrix (only include games with total data)
            if "target_ou" in final_features.columns:
                ou_games = final_features["target_ou"].notna()
                ou_matrix = final_features.loc[
                    ou_games,
                    [
                        "game_id",
                        "season",
                        "week",
                        "feature_timestamp",
                        "target_ou",
                        *feature_cols,
                    ],
                ].copy()
                feature_matrices["ou"] = ou_matrix

            logger.info(
                "Generated feature matrices",
                wp_games=len(feature_matrices.get("wp", [])),
                ats_games=len(feature_matrices.get("ats", [])),
                ou_games=len(feature_matrices.get("ou", [])),
                total_features=len(feature_cols),
            )

            return feature_matrices

        except Exception as e:
            logger.error("Failed to generate feature matrices", error=str(e))
            raise

    def save_feature_matrices(
        self,
        feature_matrices: dict[str, pd.DataFrame],
        target_season: int | None = None,
    ) -> None:
        """
        Save feature matrices to gold layer.

        Args:
            feature_matrices: Dictionary with feature matrices
            target_season: Season for partitioning
        """
        logger.info("Saving feature matrices to gold layer")

        for target, matrix_df in feature_matrices.items():
            if len(matrix_df) == 0:
                logger.warning("Empty feature matrix", target=target)
                continue

            # Determine partition columns
            partition_cols = ["season"] if target_season else None

            table_name = f"features_{target}"

            save_dataframe(
                matrix_df,
                table_name=table_name,
                layer="gold",
                partition_cols=partition_cols,
            )

            logger.info(
                "Saved feature matrix",
                target=target,
                table_name=table_name,
                records=len(matrix_df),
                features=len(matrix_df.columns) - 5,
            )  # Exclude metadata columns


def main():
    """Build unified feature matrices."""
    parser = argparse.ArgumentParser(description="Build unified feature matrices")
    parser.add_argument("--season", type=int, help="Target season (e.g., 2024)")
    parser.add_argument("--week", type=int, help="Target week (1-18)")
    parser.add_argument(
        "--save",
        action="store_true",
        default=True,
        help="Save feature matrices to gold layer",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        default=True,
        help="Validate feature matrices after building",
    )

    args = parser.parse_args()

    logger.info("Building unified feature matrices", season=args.season, week=args.week)

    try:
        # Initialize feature matrix builder
        builder = FeatureMatrixBuilder()

        # Generate feature matrices
        feature_matrices = builder.generate_feature_matrices(
            target_season=args.season, target_week=args.week
        )

        if not feature_matrices:
            logger.warning("No feature matrices generated")
            return

        # Display summary
        print("\nFeature matrices summary:")
        print("=" * 60)

        total_features = 0
        for target, matrix_df in feature_matrices.items():
            feature_count = len(matrix_df.columns) - 5  # Exclude metadata
            total_features = feature_count  # They should all have same feature count

            print(f"{target.upper()} Matrix:")
            print(f"  Games: {len(matrix_df)}")
            print(f"  Features: {feature_count}")

            if len(matrix_df) > 0:
                # Show target distribution
                target_col = f"target_{target}"
                if target_col in matrix_df.columns:
                    if target == "wp":
                        home_wins = (matrix_df[target_col] == 1).sum()
                        away_wins = (matrix_df[target_col] == 0).sum()
                        ties = (matrix_df[target_col] == 0.5).sum()
                        print(
                            f"  Home wins: {home_wins}, Away wins: {away_wins}, Ties: {ties}"
                        )
                    else:
                        target_mean = matrix_df[target_col].mean()
                        target_std = matrix_df[target_col].std()
                        print(
                            f"  Target mean: {target_mean:.3f}, std: {target_std:.3f}"
                        )

        print(f"\nTotal unique features across all matrices: {total_features}")

        # Sample feature types
        if feature_matrices:
            sample_matrix = next(iter(feature_matrices.values()))
            feature_cols = [
                col
                for col in sample_matrix.columns
                if col
                not in [
                    "game_id",
                    "season",
                    "week",
                    "feature_timestamp",
                    "target_wp",
                    "target_ats",
                    "target_ou",
                ]
            ]

            feature_types = {}
            for col in feature_cols:
                if "elo_" in col:
                    feature_types.setdefault("Elo", []).append(col)
                elif any(
                    x in col
                    for x in ["home_off_", "away_off_", "home_def_", "away_def_"]
                ):
                    feature_types.setdefault("Team Form", []).append(col)
                elif any(
                    x in col
                    for x in ["travel_", "rest_", "venue_", "thursday_", "short_"]
                ):
                    feature_types.setdefault("Contextual", []).append(col)
                elif any(x in col for x in ["wind_", "temp_", "precip_", "weather_"]):
                    feature_types.setdefault("Weather", []).append(col)
                elif any(
                    x in col
                    for x in ["ml_", "spread_", "total_", "opening_", "snapshot_"]
                ):
                    feature_types.setdefault("Market", []).append(col)
                else:
                    feature_types.setdefault("Other", []).append(col)

            print("\nFeature breakdown by type:")
            for ftype, fcols in feature_types.items():
                print(f"  {ftype}: {len(fcols)} features")

        # Save feature matrices if requested
        if args.save:
            builder.save_feature_matrices(feature_matrices, args.season)
            logger.info("Saved all feature matrices to gold layer")

        logger.info("Feature matrix building completed successfully")

    except Exception as e:
        logger.error("Failed to build feature matrices", error=str(e))
        raise


if __name__ == "__main__":
    main()
