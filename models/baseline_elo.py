#!/usr/bin/env python3
"""
Baseline Elo Model for NFL Predictions

This module implements a pure Elo-based baseline model for NFL predictions:
- Pure Elo-based win probability calculation
- Logistic regression on Elo difference
- Home field advantage integration
- Simple, interpretable baseline for comparison
- Walk-forward training and validation
- Calibrated probability outputs

The baseline model serves as a benchmark against which more complex models
can be compared and provides a strong foundation for NFL win probability.
"""

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from conf.season_partition import (
    LATEST_COMPLETED_SEASON,
    SELECTION_WINDOW_FIRST_SEASON,
)
from models.calibrate import ProbabilityCalibrator
from models.utils import ModelMetadata, WalkForwardValidator
from ratings.elo import EloRatingSystem
from utils import get_logger

logger = get_logger(__name__)


@dataclass
class EloModelPrediction:
    """
    Container for Elo model prediction results.

    Attributes:
        game_id: Game identifier
        home_team: Home team abbreviation
        away_team: Away team abbreviation
        home_elo: Home team Elo rating
        away_elo: Away team Elo rating
        elo_diff: Elo difference (home - away)
        home_advantage: Home field advantage points
        raw_win_probability: Raw win probability from Elo formula
        calibrated_win_probability: Calibrated win probability
        prediction_confidence: Model confidence (distance from 0.5)
        metadata: Additional prediction metadata
    """

    game_id: str
    home_team: str
    away_team: str
    home_elo: float
    away_elo: float
    elo_diff: float
    home_advantage: float
    raw_win_probability: float
    calibrated_win_probability: float | None = None
    prediction_confidence: float | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class EloModelResults:
    """
    Container for Elo model training and validation results.

    Attributes:
        model: Trained logistic regression model
        calibrator: Trained probability calibrator
        elo_system: Trained Elo rating system
        predictions: List of prediction results
        performance_metrics: Dictionary of performance metrics
        validation_results: Walk-forward validation results
        model_metadata: Model metadata for persistence
        training_history: Training history and parameters
    """

    model: LogisticRegression
    calibrator: Any
    elo_system: EloRatingSystem
    predictions: list[EloModelPrediction]
    performance_metrics: dict[str, float]
    validation_results: dict[str, Any]
    model_metadata: ModelMetadata
    training_history: dict[str, Any]


class BaselineEloModel:
    """
    Baseline Elo model for NFL win probability predictions.

    Implements a simple but effective baseline using only Elo ratings
    and home field advantage to predict game outcomes.
    """

    def __init__(
        self,
        initial_elo: float = 1500,
        k_factor: float = 20,
        home_advantage: float = 65,
        season_carryover: float = 0.75,
        uncertainty_factor: float = 400,
        use_calibration: bool = True,
    ):
        """
        Initialize baseline Elo model.

        Args:
            initial_elo: Starting Elo rating for new teams
            k_factor: Elo K-factor for rating updates
            home_advantage: Home field advantage in Elo points
            season_carryover: Fraction of rating carried over between seasons
            uncertainty_factor: Elo uncertainty factor (400 standard)
            use_calibration: Whether to calibrate probabilities
        """
        self.initial_elo = initial_elo
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.season_carryover = season_carryover
        self.uncertainty_factor = uncertainty_factor
        self.use_calibration = use_calibration

        self.logger = get_logger(__name__)

        # Model components
        self.elo_system = None
        self.logistic_model = None
        self.calibrator = None
        self.model_metadata = None

        # Training state
        self.is_trained = False
        self.training_history = {}

    def calculate_win_probability(
        self, home_elo: float, away_elo: float, home_advantage: float | None = None
    ) -> float:
        """
        Calculate win probability using pure Elo formula.

        Uses the standard Elo probability formula:
        P(home wins) = 1 / (1 + 10^((away_elo - (home_elo + home_advantage)) / 400))

        Args:
            home_elo: Home team Elo rating
            away_elo: Away team Elo rating
            home_advantage: Home field advantage (uses model default if None)

        Returns:
            Home team win probability (0-1)
        """
        if home_advantage is None:
            home_advantage = self.home_advantage

        # Calculate Elo difference including home advantage
        elo_diff = (home_elo + home_advantage) - away_elo

        # Standard Elo probability formula
        win_prob = 1 / (1 + 10 ** (-elo_diff / self.uncertainty_factor))

        return np.clip(win_prob, 1e-10, 1 - 1e-10)  # Avoid extreme probabilities

    def prepare_training_data(
        self, games_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Prepare training data from games DataFrame.

        Extracts Elo ratings for each game and creates feature matrix
        for logistic regression training.

        Args:
            games_df: DataFrame with game data including Elo ratings

        Returns:
            Tuple of (features_df, targets_array)
        """
        logger.info("Preparing Elo model training data", games=len(games_df))

        # Required columns
        required_cols = ["home_team", "away_team", "home_wins"]
        missing_cols = [col for col in required_cols if col not in games_df.columns]

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Check for Elo columns (should be present from Elo system)
        elo_cols = ["elo_home", "elo_away"]
        missing_elo_cols = [col for col in elo_cols if col not in games_df.columns]

        if missing_elo_cols:
            logger.warning(
                "Missing Elo columns, will need to compute from Elo system",
                missing_columns=missing_elo_cols,
            )

        # Create features DataFrame
        features_data = []

        for _, game in games_df.iterrows():
            # Get Elo ratings
            if "elo_home" in games_df.columns and "elo_away" in games_df.columns:
                home_elo = game["elo_home"]
                away_elo = game["elo_away"]
            # Get from Elo system if available
            elif self.elo_system is not None:
                home_elo = self.elo_system.get_rating(game["home_team"])
                away_elo = self.elo_system.get_rating(game["away_team"])
            else:
                # Use initial ratings as fallback
                home_elo = self.initial_elo
                away_elo = self.initial_elo

            # Calculate Elo difference and features
            elo_diff = home_elo - away_elo
            home_advantage = game.get("home_advantage", self.home_advantage)

            # Raw Elo win probability
            raw_win_prob = self.calculate_win_probability(
                home_elo, away_elo, home_advantage
            )

            feature_row = {
                "game_id": game.get("game_id", f"game_{len(features_data)}"),
                "home_team": game["home_team"],
                "away_team": game["away_team"],
                "home_elo": home_elo,
                "away_elo": away_elo,
                "elo_diff": elo_diff,
                "home_advantage": home_advantage,
                "raw_win_probability": raw_win_prob,
                "season": game.get("season"),
                "week": game.get("week"),
            }

            features_data.append(feature_row)

        features_df = pd.DataFrame(features_data)
        targets = games_df["home_wins"].values

        # Validate no missing targets
        valid_targets = ~np.isnan(targets)
        features_df = features_df[valid_targets]
        targets = targets[valid_targets]

        logger.info(
            "Training data prepared",
            features=len(features_df.columns),
            samples=len(features_df),
            positive_rate=targets.mean(),
        )

        return features_df, targets

    def train_model(
        self,
        training_data: pd.DataFrame,
        validation_data: pd.DataFrame | None = None,
        fit_elo_system: bool = True,
    ) -> EloModelResults:
        """
        Train the baseline Elo model.

        Args:
            training_data: DataFrame with training games
            validation_data: Optional validation data
            fit_elo_system: Whether to fit Elo system on training data

        Returns:
            EloModelResults with trained model and performance metrics
        """
        logger.info(
            "Training baseline Elo model",
            training_games=len(training_data),
            validation_games=len(validation_data) if validation_data is not None else 0,
        )

        # Initialize and train Elo system if requested
        if fit_elo_system:
            logger.info("Fitting Elo rating system")
            self.elo_system = EloRatingSystem(
                base_k=self.k_factor,
                hfa_init=self.home_advantage,
                season_carryover=self.season_carryover,
            )

            # Process games chronologically to build Elo ratings
            training_sorted = training_data.sort_values(["season", "week"]).copy()

            for _, game in training_sorted.iterrows():
                if pd.notna(game.get("home_wins")):
                    # Update Elo ratings based on game result
                    home_wins = bool(game["home_wins"])
                    margin = game.get("point_differential", 0)

                    # Convert to scores (simulated for Elo updates)
                    if home_wins:
                        home_score = 24 + abs(margin) if pd.notna(margin) else 24
                        away_score = 17
                    else:
                        home_score = 17
                        away_score = 24 + abs(margin) if pd.notna(margin) else 24

                    self.elo_system.update_ratings(
                        home_team=game["home_team"],
                        away_team=game["away_team"],
                        home_score=int(home_score),
                        away_score=int(away_score),
                        season=game.get("season", LATEST_COMPLETED_SEASON),
                        game_date=game.get("kickoff_et", datetime.now()),
                        game_id=game.get(
                            "game_id", f"{game['home_team']}_vs_{game['away_team']}"
                        ),
                    )

        # Prepare training features
        train_features, train_targets = self.prepare_training_data(training_data)

        # Train logistic regression model
        logger.info("Training logistic regression model")

        # Feature matrix: elo_diff, home_advantage, and raw_win_probability
        X_train = train_features[
            ["elo_diff", "home_advantage", "raw_win_probability"]
        ].values
        y_train = train_targets

        self.logistic_model = LogisticRegression(
            random_state=42, max_iter=1000, solver="lbfgs"
        )

        self.logistic_model.fit(X_train, y_train)

        # Train calibrator if requested
        if self.use_calibration:
            logger.info("Training probability calibrator")
            raw_probabilities = self.logistic_model.predict_proba(X_train)[:, 1]

            calibrator_system = ProbabilityCalibrator(primary_method="isotonic")
            calibration_results = calibrator_system.calibrate_probabilities(
                raw_probabilities, y_train
            )
            self.calibrator = calibration_results.calibrator

        # Generate predictions for training data
        train_predictions = self._generate_predictions(train_features, calibrated=True)

        # Calculate training metrics
        train_metrics = self._calculate_performance_metrics(
            train_predictions, train_targets, "training"
        )

        # Validate on validation data if provided
        validation_metrics = {}
        val_predictions = []

        if validation_data is not None:
            val_features, val_targets = self.prepare_training_data(validation_data)
            val_predictions = self._generate_predictions(val_features, calibrated=True)
            validation_metrics = self._calculate_performance_metrics(
                val_predictions, val_targets, "validation"
            )

        # Create model metadata
        self.model_metadata = ModelMetadata(
            model_name="baseline_elo",
            model_type="BaselineEloModel",
            target_type="wp",
            train_seasons=sorted(training_data["season"].unique())
            if "season" in training_data.columns
            else [],
            features_used=["elo_diff", "home_advantage", "raw_win_probability"],
            training_date=datetime.now(),
            performance_metrics={**train_metrics, **validation_metrics},
            hyperparameters={
                "initial_elo": self.initial_elo,
                "k_factor": self.k_factor,
                "home_advantage": self.home_advantage,
                "season_carryover": self.season_carryover,
                "uncertainty_factor": self.uncertainty_factor,
                "use_calibration": self.use_calibration,
            },
        )

        # Store training history
        self.training_history = {
            "training_samples": len(training_data),
            "validation_samples": len(validation_data)
            if validation_data is not None
            else 0,
            "elo_teams": len(self.elo_system.ratings) if self.elo_system else 0,
            "training_date": datetime.now(),
            "model_coefficients": self.logistic_model.coef_.flatten().tolist(),
            "model_intercept": float(self.logistic_model.intercept_[0]),
        }

        self.is_trained = True

        # Create results object
        results = EloModelResults(
            model=self.logistic_model,
            calibrator=self.calibrator,
            elo_system=self.elo_system,
            predictions=train_predictions + val_predictions,
            performance_metrics={**train_metrics, **validation_metrics},
            validation_results={
                "training": train_metrics,
                "validation": validation_metrics,
            },
            model_metadata=self.model_metadata,
            training_history=self.training_history,
        )

        logger.info(
            "Baseline Elo model training completed",
            training_accuracy=train_metrics.get("accuracy", 0),
            validation_accuracy=validation_metrics.get("accuracy", 0)
            if validation_metrics
            else None,
        )

        return results

    def predict(
        self, games_df: pd.DataFrame, calibrated: bool = True
    ) -> list[EloModelPrediction]:
        """
        Generate predictions for games.

        Args:
            games_df: DataFrame with games to predict
            calibrated: Whether to return calibrated probabilities

        Returns:
            List of EloModelPrediction objects
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        logger.info(
            "Generating Elo model predictions",
            games=len(games_df),
            calibrated=calibrated,
        )

        # Prepare features
        features_df, _ = self.prepare_training_data(games_df)

        # Generate predictions
        predictions = self._generate_predictions(features_df, calibrated)

        return predictions

    def _generate_predictions(
        self, features_df: pd.DataFrame, calibrated: bool = True
    ) -> list[EloModelPrediction]:
        """
        Generate predictions from features DataFrame.

        Args:
            features_df: Features DataFrame
            calibrated: Whether to apply calibration

        Returns:
            List of EloModelPrediction objects
        """
        predictions = []

        # Prepare feature matrix
        X = features_df[["elo_diff", "home_advantage", "raw_win_probability"]].values

        # Get raw probabilities from logistic regression
        raw_probabilities = self.logistic_model.predict_proba(X)[:, 1]

        # Apply calibration if requested and available
        calibrated_probabilities = None
        if calibrated and self.calibrator is not None:
            calibrated_probabilities = self.calibrator.predict(
                raw_probabilities.reshape(-1, 1)
            )

        # Create prediction objects
        for idx, (_, row) in enumerate(features_df.iterrows()):
            raw_prob = raw_probabilities[idx]
            cal_prob = (
                calibrated_probabilities[idx]
                if calibrated_probabilities is not None
                else None
            )

            # Calculate prediction confidence
            final_prob = cal_prob if cal_prob is not None else raw_prob
            confidence = abs(final_prob - 0.5) * 2  # Scale to 0-1

            prediction = EloModelPrediction(
                game_id=row["game_id"],
                home_team=row["home_team"],
                away_team=row["away_team"],
                home_elo=row["home_elo"],
                away_elo=row["away_elo"],
                elo_diff=row["elo_diff"],
                home_advantage=row["home_advantage"],
                raw_win_probability=raw_prob,
                calibrated_win_probability=cal_prob,
                prediction_confidence=confidence,
                metadata={
                    "season": row.get("season"),
                    "week": row.get("week"),
                    "prediction_date": datetime.now(),
                },
            )

            predictions.append(prediction)

        return predictions

    def _calculate_performance_metrics(
        self,
        predictions: list[EloModelPrediction],
        true_labels: np.ndarray,
        dataset_name: str,
    ) -> dict[str, float]:
        """
        Calculate performance metrics for predictions.

        Args:
            predictions: List of predictions
            true_labels: True binary labels
            dataset_name: Name of dataset for metric keys

        Returns:
            Dictionary of performance metrics
        """
        if len(predictions) != len(true_labels):
            raise ValueError("Predictions and labels must have same length")

        # Extract probabilities
        raw_probs = np.array([p.raw_win_probability for p in predictions])
        cal_probs = np.array(
            [
                p.calibrated_win_probability
                for p in predictions
                if p.calibrated_win_probability is not None
            ]
        )

        metrics = {}

        # Raw probability metrics
        metrics[f"{dataset_name}_raw_accuracy"] = accuracy_score(
            true_labels, raw_probs > 0.5
        )

        try:
            metrics[f"{dataset_name}_raw_log_loss"] = log_loss(true_labels, raw_probs)
        except ValueError:
            metrics[f"{dataset_name}_raw_log_loss"] = np.nan

        try:
            metrics[f"{dataset_name}_raw_brier_score"] = brier_score_loss(
                true_labels, raw_probs
            )
        except ValueError:
            metrics[f"{dataset_name}_raw_brier_score"] = np.nan

        # Calibrated probability metrics (if available)
        if len(cal_probs) > 0:
            metrics[f"{dataset_name}_calibrated_accuracy"] = accuracy_score(
                true_labels, cal_probs > 0.5
            )

            try:
                metrics[f"{dataset_name}_calibrated_log_loss"] = log_loss(
                    true_labels, cal_probs
                )
            except ValueError:
                metrics[f"{dataset_name}_calibrated_log_loss"] = np.nan

            try:
                metrics[f"{dataset_name}_calibrated_brier_score"] = brier_score_loss(
                    true_labels, cal_probs
                )
            except ValueError:
                metrics[f"{dataset_name}_calibrated_brier_score"] = np.nan

        # General metrics
        metrics[f"{dataset_name}_samples"] = len(predictions)
        metrics[f"{dataset_name}_positive_rate"] = float(true_labels.mean())

        # Use calibrated metrics as primary if available, otherwise raw
        if len(cal_probs) > 0:
            metrics[f"{dataset_name}_accuracy"] = metrics[
                f"{dataset_name}_calibrated_accuracy"
            ]
            metrics[f"{dataset_name}_log_loss"] = metrics[
                f"{dataset_name}_calibrated_log_loss"
            ]
            metrics[f"{dataset_name}_brier_score"] = metrics[
                f"{dataset_name}_calibrated_brier_score"
            ]
        else:
            metrics[f"{dataset_name}_accuracy"] = metrics[
                f"{dataset_name}_raw_accuracy"
            ]
            metrics[f"{dataset_name}_log_loss"] = metrics[
                f"{dataset_name}_raw_log_loss"
            ]
            metrics[f"{dataset_name}_brier_score"] = metrics[
                f"{dataset_name}_raw_brier_score"
            ]

        return metrics

    def run_walk_forward_validation(
        self,
        games_df: pd.DataFrame,
        start_season: int = SELECTION_WINDOW_FIRST_SEASON,
        end_season: int = LATEST_COMPLETED_SEASON,
    ) -> dict[str, Any]:
        """
        Run walk-forward validation on historical data.

        Args:
            games_df: DataFrame with historical games
            start_season: First season for validation
            end_season: Last season for validation

        Returns:
            Dictionary with walk-forward validation results
        """
        logger.info(
            "Starting walk-forward validation",
            start_season=start_season,
            end_season=end_season,
        )

        validator = WalkForwardValidator(min_train_seasons=1)

        # Ensure required columns
        if "home_wins" not in games_df.columns:
            raise ValueError("games_df must contain 'home_wins' column")

        validation_results = {
            "start_season": start_season,
            "end_season": end_season,
            "season_results": {},
            "overall_metrics": {},
            "model_stability": {},
        }

        all_predictions = []
        all_actuals = []
        season_accuracies = []

        # Run walk-forward validation
        for split in validator.create_seasonal_splits(
            games_df, "home_wins", start_season, end_season
        ):
            test_season = split.test_season
            logger.info(
                f"Validating season {test_season}",
                train_seasons=split.train_seasons,
                train_games=len(split.train_data),
                test_games=len(split.test_data),
            )

            try:
                # Reconstruct training data with targets
                train_data_full = split.train_data.copy()
                train_data_full["home_wins"] = split.train_targets

                # Reconstruct test data with targets
                test_data_full = split.test_data.copy()
                test_data_full["home_wins"] = split.test_targets

                # Train model on this split
                season_model = BaselineEloModel(
                    initial_elo=self.initial_elo,
                    k_factor=self.k_factor,
                    home_advantage=self.home_advantage,
                    season_carryover=self.season_carryover,
                    uncertainty_factor=self.uncertainty_factor,
                    use_calibration=self.use_calibration,
                )

                season_model.train_model(train_data_full, test_data_full)

                # Get test predictions
                test_predictions = season_model.predict(test_data_full)

                # Calculate season metrics
                season_metrics = season_model._calculate_performance_metrics(
                    test_predictions, split.test_targets, f"season_{test_season}"
                )

                validation_results["season_results"][test_season] = {
                    "predictions": test_predictions,
                    "metrics": season_metrics,
                    "train_seasons": split.train_seasons,
                    "train_games": len(split.train_data),
                    "test_games": len(split.test_data),
                }

                # Collect for overall metrics
                pred_probs = [
                    p.calibrated_win_probability or p.raw_win_probability
                    for p in test_predictions
                ]
                all_predictions.extend(pred_probs)
                all_actuals.extend(split.test_targets)
                season_accuracies.append(
                    season_metrics[f"season_{test_season}_accuracy"]
                )

                logger.info(
                    f"Season {test_season} validation completed",
                    accuracy=season_metrics[f"season_{test_season}_accuracy"],
                )

            except (ValueError, KeyError, TypeError, RuntimeError) as e:
                logger.error(f"Season {test_season} validation failed", error=str(e))
                validation_results["season_results"][test_season] = {
                    "error": str(e),
                    "train_seasons": split.train_seasons,
                    "train_games": len(split.train_data),
                    "test_games": len(split.test_data),
                }

        # Calculate overall metrics
        if all_predictions and all_actuals:
            all_predictions = np.array(all_predictions)
            all_actuals = np.array(all_actuals)

            validation_results["overall_metrics"] = {
                "overall_accuracy": accuracy_score(all_actuals, all_predictions > 0.5),
                "overall_log_loss": log_loss(all_actuals, all_predictions),
                "overall_brier_score": brier_score_loss(all_actuals, all_predictions),
                "total_predictions": len(all_predictions),
                "seasons_validated": len(
                    [
                        s
                        for s in validation_results["season_results"].values()
                        if "error" not in s
                    ]
                ),
            }

            # Model stability metrics
            if len(season_accuracies) > 1:
                validation_results["model_stability"] = {
                    "accuracy_mean": np.mean(season_accuracies),
                    "accuracy_std": np.std(season_accuracies),
                    "accuracy_min": np.min(season_accuracies),
                    "accuracy_max": np.max(season_accuracies),
                    "accuracy_range": np.max(season_accuracies)
                    - np.min(season_accuracies),
                }

        logger.info(
            "Walk-forward validation completed",
            overall_accuracy=validation_results["overall_metrics"].get(
                "overall_accuracy", 0
            ),
            seasons_validated=validation_results["overall_metrics"].get(
                "seasons_validated", 0
            ),
        )

        return validation_results

    def save_model(self, save_path: str):
        """
        Save trained model to disk.

        Args:
            save_path: Path to save model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")

        model_data = {
            "logistic_model": self.logistic_model,
            "calibrator": self.calibrator,
            "elo_system": self.elo_system,
            "model_metadata": self.model_metadata,
            "training_history": self.training_history,
            "hyperparameters": {
                "initial_elo": self.initial_elo,
                "k_factor": self.k_factor,
                "home_advantage": self.home_advantage,
                "season_carryover": self.season_carryover,
                "uncertainty_factor": self.uncertainty_factor,
                "use_calibration": self.use_calibration,
            },
        }

        joblib.dump(model_data, save_path)
        logger.info(f"Baseline Elo model saved to {save_path}")

    def load_model(self, load_path: str):
        """
        Load trained model from disk.

        Args:
            load_path: Path to load model from
        """
        model_data = joblib.load(load_path)

        self.logistic_model = model_data["logistic_model"]
        self.calibrator = model_data["calibrator"]
        self.elo_system = model_data["elo_system"]
        self.model_metadata = model_data["model_metadata"]
        self.training_history = model_data["training_history"]

        # Restore hyperparameters
        hyperparams = model_data["hyperparameters"]
        self.initial_elo = hyperparams["initial_elo"]
        self.k_factor = hyperparams["k_factor"]
        self.home_advantage = hyperparams["home_advantage"]
        self.season_carryover = hyperparams["season_carryover"]
        self.uncertainty_factor = hyperparams["uncertainty_factor"]
        self.use_calibration = hyperparams["use_calibration"]

        self.is_trained = True
        logger.info(f"Baseline Elo model loaded from {load_path}")

    def get_model_summary(self) -> dict[str, Any]:
        """
        Get summary of trained model.

        Returns:
            Dictionary with model summary information
        """
        if not self.is_trained:
            return {"error": "Model not trained"}

        summary = {
            "model_type": "BaselineEloModel",
            "is_trained": self.is_trained,
            "hyperparameters": {
                "initial_elo": self.initial_elo,
                "k_factor": self.k_factor,
                "home_advantage": self.home_advantage,
                "season_carryover": self.season_carryover,
                "uncertainty_factor": self.uncertainty_factor,
                "use_calibration": self.use_calibration,
            },
            "training_info": self.training_history,
            "model_coefficients": {
                "intercept": float(self.logistic_model.intercept_[0]),
                "elo_diff_coef": float(self.logistic_model.coef_[0][0]),
                "home_advantage_coef": float(self.logistic_model.coef_[0][1]),
                "raw_prob_coef": float(self.logistic_model.coef_[0][2]),
            },
            "elo_system_info": {
                "total_teams": len(self.elo_system.ratings) if self.elo_system else 0,
                "rating_range": self._get_elo_rating_range()
                if self.elo_system
                else None,
            },
            "calibration_used": self.calibrator is not None,
        }

        if self.model_metadata:
            summary["performance_metrics"] = self.model_metadata.performance_metrics

        return summary

    def _get_elo_rating_range(self) -> dict[str, float]:
        """Get Elo rating range statistics."""
        if not self.elo_system:
            return {}

        ratings_df = self.elo_system.get_current_ratings()
        if len(ratings_df) == 0:
            return {}

        all_ratings = ratings_df["rating"].values
        if len(all_ratings) == 0:
            return {}

        return {
            "min_rating": float(np.min(all_ratings)),
            "max_rating": float(np.max(all_ratings)),
            "mean_rating": float(np.mean(all_ratings)),
            "std_rating": float(np.std(all_ratings)),
        }
