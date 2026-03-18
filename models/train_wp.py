#!/usr/bin/env python3
"""
Win Probability Model Training

This module implements a comprehensive Win Probability (WP) model for NFL predictions:
- Advanced logistic regression with L1/L2 regularization
- Automated feature selection with statistical and model-based methods
- Hyperparameter tuning with cross-validation
- Walk-forward training protocol for temporal data
- Integrated probability calibration
- Model persistence and loading with versioning
- Comprehensive evaluation and validation

The WP model serves as the foundation for game outcome predictions and integrates
with the broader prediction pipeline for ATS and O/U models.
"""

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ML imports
from sklearn.feature_selection import (
    RFE,
    SelectFromModel,
    SelectKBest,
    VarianceThreshold,
    f_classif,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
    StratifiedKFold,
)
from sklearn.preprocessing import StandardScaler

from data.storage import load_dataframe
from models.calibrate import ProbabilityCalibrator
from models.evaluation import ModelEvaluationFramework

# Project imports
from models.utils import ModelMetadata, WalkForwardValidator
from utils import get_logger

logger = get_logger(__name__)


@dataclass
class WPModelPrediction:
    """
    Container for WP model prediction results.

    Attributes:
        game_id: Unique game identifier
        home_team: Home team abbreviation
        away_team: Away team abbreviation
        raw_win_probability: Raw model probability
        calibrated_win_probability: Calibrated probability
        prediction_confidence: Model confidence (distance from 0.5)
        feature_importances: Feature contribution to prediction
        model_version: Version of model used
        prediction_date: When prediction was made
        metadata: Additional prediction metadata
    """

    game_id: str
    home_team: str
    away_team: str
    raw_win_probability: float
    calibrated_win_probability: float | None = None
    prediction_confidence: float | None = None
    feature_importances: dict[str, float] | None = None
    model_version: str | None = None
    prediction_date: datetime | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class WPModelResults:
    """
    Container for WP model training and validation results.

    Attributes:
        model: Trained logistic regression model
        scaler: Fitted feature scaler
        feature_selector: Trained feature selector
        calibrator: Trained probability calibrator
        predictions: Validation predictions
        performance_metrics: Evaluation metrics
        feature_names: Names of selected features
        feature_importances: Feature importance scores
        hyperparameters: Final hyperparameters used
        model_metadata: Model metadata for persistence
        training_history: Training process details
    """

    model: LogisticRegression
    scaler: StandardScaler
    feature_selector: Any
    calibrator: Any | None
    predictions: list[WPModelPrediction]
    performance_metrics: dict[str, float]
    feature_names: list[str]
    feature_importances: dict[str, float]
    hyperparameters: dict[str, Any]
    model_metadata: ModelMetadata
    training_history: dict[str, Any]


class WinProbabilityModel:
    """
    Comprehensive Win Probability model for NFL game predictions.

    This model implements state-of-the-art techniques for binary classification
    of NFL game outcomes, including automated feature selection, hyperparameter
    optimization, and proper temporal validation.
    """

    def __init__(
        self,
        feature_selection_method: str = "recursive",
        max_features: int | None = None,
        regularization_type: str = "elasticnet",
        use_calibration: bool = True,
        validation_method: str = "walk_forward",
        hyperparameter_tuning: str = "grid_search",
        random_state: int = 42,
    ):
        """
        Initialize Win Probability model.

        Args:
            feature_selection_method: Method for feature selection
                ('variance', 'univariate', 'model_based', 'recursive', 'none')
            max_features: Maximum number of features to select (None for auto)
            regularization_type: Type of regularization ('l1', 'l2', 'elasticnet')
            use_calibration: Whether to calibrate probabilities
            validation_method: Validation strategy ('walk_forward', 'cv')
            hyperparameter_tuning: Tuning method ('grid_search', 'random_search', 'none')
            random_state: Random seed for reproducibility
        """
        self.feature_selection_method = feature_selection_method
        self.max_features = max_features
        self.regularization_type = regularization_type
        self.use_calibration = use_calibration
        self.validation_method = validation_method
        self.hyperparameter_tuning = hyperparameter_tuning
        self.random_state = random_state

        self.logger = get_logger(__name__)

        # Model components
        self.model = None
        self.scaler = None
        self.feature_selector = None
        self.calibrator = None
        self.model_metadata = None

        # Training state
        self.is_trained = False
        self.feature_names = []
        self.feature_importances = {}
        self.training_history = {}
        self.hyperparameters = {}

        # Performance tracking
        self.evaluator = ModelEvaluationFramework()

    def prepare_features(
        self, games_df: pd.DataFrame, feature_path: str | None = None
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Prepare features for training from games DataFrame.

        Args:
            games_df: DataFrame with game data or pre-built features (with target_wp column)
            feature_path: Path to pre-built features (optional)

        Returns:
            Tuple of (features_df, targets_array)
        """
        self.logger.info("Preparing WP model features", games=len(games_df))

        # Check if we already have pre-built features with target
        if "target_wp" in games_df.columns:
            self.logger.info("Using pre-built features with target")
            targets = games_df["target_wp"].values
            features_df = games_df.drop(["target_wp"], axis=1)

        elif feature_path and Path(feature_path).exists():
            # Load pre-built features
            self.logger.info("Loading pre-built features", path=feature_path)
            features_df = load_dataframe(feature_path)

            # Merge with games data for targets
            merged_df = features_df.merge(
                games_df[["game_id", "home_wins"]], on="game_id", how="inner"
            )

            if len(merged_df) == 0:
                raise ValueError("No games found after merging features with targets")

            targets = merged_df["home_wins"].values
            features_df = merged_df.drop(["home_wins"], axis=1)

        else:
            # Build features on the fly (simplified version)
            self.logger.warning(
                "Building features on the fly - consider pre-building for performance"
            )
            features_df = self._build_basic_features(games_df)
            targets = games_df["home_wins"].values

        # Clean and validate
        features_df, targets = self._clean_features(features_df, targets)

        self.logger.info(
            "WP features prepared",
            features=len(features_df.columns),
            samples=len(features_df),
            positive_rate=float(targets.mean()),
        )

        return features_df, targets

    def _build_basic_features(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """Build basic features for WP model."""
        features = []

        for _, game in games_df.iterrows():
            feature_row = {
                "game_id": game["game_id"],
                "home_team": game["home_team"],
                "away_team": game["away_team"],
                "season": game.get("season", 2024),
                "week": game.get("week", 1),
                # Basic Elo features (if available)
                "elo_home": game.get("elo_home", 1500),
                "elo_away": game.get("elo_away", 1500),
                "elo_diff": game.get("elo_home", 1500) - game.get("elo_away", 1500),
                # Basic contextual features
                "is_playoff": int(game.get("week", 1) > 18),
                "is_primetime": int(game.get("kickoff_hour", 13) >= 20),
                # Placeholder for additional features
                "rest_days_home": game.get("rest_days_home", 7),
                "rest_days_away": game.get("rest_days_away", 7),
                "temperature": game.get("temperature", 72),
                "wind_speed": game.get("wind_speed", 5),
            }

            features.append(feature_row)

        return pd.DataFrame(features)

    def _clean_features(
        self, features_df: pd.DataFrame, targets: np.ndarray
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """Clean and validate feature data."""
        # Remove non-feature columns
        potential_id_cols = [
            "game_id",
            "home_team",
            "away_team",
            "season",
            "week",
            "feature_timestamp",
        ]
        id_cols = [col for col in potential_id_cols if col in features_df.columns]

        # Also identify datetime and string columns as non-features
        datetime_cols = features_df.select_dtypes(
            include=["datetime64"]
        ).columns.tolist()
        string_cols = features_df.select_dtypes(include=["object"]).columns.tolist()

        all_id_cols = list(set(id_cols + datetime_cols + string_cols))
        feature_cols = [col for col in features_df.columns if col not in all_id_cols]

        # Keep ID columns for later use
        id_data = features_df[id_cols].copy() if id_cols else pd.DataFrame()
        X = features_df[feature_cols].copy()

        self.logger.info("Feature matrix shape before cleaning", shape=X.shape)
        self.logger.info("Feature dtypes unique", dtypes=X.dtypes.unique().tolist())

        # Ensure all feature columns are numeric
        X = X.select_dtypes(include=[np.number])
        self.logger.info("Feature matrix shape after numeric filtering", shape=X.shape)

        # Handle missing values
        X = X.fillna(X.median())

        # Remove constant features
        constant_features = X.columns[X.nunique() <= 1]
        if len(constant_features) > 0:
            self.logger.info("Removing constant features", count=len(constant_features))
            X = X.drop(columns=constant_features)

        # Handle infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(X.median())

        # Remove rows with missing targets
        valid_mask = ~np.isnan(targets)
        X = X[valid_mask]
        targets = targets[valid_mask]
        if len(id_data) > 0:
            id_data = id_data[valid_mask]

        # Combine back with ID columns
        if len(id_data) > 0:
            features_df_clean = pd.concat(
                [id_data.reset_index(drop=True), X.reset_index(drop=True)], axis=1
            )
        else:
            features_df_clean = X.copy()

        return features_df_clean, targets

    def select_features(self, X: pd.DataFrame, y: np.ndarray) -> tuple[Any, list[str]]:
        """
        Perform feature selection based on configured method.

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            Tuple of (fitted_selector, selected_feature_names)
        """
        self.logger.info(
            f"Performing feature selection using {self.feature_selection_method}"
        )

        # Get feature columns (exclude ID and non-numeric columns)
        potential_id_cols = [
            "game_id",
            "home_team",
            "away_team",
            "season",
            "week",
            "feature_timestamp",
        ]
        datetime_cols = X.select_dtypes(include=["datetime64"]).columns.tolist()
        string_cols = X.select_dtypes(include=["object"]).columns.tolist()

        all_id_cols = list(set(potential_id_cols + datetime_cols + string_cols))
        feature_cols = [col for col in X.columns if col not in all_id_cols]
        X_features = X[feature_cols]

        self.logger.info(
            "Feature selection input",
            shape=X_features.shape,
            dtypes=X_features.dtypes.unique().tolist(),
        )

        if self.feature_selection_method == "none":
            # No feature selection
            selector = None
            selected_features = feature_cols

        elif self.feature_selection_method == "variance":
            # Remove low-variance features
            threshold = 0.01
            selector = VarianceThreshold(threshold=threshold)
            selector.fit(X_features)
            mask = selector.get_support()
            selected_features = [f for i, f in enumerate(feature_cols) if mask[i]]

        elif self.feature_selection_method == "univariate":
            # Univariate statistical tests
            k = min(self.max_features or 20, len(feature_cols))
            selector = SelectKBest(score_func=f_classif, k=k)
            selector.fit(X_features, y)
            mask = selector.get_support()
            selected_features = [f for i, f in enumerate(feature_cols) if mask[i]]

        elif self.feature_selection_method == "model_based":
            # Model-based feature selection
            base_model = LogisticRegression(
                penalty="l1",
                solver="liblinear",
                random_state=self.random_state,
                max_iter=1000,
            )
            max_features = min(
                self.max_features or len(feature_cols), len(feature_cols)
            )
            selector = SelectFromModel(base_model, max_features=max_features)
            selector.fit(X_features, y)
            mask = selector.get_support()
            selected_features = [f for i, f in enumerate(feature_cols) if mask[i]]

        elif self.feature_selection_method == "recursive":
            # Recursive Feature Elimination
            base_model = LogisticRegression(
                penalty="l2",
                solver="liblinear",
                random_state=self.random_state,
                max_iter=1000,
            )
            n_features = min(self.max_features or 15, len(feature_cols))
            selector = RFE(base_model, n_features_to_select=n_features)
            selector.fit(X_features, y)
            mask = selector.get_support()
            selected_features = [f for i, f in enumerate(feature_cols) if mask[i]]

        else:
            raise ValueError(
                f"Unknown feature selection method: {self.feature_selection_method}"
            )

        self.logger.info(
            "Feature selection completed",
            original_features=len(feature_cols),
            selected_features=len(selected_features),
        )

        return selector, selected_features

    def tune_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        """
        Tune hyperparameters using cross-validation.

        Args:
            X: Feature matrix
            y: Target vector

        Returns:
            Best hyperparameters found
        """
        if self.hyperparameter_tuning == "none":
            # Use default parameters
            return self._get_default_hyperparameters()

        self.logger.info(f"Tuning hyperparameters using {self.hyperparameter_tuning}")

        # Define parameter space
        param_space = self._get_parameter_space()

        # Create base model
        base_model = LogisticRegression(
            random_state=self.random_state,
            max_iter=2000,
            solver="saga",  # Supports all penalty types
        )

        # Cross-validation setup
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)

        if self.hyperparameter_tuning == "grid_search":
            # Grid search
            search = GridSearchCV(
                base_model,
                param_space,
                cv=cv,
                scoring="neg_log_loss",
                n_jobs=-1,
                verbose=0,
            )

        elif self.hyperparameter_tuning == "random_search":
            # Random search
            search = RandomizedSearchCV(
                base_model,
                param_space,
                n_iter=50,
                cv=cv,
                scoring="neg_log_loss",
                n_jobs=-1,
                random_state=self.random_state,
                verbose=0,
            )

        # Perform search
        search.fit(X, y)

        best_params = search.best_params_
        best_score = -search.best_score_

        self.logger.info(
            "Hyperparameter tuning completed",
            best_score=best_score,
            best_params=best_params,
        )

        return best_params

    def _get_parameter_space(self) -> dict[str, Any]:
        """Get hyperparameter search space based on regularization type."""
        if self.regularization_type == "l1":
            return {
                "penalty": ["l1"],
                "C": [0.001, 0.01, 0.1, 1, 10, 100],
                "solver": ["liblinear", "saga"],
            }
        if self.regularization_type == "l2":
            return {
                "penalty": ["l2"],
                "C": [0.001, 0.01, 0.1, 1, 10, 100],
                "solver": ["liblinear", "lbfgs", "saga"],
            }
        if self.regularization_type == "elasticnet":
            return {
                "penalty": ["elasticnet"],
                "C": [0.001, 0.01, 0.1, 1, 10],
                "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],
                "solver": ["saga"],
            }
        return {"penalty": ["none"], "solver": ["lbfgs", "saga"]}

    def _get_default_hyperparameters(self) -> dict[str, Any]:
        """Get default hyperparameters."""
        if self.regularization_type == "l1":
            return {"penalty": "l1", "C": 1.0, "solver": "liblinear"}
        if self.regularization_type == "l2":
            return {"penalty": "l2", "C": 1.0, "solver": "lbfgs"}
        if self.regularization_type == "elasticnet":
            return {
                "penalty": "elasticnet",
                "C": 1.0,
                "l1_ratio": 0.5,
                "solver": "saga",
            }
        return {"penalty": "none", "solver": "lbfgs"}

    def train_model(
        self,
        training_data: pd.DataFrame,
        validation_data: pd.DataFrame | None = None,
        feature_path: str | None = None,
    ) -> WPModelResults:
        """
        Train the Win Probability model with full pipeline.

        Args:
            training_data: Training games DataFrame
            validation_data: Optional validation games DataFrame
            feature_path: Path to pre-built features

        Returns:
            WPModelResults with trained model and metrics
        """
        self.logger.info(
            "Training WP model",
            training_games=len(training_data),
            validation_games=len(validation_data) if validation_data else 0,
        )

        start_time = datetime.now()

        # Prepare features
        features_df, targets = self.prepare_features(training_data, feature_path)

        # Feature selection
        self.feature_selector, self.feature_names = self.select_features(
            features_df, targets
        )

        # Extract selected features
        X = features_df[self.feature_names].values
        y = targets

        # Feature scaling
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Hyperparameter tuning
        self.hyperparameters = self.tune_hyperparameters(X_scaled, y)

        # Train final model
        self.model = LogisticRegression(
            random_state=self.random_state, max_iter=2000, **self.hyperparameters
        )
        self.model.fit(X_scaled, y)

        # Calculate feature importances
        self.feature_importances = self._calculate_feature_importances()

        # Calibration
        if self.use_calibration:
            self.logger.info("Training probability calibrator")
            raw_probs = self.model.predict_proba(X_scaled)[:, 1]
            calibrator_system = ProbabilityCalibrator(primary_method="isotonic")
            calibration_results = calibrator_system.calibrate_probabilities(
                raw_probs, y
            )
            self.calibrator = calibration_results.calibrator

        # Generate training predictions
        train_predictions = self._generate_predictions(features_df, calibrated=True)

        # Evaluate on training data
        train_metrics = self._evaluate_predictions(
            train_predictions, targets, "training"
        )

        # Validation if provided
        val_predictions = []
        val_metrics = {}
        if validation_data is not None:
            val_predictions = self.predict(validation_data, calibrated=True)
            val_targets = validation_data["home_wins"].values
            val_metrics = self._evaluate_predictions(
                val_predictions, val_targets, "validation"
            )

        # Create model metadata
        self.model_metadata = ModelMetadata(
            model_name="win_probability_model",
            model_type="LogisticRegression",
            target_type="wp",
            train_seasons=sorted(training_data["season"].unique())
            if "season" in training_data.columns
            else [],
            features_used=self.feature_names,
            training_date=datetime.now(),
            performance_metrics={**train_metrics, **val_metrics},
            hyperparameters=self.hyperparameters,
        )

        # Training history
        self.training_history = {
            "training_samples": len(training_data),
            "validation_samples": len(validation_data) if validation_data else 0,
            "features_selected": len(self.feature_names),
            "training_time": (datetime.now() - start_time).total_seconds(),
            "feature_selection_method": self.feature_selection_method,
            "hyperparameter_tuning": self.hyperparameter_tuning,
            "regularization_type": self.regularization_type,
            "use_calibration": self.use_calibration,
        }

        self.is_trained = True

        # Create results
        results = WPModelResults(
            model=self.model,
            scaler=self.scaler,
            feature_selector=self.feature_selector,
            calibrator=self.calibrator,
            predictions=train_predictions + val_predictions,
            performance_metrics={**train_metrics, **val_metrics},
            feature_names=self.feature_names,
            feature_importances=self.feature_importances,
            hyperparameters=self.hyperparameters,
            model_metadata=self.model_metadata,
            training_history=self.training_history,
        )

        self.logger.info(
            "WP model training completed",
            training_accuracy=train_metrics.get("accuracy", 0),
            validation_accuracy=val_metrics.get("accuracy", 0) if val_metrics else None,
            features_used=len(self.feature_names),
        )

        return results

    def predict(
        self,
        games_df: pd.DataFrame,
        calibrated: bool = True,
        feature_path: str | None = None,
    ) -> list[WPModelPrediction]:
        """
        Generate predictions for games.

        Args:
            games_df: DataFrame with games to predict
            calibrated: Whether to return calibrated probabilities
            feature_path: Path to pre-built features

        Returns:
            List of WPModelPrediction objects
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        self.logger.info(
            "Generating WP predictions", games=len(games_df), calibrated=calibrated
        )

        # Prepare features (no targets needed for prediction)
        features_df, _ = self.prepare_features(games_df, feature_path)

        # Generate predictions
        predictions = self._generate_predictions(features_df, calibrated)

        return predictions

    def _generate_predictions(
        self, features_df: pd.DataFrame, calibrated: bool = True
    ) -> list[WPModelPrediction]:
        """Generate predictions from features DataFrame."""
        predictions = []

        # Ensure all required features are present
        missing_features = [
            f for f in self.feature_names if f not in features_df.columns
        ]
        if missing_features:
            self.logger.warning(
                "Missing features for prediction, using default values",
                missing=missing_features,
            )
            # Add missing features with default values
            for feature in missing_features:
                if "season" in feature.lower():
                    features_df[feature] = 2024
                elif "week" in feature.lower():
                    features_df[feature] = 1
                elif "elo" in feature.lower():
                    features_df[feature] = 1500
                elif "temperature" in feature.lower():
                    features_df[feature] = 70
                elif "wind" in feature.lower():
                    features_df[feature] = 5
                else:
                    features_df[feature] = 0

        # Extract features
        X = features_df[self.feature_names].values
        X_scaled = self.scaler.transform(X)

        # Get raw probabilities
        raw_probs = self.model.predict_proba(X_scaled)[:, 1]

        # Apply calibration if requested
        cal_probs = None
        if calibrated and self.calibrator is not None:
            cal_probs = self.calibrator.predict(raw_probs.reshape(-1, 1))

        # Calculate feature importances for each prediction (simplified)
        feature_contribs = self._calculate_prediction_contributions(X_scaled)

        # Create prediction objects
        for idx, (_, row) in enumerate(features_df.iterrows()):
            raw_prob = raw_probs[idx]
            cal_prob = cal_probs[idx] if cal_probs is not None else None

            # Calculate confidence
            final_prob = cal_prob if cal_prob is not None else raw_prob
            confidence = abs(final_prob - 0.5) * 2

            prediction = WPModelPrediction(
                game_id=row.get("game_id", f"game_{idx}"),
                home_team=row.get("home_team", "HOME"),
                away_team=row.get("away_team", "AWAY"),
                raw_win_probability=raw_prob,
                calibrated_win_probability=cal_prob,
                prediction_confidence=confidence,
                feature_importances=feature_contribs[idx] if feature_contribs else None,
                model_version=self.model_metadata.model_name
                if self.model_metadata
                else "unknown",
                prediction_date=datetime.now(),
                metadata={
                    "regularization": self.regularization_type,
                    "features_used": len(self.feature_names),
                    "calibrated": calibrated,
                },
            )

            predictions.append(prediction)

        return predictions

    def _calculate_feature_importances(self) -> dict[str, float]:
        """Calculate feature importances from trained model."""
        if not self.is_trained or self.model is None:
            return {}

        # Get coefficients
        coefs = self.model.coef_[0]

        # Calculate importances as absolute coefficients
        importances = {}
        for i, feature_name in enumerate(self.feature_names):
            importances[feature_name] = abs(coefs[i])

        # Normalize to sum to 1
        total = sum(importances.values())
        if total > 0:
            importances = {k: v / total for k, v in importances.items()}

        return importances

    def _calculate_prediction_contributions(
        self, X_scaled: np.ndarray
    ) -> list[dict[str, float]] | None:
        """Calculate feature contributions for each prediction (simplified)."""
        if len(self.feature_names) > 20:  # Skip for large feature sets
            return None

        coefs = self.model.coef_[0]
        contributions = []

        for i in range(len(X_scaled)):
            contrib = {}
            for j, feature_name in enumerate(self.feature_names):
                contrib[feature_name] = float(coefs[j] * X_scaled[i, j])
            contributions.append(contrib)

        return contributions

    def _evaluate_predictions(
        self,
        predictions: list[WPModelPrediction],
        true_targets: np.ndarray,
        dataset_name: str,
    ) -> dict[str, float]:
        """Evaluate predictions using the evaluation framework."""
        if len(predictions) == 0:
            return {}

        # Extract probabilities
        raw_probs = np.array([p.raw_win_probability for p in predictions])
        cal_probs = np.array(
            [
                p.calibrated_win_probability
                for p in predictions
                if p.calibrated_win_probability is not None
            ]
        )

        # Use evaluation framework
        eval_metrics = self.evaluator.evaluate_model(
            predictions=cal_probs if len(cal_probs) > 0 else raw_probs,
            actual_outcomes=true_targets,
            prediction_type="binary",
            model_name=f"WP_{dataset_name}",
        )

        # Extract relevant metrics with dataset prefix
        metrics = {}
        for key, value in eval_metrics.core_metrics.items():
            metrics[f"{dataset_name}_{key}"] = value

        for key, value in eval_metrics.calibration_metrics.items():
            metrics[f"{dataset_name}_cal_{key}"] = value

        return metrics

    def run_walk_forward_validation(
        self,
        games_df: pd.DataFrame,
        start_season: int = 2018,
        end_season: int = 2024,
        feature_path: str | None = None,
    ) -> dict[str, Any]:
        """
        Run walk-forward validation on historical data.

        Args:
            games_df: DataFrame with historical games
            start_season: First season for validation
            end_season: Last season for validation
            feature_path: Path to pre-built features

        Returns:
            Dictionary with walk-forward validation results
        """
        self.logger.info(
            "Starting WP model walk-forward validation",
            start_season=start_season,
            end_season=end_season,
        )

        validator = WalkForwardValidator(min_train_seasons=2)

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
            self.logger.info(
                f"Validating WP model for season {test_season}",
                train_seasons=split.train_seasons,
                train_games=len(split.train_data),
                test_games=len(split.test_data),
            )

            try:
                # Create season-specific model
                season_model = WinProbabilityModel(
                    feature_selection_method=self.feature_selection_method,
                    max_features=self.max_features,
                    regularization_type=self.regularization_type,
                    use_calibration=self.use_calibration,
                    hyperparameter_tuning=self.hyperparameter_tuning,
                    random_state=self.random_state,
                )

                # Prepare training data
                train_data_full = split.train_data.copy()
                train_data_full["home_wins"] = split.train_targets

                # Train on this split
                season_model.train_model(train_data_full, feature_path=feature_path)

                # Generate test predictions
                test_data_full = split.test_data.copy()
                test_predictions = season_model.predict(
                    test_data_full, feature_path=feature_path
                )

                # Evaluate predictions
                season_metrics = season_model._evaluate_predictions(
                    test_predictions, split.test_targets, f"season_{test_season}"
                )

                validation_results["season_results"][test_season] = {
                    "predictions": test_predictions,
                    "metrics": season_metrics,
                    "model_info": {
                        "features_used": len(season_model.feature_names),
                        "feature_names": season_model.feature_names,
                        "hyperparameters": season_model.hyperparameters,
                    },
                }

                # Collect for overall metrics
                pred_probs = [
                    p.calibrated_win_probability or p.raw_win_probability
                    for p in test_predictions
                ]
                all_predictions.extend(pred_probs)
                all_actuals.extend(split.test_targets)

                accuracy_key = f"season_{test_season}_accuracy"
                if accuracy_key in season_metrics:
                    season_accuracies.append(season_metrics[accuracy_key])

                self.logger.info(
                    f"Season {test_season} WP validation completed",
                    accuracy=season_metrics.get(accuracy_key, 0),
                )

            except (
                ValueError,
                KeyError,
                TypeError,
                RuntimeError,
                np.linalg.LinAlgError,
            ) as e:
                self.logger.error(
                    f"Season {test_season} WP validation failed", error=str(e)
                )
                validation_results["season_results"][test_season] = {
                    "error": str(e),
                    "train_seasons": split.train_seasons,
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

            # Model stability
            if len(season_accuracies) > 1:
                validation_results["model_stability"] = {
                    "accuracy_mean": np.mean(season_accuracies),
                    "accuracy_std": np.std(season_accuracies),
                    "accuracy_min": np.min(season_accuracies),
                    "accuracy_max": np.max(season_accuracies),
                }

        self.logger.info(
            "WP model walk-forward validation completed",
            overall_accuracy=validation_results["overall_metrics"].get(
                "overall_accuracy", 0
            ),
            seasons_validated=validation_results["overall_metrics"].get(
                "seasons_validated", 0
            ),
        )

        return validation_results

    def save_model(self, save_path: str):
        """Save trained model to disk."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")

        model_data = {
            "model": self.model,
            "scaler": self.scaler,
            "feature_selector": self.feature_selector,
            "calibrator": self.calibrator,
            "feature_names": self.feature_names,
            "feature_importances": self.feature_importances,
            "model_metadata": self.model_metadata,
            "training_history": self.training_history,
            "hyperparameters": self.hyperparameters,
            "config": {
                "feature_selection_method": self.feature_selection_method,
                "max_features": self.max_features,
                "regularization_type": self.regularization_type,
                "use_calibration": self.use_calibration,
                "validation_method": self.validation_method,
                "hyperparameter_tuning": self.hyperparameter_tuning,
                "random_state": self.random_state,
            },
        }

        # Ensure directory exists
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(model_data, save_path)
        self.logger.info(f"WP model saved to {save_path}")

    def load_model(self, load_path: str):
        """Load trained model from disk."""
        model_data = joblib.load(load_path)

        self.model = model_data["model"]
        self.scaler = model_data["scaler"]
        self.feature_selector = model_data["feature_selector"]
        self.calibrator = model_data["calibrator"]
        self.feature_names = model_data["feature_names"]
        self.feature_importances = model_data["feature_importances"]
        self.model_metadata = model_data["model_metadata"]
        self.training_history = model_data["training_history"]
        self.hyperparameters = model_data["hyperparameters"]

        # Restore configuration
        config = model_data["config"]
        self.feature_selection_method = config["feature_selection_method"]
        self.max_features = config["max_features"]
        self.regularization_type = config["regularization_type"]
        self.use_calibration = config["use_calibration"]
        self.validation_method = config["validation_method"]
        self.hyperparameter_tuning = config["hyperparameter_tuning"]
        self.random_state = config["random_state"]

        self.is_trained = True
        self.logger.info(f"WP model loaded from {load_path}")

    def get_model_summary(self) -> dict[str, Any]:
        """Get comprehensive model summary."""
        if not self.is_trained:
            return {"is_trained": False, "error": "Model not trained"}

        summary = {
            "model_type": "WinProbabilityModel",
            "is_trained": self.is_trained,
            "configuration": {
                "feature_selection_method": self.feature_selection_method,
                "max_features": self.max_features,
                "regularization_type": self.regularization_type,
                "use_calibration": self.use_calibration,
                "hyperparameter_tuning": self.hyperparameter_tuning,
            },
            "model_details": {
                "features_selected": len(self.feature_names),
                "feature_names": self.feature_names,
                "hyperparameters": self.hyperparameters,
                "feature_importances": dict(
                    sorted(
                        self.feature_importances.items(),
                        key=lambda x: x[1],
                        reverse=True,
                    )[:10]
                ),
            },
            "training_info": self.training_history,
            "calibration_used": self.calibrator is not None,
        }

        if self.model_metadata:
            summary["performance_metrics"] = self.model_metadata.performance_metrics

        return summary


def main():
    """Main function for command-line usage."""
    import argparse
    from pathlib import Path

    from utils.date_utils import get_current_nfl_season, get_current_nfl_week

    parser = argparse.ArgumentParser(description="Train Win Probability Model")
    parser.add_argument("--season", type=int, help="Target season (default: current)")
    parser.add_argument(
        "--week", help="Target week (default: current, 'all' for full season)"
    )

    args = parser.parse_args()

    # Determine season and week
    season = args.season or get_current_nfl_season()
    if args.week == "all":
        week = None
    else:
        week = int(args.week) if args.week else get_current_nfl_week()

    # Load features from gold layer (standard pipeline path)
    try:
        features_df = load_dataframe("features_wp", layer="gold")

        # Filter for target season/week
        if season:
            features_df = features_df[features_df["season"] == season]
        if week:
            features_df = features_df[features_df["week"] == week]

        logger.info(
            "Loaded WP features", season=season, week=week, records=len(features_df)
        )
    except (FileNotFoundError, OSError, ValueError, KeyError) as e:
        logger.error(f"Failed to load features: {e}")
        return

    # Initialize model with pipeline defaults
    model = WinProbabilityModel(
        feature_selection_method="recursive",
        regularization_type="elasticnet",
        use_calibration=True,
        hyperparameter_tuning="grid_search",
    )

    # Always do both: Train first, then validate
    model.train_model(features_df)

    # Save to standard artifacts location
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    model_path = artifacts_dir / "wp_model.pkl"
    model.save_model(str(model_path))

    # Load the saved model and run validation
    validation_model = WinProbabilityModel()
    validation_model.load_model(str(model_path))

    validation_model.run_walk_forward_validation(features_df)

    logger.info("WP model training completed successfully")


if __name__ == "__main__":
    main()
