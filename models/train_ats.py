#!/usr/bin/env python3
"""
Against The Spread (ATS) Model Training

This module implements a comprehensive ATS model for NFL predictions:
- XGBoost/LightGBM regression for expected margin prediction
- Conversion to cover probability using residual distribution
- Both classification and regression-to-margin approaches
- Proper spread betting mechanics handling
- Feature importance tracking and analysis
- Walk-forward training protocol for temporal data
- Integrated probability calibration
- Model persistence and loading with versioning

The ATS model predicts point spreads and cover probabilities, integrating
with the broader prediction pipeline for comprehensive game analysis.
"""

import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ML imports
import xgboost as xgb

try:
    import lightgbm as lgb

    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.feature_selection import (
    RFE,
    SelectFromModel,
    SelectKBest,
    VarianceThreshold,
    f_classif,
    f_regression,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    RandomizedSearchCV,
)
from sklearn.preprocessing import RobustScaler

from data.storage import load_dataframe
from models.calibrate import CalibrationResults, ProbabilityCalibrator
from models.evaluation import ModelEvaluationFramework

# Project imports
from models.utils import ModelManager, WalkForwardValidator
from utils import get_logger

logger = get_logger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass
class ATSModelPrediction:
    """
    Container for ATS model prediction results.

    Attributes:
        game_id: Unique game identifier
        home_team: Home team abbreviation
        away_team: Away team abbreviation
        predicted_margin: Expected point margin (positive = home favored)
        predicted_spread: Predicted spread line
        cover_probability: Probability home team covers the spread
        classification_cover_prob: Classification approach cover probability
        regression_cover_prob: Regression approach cover probability
        market_spread: Market spread at prediction time
        edge: Difference between model and market probabilities
        confidence: Model confidence in prediction
        feature_importances: Feature contribution to prediction
        model_version: Version of model used
        prediction_date: When prediction was made
        metadata: Additional prediction metadata
    """

    game_id: str
    home_team: str
    away_team: str
    predicted_margin: float
    predicted_spread: float
    cover_probability: float
    classification_cover_prob: float | None = None
    regression_cover_prob: float | None = None
    market_spread: float | None = None
    edge: float | None = None
    confidence: float | None = None
    feature_importances: dict[str, float] | None = None
    model_version: str | None = None
    prediction_date: datetime | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class ATSModelResults:
    """
    Container for ATS model training and validation results.

    Attributes:
        regression_model: Trained margin regression model
        classification_model: Trained cover classification model
        scaler: Fitted feature scaler
        feature_selector: Trained feature selector
        calibrator: Trained probability calibrator
        residual_std: Standard deviation of training residuals
        feature_names: List of selected feature names
        feature_importances: Dictionary of feature importances
        performance_metrics: Training and validation metrics
        hyperparameters: Final hyperparameters used
        training_history: Training progression data
        calibration_results: Probability calibration results
        model_version: Model version identifier
        training_date: When model was trained
        metadata: Additional model metadata
    """

    regression_model: Any
    classification_model: Any | None = None
    scaler: Any | None = None
    feature_selector: Any | None = None
    calibrator: ProbabilityCalibrator | None = None
    residual_std: float | None = None
    feature_names: list[str] = field(default_factory=list)
    feature_importances: dict[str, float] = field(default_factory=dict)
    performance_metrics: dict[str, float] = field(default_factory=dict)
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    training_history: dict[str, list[float]] = field(default_factory=dict)
    calibration_results: CalibrationResults | None = None
    model_version: str = "1.0.0"
    training_date: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class ResidualDistributionConverter:
    """
    Converts margin predictions to cover probabilities using residual distribution.

    This class learns the distribution of prediction residuals during training
    and uses it to convert point margin predictions to cover probabilities.
    """

    def __init__(self, distribution_type: str = "normal"):
        """
        Initialize the converter.

        Args:
            distribution_type: Type of distribution to fit ("normal", "t", "skewnorm")
        """
        self.distribution_type = distribution_type
        self.distribution_params = None
        self.residual_std = None
        self.is_fitted = False

    def fit(self, residuals: np.ndarray) -> None:
        """
        Fit the residual distribution.

        Args:
            residuals: Array of prediction residuals (actual - predicted)
        """
        self.residual_std = np.std(residuals)

        if self.distribution_type == "normal":
            self.distribution_params = {
                "loc": np.mean(residuals),
                "scale": np.std(residuals),
            }
        elif self.distribution_type == "t":
            from scipy.stats import t

            df, loc, scale = t.fit(residuals)
            self.distribution_params = {"df": df, "loc": loc, "scale": scale}
        elif self.distribution_type == "skewnorm":
            from scipy.stats import skewnorm

            a, loc, scale = skewnorm.fit(residuals)
            self.distribution_params = {"a": a, "loc": loc, "scale": scale}
        else:
            raise ValueError(f"Unsupported distribution type: {self.distribution_type}")

        self.is_fitted = True
        logger.info(
            f"Fitted {self.distribution_type} distribution with params: {self.distribution_params}"
        )

    def predict_cover_probability(
        self, predicted_margins: np.ndarray, spreads: np.ndarray
    ) -> np.ndarray:
        """
        Convert margin predictions to cover probabilities.

        Args:
            predicted_margins: Predicted point margins
            spreads: Market spreads (negative = home team favored)

        Returns:
            Array of cover probabilities
        """
        if not self.is_fitted:
            raise ValueError("Must fit the residual distribution first")

        # Calculate the margin needed to cover
        margin_needed = -spreads  # Convert spread to margin needed

        if self.distribution_type == "normal":
            from scipy.stats import norm

            loc = self.distribution_params["loc"]
            scale = self.distribution_params["scale"]

            # Probability that actual margin > margin_needed
            z_scores = (margin_needed - predicted_margins - loc) / scale
            cover_probs = 1 - norm.cdf(z_scores)

        elif self.distribution_type == "t":
            from scipy.stats import t

            df = self.distribution_params["df"]
            loc = self.distribution_params["loc"]
            scale = self.distribution_params["scale"]

            t_scores = (margin_needed - predicted_margins - loc) / scale
            cover_probs = 1 - t.cdf(t_scores, df)

        elif self.distribution_type == "skewnorm":
            from scipy.stats import skewnorm

            a = self.distribution_params["a"]
            loc = self.distribution_params["loc"]
            scale = self.distribution_params["scale"]

            z_scores = (margin_needed - predicted_margins - loc) / scale
            cover_probs = 1 - skewnorm.cdf(z_scores, a)

        return np.clip(cover_probs, 0.001, 0.999)


class ATSModel(BaseEstimator):
    """
    Comprehensive Against The Spread (ATS) model for NFL predictions.

    This model combines regression and classification approaches to predict
    both point margins and cover probabilities, with proper handling of
    spread betting mechanics.
    """

    def __init__(
        self,
        model_type: str = "xgboost",
        approach: str = "hybrid",
        feature_selection_method: str = "model_based",
        max_features: int | None = 20,
        regularization_strength: float = 0.1,
        use_calibration: bool = True,
        hyperparameter_tuning: str = "grid_search",
        distribution_type: str = "normal",
        random_state: int = 42,
    ):
        """
        Initialize the ATS model.

        Args:
            model_type: Type of base model ("xgboost", "lightgbm", "random_forest", "gradient_boosting")
            approach: Modeling approach ("regression", "classification", "hybrid")
            feature_selection_method: Method for feature selection
            max_features: Maximum number of features to select
            regularization_strength: Strength of regularization
            use_calibration: Whether to use probability calibration
            hyperparameter_tuning: Type of hyperparameter tuning
            distribution_type: Distribution for residual conversion
            random_state: Random state for reproducibility
        """
        self.model_type = model_type
        self.approach = approach
        self.feature_selection_method = feature_selection_method
        self.max_features = max_features
        self.regularization_strength = regularization_strength
        self.use_calibration = use_calibration
        self.hyperparameter_tuning = hyperparameter_tuning
        self.distribution_type = distribution_type
        self.random_state = random_state

        # Model components
        self.regression_model = None
        self.classification_model = None
        self.scaler = None
        self.feature_selector = None
        self.calibrator = None
        self.residual_converter = None

        # Training results
        self.feature_names = []
        self.feature_importances = {}
        self.residual_std = None
        self.is_trained = False

        # Validation
        self.walk_forward_validator = WalkForwardValidator()
        self.model_manager = ModelManager()
        self.evaluation_framework = ModelEvaluationFramework()

    def _create_base_model(self, task_type: str = "regression") -> Any:
        """Create the base model based on configuration."""
        if self.model_type == "xgboost":
            if task_type == "regression":
                return xgb.XGBRegressor(
                    random_state=self.random_state, n_jobs=-1, verbosity=0
                )
            return xgb.XGBClassifier(
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=0,
                eval_metric="logloss",
            )
        if self.model_type == "lightgbm" and LIGHTGBM_AVAILABLE:
            if task_type == "regression":
                return lgb.LGBMRegressor(
                    random_state=self.random_state, n_jobs=-1, verbosity=-1
                )
            return lgb.LGBMClassifier(
                random_state=self.random_state, n_jobs=-1, verbosity=-1
            )
        if self.model_type == "random_forest":
            if task_type == "regression":
                return RandomForestRegressor(random_state=self.random_state, n_jobs=-1)
            from sklearn.ensemble import RandomForestClassifier

            return RandomForestClassifier(random_state=self.random_state, n_jobs=-1)
        if self.model_type == "gradient_boosting":
            if task_type == "regression":
                return GradientBoostingRegressor(random_state=self.random_state)
            from sklearn.ensemble import GradientBoostingClassifier

            return GradientBoostingClassifier(random_state=self.random_state)
        if task_type == "regression":
            return Ridge(random_state=self.random_state)
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(random_state=self.random_state)

    def _prepare_features(
        self, data: pd.DataFrame
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """
        Prepare features and targets for ATS modeling.

        Args:
            data: Input dataframe with game data

        Returns:
            Tuple of (features, margin_targets, cover_targets)
        """
        # Feature columns (exclude targets and identifiers)
        exclude_cols = [
            "game_id",
            "season",
            "week",
            "home_team",
            "away_team",
            "home_wins",
            "actual_margin",
            "covers_spread",
            "market_spread",
            "home_score",
            "away_score",
        ]

        feature_cols = [col for col in data.columns if col not in exclude_cols]
        X = data[feature_cols].copy()

        # Handle missing values
        # Ensure all feature columns are numeric
        X = X.select_dtypes(include=[np.number])
        X = X.fillna(X.median())

        # Create targets
        margin_targets = (
            data["actual_margin"].values if "actual_margin" in data.columns else None
        )
        cover_targets = (
            data["covers_spread"].values if "covers_spread" in data.columns else None
        )

        return X, margin_targets, cover_targets

    def select_features(
        self, X: pd.DataFrame, y: np.ndarray, task_type: str = "regression"
    ) -> tuple[Any, list[str]]:
        """
        Select features using the specified method.

        Args:
            X: Feature matrix
            y: Target variable
            task_type: Type of task ("regression" or "classification")

        Returns:
            Tuple of (fitted_selector, selected_feature_names)
        """
        feature_cols = X.columns.tolist()

        # Determine max features
        max_features = min(self.max_features or len(feature_cols), len(feature_cols))

        if self.feature_selection_method == "variance":
            selector = VarianceThreshold(threshold=0.01)
            selector.fit_transform(X)
            selected_features = [
                col for i, col in enumerate(feature_cols) if selector.get_support()[i]
            ]

        elif self.feature_selection_method == "univariate":
            score_func = f_regression if task_type == "regression" else f_classif
            selector = SelectKBest(
                score_func=score_func, k=min(max_features, len(feature_cols))
            )
            selector.fit(X, y)
            selected_features = [
                feature_cols[i] for i in selector.get_support(indices=True)
            ]

        elif self.feature_selection_method == "model_based":
            base_model = self._create_base_model(task_type)
            selector = SelectFromModel(base_model, max_features=max_features)
            selector.fit(X, y)
            selected_features = [
                feature_cols[i] for i in selector.get_support(indices=True)
            ]

        elif self.feature_selection_method == "recursive":
            base_model = self._create_base_model(task_type)
            selector = RFE(base_model, n_features_to_select=max_features)
            selector.fit(X, y)
            selected_features = [
                feature_cols[i] for i in selector.get_support(indices=True)
            ]

        else:
            # Use all features
            selector = None
            selected_features = feature_cols[:max_features]

        logger.info(
            f"Selected {len(selected_features)} features using {self.feature_selection_method}"
        )
        return selector, selected_features

    def tune_hyperparameters(
        self, X: np.ndarray, y: np.ndarray, task_type: str = "regression"
    ) -> dict[str, Any]:
        """
        Tune hyperparameters using the specified method.

        Args:
            X: Feature matrix
            y: Target variable
            task_type: Type of task ("regression" or "classification")

        Returns:
            Best hyperparameters found
        """
        base_model = self._create_base_model(task_type)

        # Define parameter grids based on model type
        if self.model_type == "xgboost":
            if task_type == "regression":
                param_grid = {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [3, 4, 5, 6],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "subsample": [0.8, 0.9, 1.0],
                    "colsample_bytree": [0.8, 0.9, 1.0],
                    "reg_alpha": [0, 0.1, 0.5],
                    "reg_lambda": [1, 1.5, 2],
                }
            else:
                param_grid = {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [3, 4, 5, 6],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "subsample": [0.8, 0.9, 1.0],
                    "colsample_bytree": [0.8, 0.9, 1.0],
                    "reg_alpha": [0, 0.1, 0.5],
                    "reg_lambda": [1, 1.5, 2],
                }
        elif self.model_type == "lightgbm" and LIGHTGBM_AVAILABLE:
            param_grid = {
                "n_estimators": [100, 200, 300],
                "max_depth": [3, 4, 5, 6],
                "learning_rate": [0.01, 0.1, 0.2],
                "subsample": [0.8, 0.9, 1.0],
                "colsample_bytree": [0.8, 0.9, 1.0],
                "reg_alpha": [0, 0.1, 0.5],
                "reg_lambda": [1, 1.5, 2],
            }
        elif self.model_type == "random_forest":
            param_grid = {
                "n_estimators": [100, 200, 300],
                "max_depth": [5, 10, 15, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 4],
                "max_features": ["sqrt", "log2", None],
            }
        else:
            # Simple parameter grid for other models
            param_grid = (
                {"alpha": [0.1, 1.0, 10.0]}
                if task_type == "regression"
                else {"C": [0.1, 1.0, 10.0]}
            )

        # Choose search method
        if self.hyperparameter_tuning == "grid_search":
            search = GridSearchCV(
                base_model,
                param_grid,
                cv=5,
                scoring="neg_mean_squared_error"
                if task_type == "regression"
                else "neg_log_loss",
                n_jobs=-1,
                verbose=0,
            )
        else:
            # Random search
            search = RandomizedSearchCV(
                base_model,
                param_grid,
                n_iter=50,
                cv=5,
                scoring="neg_mean_squared_error"
                if task_type == "regression"
                else "neg_log_loss",
                n_jobs=-1,
                verbose=0,
                random_state=self.random_state,
            )

        search.fit(X, y)
        logger.info(f"Best {task_type} hyperparameters: {search.best_params_}")

        return search.best_params_

    def train_model(
        self,
        training_data: pd.DataFrame,
        validation_data: pd.DataFrame | None = None,
    ) -> ATSModelResults:
        """
        Train the ATS model using the specified approach.

        Args:
            training_data: Training dataset
            validation_data: Optional validation dataset

        Returns:
            ATSModelResults containing all training artifacts
        """
        logger.info("Starting ATS model training...")

        # Prepare features and targets
        X, margin_targets, cover_targets = self._prepare_features(training_data)

        if margin_targets is None:
            raise ValueError("Training data must include 'actual_margin' column")

        # Initialize scaler
        self.scaler = RobustScaler()
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X), columns=X.columns, index=X.index
        )

        # Train regression model (always needed)
        logger.info("Training margin regression model...")

        # Feature selection for regression
        self.feature_selector, self.feature_names = self.select_features(
            X_scaled, margin_targets, "regression"
        )

        X_selected = X_scaled[self.feature_names]

        # Hyperparameter tuning for regression
        if self.hyperparameter_tuning != "none":
            best_params = self.tune_hyperparameters(
                X_selected, margin_targets, "regression"
            )
        else:
            best_params = {}

        # Create and train regression model
        self.regression_model = self._create_base_model("regression")
        self.regression_model.set_params(**best_params)
        self.regression_model.fit(X_selected, margin_targets)

        # Calculate residuals and fit distribution
        margin_predictions = self.regression_model.predict(X_selected)
        residuals = margin_targets - margin_predictions
        self.residual_std = np.std(residuals)

        self.residual_converter = ResidualDistributionConverter(self.distribution_type)
        self.residual_converter.fit(residuals)

        # Train classification model if needed
        classification_model = None
        if self.approach in ["classification", "hybrid"] and cover_targets is not None:
            logger.info("Training cover classification model...")

            # Use same features for classification
            class_params = (
                self.tune_hyperparameters(X_selected, cover_targets, "classification")
                if self.hyperparameter_tuning != "none"
                else {}
            )

            classification_model = self._create_base_model("classification")
            classification_model.set_params(**class_params)
            classification_model.fit(X_selected, cover_targets)

        # Calculate feature importances
        if hasattr(self.regression_model, "feature_importances_"):
            self.feature_importances = dict(
                zip(
                    self.feature_names,
                    self.regression_model.feature_importances_,
                    strict=False,
                )
            )
        elif hasattr(self.regression_model, "coef_"):
            self.feature_importances = dict(
                zip(
                    self.feature_names,
                    np.abs(self.regression_model.coef_),
                    strict=False,
                )
            )

        # Probability calibration
        calibrator = None
        calibration_results = None

        if self.use_calibration and cover_targets is not None:
            logger.info("Training probability calibration...")

            # Generate cover probabilities for calibration
            if "market_spread" in training_data.columns:
                spreads = training_data["market_spread"].values
                cover_probs = self.residual_converter.predict_cover_probability(
                    margin_predictions, spreads
                )

                calibrator = ProbabilityCalibrator(primary_method="isotonic")
                calibration_results = calibrator.calibrate_probabilities(
                    cover_probs, cover_targets
                )
                self.calibrator = calibrator
                self.trained_calibrator = calibration_results.calibrator

        # Calculate training metrics
        train_mae = mean_absolute_error(margin_targets, margin_predictions)
        train_rmse = np.sqrt(mean_squared_error(margin_targets, margin_predictions))
        train_r2 = r2_score(margin_targets, margin_predictions)

        performance_metrics = {
            "training_mae": train_mae,
            "training_rmse": train_rmse,
            "training_r2": train_r2,
            "training_residual_std": self.residual_std,
        }

        if cover_targets is not None and "market_spread" in training_data.columns:
            spreads = training_data["market_spread"].values
            cover_probs = self.residual_converter.predict_cover_probability(
                margin_predictions, spreads
            )

            if calibrator and hasattr(self, "trained_calibrator"):
                cover_probs = calibrator.apply_calibration(
                    cover_probs, self.trained_calibrator
                )

            from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

            performance_metrics.update(
                {
                    "training_cover_accuracy": accuracy_score(
                        cover_targets, cover_probs > 0.5
                    ),
                    "training_cover_log_loss": log_loss(cover_targets, cover_probs),
                    "training_cover_brier_score": brier_score_loss(
                        cover_targets, cover_probs
                    ),
                }
            )

        # Validation metrics
        if validation_data is not None:
            logger.info("Calculating validation metrics...")
            # Temporarily set is_trained to allow validation predictions
            was_trained = self.is_trained
            self.is_trained = True
            val_predictions = self.predict(validation_data)
            self.is_trained = was_trained

            if "actual_margin" in validation_data.columns:
                val_margins = validation_data["actual_margin"].values
                pred_margins = np.array([p.predicted_margin for p in val_predictions])

                performance_metrics.update(
                    {
                        "validation_mae": mean_absolute_error(
                            val_margins, pred_margins
                        ),
                        "validation_rmse": np.sqrt(
                            mean_squared_error(val_margins, pred_margins)
                        ),
                        "validation_r2": r2_score(val_margins, pred_margins),
                    }
                )

            if "covers_spread" in validation_data.columns:
                val_covers = validation_data["covers_spread"].values
                pred_covers = np.array([p.cover_probability for p in val_predictions])

                performance_metrics.update(
                    {
                        "validation_cover_accuracy": accuracy_score(
                            val_covers, pred_covers > 0.5
                        ),
                        "validation_cover_log_loss": log_loss(val_covers, pred_covers),
                        "validation_cover_brier_score": brier_score_loss(
                            val_covers, pred_covers
                        ),
                    }
                )

        self.is_trained = True

        # Create results container
        results = ATSModelResults(
            regression_model=self.regression_model,
            classification_model=classification_model,
            scaler=self.scaler,
            feature_selector=self.feature_selector,
            calibrator=calibrator,
            residual_std=self.residual_std,
            feature_names=self.feature_names,
            feature_importances=self.feature_importances,
            performance_metrics=performance_metrics,
            hyperparameters=best_params,
            calibration_results=calibration_results,
            training_date=datetime.now(),
            metadata={
                "model_type": self.model_type,
                "approach": self.approach,
                "feature_selection_method": self.feature_selection_method,
                "training_samples": len(training_data),
                "distribution_type": self.distribution_type,
            },
        )

        logger.info(
            f"ATS model training complete. Training MAE: {train_mae:.3f}, RMSE: {train_rmse:.3f}"
        )
        return results

    def predict(
        self, data: pd.DataFrame, include_all_approaches: bool = True
    ) -> list[ATSModelPrediction]:
        """
        Make ATS predictions on new data.

        Args:
            data: Input data for prediction
            include_all_approaches: Whether to include both regression and classification approaches

        Returns:
            List of ATSModelPrediction objects
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Prepare features
        X, _, _ = self._prepare_features(data)

        # Handle missing features by adding them with median values
        missing_features = set(self.feature_names) - set(X.columns)
        if missing_features:
            logger.warning(f"Missing features in prediction data: {missing_features}")
            for feature in missing_features:
                X[feature] = 0.0  # Use default value

        # Reorder columns to match training
        X = X[self.feature_names]

        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.transform(X), columns=X.columns, index=X.index
        )

        # Select features
        X_selected = X_scaled[self.feature_names]

        # Make margin predictions
        margin_predictions = self.regression_model.predict(X_selected)

        # Create predictions
        predictions = []

        for i, (_idx, row) in enumerate(data.iterrows()):
            game_id = row.get("game_id", f"game_{i}")
            home_team = row.get("home_team", "HOME")
            away_team = row.get("away_team", "AWAY")
            predicted_margin = margin_predictions[i]
            predicted_spread = -predicted_margin  # Convert margin to spread

            # Get market spread if available
            market_spread = row.get("market_spread", None)

            # Calculate cover probability using residual distribution
            if market_spread is not None:
                cover_prob = self.residual_converter.predict_cover_probability(
                    np.array([predicted_margin]), np.array([market_spread])
                )[0]

                # Apply calibration if available
                if self.calibrator and hasattr(self, "trained_calibrator"):
                    cover_prob = self.calibrator.apply_calibration(
                        np.array([cover_prob]), self.trained_calibrator
                    )[0]

                # Calculate edge
                implied_prob = 0.5  # Default for even spread
                edge = cover_prob - implied_prob
            else:
                cover_prob = 0.5
                edge = None

            # Classification approach cover probability
            classification_cover_prob = None
            if self.classification_model and include_all_approaches:
                classification_cover_prob = self.classification_model.predict_proba(
                    X_selected.iloc[[i]]
                )[0][1]

            # Feature importances for this prediction
            feature_importances = None
            if hasattr(self.regression_model, "feature_importances_"):
                feature_importances = dict(
                    zip(
                        self.feature_names,
                        self.regression_model.feature_importances_,
                        strict=False,
                    )
                )

            # Calculate confidence
            confidence = abs(cover_prob - 0.5) * 2

            prediction = ATSModelPrediction(
                game_id=game_id,
                home_team=home_team,
                away_team=away_team,
                predicted_margin=predicted_margin,
                predicted_spread=predicted_spread,
                cover_probability=cover_prob,
                classification_cover_prob=classification_cover_prob,
                regression_cover_prob=cover_prob,
                market_spread=market_spread,
                edge=edge,
                confidence=confidence,
                feature_importances=feature_importances,
                model_version=getattr(self, "model_version", "1.0.0"),
                prediction_date=datetime.now(),
            )

            predictions.append(prediction)

        return predictions

    def run_walk_forward_validation(
        self, games_df: pd.DataFrame, start_season: int, end_season: int
    ) -> dict[str, Any]:
        """
        Run walk-forward validation across multiple seasons.

        Args:
            games_df: Complete games dataset
            start_season: First season to validate
            end_season: Last season to validate

        Returns:
            Dictionary containing validation results
        """
        logger.info(
            f"Running walk-forward validation from {start_season} to {end_season}"
        )

        results = {
            "season_results": [],
            "overall_metrics": {},
            "feature_importance_evolution": [],
        }

        all_predictions = []
        all_actuals_margin = []
        all_actuals_cover = []

        for season in range(start_season, end_season + 1):
            logger.info(f"Validating season {season}")

            # Split data
            train_data = games_df[games_df["season"] < season].copy()
            test_data = games_df[games_df["season"] == season].copy()

            if len(train_data) == 0 or len(test_data) == 0:
                logger.warning(f"Insufficient data for season {season}, skipping")
                continue

            # Train model for this season
            model_results = self.train_model(train_data, test_data)

            # Make predictions
            predictions = self.predict(test_data)

            # Calculate metrics for this season
            if "actual_margin" in test_data.columns:
                actual_margins = test_data["actual_margin"].values
                pred_margins = np.array([p.predicted_margin for p in predictions])

                season_mae = mean_absolute_error(actual_margins, pred_margins)
                season_rmse = np.sqrt(mean_squared_error(actual_margins, pred_margins))
                season_r2 = r2_score(actual_margins, pred_margins)

                all_predictions.extend(pred_margins)
                all_actuals_margin.extend(actual_margins)
            else:
                season_mae = season_rmse = season_r2 = None

            if "covers_spread" in test_data.columns:
                actual_covers = test_data["covers_spread"].values
                pred_covers = np.array([p.cover_probability for p in predictions])

                season_accuracy = accuracy_score(actual_covers, pred_covers > 0.5)
                season_log_loss = log_loss(actual_covers, pred_covers)
                season_brier = brier_score_loss(actual_covers, pred_covers)

                all_actuals_cover.extend(actual_covers)
            else:
                season_accuracy = season_log_loss = season_brier = None

            season_result = {
                "season": season,
                "train_games": len(train_data),
                "test_games": len(test_data),
                "mae": season_mae,
                "rmse": season_rmse,
                "r2": season_r2,
                "cover_accuracy": season_accuracy,
                "cover_log_loss": season_log_loss,
                "cover_brier_score": season_brier,
                "feature_importances": model_results.feature_importances.copy(),
            }

            results["season_results"].append(season_result)

            if model_results.feature_importances:
                results["feature_importance_evolution"].append(
                    {
                        "season": season,
                        "importances": model_results.feature_importances.copy(),
                    }
                )

        # Calculate overall metrics
        if all_actuals_margin:
            overall_mae = mean_absolute_error(all_actuals_margin, all_predictions)
            overall_rmse = np.sqrt(
                mean_squared_error(all_actuals_margin, all_predictions)
            )
            overall_r2 = r2_score(all_actuals_margin, all_predictions)

            results["overall_metrics"].update(
                {
                    "overall_mae": overall_mae,
                    "overall_rmse": overall_rmse,
                    "overall_r2": overall_r2,
                    "seasons_validated": len(results["season_results"]),
                }
            )

        if all_actuals_cover:
            all_pred_covers = np.array(
                [
                    p.cover_probability
                    for season_preds in [
                        self.predict(games_df[games_df["season"] == s])
                        for s in range(start_season, end_season + 1)
                    ]
                    for p in season_preds
                ]
            )[: len(all_actuals_cover)]

            overall_accuracy = accuracy_score(all_actuals_cover, all_pred_covers > 0.5)
            overall_log_loss = log_loss(all_actuals_cover, all_pred_covers)
            overall_brier = brier_score_loss(all_actuals_cover, all_pred_covers)

            results["overall_metrics"].update(
                {
                    "overall_cover_accuracy": overall_accuracy,
                    "overall_cover_log_loss": overall_log_loss,
                    "overall_cover_brier_score": overall_brier,
                }
            )

        logger.info(
            f"Walk-forward validation complete. Overall MAE: {results['overall_metrics'].get('overall_mae', 'N/A')}"
        )
        return results

    def get_model_summary(self) -> dict[str, Any]:
        """Get a comprehensive summary of the trained model."""
        if not self.is_trained:
            return {
                "model_type": "ATS Model",
                "is_trained": False,
                "status": "Not trained",
            }

        summary = {
            "model_type": "ATS Model",
            "is_trained": True,
            "approach": self.approach,
            "base_model_type": self.model_type,
            "distribution_type": self.distribution_type,
            "features_selected": len(self.feature_names),
            "residual_std": self.residual_std,
            "feature_importances": self.feature_importances,
            "configuration": {
                "feature_selection_method": self.feature_selection_method,
                "max_features": self.max_features,
                "use_calibration": self.use_calibration,
                "hyperparameter_tuning": self.hyperparameter_tuning,
            },
        }

        return summary

    def save_model(self, filepath: str) -> None:
        """Save the trained model to disk."""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        model_data = {
            "regression_model": self.regression_model,
            "classification_model": getattr(self, "classification_model", None),
            "scaler": self.scaler,
            "feature_selector": self.feature_selector,
            "calibrator": self.calibrator,
            "trained_calibrator": getattr(self, "trained_calibrator", None),
            "residual_converter": self.residual_converter,
            "feature_names": self.feature_names,
            "feature_importances": self.feature_importances,
            "residual_std": self.residual_std,
            "model_config": {
                "model_type": self.model_type,
                "approach": self.approach,
                "feature_selection_method": self.feature_selection_method,
                "max_features": self.max_features,
                "use_calibration": self.use_calibration,
                "distribution_type": self.distribution_type,
                "random_state": self.random_state,
            },
        }

        joblib.dump(model_data, filepath)
        logger.info(f"ATS model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load a trained model from disk."""
        model_data = joblib.load(filepath)

        self.regression_model = model_data["regression_model"]
        self.classification_model = model_data.get("classification_model")
        self.scaler = model_data["scaler"]
        self.feature_selector = model_data.get("feature_selector")
        self.calibrator = model_data.get("calibrator")
        self.trained_calibrator = model_data.get("trained_calibrator")
        self.residual_converter = model_data["residual_converter"]
        self.feature_names = model_data["feature_names"]
        self.feature_importances = model_data["feature_importances"]
        self.residual_std = model_data["residual_std"]

        # Restore configuration
        config = model_data["model_config"]
        self.model_type = config["model_type"]
        self.approach = config["approach"]
        self.feature_selection_method = config["feature_selection_method"]
        self.max_features = config["max_features"]
        self.use_calibration = config["use_calibration"]
        self.distribution_type = config["distribution_type"]
        self.random_state = config["random_state"]

        self.is_trained = True
        logger.info(f"ATS model loaded from {filepath}")


def main():
    """Main function for command-line usage."""
    import argparse
    from pathlib import Path

    from utils.date_utils import get_current_nfl_season, get_current_nfl_week

    parser = argparse.ArgumentParser(description="Train Against the Spread (ATS) Model")
    parser.add_argument("--season", type=int, help="Target season (default: current)")
    parser.add_argument(
        "--week", help="Target week (default: current, 'all' for full season)"
    )
    parser.add_argument(
        "--no-market-anchors", action="store_true", help="Exclude betting line features"
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
        features_df = load_dataframe("features_ats", layer="gold")

        # COLD-02 / T-33-18: refuse a PROVISIONAL Elo row as a TRAINING input, HERE at
        # the gold-loading boundary and BEFORE any filtering. This trainer reads gold
        # DIRECTLY and never passes through features/elo_features.build_features, so a
        # guard placed only in the feature builder protects none of this path. Filtering
        # first would let a provisional row drop out of sight and still be trained on in
        # a different slice. The refusal is a RuntimeError subclass and is therefore NOT
        # caught by the surrounding handler tuple -- a refusal converted into a logged
        # `return` would be a silent no-train.
        from features.elo_features import assert_no_provisional_training_rows

        assert_no_provisional_training_rows(features_df, "train:ats")

        # Filter for target season/week
        if season:
            # Extract season from game_id (format: YYYY_WXX_TEAM@TEAM)
            features_df["season"] = features_df["game_id"].str[:4].astype(int)
            features_df = features_df[features_df["season"] == season]
        if week:
            features_df = features_df[features_df["week"] == week]

        logger.info(
            "Loaded ATS features", season=season, week=week, records=len(features_df)
        )
    except (FileNotFoundError, OSError, ValueError, KeyError) as e:
        logger.error(f"Failed to load features: {e}")
        return

    # Initialize model with pipeline defaults
    model = ATSModel(
        feature_selection_method="recursive",
        regularization_strength=0.1,
        use_calibration=True,
        hyperparameter_tuning="grid_search",
    )

    # Always do both: Train first, then validate
    model.train_model(features_df)

    # Save to standard artifacts location
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    model_path = artifacts_dir / "ats_model.pkl"
    model.save_model(str(model_path))

    # Load the saved model and run validation
    validation_model = ATSModel()
    validation_model.load_model(str(model_path))

    validation_model.run_walk_forward_validation(features_df)

    logger.info("ATS model training completed successfully")


if __name__ == "__main__":
    main()
