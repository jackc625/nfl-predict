#!/usr/bin/env python3
"""
Over/Under (O/U) Model Training

This module implements a comprehensive Over/Under model for NFL predictions:
- XGBoost/LightGBM regression for total points prediction
- Conversion to over/under probabilities using residual distribution
- Optional Poisson score model for total points simulation
- Weather impact modeling specifically for totals
- Proper total betting mechanics handling
- Feature importance tracking and analysis
- Walk-forward training protocol for temporal data
- Integrated probability calibration
- Model persistence and loading with versioning

The O/U model predicts game totals and over/under probabilities, integrating
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import (
    RFE,
    SelectFromModel,
    SelectKBest,
    VarianceThreshold,
    f_regression,
)
from sklearn.linear_model import (
    LinearRegression,
    PoissonRegressor,
    Ridge,
)
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
class OUModelPrediction:
    """
    Container for O/U model prediction results.

    Attributes:
        game_id: Unique game identifier
        home_team: Home team abbreviation
        away_team: Away team abbreviation
        predicted_total: Expected total points
        over_probability: Probability total goes over market line
        under_probability: Probability total goes under market line
        poisson_over_prob: Poisson model over probability (if available)
        weather_adjusted_total: Weather-adjusted total prediction
        market_total: Market total line at prediction time
        edge_over: Edge for over bet vs market
        edge_under: Edge for under bet vs market
        confidence: Model confidence in prediction
        weather_impact: Estimated weather impact on total
        home_team_total: Predicted home team points
        away_team_total: Predicted away team points
        feature_importances: Feature contribution to prediction
        model_version: Version of model used
        prediction_date: When prediction was made
        metadata: Additional prediction metadata
    """

    game_id: str
    home_team: str
    away_team: str
    predicted_total: float
    over_probability: float
    under_probability: float
    poisson_over_prob: float | None = None
    weather_adjusted_total: float | None = None
    market_total: float | None = None
    edge_over: float | None = None
    edge_under: float | None = None
    confidence: float | None = None
    weather_impact: float | None = None
    home_team_total: float | None = None
    away_team_total: float | None = None
    feature_importances: dict[str, float] | None = None
    model_version: str | None = None
    prediction_date: datetime | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class OUModelResults:
    """
    Container for O/U model training and validation results.

    Attributes:
        total_regression_model: Trained total points regression model
        poisson_model: Optional Poisson score model
        home_score_model: Optional home team score model
        away_score_model: Optional away team score model
        weather_model: Optional weather impact model
        scaler: Fitted feature scaler
        feature_selector: Trained feature selector
        calibrator: Trained probability calibrator
        residual_std: Standard deviation of training residuals
        weather_coefficients: Weather impact coefficients
        feature_names: List of selected feature names
        feature_importances: Dictionary of feature importances
        performance_metrics: Training and validation metrics
        hyperparameters: Final hyperparameters used
        training_history: Training progression data
        calibration_results: Probability calibration results
        poisson_params: Poisson model parameters
        model_version: Model version identifier
        training_date: When model was trained
        metadata: Additional model metadata
    """

    total_regression_model: Any
    poisson_model: Any | None = None
    home_score_model: Any | None = None
    away_score_model: Any | None = None
    weather_model: Any | None = None
    scaler: Any | None = None
    feature_selector: Any | None = None
    calibrator: ProbabilityCalibrator | None = None
    residual_std: float | None = None
    weather_coefficients: dict[str, float] = field(default_factory=dict)
    feature_names: list[str] = field(default_factory=list)
    feature_importances: dict[str, float] = field(default_factory=dict)
    performance_metrics: dict[str, float] = field(default_factory=dict)
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    training_history: dict[str, list[float]] = field(default_factory=dict)
    calibration_results: CalibrationResults | None = None
    poisson_params: dict[str, Any] = field(default_factory=dict)
    model_version: str = "1.0.0"
    training_date: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class TotalDistributionConverter:
    """
    Converts total points predictions to over/under probabilities using residual distribution.

    This class learns the distribution of prediction residuals during training
    and uses it to convert total points predictions to over/under probabilities.
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

    def predict_over_under_probabilities(
        self, predicted_totals: np.ndarray, market_totals: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convert total predictions to over/under probabilities.

        Args:
            predicted_totals: Predicted total points
            market_totals: Market total lines

        Returns:
            Tuple of (over_probabilities, under_probabilities)
        """
        if not self.is_fitted:
            raise ValueError("Must fit the residual distribution first")

        if self.distribution_type == "normal":
            from scipy.stats import norm

            loc = self.distribution_params["loc"]
            scale = self.distribution_params["scale"]

            # Probability that actual total > market total
            z_scores = (market_totals - predicted_totals - loc) / scale
            over_probs = 1 - norm.cdf(z_scores)

        elif self.distribution_type == "t":
            from scipy.stats import t

            df = self.distribution_params["df"]
            loc = self.distribution_params["loc"]
            scale = self.distribution_params["scale"]

            t_scores = (market_totals - predicted_totals - loc) / scale
            over_probs = 1 - t.cdf(t_scores, df)

        elif self.distribution_type == "skewnorm":
            from scipy.stats import skewnorm

            a = self.distribution_params["a"]
            loc = self.distribution_params["loc"]
            scale = self.distribution_params["scale"]

            z_scores = (market_totals - predicted_totals - loc) / scale
            over_probs = 1 - skewnorm.cdf(z_scores, a)

        over_probs = np.clip(over_probs, 0.001, 0.999)
        under_probs = 1 - over_probs

        return over_probs, under_probs


class PoissonScoreModel:
    """
    Poisson-based score model for total points simulation.

    This model predicts individual team scores using Poisson distributions,
    which can be combined to generate total points probabilities.
    """

    def __init__(self, random_state: int = 42):
        """Initialize the Poisson score model."""
        self.random_state = random_state
        self.home_model = None
        self.away_model = None
        self.correlation_factor = 0.0
        self.is_fitted = False

    def fit(
        self, X: pd.DataFrame, home_scores: np.ndarray, away_scores: np.ndarray
    ) -> None:
        """
        Fit Poisson models for home and away team scores.

        Args:
            X: Feature matrix
            home_scores: Home team scores
            away_scores: Away team scores
        """
        # Fit home team score model
        self.home_model = PoissonRegressor()
        self.home_model.fit(X, home_scores)

        # Fit away team score model
        self.away_model = PoissonRegressor()
        self.away_model.fit(X, away_scores)

        # Estimate correlation between team scores
        home_pred = self.home_model.predict(X)
        away_pred = self.away_model.predict(X)
        home_residuals = home_scores - home_pred
        away_residuals = away_scores - away_pred
        self.correlation_factor = np.corrcoef(home_residuals, away_residuals)[0, 1]

        self.is_fitted = True
        logger.info(
            f"Fitted Poisson score models with correlation: {self.correlation_factor:.3f}"
        )

    def predict_scores(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict individual team scores.

        Args:
            X: Feature matrix

        Returns:
            Tuple of (home_scores, away_scores)
        """
        if not self.is_fitted:
            raise ValueError("Must fit the Poisson models first")

        home_scores = self.home_model.predict(X)
        away_scores = self.away_model.predict(X)

        return home_scores, away_scores

    def simulate_game_totals(
        self, X: pd.DataFrame, n_simulations: int = 10000
    ) -> np.ndarray:
        """
        Simulate game totals using Poisson distributions.

        Args:
            X: Feature matrix for single game
            n_simulations: Number of simulations to run

        Returns:
            Array of simulated total scores
        """
        if not self.is_fitted:
            raise ValueError("Must fit the Poisson models first")

        home_lambda, away_lambda = self.predict_scores(X)
        home_lambda = home_lambda[0] if len(home_lambda) == 1 else home_lambda
        away_lambda = away_lambda[0] if len(away_lambda) == 1 else away_lambda

        # Simulate scores
        np.random.seed(self.random_state)
        home_sims = np.random.poisson(home_lambda, n_simulations)
        away_sims = np.random.poisson(away_lambda, n_simulations)

        # Add correlation adjustment
        if abs(self.correlation_factor) > 0.1:
            # Simple correlation adjustment
            correlation_noise = np.random.normal(
                0, abs(self.correlation_factor) * 5, n_simulations
            )
            away_sims = np.maximum(0, away_sims + correlation_noise.astype(int))

        total_sims = home_sims + away_sims
        return total_sims

    def calculate_over_under_probabilities(
        self, X: pd.DataFrame, market_total: float, n_simulations: int = 10000
    ) -> tuple[float, float]:
        """
        Calculate over/under probabilities using simulation.

        Args:
            X: Feature matrix for single game
            market_total: Market total line
            n_simulations: Number of simulations

        Returns:
            Tuple of (over_probability, under_probability)
        """
        simulated_totals = self.simulate_game_totals(X, n_simulations)

        over_count = np.sum(simulated_totals > market_total)
        under_count = np.sum(simulated_totals < market_total)
        push_count = n_simulations - over_count - under_count

        over_prob = (over_count + push_count / 2) / n_simulations
        under_prob = (under_count + push_count / 2) / n_simulations

        return over_prob, under_prob


class WeatherImpactModel:
    """
    Specialized model for weather impact on total points.

    This model focuses on weather conditions that significantly affect scoring,
    particularly wind, temperature, and precipitation.
    """

    def __init__(self):
        """Initialize the weather impact model."""
        self.coefficients = {}
        self.is_fitted = False

    def fit(self, weather_features: pd.DataFrame, total_residuals: np.ndarray) -> None:
        """
        Fit weather impact coefficients.

        Args:
            weather_features: Weather feature matrix
            total_residuals: Residuals from base total model
        """

        # Focus on key weather variables for totals
        weather_cols = [
            col
            for col in weather_features.columns
            if any(
                weather_term in col.lower()
                for weather_term in ["wind", "temp", "precip", "humidity"]
            )
        ]

        if len(weather_cols) == 0:
            logger.warning("No weather features found for impact modeling")
            self.is_fitted = True
            return

        X_weather = weather_features[weather_cols].fillna(0)

        # Fit linear model to predict weather impact on totals
        model = LinearRegression()
        model.fit(X_weather, total_residuals)

        # Store coefficients
        self.coefficients = dict(zip(weather_cols, model.coef_, strict=False))
        self.baseline_intercept = model.intercept_

        self.is_fitted = True
        logger.info(f"Weather impact model fitted with {len(weather_cols)} features")

    def predict_weather_impact(self, weather_features: pd.DataFrame) -> np.ndarray:
        """
        Predict weather impact on total points.

        Args:
            weather_features: Weather feature matrix

        Returns:
            Array of weather impact adjustments
        """
        if not self.is_fitted or len(self.coefficients) == 0:
            return np.zeros(len(weather_features))

        weather_cols = list(self.coefficients.keys())
        X_weather = weather_features[weather_cols].fillna(0)

        impact = self.baseline_intercept
        for col, coef in self.coefficients.items():
            impact += coef * X_weather[col].values

        return impact


class OUModel(BaseEstimator):
    """
    Comprehensive Over/Under (O/U) model for NFL predictions.

    This model predicts game totals and over/under probabilities using
    multiple approaches including regression, Poisson simulation, and
    weather impact modeling.
    """

    def __init__(
        self,
        model_type: str = "xgboost",
        use_poisson: bool = True,
        use_weather_model: bool = True,
        feature_selection_method: str = "model_based",
        max_features: int | None = 20,
        regularization_strength: float = 0.1,
        use_calibration: bool = True,
        hyperparameter_tuning: str = "grid_search",
        distribution_type: str = "normal",
        random_state: int = 42,
    ):
        """
        Initialize the O/U model.

        Args:
            model_type: Type of base model ("xgboost", "lightgbm", "random_forest")
            use_poisson: Whether to use Poisson score model
            use_weather_model: Whether to use weather impact modeling
            feature_selection_method: Method for feature selection
            max_features: Maximum number of features to select
            regularization_strength: Strength of regularization
            use_calibration: Whether to use probability calibration
            hyperparameter_tuning: Type of hyperparameter tuning
            distribution_type: Distribution for residual conversion
            random_state: Random state for reproducibility
        """
        self.model_type = model_type
        self.use_poisson = use_poisson
        self.use_weather_model = use_weather_model
        self.feature_selection_method = feature_selection_method
        self.max_features = max_features
        self.regularization_strength = regularization_strength
        self.use_calibration = use_calibration
        self.hyperparameter_tuning = hyperparameter_tuning
        self.distribution_type = distribution_type
        self.random_state = random_state

        # Model components
        self.total_model = None
        self.poisson_model = None
        self.weather_model = None
        self.scaler = None
        self.feature_selector = None
        self.calibrator = None
        self.total_converter = None

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
        if task_type == "regression":
            return Ridge(random_state=self.random_state)
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(random_state=self.random_state)

    def _prepare_features(
        self, data: pd.DataFrame
    ) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare features and targets for O/U modeling.

        Args:
            data: Input dataframe with game data

        Returns:
            Tuple of (features, total_targets, home_scores, away_scores)
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
            "total_points",
            "home_score",
            "away_score",
            "over_under",
            "market_total",
        ]

        feature_cols = [col for col in data.columns if col not in exclude_cols]
        X = data[feature_cols].copy()

        # Handle missing values
        X = X.fillna(X.median())

        # Create targets
        total_targets = None
        home_scores = None
        away_scores = None

        if "total_points" in data.columns:
            total_targets = data["total_points"].values
        elif "home_score" in data.columns and "away_score" in data.columns:
            home_scores = data["home_score"].values
            away_scores = data["away_score"].values
            total_targets = home_scores + away_scores

        return X, total_targets, home_scores, away_scores

    def select_features(self, X: pd.DataFrame, y: np.ndarray) -> tuple[Any, list[str]]:
        """
        Select features using the specified method.

        Args:
            X: Feature matrix
            y: Target variable

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
            selector = SelectKBest(
                score_func=f_regression, k=min(max_features, len(feature_cols))
            )
            selector.fit(X, y)
            selected_features = [
                feature_cols[i] for i in selector.get_support(indices=True)
            ]

        elif self.feature_selection_method == "model_based":
            base_model = self._create_base_model("regression")
            selector = SelectFromModel(base_model, max_features=max_features)
            selector.fit(X, y)
            selected_features = [
                feature_cols[i] for i in selector.get_support(indices=True)
            ]

        elif self.feature_selection_method == "recursive":
            base_model = self._create_base_model("regression")
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

    def tune_hyperparameters(self, X: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        """
        Tune hyperparameters using the specified method.

        Args:
            X: Feature matrix
            y: Target variable

        Returns:
            Best hyperparameters found
        """
        base_model = self._create_base_model("regression")

        # Define parameter grids based on model type
        if self.model_type == "xgboost" or (
            self.model_type == "lightgbm" and LIGHTGBM_AVAILABLE
        ):
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
            # Simple parameter grid for Ridge
            param_grid = {"alpha": [0.1, 1.0, 10.0]}

        # Choose search method
        if self.hyperparameter_tuning == "grid_search":
            search = GridSearchCV(
                base_model,
                param_grid,
                cv=5,
                scoring="neg_mean_squared_error",
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
                scoring="neg_mean_squared_error",
                n_jobs=-1,
                verbose=0,
                random_state=self.random_state,
            )

        search.fit(X, y)
        logger.info(f"Best O/U hyperparameters: {search.best_params_}")

        return search.best_params_

    def train_model(
        self,
        training_data: pd.DataFrame,
        validation_data: pd.DataFrame | None = None,
    ) -> OUModelResults:
        """
        Train the O/U model using multiple approaches.

        Args:
            training_data: Training dataset
            validation_data: Optional validation dataset

        Returns:
            OUModelResults containing all training artifacts
        """
        logger.info("Starting O/U model training...")

        # Prepare features and targets
        X, total_targets, home_scores, away_scores = self._prepare_features(
            training_data
        )

        if total_targets is None:
            raise ValueError("Training data must include total points information")

        # Initialize scaler
        self.scaler = RobustScaler()
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X), columns=X.columns, index=X.index
        )

        # Train total points regression model
        logger.info("Training total points regression model...")

        # Feature selection
        self.feature_selector, self.feature_names = self.select_features(
            X_scaled, total_targets
        )

        X_selected = X_scaled[self.feature_names]

        # Hyperparameter tuning
        if self.hyperparameter_tuning != "none":
            best_params = self.tune_hyperparameters(X_selected, total_targets)
        else:
            best_params = {}

        # Create and train total model
        self.total_model = self._create_base_model("regression")
        self.total_model.set_params(**best_params)
        self.total_model.fit(X_selected, total_targets)

        # Calculate residuals and fit distribution
        total_predictions = self.total_model.predict(X_selected)
        residuals = total_targets - total_predictions
        self.residual_std = np.std(residuals)

        self.total_converter = TotalDistributionConverter(self.distribution_type)
        self.total_converter.fit(residuals)

        # Train Poisson score model if enabled
        poisson_model = None
        if self.use_poisson and home_scores is not None and away_scores is not None:
            logger.info("Training Poisson score model...")
            poisson_model = PoissonScoreModel(random_state=self.random_state)
            poisson_model.fit(X_selected, home_scores, away_scores)

        # Train weather impact model if enabled
        weather_model = None
        weather_coefficients = {}
        if self.use_weather_model:
            logger.info("Training weather impact model...")
            weather_model = WeatherImpactModel()
            weather_model.fit(X_selected, residuals)
            weather_coefficients = weather_model.coefficients

        # Calculate feature importances
        if hasattr(self.total_model, "feature_importances_"):
            self.feature_importances = dict(
                zip(
                    self.feature_names,
                    self.total_model.feature_importances_,
                    strict=False,
                )
            )
        elif hasattr(self.total_model, "coef_"):
            self.feature_importances = dict(
                zip(self.feature_names, np.abs(self.total_model.coef_), strict=False)
            )

        # Probability calibration
        calibrator = None
        calibration_results = None

        if (
            self.use_calibration
            and "market_total" in training_data.columns
            and "over_under" in training_data.columns
        ):
            logger.info("Training probability calibration...")

            market_totals = training_data["market_total"].values
            over_under_results = training_data["over_under"].values

            over_probs, _ = self.total_converter.predict_over_under_probabilities(
                total_predictions, market_totals
            )

            calibrator = ProbabilityCalibrator(primary_method="isotonic")
            calibration_results = calibrator.calibrate_probabilities(
                over_probs, over_under_results
            )
            self.calibrator = calibrator
            self.trained_calibrator = calibration_results.calibrator

        # Calculate training metrics
        train_mae = mean_absolute_error(total_targets, total_predictions)
        train_rmse = np.sqrt(mean_squared_error(total_targets, total_predictions))
        train_r2 = r2_score(total_targets, total_predictions)

        performance_metrics = {
            "training_mae": train_mae,
            "training_rmse": train_rmse,
            "training_r2": train_r2,
            "training_residual_std": self.residual_std,
        }

        # Add O/U specific metrics if available
        if (
            "market_total" in training_data.columns
            and "over_under" in training_data.columns
        ):
            market_totals = training_data["market_total"].values
            over_under_results = training_data["over_under"].values

            over_probs, _ = self.total_converter.predict_over_under_probabilities(
                total_predictions, market_totals
            )

            if calibrator and hasattr(self, "trained_calibrator"):
                over_probs = calibrator.apply_calibration(
                    over_probs, self.trained_calibrator
                )

            from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

            performance_metrics.update(
                {
                    "training_over_accuracy": accuracy_score(
                        over_under_results, over_probs > 0.5
                    ),
                    "training_over_log_loss": log_loss(over_under_results, over_probs),
                    "training_over_brier_score": brier_score_loss(
                        over_under_results, over_probs
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

            if "total_points" in validation_data.columns or (
                "home_score" in validation_data.columns
                and "away_score" in validation_data.columns
            ):
                if "total_points" in validation_data.columns:
                    val_totals = validation_data["total_points"].values
                else:
                    val_totals = (
                        validation_data["home_score"].values
                        + validation_data["away_score"].values
                    )

                pred_totals = np.array([p.predicted_total for p in val_predictions])

                performance_metrics.update(
                    {
                        "validation_mae": mean_absolute_error(val_totals, pred_totals),
                        "validation_rmse": np.sqrt(
                            mean_squared_error(val_totals, pred_totals)
                        ),
                        "validation_r2": r2_score(val_totals, pred_totals),
                    }
                )

            if "over_under" in validation_data.columns:
                val_over_under = validation_data["over_under"].values
                pred_over_probs = np.array(
                    [p.over_probability for p in val_predictions]
                )

                performance_metrics.update(
                    {
                        "validation_over_accuracy": accuracy_score(
                            val_over_under, pred_over_probs > 0.5
                        ),
                        "validation_over_log_loss": log_loss(
                            val_over_under, pred_over_probs
                        ),
                        "validation_over_brier_score": brier_score_loss(
                            val_over_under, pred_over_probs
                        ),
                    }
                )

        self.is_trained = True

        # Create results container
        results = OUModelResults(
            total_regression_model=self.total_model,
            poisson_model=poisson_model,
            weather_model=weather_model,
            scaler=self.scaler,
            feature_selector=self.feature_selector,
            calibrator=calibrator,
            residual_std=self.residual_std,
            weather_coefficients=weather_coefficients,
            feature_names=self.feature_names,
            feature_importances=self.feature_importances,
            performance_metrics=performance_metrics,
            hyperparameters=best_params,
            calibration_results=calibration_results,
            training_date=datetime.now(),
            metadata={
                "model_type": self.model_type,
                "use_poisson": self.use_poisson,
                "use_weather_model": self.use_weather_model,
                "feature_selection_method": self.feature_selection_method,
                "training_samples": len(training_data),
                "distribution_type": self.distribution_type,
            },
        )

        logger.info(
            f"O/U model training complete. Training MAE: {train_mae:.3f}, RMSE: {train_rmse:.3f}"
        )
        return results

    def predict(
        self, data: pd.DataFrame, include_simulation: bool = True
    ) -> list[OUModelPrediction]:
        """
        Make O/U predictions on new data.

        Args:
            data: Input data for prediction
            include_simulation: Whether to include Poisson simulation results

        Returns:
            List of OUModelPrediction objects
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Prepare features
        X, _, _, _ = self._prepare_features(data)

        # Handle missing features by adding them with default values
        missing_features = set(self.feature_names) - set(X.columns)
        if missing_features:
            logger.warning(f"Missing features in prediction data: {missing_features}")
            for feature in missing_features:
                X[feature] = 0.0

        # Reorder columns to match training
        X = X[self.feature_names]

        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.transform(X), columns=X.columns, index=X.index
        )

        # Make total predictions
        total_predictions = self.total_model.predict(X_scaled)

        # Weather adjustments if available
        weather_adjustments = np.zeros(len(total_predictions))
        if (
            self.use_weather_model
            and hasattr(self, "weather_model")
            and self.weather_model
        ):
            weather_adjustments = self.weather_model.predict_weather_impact(X_scaled)

        weather_adjusted_totals = total_predictions + weather_adjustments

        # Create predictions
        predictions = []

        for i, (_idx, row) in enumerate(data.iterrows()):
            game_id = row.get("game_id", f"game_{i}")
            home_team = row.get("home_team", "HOME")
            away_team = row.get("away_team", "AWAY")
            predicted_total = total_predictions[i]
            weather_adjusted_total = weather_adjusted_totals[i]

            # Get market total if available
            market_total = row.get("market_total", None)

            # Calculate over/under probabilities
            if market_total is not None:
                over_prob, under_prob = (
                    self.total_converter.predict_over_under_probabilities(
                        np.array([weather_adjusted_total]), np.array([market_total])
                    )
                )
                over_prob = over_prob[0]
                under_prob = under_prob[0]

                # Apply calibration if available
                if self.calibrator and hasattr(self, "trained_calibrator"):
                    over_prob = self.calibrator.apply_calibration(
                        np.array([over_prob]), self.trained_calibrator
                    )[0]
                    under_prob = 1 - over_prob

                # Calculate edges (assuming -110 juice)
                implied_over_prob = 0.524  # -110 juice
                implied_under_prob = 0.524
                edge_over = over_prob - implied_over_prob
                edge_under = under_prob - implied_under_prob
            else:
                over_prob = under_prob = 0.5
                edge_over = edge_under = None

            # Poisson simulation if enabled
            poisson_over_prob = None
            home_team_total = None
            away_team_total = None

            if (
                include_simulation
                and self.use_poisson
                and hasattr(self, "poisson_model")
                and self.poisson_model
            ):
                try:
                    game_features = X_scaled.iloc[[i]]
                    poisson_over_prob, _ = (
                        self.poisson_model.calculate_over_under_probabilities(
                            game_features,
                            market_total if market_total else predicted_total,
                        )
                    )
                    home_team_total, away_team_total = (
                        self.poisson_model.predict_scores(game_features)
                    )
                    home_team_total = home_team_total[0]
                    away_team_total = away_team_total[0]
                except (ValueError, TypeError, IndexError, RuntimeError):
                    pass  # Silently handle Poisson errors

            # Feature importances for this prediction
            feature_importances = None
            if hasattr(self.total_model, "feature_importances_"):
                feature_importances = dict(
                    zip(
                        self.feature_names,
                        self.total_model.feature_importances_,
                        strict=False,
                    )
                )

            # Calculate confidence
            confidence = abs(over_prob - 0.5) * 2 if market_total else None

            # Weather impact
            weather_impact = weather_adjustments[i] if self.use_weather_model else None

            prediction = OUModelPrediction(
                game_id=game_id,
                home_team=home_team,
                away_team=away_team,
                predicted_total=predicted_total,
                over_probability=over_prob,
                under_probability=under_prob,
                poisson_over_prob=poisson_over_prob,
                weather_adjusted_total=weather_adjusted_total,
                market_total=market_total,
                edge_over=edge_over,
                edge_under=edge_under,
                confidence=confidence,
                weather_impact=weather_impact,
                home_team_total=home_team_total,
                away_team_total=away_team_total,
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
            f"Running O/U walk-forward validation from {start_season} to {end_season}"
        )

        results = {
            "season_results": [],
            "overall_metrics": {},
            "feature_importance_evolution": [],
        }

        all_predictions = []
        all_actuals_total = []
        all_actuals_over = []

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
            if "total_points" in test_data.columns or (
                "home_score" in test_data.columns and "away_score" in test_data.columns
            ):
                if "total_points" in test_data.columns:
                    actual_totals = test_data["total_points"].values
                else:
                    actual_totals = (
                        test_data["home_score"].values + test_data["away_score"].values
                    )

                pred_totals = np.array([p.predicted_total for p in predictions])

                season_mae = mean_absolute_error(actual_totals, pred_totals)
                season_rmse = np.sqrt(mean_squared_error(actual_totals, pred_totals))
                season_r2 = r2_score(actual_totals, pred_totals)

                all_predictions.extend(pred_totals)
                all_actuals_total.extend(actual_totals)
            else:
                season_mae = season_rmse = season_r2 = None

            if "over_under" in test_data.columns:
                actual_over_under = test_data["over_under"].values
                pred_over_probs = np.array([p.over_probability for p in predictions])

                season_accuracy = accuracy_score(
                    actual_over_under, pred_over_probs > 0.5
                )
                season_log_loss = log_loss(actual_over_under, pred_over_probs)
                season_brier = brier_score_loss(actual_over_under, pred_over_probs)

                all_actuals_over.extend(actual_over_under)
            else:
                season_accuracy = season_log_loss = season_brier = None

            season_result = {
                "season": season,
                "train_games": len(train_data),
                "test_games": len(test_data),
                "mae": season_mae,
                "rmse": season_rmse,
                "r2": season_r2,
                "over_accuracy": season_accuracy,
                "over_log_loss": season_log_loss,
                "over_brier_score": season_brier,
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
        if all_actuals_total:
            overall_mae = mean_absolute_error(all_actuals_total, all_predictions)
            overall_rmse = np.sqrt(
                mean_squared_error(all_actuals_total, all_predictions)
            )
            overall_r2 = r2_score(all_actuals_total, all_predictions)

            results["overall_metrics"].update(
                {
                    "overall_mae": overall_mae,
                    "overall_rmse": overall_rmse,
                    "overall_r2": overall_r2,
                    "seasons_validated": len(results["season_results"]),
                }
            )

        logger.info(
            f"O/U walk-forward validation complete. Overall MAE: {results['overall_metrics'].get('overall_mae', 'N/A')}"
        )
        return results

    def get_model_summary(self) -> dict[str, Any]:
        """Get a comprehensive summary of the trained model."""
        if not self.is_trained:
            return {
                "model_type": "O/U Model",
                "is_trained": False,
                "status": "Not trained",
            }

        summary = {
            "model_type": "O/U Model",
            "is_trained": True,
            "base_model_type": self.model_type,
            "use_poisson": self.use_poisson,
            "use_weather_model": self.use_weather_model,
            "distribution_type": self.distribution_type,
            "features_selected": len(self.feature_names),
            "residual_std": self.residual_std,
            "feature_importances": self.feature_importances,
            "weather_coefficients": getattr(self, "weather_coefficients", {}),
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
            "total_regression_model": self.total_model,
            "poisson_model": getattr(self, "poisson_model", None),
            "weather_model": getattr(self, "weather_model", None),
            "scaler": self.scaler,
            "feature_selector": self.feature_selector,
            "calibrator": self.calibrator,
            "trained_calibrator": getattr(self, "trained_calibrator", None),
            "total_converter": self.total_converter,
            "feature_names": self.feature_names,
            "feature_importances": self.feature_importances,
            "residual_std": self.residual_std,
            "model_config": {
                "model_type": self.model_type,
                "use_poisson": self.use_poisson,
                "use_weather_model": self.use_weather_model,
                "feature_selection_method": self.feature_selection_method,
                "max_features": self.max_features,
                "use_calibration": self.use_calibration,
                "distribution_type": self.distribution_type,
                "random_state": self.random_state,
            },
        }

        joblib.dump(model_data, filepath)
        logger.info(f"O/U model saved to {filepath}")

    def load_model(self, filepath: str) -> None:
        """Load a trained model from disk."""
        model_data = joblib.load(filepath)

        self.total_model = model_data["total_regression_model"]
        self.poisson_model = model_data.get("poisson_model")
        self.weather_model = model_data.get("weather_model")
        self.scaler = model_data["scaler"]
        self.feature_selector = model_data.get("feature_selector")
        self.calibrator = model_data.get("calibrator")
        self.trained_calibrator = model_data.get("trained_calibrator")
        self.total_converter = model_data["total_converter"]
        self.feature_names = model_data["feature_names"]
        self.feature_importances = model_data["feature_importances"]
        self.residual_std = model_data["residual_std"]

        # Restore configuration
        config = model_data["model_config"]
        self.model_type = config["model_type"]
        self.use_poisson = config["use_poisson"]
        self.use_weather_model = config["use_weather_model"]
        self.feature_selection_method = config["feature_selection_method"]
        self.max_features = config["max_features"]
        self.use_calibration = config["use_calibration"]
        self.distribution_type = config["distribution_type"]
        self.random_state = config["random_state"]

        self.is_trained = True
        logger.info(f"O/U model loaded from {filepath}")


def main():
    """Main function for command-line usage."""
    import argparse
    from pathlib import Path

    from utils.date_utils import get_current_nfl_season, get_current_nfl_week

    parser = argparse.ArgumentParser(description="Train Over/Under (O/U) Model")
    parser.add_argument("--season", type=int, help="Target season (default: current)")
    parser.add_argument(
        "--week", help="Target week (default: current, 'all' for full season)"
    )
    parser.add_argument(
        "--model-type",
        default="xgboost",
        choices=["xgboost", "lightgbm", "random_forest"],
        help="Model algorithm (default: xgboost)",
    )
    parser.add_argument(
        "--no-poisson", action="store_true", help="Disable Poisson score modeling"
    )
    parser.add_argument(
        "--no-weather", action="store_true", help="Disable weather impact modeling"
    )
    parser.add_argument(
        "--distribution",
        default="normal",
        choices=["normal", "poisson", "negative_binomial"],
        help="Distribution type for modeling (default: normal)",
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
        features_df = load_dataframe("features_ou", layer="gold")

        # COLD-02 / T-33-18: refuse a PROVISIONAL Elo row as a TRAINING input, HERE at
        # the gold-loading boundary and BEFORE any filtering. This trainer reads gold
        # DIRECTLY and never passes through features/elo_features.build_features, so a
        # guard placed only in the feature builder protects none of this path. Filtering
        # first would let a provisional row drop out of sight and still be trained on in
        # a different slice. The refusal is a RuntimeError subclass and is therefore NOT
        # caught by the surrounding handler tuple -- a refusal converted into a logged
        # `return` would be a silent no-train.
        from features.elo_features import assert_no_provisional_training_rows

        assert_no_provisional_training_rows(features_df, "train:ou")

        # Filter for target season/week
        if season:
            features_df = features_df[features_df["season"] == season]
        if week:
            features_df = features_df[features_df["week"] == week]

        logger.info(
            "Loaded O/U features", season=season, week=week, records=len(features_df)
        )
    except (FileNotFoundError, OSError, ValueError, KeyError) as e:
        logger.error(f"Failed to load features: {e}")
        return

    # Initialize model with pipeline defaults
    model = OUModel(
        model_type=args.model_type,
        use_poisson=not args.no_poisson,  # Default enabled
        use_weather_model=not args.no_weather,  # Default enabled
        feature_selection_method="model_based",
        use_calibration=True,
        hyperparameter_tuning="grid_search",
        distribution_type=args.distribution,
    )

    # Always do both: Train first, then validate
    model.train_model(features_df)

    # Save to standard artifacts location
    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)
    model_path = artifacts_dir / "ou_model.pkl"
    model.save_model(str(model_path))

    # Load the saved model and run validation
    validation_model = OUModel()
    validation_model.load_model(str(model_path))

    validation_model.run_walk_forward_validation(features_df)

    logger.info("O/U model training completed successfully")


if __name__ == "__main__":
    main()
