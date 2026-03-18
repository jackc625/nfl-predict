#!/usr/bin/env python3
"""
Model Training Utilities

This module provides utilities for NFL prediction model training:
- Walk-forward validation framework
- Data splitting by season/week with no-leakage guarantee
- Cross-validation utilities for within-season validation
- Model serialization and loading with versioning
- Training pipeline orchestration
- Performance tracking and logging

All utilities enforce strict temporal ordering to prevent look-ahead bias.
"""

import json
import pickle
import sys
from collections.abc import Iterator
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

from utils import get_logger

logger = get_logger(__name__)


@dataclass
class TrainTestSplit:
    """
    Container for train/test split data with metadata.

    Attributes:
        train_data: Training features DataFrame
        train_targets: Training targets Series/DataFrame
        test_data: Testing features DataFrame
        test_targets: Testing targets Series/DataFrame
        train_seasons: Seasons used for training
        test_season: Season used for testing
        train_weeks: Weeks used for training (if applicable)
        test_weeks: Weeks used for testing (if applicable)
        split_date: Date of the split for temporal validation
        metadata: Additional split metadata
    """

    train_data: pd.DataFrame
    train_targets: pd.Series
    test_data: pd.DataFrame
    test_targets: pd.Series
    train_seasons: list[int]
    test_season: int
    train_weeks: list[int] | None = None
    test_weeks: list[int] | None = None
    split_date: datetime | None = None
    metadata: dict[str, Any] | None = None


@dataclass
class ModelMetadata:
    """
    Metadata for trained models.

    Attributes:
        model_name: Name/identifier of the model
        model_type: Type of model (e.g., 'logistic', 'xgboost')
        target_type: Target type ('wp', 'ats', 'ou')
        train_seasons: Seasons used for training
        features_used: List of feature names used
        training_date: Date when model was trained
        performance_metrics: Dictionary of performance metrics
        hyperparameters: Dictionary of model hyperparameters
        feature_importance: Dictionary of feature importances (if available)
        version: Model version string
        notes: Additional notes about the model
    """

    model_name: str
    model_type: str
    target_type: str
    train_seasons: list[int]
    features_used: list[str]
    training_date: datetime
    performance_metrics: dict[str, float]
    hyperparameters: dict[str, Any]
    feature_importance: dict[str, float] | None = None
    version: str = "1.0.0"
    notes: str = ""


class WalkForwardValidator:
    """
    Walk-forward validation framework for NFL prediction models.

    Implements strict temporal ordering with no look-ahead bias.
    Supports both season-by-season and weekly walk-forward validation.
    """

    def __init__(
        self,
        min_train_seasons: int = 3,
        max_train_seasons: int | None = None,
        validation_method: str = "expanding",
        allow_partial_seasons: bool = False,
    ):
        """
        Initialize walk-forward validator.

        Args:
            min_train_seasons: Minimum seasons required for training
            max_train_seasons: Maximum seasons to include in training (sliding window)
            validation_method: 'expanding' (use all past data) or 'sliding' (fixed window)
            allow_partial_seasons: Whether to allow training on partial seasons
        """
        self.min_train_seasons = min_train_seasons
        self.max_train_seasons = max_train_seasons
        self.validation_method = validation_method
        self.allow_partial_seasons = allow_partial_seasons
        self.logger = get_logger(__name__)

    def create_seasonal_splits(
        self,
        data: pd.DataFrame,
        target_column: str,
        start_season: int = 2018,
        end_season: int = 2024,
    ) -> Iterator[TrainTestSplit]:
        """
        Create walk-forward splits by season.

        For each test season, uses all previous seasons as training data
        (subject to min/max season constraints).

        Args:
            data: DataFrame with features and targets
            target_column: Name of target column
            start_season: First season to include
            end_season: Last season to include

        Yields:
            TrainTestSplit objects for each season
        """
        if "season" not in data.columns:
            raise ValueError("Data must contain 'season' column for seasonal splits")

        available_seasons = sorted(data["season"].unique())
        valid_seasons = [
            s for s in available_seasons if start_season <= s <= end_season
        ]

        self.logger.info(
            "Creating seasonal walk-forward splits",
            available_seasons=len(available_seasons),
            valid_seasons=len(valid_seasons),
            min_train_seasons=self.min_train_seasons,
        )

        # Start testing from the season that has enough training history
        test_start_season = start_season + self.min_train_seasons

        for test_season in range(test_start_season, end_season + 1):
            if test_season not in valid_seasons:
                self.logger.warning(
                    f"Skipping season {test_season} - no data available"
                )
                continue

            # Determine training seasons
            if self.validation_method == "expanding":
                # Use all available seasons before test season
                train_seasons = [s for s in valid_seasons if s < test_season]
            else:  # sliding window
                # Use fixed window of previous seasons
                window_start = max(start_season, test_season - self.max_train_seasons)
                train_seasons = [
                    s for s in valid_seasons if window_start <= s < test_season
                ]

            # Check minimum training seasons requirement
            if len(train_seasons) < self.min_train_seasons:
                self.logger.warning(
                    f"Skipping season {test_season} - insufficient training seasons "
                    f"({len(train_seasons)} < {self.min_train_seasons})"
                )
                continue

            # Split data
            train_mask = data["season"].isin(train_seasons)
            test_mask = data["season"] == test_season

            train_data = data[train_mask].drop(columns=[target_column])
            train_targets = data[train_mask][target_column]
            test_data = data[test_mask].drop(columns=[target_column])
            test_targets = data[test_mask][target_column]

            # Remove any rows with missing targets
            train_valid = train_targets.notna()
            test_valid = test_targets.notna()

            if train_valid.sum() == 0:
                self.logger.warning(
                    f"No valid training targets for season {test_season}"
                )
                continue

            if test_valid.sum() == 0:
                self.logger.warning(f"No valid test targets for season {test_season}")
                continue

            train_data = train_data[train_valid]
            train_targets = train_targets[train_valid]
            test_data = test_data[test_valid]
            test_targets = test_targets[test_valid]

            # Create split object
            split = TrainTestSplit(
                train_data=train_data,
                train_targets=train_targets,
                test_data=test_data,
                test_targets=test_targets,
                train_seasons=train_seasons,
                test_season=test_season,
                split_date=datetime.now(),
                metadata={
                    "validation_method": self.validation_method,
                    "train_games": len(train_data),
                    "test_games": len(test_data),
                },
            )

            self.logger.info(
                f"Created split for season {test_season}",
                train_seasons=train_seasons,
                train_games=len(train_data),
                test_games=len(test_data),
            )

            yield split

    def create_weekly_splits(
        self,
        data: pd.DataFrame,
        target_column: str,
        season: int,
        start_week: int = 5,  # Start predictions from week 5
        end_week: int = 18,
    ) -> Iterator[TrainTestSplit]:
        """
        Create walk-forward splits by week within a season.

        For each test week, uses all previous weeks in the season plus
        all data from previous seasons as training data.

        Args:
            data: DataFrame with features and targets
            target_column: Name of target column
            season: Season to create weekly splits for
            start_week: First week to predict
            end_week: Last week to predict

        Yields:
            TrainTestSplit objects for each week
        """
        if "season" not in data.columns or "week" not in data.columns:
            raise ValueError(
                "Data must contain 'season' and 'week' columns for weekly splits"
            )

        season_data = data[data["season"] == season]
        available_weeks = sorted(season_data["week"].unique())

        # Get all historical data (previous seasons)
        historical_data = data[data["season"] < season]

        self.logger.info(
            "Creating weekly walk-forward splits",
            season=season,
            available_weeks=len(available_weeks),
            historical_games=len(historical_data),
        )

        for test_week in range(start_week, end_week + 1):
            if test_week not in available_weeks:
                self.logger.warning(f"Skipping week {test_week} - no data available")
                continue

            # Training data: all historical + current season up to test week
            train_weeks = [w for w in available_weeks if w < test_week]
            current_season_train = season_data[season_data["week"].isin(train_weeks)]

            # Combine historical and current season training data
            if len(historical_data) > 0:
                train_data_full = pd.concat(
                    [historical_data, current_season_train], ignore_index=True
                )
            else:
                train_data_full = current_season_train

            # Test data: current week only
            test_data_full = season_data[season_data["week"] == test_week]

            if len(train_data_full) == 0:
                self.logger.warning(f"No training data for week {test_week}")
                continue

            if len(test_data_full) == 0:
                self.logger.warning(f"No test data for week {test_week}")
                continue

            # Split features and targets
            train_data = train_data_full.drop(columns=[target_column])
            train_targets = train_data_full[target_column]
            test_data = test_data_full.drop(columns=[target_column])
            test_targets = test_data_full[target_column]

            # Remove missing targets
            train_valid = train_targets.notna()
            test_valid = test_targets.notna()

            if train_valid.sum() == 0 or test_valid.sum() == 0:
                self.logger.warning(f"Invalid targets for week {test_week}")
                continue

            train_data = train_data[train_valid]
            train_targets = train_targets[train_valid]
            test_data = test_data[test_valid]
            test_targets = test_targets[test_valid]

            # Create split object
            split = TrainTestSplit(
                train_data=train_data,
                train_targets=train_targets,
                test_data=test_data,
                test_targets=test_targets,
                train_seasons=sorted(train_data_full["season"].unique()),
                test_season=season,
                train_weeks=train_weeks,
                test_weeks=[test_week],
                split_date=datetime.now(),
                metadata={
                    "validation_method": "weekly",
                    "train_games": len(train_data),
                    "test_games": len(test_data),
                    "historical_games": len(historical_data),
                },
            )

            self.logger.info(
                f"Created weekly split for season {season}, week {test_week}",
                train_games=len(train_data),
                test_games=len(test_data),
            )

            yield split


class CrossValidator:
    """
    Cross-validation utilities for model selection and hyperparameter tuning.

    Implements time-aware cross-validation that respects temporal ordering.
    """

    def __init__(
        self, n_folds: int = 5, validation_method: str = "temporal", gap_weeks: int = 1
    ):
        """
        Initialize cross-validator.

        Args:
            n_folds: Number of cross-validation folds
            validation_method: 'temporal' (time-aware) or 'random' (standard CV)
            gap_weeks: Weeks to skip between train/validation to prevent leakage
        """
        self.n_folds = n_folds
        self.validation_method = validation_method
        self.gap_weeks = gap_weeks
        self.logger = get_logger(__name__)

    def create_temporal_folds(
        self, data: pd.DataFrame, target_column: str
    ) -> Iterator[tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]]:
        """
        Create temporal cross-validation folds.

        Splits data chronologically, ensuring validation data is always
        after training data with optional gap to prevent leakage.

        Args:
            data: DataFrame with features and targets
            target_column: Name of target column

        Yields:
            Tuples of (train_features, train_targets, val_features, val_targets)
        """
        if "season" not in data.columns or "week" not in data.columns:
            raise ValueError("Temporal CV requires 'season' and 'week' columns")

        # Sort data chronologically
        data_sorted = data.sort_values(["season", "week"]).reset_index(drop=True)
        n_samples = len(data_sorted)
        fold_size = n_samples // self.n_folds

        self.logger.info(
            "Creating temporal CV folds",
            n_samples=n_samples,
            n_folds=self.n_folds,
            fold_size=fold_size,
        )

        for fold in range(self.n_folds):
            # Calculate split points
            val_start = fold * fold_size
            val_end = (fold + 1) * fold_size if fold < self.n_folds - 1 else n_samples

            # Skip first fold if it would have no training data
            if val_start == 0:
                self.logger.warning(
                    f"Skipping fold {fold} - no training data available"
                )
                continue

            # Use all data before validation set for training
            train_end = val_start

            # Apply gap if specified
            if self.gap_weeks > 0 and train_end > 0:
                # Find games within gap_weeks of validation start
                val_start_season = data_sorted.iloc[val_start]["season"]
                val_start_week = data_sorted.iloc[val_start]["week"]

                gap_mask = (
                    (data_sorted["season"] == val_start_season)
                    & (data_sorted["week"] >= val_start_week - self.gap_weeks)
                    & (data_sorted["week"] < val_start_week)
                )

                # Exclude gap games from training
                train_mask = (data_sorted.index < val_start) & (~gap_mask)
            else:
                train_mask = data_sorted.index < train_end

            val_mask = (data_sorted.index >= val_start) & (data_sorted.index < val_end)

            train_data = data_sorted[train_mask]
            val_data = data_sorted[val_mask]

            if len(train_data) == 0 or len(val_data) == 0:
                self.logger.warning(f"Empty fold {fold}, skipping")
                continue

            # Split features and targets
            train_features = train_data.drop(columns=[target_column])
            train_targets = train_data[target_column]
            val_features = val_data.drop(columns=[target_column])
            val_targets = val_data[target_column]

            # Remove missing targets
            train_valid = train_targets.notna()
            val_valid = val_targets.notna()

            if train_valid.sum() == 0 or val_valid.sum() == 0:
                self.logger.warning(f"Invalid targets in fold {fold}, skipping")
                continue

            train_features = train_features[train_valid]
            train_targets = train_targets[train_valid]
            val_features = val_features[val_valid]
            val_targets = val_targets[val_valid]

            self.logger.info(
                f"Created fold {fold}",
                train_samples=len(train_features),
                val_samples=len(val_features),
            )

            yield train_features, train_targets, val_features, val_targets


class ModelManager:
    """
    Model serialization, loading, and versioning manager.

    Handles saving trained models with metadata, loading models,
    and managing model versions and artifacts.
    """

    def __init__(self, models_dir: str | Path | None = None):
        """
        Initialize model manager.

        Args:
            models_dir: Directory to store model artifacts
        """
        self.models_dir = (
            Path(models_dir) if models_dir else Path.cwd() / "models" / "artifacts"
        )
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(__name__)

    def save_model(
        self, model: Any, metadata: ModelMetadata, model_format: str = "joblib"
    ) -> Path:
        """
        Save a trained model with metadata.

        Args:
            model: Trained model object
            metadata: Model metadata
            model_format: Format to save model ('joblib', 'pickle')

        Returns:
            Path to saved model file
        """
        # Create model filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"{metadata.model_name}_{metadata.target_type}_{timestamp}"

        # Save model
        if model_format == "joblib":
            model_path = self.models_dir / f"{model_filename}.joblib"
            joblib.dump(model, model_path)
        elif model_format == "pickle":
            model_path = self.models_dir / f"{model_filename}.pkl"
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
        else:
            raise ValueError(f"Unsupported model format: {model_format}")

        # Save metadata
        metadata_path = self.models_dir / f"{model_filename}_metadata.json"
        metadata_dict = {
            "model_name": metadata.model_name,
            "model_type": metadata.model_type,
            "target_type": metadata.target_type,
            "train_seasons": metadata.train_seasons,
            "features_used": metadata.features_used,
            "training_date": metadata.training_date.isoformat(),
            "performance_metrics": metadata.performance_metrics,
            "hyperparameters": metadata.hyperparameters,
            "feature_importance": metadata.feature_importance,
            "version": metadata.version,
            "notes": metadata.notes,
            "model_path": str(model_path),
            "model_format": model_format,
        }

        with open(metadata_path, "w") as f:
            json.dump(metadata_dict, f, indent=2)

        self.logger.info(
            "Saved model", model_path=str(model_path), metadata_path=str(metadata_path)
        )

        return model_path

    def load_model(self, model_path: str | Path) -> tuple[Any, ModelMetadata]:
        """
        Load a trained model with its metadata.

        Args:
            model_path: Path to model file

        Returns:
            Tuple of (model_object, metadata)
        """
        model_path = Path(model_path)

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load model
        if model_path.suffix == ".joblib":
            model = joblib.load(model_path)
        elif model_path.suffix == ".pkl":
            with open(model_path, "rb") as f:
                model = pickle.load(f)
        else:
            raise ValueError(f"Unsupported model file format: {model_path.suffix}")

        # Load metadata
        metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"

        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata_dict = json.load(f)

            metadata = ModelMetadata(
                model_name=metadata_dict["model_name"],
                model_type=metadata_dict["model_type"],
                target_type=metadata_dict["target_type"],
                train_seasons=metadata_dict["train_seasons"],
                features_used=metadata_dict["features_used"],
                training_date=datetime.fromisoformat(metadata_dict["training_date"]),
                performance_metrics=metadata_dict["performance_metrics"],
                hyperparameters=metadata_dict["hyperparameters"],
                feature_importance=metadata_dict.get("feature_importance"),
                version=metadata_dict.get("version", "1.0.0"),
                notes=metadata_dict.get("notes", ""),
            )
        else:
            # Create minimal metadata if not found
            self.logger.warning(f"Metadata file not found for {model_path}")
            metadata = ModelMetadata(
                model_name=model_path.stem,
                model_type="unknown",
                target_type="unknown",
                train_seasons=[],
                features_used=[],
                training_date=datetime.now(),
                performance_metrics={},
                hyperparameters={},
            )

        self.logger.info(
            "Loaded model", model_path=str(model_path), model_type=metadata.model_type
        )

        return model, metadata

    def list_models(
        self, target_type: str | None = None, model_type: str | None = None
    ) -> list[dict[str, Any]]:
        """
        List available models with filtering.

        Args:
            target_type: Filter by target type ('wp', 'ats', 'ou')
            model_type: Filter by model type

        Returns:
            List of model information dictionaries
        """
        model_files = list(self.models_dir.glob("*.joblib")) + list(
            self.models_dir.glob("*.pkl")
        )
        models_info = []

        for model_path in model_files:
            metadata_path = model_path.parent / f"{model_path.stem}_metadata.json"

            if metadata_path.exists():
                try:
                    with open(metadata_path) as f:
                        metadata_dict = json.load(f)

                    # Apply filters
                    if target_type and metadata_dict.get("target_type") != target_type:
                        continue
                    if model_type and metadata_dict.get("model_type") != model_type:
                        continue

                    models_info.append(
                        {
                            "model_path": str(model_path),
                            "model_name": metadata_dict.get("model_name"),
                            "model_type": metadata_dict.get("model_type"),
                            "target_type": metadata_dict.get("target_type"),
                            "training_date": metadata_dict.get("training_date"),
                            "train_seasons": metadata_dict.get("train_seasons"),
                            "performance_metrics": metadata_dict.get(
                                "performance_metrics", {}
                            ),
                            "version": metadata_dict.get("version", "1.0.0"),
                        }
                    )
                except Exception as e:
                    self.logger.warning(f"Error reading metadata for {model_path}: {e}")

        # Sort by training date (newest first)
        models_info.sort(key=lambda x: x.get("training_date", ""), reverse=True)

        self.logger.info(
            "Listed models",
            total_models=len(models_info),
            target_filter=target_type,
            model_filter=model_type,
        )

        return models_info


class TrainingPipeline:
    """
    Orchestrates model training pipelines with walk-forward validation.

    Coordinates data loading, feature selection, model training,
    validation, and result tracking.
    """

    def __init__(
        self,
        models_dir: str | Path | None = None,
        results_dir: str | Path | None = None,
    ):
        """
        Initialize training pipeline.

        Args:
            models_dir: Directory to store model artifacts
            results_dir: Directory to store training results
        """
        self.model_manager = ModelManager(models_dir)
        self.results_dir = (
            Path(results_dir) if results_dir else Path.cwd() / "outputs" / "training"
        )
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(__name__)

    def run_walk_forward_training(
        self,
        data: pd.DataFrame,
        target_column: str,
        model_class: Any,
        model_params: dict[str, Any],
        model_name: str,
        target_type: str,
        start_season: int = 2018,
        end_season: int = 2024,
        save_models: bool = True,
    ) -> dict[str, Any]:
        """
        Run walk-forward training and validation.

        Args:
            data: DataFrame with features and targets
            target_column: Name of target column
            model_class: Model class to instantiate
            model_params: Model hyperparameters
            model_name: Name for the model
            target_type: Target type ('wp', 'ats', 'ou')
            start_season: First season for training
            end_season: Last season for validation
            save_models: Whether to save trained models

        Returns:
            Dictionary with training results and performance metrics
        """
        validator = WalkForwardValidator(min_train_seasons=1)
        results = {
            "model_name": model_name,
            "target_type": target_type,
            "start_season": start_season,
            "end_season": end_season,
            "model_params": model_params,
            "splits": [],
            "overall_metrics": {},
            "training_date": datetime.now().isoformat(),
        }

        self.logger.info(
            "Starting walk-forward training",
            model_name=model_name,
            target_type=target_type,
            seasons=f"{start_season}-{end_season}",
        )

        all_predictions = []
        all_actuals = []
        trained_models = []

        for split in validator.create_seasonal_splits(
            data, target_column, start_season, end_season
        ):
            split_results = {
                "test_season": int(split.test_season),
                "train_seasons": [int(s) for s in split.train_seasons],
                "train_games": len(split.train_data),
                "test_games": len(split.test_data),
                "features_used": list(split.train_data.columns),
            }

            try:
                # Train model
                model = model_class(**model_params)
                model.fit(split.train_data, split.train_targets)

                # Make predictions
                if hasattr(model, "predict_proba"):
                    # For classification, get probabilities
                    pred_proba = model.predict_proba(split.test_data)
                    if pred_proba.shape[1] == 2:
                        predictions = pred_proba[:, 1]  # Probability of positive class
                    else:
                        predictions = pred_proba
                else:
                    # For regression
                    predictions = model.predict(split.test_data)

                split_results["predictions"] = predictions.tolist()
                split_results["actuals"] = split.test_targets.tolist()

                # Collect for overall metrics
                all_predictions.extend(predictions)
                all_actuals.extend(split.test_targets)

                # Save model if requested
                if save_models:
                    # Get feature importance if available
                    feature_importance = None
                    if hasattr(model, "feature_importances_"):
                        feature_importance = dict(
                            zip(
                                split.train_data.columns,
                                model.feature_importances_,
                                strict=False,
                            )
                        )
                    elif hasattr(model, "coef_"):
                        feature_importance = dict(
                            zip(
                                split.train_data.columns,
                                model.coef_.flatten()
                                if model.coef_.ndim > 1
                                else model.coef_,
                                strict=False,
                            )
                        )

                    # Create metadata
                    metadata = ModelMetadata(
                        model_name=f"{model_name}_season_{split.test_season}",
                        model_type=model_class.__name__,
                        target_type=target_type,
                        train_seasons=[int(s) for s in split.train_seasons],
                        features_used=list(split.train_data.columns),
                        training_date=datetime.now(),
                        performance_metrics={},  # Will be filled later
                        hyperparameters=model_params,
                        feature_importance=feature_importance,
                    )

                    model_path = self.model_manager.save_model(model, metadata)
                    split_results["model_path"] = str(model_path)
                    trained_models.append((model, metadata))

                split_results["success"] = True
                self.logger.info(
                    f"Completed training for season {split.test_season}",
                    train_games=len(split.train_data),
                    test_games=len(split.test_data),
                )

            except Exception as e:
                split_results["success"] = False
                split_results["error"] = str(e)
                self.logger.error(
                    f"Training failed for season {split.test_season}", error=str(e)
                )

            results["splits"].append(split_results)

        # Calculate overall metrics
        if all_predictions and all_actuals:
            results["overall_metrics"] = self._calculate_metrics(
                all_predictions, all_actuals, target_type
            )

        # Save results
        results_file = self.results_dir / f"{model_name}_{target_type}_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        self.logger.info(
            "Walk-forward training completed",
            model_name=model_name,
            successful_splits=sum(1 for s in results["splits"] if s.get("success")),
            total_splits=len(results["splits"]),
        )

        return results

    def _calculate_metrics(
        self, predictions: list[float], actuals: list[float], target_type: str
    ) -> dict[str, float]:
        """
        Calculate performance metrics for predictions.

        Args:
            predictions: Model predictions
            actuals: Actual target values
            target_type: Type of target for appropriate metrics

        Returns:
            Dictionary of performance metrics
        """
        predictions = np.array(predictions)
        actuals = np.array(actuals)

        metrics = {}

        # Common metrics
        mse = np.mean((predictions - actuals) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - actuals))

        metrics.update(
            {
                "mse": float(mse),
                "rmse": float(rmse),
                "mae": float(mae),
                "n_samples": len(predictions),
            }
        )

        # Target-specific metrics
        if target_type == "wp":
            # Win probability metrics (binary classification)
            # Convert probabilities to binary predictions
            binary_preds = (predictions > 0.5).astype(int)
            accuracy = np.mean(binary_preds == actuals)

            # Log loss (if actuals are binary)
            if set(actuals) <= {0, 1}:
                # Clip predictions to avoid log(0)
                clipped_preds = np.clip(predictions, 1e-15, 1 - 1e-15)
                log_loss = -np.mean(
                    actuals * np.log(clipped_preds)
                    + (1 - actuals) * np.log(1 - clipped_preds)
                )
                metrics["log_loss"] = float(log_loss)

            metrics["accuracy"] = float(accuracy)

        elif target_type in ["ats", "ou"]:
            # Spread/total metrics (can be both classification and regression)
            # If binary (cover/not cover), calculate accuracy
            if set(actuals) <= {0, 1}:
                binary_preds = (predictions > 0.5).astype(int)
                accuracy = np.mean(binary_preds == actuals)
                metrics["accuracy"] = float(accuracy)

            # Correlation
            if len(predictions) > 1:
                correlation = np.corrcoef(predictions, actuals)[0, 1]
                if not np.isnan(correlation):
                    metrics["correlation"] = float(correlation)

        return metrics


def probability_to_odds(probability: float) -> int:
    """
    Convert probability to American odds.

    Args:
        probability: Probability between 0 and 1

    Returns:
        American odds (negative for favorites, positive for underdogs)
    """
    if probability <= 0 or probability >= 1:
        raise ValueError("Probability must be between 0 and 1")

    if probability > 0.5:
        # Favorite (negative odds)
        return int(-100 * probability / (1 - probability))
    # Underdog (positive odds)
    return int(100 * (1 - probability) / probability)


def odds_to_probability(american_odds: int) -> float:
    """
    Convert American odds to probability.

    Args:
        american_odds: American odds (negative for favorites, positive for underdogs)

    Returns:
        Probability between 0 and 1
    """
    if american_odds == 0:
        raise ValueError("American odds cannot be 0")

    if american_odds < 0:
        # Favorite
        return abs(american_odds) / (abs(american_odds) + 100)
    # Underdog
    return 100 / (american_odds + 100)


def kelly_criterion(win_probability: float, decimal_odds: float) -> float:
    """
    Calculate Kelly criterion bet size.

    Args:
        win_probability: Probability of winning the bet
        decimal_odds: Decimal odds (e.g., 2.0 for even money)

    Returns:
        Kelly fraction (proportion of bankroll to bet)
    """
    if win_probability <= 0 or win_probability >= 1:
        raise ValueError("Win probability must be between 0 and 1")

    if decimal_odds <= 1:
        raise ValueError("Decimal odds must be greater than 1")

    # Kelly formula: f = (bp - q) / b
    # where b = decimal_odds - 1, p = win_probability, q = 1 - win_probability
    b = decimal_odds - 1
    p = win_probability
    q = 1 - win_probability

    kelly_fraction = (b * p - q) / b

    # Return 0 if negative (no bet recommended)
    return max(0, kelly_fraction)
