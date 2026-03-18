"""
Walk-Forward Backtesting System for NFL Prediction System.

This module implements rigorous walk-forward validation with strict temporal ordering,
data leakage prevention, and comprehensive performance tracking across multiple seasons.
"""

import json
import os
import pickle
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from utils.exceptions import BacktestError, DataValidationError, ModelTrainingError
from utils.logging_config import get_logger

logger = get_logger(__name__)


class BacktestPhase(Enum):
    """Phases of walk-forward backtesting."""

    INITIALIZATION = "initialization"
    DATA_PREPARATION = "data_preparation"
    FEATURE_BUILDING = "feature_building"
    MODEL_TRAINING = "model_training"
    PREDICTION = "prediction"
    EVALUATION = "evaluation"
    VALIDATION = "validation"
    COMPLETION = "completion"


class ValidationLevel(Enum):
    """Levels of validation strictness."""

    STRICT = "strict"  # Maximum validation, slowest
    MODERATE = "moderate"  # Balanced validation
    BASIC = "basic"  # Minimal validation, fastest


@dataclass
class SeasonSplit:
    """Defines training and validation periods for a season."""

    validation_season: int
    validation_weeks: list[int]

    # Training data includes all prior seasons + early weeks of validation season
    training_seasons: list[int]
    training_weeks_current_season: list[
        int
    ]  # Weeks from validation season used for training

    # Metadata
    total_training_games: int = 0
    total_validation_games: int = 0
    split_date: datetime | None = None


@dataclass
class BacktestConfig:
    """Configuration for walk-forward backtesting."""

    # Season range
    start_season: int = 2018
    end_season: int = 2024

    # Validation strategy
    min_training_seasons: int = 3  # Minimum seasons needed before validation
    validation_start_week: int = 5  # Start validation from week N of each season

    # Data paths
    data_path: str = "data"
    models_path: str = "artifacts/models"
    results_path: str = "outputs/backtest"

    # Model types to backtest
    model_types: list[str] = field(default_factory=lambda: ["wp", "ats", "ou"])

    # Validation settings
    validation_level: ValidationLevel = ValidationLevel.MODERATE
    check_data_leakage: bool = True
    validate_temporal_order: bool = True
    save_intermediate_results: bool = True

    # Performance settings
    parallel_models: bool = False
    cache_features: bool = True
    checkpoint_frequency: int = 5  # Save progress every N weeks

    # Betting simulation
    simulate_betting: bool = True
    starting_bankroll: float = 10000.0
    bet_selection_criteria: dict[str, Any] = field(default_factory=dict)


@dataclass
class BacktestResult:
    """Results from a single backtest period."""

    season: int
    week: int
    model_type: str

    # Performance metrics
    predictions_made: int
    accuracy: float | None = None
    log_loss: float | None = None
    brier_score: float | None = None
    mae: float | None = None
    rmse: float | None = None

    # Betting metrics (if applicable)
    bets_placed: int = 0
    betting_roi: float | None = None
    betting_profit: float | None = None
    units_wagered: float | None = None

    # Metadata
    execution_time: float | None = None
    data_quality_score: float | None = None
    feature_count: int | None = None
    model_version: str | None = None


@dataclass
class BacktestSummary:
    """Summary of complete backtesting run."""

    config: BacktestConfig
    total_seasons: int
    total_weeks: int
    total_predictions: int

    # Aggregated performance
    results_by_season: dict[int, list[BacktestResult]]
    results_by_model: dict[str, list[BacktestResult]]

    # Summary metrics
    overall_metrics: dict[str, float]
    seasonal_trends: dict[str, list[float]]

    # Execution metadata
    start_time: datetime
    end_time: datetime | None = None
    total_execution_time: float | None = None
    validation_errors: list[str] = field(default_factory=list)

    # Betting simulation results
    final_bankroll: float | None = None
    total_roi: float | None = None
    sharpe_ratio: float | None = None


class DataLeakageValidator:
    """Validates that no future data leaks into training."""

    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode
        self.violations = []

    def validate_training_data(
        self, training_data: pd.DataFrame, validation_cutoff: datetime
    ) -> bool:
        """Validate that training data contains no future information."""

        if (
            "date" not in training_data.columns
            and "game_date" not in training_data.columns
        ):
            logger.warning("No date column found for leakage validation")
            return True

        date_col = "date" if "date" in training_data.columns else "game_date"

        # Check for future dates
        future_data = training_data[training_data[date_col] >= validation_cutoff]

        if len(future_data) > 0:
            violation = f"Found {len(future_data)} training records from future dates"
            self.violations.append(violation)
            logger.error(violation)

            if self.strict_mode:
                raise DataValidationError(f"Data leakage detected: {violation}")

            return False

        return True

    def validate_feature_timestamps(
        self, features: pd.DataFrame, prediction_date: datetime
    ) -> bool:
        """Validate that all features are based on data before prediction date."""

        # Check for any timestamp columns that might indicate leakage
        timestamp_cols = [
            col
            for col in features.columns
            if "timestamp" in col.lower() or "date" in col.lower()
        ]

        for col in timestamp_cols:
            if features[col].dtype == "datetime64[ns]":
                future_features = features[features[col] >= prediction_date]
                if len(future_features) > 0:
                    violation = f"Feature column '{col}' contains {len(future_features)} future timestamps"
                    self.violations.append(violation)
                    logger.error(violation)

                    if self.strict_mode:
                        raise DataValidationError(
                            f"Feature leakage detected: {violation}"
                        )

                    return False

        return True


class WalkForwardBacktester:
    """
    Implements rigorous walk-forward backtesting for NFL prediction models.

    Ensures strict temporal ordering and prevents data leakage while providing
    comprehensive performance evaluation across multiple seasons.
    """

    def __init__(self, config: BacktestConfig):
        """Initialize backtester with configuration."""
        self.config = config
        self.leakage_validator = DataLeakageValidator(
            strict_mode=(config.validation_level == ValidationLevel.STRICT)
        )

        # Create output directories
        self._create_directories()

        logger.info(
            f"WalkForwardBacktester initialized for seasons {config.start_season}-{config.end_season}"
        )

    def run_backtest(
        self,
        data_loader: Callable,
        feature_builder: Callable,
        model_trainers: dict[str, Callable],
        evaluator: Callable,
    ) -> BacktestSummary:
        """
        Execute complete walk-forward backtesting.

        Args:
            data_loader: Function to load historical data
            feature_builder: Function to build features from raw data
            model_trainers: Dict of model type -> training function
            evaluator: Function to evaluate predictions

        Returns:
            BacktestSummary with complete results
        """

        logger.info("Starting walk-forward backtesting")
        start_time = datetime.now()

        # Initialize tracking
        all_results = []
        results_by_season = {}
        results_by_model = {model_type: [] for model_type in self.config.model_types}

        try:
            # Generate season splits
            season_splits = self._generate_season_splits()
            total_splits = len(season_splits)

            logger.info(f"Generated {total_splits} season splits for validation")

            # Execute backtest for each split
            for i, split in enumerate(season_splits):
                logger.info(
                    f"Processing split {i + 1}/{total_splits}: Season {split.validation_season}"
                )

                try:
                    split_results = self._execute_season_split(
                        split, data_loader, feature_builder, model_trainers, evaluator
                    )

                    # Aggregate results
                    all_results.extend(split_results)

                    if split.validation_season not in results_by_season:
                        results_by_season[split.validation_season] = []
                    results_by_season[split.validation_season].extend(split_results)

                    for result in split_results:
                        results_by_model[result.model_type].append(result)

                    # Checkpoint progress
                    if (i + 1) % self.config.checkpoint_frequency == 0:
                        self._save_checkpoint(all_results, i + 1)

                except (
                    ValueError,
                    KeyError,
                    TypeError,
                    RuntimeError,
                    FileNotFoundError,
                ) as e:
                    logger.error(f"Error in split {i + 1}: {e}")
                    if self.config.validation_level == ValidationLevel.STRICT:
                        raise BacktestError(
                            f"Backtest failed at split {i + 1}: {e}"
                        ) from e
                    # Continue with next split in non-strict mode
                    continue

            # Create final summary
            end_time = datetime.now()
            execution_time = (end_time - start_time).total_seconds()

            summary = BacktestSummary(
                config=self.config,
                total_seasons=len(results_by_season),
                total_weeks=sum(len(results) for results in results_by_season.values()),
                total_predictions=sum(r.predictions_made for r in all_results),
                results_by_season=results_by_season,
                results_by_model=results_by_model,
                overall_metrics=self._calculate_overall_metrics(all_results),
                seasonal_trends=self._calculate_seasonal_trends(results_by_season),
                start_time=start_time,
                end_time=end_time,
                total_execution_time=execution_time,
                validation_errors=self.leakage_validator.violations,
            )

            # Save final results
            self._save_final_results(summary)

            logger.info(f"Backtesting completed in {execution_time:.1f} seconds")
            logger.info(
                f"Processed {summary.total_predictions} predictions across {summary.total_weeks} weeks"
            )

            return summary

        except (BacktestError, DataValidationError, ModelTrainingError) as e:
            logger.error(f"Backtesting failed: {e}")
            raise BacktestError(f"Walk-forward backtest failed: {e}") from e

    def _generate_season_splits(self) -> list[SeasonSplit]:
        """Generate all season splits for walk-forward validation."""

        splits = []

        for validation_season in range(
            self.config.start_season + self.config.min_training_seasons,
            self.config.end_season + 1,
        ):
            # Training seasons: all previous seasons
            training_seasons = list(range(self.config.start_season, validation_season))

            # Validation weeks: from validation_start_week to end of season
            validation_weeks = list(
                range(self.config.validation_start_week, 19)
            )  # NFL weeks 1-18

            # Training weeks from current season: weeks before validation starts
            training_weeks_current = list(range(1, self.config.validation_start_week))

            split = SeasonSplit(
                validation_season=validation_season,
                validation_weeks=validation_weeks,
                training_seasons=training_seasons,
                training_weeks_current_season=training_weeks_current,
                split_date=datetime(
                    validation_season, 9, 1
                ),  # Approximate season start
            )

            splits.append(split)

        logger.info(f"Generated {len(splits)} season splits")
        return splits

    def _execute_season_split(
        self,
        split: SeasonSplit,
        data_loader: Callable,
        feature_builder: Callable,
        model_trainers: dict[str, Callable],
        evaluator: Callable,
    ) -> list[BacktestResult]:
        """Execute backtesting for a single season split."""

        logger.info(f"Executing season split: {split.validation_season}")

        # Phase 1: Load and validate training data
        logger.info("Phase 1: Loading training data")
        training_data = self._load_training_data(split, data_loader)

        # Phase 2: Build features
        logger.info("Phase 2: Building features")
        training_features = self._build_training_features(
            split, training_data, feature_builder
        )

        # Phase 3: Train models
        logger.info("Phase 3: Training models")
        trained_models = self._train_models(split, training_features, model_trainers)

        # Phase 4: Execute week-by-week validation
        logger.info("Phase 4: Week-by-week validation")
        week_results = []

        for week in split.validation_weeks:
            try:
                week_result = self._execute_week_validation(
                    split.validation_season,
                    week,
                    split,
                    trained_models,
                    data_loader,
                    feature_builder,
                    evaluator,
                )
                week_results.extend(week_result)

            except (ValueError, KeyError, TypeError, RuntimeError) as e:
                logger.error(
                    f"Error in week {week} of season {split.validation_season}: {e}"
                )
                if self.config.validation_level == ValidationLevel.STRICT:
                    raise

        logger.info(
            f"Completed season {split.validation_season}: {len(week_results)} results"
        )
        return week_results

    def _load_training_data(
        self, split: SeasonSplit, data_loader: Callable
    ) -> pd.DataFrame:
        """Load and validate training data for a split."""

        # Load data for all training seasons
        training_data = []

        for season in split.training_seasons:
            season_data = data_loader(season=season, weeks=None)  # All weeks
            training_data.append(season_data)

        # Add early weeks from validation season
        if split.training_weeks_current_season:
            current_season_data = data_loader(
                season=split.validation_season,
                weeks=split.training_weeks_current_season,
            )
            training_data.append(current_season_data)

        # Combine all training data
        combined_data = pd.concat(training_data, ignore_index=True)

        # Validate data leakage
        if self.config.check_data_leakage:
            validation_cutoff = datetime(
                split.validation_season, 10, 1
            )  # Conservative cutoff
            self.leakage_validator.validate_training_data(
                combined_data, validation_cutoff
            )

        logger.info(
            f"Loaded {len(combined_data)} training records for season {split.validation_season}"
        )
        return combined_data

    def _build_training_features(
        self, split: SeasonSplit, training_data: pd.DataFrame, feature_builder: Callable
    ) -> pd.DataFrame:
        """Build features for training data."""

        # Build features with strict temporal constraints
        features = feature_builder(
            data=training_data,
            as_of_date=datetime(split.validation_season, 9, 1),  # No future data
            include_targets=True,
        )

        # Validate feature timestamps
        if self.config.validate_temporal_order:
            prediction_date = datetime(split.validation_season, 9, 1)
            self.leakage_validator.validate_feature_timestamps(
                features, prediction_date
            )

        logger.info(
            f"Built {len(features)} feature records with {len(features.columns)} features"
        )
        return features

    def _train_models(
        self,
        split: SeasonSplit,
        training_features: pd.DataFrame,
        model_trainers: dict[str, Callable],
    ) -> dict[str, Any]:
        """Train all models for the split."""

        trained_models = {}

        for model_type in self.config.model_types:
            if model_type not in model_trainers:
                logger.warning(f"No trainer found for model type: {model_type}")
                continue

            try:
                logger.info(f"Training {model_type} model")
                trainer = model_trainers[model_type]

                # Train model with validation split
                model = trainer(
                    features=training_features,
                    target_column=f"{model_type}_target",
                    validation_split=0.2,
                    random_state=42,
                )

                trained_models[model_type] = model

                # Save model checkpoint
                if self.config.save_intermediate_results:
                    model_path = os.path.join(
                        self.config.models_path,
                        f"{model_type}_season_{split.validation_season}.pkl",
                    )
                    os.makedirs(os.path.dirname(model_path), exist_ok=True)

                    with open(model_path, "wb") as f:
                        pickle.dump(model, f)

                logger.info(f"Successfully trained {model_type} model")

            except (ValueError, KeyError, TypeError, RuntimeError) as e:
                logger.error(f"Failed to train {model_type} model: {e}")
                if self.config.validation_level == ValidationLevel.STRICT:
                    raise ModelTrainingError(
                        f"Model training failed for {model_type}: {e}"
                    ) from e

        return trained_models

    def _execute_week_validation(
        self,
        season: int,
        week: int,
        split: SeasonSplit,
        trained_models: dict[str, Any],
        data_loader: Callable,
        feature_builder: Callable,
        evaluator: Callable,
    ) -> list[BacktestResult]:
        """Execute validation for a single week."""

        logger.debug(f"Validating season {season}, week {week}")

        # Load validation data for this week
        week_data = data_loader(season=season, weeks=[week])

        if len(week_data) == 0:
            logger.warning(f"No data available for season {season}, week {week}")
            return []

        # Build features for this week
        week_features = feature_builder(
            data=week_data,
            as_of_date=datetime(season, 9, 1) + timedelta(weeks=week - 1),
            include_targets=False,  # No targets for prediction
        )

        results = []

        # Generate predictions for each model
        for model_type, model in trained_models.items():
            try:
                # Make predictions
                predictions = model.predict(
                    week_features.drop(columns=["game_id"], errors="ignore")
                )

                # Load actual outcomes for evaluation
                actuals = week_data[f"{model_type}_actual"].values

                # Evaluate predictions
                metrics = evaluator(predictions, actuals, model_type=model_type)

                # Create result record
                result = BacktestResult(
                    season=season,
                    week=week,
                    model_type=model_type,
                    predictions_made=len(predictions),
                    accuracy=metrics.get("accuracy"),
                    log_loss=metrics.get("log_loss"),
                    brier_score=metrics.get("brier_score"),
                    mae=metrics.get("mae"),
                    rmse=metrics.get("rmse"),
                    feature_count=len(week_features.columns),
                    model_version=f"{model_type}_v1.0",
                )

                results.append(result)

                logger.debug(
                    f"Season {season} Week {week} {model_type}: {result.predictions_made} predictions"
                )

            except (ValueError, KeyError, TypeError, RuntimeError) as e:
                logger.error(
                    f"Prediction failed for {model_type} in season {season}, week {week}: {e}"
                )
                if self.config.validation_level == ValidationLevel.STRICT:
                    raise

        return results

    def _calculate_overall_metrics(
        self, all_results: list[BacktestResult]
    ) -> dict[str, float]:
        """Calculate overall performance metrics."""

        if not all_results:
            return {}

        metrics = {}

        # Group by model type
        by_model = {}
        for result in all_results:
            if result.model_type not in by_model:
                by_model[result.model_type] = []
            by_model[result.model_type].append(result)

        # Calculate metrics for each model type
        for model_type, results in by_model.items():
            valid_results = [r for r in results if r.accuracy is not None]

            if valid_results:
                metrics[f"{model_type}_mean_accuracy"] = np.mean(
                    [r.accuracy for r in valid_results]
                )
                metrics[f"{model_type}_std_accuracy"] = np.std(
                    [r.accuracy for r in valid_results]
                )

                log_loss_results = [r for r in results if r.log_loss is not None]
                if log_loss_results:
                    metrics[f"{model_type}_mean_log_loss"] = np.mean(
                        [r.log_loss for r in log_loss_results]
                    )

                mae_results = [r for r in results if r.mae is not None]
                if mae_results:
                    metrics[f"{model_type}_mean_mae"] = np.mean(
                        [r.mae for r in mae_results]
                    )

        # Overall statistics
        metrics["total_predictions"] = sum(r.predictions_made for r in all_results)
        metrics["total_weeks"] = len({(r.season, r.week) for r in all_results})

        return metrics

    def _calculate_seasonal_trends(
        self, results_by_season: dict[int, list[BacktestResult]]
    ) -> dict[str, list[float]]:
        """Calculate seasonal performance trends."""

        trends = {}
        model_types = set()

        # Collect all model types
        for results in results_by_season.values():
            for result in results:
                model_types.add(result.model_type)

        # Calculate trends for each model type
        for model_type in model_types:
            trends[f"{model_type}_accuracy"] = []

            for season in sorted(results_by_season.keys()):
                season_results = [
                    r for r in results_by_season[season] if r.model_type == model_type
                ]
                valid_accuracies = [
                    r.accuracy for r in season_results if r.accuracy is not None
                ]

                if valid_accuracies:
                    trends[f"{model_type}_accuracy"].append(np.mean(valid_accuracies))
                else:
                    trends[f"{model_type}_accuracy"].append(None)

        return trends

    def _create_directories(self):
        """Create necessary output directories."""
        dirs_to_create = [
            self.config.results_path,
            self.config.models_path,
            os.path.join(self.config.results_path, "checkpoints"),
        ]

        for dir_path in dirs_to_create:
            os.makedirs(dir_path, exist_ok=True)

    def _save_checkpoint(self, results: list[BacktestResult], checkpoint_num: int):
        """Save intermediate results checkpoint."""
        checkpoint_path = os.path.join(
            self.config.results_path, "checkpoints", f"checkpoint_{checkpoint_num}.json"
        )

        checkpoint_data = {
            "checkpoint_number": checkpoint_num,
            "timestamp": datetime.now().isoformat(),
            "results_count": len(results),
            "results": [
                {
                    "season": r.season,
                    "week": r.week,
                    "model_type": r.model_type,
                    "predictions_made": r.predictions_made,
                    "accuracy": r.accuracy,
                    "log_loss": r.log_loss,
                }
                for r in results[-50:]  # Save last 50 results
            ],
        }

        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint_data, f, indent=2)

        logger.info(f"Saved checkpoint {checkpoint_num} with {len(results)} results")

    def _save_final_results(self, summary: BacktestSummary):
        """Save final backtesting results."""

        # Save summary as JSON
        summary_path = os.path.join(self.config.results_path, "backtest_summary.json")

        summary_data = {
            "config": {
                "start_season": summary.config.start_season,
                "end_season": summary.config.end_season,
                "model_types": summary.config.model_types,
                "validation_level": summary.config.validation_level.value,
            },
            "execution": {
                "start_time": summary.start_time.isoformat(),
                "end_time": summary.end_time.isoformat() if summary.end_time else None,
                "total_execution_time": summary.total_execution_time,
                "total_predictions": summary.total_predictions,
            },
            "performance": summary.overall_metrics,
            "trends": summary.seasonal_trends,
            "validation_errors": summary.validation_errors,
        }

        with open(summary_path, "w") as f:
            json.dump(summary_data, f, indent=2)

        # Save detailed results as pickle
        results_path = os.path.join(self.config.results_path, "detailed_results.pkl")
        with open(results_path, "wb") as f:
            pickle.dump(summary, f)

        logger.info(f"Saved final results to {self.config.results_path}")


def create_default_config() -> BacktestConfig:
    """Create default backtesting configuration."""
    return BacktestConfig(
        start_season=2018,
        end_season=2024,
        min_training_seasons=3,
        validation_start_week=5,
        validation_level=ValidationLevel.MODERATE,
        check_data_leakage=True,
        validate_temporal_order=True,
        save_intermediate_results=True,
    )


def create_strict_config() -> BacktestConfig:
    """Create strict backtesting configuration with maximum validation."""
    return BacktestConfig(
        start_season=2018,
        end_season=2024,
        min_training_seasons=3,
        validation_start_week=4,
        validation_level=ValidationLevel.STRICT,
        check_data_leakage=True,
        validate_temporal_order=True,
        save_intermediate_results=True,
        checkpoint_frequency=2,
    )
