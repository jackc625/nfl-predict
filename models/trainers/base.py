"""Abstract base trainer for WP, ATS, and O/U models.

Provides the shared interface that all model trainers implement:
- Feature selection on initial training window
- Hyperparameter tuning with temporal CV
- Walk-forward training and evaluation on holdout
- Versioned artifact saving

Subclasses implement model-specific logic via abstract methods.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import RandomizedSearchCV

from models.artifacts import save_model_artifact
from models.clv import compute_clv_for_predictions
from models.temporal import (
    TemporalSplitConfig,
    WalkForwardSplitter,
    make_temporal_cv_splits,
)
from utils import get_logger


class BaseTrainer(ABC):
    """Abstract base class for model trainers.

    Provides the shared orchestration logic for training any of the three
    model types (WP, ATS, O/U) while delegating model-specific behavior
    to subclass abstract methods.

    Attributes:
        target: Model target type ("wp", "ats", "ou").
        config: Temporal split configuration.
        model: Trained model (set after training).
        calibrator: Fitted calibrator (set after calibration).
        feature_names: Selected feature names (set after feature selection).
        metadata: Training metadata dict (populated during training).
    """

    def __init__(
        self,
        target: str,
        config: TemporalSplitConfig | None = None,
    ) -> None:
        """Initialize the trainer.

        Args:
            target: Model target type ("wp", "ats", "ou").
            config: Temporal split configuration. Defaults to default split.
        """
        if target not in ("wp", "ats", "ou"):
            msg = f"target must be 'wp', 'ats', or 'ou', got '{target}'"
            raise ValueError(msg)

        self.target = target
        self.config = config or TemporalSplitConfig.default()
        self.config.validate()

        self.logger = get_logger(f"models.trainers.{target}")

        self.model: Any = None
        self.calibrator: Any = None
        self.feature_names: list[str] = []
        self.metadata: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Abstract methods -- subclasses must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def _create_model(self, params: dict) -> Any:
        """Create a new model instance with the given parameters.

        Args:
            params: Model hyperparameters.

        Returns:
            Unfitted model instance.
        """

    @abstractmethod
    def _get_target_column(self) -> str:
        """Return the name of the target column in the feature matrix.

        Returns:
            Target column name (e.g., "home_win", "home_margin", "total_points").
        """

    @abstractmethod
    def _predict_raw(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        """Generate raw predictions from a fitted model.

        Args:
            model: Fitted model.
            X: Feature matrix.

        Returns:
            Array of raw predictions.
        """

    @abstractmethod
    def _get_default_params(self) -> dict:
        """Return default hyperparameters for this model type.

        Returns:
            Dict of default parameter values.
        """

    # ------------------------------------------------------------------
    # Concrete methods -- shared across all trainers
    # ------------------------------------------------------------------

    def select_features(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        max_features: int | None = None,
    ) -> list[str]:
        """Select features using model-based importance on the training window.

        Uses SelectFromModel with the model type to rank features.
        If max_features is specified, limits to that count.

        Args:
            X: Training features.
            y: Training targets.
            max_features: Maximum number of features. None = use all that pass.

        Returns:
            Sorted list of selected feature names.
        """
        model = self._create_model(self._get_default_params())
        model.fit(X, y)

        selector = SelectFromModel(model, max_features=max_features, prefit=True)
        selected_mask = selector.get_support()
        selected_features = sorted(X.columns[selected_mask].tolist())

        self.logger.info(
            "Feature selection completed",
            total_features=len(X.columns),
            selected_features=len(selected_features),
            max_features=max_features,
        )

        return selected_features

    def tune_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_iter: int = 50,
    ) -> dict:
        """Tune hyperparameters using temporal CV within the training window.

        Uses make_temporal_cv_splits for CV folds and RandomizedSearchCV
        for parameter search. Subclasses should override
        _get_param_distributions() to define the search space.

        Args:
            X_train: Training features (train + hp_val combined).
            y_train: Training targets.
            n_iter: Number of random parameter combinations to try.

        Returns:
            Best parameters dict.
        """
        param_distributions = self._get_param_distributions()
        if not param_distributions:
            self.logger.info("No param distributions defined, using defaults")
            return self._get_default_params()

        # Create temporal CV folds
        # We need a df with season and week for temporal splits
        # Since we're working with feature matrices, create index-based splits
        cv_splits = make_temporal_cv_splits(
            pd.DataFrame({"season": [0] * len(X_train), "week": range(len(X_train))}),
            n_splits=3,
        )

        model = self._create_model(self._get_default_params())

        search = RandomizedSearchCV(
            model,
            param_distributions,
            n_iter=n_iter,
            cv=cv_splits,
            scoring=self._get_scoring_metric(),
            random_state=42,
            n_jobs=1,
            verbose=0,
        )

        search.fit(X_train, y_train)

        self.logger.info(
            "Hyperparameter tuning completed",
            best_score=search.best_score_,
            best_params=search.best_params_,
            n_iter=n_iter,
        )

        return search.best_params_

    def train_and_evaluate(
        self,
        features_df: pd.DataFrame,
        closing_odds_df: pd.DataFrame | None = None,
    ) -> dict:
        """Orchestrate full training and evaluation pipeline.

        1. Select features on training window
        2. Tune hyperparameters on train + HP-val window
        3. Walk-forward through holdout seasons
        4. Collect per-season metrics
        5. Compute CLV if closing odds provided

        Args:
            features_df: Full feature matrix with ID cols, features, and target.
            closing_odds_df: Optional DataFrame with closing odds for CLV.

        Returns:
            Dict with per-season metrics, overall metrics, feature names,
            best parameters, and CLV results.
        """
        target_col = self._get_target_column()
        splitter = WalkForwardSplitter(
            config=self.config,
            target_col=target_col,
        )

        # Step 1: Feature selection on training window
        train_val_split = splitter.get_train_val_split(features_df)
        self.feature_names = self.select_features(
            train_val_split.train_data,
            train_val_split.train_targets,
        )

        self.logger.info(
            "Features selected",
            n_features=len(self.feature_names),
            features=self.feature_names[:10],
        )

        # Step 2: Tune hyperparameters on train + hp_val
        combined_train = pd.concat(
            [train_val_split.train_data, train_val_split.test_data]
        )
        combined_targets = pd.concat(
            [train_val_split.train_targets, train_val_split.test_targets]
        )
        best_params = self.tune_hyperparameters(
            combined_train[self.feature_names],
            combined_targets,
        )

        # Step 3: Walk-forward through holdout
        season_results = []
        all_predictions = []

        for split in splitter.generate_splits(features_df):
            X_train = split.train_data[self.feature_names]
            y_train = split.train_targets
            X_test = split.test_data[self.feature_names]
            y_test = split.test_targets

            model = self._create_model(best_params)
            model.fit(X_train, y_train)

            predictions = self._predict_raw(model, X_test)

            season_metrics = self._compute_season_metrics(
                predictions, y_test.values, split.test_season
            )
            season_results.append(season_metrics)

            # Collect predictions for CLV
            if closing_odds_df is not None:
                pred_df = pd.DataFrame(
                    {
                        "game_id": split.test_data.index,
                        "model_prob": predictions,
                        "season": split.test_season,
                    }
                )
                all_predictions.append(pred_df)

            self.logger.info(
                "Holdout season evaluated",
                test_season=split.test_season,
                n_train=len(X_train),
                n_test=len(X_test),
                metrics=season_metrics,
            )

        # Store the final model (trained on all data before last holdout)
        self.model = model
        self.metadata = {
            "target": self.target,
            "feature_names": self.feature_names,
            "best_params": best_params,
            "season_results": season_results,
            "config": {
                "train_seasons": self.config.train_seasons,
                "hp_val_seasons": self.config.hp_val_seasons,
                "holdout_seasons": self.config.holdout_seasons,
            },
        }

        # Step 4: Compute CLV if closing odds provided
        clv_results = None
        if closing_odds_df is not None and all_predictions:
            predictions_combined = pd.concat(all_predictions, ignore_index=True)
            clv_results = compute_clv_for_predictions(
                predictions_combined,
                closing_odds_df,
                target=self.target,
            )
            self.metadata["clv_summary"] = {
                "mean_clv": float(clv_results["probability_clv"].mean()),
                "n_games_with_odds": int(clv_results["has_closing_odds"].sum()),
            }

        return {
            "season_results": season_results,
            "feature_names": self.feature_names,
            "best_params": best_params,
            "clv_results": clv_results,
            "metadata": self.metadata,
        }

    def save(self, artifacts_dir: Path = Path("artifacts")) -> Path:
        """Save the trained model and metadata as a versioned artifact.

        Args:
            artifacts_dir: Root directory for artifacts.

        Returns:
            Path to the created artifact directory.

        Raises:
            RuntimeError: If model has not been trained yet.
        """
        if self.model is None:
            msg = "Cannot save: model has not been trained yet"
            raise RuntimeError(msg)

        return save_model_artifact(
            model=self.model,
            target=self.target,
            metadata=self.metadata,
            feature_list=self.feature_names,
            calibrator=self.calibrator,
            artifacts_dir=artifacts_dir,
        )

    # ------------------------------------------------------------------
    # Overridable hooks
    # ------------------------------------------------------------------

    def _get_param_distributions(self) -> dict:
        """Return parameter distributions for RandomizedSearchCV.

        Subclasses should override this to define their search space.
        Returns empty dict to skip tuning and use defaults.
        """
        return {}

    def _get_scoring_metric(self) -> str:
        """Return the scoring metric for hyperparameter tuning.

        Subclasses can override for target-specific metrics.
        """
        return "neg_log_loss"

    def _compute_season_metrics(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        season: int,
    ) -> dict[str, Any]:
        """Compute evaluation metrics for a single holdout season.

        Subclasses can override for target-specific metrics.

        Args:
            predictions: Model predictions.
            actuals: Actual target values.
            season: Season year.

        Returns:
            Dict of metric name to value.
        """
        from sklearn.metrics import accuracy_score, mean_absolute_error

        metrics: dict[str, Any] = {"season": season, "n_games": len(predictions)}

        # Binary classification metrics (WP, ATS cover, O/U over)
        if set(np.unique(actuals)) <= {0, 1}:
            binary_preds = (predictions > 0.5).astype(int)
            metrics["accuracy"] = float(accuracy_score(actuals, binary_preds))

        # Regression metrics
        metrics["mae"] = float(mean_absolute_error(actuals, predictions))

        return metrics
