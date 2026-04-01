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
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
from sklearn.feature_selection import SelectFromModel

from models.artifacts import save_model_artifact
from models.clv import compute_clv_for_predictions
from models.temporal import (
    TemporalSplitConfig,
    WalkForwardSplitter,
    make_temporal_cv_splits,
)
from models.tuning import OptunaTuner, TuningResult
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
        self._tuning_result: TuningResult | None = None

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

    @abstractmethod
    def _define_search_space(self, trial: optuna.Trial) -> dict:
        """Define Optuna search space using trial.suggest_* API.

        Each subclass defines its own hyperparameter search space
        using Optuna's trial.suggest_* methods (suggest_float,
        suggest_int, suggest_categorical).

        Args:
            trial: Optuna trial for parameter suggestion.

        Returns:
            Dict of parameter name to suggested value.
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

    def _make_objective(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        cv_splits: list[tuple[np.ndarray, np.ndarray]],
    ) -> Callable[[optuna.Trial], float]:
        """Create Optuna objective function for temporal CV evaluation.

        The objective trains the model on each CV fold's train split,
        evaluates on the validation split, and reports intermediate
        scores for Hyperband pruning (per D-02).

        Args:
            X_train: Training features (combined train + hp_val).
            y_train: Training targets.
            cv_splits: Temporal CV fold indices.

        Returns:
            Callable that takes an Optuna Trial and returns mean CV score.
        """

        def objective(trial: optuna.Trial) -> float:
            params = self._define_search_space(trial)
            scores = []
            for fold_idx, (train_idx, val_idx) in enumerate(cv_splits):
                model = self._create_model(params)
                model.fit(X_train.iloc[train_idx], y_train.iloc[train_idx])
                preds = self._predict_raw(model, X_train.iloc[val_idx])
                score = self._compute_cv_score(preds, y_train.iloc[val_idx])
                scores.append(score)

                # Report intermediate value for Hyperband pruning (per D-02)
                trial.report(float(np.mean(scores)), step=fold_idx)
                if trial.should_prune():
                    raise optuna.TrialPruned()

            return float(np.mean(scores))

        return objective

    def _compute_cv_score(
        self,
        predictions: np.ndarray,
        actuals: pd.Series,
    ) -> float:
        """Compute CV score for a single fold. Lower is better (minimized).

        Default implementation uses MAE (appropriate for ATS/O/U).
        WP subclass should override to use log_loss.

        Args:
            predictions: Model predictions for validation fold.
            actuals: Actual target values.

        Returns:
            Score value (lower is better for Optuna minimization).
        """
        from sklearn.metrics import mean_absolute_error

        return float(mean_absolute_error(actuals.values, predictions))

    def tune_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_trials: int = 100,
        season_week_df: pd.DataFrame | None = None,
    ) -> dict:
        """Tune hyperparameters using Optuna with temporal CV folds.

        Per D-01, delegates to OptunaTuner. Per D-02, uses TPE sampler
        with 100+ trials and Hyperband pruner. Per D-03, uses SQLite
        storage for resumability. Per D-05, uses target-specific
        optimization metric (direction derived from _get_scoring_metric).

        Args:
            X_train: Training features (train + hp_val combined).
            y_train: Training targets.
            n_trials: Number of Optuna trials (default 100 per D-02).
            season_week_df: DataFrame with "season" and "week" columns
                for temporal CV splits. If None, creates index-based splits.

        Returns:
            Best parameters dict.
        """
        # Create temporal CV folds
        if season_week_df is not None:
            cv_splits = make_temporal_cv_splits(season_week_df, n_splits=3)
        else:
            cv_splits = make_temporal_cv_splits(
                pd.DataFrame(
                    {
                        "season": [0] * len(X_train),
                        "week": range(len(X_train)),
                    }
                ),
                n_splits=3,
            )

        # Determine optimization direction from scoring metric
        # neg_log_loss and neg_mean_absolute_error are both "higher is better"
        # in sklearn convention, but we want to minimize the raw metric
        direction = "minimize"

        study_name = f"{self.target}_tuning_v1"

        tuner = OptunaTuner(
            study_name=study_name,
            direction=direction,
            n_trials=n_trials,
        )

        objective = self._make_objective(X_train, y_train, cv_splits)
        result = tuner.optimize(objective)

        # Replay best params through _define_search_space to get
        # model-compatible parameter names (e.g., "solver_l2" -> "solver")
        fixed_trial = optuna.trial.FixedTrial(result.best_params)
        best_params = self._define_search_space(fixed_trial)

        self.logger.info(
            "Optuna tuning completed",
            target=self.target,
            best_value=result.best_value,
            best_params=best_params,
            n_trials=result.n_trials,
            top_importances=dict(list(result.param_importances.items())[:5]),
        )

        # Store tuning result for later use in save()
        self._tuning_result = result

        return best_params

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

        Passes tuning metadata (if available) to save_model_artifact
        for JSON params sidecar creation (per D-12, D-13).

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

        # Get tuning result if available
        tuning_result = getattr(self, "_tuning_result", None)
        best_params = self.metadata.get("best_params")
        tuning_metadata = None
        if tuning_result is not None:
            tuning_metadata = {
                "study_name": tuning_result.study_name,
                "best_value": tuning_result.best_value,
                "n_trials": tuning_result.n_trials,
                "param_importances": tuning_result.param_importances,
                "optimization_metric": self._get_scoring_metric(),
            }

        return save_model_artifact(
            model=self.model,
            target=self.target,
            metadata=self.metadata,
            feature_list=self.feature_names,
            calibrator=self.calibrator,
            best_params=best_params,
            tuning_metadata=tuning_metadata,
            artifacts_dir=artifacts_dir,
        )

    # ------------------------------------------------------------------
    # Overridable hooks
    # ------------------------------------------------------------------

    def _get_param_distributions(self) -> dict:
        """Return parameter distributions for RandomizedSearchCV.

        .. deprecated::
            No longer used by BaseTrainer. Tuning now delegates to
            OptunaTuner via _define_search_space(). Kept for backward
            compatibility with any external code referencing this method.

        Returns:
            Empty dict (no-op).
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
