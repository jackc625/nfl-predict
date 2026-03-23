"""O/U (Over/Under) model trainer using XGBoost regression.

Predicts total points (home_score + away_score) and converts to
over/under probabilities via TotalDistributionConverter fitted on
HP-validation residuals.

Extends BaseTrainer from Plan 01. Reuses the existing
TotalDistributionConverter from models.train_ou.

No weather sub-model (redundant with main model features).
No sub-model for individual team scoring (dropped per research).

Requirements addressed: MODL-04 (O/U XGBoost), MODL-07 (feature count).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from models.temporal import TemporalSplitConfig, WalkForwardSplitter
from models.train_ou import TotalDistributionConverter
from models.trainers.base import BaseTrainer
from utils import get_logger

logger = get_logger(__name__)

# Soft target for feature count (MODL-07: ~21 features for O/U)
_OU_MAX_FEATURES = 25


class OUTrainer(BaseTrainer):
    """O/U (Over/Under) trainer using XGBoost regression for total points prediction.

    Predicts the combined score (home_score + away_score).
    Over/under probabilities are derived from the residual distribution
    fitted on HP-validation predictions -- regression only, no sub-models.

    Attributes:
        total_converter: Fitted TotalDistributionConverter for
            converting total predictions to over/under probabilities.
    """

    def __init__(self, config: TemporalSplitConfig | None = None) -> None:
        """Initialize the O/U trainer.

        Args:
            config: Temporal split configuration. Defaults to default split.
        """
        super().__init__(target="ou", config=config)
        self.total_converter: TotalDistributionConverter | None = None

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def _get_target_column(self) -> str:
        """Return the target column for O/U: total_points."""
        return "total_points"

    def _create_model(self, params: dict) -> XGBRegressor:
        """Create an XGBRegressor with the given parameters.

        Args:
            params: Model hyperparameters (merged with defaults).

        Returns:
            Unfitted XGBRegressor instance.
        """
        default_params = self._get_default_params()
        default_params.update(params)
        return XGBRegressor(**default_params)

    def _get_default_params(self) -> dict:
        """Return default XGBoost hyperparameters for O/U total prediction."""
        return {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 1.0,
            "random_state": 42,
            "verbosity": 0,
            "n_jobs": -1,
        }

    def _predict_raw(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        """Generate raw total points predictions from a fitted model.

        Args:
            model: Fitted XGBRegressor.
            X: Feature matrix.

        Returns:
            Array of predicted total points (float), NOT probabilities.
        """
        return model.predict(X)

    # ------------------------------------------------------------------
    # Overridable hooks
    # ------------------------------------------------------------------

    def _get_param_distributions(self) -> dict:
        """Return parameter distributions for RandomizedSearchCV."""
        return {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [3, 4, 5, 6],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.7, 0.8, 0.9],
            "colsample_bytree": [0.7, 0.8, 0.9],
        }

    def _get_scoring_metric(self) -> str:
        """O/U uses negative MAE for hyperparameter tuning."""
        return "neg_mean_absolute_error"

    def _compute_season_metrics(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        season: int,
    ) -> dict[str, Any]:
        """Compute O/U-specific evaluation metrics for a holdout season.

        Args:
            predictions: Predicted totals.
            actuals: Actual totals.
            season: Season year.

        Returns:
            Dict of metric name to value.
        """
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        mae = float(mean_absolute_error(actuals, predictions))
        rmse = float(np.sqrt(mean_squared_error(actuals, predictions)))
        r2 = float(r2_score(actuals, predictions))

        return {
            "season": season,
            "n_games": len(predictions),
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
        }

    # ------------------------------------------------------------------
    # O/U-specific methods
    # ------------------------------------------------------------------

    def train_and_evaluate(
        self,
        features_df: pd.DataFrame,
        closing_odds_df: pd.DataFrame | None = None,
    ) -> dict:
        """Orchestrate O/U training with total distribution converter fitting.

        Extends the base train_and_evaluate to additionally fit a
        TotalDistributionConverter on HP-validation residuals.

        Args:
            features_df: Full feature matrix with ID cols, features, and target.
            closing_odds_df: Optional DataFrame with closing odds for CLV.

        Returns:
            Dict with per-season metrics, feature names, best params, etc.
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
            max_features=_OU_MAX_FEATURES,
        )

        self.logger.info(
            "Features selected for O/U",
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

        # Step 2b: Fit total distribution converter on HP-validation residuals
        hp_model = self._create_model(best_params)
        hp_model.fit(
            train_val_split.train_data[self.feature_names],
            train_val_split.train_targets,
        )
        hp_val_preds = self._predict_raw(
            hp_model,
            train_val_split.test_data[self.feature_names],
        )
        self.total_converter = self._fit_total_converter(
            train_val_split.test_targets.values, hp_val_preds
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
                        "model_total": predictions,
                        "actual": y_test.values,
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
        self.metadata = self._build_metadata(best_params, season_results)

        # Step 4: Compute CLV if closing odds provided
        clv_results = None
        if closing_odds_df is not None and all_predictions:
            from models.clv import compute_clv_for_predictions

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

    def predict_over_under_probability(
        self,
        predicted_totals: np.ndarray,
        market_totals: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert total predictions to over/under probabilities.

        Uses the fitted TotalDistributionConverter to compute P(over)
        and P(under) for each game given predicted total and market line.

        Args:
            predicted_totals: Array of predicted total points.
            market_totals: Array of market total lines.

        Returns:
            Tuple of (over_probabilities, under_probabilities) where
            under_probability = 1 - over_probability.

        Raises:
            ValueError: If total converter not fitted (model not trained).
        """
        if self.total_converter is None or not self.total_converter.is_fitted:
            msg = "Total converter not fitted. Train the model first."
            raise ValueError(msg)
        over_probs, under_probs = self.total_converter.predict_over_under_probabilities(
            predicted_totals, market_totals
        )
        return over_probs, under_probs

    def _fit_total_converter(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> TotalDistributionConverter:
        """Fit a TotalDistributionConverter on prediction residuals.

        Args:
            y_true: Actual target values.
            y_pred: Predicted target values.

        Returns:
            Fitted TotalDistributionConverter instance.
        """
        residuals = y_true - y_pred
        converter = TotalDistributionConverter(distribution_type="normal")
        converter.fit(residuals)
        return converter

    def _get_feature_importances(
        self,
        model: XGBRegressor,
        feature_names: list[str],
    ) -> dict[str, float]:
        """Extract sorted feature importances from a trained model.

        Args:
            model: Fitted XGBRegressor.
            feature_names: List of feature names.

        Returns:
            Dict of feature name to importance, sorted descending.
        """
        importances = model.feature_importances_
        importance_dict = dict(zip(feature_names, importances.tolist()))
        return dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))

    def _build_metadata(
        self,
        best_params: dict,
        season_results: list[dict],
    ) -> dict[str, Any]:
        """Build metadata dict for artifact storage.

        Args:
            best_params: Best hyperparameters from tuning.
            season_results: Per-season evaluation metrics.

        Returns:
            Metadata dict.
        """
        metadata: dict[str, Any] = {
            "target": self.target,
            "model_type": "XGBRegressor",
            "approach": "regression_only",
            "feature_names": self.feature_names,
            "best_params": best_params,
            "season_results": season_results,
            "config": {
                "train_seasons": self.config.train_seasons,
                "hp_val_seasons": self.config.hp_val_seasons,
                "holdout_seasons": self.config.holdout_seasons,
            },
        }

        # Add feature importances from the final model if available
        if self.model is not None and hasattr(self.model, "feature_importances_"):
            top_importances = self._get_feature_importances(
                self.model, self.feature_names
            )
            # Store top 10
            top_10 = dict(list(top_importances.items())[:10])
            metadata["top_feature_importances"] = top_10

        return metadata
