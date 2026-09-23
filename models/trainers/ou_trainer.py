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
import optuna
import pandas as pd
from xgboost import XGBRegressor

from config.tuning_preregistration import SEARCH_SPACE_BY_TARGET
from models.temporal import TemporalSplitConfig, WalkForwardSplitter
from models.train_ou import TotalDistributionConverter
from models.trainers.base import BaseTrainer, concat_holdout_predictions
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

    def _define_search_space(self, trial: optuna.Trial) -> dict:
        """The XGBoost search space, READ from the committed pre-registration.

        NO NUMERIC BOUND IS A LITERAL HERE ANY MORE (D33.2-17, Plan 33.2-23). Every bound
        comes from ``config.tuning_preregistration.SEARCH_SPACE_BY_TARGET``, which records
        TODAY'S bound beside the widened one. Two things follow that could not both follow
        while the bounds were inline: the widening happened ONCE, in a file locked by a
        content hash and a git-ancestry assertion, and the RandomSampler baseline searches
        the IDENTICAL space BY CONSTRUCTION rather than by a second declaration that can
        drift away from this one.

        Args:
            trial: Optuna trial for parameter suggestion.

        Returns:
            Dict of parameter name to suggested value.
        """
        space = SEARCH_SPACE_BY_TARGET[self.target]
        learning_rate = space["learning_rate"]
        max_depth = space["max_depth"]
        n_estimators = space["n_estimators"]
        subsample = space["subsample"]
        colsample_bytree = space["colsample_bytree"]
        min_child_weight = space["min_child_weight"]
        reg_alpha = space["reg_alpha"]
        reg_lambda = space["reg_lambda"]
        gamma = space["gamma"]
        return {
            "learning_rate": trial.suggest_float(
                "learning_rate",
                learning_rate.low,
                learning_rate.high,
                log=learning_rate.log,
            ),
            "max_depth": trial.suggest_int("max_depth", max_depth.low, max_depth.high),
            "n_estimators": trial.suggest_int(
                "n_estimators", n_estimators.low, n_estimators.high
            ),
            "subsample": trial.suggest_float(
                "subsample", subsample.low, subsample.high
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", colsample_bytree.low, colsample_bytree.high
            ),
            "min_child_weight": trial.suggest_int(
                "min_child_weight", min_child_weight.low, min_child_weight.high
            ),
            "reg_alpha": trial.suggest_float(
                "reg_alpha", reg_alpha.low, reg_alpha.high, log=reg_alpha.log
            ),
            "reg_lambda": trial.suggest_float(
                "reg_lambda", reg_lambda.low, reg_lambda.high, log=reg_lambda.log
            ),
            # NEW axis: gamma was fixed at XGBoost's 0.0 and never searched before.
            "gamma": trial.suggest_float("gamma", gamma.low, gamma.high),
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
        tune: bool = True,
    ) -> dict:
        """Orchestrate O/U training with total distribution converter fitting.

        Extends the base train_and_evaluate to additionally fit a
        TotalDistributionConverter on HP-validation residuals.

        Args:
            features_df: Full feature matrix with ID cols, features, and target.
            closing_odds_df: Optional DataFrame with closing odds for CLV.
            tune: When True (default), tune hyperparameters via Optuna. When
                False, perform a straight re-fit using _get_default_params() and
                skip the Optuna study (D24-12). The total-converter fitting is
                unchanged -- only the params source changes.

        Returns:
            Dict with per-season metrics, feature names, best params, CLV results,
            and ``holdout_predictions`` -- the per-game out-of-sample record
            (``models.trainers.base.HOLDOUT_PREDICTION_COLUMNS``), returned whether
            or not a closing-odds frame was passed.
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
        best_params = (
            self.tune_hyperparameters(
                combined_train[self.feature_names],
                combined_targets,
                # D33.2-17: the FULL frame, so the pre-registered adoption gate can
                # resolve and score the OUTER comparison season -- a season that is by
                # construction absent from combined_train. Ignored on every other path.
                full_features_df=features_df,
            )
            if tune
            else self._get_default_params()
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
        all_holdout_dfs: list[pd.DataFrame] = []

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

            # THE PER-GAME OUT-OF-SAMPLE RECORD, BUILT ON EVERY SPLIT (Plan 33.2-22). It used
            # to be built only `if closing_odds_df is not None`, so no caller could obtain
            # this model's own out-of-sample predictions without handing it a closing line.
            holdout_frame = pd.DataFrame(
                {
                    "game_id": split.test_data.index,
                    "season": split.test_season,
                    "prediction": predictions,
                    "actual": y_test.values,
                }
            )
            all_holdout_dfs.append(holdout_frame)

            # Collect predictions for CLV, DERIVED from the same frame in exactly the columns
            # and the order the CLV path has always received.
            if closing_odds_df is not None:
                pred_df = holdout_frame.rename(columns={"prediction": "model_prob"})
                pred_df["model_total"] = holdout_frame["prediction"].to_numpy()
                all_predictions.append(
                    pred_df[
                        ["game_id", "model_prob", "model_total", "actual", "season"]
                    ]
                )

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
            "holdout_predictions": concat_holdout_predictions(all_holdout_dfs),
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

    def converter_params(self) -> dict[str, Any] | None:
        """The fitted total converter's parameters, as plain JSON-able values.

        THE FORM IS THE DECISION (D33.1-R1, Plan 33.1-10 Ruling T2 part 2). Three fields
        describe this converter completely and reconstruct it exactly, so they are
        persisted as JSON in ``metadata.json`` rather than as a second pickle -- readable a
        year from now by a human auditing why an over probability was what it was, and
        without widening the ``joblib.load`` deserialisation surface.

        Until this existed the converter was never persisted under ANY name, and serving
        rebuilt conversion from ``metadata.get("residual_std", 13.0)`` -- a hardcoded
        fallback standing in for a fitted object.

        Returns:
            The three parameters, or None when the converter is unfitted.
        """
        converter = self.total_converter
        if converter is None or not converter.is_fitted:
            return None
        return {
            "distribution_type": converter.distribution_type,
            "residual_std": float(converter.residual_std),
            "distribution_params": {
                key: float(value)
                for key, value in (converter.distribution_params or {}).items()
            },
        }

    def final_fit(
        self,
        features_df: pd.DataFrame,
        partition: Any,
        *,
        closing_odds_df: pd.DataFrame | None = None,
    ) -> Any:
        """Fit the SHIPPED model on every completed season in *partition*.

        A thin wrapper over
        ``models.trainers.final_fit.final_fit_over_completed_seasons``; the decisions and
        their reasons live in that module. It WRITES NOTHING, and no production path in
        Phase 33.1 calls it -- the re-fit is Phase 33 Wave 15's act.
        """
        from models.trainers.final_fit import final_fit_over_completed_seasons

        return final_fit_over_completed_seasons(
            self, features_df, partition, closing_odds_df=closing_odds_df
        )

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
