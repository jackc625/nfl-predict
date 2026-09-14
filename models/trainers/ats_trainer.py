"""ATS (Against the Spread) model trainer using XGBoost regression.

Predicts point margin (home_score - away_score) and converts to
cover probabilities via ResidualDistributionConverter fitted on
HP-validation residuals.

Extends BaseTrainer from Plan 01. Reuses the existing
ResidualDistributionConverter from models.train_ats.

Requirements addressed: MODL-03 (ATS XGBoost), MODL-07 (feature count).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import optuna
import pandas as pd
from xgboost import XGBRegressor

from models.temporal import TemporalSplitConfig, WalkForwardSplitter
from models.train_ats import ResidualDistributionConverter
from models.trainers.base import BaseTrainer
from utils import get_logger

logger = get_logger(__name__)

# Soft target for feature count (MODL-07: ~22 features for ATS)
_ATS_MAX_FEATURES = 25


class ATSTrainer(BaseTrainer):
    """ATS (Against the Spread) trainer using XGBoost regression for margin prediction.

    Predicts the home team's expected margin (home_score - away_score).
    Cover probabilities are derived from the residual distribution fitted
    on HP-validation predictions -- regression only, no binary model.

    Attributes:
        residual_converter: Fitted ResidualDistributionConverter for
            converting margin predictions to cover probabilities.
    """

    def __init__(self, config: TemporalSplitConfig | None = None) -> None:
        """Initialize the ATS trainer.

        Args:
            config: Temporal split configuration. Defaults to default split.
        """
        super().__init__(target="ats", config=config)
        self.residual_converter: ResidualDistributionConverter | None = None

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def _get_target_column(self) -> str:
        """Return the target column for ATS: home_margin."""
        return "home_margin"

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
        """Return default XGBoost hyperparameters for ATS margin prediction."""
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
        """8-parameter XGBoost search space for ATS (per D-07).

        Defines the full hyperparameter search space for XGBoost
        margin prediction. Ranges follow D-07 specifications.

        Args:
            trial: Optuna trial for parameter suggestion.

        Returns:
            Dict of parameter name to suggested value.
        """
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 8),
            "n_estimators": trial.suggest_int("n_estimators", 50, 500),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
            "random_state": 42,
            "verbosity": 0,
            "n_jobs": -1,
        }

    def _predict_raw(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        """Generate raw margin predictions from a fitted model.

        Args:
            model: Fitted XGBRegressor.
            X: Feature matrix.

        Returns:
            Array of predicted margins (float), NOT probabilities.
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
        """ATS uses negative MAE for hyperparameter tuning."""
        return "neg_mean_absolute_error"

    def _compute_season_metrics(
        self,
        predictions: np.ndarray,
        actuals: np.ndarray,
        season: int,
    ) -> dict[str, Any]:
        """Compute ATS-specific evaluation metrics for a holdout season.

        Args:
            predictions: Predicted margins.
            actuals: Actual margins.
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
    # ATS-specific methods
    # ------------------------------------------------------------------

    def train_and_evaluate(
        self,
        features_df: pd.DataFrame,
        closing_odds_df: pd.DataFrame | None = None,
        tune: bool = True,
    ) -> dict:
        """Orchestrate ATS training with residual converter fitting.

        Extends the base train_and_evaluate to additionally fit a
        ResidualDistributionConverter on HP-validation residuals.

        Args:
            features_df: Full feature matrix with ID cols, features, and target.
            closing_odds_df: Optional DataFrame with closing odds for CLV.
            tune: When True (default), tune hyperparameters via Optuna. When
                False, perform a straight re-fit using _get_default_params() and
                skip the Optuna study (D24-12). The residual-converter fitting is
                unchanged -- only the params source changes.

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
            max_features=_ATS_MAX_FEATURES,
        )

        self.logger.info(
            "Features selected for ATS",
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
            )
            if tune
            else self._get_default_params()
        )

        # Step 2b: Fit residual converter on HP-validation residuals
        hp_model = self._create_model(best_params)
        hp_model.fit(
            train_val_split.train_data[self.feature_names],
            train_val_split.train_targets,
        )
        hp_val_preds = self._predict_raw(
            hp_model,
            train_val_split.test_data[self.feature_names],
        )
        self.residual_converter = self._fit_residual_converter(
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
                        "model_spread": predictions,
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

    def predict_cover_probability(
        self,
        predicted_margins: np.ndarray,
        spreads: np.ndarray,
    ) -> np.ndarray:
        """Convert margin predictions to cover probabilities.

        Uses the fitted ResidualDistributionConverter to compute P(cover)
        for each game given predicted margin and market spread.

        Args:
            predicted_margins: Array of predicted point margins.
            spreads: Array of market spreads (negative = home favored).

        Returns:
            Array of cover probabilities in [0, 1].

        Raises:
            ValueError: If residual converter not fitted (model not trained).
        """
        if self.residual_converter is None or not self.residual_converter.is_fitted:
            msg = "Residual converter not fitted. Train the model first."
            raise ValueError(msg)
        return self.residual_converter.predict_cover_probability(
            predicted_margins, spreads
        )

    def converter_params(self) -> dict[str, Any] | None:
        """The fitted residual converter's parameters, as plain JSON-able values.

        THE FORM IS THE DECISION (D33.1-R1, Plan 33.1-10 Ruling T2 part 2). Three fields
        describe this converter completely and reconstruct it exactly, so they are
        persisted as JSON in ``metadata.json`` rather than as a second pickle. Two reasons,
        both deliberate: a JSON record is readable a year from now by a human auditing why
        a cover probability was what it was, which a pickle is not; and it does not widen
        the ``joblib.load`` deserialisation surface.

        Until this existed the converter was never persisted under ANY name --
        ``BaseTrainer.save`` passes ``self.calibrator``, which ATS leaves None -- and
        serving rebuilt conversion from ``metadata.get("residual_std", 13.5)``, a hardcoded
        fallback standing in for a fitted object.

        Returns:
            The three parameters, or None when the converter is unfitted (so a caller
            that saves before training does not write a half-built record).
        """
        converter = self.residual_converter
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

    def _fit_residual_converter(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> ResidualDistributionConverter:
        """Fit a ResidualDistributionConverter on prediction residuals.

        Args:
            y_true: Actual target values.
            y_pred: Predicted target values.

        Returns:
            Fitted ResidualDistributionConverter instance.
        """
        residuals = y_true - y_pred
        converter = ResidualDistributionConverter(distribution_type="normal")
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
