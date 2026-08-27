"""Win Probability trainer using Logistic Regression with isotonic calibration.

Extends BaseTrainer to implement WP-specific model logic:
- LogisticRegression as the core model (MODL-02)
- Isotonic calibration fitted on HP-validation data, not training data
- StandardScaler on features (LogReg benefits from scaling)
- ECE computed on holdout predictions (MODL-06)
- Feature selection soft target of ~16-20 features (MODL-07)
- No random CV -- only temporal validation via walk-forward
"""

from __future__ import annotations

import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from models.calibrate import ProbabilityCalibrator
from models.clv import compute_clv_for_predictions
from models.temporal import TemporalSplitConfig, WalkForwardSplitter
from models.trainers.base import BaseTrainer

# Maximum number of features to select (soft target ~16, hard cap 20)
_WP_MAX_FEATURES = 20


class WPTrainer(BaseTrainer):
    """Win Probability trainer using Logistic Regression with isotonic calibration.

    Produces calibrated home-team win probabilities in [0, 1] range.
    Calibration is fitted on HP-validation predictions (not training data)
    to avoid overfitting the calibrator to the same data used for model fitting.

    Attributes:
        scaler: StandardScaler fitted on training data.
    """

    def __init__(self, config: TemporalSplitConfig | None = None) -> None:
        """Initialize the WP trainer.

        Args:
            config: Temporal split configuration. Defaults to default split.
        """
        super().__init__(target="wp", config=config)
        self.scaler: StandardScaler | None = None

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def _get_target_column(self) -> str:
        """Return the target column name for WP prediction."""
        return "home_win"

    def _create_model(self, params: dict) -> LogisticRegression:
        """Create a LogisticRegression model with the given parameters.

        Args:
            params: Model hyperparameters. Merged with defaults.

        Returns:
            Unfitted LogisticRegression instance.
        """
        default_params = self._get_default_params()
        default_params.update(params)
        return LogisticRegression(**default_params)

    def _get_default_params(self) -> dict:
        """Return default hyperparameters for LogisticRegression."""
        return {
            "max_iter": 1000,
            "solver": "lbfgs",
            "C": 1.0,
            "random_state": 42,
        }

    def _define_search_space(self, trial: optuna.Trial) -> dict:
        """LogReg search space with conditional solver-penalty (per D-09).

        Uses trial.suggest_* to define the search space. Handles
        solver-penalty compatibility: l1 and elasticnet require saga,
        l2 allows lbfgs or saga.

        The solver_l2 parameter is suggested under a separate Optuna name
        to avoid conditional parameter conflicts, but is returned as
        "solver" in the output dict so LogisticRegression receives valid kwargs.

        Args:
            trial: Optuna trial for parameter suggestion.

        Returns:
            Dict of parameter name to suggested value, ready for
            LogisticRegression(**params).
        """
        C = trial.suggest_float("C", 0.001, 100.0, log=True)
        penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet"])

        params: dict = {
            "C": C,
            "penalty": penalty,
            "max_iter": 1000,
            "random_state": 42,
        }

        if penalty == "l1":
            params["solver"] = "saga"
        elif penalty == "elasticnet":
            params["solver"] = "saga"
            params["l1_ratio"] = trial.suggest_float("l1_ratio", 0.0, 1.0)
        else:  # l2
            # Use distinct Optuna name to avoid conflicts with fixed solver values,
            # but map back to "solver" for LogisticRegression compatibility
            solver_choice = trial.suggest_categorical("solver_l2", ["lbfgs", "saga"])
            params["solver"] = solver_choice

        return params

    def _predict_raw(self, model: LogisticRegression, X: pd.DataFrame) -> np.ndarray:
        """Generate raw probabilities from a fitted LogisticRegression.

        Args:
            model: Fitted LogisticRegression.
            X: Feature matrix.

        Returns:
            Array of P(home_win=1) probabilities.
        """
        return model.predict_proba(X)[:, 1]

    # ------------------------------------------------------------------
    # Overridable hooks
    # ------------------------------------------------------------------

    def _get_param_distributions(self) -> dict:
        """Return parameter distributions for HP tuning."""
        return {
            "C": [0.001, 0.01, 0.1, 1.0, 10.0],
            "solver": ["lbfgs"],
            "max_iter": [1000],
            "penalty": ["l2"],
        }

    def _get_scoring_metric(self) -> str:
        """WP uses log loss for HP tuning."""
        return "neg_log_loss"

    def _compute_cv_score(
        self,
        predictions: np.ndarray,
        actuals: pd.Series,
    ) -> float:
        """WP uses log_loss for CV scoring (per D-05).

        Overrides the base class MAE default to use log_loss,
        which is the appropriate metric for binary probability
        calibration.

        Args:
            predictions: Model probability predictions for validation fold.
            actuals: Actual binary target values (0/1).

        Returns:
            Log loss score (lower is better).
        """
        from sklearn.metrics import log_loss

        # Clip predictions to avoid log(0)
        clipped = np.clip(predictions, 1e-7, 1 - 1e-7)
        return float(log_loss(actuals.values, clipped))

    # ------------------------------------------------------------------
    # WP-specific feature importance
    # ------------------------------------------------------------------

    def _get_feature_importances(
        self,
        model: LogisticRegression,
        feature_names: list[str],
    ) -> dict[str, float]:
        """Extract feature importances from LogReg coefficients.

        Uses absolute coefficient values, sorted descending.

        Args:
            model: Fitted LogisticRegression.
            feature_names: List of feature column names.

        Returns:
            Dict mapping feature name to absolute coefficient value.
        """
        coefficients = model.coef_[0]
        importance = dict(
            zip(feature_names, np.abs(coefficients).tolist(), strict=False)
        )
        return dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))

    # ------------------------------------------------------------------
    # Override train_and_evaluate for WP-specific calibration and ECE
    # ------------------------------------------------------------------

    def train_and_evaluate(
        self,
        features_df: pd.DataFrame,
        closing_odds_df: pd.DataFrame | None = None,
        tune: bool = True,
    ) -> dict:
        """Orchestrate WP training with isotonic calibration and ECE.

        Extends the base training pipeline with WP-specific steps:
        1. Select features on training window (locked for all holdout)
        2. Fit StandardScaler on training data
        3. Tune hyperparameters on train + HP-val (temporal CV) or use defaults
        4. Fit calibrator on HP-validation predictions (NOT training data)
        5. Walk-forward through holdout applying locked features + calibrator
        6. Compute ECE on all holdout predictions
        7. Compute CLV if closing odds provided

        Args:
            features_df: Full feature matrix with ID cols, features, and target.
            closing_odds_df: Optional DataFrame with closing odds for CLV.
            tune: When True (default), tune hyperparameters via Optuna. When
                False, perform a straight re-fit using _get_default_params() and
                skip the Optuna study (D24-12). The scaler and calibration steps
                are unchanged -- only the params source changes.

        Returns:
            Dict with per-season metrics, overall metrics, feature names,
            best parameters, and CLV results.

        Raises:
            ValueError: If target column "home_win" is missing.
        """
        target_col = self._get_target_column()

        # Hard-fail if target column missing
        if target_col not in features_df.columns:
            # Check if we can derive it from scores
            if (
                "home_score" in features_df.columns
                and "away_score" in features_df.columns
            ):
                features_df = features_df.copy()
                features_df[target_col] = (
                    features_df["home_score"] > features_df["away_score"]
                ).astype(int)
                self.logger.info("Created home_win column from home_score > away_score")
            else:
                msg = (
                    f"Target column '{target_col}' not found in features_df. "
                    f"Available columns: {list(features_df.columns)}"
                )
                raise ValueError(msg)

        splitter = WalkForwardSplitter(
            config=self.config,
            target_col=target_col,
        )

        # Step 1: Feature selection on training window
        train_val_split = splitter.get_train_val_split(features_df)
        self.feature_names = self.select_features(
            train_val_split.train_data,
            train_val_split.train_targets,
            max_features=_WP_MAX_FEATURES,
        )

        self.logger.info(
            "Features selected for WP",
            n_features=len(self.feature_names),
            features=self.feature_names[:10],
        )

        # Step 2: Fit StandardScaler on training data
        self.scaler = StandardScaler()
        self.scaler.fit(train_val_split.train_data[self.feature_names])

        # Step 3: Tune hyperparameters on train + hp_val
        combined_train = pd.concat(
            [train_val_split.train_data, train_val_split.test_data]
        )
        combined_targets = pd.concat(
            [train_val_split.train_targets, train_val_split.test_targets]
        )
        combined_X_scaled = pd.DataFrame(
            self.scaler.transform(combined_train[self.feature_names]),
            columns=self.feature_names,
            index=combined_train.index,
        )
        best_params = (
            self.tune_hyperparameters(
                combined_X_scaled,
                combined_targets,
            )
            if tune
            else self._get_default_params()
        )

        # Step 4: Fit calibrator on HP-validation predictions
        # Train model on train data only, predict HP-val, fit calibrator on those
        X_train_scaled = pd.DataFrame(
            self.scaler.transform(train_val_split.train_data[self.feature_names]),
            columns=self.feature_names,
            index=train_val_split.train_data.index,
        )
        hp_val_model = self._create_model(best_params)
        hp_val_model.fit(X_train_scaled, train_val_split.train_targets)

        X_hp_val_scaled = pd.DataFrame(
            self.scaler.transform(train_val_split.test_data[self.feature_names]),
            columns=self.feature_names,
            index=train_val_split.test_data.index,
        )
        hp_val_predictions = self._predict_raw(hp_val_model, X_hp_val_scaled)

        # Fit Platt calibrator on HP-val predictions (isotonic is the fallback)
        prob_calibrator = ProbabilityCalibrator(
            primary_method="platt",
            fallback_method="isotonic",
        )
        calibration_results = prob_calibrator.calibrate_probabilities(
            raw_probabilities=hp_val_predictions,
            true_labels=train_val_split.test_targets.values,
        )
        self.calibrator = calibration_results.calibrator

        self.logger.info(
            "Calibration fitted on HP-validation data",
            method=calibration_results.method,
            hp_val_ece=calibration_results.calibration_metrics.get(
                "expected_calibration_error"
            ),
            n_hp_val_samples=len(hp_val_predictions),
        )

        # Step 5: Walk-forward through holdout with locked features + calibrator
        season_results = []
        all_predictions = []
        all_actuals = []
        all_pred_dfs = []

        for split in splitter.generate_splits(features_df):
            X_train = split.train_data[self.feature_names]
            y_train = split.train_targets
            X_test = split.test_data[self.feature_names]
            y_test = split.test_targets

            # Scale features
            X_train_s = pd.DataFrame(
                self.scaler.transform(X_train),
                columns=self.feature_names,
                index=X_train.index,
            )
            X_test_s = pd.DataFrame(
                self.scaler.transform(X_test),
                columns=self.feature_names,
                index=X_test.index,
            )

            model = self._create_model(best_params)
            model.fit(X_train_s, y_train)

            raw_predictions = self._predict_raw(model, X_test_s)

            # Apply calibration
            calibrated_predictions = prob_calibrator.apply_calibration(
                raw_predictions, self.calibrator
            )

            season_metrics = self._compute_season_metrics(
                calibrated_predictions, y_test.values, split.test_season
            )
            season_results.append(season_metrics)

            # Collect predictions for ECE and CLV
            all_predictions.extend(calibrated_predictions.tolist())
            all_actuals.extend(y_test.values.tolist())

            if closing_odds_df is not None:
                pred_df = pd.DataFrame(
                    {
                        "game_id": split.test_data.index,
                        "model_prob": calibrated_predictions,
                        "actual": y_test.values,
                        "season": split.test_season,
                    }
                )
                all_pred_dfs.append(pred_df)

            self.logger.info(
                "WP holdout season evaluated",
                test_season=split.test_season,
                n_train=len(X_train),
                n_test=len(X_test),
                metrics=season_metrics,
            )

        # Store the final model
        self.model = model

        # Step 6: Compute ECE on all holdout predictions
        all_predictions_arr = np.array(all_predictions)
        all_actuals_arr = np.array(all_actuals)

        holdout_ece = prob_calibrator._calculate_ece(
            all_predictions_arr, all_actuals_arr
        )

        # Build metadata
        self.metadata = {
            "target": self.target,
            "feature_names": self.feature_names,
            "best_params": best_params,
            "season_results": season_results,
            "ece": float(holdout_ece),
            "calibration": {
                "method": calibration_results.method,
                "hp_val_ece": float(
                    calibration_results.calibration_metrics.get(
                        "expected_calibration_error", float("nan")
                    )
                ),
            },
            "feature_importances": self._get_feature_importances(
                model, self.feature_names
            ),
            "config": {
                "train_seasons": self.config.train_seasons,
                "hp_val_seasons": self.config.hp_val_seasons,
                "holdout_seasons": self.config.holdout_seasons,
            },
        }

        # Step 7: Compute CLV if closing odds provided
        clv_results = None
        if closing_odds_df is not None and all_pred_dfs:
            predictions_combined = pd.concat(all_pred_dfs, ignore_index=True)
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
