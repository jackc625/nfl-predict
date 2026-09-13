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
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from models.calibrate import ProbabilityCalibrator
from models.clv import compute_clv_for_predictions
from models.temporal import TemporalSplitConfig, WalkForwardSplitter
from models.trainers.base import BaseTrainer

# Maximum number of features to select (soft target ~16, hard cap 20)
_WP_MAX_FEATURES = 20

# THE FOUR PIPELINE STEP NAMES, DECLARED ONCE (D33.1-R3, composing with
# D33.1-R1). Plan 33.1-10 IMPORTS this when it persists the pipeline, so the two
# plans read one declaration and not two.
WP_PIPELINE_STEP_NAMES: tuple[str, ...] = (
    "imputer",
    "missing_indicator",
    "scaler",
    "estimator",
)

# The suffix every missing-indicator column carries. These columns are part of
# WP's feature space from here on, so they are new symbols of this phase and are
# named rather than positional.
MISSING_INDICATOR_SUFFIX: str = "_was_missing"


class _ImputeAndCarryRaw(BaseEstimator, TransformerMixin):
    """Pipeline step ``imputer``: median-impute, and CARRY the original beside it.

    WHY THIS IS NOT A BARE ``SimpleImputer``. A linear ``Pipeline`` hands step
    N+1 whatever step N returned, so a stock ``MissingIndicator`` placed after a
    stock imputer would see data with no missing values left and emit an
    all-False indicator -- the exact opposite of what it is there for. The
    indicator has to see the ORIGINAL, so this step emits both halves and
    :class:`_FoldRawIntoMissingIndicator` folds the second one.

    MEDIAN, and the choice is deliberate: a median is robust to the heavy tails
    a temperature or a wind distribution has. The imputed VALUE is not asked to
    mean anything -- the indicator column beside it is what carries the
    information that the reading was absent.

    ``statistics_`` is re-exposed so the per-fold temporal-safety assertion can
    read the fitted statistic off a fitted pipeline.
    """

    def __init__(self, strategy: str = "median") -> None:
        self.strategy = strategy

    def fit(self, X, y=None):
        self.imputer_ = SimpleImputer(strategy=self.strategy)
        self.imputer_.fit(np.asarray(X, dtype=float))
        self.n_features_in_ = np.asarray(X, dtype=float).shape[1]
        return self

    def transform(self, X):
        raw = np.asarray(X, dtype=float)
        return np.hstack([self.imputer_.transform(raw), raw])

    @property
    def statistics_(self) -> np.ndarray:
        """The fitted per-column imputation statistic."""
        return self.imputer_.statistics_


class _FoldRawIntoMissingIndicator(BaseEstimator, TransformerMixin):
    """Pipeline step ``missing_indicator``: append one indicator per column.

    Takes the ``[imputed | raw]`` pair the step above emits and returns
    ``[imputed | indicators]``, so the indicator columns are APPENDED rather
    than replacing the imputed ones. Both are needed: the imputed value is what
    the estimator fits on, and the indicator is what tells it the value was
    absent.

    ``features="all"`` is what makes the emitted column set a deterministic
    function of the INPUT COLUMNS rather than of which rows happened to be null
    in the fit window. A data-dependent indicator set would make two re-fits
    produce two different feature spaces, which is the reproducibility
    constraint CLAUDE.md states.
    """

    def fit(self, X, y=None):
        paired = np.asarray(X, dtype=float)
        self.n_features_ = paired.shape[1] // 2
        self.indicator_ = MissingIndicator(features="all")
        self.indicator_.fit(paired[:, self.n_features_ :])
        return self

    def transform(self, X):
        paired = np.asarray(X, dtype=float)
        split = self.n_features_
        indicators = self.indicator_.transform(paired[:, split:]).astype(float)
        return np.hstack([paired[:, :split], indicators])


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

    # ------------------------------------------------------------------
    # Abstract method implementations
    # ------------------------------------------------------------------

    def _get_target_column(self) -> str:
        """Return the target column name for WP prediction."""
        return "home_win"

    def _create_model(self, params: dict) -> Pipeline:
        """Create WP's four-step Pipeline with the given parameters.

        D33.1-R3, owner-ratified 2026-09-12 and attributed to R4 as this
        phase's nullable-input handling.

        THIS IS THE ONLY CONSTRUCTION SITE, so there is NO path that produces a
        bare ``LogisticRegression``. That matters because
        ``BaseTrainer.select_features`` fits ``self._create_model(...)`` BEFORE
        anything is scaled, over every informative candidate column -- and after
        Plan 33.1-07 those candidates include 45 weather columns carrying NaN,
        which a bare ``LogisticRegression`` rejects. The currently-deployed
        feature list containing zero weather features does not protect
        re-selection: selection re-runs on every ``train_and_evaluate`` call and
        ``models.train`` exposes no feature-list flag.

        The scaler lives INSIDE the returned object, which is the property
        D33.1-R1 exists to guarantee and the reason ``wp_20260824_113325`` is
        trained-scaled and served-raw today.

        Args:
            params: Model hyperparameters. Merged with defaults.

        Returns:
            Unfitted, serializable Pipeline whose step names are exactly
            :data:`WP_PIPELINE_STEP_NAMES` and whose final step is a
            ``LogisticRegression``.
        """
        default_params = self._get_default_params()
        default_params.update(params)
        return Pipeline(
            [
                ("imputer", _ImputeAndCarryRaw(strategy="median")),
                ("missing_indicator", _FoldRawIntoMissingIndicator()),
                ("scaler", StandardScaler()),
                ("estimator", LogisticRegression(**default_params)),
            ]
        )

    # ------------------------------------------------------------------
    # Pipeline accessors (D33.1-R3)
    # ------------------------------------------------------------------

    @property
    def scaler(self) -> StandardScaler | None:
        """The fitted scaler, read off the pipeline rather than held beside it.

        Kept as an accessor so existing callers and Plan 33.1-10's persistence
        contract have ONE name for it. The scaler is now INSEPARABLE from the
        estimator by construction -- which is exactly the property D33.1-R1
        exists to guarantee, and the reason a separately-held scaler was how
        ``wp_20260824_113325`` came to be trained-scaled and served-raw.
        """
        if isinstance(self.model, Pipeline):
            return self.model.named_steps.get("scaler")
        return None

    @staticmethod
    def imputer_statistics(model: Pipeline) -> np.ndarray:
        """The fitted per-column imputation statistic of *model*.

        Exposed so the per-fold temporal-safety assertion reads the statistic
        through one named accessor rather than reaching into step internals.
        """
        return model.named_steps["imputer"].statistics_

    @staticmethod
    def indicator_feature_names(model: Pipeline, feature_names: list[str]) -> list[str]:
        """The indicator column names *model* emits for *feature_names*.

        One per input column, always, because the indicator is configured
        ``features="all"``. The count is therefore a function of the columns and
        not of the fit window's null pattern.
        """
        count = model.named_steps["missing_indicator"].n_features_
        return [f"{name}{MISSING_INDICATOR_SUFFIX}" for name in feature_names[:count]]

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

    def _predict_raw(self, model: Pipeline, X: pd.DataFrame) -> np.ndarray:
        """Generate raw probabilities from a fitted WP pipeline.

        Args:
            model: Fitted Pipeline (imputer -> indicator -> scaler -> estimator).
            X: RAW feature matrix. It is not pre-scaled: the pipeline scales
                internally, and handing it an already-scaled frame would scale
                twice.

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
        model: Pipeline,
        feature_names: list[str],
    ) -> dict[str, float]:
        """Extract feature importances from the pipeline's LogReg coefficients.

        Uses absolute coefficient values, sorted descending.

        The fitted estimator carries TWICE as many coefficients as there are
        input columns -- the imputed values and then the missing indicators --
        so the indicator half is reported under its own
        ``<column>_was_missing`` names rather than dropped. "This feature was
        absent" is a thing WP can learn from, and hiding its weight would make
        the importance table describe a model that is not the one fitted.

        Args:
            model: Fitted Pipeline.
            feature_names: List of feature column names.

        Returns:
            Dict mapping feature name to absolute coefficient value.
        """
        estimator = model.named_steps["estimator"]
        coefficients = estimator.coef_[0]
        names = list(feature_names)
        if len(coefficients) == 2 * len(names):
            names = names + [
                f"{name}{MISSING_INDICATOR_SUFFIX}" for name in feature_names
            ]
        importance = dict(zip(names, np.abs(coefficients).tolist(), strict=False))
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

        # Step 2: THERE IS NO SEPARATE SCALER STEP ANY MORE (D33.1-R3).
        #
        # The scaler is the third step of the Pipeline `_create_model` returns,
        # so every fit below scales on the rows it is handed and every predict
        # scales with that fit's statistics. `self.scaler` remains readable as a
        # property that reads the fitted pipeline.

        # Step 3: Tune hyperparameters on train + hp_val
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

        # Step 4: Fit calibrator on HP-validation predictions
        # Train model on train data only, predict HP-val, fit calibrator on those
        hp_val_model = self._create_model(best_params)
        hp_val_model.fit(
            train_val_split.train_data[self.feature_names],
            train_val_split.train_targets,
        )

        hp_val_predictions = self._predict_raw(
            hp_val_model, train_val_split.test_data[self.feature_names]
        )

        # Fit isotonic calibrator on HP-val predictions
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

            model = self._create_model(best_params)

            # THE FIT RECEIVES THE RAW FOLD FRAME, AND THAT IS WHAT MAKES THE
            # IMPUTER FOLD-FITTED. This is a TEMPORAL-SAFETY property, not an
            # implementation detail: `Pipeline.fit` fits EVERY step on the rows
            # it is handed, and the rows it is handed are this fold's
            # pre-holdout rows. An imputation statistic computed over the whole
            # frame and applied inside a fold leaks the holdout's distribution
            # into training -- which CLAUDE.md's walk-forward constraint forbids
            # and which this milestone exists to detect. The same now goes for
            # the scaler, which used to be fitted ONCE outside this loop.
            model.fit(X_train, y_train)

            raw_predictions = self._predict_raw(model, X_test)

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
