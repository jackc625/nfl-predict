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

# ---------------------------------------------------------------------------
# Optuna study identity + storage (Plan 30-01, T-30-02 / T-30-14)
# ---------------------------------------------------------------------------
#
# There are TWO identities here on purpose, and which one a trainer uses is an EXPLICIT
# opt-in, never a global default. The Phase-30 identity changes what a tuned train actually
# searches, and this method is shared by callers with very different contracts:
#
#   * ``models.train`` (the Stage-2 candidate train that scripts/promote_models invokes) MUST
#     genuinely search -- SPEC R5. It opts in via ``use_phase30_tuning()``.
#   * ``backtest.engine.run_backtest`` (engine.py:339) trains with tune=True as a DIAGNOSTIC,
#     and the frozen v2.1 AUDIT-REPORT anchors in tests/integration/test_diag_diagnosis.py were
#     measured with it resuming the v2.0 study. Handing it a fresh study makes it search real
#     trials, land on different parameters, and drift those anchors -- verified empirically,
#     twice, during Plan 30-01. It therefore keeps the LEGACY identity, byte-for-byte.
#   * ``scripts.retrain_models`` likewise keeps the legacy identity.
#
# The legacy path's resume-at-budget IS a genuine latent issue (those "tuned" runs have been
# returning stored v2.0 parameters since March), but it is PRE-EXISTING and out of this plan's
# scope: silently changing what the backtest trains with inside a plumbing tracer is exactly
# the uninstructed side effect this phase is careful about. It is recorded for a later phase.

# The v2.0 identity every pre-Phase-30 caller keeps. Reusing it is what makes a search vacuous
# (those studies are already at budget), which is precisely why the Stage-2 path must not.
LEGACY_TUNING_STUDY_TAG: str = "v1"
LEGACY_TUNING_STORAGE_DIR: Path = Path("data/optuna")

# The per-PHASE study-identity tag. It exists because OptunaTuner.optimize computes remaining
# trials as ``max(0, n_trials - len(study.trials))`` under ``load_if_exists=True``: resuming a
# study that already holds the full budget runs ZERO new trials while still reporting a full
# trial count, so a "tuned" candidate would silently carry the OLD phase's parameters.
#
# The rule, plainly: a study identity is per-phase. A future phase that re-tunes MUST bump this
# tag. What this tag is NOT allowed to do is revert to the v2.0 ``_tuning_v1`` literal -- those
# three studies are the v2.0 historical record and are already at budget, so reusing their
# identity guarantees a vacuous search. Nothing in this phase deletes them either.
TUNING_STUDY_TAG: str = "p30"

# Where the Phase-30 SQLite study files live. OptunaTuner defaults storage_dir to
# ``data/optuna``, which collides with this phase's own prohibition on writing under ``data/``
# outside the one sanctioned fingerprinted rebuild -- and the prohibition's before/after hash
# manifest reads through ``load_dataframe``, so it would NOT have caught a ``data/optuna/``
# write. ``outputs/`` is gitignored and is the right home. What this constant is NOT allowed to
# be is any path under ``data/``; tests/unit/test_promote_models_tuned_path.py asserts that.
TUNING_STORAGE_DIR: Path = Path("outputs/optuna")


def _existing_trial_count(tuner: OptunaTuner) -> int:
    """Return how many trials the tuner's study ALREADY holds in storage.

    Read BEFORE ``optimize`` so the caller can compute how many trials the search genuinely
    added. A missing study file, or a storage file with no such study, is 0 -- the fresh case.

    Args:
        tuner: The configured OptunaTuner (read-only; its storage is not created here).

    Returns:
        The stored trial count, or 0 when the study does not exist yet.
    """
    db_path = tuner.storage_dir / f"{tuner.study_name}.db"
    if not db_path.exists():
        return 0
    try:
        study = optuna.load_study(
            study_name=tuner.study_name, storage=tuner.storage_url
        )
    except KeyError:
        # The storage file exists but holds no study by this name (the fresh-identity case,
        # e.g. right after TUNING_STUDY_TAG was bumped).
        return 0
    return len(study.trials)


# ---------------------------------------------------------------------------
# Feature-selection semantics (Plan 30-17, D30-OWNER-07)
# ---------------------------------------------------------------------------
#
# THE RULE, stated once so nobody has to re-derive it from a library default:
# ``select_features`` fits a scoring model on the training window, then keeps the
# top ``max_features`` columns by importance among those scoring at or above the MEAN
# importance. That is an INTERSECTION of a cap and a threshold, not a cap alone --
# scikit-learn's ``SelectFromModel._get_support_mask`` takes the top-K by score and then
# discards anything below the threshold. Measured on the accepted rung-4 gold
# (SELECTION-CENSUS.md), the CAP is what binds for every target: 56 / 85 / 91 features
# clear the mean against caps of 20 / 25 / 25, so the threshold currently removes nothing
# and the effective rule is pure top-K.
#
# THE PRE-FILTER, and why it is a pre-filter rather than a different threshold. Before
# the scoring model is fitted, columns carrying no information over the fit window --
# zero variance, including all-NaN -- are withheld from the FIT. Without this, selection
# depends on how many dead columns happen to be in the frame: the census measured that
# adding ten literally-constant columns moved 5 of ATS's 25 selected features and 9 of
# O/U's 25, and that removing the dead block moved 9 and 5.
#
# The mechanism is the FIT, not the threshold, which is why changing the threshold could
# not have fixed it. ``XGBRegressor`` runs with ``colsample_bytree=0.8`` and
# ``subsample=0.8``, so every added column changes which columns each tree samples and
# therefore the gain importances of the REAL features. ``LogisticRegression`` has no such
# sampling, which is why WP was already invariant to constants on live gold. Withholding
# the dead columns makes the fit input identical regardless of how many were present, so
# the invariance holds BY CONSTRUCTION.
#
# WHAT THE PRE-FILTER DOES NOT DO. It cannot make selection invariant to columns that
# have variance but no relationship to the target -- pure noise is indistinguishable from
# a weak signal without looking at the target, and the census measured noise columns
# being selected outright. That boundary is pinned by
# tests/unit/test_feature_selection_stability.py rather than papered over.
#
# WHAT IT DOES NOT CHANGE. No estimator and no default hyperparameter is touched, and the
# per-target budgets (``_WP_MAX_FEATURES`` 20, ``_ATS_MAX_FEATURES`` 25,
# ``_OU_MAX_FEATURES`` 25) are incumbent values that this change deliberately leaves
# alone. It changes the RULE, not the budget. A withheld column was never selectable in
# practice anyway: on the accepted gold no target selected a single column that was
# constant over its own fit window.


def informative_columns(X: pd.DataFrame) -> list[str]:
    """Return the columns of ``X`` that vary over these rows, in their original order.

    "Informative" here means only "not constant over the fit window". A column that is
    constant, or entirely NaN, carries nothing a model can learn from -- but it still
    perturbs a column-sampling estimator's fit, which is the defect this exists to close.

    Args:
        X: The frame the scoring model is about to be fitted on.

    Returns:
        The subset of ``X.columns`` with more than one distinct value (NaN counted as a
        value, so an all-NaN column is correctly treated as constant).
    """
    distinct = X.nunique(dropna=False)
    return [column for column in X.columns if distinct[column] > 1]


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

        # Tuning identity defaults to LEGACY so every pre-Phase-30 caller (backtest.engine,
        # scripts.retrain_models) is byte-identical to its prior behaviour. The Stage-2 train
        # opts in explicitly via use_phase30_tuning().
        self.tuning_study_tag: str = LEGACY_TUNING_STUDY_TAG
        self.tuning_storage_dir: Path = LEGACY_TUNING_STORAGE_DIR
        self.require_fresh_search: bool = False

    def use_phase30_tuning(self) -> None:
        """Opt this trainer into the Phase-30 study identity, storage and freshness guard.

        Called by ``models.train.train_target`` on the tuned path -- the Stage-2 candidate
        train that ``scripts/promote_models`` STEP 1 invokes. After this call the trainer
        searches a FRESH per-phase study under a non-``data/`` storage dir, and a search that
        adds zero trials is a hard failure rather than a silent return of stored parameters.

        Deliberately NOT the default: see the module-level note above. Making it the default
        would change what ``backtest.engine`` trains with and drift the frozen v2.1 anchors.
        """
        self.tuning_study_tag = TUNING_STUDY_TAG
        self.tuning_storage_dir = TUNING_STORAGE_DIR
        self.require_fresh_search = True

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
        """Select features by model importance on the training window.

        THE RULE: fit a scoring model on the columns that VARY over this window, then
        keep the top ``max_features`` of them by importance among those scoring at or
        above the mean importance. See the module-level "Feature-selection semantics"
        note for why the zero-variance pre-filter is there, what it does not promise,
        and which of the cap and the threshold actually binds in production.

        ``threshold="mean"`` is stated rather than left to scikit-learn's
        ``threshold=None`` default. It is the same rule that default already resolved to
        for this codebase's two estimators (an L2 ``LogisticRegression`` and an
        ``XGBRegressor``); stating it means a future switch to an L1 penalty would not
        silently re-resolve the rule to ``1e-5`` without anyone deciding to.

        Args:
            X: Training features.
            y: Training targets.
            max_features: Maximum number of features. None = use all that pass.

        Returns:
            Sorted list of selected feature names.
        """
        # Withhold columns that carry no information over THIS window from the fit. The
        # fallback covers a degenerate frame in which nothing varies: selecting from an
        # empty frame would raise, and refusing to select at all is worse than scoring
        # the frame as it stands.
        informative = informative_columns(X)
        fit_frame = X[informative] if informative else X

        model = self._create_model(self._get_default_params())
        model.fit(fit_frame, y)

        selector = SelectFromModel(
            model, max_features=max_features, threshold="mean", prefit=True
        )
        selected_mask = selector.get_support()
        selected_features = sorted(fit_frame.columns[selected_mask].tolist())

        self.logger.info(
            "Feature selection completed",
            total_features=len(X.columns),
            informative_features=len(fit_frame.columns),
            withheld_zero_variance=len(X.columns) - len(fit_frame.columns),
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

        # Study identity + storage come from INSTANCE state, so the Stage-2 opt-in
        # (use_phase30_tuning) can require a genuinely fresh search without changing what any
        # other caller trains with (T-30-02 / T-30-14). storage_dir is passed explicitly --
        # omitting it lets OptunaTuner default to ``data/optuna`` regardless of this tag.
        study_name = f"{self.target}_tuning_{self.tuning_study_tag}"

        tuner = OptunaTuner(
            study_name=study_name,
            direction=direction,
            storage_dir=self.tuning_storage_dir,
            n_trials=n_trials,
        )

        # Read the stored trial count BEFORE the search so a vacuous resume is detectable.
        trials_before = _existing_trial_count(tuner)

        objective = self._make_objective(X_train, y_train, cv_splits)
        result = tuner.optimize(objective)

        # HARD-fail a search that added nothing. A resumed full study returns the STORED best
        # parameters while reporting a full trial count, so without this assertion an untuned
        # candidate is indistinguishable from a tuned one in every downstream artifact.
        trials_added = result.n_trials - trials_before
        if self.require_fresh_search and trials_added <= 0:
            msg = (
                f"Optuna study '{study_name}' added ZERO new trials "
                f"(stored before={trials_before}, after={result.n_trials}, budget={n_trials}). "
                "The study was resumed already at budget, so the 'best params' returned are the "
                "STORED ones -- this candidate would be reported as tuned without having been "
                f"tuned. Remediation: bump TUNING_STUDY_TAG in {__name__} to open a fresh study "
                "identity. Do NOT delete the existing study file to work around this -- study "
                "files are the historical record of what was searched."
            )
            raise RuntimeError(msg)

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
            trials_added=trials_added,
            study_name=study_name,
            storage_dir=str(self.tuning_storage_dir),
            top_importances=dict(list(result.param_importances.items())[:5]),
        )

        # Store tuning result for later use in save()
        self._tuning_result = result

        return best_params

    def train_and_evaluate(
        self,
        features_df: pd.DataFrame,
        closing_odds_df: pd.DataFrame | None = None,
        tune: bool = True,
    ) -> dict:
        """Orchestrate full training and evaluation pipeline.

        1. Select features on training window
        2. Tune hyperparameters on train + HP-val window (or use defaults)
        3. Walk-forward through holdout seasons
        4. Collect per-season metrics
        5. Compute CLV if closing odds provided

        Args:
            features_df: Full feature matrix with ID cols, features, and target.
            closing_odds_df: Optional DataFrame with closing odds for CLV.
            tune: When True (default), tune hyperparameters via Optuna on the
                train + HP-val window. When False, perform a straight re-fit
                using _get_default_params() and skip the Optuna study entirely
                (D24-12). The default preserves the existing tuned behavior.

        Returns:
            Dict with per-season metrics, overall metrics, feature names,
            best parameters, and CLV results.
        """
        target_col = self._get_target_column()
        splitter = WalkForwardSplitter(
            config=self.config,
            target_col=target_col,
        )

        # Step 1: Feature selection on training window.
        #
        # ASYMMETRY, stated rather than left to be discovered (Plan 30-17). This call
        # passes NO max_features, i.e. it selects every feature clearing the mean
        # importance -- a different rule from the one production runs. On the accepted
        # rung-4 gold it would select 56 / 85 / 91 features where the three concrete
        # trainers select 20 / 25 / 25.
        #
        # It is UNREACHABLE on any production path, measured rather than assumed:
        # BaseTrainer is abstract (five abstract methods), and all three concrete
        # subclasses -- the only ones in the codebase, and the only values in
        # backtest.engine._TRAINER_MAP and backtest.signal_lift._TRAINER_FOR -- override
        # train_and_evaluate and pass their own budget. Instrumented production runs
        # confirmed this method is never entered (SELECTION-CENSUS.md section 6), and
        # tests/unit/test_feature_selection_stability.py fails if a trainer ever stops
        # overriding it. It is documented rather than deleted because removing a base
        # method to resolve an inconsistency that has never fired is the larger change.
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
        best_params = (
            self.tune_hyperparameters(
                combined_train[self.feature_names],
                combined_targets,
            )
            if tune
            else self._get_default_params()
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
            # D24-08: a train run (normal or staging) never auto-swaps the
            # served model. Production promotion is the sole latest.json writer
            # via models.artifacts.update_manifest / scripts/promote_models.
            update_latest=False,
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
