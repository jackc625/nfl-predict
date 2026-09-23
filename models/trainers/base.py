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
from sklearn.pipeline import Pipeline

from config.tuning_preregistration import (
    BEAT_RANDOM_MARGIN_BY_TARGET,
    NOT_CLEARED_ARM,
    NOT_CLEARED_RULE,
    OUTER_COMPARISON_RULE,
    PINNED_THREAD_COUNT,
    PRUNER_CONFIG,
    RANDOM_SAMPLER_SEED,
    SEARCH_SPACE_BY_TARGET,
    STUDY_ARM_RANDOM,
    STUDY_ARM_TPE,
    TPE_SAMPLER_SEED,
    TPE_SAMPLER_STARTUP_TRIALS,
    TRIAL_BUDGET_BY_TARGET,
    outer_comparison_season,
    search_space_digest,
)
from config.tuning_preregistration import (
    study_name as preregistered_study_name,
)
from models.artifacts import CONVERTER_PARAMS_METADATA_KEY, save_model_artifact
from models.clv import compute_clv_for_predictions
from models.temporal import (
    TemporalSplitConfig,
    WalkForwardSplitter,
    make_temporal_cv_splits,
)
from models.tuning import OptunaTuner, TuningResult, count_completed_trials
from utils import get_logger

# ---------------------------------------------------------------------------
# THE PER-GAME OUT-OF-SAMPLE RECORD (Plan 33.2-22, D33.2-15)
# ---------------------------------------------------------------------------
#
# Every concrete trainer returns ``holdout_predictions`` from ``train_and_evaluate`` --
# whether or not a closing-odds frame was passed. Before Plan 33.2-22 that per-game frame was
# built only ``if closing_odds_df is not None``, so no caller could obtain a model's OWN
# out-of-sample predictions without handing it a closing line. That made an outcome-based
# feature-group screen impossible to compute without the very market line D33.2-03 removes
# from every fit decision.
#
# The columns and the concatenation live HERE, on the shared base, because all three concrete
# trainers emit the frame and a declaration in any one of them would be a declaration in the
# wrong place -- three copies of a column list are three contracts wearing one name.
HOLDOUT_PREDICTION_COLUMNS: tuple[str, ...] = (
    "game_id",
    "season",
    "prediction",
    "actual",
)


def concat_holdout_predictions(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate per-split holdout frames into ONE per-game out-of-sample record.

    Returns an EMPTY frame carrying :data:`HOLDOUT_PREDICTION_COLUMNS` when a run produced no
    split at all, so a consumer always receives the same shape and never has to tell ``None``
    apart from "no holdout season had rows".

    Args:
        frames: One frame per walk-forward split, each already carrying the four columns.

    Returns:
        The concatenated frame, columns in :data:`HOLDOUT_PREDICTION_COLUMNS` order.
    """
    if not frames:
        return pd.DataFrame(columns=list(HOLDOUT_PREDICTION_COLUMNS))
    return pd.concat(frames, ignore_index=True)[list(HOLDOUT_PREDICTION_COLUMNS)]


# ---------------------------------------------------------------------------
# Optuna study identity + storage (Plan 30-01 T-30-02/T-30-14; Plan 30-16 T-30-63..68)
# ---------------------------------------------------------------------------
#
# There are THREE identities here on purpose, with THREE different lifetimes, and which one a
# trainer uses is an EXPLICIT opt-in -- only the first is a default. An identity decides what a
# "tuned" train actually searches, and ``tune_hyperparameters`` is shared by callers with very
# different contracts:
#
#   * DEFAULT / LEGACY -- lifetime FOREVER. ``{target}_tuning_v1`` under ``data/optuna/``. Every
#     caller that does not opt in keeps it, byte-for-byte: ``scripts.retrain_models`` and any
#     future caller. Those three study files are already at budget, so this identity's searches
#     resume and return the stored v2.0 parameters. That is a known property of the default and
#     is NOT changed here -- Plan 30-16's dispatched scope (D30-OWNER-02) is the backtest alone.
#   * PER-PHASE -- ``models.train`` (the Stage-2 candidate train that scripts/promote_models
#     invokes) MUST genuinely search, SPEC R5. It opts in via ``use_phase30_tuning()``, which
#     also ARMS ``require_fresh_search`` so a zero-trial resume is a hard RuntimeError. A phase
#     that re-tunes bumps ``TUNING_STUDY_TAG`` (D30-DEFER-02).
#   * PER-RUN -- ``backtest.engine`` opts in via ``use_backtest_tuning(run_id)``. Added by Plan
#     30-16 under owner ruling **D30-OWNER-02**, which selected Option 2 of D30-DEFER-01.
#
# WHAT D30-OWNER-02 OVERTURNED, stated plainly because this note previously asserted the
# opposite. Until Plan 30-16 this comment read that ``backtest.engine`` "therefore keeps the
# LEGACY identity, byte-for-byte" and that its resume-at-budget was "PRE-EXISTING and out of
# this plan's scope ... recorded for a later phase". That was true when Plan 30-01 wrote it.
# It is no longer true. Every "tuned" backtest between 2026-03-31 and Plan 30-16 was a straight
# re-fit on frozen v2.0 hyperparameters that still reported ``n_trials=100``; the owner ruled
# that the backtest must genuinely search, that the frozen v2.1 AUDIT-REPORT anchors would move,
# and that they be re-ratified with the drift RECORDED (Plan 30-16 Task 3). Option 1 -- document
# the freeze and move on -- was considered and explicitly NOT taken.
#
# WHY THE BACKTEST IDENTITY IS PER-RUN AND NOT PER-PHASE. The backtest is run repeatedly: three
# engine runs in the canonical PIPELINE sequence alone (``--blend`` runs the engine twice --
# blended, then the unblended baseline) plus every execution of the anchor test. A per-phase
# identity would make run 1 genuine and every later run a resume at budget: the same vacuity
# with a newer date on it.
#
# WHERE THE RUN ID GOES, AND WHY IT IS NOT IN THE STUDY NAME. The run id lives in the STORAGE
# PATH; the study NAME is the constant ``BACKTEST_TUNING_STUDY_TAG`` below. Both placements
# defeat cross-run vacuity equally -- a fresh run gets empty storage either way, so the stored
# trial count is 0 and the search genuinely runs. Only the storage placement ALSO keeps the
# search REPRODUCIBLE, because ``optuna.pruners.HyperbandPruner._get_bracket_id`` brackets each
# trial by a crc32 of the STUDY NAME (see that constant's own note for the citation and the
# arithmetic). Plan 30-16 first put the run id in the NAME and measured the consequence rather
# than assuming it: two runs of the identical anchor command came out 3.51e-3 apart on WP pooled
# accuracy -- 70% of the 5e-3 anchor band -- and five runs selected THREE different WP penalties.
# The owner ruled the split on 2026-08-24. Dropping or reconfiguring the pruner was considered
# and REJECTED: the backtest is a DIAGNOSTIC of the production models and the Stage-2 search runs
# under HyperbandPruner, so a backtest that searched without it would measure something
# production does not do.
#
# WHAT THE PER-RUN CHOICE COSTS, stated rather than left to be discovered. The storage carries
# the RUN, not the holdout season, so within one run the EARLIEST holdout season fills the study
# and the later ones resume it and add zero. Every fold of a run therefore trains under
# parameters searched on the earliest fold's train+hp_val window. Because holdout seasons are
# processed in ASCENDING order that is the temporally safe direction -- later folds use
# parameters chosen from strictly prior data -- and ``BacktestEngine.run()`` hard-fails a
# non-ascending holdout list so the safety is guarded rather than emergent. A per-FOLD identity
# would remove the reuse entirely and was considered; it multiplies every engine run by roughly
# four, forever, including inside the suite, and its only additional protection is against a
# reordered holdout list, which the guard closes directly and for free.

# The v2.0 identity every pre-Phase-30 caller keeps. Reusing it is what makes a search vacuous
# (those studies are already at budget), which is precisely why the Stage-2 path must not.
LEGACY_TUNING_STUDY_TAG: str = "v1"
LEGACY_TUNING_STORAGE_DIR: Path = Path("data/optuna")

# The per-PHASE study-identity tag. It exists because OptunaTuner.optimize computes remaining
# trials as ``max(0, n_trials - len(study.trials))`` under ``load_if_exists=True``: resuming a
# study that already holds the full budget runs ZERO new trials while still reporting a full
# trial count, so a "tuned" candidate would silently carry the OLD phase's parameters.
#
# The rule, plainly: a study identity is per-RUN of a binding search, not merely per-phase. A
# future phase (or a second binding search inside one phase) that re-tunes MUST bump this tag.
# What this tag is NOT allowed to do is revert to the v2.0 ``_tuning_v1`` literal -- those three
# studies are the v2.0 historical record and are already at budget, so reusing their identity
# guarantees a vacuous search. Nothing in this phase deletes them either.
#
# BUMPED ``p30`` -> ``p30s2`` by Plan 30-11 before the BINDING Stage-2 run, closing the
# D30-DEFER-02 item Plan 30-01 recorded. The Plan 30-01 tracer's dry run filled
# ``outputs/optuna/{target}_tuning_p30.db`` with the full 100-trial budget for all three
# targets (measured: 100/100/100 stored). ``OptunaTuner.optimize`` computes remaining as
# ``max(0, n_trials - len(study.trials))``, so a binding run on the ``p30`` identity would
# have added ZERO trials and hit the RuntimeError below -- at the moment of the phase's one
# irreversible action. The tag is bumped rather than the tracer's study files deleted:
# study files are the historical record of what was searched, and the guard's own remediation
# message says so.
#
# BUMPED ``p30s2`` -> ``p332s23`` by Plan 33.2-23 before the BINDING Phase-33.2 re-fit.
# EVERY study stored under the old tag was searched with MARKET-LINE COLUMNS inside the
# selected feature set, which D33.2-03 removes from every fit decision, so none of them is
# reusable -- and reusing the identity would resume a study already at budget and return
# those unusable parameters while reporting a full trial count.
#
# CHANGING THIS TAG CHANGES THE SEARCH, not merely its label, and that is INTENDED here.
# ``optuna.pruners.HyperbandPruner._get_bracket_id`` brackets each trial by a crc32 of the
# STUDY NAME (see BACKTEST_TUNING_STUDY_TAG's note for the citation and the measured
# consequence), so a different name prunes a different set of trials. Under this plan the
# two arms are ``{target}_tuning_{TAG}_tpe`` and ``{target}_tuning_{TAG}_random``, which
# therefore prune on DIFFERENT schedules even under one PRUNER_CONFIG -- declared rather
# than hidden, and made harmless because NO adoption decision reads an in-search value.
TUNING_STUDY_TAG: str = "p332s23"

# Where the Phase-30 SQLite study files live. OptunaTuner defaults storage_dir to
# ``data/optuna``, which collides with this phase's own prohibition on writing under ``data/``
# outside the one sanctioned fingerprinted rebuild -- and the prohibition's before/after hash
# manifest reads through ``load_dataframe``, so it would NOT have caught a ``data/optuna/``
# write. ``outputs/`` is gitignored and is the right home. What this constant is NOT allowed to
# be is any path under ``data/``; tests/unit/test_promote_models_tuned_path.py asserts that.
TUNING_STORAGE_DIR: Path = Path("outputs/optuna")

# The backtest study tag (Plan 30-16, D30-OWNER-02, as amended by the owner's 2026-08-24 ruling
# on the Task 3 halt). It is CONSTANT across runs. What varies per run is the STORAGE DIRECTORY
# below, not this name.
#
# THIS LITERAL'S VALUE IS LOAD-BEARING. It is not a cosmetic label, and it must NOT be bumped the
# way ``TUNING_STUDY_TAG`` above is deliberately bumped per phase.
# ``optuna.pruners.HyperbandPruner._get_bracket_id``
# (``optuna/pruners/_hyperband.py:255-258``, verified against the installed library) assigns each
# trial to a Hyperband bracket by taking the CRC32 of the study name joined to the trial number
# by an underscore, modulo the pruner's total trial allocation budget. So the study NAME decides
# which trials get PRUNED. Change this string and a different set of
# trials is pruned, the search returns different hyperparameters, and the v2.1 AUDIT-REPORT
# anchors that tests/integration/test_diag_diagnosis.py pins to 5e-3 MOVE -- silently, because
# nothing else would fail. Measured, not theorised: with the run id in the study name (this
# plan's first attempt) two runs of the identical anchor command differed by 3.51e-3 on WP
# pooled accuracy, 70% of the anchor band, and five runs chose three different WP penalties.
# The control -- constant name, separate storage -- returned byte-identical best parameters.
# tests/unit/test_backtest_tuning_identity.py asserts this EXACT string, so a rename fails
# loudly instead of drifting the anchors under a green suite.
#
# If it must ever change, that change is a RE-RATIFICATION, not a rename: re-measure the anchors
# TWICE, confirm the two readings agree, and move them with the drift recorded (IN-03).
#
# What this tag must never be: ``v1`` (the v2.0 identity -- reusing it guarantees a vacuous
# resume) or ``TUNING_STUDY_TAG`` (the Stage-2 identity -- a diagnostic backtest would then eat
# the binding candidate search's trial budget).
BACKTEST_TUNING_STUDY_TAG: str = "p30bt"

# Where the per-run backtest study files live: ``outputs/optuna/backtest/{run_id}/``. The
# per-run SUBDIRECTORY is what carries the run id, and it is the whole anti-vacuity property:
# a fresh run resolves a directory that does not exist yet, so the stored trial count is 0 and
# the search actually runs. The same T-30-14 constraint the Stage-2 storage carries applies --
# NEVER under ``data/`` -- because OptunaTuner defaults its storage to ``data/optuna`` and this
# phase forbids writes there. The per-run subdirectory also keeps the tuning provenance record
# beside the studies it describes.
BACKTEST_TUNING_STORAGE_DIR: Path = Path("outputs/optuna/backtest")


def _existing_trial_count(tuner: OptunaTuner) -> int:
    """Return how many trials the tuner's study ALREADY holds in storage.

    Read BEFORE ``optimize`` so the caller can compute how many trials the search genuinely
    added. A missing study file, or a storage file with no such study, is 0 -- the fresh case.

    Args:
        tuner: The configured OptunaTuner (read-only; its storage is not created here).

    Returns:
        The stored trial count, or 0 when the study does not exist yet.
    """
    return _stored_trial_counts(tuner)[0]


def _existing_completed_trial_count(tuner: OptunaTuner) -> int:
    """Return how many of the tuner's stored trials are in the ``COMPLETE`` state.

    WR-08: the sibling above counts trials STARTED, which is the right question for the
    anti-vacuity check but the wrong one for "how much searching happened". Under
    ``HyperbandPruner`` most trials are pruned, so the two numbers differ substantially.
    """
    return _stored_trial_counts(tuner)[1]


def _stored_trial_counts(tuner: OptunaTuner) -> tuple[int, int]:
    """Return ``(started, completed)`` trial counts for the tuner's stored study.

    One read of storage answers both, so the two counts can never disagree about which
    study state they describe. A missing study file, or a storage file with no such
    study, is ``(0, 0)`` -- the fresh case.
    """
    db_path = tuner.storage_dir / f"{tuner.study_name}.db"
    if not db_path.exists():
        return 0, 0
    try:
        study = optuna.load_study(
            study_name=tuner.study_name, storage=tuner.storage_url
        )
    except KeyError:
        # The storage file exists but holds no study by this name (the fresh-identity case,
        # e.g. right after TUNING_STUDY_TAG was bumped).
        return 0, 0
    return len(study.trials), count_completed_trials(study)


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


def _zero_trial_message(
    study_name: str,
    trials_before: int,
    trials_after: int,
    budget: int,
) -> str:
    """The message a vacuous resume raises with, declared ONCE.

    Both the legacy single-arm path and the pre-registered two-arm path raise it, and
    ``tests/unit/test_promote_models_tuned_path.py`` asserts it names both the study and
    the remediation. Two copies of it would be two contracts wearing one name.
    """
    return (
        f"Optuna study '{study_name}' added ZERO new trials "
        f"(stored before={trials_before}, after={trials_after}, budget={budget}). "
        "The study was resumed already at budget, so the 'best params' returned are the "
        "STORED ones -- this candidate would be reported as tuned without having been "
        "tuned. Remediation: bump TUNING_STUDY_TAG in models.trainers.base to open a "
        "fresh study identity. Do NOT delete the existing study file to work around "
        "this -- study files are the historical record of what was searched."
    )


def decide_adoption(
    target: str,
    arms: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Decide whether *target*'s SEARCHED winner beat the RANDOM winner by the bar.

    THE DECISION READS ONE NUMBER PER ARM, AND IT IS THE OUT-OF-SAMPLE ONE. Both arms
    optimise the same three temporal CV folds (:meth:`BaseTrainer._make_objective` over
    ``make_temporal_cv_splits``), so the gap between their ``in_search_best_value`` entries
    is a gap between two winners selected on the SAME noise -- it cannot show that the
    searched winner generalises. Each arm's ``outer_score`` is its best setting REFIT on
    the same train+hp_val data and scored ONCE on the season named by
    ``config.tuning_preregistration.OUTER_COMPARISON_RULE``, which neither search ever saw.
    ``in_search_best_value`` is recorded and decides nothing.

    THE MARGIN IS READ FROM THE PRE-REGISTRATION AND IS NEVER A LITERAL HERE. A number
    written into this function could be changed after the search ran with nothing to catch
    it; a number in ``config/tuning_preregistration.py`` is locked by a content hash and a
    git-ancestry assertion (``tests/unit/test_tuning_preregistration.py``).

    Every target's metric is LOWER-IS-BETTER (``METRIC_BY_TARGET``), so the gap in the
    searched winner's favour is ``random.outer_score - tpe.outer_score``.

    Args:
        target: ``"wp"``, ``"ats"`` or ``"ou"``.
        arms: The per-arm record, keyed by ``STUDY_ARM_TPE`` / ``STUDY_ARM_RANDOM``. Each
            must carry ``outer_score``.

    Returns:
        The adoption record: the margin that had to be cleared, the observed gap, whether
        it cleared, which arm was adopted, and -- when it did NOT clear -- the
        pre-registered not-cleared rule that was applied, so the outcome is a RECORDED
        FINDING rather than a silent fall-through.
    """
    margin = BEAT_RANDOM_MARGIN_BY_TARGET[target]
    tpe_outer = float(arms[STUDY_ARM_TPE]["outer_score"])
    random_outer = float(arms[STUDY_ARM_RANDOM]["outer_score"])
    gap = random_outer - tpe_outer
    cleared = bool(gap >= margin)
    return {
        "margin": margin,
        "outer_gap": gap,
        "margin_cleared": cleared,
        "adopted_arm": STUDY_ARM_TPE if cleared else NOT_CLEARED_ARM,
        "not_cleared_rule": "" if cleared else NOT_CLEARED_RULE,
        "outer_comparison_rule": OUTER_COMPARISON_RULE,
    }


def informative_columns(X: pd.DataFrame) -> list[str]:
    """Return the columns of ``X`` that vary over these rows, in their original order.

    "Informative" here means only "not constant over the fit window". A column that is
    constant, or entirely NaN, carries nothing a model can learn from -- but it still
    perturbs a column-sampling estimator's fit, which is the defect this exists to close.

    WR-13: distinctness is counted over the OBSERVED values (``dropna=True``). The
    original used ``dropna=False``, which counts NaN as a distinct level -- so an all-NaN
    column scored 1 and was correctly withheld, but a column that is a single constant on
    its observed rows and NaN elsewhere (``[5.0, 5.0, NaN, 5.0]``) scored 2 and reached
    the fit, where it perturbs ``XGBRegressor``'s ``colsample_bytree=0.8`` sampling and
    therefore the gain importances of every real feature. That is exactly the
    count-dependence this pre-filter exists to remove, and the partial case is the one it
    most needs to catch.

    Measured on the shipped gold no such column exists in either the 2018-2019 or the
    2015-2019 window, so the defect was latent and this change withholds nothing new
    today. It becomes live the moment a late-arriving upstream source lands a
    neutral-default column -- and ``handle_missing_data_and_outliers`` now deliberately
    LEAVES NaNs in place where no prior fit source exists
    (``_impute_game_level_features``), which makes partially-NaN constant columns MORE
    likely than before, not less.

    Args:
        X: The frame the scoring model is about to be fitted on.

    Returns:
        The subset of ``X.columns`` carrying more than one OBSERVED value. An all-NaN
        column has zero observed values and is withheld; so is a column with one.

    Raises:
        ValueError: If ``X`` carries duplicated column labels. ``distinct[column]`` would
            return a Series and the ``> 1`` comparison would raise
            ``ValueError: truth value ambiguous`` from inside a comprehension, which is a
            far worse diagnostic than saying so here.
    """
    if X.columns.has_duplicates:
        duplicated = sorted({str(name) for name in X.columns[X.columns.duplicated()]})
        msg = (
            f"informative_columns received duplicated column labels {duplicated}; "
            "per-column nunique is ambiguous and the zero-variance pre-filter cannot "
            "decide what to withhold. De-duplicate the frame before selection."
        )
        raise ValueError(msg)

    # A column that is a single OBSERVED value carries nothing a model can learn from,
    # whether or not it is also missing somewhere.
    distinct = X.nunique(dropna=True)
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
        preprocessing: The fitted preprocessing object persisted ALONGSIDE the model
            (D33.1-R1), or None for a trainer that has none.
    """

    # The class DEFAULT for the persisted-preprocessing contract (D33.1-R1, Plan
    # 33.1-10). It is a DEFAULT and not reachable behaviour: ATS and O/U carry the
    # attribute with a None value so `save` can read it unconditionally, and neither
    # trainer's behaviour changes by one byte. WP assigns the fitted four-step Pipeline
    # in its own `train_and_evaluate`.
    #
    # `models/trainers/base.py`'s statement that `train_and_evaluate` is unreachable dead
    # code is untouched by this: a class attribute is not a reachable sibling method, and
    # there is deliberately NO `final_fit` here -- that entry point is a module beside the
    # three concrete trainers (Ruling T), so this base class's own documentation stays
    # true.
    preprocessing: Any = None

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

        # Tuning identity defaults to LEGACY so every caller that does not opt in
        # (scripts.retrain_models, and any future one) is byte-identical to its prior
        # behaviour. The Stage-2 train opts in via use_phase30_tuning(); backtest.engine opts
        # in via use_backtest_tuning(run_id).
        self.tuning_study_tag: str = LEGACY_TUNING_STUDY_TAG
        self.tuning_storage_dir: Path = LEGACY_TUNING_STORAGE_DIR
        self.require_fresh_search: bool = False

        # Plan 33.2-23 / D33.2-17. FALSE by default, so every existing caller keeps the
        # single-arm, 100-trial-defaulted search byte-for-byte. The Phase-33.2 re-fit opts
        # in via use_phase332_tuning(), after which the budget comes from the
        # pre-registration, a RandomSampler baseline runs FIRST over the same objective and
        # the same budget, and the searched winner is adopted only if it beats that
        # baseline on a season neither arm saw.
        self.preregistered_search: bool = False
        self.adoption_record: dict[str, Any] | None = None

        # What the LAST tuning search on this instance actually did. Initialised here so the
        # attributes exist even on an untuned path (tune=False never sets them). These three
        # values already existed as locals inside tune_hyperparameters and were already logged;
        # keeping them on the instance is what lets backtest.engine write a tuning provenance
        # record a reader can check, instead of a plan asserting the search was real.
        self.last_tuning_study_name: str | None = None
        self.last_tuning_trials_before: int | None = None
        self.last_tuning_trials_added: int | None = None
        # WR-08: the two above count trials STARTED (len(study.trials), which includes
        # PRUNED and FAIL). These count the ones that ran to completion.
        self.last_tuning_completed_before: int | None = None
        self.last_tuning_completed_added: int | None = None

    def use_phase30_tuning(self) -> None:
        """Opt this trainer into the PER-PHASE Stage-2 identity, storage and freshness guard.

        Called by ``models.train.train_target`` on the tuned path -- the Stage-2 candidate
        train that ``scripts/promote_models`` STEP 1 invokes. After this call the trainer
        searches a FRESH per-phase study under a non-``data/`` storage dir, and a search that
        adds zero trials is a hard failure rather than a silent return of stored parameters.

        Deliberately NOT the default: see the module-level note above. Making any non-legacy
        identity the default would change what every non-opted-in caller trains with in one
        move, which is the elevation-of-scope this repository's three-identity split prevents.
        This is the PER-PHASE identity; ``use_backtest_tuning`` is the PER-RUN one, and they
        are separate so a diagnostic backtest can never consume the binding search's budget.
        """
        self.tuning_study_tag = TUNING_STUDY_TAG
        self.tuning_storage_dir = TUNING_STORAGE_DIR
        self.require_fresh_search = True

    def use_phase332_tuning(self) -> None:
        """Opt this trainer into the PRE-REGISTERED two-arm search (D33.2-17, Plan 33.2-23).

        It is the PER-PHASE identity plus three properties that identity alone does not
        carry, each of which silently does not happen if it is not switched on:

        * the trial BUDGET comes from ``TRIAL_BUDGET_BY_TARGET`` and the 100-trial default
          that both ``models/tuning.py`` and this module carry becomes UNREACHABLE. A
          caller that passes a contradicting budget fails loudly rather than quietly
          shipping a hundred-trial search described as a deep one;
        * a RANDOM-sampler baseline runs FIRST, over the SAME objective object and the
          SAME budget, as a SEPARATE study that cannot resume the tuned arm's;
        * the searched winner is adopted only if it beats that baseline by the
          pre-registered per-target margin on the OUTER season -- and when it does not, the
          pre-registered not-cleared rule is applied and RECORDED.

        Deliberately NOT the default and deliberately NOT folded into
        :meth:`use_phase30_tuning`: that opt-in is what ``scripts/promote_models`` and
        every other Stage-2 caller already use, and changing what they search in one move
        is the elevation of scope this class's identity split exists to prevent.
        """
        self.use_phase30_tuning()
        self.preregistered_search = True

    def use_backtest_tuning(self, run_id: str) -> None:
        """Opt this trainer into the PER-RUN backtest study storage (Plan 30-16).

        Called by ``BacktestEngine._create_trainer`` for every trainer it builds -- every
        target, every holdout season -- so the identity cannot be forgotten for one of them.
        After this call the trainer searches under ``outputs/optuna/backtest/{run_id}/``, a
        directory no earlier run has written, so no backtest run can inherit another run's
        stored parameters. That is the defect D30-DEFER-01 recorded and D30-OWNER-02 ruled must
        be fixed.

        THE STUDY NAME IS CONSTANT, AND ONLY THE STORAGE IS PER RUN. Putting the run id in the
        NAME would defeat vacuity just as well and would ALSO break reproducibility, because
        ``HyperbandPruner`` brackets trials by a crc32 of the study name -- see the
        ``BACKTEST_TUNING_STUDY_TAG`` note above for the citation and the measured consequence.
        The owner ruled this split on 2026-08-24 after Plan 30-16 measured two anchor readings
        3.51e-3 apart under per-run naming.

        WHY THE FRESHNESS GUARD IS DELIBERATELY LEFT DISARMED. ``require_fresh_search`` raises
        when a search adds zero trials. Within ONE engine run that is the normal, correct case
        for every holdout season after the first: the storage carries the run and not the
        season, so the earliest season fills the study and the later ones resume it. Arming the
        guard here would make every multi-season backtest raise on its second fold. The property
        that replaces the guard is the per-run storage itself -- cross-run vacuity is impossible
        by construction, and within-run reuse is stated here, recorded in the run's tuning
        provenance file, and made temporally safe by the engine's ascending-holdout hard
        failure.

        Args:
            run_id: The engine instance's unique run id. Must be unique per BacktestEngine
                instance, not merely per second -- ``run_backtest(blend=True)`` constructs two
                engines in immediate succession, and a colliding id would make them share one
                storage directory.
        """
        self.tuning_study_tag = BACKTEST_TUNING_STUDY_TAG
        self.tuning_storage_dir = BACKTEST_TUNING_STORAGE_DIR / run_id
        self.require_fresh_search = False

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
        if not informative:
            # WR-13: the fallback silently restored the count-dependent behaviour it
            # exists to remove. It is still the right disposition -- refusing to select
            # at all is worse -- but a frame in which NOTHING varies means the window is
            # degenerate, and that has to be said rather than absorbed.
            self.logger.warning(
                "No informative columns over this fit window; scoring the frame as it "
                "stands, which restores the count-dependent selection the pre-filter "
                "exists to remove",
                target=self.target,
                total_features=len(X.columns),
                rows=len(X),
            )
        fit_frame = X[informative] if informative else X

        # D33.1-R3: for WP this is now a four-step Pipeline, and that IS the fix.
        #
        # `informative_columns` above withholds a column that does not VARY. A
        # NaN-bearing weather column DOES vary, so it is offered to the
        # estimator -- and a bare `LogisticRegression` raises
        # `ValueError: Input X contains NaN` on it. The currently-deployed WP
        # feature list containing zero weather features does NOT protect
        # re-selection: this function re-runs on every `train_and_evaluate`
        # call and `models.train` exposes no feature-list flag, so a re-fit
        # re-selects from the whole informative candidate pool.
        #
        # Behaviour-preserving for ATS and O/U, whose `_create_model` returns an
        # XGBoost model that takes NaN natively.
        model = self._create_model(self._get_default_params())
        model.fit(fit_frame, y)

        selected_mask = self._selection_support(model, fit_frame, max_features)
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

    @staticmethod
    def _selection_support(
        model: Any,
        fit_frame: pd.DataFrame,
        max_features: int | None,
    ) -> np.ndarray:
        """Return a boolean mask over ``fit_frame.columns``.

        Two things a plain ``SelectFromModel(model, prefit=True)`` cannot do
        once WP's ``_create_model`` returns a Pipeline (D33.1-R3), both of them
        mechanical rather than a change of rule:

        * A ``Pipeline`` exposes neither ``coef_`` nor ``feature_importances_``,
          so the importances are read off its FINAL STEP. Which of the two
          shapes was needed is recorded here rather than left to be rediscovered:
          it is the final step, because sklearn's importance getter does not
          traverse a pipeline.
        * WP's estimator sees TWICE the input width -- the imputed values and
          then one missing indicator per column. The mask therefore comes back
          at double width and is OR-reduced back onto the original columns: a
          column is selected if its VALUE or its ABSENCE scored at or above the
          mean. Dropping the indicator half instead would silently discard the
          half of the signal that "this reading was missing" carries.

        Byte-identical for a non-Pipeline estimator of matching width, which is
        every ATS and O/U selection.

        Args:
            model: The fitted scoring model.
            fit_frame: The frame it was fitted on.
            max_features: Cap passed through to ``SelectFromModel``.

        Returns:
            Boolean mask of length ``len(fit_frame.columns)``.
        """
        scoring_model = model
        if isinstance(model, Pipeline):
            scoring_model = model.steps[-1][1]

        selector = SelectFromModel(
            scoring_model, max_features=max_features, threshold="mean", prefit=True
        )
        support = np.asarray(selector.get_support())

        width = len(fit_frame.columns)
        if len(support) == width:
            return support
        if len(support) % width == 0:
            blocks = support.reshape(len(support) // width, width)
            return blocks.any(axis=0)
        msg = (
            f"feature-selection support has length {len(support)}, which is "
            f"neither {width} nor a multiple of it. The scoring model's feature "
            "space cannot be mapped back onto the fit frame's columns, and "
            "guessing an alignment here would silently select the wrong ones."
        )
        raise ValueError(msg)

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

    def _resolve_trial_budget(self, n_trials: int | None) -> int:
        """Resolve how many trials this search may START, and refuse a silent default.

        THE DEFAULT IS THE DEFECT THIS EXISTS TO CLOSE. ``models/tuning.py`` and this
        module BOTH default ``n_trials`` to 100, so a plan that says "deep search" without
        setting the budget ships a hundred-trial search and nothing in the run says
        otherwise. On the pre-registered path the budget therefore comes from
        ``config.tuning_preregistration.TRIAL_BUDGET_BY_TARGET`` and a caller that
        contradicts it fails LOUDLY.

        Args:
            n_trials: What the caller asked for, or None for "resolve it".

        Returns:
            The budget, in trials STARTED.

        Raises:
            RuntimeError: On the pre-registered path, when the caller named a budget that
                is not the pre-registered one.
        """
        if not self.preregistered_search:
            # The LEGACY resolution, byte-identical to the former signature default.
            return 100 if n_trials is None else n_trials

        budget = TRIAL_BUDGET_BY_TARGET[self.target]
        if n_trials is not None and n_trials != budget:
            msg = (
                f"tune_hyperparameters was called with n_trials={n_trials} on the "
                f"PRE-REGISTERED path, where the budget for target '{self.target}' is "
                f"{budget} (config.tuning_preregistration.TRIAL_BUDGET_BY_TARGET). The "
                "budget is pre-registered precisely because both models/tuning.py and "
                "models/trainers/base.py default it to 100, so an unnoticed argument "
                "would ship a hundred-trial search described as a deep one. Pass the "
                "pre-registered budget or pass nothing."
            )
            raise RuntimeError(msg)
        return budget

    def tune_hyperparameters(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        n_trials: int | None = None,
        season_week_df: pd.DataFrame | None = None,
        full_features_df: pd.DataFrame | None = None,
    ) -> dict:
        """Tune hyperparameters using Optuna with temporal CV folds.

        Per D-01, delegates to OptunaTuner. Per D-02, uses TPE sampler
        with 100+ trials and Hyperband pruner. Per D-03, uses SQLite
        storage for resumability. Per D-05, uses target-specific
        optimization metric (direction derived from _get_scoring_metric).

        TWO PATHS, and which one runs is an EXPLICIT opt-in. Without
        :meth:`use_phase332_tuning` this is the single-arm search it has always been, with
        the same 100-trial default and the same study name. With it, the pre-registered
        two-arm search of D33.2-17 runs instead -- see
        :meth:`_run_preregistered_two_arm_search`.

        Args:
            X_train: Training features (train + hp_val combined).
            y_train: Training targets.
            n_trials: Number of Optuna trials. None resolves to the legacy default of 100,
                or -- on the pre-registered path -- to the pre-registered per-target
                budget. See :meth:`_resolve_trial_budget`.
            season_week_df: DataFrame with "season" and "week" columns
                for temporal CV splits. If None, creates index-based splits.
            full_features_df: The FULL feature matrix, including the outer comparison
                season. Required on the pre-registered path and IGNORED otherwise: the
                adoption gate scores each arm on a season that is by construction absent
                from ``X_train``.

        Returns:
            Best parameters dict.
        """
        budget = self._resolve_trial_budget(n_trials)

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

        objective = self._make_objective(X_train, y_train, cv_splits)

        if self.preregistered_search:
            return self._run_preregistered_two_arm_search(
                objective=objective,
                X_train=X_train,
                y_train=y_train,
                budget=budget,
                direction=direction,
                full_features_df=full_features_df,
            )

        # Study identity + storage come from INSTANCE state, so the Stage-2 opt-in
        # (use_phase30_tuning) can require a genuinely fresh search without changing what any
        # other caller trains with (T-30-02 / T-30-14). storage_dir is passed explicitly --
        # omitting it lets OptunaTuner default to ``data/optuna`` regardless of this tag.
        study_name = f"{self.target}_tuning_{self.tuning_study_tag}"

        tuner = OptunaTuner(
            study_name=study_name,
            direction=direction,
            storage_dir=self.tuning_storage_dir,
            n_trials=budget,
        )

        # Read the stored trial counts BEFORE the search so a vacuous resume is detectable.
        # Both are read: STARTED answers "did this search add anything at all", which is the
        # anti-vacuity question; COMPLETED answers "how much searching actually happened",
        # which is what a reader assumes a "trials" figure means (WR-08).
        trials_before, completed_before = _stored_trial_counts(tuner)

        result = tuner.optimize(objective)

        # HARD-fail a search that added nothing. A resumed full study returns the STORED best
        # parameters while reporting a full trial count, so without this assertion an untuned
        # candidate is indistinguishable from a tuned one in every downstream artifact.
        trials_added = result.n_trials - trials_before

        # WR-08: trials_added counts trials STARTED, which is what the anti-vacuity check
        # below needs -- a study resumed at budget starts none. It is NOT the number of
        # searches that ran to completion: HyperbandPruner prunes the majority, and
        # `tuning_provenance.json` and GATED-REFIT-READOUT's "NEW completed trials" column
        # were both reading a started count under a completed label. Both are recorded now,
        # each under its real name.
        completed_added = result.n_completed_trials - completed_before

        # Keep what the search DID where a caller can read it after train_and_evaluate returns.
        # backtest.engine reads these per (target, holdout season) to write its tuning
        # provenance record, which is what makes "this run genuinely searched" checkable by a
        # reader rather than asserted by a plan (Plan 30-16, T-30-63).
        self.last_tuning_study_name = study_name
        self.last_tuning_trials_before = trials_before
        self.last_tuning_trials_added = trials_added
        self.last_tuning_completed_before = completed_before
        self.last_tuning_completed_added = completed_added

        if self.require_fresh_search and trials_added <= 0:
            raise RuntimeError(
                _zero_trial_message(study_name, trials_before, result.n_trials, budget)
            )

        # Replay best params through _define_search_space to get
        # model-compatible parameter names (e.g., "solver_l2" -> "solver")
        fixed_trial = optuna.trial.FixedTrial(result.best_params)
        best_params = self._define_search_space(fixed_trial)

        self.logger.info(
            "Optuna tuning completed",
            target=self.target,
            best_value=result.best_value,
            best_params=best_params,
            n_trials_started=result.n_trials,
            n_trials_completed=result.n_completed_trials,
            trials_started_added=trials_added,
            trials_completed_added=completed_added,
            study_name=study_name,
            storage_dir=str(self.tuning_storage_dir),
            top_importances=dict(list(result.param_importances.items())[:5]),
        )

        # Store tuning result for later use in save()
        self._tuning_result = result

        return best_params

    # ------------------------------------------------------------------
    # The PRE-REGISTERED two-arm search (D33.2-17, Plan 33.2-23)
    # ------------------------------------------------------------------

    def _run_one_arm(
        self,
        arm: str,
        objective: Callable[[optuna.Trial], float],
        budget: int,
        direction: str,
    ) -> tuple[TuningResult, dict[str, Any]]:
        """Run ONE arm of the pre-registered search and return its result and its record.

        Both arms come through here, with the SAME ``objective`` OBJECT and the SAME
        ``budget``. That is what makes "the two arms searched the same space with the same
        budget" true BY CONSTRUCTION rather than by a second declaration that can drift --
        and a smaller random budget would rig the comparison in the searched winner's
        favour.

        Args:
            arm: ``STUDY_ARM_RANDOM`` or ``STUDY_ARM_TPE``.
            objective: The shared objective, built ONCE by the caller.
            budget: Trials to START, from the pre-registration.
            direction: Optuna's optimisation direction.

        Returns:
            ``(result, record)`` -- the raw ``TuningResult`` and the per-arm record that
            goes into the artifact's tuning metadata.

        Raises:
            RuntimeError: If the arm resumed a study already at budget (zero new trials),
                or if it did not START exactly the pre-registered budget.
        """
        sampler = (
            optuna.samplers.RandomSampler(seed=RANDOM_SAMPLER_SEED)
            if arm == STUDY_ARM_RANDOM
            else optuna.samplers.TPESampler(
                seed=TPE_SAMPLER_SEED, n_startup_trials=TPE_SAMPLER_STARTUP_TRIALS
            )
        )
        name = preregistered_study_name(self.target, self.tuning_study_tag, arm)
        tuner = OptunaTuner(
            study_name=name,
            direction=direction,
            storage_dir=self.tuning_storage_dir,
            n_trials=budget,
            sampler=sampler,
            # ONE pruner configuration for both arms. The crc32-of-the-study-name bracket
            # still differs between two differently-named studies; that is declared in
            # RANDOM_COMPARATOR and is harmless because no decision reads an in-search
            # value.
            pruner=optuna.pruners.HyperbandPruner(**PRUNER_CONFIG),
        )

        trials_before, completed_before = _stored_trial_counts(tuner)
        result = tuner.optimize(objective)
        trials_added = result.n_trials - trials_before

        if self.require_fresh_search and trials_added <= 0:
            raise RuntimeError(
                _zero_trial_message(name, trials_before, result.n_trials, budget)
            )

        # The POSITIVE form of the anti-vacuity check, and the only count the two arms can
        # be held EQUAL on. A completed-trial floor cannot be relied on to pass: under
        # HyperbandPruner most trials are pruned BY DESIGN, and two arms with different
        # samplers and different crc32 brackets will not complete the same number.
        if result.n_trials != budget:
            msg = (
                f"arm '{arm}' of target '{self.target}' recorded "
                f"trials_started={result.n_trials}, but the pre-registered budget is "
                f"{budget}. The budget is asserted in trials STARTED because that is the "
                "only count both arms can be held equal on; a search that started a "
                "different number is not the search that was pre-registered."
            )
            raise RuntimeError(msg)

        record = {
            "study_name": name,
            "sampler": type(tuner.sampler).__name__,
            "trials_started": result.n_trials,
            "trials_started_added": trials_added,
            "trials_completed": result.n_completed_trials,
            "trials_completed_added": result.n_completed_trials - completed_before,
            "trials_pruned": result.n_pruned_trials,
            "trials_failed": result.n_failed_trials,
            # RECORDED AND DECIDES NOTHING. Two winners selected on the same folds can
            # differ by noise alone; only ``outer_score`` enters the adoption gate.
            "in_search_best_value": result.best_value,
            "outer_score": None,
        }
        return result, record

    def _score_on_outer_season(
        self,
        best_params: dict,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_outer: pd.DataFrame,
        y_outer: pd.Series,
    ) -> float:
        """Refit *best_params* on train+hp_val and score it ONCE on the outer season.

        The SAME training data for both arms and the SAME metric the tuner minimised
        (:meth:`_compute_cv_score`), so the only thing that differs between the two numbers
        the adoption gate compares is the hyperparameter setting.
        """
        model = self._create_model(best_params)
        model.fit(X_train, y_train)
        predictions = self._predict_raw(model, X_outer)
        return float(self._compute_cv_score(predictions, y_outer))

    def _run_preregistered_two_arm_search(
        self,
        objective: Callable[[optuna.Trial], float],
        X_train: pd.DataFrame,
        y_train: pd.Series,
        budget: int,
        direction: str,
        full_features_df: pd.DataFrame | None,
    ) -> dict:
        """Run the RANDOM baseline, then the TPE search, then the outer-season gate.

        THE ORDER IS PART OF THE DESIGN. The random baseline runs FIRST because it is the
        comparator: a baseline established after seeing the searched winner is not a
        baseline. Each arm is a SEPARATE study under a name the other cannot resume.

        Returns:
            The adopted parameters -- the searched winner when it beat the random winner by
            the pre-registered margin on the outer season, otherwise the pre-registered
            not-cleared rule's answer, which is the trainer's standard defaults.

        Raises:
            RuntimeError: If no ``full_features_df`` was supplied. Without it there is no
                outer season, and a "margin" computed on the folds both arms optimised
                would prove nothing -- so this refuses rather than degrading quietly.
        """
        if full_features_df is None or "season" not in getattr(
            full_features_df, "columns", []
        ):
            msg = (
                "the pre-registered search needs full_features_df (a frame carrying a "
                "'season' column) to resolve and score the OUTER comparison season. "
                f"{OUTER_COMPARISON_RULE} Without it the only available comparison is "
                "between two winners selected on the same folds, which cannot show "
                "generalisation -- so this is a refusal, not a fallback."
            )
            raise RuntimeError(msg)

        # NON-VACUITY, asserted rather than assumed. An EMPTY search space would make both
        # arms search nothing, both best values identical and the gap trivially zero -- a
        # guard that passes by measuring nothing. The space is the pre-registered one, read
        # here from the same declaration the three trainers' suggest_* calls read.
        searched_space = SEARCH_SPACE_BY_TARGET[self.target]
        if not searched_space:
            msg = (
                f"the pre-registered search space for target '{self.target}' is EMPTY. "
                "Both arms would search nothing, their best values would be identical and "
                "the margin gate would be satisfied by measuring nothing."
            )
            raise RuntimeError(msg)

        outer_season = outer_comparison_season(full_features_df["season"])

        arms: dict[str, dict[str, Any]] = {}
        results: dict[str, TuningResult] = {}
        adopted_params_by_arm: dict[str, dict] = {}

        # RANDOM FIRST. Both arms receive the SAME objective object and the SAME budget.
        for arm in (STUDY_ARM_RANDOM, STUDY_ARM_TPE):
            result, record = self._run_one_arm(arm, objective, budget, direction)
            results[arm] = result
            arms[arm] = record
            # Replay through _define_search_space so the parameter NAMES are the ones the
            # estimator takes (e.g. WP's "solver_l2" -> "solver").
            adopted_params_by_arm[arm] = self._define_search_space(
                optuna.trial.FixedTrial(result.best_params)
            )

        feature_columns = list(X_train.columns)
        outer_rows = full_features_df[full_features_df["season"] == outer_season]
        X_outer = outer_rows[feature_columns]
        y_outer = outer_rows[self._get_target_column()]

        for arm in (STUDY_ARM_RANDOM, STUDY_ARM_TPE):
            arms[arm]["outer_score"] = self._score_on_outer_season(
                adopted_params_by_arm[arm], X_train, y_train, X_outer, y_outer
            )

        decision = decide_adoption(self.target, arms)
        adopted_arm = decision["adopted_arm"]
        best_params = (
            adopted_params_by_arm[STUDY_ARM_TPE]
            if adopted_arm == STUDY_ARM_TPE
            else self._get_default_params()
        )

        # The tuned arm is the headline TuningResult the save path already reads; the
        # per-arm record below is what an auditor actually needs.
        tuned = results[STUDY_ARM_TPE]
        self._tuning_result = tuned
        self.last_tuning_study_name = arms[STUDY_ARM_TPE]["study_name"]
        self.last_tuning_trials_before = (
            tuned.n_trials - arms[STUDY_ARM_TPE]["trials_started_added"]
        )
        self.last_tuning_trials_added = arms[STUDY_ARM_TPE]["trials_started_added"]
        self.last_tuning_completed_before = (
            tuned.n_completed_trials - arms[STUDY_ARM_TPE]["trials_completed_added"]
        )
        self.last_tuning_completed_added = arms[STUDY_ARM_TPE]["trials_completed_added"]

        self.adoption_record = {
            "arms": arms,
            "outer_season": outer_season,
            "outer_season_n_games": len(outer_rows),
            "search_space_digest": search_space_digest(self.target),
            "trial_budget": budget,
            "thread_limit": PINNED_THREAD_COUNT,
            "study_tag": self.tuning_study_tag,
            "adopted_params": best_params,
            **decision,
        }

        self.logger.info(
            "Pre-registered two-arm search completed",
            target=self.target,
            outer_season=outer_season,
            margin=decision["margin"],
            outer_gap=decision["outer_gap"],
            margin_cleared=decision["margin_cleared"],
            adopted_arm=adopted_arm,
            tpe_outer=arms[STUDY_ARM_TPE]["outer_score"],
            random_outer=arms[STUDY_ARM_RANDOM]["outer_score"],
            trials_started=budget,
            thread_limit=PINNED_THREAD_COUNT,
        )

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
                # D33.2-17: the FULL frame, so the pre-registered adoption gate can
                # resolve and score the OUTER comparison season -- a season that is by
                # construction absent from combined_train. Ignored on every other path.
                full_features_df=features_df,
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

        # D33.1-R1: record the converter parameters where SERVING can read them.
        #
        # This method has always passed `self.calibrator`, which ATS and O/U leave None --
        # their fitted `residual_converter` / `total_converter` were never persisted under
        # ANY name, and `prediction_pipeline` rebuilt conversion from
        # `metadata.get("residual_std", 13.5)` / `13.0`, hardcoded fallbacks standing in
        # for a fitted object. The hook is read through `getattr` so a trainer that does
        # not define it is unaffected, byte for byte.
        converter_params_hook = getattr(self, "converter_params", None)
        if callable(converter_params_hook):
            converter_params = converter_params_hook()
            if converter_params is not None:
                self.metadata[CONVERTER_PARAMS_METADATA_KEY] = converter_params

        # Get tuning result if available
        tuning_result = getattr(self, "_tuning_result", None)
        best_params = self.metadata.get("best_params")
        tuning_metadata = None
        if tuning_result is not None:
            tuning_metadata = {
                "study_name": tuning_result.study_name,
                "best_value": tuning_result.best_value,
                "n_trials": tuning_result.n_trials,
                "n_completed_trials": tuning_result.n_completed_trials,
                "n_pruned_trials": tuning_result.n_pruned_trials,
                "n_failed_trials": tuning_result.n_failed_trials,
                "param_importances": tuning_result.param_importances,
                "optimization_metric": self._get_scoring_metric(),
            }
            # The PRE-REGISTERED search's record (D33.2-17): both arms, the outer season,
            # the margin verdict, the adopted arm and the search-space digest. It lands in
            # `{target}_params.json` because that is where models/artifacts.py already
            # writes the tuning record -- NOT in metadata.json, which carries the training
            # metadata and the gold digest.
            if self.adoption_record is not None:
                tuning_metadata.update(self.adoption_record)

        return save_model_artifact(
            model=self.model,
            target=self.target,
            metadata=self.metadata,
            feature_list=self.feature_names,
            calibrator=self.calibrator,
            # D33.1-R1: None for every trainer that has no preprocessing, which is the
            # BaseTrainer class default, so ATS and O/U produce a byte-identical file set.
            preprocessing=getattr(self, "preprocessing", None),
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
