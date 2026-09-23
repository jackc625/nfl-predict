"""Optuna-backed hyperparameter tuning infrastructure.

Provides:
- TuningResult: Dataclass holding optimization results
- OptunaTuner: SQLite-backed Optuna study runner with TPE sampler,
  Hyperband pruner, and parameter importance logging

Usage:
    tuner = OptunaTuner(study_name="wp_tuning", n_trials=100)
    result = tuner.optimize(objective_fn)
    print(result.best_params, result.best_value)

All studies are persisted to SQLite for resumability across sessions.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import optuna

from utils import get_logger

# Suppress Optuna's verbose default logging (per plan requirement)
optuna.logging.set_verbosity(optuna.logging.WARNING)

logger = get_logger(__name__)


def default_tpe_sampler() -> optuna.samplers.BaseSampler:
    """The sampler every caller that does not ask for another one gets.

    IT LIVES HERE AND NOT INSIDE ``optimize`` (Plan 33.2-23). While ``optimize``
    constructed a ``TPESampler`` in its own body, an explicitly-passed ``RandomSampler``
    had nowhere to go: the "random baseline" the noise guard rests on would have been a
    second TPE run wearing a different name. Building the default at CONSTRUCTION time
    keeps every legacy caller byte-identical -- same class, same seed, same
    ``n_startup_trials`` -- while making the sampler a thing a caller can replace.
    """
    return optuna.samplers.TPESampler(seed=42, n_startup_trials=10)


def default_hyperband_pruner() -> optuna.pruners.BasePruner:
    """The pruner every caller that does not ask for another one gets.

    The same configuration every study in this repository has always run under. Moved out
    of ``optimize`` for the same reason as the sampler, and so that BOTH arms of the
    Phase-33.2 search can be handed ONE pre-registered pruner configuration
    (``config.tuning_preregistration.PRUNER_CONFIG``) rather than each constructing its
    own.
    """
    return optuna.pruners.HyperbandPruner(
        min_resource=1,
        max_resource=3,
        reduction_factor=3,
    )


def count_trials_by_state(study: optuna.Study) -> dict[str, int]:
    """Return how many of *study*'s trials are in each terminal state.

    The three counts a tuning record must PUBLISH, taken from one read of the study so
    they can never disagree about which state they describe: ``complete``, ``pruned`` and
    ``failed``. Under ``HyperbandPruner`` the pruned count is normally the largest of the
    three by design, which is exactly why a completed-trial FLOOR is not a usable pass bar
    and these are reported facts rather than a gate.

    Args:
        study: The finished (or resumed) Optuna study.

    Returns:
        ``{"complete": int, "pruned": int, "failed": int}``.
    """
    states = {
        "complete": optuna.trial.TrialState.COMPLETE,
        "pruned": optuna.trial.TrialState.PRUNED,
        "failed": optuna.trial.TrialState.FAIL,
    }
    return {
        name: sum(1 for trial in study.trials if trial.state == state)
        for name, state in states.items()
    }


def count_completed_trials(study: optuna.Study) -> int:
    """Return how many of *study*'s trials reached the ``COMPLETE`` state.

    WR-08: ``len(study.trials)`` counts every trial STARTED, including ``PRUNED`` and
    ``FAIL``. The searches in this project run under
    ``HyperbandPruner(min_resource=1, max_resource=3, reduction_factor=3)``, whose whole
    purpose is to prune the majority of them -- so a "trials" figure taken from the raw
    length is a budget, not an amount of work done, and anything labelled "completed
    trials" from it is overstating.

    Kept here beside ``OptunaTuner`` so both the tuner and ``BaseTrainer`` ask the same
    question of a study rather than each re-deriving the state filter.
    """
    return sum(
        1 for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE
    )


# ---------------------------------------------------------------------------
# TuningResult dataclass
# ---------------------------------------------------------------------------


@dataclass
class TuningResult:
    """Result of an Optuna optimization study.

    Attributes:
        best_params: Best hyperparameter values found.
        best_value: Best objective function value achieved.
        n_trials: Total number of trials STARTED in the study -- ``len(study.trials)``,
            which includes ``PRUNED`` and ``FAIL`` states. WR-08: this is a budget
            figure, not a work figure. The search runs under
            ``HyperbandPruner(min_resource=1, max_resource=3, reduction_factor=3)``,
            which prunes the majority of trials, so this materially overstates how many
            searches ran to completion. Use ``n_completed_trials`` for that.
        n_completed_trials: Trials in the ``COMPLETE`` state -- the ones that actually
            produced an objective value. Reported alongside rather than instead of
            ``n_trials``: the anti-vacuity check ("did this resumed study add anything
            at all?") is correctly a question about STARTED trials.
        param_importances: Parameter importance rankings (fANOVA-based).
            May be empty if insufficient trials for importance computation.
        study_name: Name of the Optuna study.
    """

    best_params: dict[str, Any]
    best_value: float
    n_trials: int
    n_completed_trials: int = 0
    # Plan 33.2-23: the other two terminal states, so a tuning record can PUBLISH what the
    # search actually did rather than leaving "1,000 started, 40 completed" unexplained.
    # Under HyperbandPruner the pruned count is normally the largest of the three.
    n_pruned_trials: int = 0
    n_failed_trials: int = 0
    param_importances: dict[str, float] = field(default_factory=dict)
    study_name: str = ""


# ---------------------------------------------------------------------------
# OptunaTuner
# ---------------------------------------------------------------------------


class OptunaTuner:
    """Optuna-backed hyperparameter tuner with SQLite persistence.

    Creates and manages Optuna studies with:
    - TPE sampler (seed=42, n_startup_trials=10) for Bayesian optimization
    - Hyperband pruner for early stopping of poor trials
    - SQLite storage for resumability across sessions
    - Parameter importance logging after study completion

    Args:
        study_name: Name for the Optuna study (used in storage filename).
        direction: Optimization direction ("minimize" or "maximize").
        storage_dir: Directory for SQLite study databases.
        n_trials: Total number of trials to run.
    """

    def __init__(
        self,
        study_name: str,
        direction: str = "minimize",
        storage_dir: Path = Path("data/optuna"),
        n_trials: int = 100,
        sampler: optuna.samplers.BaseSampler | None = None,
        pruner: optuna.pruners.BasePruner | None = None,
    ) -> None:
        self.study_name = study_name
        self.direction = direction
        self.storage_dir = storage_dir
        self.n_trials = n_trials

        # The sampler and the pruner are resolved HERE rather than inside optimize (Plan
        # 33.2-23). Omitting either reproduces the previous behaviour exactly -- TPE seed
        # 42 with 10 startup trials, Hyperband(1, 3, 3) -- so every legacy caller is
        # byte-identical; passing one is what makes a RandomSampler baseline possible at
        # all. See default_tpe_sampler's own note.
        self.sampler = sampler if sampler is not None else default_tpe_sampler()
        self.pruner = pruner if pruner is not None else default_hyperband_pruner()

        # Ensure storage directory exists
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    @property
    def storage_url(self) -> str:
        """Return the SQLite storage URL for this study.

        Returns:
            URL string in the format sqlite:///{storage_dir}/{study_name}.db
        """
        db_path = self.storage_dir / f"{self.study_name}.db"
        return f"sqlite:///{db_path}"

    def optimize(
        self,
        objective_fn: Callable[[optuna.Trial], float],
    ) -> TuningResult:
        """Run optimization and return results.

        Creates or resumes an Optuna study with TPE sampler and Hyperband
        pruner. If a study with the same name already exists in storage,
        it resumes from where it left off and only runs remaining trials.

        Args:
            objective_fn: Callable that takes an optuna.Trial and returns
                the objective value to optimize.

        Returns:
            TuningResult with best parameters, value, trial count,
            and parameter importances.
        """
        study = optuna.create_study(
            study_name=self.study_name,
            storage=self.storage_url,
            direction=self.direction,
            sampler=self.sampler,
            pruner=self.pruner,
            load_if_exists=True,
        )

        # Only run remaining trials if study is being resumed
        remaining = max(0, self.n_trials - len(study.trials))
        if remaining > 0:
            study.optimize(objective_fn, n_trials=remaining)

        logger.info(
            "Optimization completed",
            study_name=self.study_name,
            best_params=study.best_params,
            best_value=study.best_value,
            n_trials=len(study.trials),
        )

        # Attempt parameter importance computation
        importances: dict[str, float] = {}
        try:
            importances = optuna.importance.get_param_importances(study)
            # Log importance rankings sorted descending
            sorted_importances = sorted(
                importances.items(), key=lambda x: x[1], reverse=True
            )
            logger.info(
                "Parameter importance rankings",
                study_name=self.study_name,
                importances={
                    name: round(value, 4) for name, value in sorted_importances
                },
            )
        except (RuntimeError, ValueError, ZeroDivisionError):
            logger.info(
                "Could not compute parameter importances "
                "(insufficient trials or single-parameter study)",
                study_name=self.study_name,
            )

        by_state = count_trials_by_state(study)
        return TuningResult(
            best_params=study.best_params,
            best_value=study.best_value,
            n_trials=len(study.trials),
            n_completed_trials=by_state["complete"],
            n_pruned_trials=by_state["pruned"],
            n_failed_trials=by_state["failed"],
            param_importances=importances,
            study_name=self.study_name,
        )
