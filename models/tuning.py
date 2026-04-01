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


# ---------------------------------------------------------------------------
# TuningResult dataclass
# ---------------------------------------------------------------------------


@dataclass
class TuningResult:
    """Result of an Optuna optimization study.

    Attributes:
        best_params: Best hyperparameter values found.
        best_value: Best objective function value achieved.
        n_trials: Total number of trials completed in the study.
        param_importances: Parameter importance rankings (fANOVA-based).
            May be empty if insufficient trials for importance computation.
        study_name: Name of the Optuna study.
    """

    best_params: dict[str, Any]
    best_value: float
    n_trials: int
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
    ) -> None:
        self.study_name = study_name
        self.direction = direction
        self.storage_dir = storage_dir
        self.n_trials = n_trials

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
            sampler=optuna.samplers.TPESampler(
                seed=42,
                n_startup_trials=10,
            ),
            pruner=optuna.pruners.HyperbandPruner(
                min_resource=1,
                max_resource=3,
                reduction_factor=3,
            ),
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
                    name: round(value, 4)
                    for name, value in sorted_importances
                },
            )
        except (RuntimeError, ValueError, ZeroDivisionError):
            logger.info(
                "Could not compute parameter importances "
                "(insufficient trials or single-parameter study)",
                study_name=self.study_name,
            )

        return TuningResult(
            best_params=study.best_params,
            best_value=study.best_value,
            n_trials=len(study.trials),
            param_importances=importances,
            study_name=self.study_name,
        )
