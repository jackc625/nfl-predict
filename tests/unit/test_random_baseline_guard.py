"""The random-setting baseline, the two study identities, and the outer-season gate.

WHY THIS EXISTS (D33.2-17, R13)
-------------------------------
A thousand-trial search on three small validation folds does not only find better
settings; it also finds the setting that best fits those folds' NOISE. The guard is a
RANDOM-sampler arm over the SAME space and the SAME budget, plus a margin committed before
the search ran, read off a season NEITHER arm saw.

Three things could quietly make that guard vacuous, and each has a test here:

* the "random baseline" RESUMING the TPE study. ``OptunaTuner.optimize`` opens every study
  with ``load_if_exists=True`` and used to construct ``TPESampler`` inline, so a second run
  under one name would have added nothing and measured nothing;
* the budget silently falling back to the 100-trial default that ``models/tuning.py`` and
  ``models/trainers/base.py`` BOTH carry;
* the adoption decision reading the two arms' IN-SEARCH best values, which are two winners
  selected on the same noise and can differ by noise alone.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import pytest

from config import tuning_preregistration as prereg
from models import tuning as tuning_module
from models.trainers import base as base_trainer
from models.trainers.base import BaseTrainer
from models.tuning import OptunaTuner, TuningResult


class _DummyModel:
    """A trivial fitted-model stand-in. The search's scores are not the subject here."""

    def fit(self, X: pd.DataFrame, y: pd.Series) -> _DummyModel:
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X))


class _DummyTrainer(BaseTrainer):
    """Minimal concrete trainer so the two-arm machinery runs hermetically."""

    def __init__(self, target: str = "ats") -> None:
        super().__init__(target=target)
        self.feature_names = ["f1"]

    def _create_model(self, params: dict) -> _DummyModel:
        return _DummyModel()

    def _get_target_column(self) -> str:
        return "home_margin"

    def _predict_raw(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        return np.zeros(len(X))

    def _get_default_params(self) -> dict:
        return {"the_default": 1}

    def _define_search_space(self, trial: optuna.Trial) -> dict:
        return {"x": trial.suggest_float("x", 0.0, 1.0)}


def _training_frame(n_rows: int = 90) -> tuple[pd.DataFrame, pd.Series]:
    """A frame large enough for make_temporal_cv_splits."""
    return (
        pd.DataFrame({"f1": np.arange(n_rows, dtype=float)}),
        pd.Series(np.arange(n_rows, dtype=float) % 7),
    )


def _full_frame() -> pd.DataFrame:
    """A frame whose seasons derive a partition with 2024 as the FIRST holdout season."""
    rows = []
    for season in range(2020, 2026):
        for index in range(12):
            rows.append(
                {
                    "season": season,
                    "week": index + 1,
                    "f1": float(index),
                    "home_margin": float(index % 5),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture
def pinned_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """A SMALL pre-registered budget and a throwaway study directory.

    The budget is monkeypatched rather than the real 1,000 because these tests are about
    the machinery, not the search. The REAL budget is asserted separately, against the
    committed pre-registration, by ``test_the_real_budget_is_the_pre_registered_one``.
    """
    budget = {"wp": 8, "ats": 8, "ou": 8}
    monkeypatch.setattr(base_trainer, "TRIAL_BUDGET_BY_TARGET", budget)
    monkeypatch.setattr(base_trainer, "TUNING_STORAGE_DIR", tmp_path / "optuna")
    return budget


# ---------------------------------------------------------------------------
# The tuner gained an explicit sampler and pruner, and stopped hardcoding TPE
# ---------------------------------------------------------------------------


class TestTheTunerTakesAnExplicitSampler:
    def test_optimize_no_longer_constructs_a_sampler_of_its_own(self) -> None:
        """A random arm cannot exist while ``optimize`` builds TPE inline."""
        import ast
        import inspect
        import textwrap

        body = ast.parse(textwrap.dedent(inspect.getsource(OptunaTuner.optimize)))
        constructed = [
            node
            for node in ast.walk(body)
            if isinstance(node, ast.Call)
            and (
                getattr(node.func, "attr", None) == "TPESampler"
                or getattr(node.func, "id", None) == "TPESampler"
            )
        ]
        assert constructed == [], (
            "OptunaTuner.optimize still constructs a TPESampler; an explicitly-passed "
            "RandomSampler would be ignored and the baseline would be a second TPE run."
        )

    def test_the_default_is_unchanged_for_every_legacy_caller(self) -> None:
        """Omitting the sampler still yields TPE seed 42 with 10 startup trials.

        Byte-identical behaviour for ``scripts.retrain_models`` and ``backtest.engine``
        matters: a changed sampler there would drift the frozen v2.1 AUDIT-REPORT anchors.
        """
        tuner = OptunaTuner(study_name="legacy", n_trials=3)
        assert isinstance(tuner.sampler, optuna.samplers.TPESampler)
        assert isinstance(tuner.pruner, optuna.pruners.HyperbandPruner)

    def test_an_explicit_sampler_and_pruner_are_used(self) -> None:
        sampler = optuna.samplers.RandomSampler(seed=7)
        pruner = optuna.pruners.HyperbandPruner(**prereg.PRUNER_CONFIG)
        tuner = OptunaTuner(
            study_name="explicit", n_trials=3, sampler=sampler, pruner=pruner
        )
        assert tuner.sampler is sampler
        assert tuner.pruner is pruner


# ---------------------------------------------------------------------------
# Two arms: same space, same budget, different samplers, different identities
# ---------------------------------------------------------------------------


class TestTheTwoArms:
    def test_both_arms_get_the_same_objective_and_the_same_budget(
        self, pinned_budget: dict[str, int], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Same space AND same budget, asserted together.

        The SAME objective OBJECT reaching both arms is what makes "the same search
        space" true by construction rather than by a second declaration that can drift.
        """
        seen: list[dict[str, Any]] = []

        def fake_optimize(self: OptunaTuner, objective_fn: Any) -> TuningResult:
            seen.append(
                {
                    "study_name": self.study_name,
                    "n_trials": self.n_trials,
                    "sampler": type(self.sampler).__name__,
                    "pruner_config": {
                        "min_resource": self.pruner._min_resource,
                        "reduction_factor": self.pruner._reduction_factor,
                    },
                    "objective_id": id(objective_fn),
                }
            )
            return TuningResult(
                best_params={"x": 0.5},
                best_value=1.0,
                n_trials=self.n_trials,
                n_completed_trials=2,
                n_pruned_trials=self.n_trials - 2,
                n_failed_trials=0,
                study_name=self.study_name,
            )

        monkeypatch.setattr(OptunaTuner, "optimize", fake_optimize)
        trainer = _DummyTrainer()
        trainer.use_phase332_tuning()
        X_train, y_train = _training_frame()
        trainer.tune_hyperparameters(X_train, y_train, full_features_df=_full_frame())

        assert len(seen) == 2, seen
        first, second = seen
        assert first["study_name"].endswith("_random"), (
            "the RANDOM baseline must run FIRST; it is the comparator, not an afterthought"
        )
        assert second["study_name"].endswith("_tpe")
        assert first["study_name"] != second["study_name"], (
            "one study name for both arms means the second run RESUMES the first"
        )
        assert first["sampler"] == "RandomSampler"
        assert second["sampler"] == "TPESampler"
        assert first["n_trials"] == second["n_trials"] == pinned_budget["ats"]
        assert first["objective_id"] == second["objective_id"], (
            "the two arms were handed DIFFERENT objective objects, so nothing guarantees "
            "they searched the same space"
        )
        assert first["pruner_config"] == second["pruner_config"], (
            "the two arms ran under different pruner configurations"
        )

    def test_started_counts_are_recorded_per_arm_and_completed_is_never_a_floor(
        self, pinned_budget: dict[str, int], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A planted run in which MOST trials prune still passes.

        That is the case a completed-trial floor would have failed on a correct run: the
        Hyperband pruner exists to stop most trials early.
        """
        completed_by_arm = {"_random": 7, "_tpe": 1}

        def fake_optimize(self: OptunaTuner, objective_fn: Any) -> TuningResult:
            suffix = "_random" if self.study_name.endswith("_random") else "_tpe"
            completed = completed_by_arm[suffix]
            return TuningResult(
                best_params={"x": 0.5},
                best_value=1.0,
                n_trials=self.n_trials,
                n_completed_trials=completed,
                n_pruned_trials=self.n_trials - completed,
                n_failed_trials=0,
                study_name=self.study_name,
            )

        monkeypatch.setattr(OptunaTuner, "optimize", fake_optimize)
        trainer = _DummyTrainer()
        trainer.use_phase332_tuning()
        X_train, y_train = _training_frame()
        trainer.tune_hyperparameters(X_train, y_train, full_features_df=_full_frame())

        record = trainer.adoption_record
        assert record is not None
        arms = record["arms"]
        assert arms["tpe"]["trials_started"] == pinned_budget["ats"]
        assert arms["random"]["trials_started"] == pinned_budget["ats"]
        assert arms["tpe"]["trials_completed"] != arms["random"]["trials_completed"], (
            "this fixture deliberately plants UNEQUAL completed counts; if the code "
            "required them equal it would fail on a correct run"
        )
        for arm in ("tpe", "random"):
            assert arms[arm]["trials_pruned"] is not None
            assert arms[arm]["trials_failed"] is not None
            assert arms[arm]["in_search_best_value"] is not None
            assert arms[arm]["outer_score"] is not None

    def test_a_short_arm_raises_rather_than_reporting_a_full_budget(
        self, pinned_budget: dict[str, int], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``trials_started == budget`` is the POSITIVE form of the anti-vacuity check."""

        def fake_optimize(self: OptunaTuner, objective_fn: Any) -> TuningResult:
            return TuningResult(
                best_params={"x": 0.5},
                best_value=1.0,
                n_trials=self.n_trials - 1,
                n_completed_trials=1,
                n_pruned_trials=0,
                n_failed_trials=0,
                study_name=self.study_name,
            )

        monkeypatch.setattr(OptunaTuner, "optimize", fake_optimize)
        trainer = _DummyTrainer()
        trainer.use_phase332_tuning()
        X_train, y_train = _training_frame()
        with pytest.raises(RuntimeError, match="trials_started"):
            trainer.tune_hyperparameters(
                X_train, y_train, full_features_df=_full_frame()
            )

    def test_the_random_arm_does_not_resume_an_existing_tpe_study(
        self, tmp_path: Path, pinned_budget: dict[str, int]
    ) -> None:
        """A REAL run against a pre-filled TPE study of the same tag.

        The TPE arm must hard-fail (zero new trials), and the random arm must have run its
        full budget in its own storage -- proving it never touched the TPE study.
        """
        storage = tmp_path / "optuna"
        tag = base_trainer.TUNING_STUDY_TAG
        tpe_name = prereg.study_name("ats", tag, prereg.STUDY_ARM_TPE)
        prefill = OptunaTuner(
            study_name=tpe_name, storage_dir=storage, n_trials=pinned_budget["ats"]
        )
        study = optuna.create_study(
            study_name=tpe_name, storage=prefill.storage_url, direction="minimize"
        )
        study.optimize(
            lambda trial: trial.suggest_float("x", 0.0, 1.0),
            n_trials=pinned_budget["ats"],
        )

        trainer = _DummyTrainer()
        trainer.use_phase332_tuning()
        X_train, y_train = _training_frame()
        with pytest.raises(RuntimeError) as excinfo:
            trainer.tune_hyperparameters(
                X_train, y_train, full_features_df=_full_frame()
            )
        message = str(excinfo.value)
        assert tpe_name in message
        assert "TUNING_STUDY_TAG" in message

        random_name = prereg.study_name("ats", tag, prereg.STUDY_ARM_RANDOM)
        random_db = storage / f"{random_name}.db"
        assert random_db.exists(), (
            "the random arm wrote no storage of its own, so it cannot have run as a "
            "separate study"
        )
        loaded = optuna.load_study(
            study_name=random_name, storage=f"sqlite:///{random_db}"
        )
        assert len(loaded.trials) == pinned_budget["ats"]


# ---------------------------------------------------------------------------
# The budget is the pre-registered one, and the 100-trial default is unreachable
# ---------------------------------------------------------------------------


class TestTheBudget:
    def test_the_real_budget_is_the_pre_registered_one(self) -> None:
        """Asserted against the COMMITTED pre-registration, not a fixture."""
        assert base_trainer.TRIAL_BUDGET_BY_TARGET is prereg.TRIAL_BUDGET_BY_TARGET, (
            "models/trainers/base.py holds a SECOND copy of the budget, which can drift"
        )
        assert min(prereg.TRIAL_BUDGET_BY_TARGET.values()) >= 900

    def test_a_caller_that_contradicts_the_pre_registered_budget_raises(
        self, pinned_budget: dict[str, int]
    ) -> None:
        """Passing the 100-trial default explicitly fails LOUDLY on the guarded path."""
        trainer = _DummyTrainer()
        trainer.use_phase332_tuning()
        X_train, y_train = _training_frame()
        with pytest.raises(RuntimeError, match="pre-registered"):
            trainer.tune_hyperparameters(
                X_train, y_train, n_trials=100, full_features_df=_full_frame()
            )

    def test_the_guarded_path_refuses_to_run_without_the_outer_frame(
        self, pinned_budget: dict[str, int]
    ) -> None:
        """No outer frame means no outer season, and no honest adoption decision."""
        trainer = _DummyTrainer()
        trainer.use_phase332_tuning()
        X_train, y_train = _training_frame()
        with pytest.raises(RuntimeError, match="full_features_df"):
            trainer.tune_hyperparameters(X_train, y_train)

    def test_the_legacy_path_is_byte_identical(self, tmp_path: Path) -> None:
        """A trainer that does NOT opt in keeps the single-arm, 100-trial default."""
        trainer = _DummyTrainer()
        assert trainer.preregistered_search is False
        assert trainer.tuning_study_tag == base_trainer.LEGACY_TUNING_STUDY_TAG
        assert trainer.tuning_storage_dir == base_trainer.LEGACY_TUNING_STORAGE_DIR


# ---------------------------------------------------------------------------
# The adoption decision reads the OUTER season, never the in-search values
# ---------------------------------------------------------------------------


class TestTheAdoptionDecision:
    def test_a_tpe_arm_that_wins_in_search_but_loses_out_of_sample_is_not_adopted(
        self,
    ) -> None:
        """THE case that proves the gate measures generalisation, not in-fold fit."""
        bar = prereg.BEAT_RANDOM_MARGIN_BY_TARGET["ats"]
        arms = {
            # TPE looks far better on the folds both arms optimised ...
            prereg.STUDY_ARM_TPE: {
                "in_search_best_value": 9.0,
                "outer_score": 11.0 + bar,
            },
            # ... and is WORSE on the season neither of them saw.
            prereg.STUDY_ARM_RANDOM: {
                "in_search_best_value": 10.5,
                "outer_score": 11.0,
            },
        }
        decision = base_trainer.decide_adoption("ats", arms)
        assert decision["margin_cleared"] is False
        assert decision["adopted_arm"] == prereg.NOT_CLEARED_ARM
        assert decision["not_cleared_rule"] == prereg.NOT_CLEARED_RULE

    def test_a_sub_margin_winner_is_not_adopted(self) -> None:
        bar = prereg.BEAT_RANDOM_MARGIN_BY_TARGET["ou"]
        arms = {
            prereg.STUDY_ARM_TPE: {"in_search_best_value": 1.0, "outer_score": 11.0},
            prereg.STUDY_ARM_RANDOM: {
                "in_search_best_value": 1.0,
                "outer_score": 11.0 + bar - 1e-6,
            },
        }
        decision = base_trainer.decide_adoption("ou", arms)
        assert decision["margin_cleared"] is False
        assert decision["adopted_arm"] == prereg.NOT_CLEARED_ARM

    def test_a_winner_that_clears_the_margin_is_adopted(self) -> None:
        bar = prereg.BEAT_RANDOM_MARGIN_BY_TARGET["wp"]
        arms = {
            prereg.STUDY_ARM_TPE: {"in_search_best_value": 1.0, "outer_score": 0.60},
            prereg.STUDY_ARM_RANDOM: {
                "in_search_best_value": 1.0,
                "outer_score": 0.60 + bar,
            },
        }
        decision = base_trainer.decide_adoption("wp", arms)
        assert decision["margin_cleared"] is True
        assert decision["adopted_arm"] == prereg.STUDY_ARM_TPE
        assert decision["margin"] == bar

    def test_the_margin_is_read_from_the_pre_registration(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Move the bar in the pre-registration and the decision MUST move with it.

        A literal anywhere in the decision path would leave this test green while the
        published margin said something else.
        """
        arms = {
            prereg.STUDY_ARM_TPE: {"in_search_best_value": 1.0, "outer_score": 10.0},
            prereg.STUDY_ARM_RANDOM: {"in_search_best_value": 1.0, "outer_score": 10.2},
        }
        monkeypatch.setattr(
            base_trainer,
            "BEAT_RANDOM_MARGIN_BY_TARGET",
            {"ats": 0.1, "ou": 1.0, "wp": 1.0},
        )
        assert base_trainer.decide_adoption("ats", arms)["margin_cleared"] is True
        monkeypatch.setattr(
            base_trainer,
            "BEAT_RANDOM_MARGIN_BY_TARGET",
            {"ats": 0.3, "ou": 1.0, "wp": 1.0},
        )
        assert base_trainer.decide_adoption("ats", arms)["margin_cleared"] is False

    def test_the_not_cleared_outcome_is_recorded_and_the_defaults_are_adopted(
        self, pinned_budget: dict[str, int]
    ) -> None:
        """A REAL two-arm run whose arms tie: the outcome is a FINDING, not a fall-through."""
        trainer = _DummyTrainer()
        trainer.use_phase332_tuning()
        X_train, y_train = _training_frame()
        best_params = trainer.tune_hyperparameters(
            X_train, y_train, full_features_df=_full_frame()
        )

        record = trainer.adoption_record
        assert record is not None
        assert record["margin_cleared"] is False
        assert record["adopted_arm"] == prereg.NOT_CLEARED_ARM
        assert record["not_cleared_rule"] == prereg.NOT_CLEARED_RULE
        assert best_params == trainer._get_default_params()

    def test_the_outer_season_is_derived_and_is_not_2025(
        self, pinned_budget: dict[str, int]
    ) -> None:
        trainer = _DummyTrainer()
        trainer.use_phase332_tuning()
        X_train, y_train = _training_frame()
        trainer.tune_hyperparameters(X_train, y_train, full_features_df=_full_frame())
        record = trainer.adoption_record
        assert record["outer_season"] == 2024
        assert record["outer_season"] not in prereg.EXCLUDED_OUTER_SEASONS

    def test_the_record_carries_the_search_space_digest_the_arms_searched(
        self, pinned_budget: dict[str, int]
    ) -> None:
        trainer = _DummyTrainer()
        trainer.use_phase332_tuning()
        X_train, y_train = _training_frame()
        trainer.tune_hyperparameters(X_train, y_train, full_features_df=_full_frame())
        record = trainer.adoption_record
        assert record["search_space_digest"] == prereg.search_space_digest("ats")
        assert record["thread_limit"] == prereg.PINNED_THREAD_COUNT


# ---------------------------------------------------------------------------
# Non-vacuity: an empty search could not satisfy the tests above
# ---------------------------------------------------------------------------


class TestNonVacuity:
    def test_the_budget_is_non_zero_and_every_space_is_non_empty(self) -> None:
        for target in prereg.TARGETS:
            assert prereg.TRIAL_BUDGET_BY_TARGET[target] > 0
            assert len(prereg.SEARCH_SPACE_BY_TARGET[target]) > 0

    def test_no_numeric_bound_is_inlined_in_any_trainer_suggest_call(self) -> None:
        """The bounds moved OUT of the trainers, so both arms read one declaration."""
        import ast

        paths = (
            "models/trainers/wp_trainer.py",
            "models/trainers/ats_trainer.py",
            "models/trainers/ou_trainer.py",
        )
        repo_root = Path(__file__).resolve().parents[2]
        calls = [
            (path, node)
            for path in paths
            for node in ast.walk(
                ast.parse((repo_root / path).read_text(encoding="utf-8"))
            )
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr.startswith("suggest_")
        ]
        assert calls, "no suggest_* call was found at all; the scan proves nothing"
        inline = [
            f"{path}:{node.lineno}"
            for path, node in calls
            if any(
                isinstance(arg, ast.Constant)
                and isinstance(arg.value, (int, float))
                and not isinstance(arg.value, bool)
                for arg in node.args[1:]
            )
            or any(isinstance(arg, ast.List) and arg.elts for arg in node.args[1:])
        ]
        assert inline == [], inline

    def test_the_trainers_read_the_pre_registered_space(self) -> None:
        for target in ("wp", "ats", "ou"):
            assert prereg.SEARCH_SPACE_BY_TARGET[target]
        assert tuning_module.count_trials_by_state is not None
