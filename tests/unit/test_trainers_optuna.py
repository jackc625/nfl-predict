"""Tests for Optuna integration in WP, ATS, and O/U trainers.

Verifies that each trainer subclass correctly defines its search space
via _define_search_space, uses appropriate CV scoring, and can be
instantiated (not abstract).
"""

from __future__ import annotations

import optuna

from config.tuning_preregistration import SEARCH_SPACE_BY_TARGET
from models.trainers.ats_trainer import ATSTrainer
from models.trainers.ou_trainer import OUTrainer
from models.trainers.wp_trainer import WPTrainer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run_single_trial_and_get_params(
    trainer_cls: type,
    **kwargs: object,
) -> dict:
    """Create a trainer, run one Optuna trial, and return suggested params.

    Uses an in-memory Optuna study to capture the parameters that
    _define_search_space produces via trial.suggest_* calls.
    """
    trainer = trainer_cls(**kwargs)
    study = optuna.create_study(direction="minimize")

    captured_params: dict = {}

    def objective(trial: optuna.Trial) -> float:
        params = trainer._define_search_space(trial)
        captured_params.update(params)
        return 0.0

    study.optimize(objective, n_trials=1)
    return captured_params


# ---------------------------------------------------------------------------
# Test 1: WPTrainer._define_search_space returns expected keys
# ---------------------------------------------------------------------------


class TestWPSearchSpace:
    """Tests for WPTrainer Optuna search space."""

    def test_wp_search_space_has_required_keys(self) -> None:
        """WPTrainer._define_search_space returns dict with keys
        C, penalty, solver, max_iter."""
        params = _run_single_trial_and_get_params(WPTrainer)
        assert "C" in params, "Missing 'C' parameter"
        assert "penalty" in params, "Missing 'penalty' parameter"
        # solver may be keyed as 'solver' or 'solver_l2' depending on penalty
        has_solver = "solver" in params or "solver_l2" in params
        assert has_solver, "Missing solver parameter (neither 'solver' nor 'solver_l2')"
        assert "max_iter" in params, "Missing 'max_iter' parameter"

    def test_wp_penalty_l1_forces_saga(self) -> None:
        """WPTrainer._define_search_space with penalty='l1' always uses solver='saga'."""
        trainer = WPTrainer()

        # Run many trials to ensure we capture at least one l1 penalty
        study = optuna.create_study(direction="minimize")
        l1_solvers: list[str] = []

        def objective(trial: optuna.Trial) -> float:
            params = trainer._define_search_space(trial)
            if params.get("penalty") == "l1":
                l1_solvers.append(params.get("solver", ""))
            return 0.0

        study.optimize(objective, n_trials=50)

        # Ensure we found at least one l1 trial
        assert len(l1_solvers) > 0, "No l1 penalty trials found in 50 trials"
        for solver in l1_solvers:
            assert solver == "saga", (
                f"l1 penalty should use solver='saga', got '{solver}'"
            )

    def test_wp_penalty_l2_solver_stays_inside_the_pre_registered_choices(self) -> None:
        """Every l2 solver suggested is one the PRE-REGISTRATION declares.

        RE-POINTED by Plan 33.2-23 from the literal pair ``{"lbfgs", "saga"}``. D33.2-17
        WIDENS the search space, and the widening is declared ONCE in
        ``config/tuning_preregistration.py`` -- a file locked by a content hash and a
        git-ancestry assertion. A second copy of the choice set here would have to be
        edited by hand every time the declared one moved, which is exactly the drift the
        pre-registration exists to remove. Read against the declaration this assertion is
        STRICTLY STRONGER than the literal: it fails for ANY suggested solver outside the
        declared set, whatever that set says today.
        """
        trainer = WPTrainer()

        study = optuna.create_study(direction="minimize")
        l2_solvers: set[str] = set()

        def objective(trial: optuna.Trial) -> float:
            params = trainer._define_search_space(trial)
            if params.get("penalty") == "l2":
                # solver is stored directly when penalty is l2
                solver = params.get("solver", params.get("solver_l2", ""))
                l2_solvers.add(solver)
            return 0.0

        study.optimize(objective, n_trials=50)

        assert len(l2_solvers) > 0, "No l2 penalty trials found in 50 trials"
        declared = set(SEARCH_SPACE_BY_TARGET["wp"]["solver_l2"].choices)
        assert declared, (
            "the pre-registered l2 solver set is empty; this proves nothing"
        )
        assert l2_solvers <= declared, (
            f"Unexpected solvers for l2: {l2_solvers - declared}; the pre-registration "
            f"declares {sorted(declared)}"
        )


# ---------------------------------------------------------------------------
# Test 4: WPTrainer._compute_cv_score uses log_loss
# ---------------------------------------------------------------------------


class TestWPCVScore:
    """Tests for WPTrainer CV scoring."""

    def test_wp_compute_cv_score_uses_log_loss(self) -> None:
        """WPTrainer._compute_cv_score uses log_loss (not MAE)."""
        import numpy as np
        import pandas as pd
        from sklearn.metrics import log_loss

        trainer = WPTrainer()

        predictions = np.array([0.8, 0.3, 0.6, 0.9, 0.1])
        actuals = pd.Series([1, 0, 1, 1, 0])

        score = trainer._compute_cv_score(predictions, actuals)
        expected = float(log_loss(actuals.values, np.clip(predictions, 1e-7, 1 - 1e-7)))

        assert abs(score - expected) < 1e-6, (
            f"WP CV score should be log_loss ({expected}), got {score}"
        )


# ---------------------------------------------------------------------------
# Test 5: ATSTrainer._define_search_space returns 8 XGBoost params
# ---------------------------------------------------------------------------


class TestATSSearchSpace:
    """Tests for ATSTrainer Optuna search space."""

    def test_ats_search_space_has_8_xgboost_params(self) -> None:
        """ATSTrainer._define_search_space returns dict with all 8 XGBoost
        params: learning_rate, max_depth, n_estimators, subsample,
        colsample_bytree, min_child_weight, reg_alpha, reg_lambda."""
        params = _run_single_trial_and_get_params(ATSTrainer)
        expected_keys = {
            "learning_rate",
            "max_depth",
            "n_estimators",
            "subsample",
            "colsample_bytree",
            "min_child_weight",
            "reg_alpha",
            "reg_lambda",
        }
        for key in expected_keys:
            assert key in params, f"Missing '{key}' in ATS search space"


# ---------------------------------------------------------------------------
# Test 6: OUTrainer._define_search_space returns 8 XGBoost params
# ---------------------------------------------------------------------------


class TestOUSearchSpace:
    """Tests for OUTrainer Optuna search space."""

    def test_ou_search_space_has_8_xgboost_params(self) -> None:
        """OUTrainer._define_search_space returns dict with same 8 XGBoost params."""
        params = _run_single_trial_and_get_params(OUTrainer)
        expected_keys = {
            "learning_rate",
            "max_depth",
            "n_estimators",
            "subsample",
            "colsample_bytree",
            "min_child_weight",
            "reg_alpha",
            "reg_lambda",
        }
        for key in expected_keys:
            assert key in params, f"Missing '{key}' in O/U search space"


# ---------------------------------------------------------------------------
# Test 7: ATSTrainer search space ranges match D-07
# ---------------------------------------------------------------------------


class TestATSSearchSpaceRanges:
    """Tests for ATSTrainer search space ranges per D-07."""

    def test_ats_learning_rate_stays_inside_the_pre_registered_range(self) -> None:
        """Every suggested learning_rate lies inside the PRE-REGISTERED bound.

        RE-POINTED by Plan 33.2-23 from the literal ``[0.005, 0.3]``, which D33.2-17
        deliberately widened. The bound now comes from the one declaration the trainer
        itself reads, so this test can never disagree with what is actually searched --
        and it still fails for any value outside it. The OLD bound is not lost: the
        pre-registration records it beside the new one under ``ParamSpec.current``, and
        the assertion below proves the widening is real rather than asserted.
        """
        trainer = ATSTrainer()
        spec = SEARCH_SPACE_BY_TARGET["ats"]["learning_rate"]

        study = optuna.create_study(direction="minimize")
        learning_rates: list[float] = []

        def objective(trial: optuna.Trial) -> float:
            params = trainer._define_search_space(trial)
            learning_rates.append(params["learning_rate"])
            return 0.0

        study.optimize(objective, n_trials=30)

        assert learning_rates, "no trial ran; the range check proves nothing"
        for lr in learning_rates:
            assert spec.low <= lr <= spec.high, (
                f"learning_rate {lr} outside the pre-registered "
                f"[{spec.low}, {spec.high}]"
            )
        assert spec.current == "0.005 - 0.3 (log)", (
            "the pre-registration no longer records the bound this test used to pin, so "
            "the widening can no longer be read off the file"
        )
        assert spec.low < 0.005 and spec.high > 0.3, (
            "the range did not actually widen; D33.2-17 requires wider ranges, and a "
            "re-pointed test must not quietly accept an unchanged one"
        )

    def test_ats_max_depth_stays_inside_the_pre_registered_range(self) -> None:
        """Every suggested max_depth lies inside the PRE-REGISTERED bound.

        RE-POINTED from the literal ``[2, 8]`` for the same reason as the sibling above,
        with the same two controls: the prior bound is still readable, and the new one is
        genuinely wider.
        """
        trainer = ATSTrainer()
        spec = SEARCH_SPACE_BY_TARGET["ats"]["max_depth"]

        study = optuna.create_study(direction="minimize")
        depths: list[int] = []

        def objective(trial: optuna.Trial) -> float:
            params = trainer._define_search_space(trial)
            depths.append(params["max_depth"])
            return 0.0

        study.optimize(objective, n_trials=30)

        assert depths, "no trial ran; the range check proves nothing"
        for d in depths:
            assert spec.low <= d <= spec.high, (
                f"max_depth {d} outside the pre-registered [{spec.low}, {spec.high}]"
            )
        assert spec.current == "2 - 8"
        assert spec.low < 2 and spec.high > 8, (
            "the range did not actually widen; D33.2-17 requires wider ranges"
        )


# ---------------------------------------------------------------------------
# Test 8: All trainers are instantiable (not abstract)
# ---------------------------------------------------------------------------


class TestTrainersInstantiable:
    """Tests that all trainers can be instantiated."""

    def test_wp_trainer_not_abstract(self) -> None:
        """WPTrainer can be instantiated (all abstract methods implemented)."""
        trainer = WPTrainer()
        assert trainer.target == "wp"

    def test_ats_trainer_not_abstract(self) -> None:
        """ATSTrainer can be instantiated."""
        trainer = ATSTrainer()
        assert trainer.target == "ats"

    def test_ou_trainer_not_abstract(self) -> None:
        """OUTrainer can be instantiated."""
        trainer = OUTrainer()
        assert trainer.target == "ou"
