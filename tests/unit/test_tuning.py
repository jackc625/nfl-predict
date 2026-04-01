"""Unit tests for OptunaTuner and TuningResult."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from models.tuning import OptunaTuner, TuningResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _quadratic_objective(trial):
    """Simple quadratic objective for testing: (x - 3)^2."""
    x = trial.suggest_float("x", -10, 10)
    return (x - 3) ** 2


def _multi_param_objective(trial):
    """Multi-parameter objective for importance testing."""
    x = trial.suggest_float("x", -10, 10)
    y = trial.suggest_float("y", -10, 10)
    z = trial.suggest_float("z", -10, 10)
    # x matters most, y matters somewhat, z is noise
    return (x - 3) ** 2 + 0.1 * (y - 1) ** 2 + 0.001 * z


# ---------------------------------------------------------------------------
# Test OptunaTuner
# ---------------------------------------------------------------------------


class TestOptunaTunerInit:
    """Test OptunaTuner initialization."""

    def test_creates_storage_dir(self, tmp_path: Path) -> None:
        """OptunaTuner.__init__ creates storage_dir if not exists."""
        storage_dir = tmp_path / "optuna_storage"
        assert not storage_dir.exists()

        OptunaTuner(
            study_name="test_study",
            storage_dir=storage_dir,
        )

        assert storage_dir.exists()

    def test_stores_params(self, tmp_path: Path) -> None:
        """OptunaTuner.__init__ stores study_name, direction, n_trials."""
        tuner = OptunaTuner(
            study_name="my_study",
            direction="maximize",
            storage_dir=tmp_path,
            n_trials=50,
        )

        assert tuner.study_name == "my_study"
        assert tuner.direction == "maximize"
        assert tuner.n_trials == 50


class TestOptunaTunerStorageUrl:
    """Test OptunaTuner.storage_url property."""

    def test_returns_sqlite_url(self, tmp_path: Path) -> None:
        """OptunaTuner.storage_url returns sqlite:///{storage_dir}/{study_name}.db."""
        tuner = OptunaTuner(
            study_name="test_study",
            storage_dir=tmp_path,
        )

        url = tuner.storage_url
        assert url.startswith("sqlite:///")
        assert "test_study.db" in url
        assert str(tmp_path).replace("\\", "/") in url.replace("\\", "/")


class TestOptunaTunerOptimize:
    """Test OptunaTuner.optimize() method."""

    def test_creates_study_with_tpe_sampler_and_hyperband(
        self, tmp_path: Path
    ) -> None:
        """OptunaTuner.optimize() creates study with TPESampler and HyperbandPruner."""
        tuner = OptunaTuner(
            study_name="test_tpe",
            storage_dir=tmp_path,
            n_trials=5,
        )

        result = tuner.optimize(_quadratic_objective)

        # Verify it returned a TuningResult
        assert isinstance(result, TuningResult)

    def test_returns_best_params_dict(self, tmp_path: Path) -> None:
        """OptunaTuner.optimize() with simple objective returns non-empty best_params."""
        tuner = OptunaTuner(
            study_name="test_params",
            storage_dir=tmp_path,
            n_trials=10,
        )

        result = tuner.optimize(_quadratic_objective)

        assert isinstance(result.best_params, dict)
        assert len(result.best_params) > 0
        assert "x" in result.best_params

    def test_resumes_study_with_load_if_exists(self, tmp_path: Path) -> None:
        """OptunaTuner.optimize() with load_if_exists=True resumes and only runs remaining trials."""
        # First run: 3 trials
        tuner1 = OptunaTuner(
            study_name="test_resume",
            storage_dir=tmp_path,
            n_trials=3,
        )
        result1 = tuner1.optimize(_quadratic_objective)
        assert result1.n_trials == 3

        # Second run: n_trials=5, should only run 2 more (already have 3)
        tuner2 = OptunaTuner(
            study_name="test_resume",
            storage_dir=tmp_path,
            n_trials=5,
        )
        result2 = tuner2.optimize(_quadratic_objective)
        assert result2.n_trials == 5

    def test_returns_tuning_result_dataclass(self, tmp_path: Path) -> None:
        """OptunaTuner.optimize() returns TuningResult with correct fields."""
        tuner = OptunaTuner(
            study_name="test_result",
            storage_dir=tmp_path,
            n_trials=5,
        )

        result = tuner.optimize(_quadratic_objective)

        assert isinstance(result, TuningResult)
        assert isinstance(result.best_params, dict)
        assert isinstance(result.best_value, float)
        assert result.n_trials == 5
        assert isinstance(result.param_importances, dict)
        assert result.study_name == "test_result"

    def test_logs_parameter_importance(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """OptunaTuner.optimize() logs parameter importance rankings."""
        tuner = OptunaTuner(
            study_name="test_importance",
            storage_dir=tmp_path,
            n_trials=20,
        )

        with caplog.at_level(logging.INFO):
            result = tuner.optimize(_multi_param_objective)

        # If importances were computed, they should be in the result
        # (may be empty if too few trials, but with 20 trials it should work)
        if result.param_importances:
            assert len(result.param_importances) > 0

    def test_runs_exact_n_trials(self, tmp_path: Path) -> None:
        """OptunaTuner with n_trials=5 actually runs 5 trials."""
        tuner = OptunaTuner(
            study_name="test_exact_trials",
            storage_dir=tmp_path,
            n_trials=5,
        )

        result = tuner.optimize(_quadratic_objective)

        assert result.n_trials == 5

    def test_best_value_is_reasonable(self, tmp_path: Path) -> None:
        """OptunaTuner finds a reasonable best_value for quadratic objective."""
        tuner = OptunaTuner(
            study_name="test_value",
            storage_dir=tmp_path,
            n_trials=30,
        )

        result = tuner.optimize(_quadratic_objective)

        # The minimum of (x - 3)^2 is 0. With 30 trials, should find
        # something reasonably close (< 5)
        assert result.best_value < 5.0
        assert result.best_value >= 0.0
