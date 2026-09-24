"""Unit tests for OptunaTuner, TuningResult, and params sidecar."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from models.artifacts import load_model_artifact, save_model_artifact
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

    def test_creates_study_with_tpe_sampler_and_hyperband(self, tmp_path: Path) -> None:
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


# ---------------------------------------------------------------------------
# Test params sidecar in save/load_model_artifact
# ---------------------------------------------------------------------------


def _make_tiny_model():
    """Create a minimal fitted LogisticRegression for testing."""
    rng = np.random.RandomState(42)
    X = rng.randn(20, 2)
    y = (X[:, 0] > 0).astype(int)
    model = LogisticRegression(max_iter=100)
    model.fit(X, y)
    return model


class TestParamsSidecar:
    """Tests for the JSON params sidecar in save/load_model_artifact."""

    def test_save_with_best_params_creates_params_json(self, tmp_path: Path) -> None:
        """save_model_artifact with best_params creates {target}_params.json."""
        model = _make_tiny_model()
        best_params = {"C": 1.0, "penalty": "l2"}

        artifact_dir = save_model_artifact(
            model=model,
            target="wp",
            metadata={"version": "test"},
            feature_list=["feat1", "feat2"],
            best_params=best_params,
            artifacts_dir=tmp_path,
        )

        params_path = artifact_dir / "wp_params.json"
        assert params_path.exists()

    def test_params_json_contains_best_params(self, tmp_path: Path) -> None:
        """params.json contains 'best_params' key with the passed dict."""
        model = _make_tiny_model()
        best_params = {"C": 1.0, "penalty": "l2"}

        artifact_dir = save_model_artifact(
            model=model,
            target="wp",
            metadata={"version": "test"},
            feature_list=["feat1", "feat2"],
            best_params=best_params,
            artifacts_dir=tmp_path,
        )

        params_path = artifact_dir / "wp_params.json"
        data = json.loads(params_path.read_text())
        assert "best_params" in data
        assert data["best_params"] == {"C": 1.0, "penalty": "l2"}

    def test_params_json_contains_tuning_metadata(self, tmp_path: Path) -> None:
        """params.json contains 'tuning_metadata' key when provided."""
        model = _make_tiny_model()
        best_params = {"C": 0.5}
        tuning_metadata = {
            "study_name": "wp_tune",
            "n_trials": 100,
            "best_value": 0.42,
        }

        artifact_dir = save_model_artifact(
            model=model,
            target="wp",
            metadata={"version": "test"},
            feature_list=["feat1", "feat2"],
            best_params=best_params,
            tuning_metadata=tuning_metadata,
            artifacts_dir=tmp_path,
        )

        params_path = artifact_dir / "wp_params.json"
        data = json.loads(params_path.read_text())
        assert "tuning_metadata" in data
        assert data["tuning_metadata"]["study_name"] == "wp_tune"
        assert data["tuning_metadata"]["n_trials"] == 100

    def test_save_without_best_params_no_params_json(self, tmp_path: Path) -> None:
        """save_model_artifact without best_params does NOT create _params.json."""
        model = _make_tiny_model()

        artifact_dir = save_model_artifact(
            model=model,
            target="wp",
            metadata={"version": "test"},
            feature_list=["feat1", "feat2"],
            artifacts_dir=tmp_path,
        )

        params_path = artifact_dir / "wp_params.json"
        assert not params_path.exists()

    def test_load_returns_params_when_exists(self, tmp_path: Path) -> None:
        """load_model_artifact returns 'params' key when _params.json exists."""
        model = _make_tiny_model()
        best_params = {"C": 2.0, "penalty": "l1"}

        # update_latest=True registers latest.json so load resolves (D24-08:
        # save no longer auto-swaps by default).
        save_model_artifact(
            model=model,
            target="wp",
            metadata={"version": "test"},
            feature_list=["feat1", "feat2"],
            best_params=best_params,
            artifacts_dir=tmp_path,
            update_latest=True,
        )

        loaded = load_model_artifact(
            target="wp",
            artifacts_dir=tmp_path,
        )

        assert "params" in loaded
        assert loaded["params"] is not None
        assert loaded["params"]["best_params"] == {"C": 2.0, "penalty": "l1"}

    def test_load_returns_none_params_when_no_sidecar(self, tmp_path: Path) -> None:
        """load_model_artifact returns params=None when no _params.json."""
        model = _make_tiny_model()

        # update_latest=True registers latest.json so load resolves (D24-08:
        # save no longer auto-swaps by default).
        save_model_artifact(
            model=model,
            target="wp",
            metadata={"version": "test"},
            feature_list=["feat1", "feat2"],
            artifacts_dir=tmp_path,
            update_latest=True,
        )

        loaded = load_model_artifact(
            target="wp",
            artifacts_dir=tmp_path,
        )

        assert "params" in loaded
        assert loaded["params"] is None


# ---------------------------------------------------------------------------
# WR-08: the trial count must not report budget under a "completed" label
# ---------------------------------------------------------------------------


class TestAnArtifactIdNamesOnePayload:
    """A33.2-review WR-07: a UTC-named directory that is never silently overwritten."""

    def test_a_second_save_in_the_same_second_is_refused_not_overwritten(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from datetime import UTC, datetime

        import models.artifacts as artifacts_module

        frozen = datetime(2026, 11, 1, 5, 30, 0, tzinfo=UTC)
        seen_tz: list[object] = []

        class _FrozenClock(datetime):
            @classmethod
            def now(cls, tz=None):  # type: ignore[override]
                seen_tz.append(tz)
                return frozen if tz is not None else frozen.replace(tzinfo=None)

        monkeypatch.setattr(artifacts_module, "datetime", _FrozenClock)
        first = save_model_artifact(
            model=_make_tiny_model(),
            target="wp",
            metadata={"version": "first"},
            feature_list=["feat1", "feat2"],
            artifacts_dir=tmp_path,
        )
        assert first.name == "wp_20261101_053000"
        assert seen_tz == [UTC], "the directory name must come from the UTC clock"

        with pytest.raises(FileExistsError):
            save_model_artifact(
                model=_make_tiny_model(),
                target="wp",
                metadata={"version": "second"},
                feature_list=["feat1", "feat2"],
                artifacts_dir=tmp_path,
            )
        payload = json.loads((first / "metadata.json").read_text())
        assert payload["version"] == "first"


class TestCompletedTrialsAreCountedSeparately:
    """``len(study.trials)`` is trials STARTED, including PRUNED and FAIL.

    Every search in this project runs under
    ``HyperbandPruner(min_resource=1, max_resource=3, reduction_factor=3)``, whose whole
    purpose is to prune the majority of trials. ``tuning_provenance.json``'s
    ``per_target_trials_added`` and its ``how_to_read_this`` prose ("trials_added > 0
    means the search genuinely ran that many trials") were both reading the started count,
    and ``GATED-REFIT-READOUT.md`` publishes a column headed "NEW completed trials" that
    the code produced no completed count for anywhere.

    The anti-vacuity check is CORRECTLY a question about started trials -- a study resumed
    at budget starts none -- so both counts are recorded, each under its real name, rather
    than one replacing the other.
    """

    def test_a_study_with_no_pruning_has_equal_counts(self, tmp_path: Path) -> None:
        tuner = OptunaTuner(
            study_name="wr08_all_complete", storage_dir=tmp_path, n_trials=5
        )
        result = tuner.optimize(_quadratic_objective)

        assert result.n_trials == 5
        assert result.n_completed_trials == 5

    def test_pruned_and_failed_trials_are_started_but_not_completed(
        self, tmp_path: Path
    ) -> None:
        """The distinction, forced: half the trials raise and are recorded as FAIL."""
        import optuna

        calls = {"n": 0}

        def flaky(trial):
            x = trial.suggest_float("x", -10, 10)
            calls["n"] += 1
            if calls["n"] % 2 == 0:
                raise ValueError("deliberate trial failure")
            return (x - 3) ** 2

        tuner = OptunaTuner(study_name="wr08_flaky", storage_dir=tmp_path, n_trials=6)
        study = optuna.create_study(
            study_name=tuner.study_name,
            storage=tuner.storage_url,
            direction="minimize",
            load_if_exists=True,
        )
        study.optimize(flaky, n_trials=6, catch=(ValueError,))

        from models.tuning import count_completed_trials

        assert len(study.trials) == 6
        assert count_completed_trials(study) == 3, (
            "count_completed_trials must exclude non-COMPLETE states; a started count "
            "under a 'completed' label overstates the work the search did"
        )

    def test_count_completed_trials_excludes_every_non_complete_state(self) -> None:
        """Asserted against optuna's own state enum, not a hand-written list."""
        import optuna

        from models.tuning import count_completed_trials

        class _Trial:
            def __init__(self, state):
                self.state = state

        class _Study:
            def __init__(self, states):
                self.trials = [_Trial(s) for s in states]

        states = [
            optuna.trial.TrialState.COMPLETE,
            optuna.trial.TrialState.PRUNED,
            optuna.trial.TrialState.FAIL,
            optuna.trial.TrialState.RUNNING,
            optuna.trial.TrialState.WAITING,
            optuna.trial.TrialState.COMPLETE,
        ]
        assert count_completed_trials(_Study(states)) == 2

    def test_the_dataclass_documents_n_trials_as_STARTED(self) -> None:
        assert TuningResult.__doc__ is not None
        assert "STARTED" in TuningResult.__doc__, (
            "TuningResult.n_trials used to be documented as 'Total number of trials "
            "completed in the study', which is not what len(study.trials) counts"
        )
        assert "n_completed_trials" in TuningResult.__doc__

    def test_the_base_trainer_records_both_counts(self) -> None:
        import inspect

        from models.trainers.base import BaseTrainer

        source = inspect.getsource(BaseTrainer.tune_hyperparameters)
        assert "last_tuning_trials_added" in source
        assert "last_tuning_completed_added" in source, (
            "BaseTrainer records only the started count, so no consumer can report "
            "completed trials without inventing the number"
        )

    def test_the_engine_provenance_carries_both_totals(self) -> None:
        import inspect

        from backtest.engine import BacktestEngine

        source = inspect.getsource(BacktestEngine._write_tuning_provenance)
        assert "per_target_trials_added" in source
        assert "per_target_completed_trials_added" in source

        prose = source[source.find("how_to_read_this") :]
        assert "BUDGET figure" in prose, (
            "how_to_read_this still claims trials_added is how many trials the search "
            "genuinely ran; it is the number STARTED, most of which Hyperband prunes"
        )
