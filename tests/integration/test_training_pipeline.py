"""Integration tests for the end-to-end training pipeline (Gap 12-02-02, ACCU-05).

Verifies the full flow:
  trainer -> tune_hyperparameters (OptunaTuner delegation)
          -> train (on synthetic data)
          -> save_model_artifact (with params sidecar)
          -> load_model_artifact (reads back params sidecar)

Uses tiny synthetic data (32 rows, 3 features) and n_trials=5 to stay fast.
The synthetic data has alternating home_win labels to guarantee both classes
appear in every CV fold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.artifacts import load_model_artifact, update_manifest
from models.temporal import TemporalSplitConfig
from models.trainers.ats_trainer import ATSTrainer
from models.trainers.wp_trainer import WPTrainer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_wp_df(n_per_season: int = 8) -> pd.DataFrame:
    """Synthetic WP feature matrix with balanced binary labels.

    Seasons: [2018, 2019] train, [2020] hp_val, [2021] holdout.
    Labels alternate 0/1 to ensure both classes appear in every CV fold.
    """
    rng = np.random.RandomState(42)
    rows = []
    seasons = (
        [2018] * n_per_season
        + [2019] * n_per_season
        + [2020] * n_per_season
        + [2021] * n_per_season
    )
    for i, season in enumerate(seasons):
        rows.append(
            {
                "game_id": f"g{i:03d}",
                "season": season,
                "week": (i % n_per_season) + 1,
                "home_team": "KC",
                "away_team": "BAL",
                "home_score": 0,
                "away_score": 0,
                "feature_timestamp": None,
                "target_wp": None,
                "target_ats": None,
                "target_ou": None,
                # Alternating label guarantees balanced classes in CV folds
                "home_win": i % 2,
                "home_margin": 0.0,
                "point_differential": 0.0,
                "total_points": 0.0,
                "home_covered_spread": 0,
                "game_went_over": 0,
                "feat_a": rng.randn(),
                "feat_b": rng.randn(),
                "feat_c": rng.randn(),
            }
        )

    df = pd.DataFrame(rows)
    return df.set_index("game_id")


def _make_ats_df(n_per_season: int = 8) -> pd.DataFrame:
    """Synthetic ATS feature matrix with continuous home_margin target."""
    rng = np.random.RandomState(99)
    rows = []
    seasons = (
        [2018] * n_per_season
        + [2019] * n_per_season
        + [2020] * n_per_season
        + [2021] * n_per_season
    )
    for i, season in enumerate(seasons):
        feat_a = rng.randn()
        rows.append(
            {
                "game_id": f"g{i:03d}",
                "season": season,
                "week": (i % n_per_season) + 1,
                "home_team": "KC",
                "away_team": "BAL",
                "home_score": 0,
                "away_score": 0,
                "feature_timestamp": None,
                "target_wp": None,
                "target_ats": None,
                "target_ou": None,
                "home_win": i % 2,
                "point_differential": 0.0,
                "total_points": 0.0,
                "home_covered_spread": 0,
                "game_went_over": 0,
                "feat_a": feat_a,
                "feat_b": rng.randn(),
                "feat_c": rng.randn(),
                "home_margin": feat_a * 3.5 + rng.randn() * 2,
            }
        )

    df = pd.DataFrame(rows)
    return df.set_index("game_id")


@pytest.fixture
def tiny_wp_features_df():
    return _make_wp_df()


@pytest.fixture
def tiny_ats_features_df():
    return _make_ats_df()


@pytest.fixture
def small_config():
    """TemporalSplitConfig with tiny train/hp_val/holdout windows."""
    return TemporalSplitConfig(
        train_seasons=[2018, 2019],
        hp_val_seasons=[2020],
        holdout_seasons=[2021],
    )


@pytest.fixture
def patched_optuna_tuner(tmp_path):
    """Redirects OptunaTuner's storage_dir to tmp_path and caps n_trials at 5."""
    import models.trainers.base as base_module
    from models.tuning import OptunaTuner

    optuna_storage = tmp_path / "optuna"

    class PatchedTuner(OptunaTuner):
        def __init__(
            self, study_name, direction="minimize", storage_dir=None, n_trials=100
        ):
            # Cap n_trials for test speed; redirect storage to tmp
            super().__init__(
                study_name=study_name,
                direction=direction,
                storage_dir=optuna_storage,
                n_trials=min(n_trials, 5),
            )

    original_tuner_cls = base_module.OptunaTuner
    base_module.OptunaTuner = PatchedTuner
    yield
    base_module.OptunaTuner = original_tuner_cls


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestWPTrainerEndToEnd:
    """WPTrainer: tune -> train -> save -> load round trip."""

    def test_wp_trainer_tuning_result_set_after_train(
        self,
        tiny_wp_features_df,
        small_config,
        tmp_path,
        patched_optuna_tuner,
    ):
        """train_and_evaluate triggers OptunaTuner delegation and sets _tuning_result."""
        trainer = WPTrainer(config=small_config)

        trainer.train_and_evaluate(tiny_wp_features_df)

        # _tuning_result populated means OptunaTuner.optimize() was called
        assert trainer._tuning_result is not None
        assert isinstance(trainer._tuning_result.best_params, dict)
        assert trainer._tuning_result.n_trials > 0

    def test_wp_trainer_best_params_contain_logreg_keys(
        self,
        tiny_wp_features_df,
        small_config,
        tmp_path,
        patched_optuna_tuner,
    ):
        """After train_and_evaluate, returned best_params contain LogReg keys."""
        trainer = WPTrainer(config=small_config)

        result = trainer.train_and_evaluate(tiny_wp_features_df)

        best = result["best_params"]
        assert isinstance(best, dict)
        assert "C" in best
        assert "penalty" in best

    def test_wp_trainer_save_creates_artifact_files(
        self,
        tiny_wp_features_df,
        small_config,
        tmp_path,
        patched_optuna_tuner,
    ):
        """WPTrainer.train_and_evaluate then save creates required artifact files."""
        artifacts_dir = tmp_path / "artifacts"
        trainer = WPTrainer(config=small_config)

        trainer.train_and_evaluate(tiny_wp_features_df)
        artifact_path = trainer.save(artifacts_dir=artifacts_dir)

        assert artifact_path.exists()
        assert (artifact_path / "model.pkl").exists()
        assert (artifact_path / "metadata.json").exists()
        assert (artifact_path / "feature_list.json").exists()

    def test_wp_trainer_save_produces_params_sidecar(
        self,
        tiny_wp_features_df,
        small_config,
        tmp_path,
        patched_optuna_tuner,
    ):
        """WPTrainer.save produces wp_params.json sidecar when tuning was run."""
        artifacts_dir = tmp_path / "artifacts"
        trainer = WPTrainer(config=small_config)

        trainer.train_and_evaluate(tiny_wp_features_df)
        artifact_path = trainer.save(artifacts_dir=artifacts_dir)

        params_file = artifact_path / "wp_params.json"
        assert params_file.exists(), (
            "Expected wp_params.json sidecar alongside model.pkl"
        )

    def test_wp_trainer_artifact_round_trip_via_load(
        self,
        tiny_wp_features_df,
        small_config,
        tmp_path,
        patched_optuna_tuner,
    ):
        """load_model_artifact on a WPTrainer artifact returns params sidecar."""
        artifacts_dir = tmp_path / "artifacts"
        trainer = WPTrainer(config=small_config)

        trainer.train_and_evaluate(tiny_wp_features_df)
        artifact_path = trainer.save(artifacts_dir=artifacts_dir)
        # .save() no longer auto-swaps (D24-08); register the manifest explicitly
        # so load_model_artifact resolves the version.
        update_manifest("wp", artifact_path.name, artifacts_dir=artifacts_dir)

        loaded = load_model_artifact(target="wp", artifacts_dir=artifacts_dir)

        assert "model" in loaded
        assert "metadata" in loaded
        assert "feature_list" in loaded
        assert "params" in loaded
        assert loaded["params"] is not None
        assert "best_params" in loaded["params"]
        assert isinstance(loaded["params"]["best_params"], dict)


class TestATSTrainerSaveLoad:
    """ATSTrainer: save then load round trip produces valid artifact."""

    def test_ats_trainer_save_creates_loadable_artifact(
        self,
        tiny_ats_features_df,
        small_config,
        tmp_path,
        patched_optuna_tuner,
    ):
        """ATSTrainer train -> save -> load_model_artifact returns valid artifact dict."""
        artifacts_dir = tmp_path / "artifacts"
        trainer = ATSTrainer(config=small_config)

        trainer.train_and_evaluate(tiny_ats_features_df)
        artifact_path = trainer.save(artifacts_dir=artifacts_dir)
        # .save() no longer auto-swaps (D24-08); register the manifest explicitly
        # so load_model_artifact resolves the version.
        update_manifest("ats", artifact_path.name, artifacts_dir=artifacts_dir)

        loaded = load_model_artifact(target="ats", artifacts_dir=artifacts_dir)

        assert "model" in loaded
        assert "feature_list" in loaded
        assert len(loaded["feature_list"]) > 0
        assert hasattr(loaded["model"], "predict")
