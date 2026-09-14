"""The save / load / predict ROUND TRIP this repository did not have (D33.1-R1).

WHY EXACT EQUALITY AND NOT A TOLERANCE
--------------------------------------
Every assertion of served-equals-in-memory below uses
``numpy.testing.assert_array_equal`` and NEVER ``assert_allclose``. The defect this test
exists to catch is a MISSING TRANSFORM, and a missing standardisation does not produce a
nearly-identical answer -- it produces a plausible WRONG one. A tolerance would hide
exactly the class of bug the test is for.

WHAT THE CONTRACT IS
--------------------
Before Plan 33.1-10, ``save_model_artifact`` had no preprocessing parameter,
``load_model_artifact`` returned none, and the string ``scaler`` appeared nowhere in
``models/artifacts.py``. A WP estimator fitted on standardised features and served on
unstandardised ones therefore produced a plausible wrong answer rather than an error. The
structural defence is that the transform and the estimator are ONE persisted object: for
WP the persisted preprocessing IS the four-step Pipeline whose final step is the estimator,
so there is no way to load the estimator without the scaler.

Separately, ATS and O/U conversion was re-derived at serving time from
``metadata.get("residual_std", 13.5)`` / ``13.0`` -- hardcoded fallbacks standing in for a
fitted object that ``BaseTrainer.save`` never persisted under any name. The parameters are
now written as JSON in metadata and consumed by serving.

NOTHING UNDER PRODUCTION ``artifacts/`` IS WRITTEN. Every save here targets ``tmp_path``
and ``update_latest`` is never True against the production root.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from models.artifacts import (
    CONVERTER_PARAMS_METADATA_KEY,
    PREPROCESSING_FILENAME,
    load_model_artifact,
    save_model_artifact,
    update_manifest,
)
from models.prediction_pipeline import NFLPredictionPipeline
from models.train_ats import ResidualDistributionConverter
from models.trainers.ats_trainer import ATSTrainer
from models.trainers.ou_trainer import OUTrainer
from models.trainers.wp_trainer import WPTrainer

# The file set `save_model_artifact` produced BEFORE this contract existed, for a save
# carrying no calibrator and no best_params. Written out rather than derived, because the
# whole point of the control is to notice a change in it.
_PRE_CHANGE_FILE_SET: frozenset[str] = frozenset(
    {"model.pkl", "metadata.json", "feature_list.json"}
)

_FEATURES: tuple[str, ...] = (
    "feature_a",
    "feature_b",
    "feature_c",
    "feature_d",
    "feature_e",
    "feature_f",
)

_N_ROWS = 120
_SEED = 3311020


def _synthetic_games() -> pd.DataFrame:
    """A small frame carrying every feature, both market lines and the three targets."""
    rng = np.random.default_rng(_SEED)
    data: dict[str, Any] = {
        "game_id": [f"synthetic_{i:03d}" for i in range(_N_ROWS)],
        "home_team": ["KC"] * _N_ROWS,
        "away_team": ["BUF"] * _N_ROWS,
    }
    # Deliberately DIFFERENT scales per column. A model fitted on standardised features
    # and served on unstandardised ones only misbehaves visibly when the columns are not
    # already unit-variance, which is the realistic case and the one worth testing.
    for index, name in enumerate(_FEATURES):
        data[name] = rng.normal(loc=10.0 * index, scale=1.0 + 3.0 * index, size=_N_ROWS)

    frame = pd.DataFrame(data)
    signal = (frame["feature_a"] - frame["feature_a"].mean()) / frame["feature_a"].std()
    frame["home_win"] = (signal + rng.normal(scale=0.3, size=_N_ROWS) > 0).astype(int)
    frame["home_margin"] = 6.0 * signal + rng.normal(scale=2.0, size=_N_ROWS)
    frame["total_points"] = 44.0 + 5.0 * signal + rng.normal(scale=3.0, size=_N_ROWS)
    frame["market_spread"] = rng.normal(scale=3.0, size=_N_ROWS)
    frame["market_total"] = 44.0 + rng.normal(scale=3.0, size=_N_ROWS)
    return frame


def _fit_wp(frame: pd.DataFrame) -> WPTrainer:
    """A WP trainer holding the fitted four-step Pipeline as BOTH model and preprocessing."""
    trainer = WPTrainer()
    trainer.feature_names = list(_FEATURES)
    pipeline = trainer._create_model({})
    pipeline.fit(frame[list(_FEATURES)], frame["home_win"])
    trainer.model = pipeline
    trainer.preprocessing = pipeline
    trainer.metadata = {"target": "wp", "feature_names": list(_FEATURES)}
    return trainer


def _fit_ats(frame: pd.DataFrame) -> ATSTrainer:
    """An ATS trainer holding a fitted XGBRegressor and a fitted residual converter."""
    trainer = ATSTrainer()
    trainer.feature_names = list(_FEATURES)
    model = trainer._create_model({})
    model.fit(frame[list(_FEATURES)], frame["home_margin"])
    trainer.model = model
    trainer.residual_converter = trainer._fit_residual_converter(
        frame["home_margin"].to_numpy(), model.predict(frame[list(_FEATURES)])
    )
    trainer.metadata = {"target": "ats", "feature_names": list(_FEATURES)}
    return trainer


def _fit_ou(frame: pd.DataFrame) -> OUTrainer:
    """An O/U trainer holding a fitted XGBRegressor and a fitted total converter."""
    trainer = OUTrainer()
    trainer.feature_names = list(_FEATURES)
    model = trainer._create_model({})
    model.fit(frame[list(_FEATURES)], frame["total_points"])
    trainer.model = model
    trainer.total_converter = trainer._fit_total_converter(
        frame["total_points"].to_numpy(), model.predict(frame[list(_FEATURES)])
    )
    trainer.metadata = {"target": "ou", "feature_names": list(_FEATURES)}
    return trainer


@pytest.fixture
def frame() -> pd.DataFrame:
    return _synthetic_games()


@pytest.fixture
def trainers(frame: pd.DataFrame) -> dict[str, Any]:
    return {"wp": _fit_wp(frame), "ats": _fit_ats(frame), "ou": _fit_ou(frame)}


@pytest.fixture
def artifacts_root(tmp_path: Path, trainers: dict[str, Any]) -> Path:
    """Three artifacts saved under the NEW contract, registered in a tmp latest.json."""
    root = tmp_path / "artifacts"
    root.mkdir()
    for target, trainer in trainers.items():
        artifact_dir = trainer.save(artifacts_dir=root)
        update_manifest(target, artifact_dir.name, root)
    return root


def _serve(artifacts_root: Path, frame: pd.DataFrame) -> list[Any]:
    pipeline = NFLPredictionPipeline()
    pipeline.load_models(artifacts_dir=artifacts_root)
    return pipeline.predict_games(frame)


class TestTheRoundTripServesExactlyWhatWasFitted:
    """Fit -> save -> reload -> serve, and the numbers must be IDENTICAL."""

    def test_wp_roundtrip_served_equals_in_memory_exactly(
        self, artifacts_root: Path, frame: pd.DataFrame, trainers: dict[str, Any]
    ) -> None:
        expected = trainers["wp"].model.predict_proba(frame[list(_FEATURES)])[:, 1]

        served = np.array(
            [p.wp_home_probability for p in _serve(artifacts_root, frame)]
        )

        np.testing.assert_array_equal(served, expected)

    def test_ats_roundtrip_served_equals_in_memory_exactly(
        self, artifacts_root: Path, frame: pd.DataFrame, trainers: dict[str, Any]
    ) -> None:
        trainer = trainers["ats"]
        margins = trainer.model.predict(frame[list(_FEATURES)])
        expected = trainer.residual_converter.predict_cover_probability(
            np.asarray(margins), frame["market_spread"].to_numpy()
        )

        predictions = _serve(artifacts_root, frame)
        np.testing.assert_array_equal(
            np.array([p.predicted_margin for p in predictions]),
            np.asarray(margins, dtype=float),
        )
        np.testing.assert_array_equal(
            np.array([p.ats_cover_probability for p in predictions]), expected
        )

    def test_ou_roundtrip_served_equals_in_memory_exactly(
        self, artifacts_root: Path, frame: pd.DataFrame, trainers: dict[str, Any]
    ) -> None:
        trainer = trainers["ou"]
        totals = trainer.model.predict(frame[list(_FEATURES)])
        over, under = trainer.total_converter.predict_over_under_probabilities(
            np.asarray(totals), frame["market_total"].to_numpy()
        )

        predictions = _serve(artifacts_root, frame)
        np.testing.assert_array_equal(
            np.array([p.over_probability for p in predictions]), over
        )
        np.testing.assert_array_equal(
            np.array([p.under_probability for p in predictions]), under
        )


class TestTheDeletedPreprocessingControls:
    """What the round-trip equality is actually asserting, measured rather than assumed."""

    def test_deleted_preprocessing_is_benign_when_model_pkl_carries_the_pipeline(
        self, artifacts_root: Path, frame: pd.DataFrame, trainers: dict[str, Any]
    ) -> None:
        """MEASURED, and it contradicts the plan's stated expectation -- so it is recorded.

        Under D33.1-R3 ``WPTrainer._create_model`` returns the four-step Pipeline, and
        ``BaseTrainer.save`` writes THAT as ``model.pkl``. So ``model.pkl`` and
        ``preprocessing.pkl`` hold the same object, and deleting the second one makes
        serving fall back to a path that produces IDENTICAL numbers.

        That is a property worth knowing rather than a defect: it means WP is now
        belt-and-braces, and it means this deletion cannot be the control that proves the
        equality asserts the transform. The sibling test below is that control.
        """
        wp_dir = Path(json.loads((artifacts_root / "latest.json").read_text())["wp"])
        preprocessing_path = artifacts_root / wp_dir / PREPROCESSING_FILENAME
        assert preprocessing_path.exists()

        before = np.array(
            [p.wp_home_probability for p in _serve(artifacts_root, frame)]
        )
        preprocessing_path.unlink()
        after = np.array([p.wp_home_probability for p in _serve(artifacts_root, frame)])

        np.testing.assert_array_equal(after, before)

    def _save_split_artifacts(
        self,
        root: Path,
        model: Any,
        preprocessing: Any,
        trainers: dict[str, Any],
    ) -> Path:
        """Save a SPLIT WP artifact -- bare estimator in model.pkl, transform beside it."""
        root.mkdir(parents=True, exist_ok=True)
        wp_dir = save_model_artifact(
            model=model,
            target="wp",
            metadata={"target": "wp"},
            feature_list=list(_FEATURES),
            preprocessing=preprocessing,
            artifacts_dir=root,
            update_latest=True,
        )
        for target in ("ats", "ou"):
            other_dir = save_model_artifact(
                model=trainers[target].model,
                target=target,
                metadata={"target": target},
                feature_list=list(_FEATURES),
                artifacts_dir=root,
            )
            update_manifest(target, other_dir.name, root)
        return wp_dir

    def test_deleted_preprocessing_breaks_the_roundtrip_for_a_bare_estimator_artifact(
        self, tmp_path: Path, frame: pd.DataFrame, trainers: dict[str, Any]
    ) -> None:
        """MEASURED: for a POST-D33.1-R3 estimator the failure is loud, not silent.

        Split the four-step Pipeline: the bare ``estimator`` step into ``model.pkl``, the
        whole Pipeline into ``preprocessing.pkl``. With the preprocessing present the
        served values equal the in-memory Pipeline's EXACTLY. Delete it and serving does
        not merely disagree -- it RAISES, because the estimator was fitted on twice the
        input width (the imputed values and then one missing indicator per column) and
        the raw frame is half that.

        That is a stronger protection than a wrong number, and it is recorded here
        because it is not what the plan predicted. The sibling below covers the shape the
        plan DID predict, which is the one the deployed artifact is actually in.
        """
        pipeline = trainers["wp"].model
        expected = pipeline.predict_proba(frame[list(_FEATURES)])[:, 1]
        root = tmp_path / "bare_estimator_artifacts"
        wp_dir = self._save_split_artifacts(
            root, pipeline.named_steps["estimator"], pipeline, trainers
        )

        with_preprocessing = np.array(
            [p.wp_home_probability for p in _serve(root, frame)]
        )
        np.testing.assert_array_equal(with_preprocessing, expected)

        (wp_dir / PREPROCESSING_FILENAME).unlink()
        with pytest.raises(ValueError, match="features"):
            _serve(root, frame)

    def test_deleted_preprocessing_gives_a_plausible_wrong_answer_for_a_scaled_estimator(
        self, tmp_path: Path, frame: pd.DataFrame, trainers: dict[str, Any]
    ) -> None:
        """THE CONTROL THAT FIRES, in the shape the DEPLOYED artifact is actually in.

        ``wp_20260824_113325`` was fitted before D33.1-R3: a bare ``LogisticRegression``
        fitted on SCALED features of the same width as the raw frame, served raw. That
        combination raises nothing. It returns a different probability per game, silently,
        which is why the defect survived to be found by reading rather than by a failure.

        Reproduced here with a two-step scaler + estimator Pipeline. With the
        preprocessing persisted the served values are EXACT; delete it and they differ
        while every single one stays a perfectly plausible probability in [0, 1].
        """
        features = frame[list(_FEATURES)]
        scaled_pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("estimator", LogisticRegression(max_iter=1000)),
            ]
        )
        scaled_pipeline.fit(features, frame["home_win"])
        expected = scaled_pipeline.predict_proba(features)[:, 1]

        root = tmp_path / "scaled_estimator_artifacts"
        wp_dir = self._save_split_artifacts(
            root, scaled_pipeline.named_steps["estimator"], scaled_pipeline, trainers
        )

        with_preprocessing = np.array(
            [p.wp_home_probability for p in _serve(root, frame)]
        )
        np.testing.assert_array_equal(with_preprocessing, expected)

        (wp_dir / PREPROCESSING_FILENAME).unlink()
        without_preprocessing = np.array(
            [p.wp_home_probability for p in _serve(root, frame)]
        )

        assert not np.array_equal(without_preprocessing, expected), (
            "deleting preprocessing.pkl changed nothing, so the round-trip equality is "
            "not asserting the transform"
        )
        assert np.all(
            (without_preprocessing >= 0.0) & (without_preprocessing <= 1.0)
        ), (
            "the wrong answers must still LOOK like probabilities -- that is what makes "
            "this defect class survive unnoticed"
        )


class TestTheContractItself:
    """The shape of the extension, and that it binds new artifacts only."""

    def test_the_signature_carries_preprocessing_and_load_returns_it(
        self, artifacts_root: Path
    ) -> None:
        signature = inspect.signature(save_model_artifact)
        assert "preprocessing" in signature.parameters
        assert signature.parameters["preprocessing"].default is None

        loaded = load_model_artifact("wp", None, artifacts_root)
        assert "preprocessing" in loaded
        assert loaded["preprocessing"] is not None

    def test_a_save_without_preprocessing_produces_the_pre_change_file_set(
        self, tmp_path: Path, trainers: dict[str, Any]
    ) -> None:
        artifact_dir = save_model_artifact(
            model=trainers["ats"].model,
            target="ats",
            metadata={"target": "ats"},
            feature_list=list(_FEATURES),
            artifacts_dir=tmp_path / "no_preprocessing",
        )

        assert {p.name for p in artifact_dir.iterdir()} == _PRE_CHANGE_FILE_SET
        assert not (artifact_dir / PREPROCESSING_FILENAME).exists()

    def test_an_absent_preprocessing_file_loads_as_none(
        self, tmp_path: Path, trainers: dict[str, Any]
    ) -> None:
        root = tmp_path / "legacy_shaped"
        artifact_dir = save_model_artifact(
            model=trainers["ats"].model,
            target="ats",
            metadata={"target": "ats"},
            feature_list=list(_FEATURES),
            artifacts_dir=root,
        )

        loaded = load_model_artifact("ats", artifact_dir.name, root)
        assert loaded["preprocessing"] is None

    def test_an_ats_artifact_metadata_carries_the_converter_parameters(
        self, artifacts_root: Path
    ) -> None:
        loaded = load_model_artifact("ats", None, artifacts_root)
        params = loaded["metadata"][CONVERTER_PARAMS_METADATA_KEY]

        assert sorted(params) == [
            "distribution_params",
            "distribution_type",
            "residual_std",
        ]
        assert params["distribution_type"] == "normal"
        assert params["residual_std"] > 0.0

    def test_serving_a_converter_params_artifact_does_not_use_the_legacy_default(
        self, artifacts_root: Path, frame: pd.DataFrame, trainers: dict[str, Any]
    ) -> None:
        """The 13.5 fallback must be REACHABLE for legacy artifacts and UNUSED for new ones."""
        loaded = load_model_artifact("ats", None, artifacts_root)
        params = loaded["metadata"][CONVERTER_PARAMS_METADATA_KEY]
        assert params["residual_std"] != pytest.approx(13.5)

        legacy_converter = ResidualDistributionConverter(distribution_type="normal")
        legacy_converter.is_fitted = True
        legacy_converter.residual_std = 13.5
        legacy_converter.distribution_params = {"loc": 0.0, "scale": 13.5}
        margins = trainers["ats"].model.predict(frame[list(_FEATURES)])
        legacy_values = legacy_converter.predict_cover_probability(
            np.asarray(margins), frame["market_spread"].to_numpy()
        )

        served = np.array(
            [p.ats_cover_probability for p in _serve(artifacts_root, frame)]
        )

        assert not np.array_equal(served, legacy_values)

    def test_nothing_under_production_artifacts_was_written(self) -> None:
        """The autouse COLD-05 guard judges this too; the explicit assertion is cheap."""
        production_latest = Path("artifacts") / "latest.json"
        if not production_latest.exists():
            pytest.skip(
                "artifacts/latest.json is absent; it is produced by "
                "scripts/promote_models.py on a passing deploy gate"
            )
        manifest = json.loads(production_latest.read_text())
        assert set(manifest) >= {"wp", "ats", "ou"}
