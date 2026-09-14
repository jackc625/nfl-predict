"""The new artifact contract binds NEW artifacts ONLY -- proven, not asserted (D33.1-R2).

WHAT THIS MODULE IS FOR
-----------------------
Plan 33.1-10 extended the artifact contract so a fitted preprocessing object and the ATS /
O/U converter parameters are persisted and consumed. That change rewrites an existing
execution path, ``models.prediction_pipeline.predict_games``, and the three artifacts
serving production today were all saved before it. If the rewrite moved even one of their
numbers, a DATA-CORRECTION phase would have silently changed live predictions with no
re-fit -- which breaks this phase's SPEC fence (``artifacts/latest.json`` byte-identical,
no re-fit, no promotion) and is exactly the defect class this milestone exists to detect.

So the values below are compared against ``tests.phase33_state.LEGACY_ARTIFACT_SERVING_BASELINE``,
captured from the PRE-CHANGE code path at commit ``a916504`` and pinned before the change
was made. The ordering is load-bearing: a baseline captured after the change is a
transcription of the change, not a baseline.

The frame is REBUILT from the recorded rule rather than stored, and the pinned predictions
are what check the rebuild: if the rule and the capture had drifted apart, the equality
below would fail rather than pass on a different frame.

SERVING MUST NOT REFUSE A LEGACY ARTIFACT. The owner rejected a refusal explicitly -- it
would take current-week prediction generation down on purpose -- so the branch is
``if preprocessing is not None`` and never an assertion, and there is an explicit
non-refusal test for it.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models.artifacts import (
    CONVERTER_PARAMS_METADATA_KEY,
    PREPROCESSING_FILENAME,
    load_model_artifact,
)
from models.prediction_pipeline import NFLPredictionPipeline
from models.trainers.wp_trainer import WP_PIPELINE_STEP_NAMES
from tests.phase33_state import (
    INCUMBENT_ARTIFACTS,
    LEGACY_ARTIFACT_SERVING_BASELINE,
    WP_TRAINED_SCALED_SERVED_RAW,
)

_ARTIFACTS_ROOT = Path("artifacts")
_LATEST = _ARTIFACTS_ROOT / "latest.json"
_TARGETS = ("wp", "ats", "ou")

_SKIP_REASON = (
    "artifacts/latest.json is absent. It is the deployed-model manifest, it is "
    "gitignored, and it is written by models.artifacts.update_manifest via "
    "scripts/promote_models.py on a passing deploy gate."
)

requires_deployed_artifacts = pytest.mark.skipif(
    not _LATEST.exists(), reason=_SKIP_REASON
)


def _build_baseline_frame(feature_union: list[str]) -> pd.DataFrame:
    """Rebuild the pinned frame from the RULE recorded in phase33_state.

    Deterministic from the record alone: seed, row count and the two offsets are all
    constants there, and the column order is the sorted union of the three artifacts'
    feature lists. Nothing here is a second copy of the data -- the pinned predictions are
    what prove the rebuild agrees with the capture.
    """
    seed = int(LEGACY_ARTIFACT_SERVING_BASELINE["baseline_seed"])
    n_rows = int(LEGACY_ARTIFACT_SERVING_BASELINE["n_rows"])
    spread_offset = int(LEGACY_ARTIFACT_SERVING_BASELINE["spread_seed_offset"])
    total_offset = int(LEGACY_ARTIFACT_SERVING_BASELINE["total_seed_offset"])

    data: dict[str, object] = {
        "game_id": [f"legacy_{i:02d}" for i in range(n_rows)],
        "home_team": ["KC"] * n_rows,
        "away_team": ["BUF"] * n_rows,
    }
    for index, name in enumerate(feature_union):
        data[name] = np.random.default_rng(seed + index).standard_normal(n_rows)
    data["market_spread"] = (
        np.random.default_rng(seed + spread_offset).standard_normal(n_rows) * 3.0
    )
    data["market_total"] = 44.0 + (
        np.random.default_rng(seed + total_offset).standard_normal(n_rows) * 3.0
    )
    return pd.DataFrame(data)


@pytest.fixture
def deployed_artifacts() -> dict[str, dict]:
    return {
        target: load_model_artifact(target, None, _ARTIFACTS_ROOT)
        for target in _TARGETS
    }


@pytest.fixture
def baseline_frame(deployed_artifacts: dict[str, dict]) -> pd.DataFrame:
    union = sorted(
        set(deployed_artifacts["wp"]["feature_list"])
        | set(deployed_artifacts["ats"]["feature_list"])
        | set(deployed_artifacts["ou"]["feature_list"])
    )
    assert len(union) == LEGACY_ARTIFACT_SERVING_BASELINE["feature_union_size"], (
        "the deployed feature lists changed size, so the pinned baseline describes a "
        "different frame than the one about to be built"
    )
    return _build_baseline_frame(union)


@requires_deployed_artifacts
class TestALegacyArtifactIsServedExactlyAsBefore:
    """Zero change to current predictions, measured against a pre-change capture."""

    def test_the_deployed_pointers_are_the_recorded_incumbents(self) -> None:
        recorded = dict(INCUMBENT_ARTIFACTS)
        pinned = LEGACY_ARTIFACT_SERVING_BASELINE["artifacts"]

        for target, artifact_id in zip(_TARGETS, pinned, strict=True):
            assert recorded[target] == artifact_id

    def test_no_deployed_artifact_carries_persisted_preprocessing(
        self, deployed_artifacts: dict[str, dict]
    ) -> None:
        """Asserted rather than assumed -- it is what puts all three on the legacy path."""
        for target in _TARGETS:
            assert deployed_artifacts[target]["preprocessing"] is None, target

    def test_no_deployed_artifact_carries_converter_params(
        self, deployed_artifacts: dict[str, dict]
    ) -> None:
        for target in _TARGETS:
            metadata = deployed_artifacts[target]["metadata"]
            assert CONVERTER_PARAMS_METADATA_KEY not in metadata, target

    def test_served_values_equal_the_pinned_pre_change_baseline(
        self, deployed_artifacts: dict[str, dict], baseline_frame: pd.DataFrame
    ) -> None:
        pipeline = NFLPredictionPipeline(
            wp_artifact=deployed_artifacts["wp"],
            ats_artifact=deployed_artifacts["ats"],
            ou_artifact=deployed_artifacts["ou"],
        )
        predictions = pipeline.predict_games(baseline_frame)
        series = LEGACY_ARTIFACT_SERVING_BASELINE["series"]

        for name, expected in series.items():
            served = np.array([getattr(p, name) for p in predictions])
            np.testing.assert_array_equal(
                served,
                np.array(expected),
                err_msg=(
                    f"'{name}' moved against the pre-change baseline. The new contract "
                    "binds artifacts saved UNDER it only; a legacy artifact's served "
                    "values must not change by one bit (D33.1-R2)."
                ),
            )

    def test_a_legacy_load_and_predict_completes_and_does_not_refuse(
        self, deployed_artifacts: dict[str, dict], baseline_frame: pd.DataFrame
    ) -> None:
        """The branch is a conditional, never an assertion.

        A refusal would have been the tidier-looking design and the owner rejected it
        outright: refusing a legacy artifact takes current-week prediction generation down
        on purpose.
        """
        pipeline = NFLPredictionPipeline(
            wp_artifact=deployed_artifacts["wp"],
            ats_artifact=deployed_artifacts["ats"],
            ou_artifact=deployed_artifacts["ou"],
        )

        predictions = pipeline.predict_games(baseline_frame)

        assert len(predictions) == len(baseline_frame)
        assert all(0.0 <= p.wp_home_probability <= 1.0 for p in predictions)

    def test_serving_applies_the_deployed_platt_calibrator(
        self, deployed_artifacts: dict[str, dict], baseline_frame: pd.DataFrame
    ) -> None:
        """Regression cover for the Rule-1 fix at commit a916504.

        ``predict_games`` was the one site calling ``calibrator.transform()``. The
        deployed WP calibrator is a ``models.calibrate.PlattCalibrator``, which exposes
        ``predict`` and NOT ``transform``, so this whole module was unservable until that
        was corrected. No synthetic fixture caught it because every one of them saves its
        artifacts WITHOUT a calibrator, so the branch was never entered.
        """
        calibrator = deployed_artifacts["wp"]["calibrator"]
        assert calibrator is not None
        assert not hasattr(calibrator, "transform"), (
            "the deployed calibrator grew a transform method, so this regression no "
            "longer covers what it was written for"
        )

        wp_artifact = deployed_artifacts["wp"]
        raw = wp_artifact["model"].predict_proba(
            baseline_frame[wp_artifact["feature_list"]]
        )[:, 1]
        expected = np.asarray(calibrator.predict(raw))

        pipeline = NFLPredictionPipeline(
            wp_artifact=wp_artifact,
            ats_artifact=deployed_artifacts["ats"],
            ou_artifact=deployed_artifacts["ou"],
        )
        served = np.array(
            [p.wp_home_probability for p in pipeline.predict_games(baseline_frame)]
        )

        np.testing.assert_array_equal(served, expected)


class TestTheDeployedWPArtifactIsTrainedScaledAndServedRaw:
    """THE CURRENT LIVE STATE OF PRODUCTION, recorded as a measured fact (D33.1-R2).

    ``wp_20260824_113325`` was fitted on STANDARDISED features and is served on
    UNSTANDARDISED ones. That is true right now, in production, and it predates Phase
    33.1: it was found by the orchestrator reading source during the cross-AI review, not
    introduced here.

    THIS PHASE DOES NOT FIX IT, and the reason is not timidity. Serving the existing
    artifact scaled would change live predictions with NO re-fit, which breaks this
    phase's SPEC fence (no re-fit, no promotion, ``artifacts/latest.json`` byte-identical)
    and R8's rule against presenting a data-correction phase as an accuracy change. The
    honest act is to MEASURE the defect, not to repair it quietly.

    THIS TEST IS EXPECTED TO KEEP PASSING until Phase 33's Wave 15 ships a WP artifact
    under the new contract. At that point it must be UPDATED with a recorded reason --
    never deleted, because deleting it erases the disclosure rather than closing it.

    It is deliberately NOT in ``tests.phase33_state.DELIBERATE_TRIPWIRE_NODE_IDS``: it is
    GREEN and describes a fact, not RED encoding an accepted failure.
    """

    def test_wp_20260824_113325_carries_no_persisted_preprocessing(self) -> None:
        artifact_dir = _ARTIFACTS_ROOT / "wp_20260824_113325"
        if not artifact_dir.exists():
            pytest.skip(_SKIP_REASON)

        assert not (artifact_dir / PREPROCESSING_FILENAME).exists(), (
            "wp_20260824_113325 grew a preprocessing.pkl. If Wave 15 shipped a "
            "replacement under the new contract, UPDATE this class with a recorded "
            "reason rather than deleting it."
        )
        assert WP_TRAINED_SCALED_SERVED_RAW["artifact_id"] == "wp_20260824_113325"

    def test_the_wp_trainer_fits_its_estimator_on_scaled_features(self) -> None:
        """Checked against CODE STRUCTURE, not against prose.

        Under D33.1-R3 the scaling no longer happens through a separately-named frame --
        it is the third step of the Pipeline whose fourth step is the estimator. So the
        claim "the estimator is fitted on scaled features" is now a statement about STEP
        ORDER, and that is what is asserted: the scaler is constructed immediately before
        the estimator inside the single construction site.
        """
        source = Path("models/trainers/wp_trainer.py").read_text(encoding="utf-8")
        tree = ast.parse(source)

        create_model = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_create_model"
        )
        constructed = [
            node.func.id
            for node in ast.walk(create_model)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        ]

        assert "StandardScaler" in constructed
        assert "LogisticRegression" in constructed
        assert constructed.index("StandardScaler") < constructed.index(
            "LogisticRegression"
        ), (
            "the estimator is no longer preceded by the scaler in wp_trainer._create_model"
        )
        assert WP_PIPELINE_STEP_NAMES.index("scaler") < WP_PIPELINE_STEP_NAMES.index(
            "estimator"
        )

    def test_predict_games_serves_raw_features_when_no_preprocessing_is_persisted(
        self,
    ) -> None:
        """The legacy branch feeds the un-transformed selected columns to the model."""
        source = inspect.getsource(NFLPredictionPipeline.predict_games)
        body = [
            line.strip()
            for line in source.splitlines()
            if not line.lstrip().startswith("#")
        ]

        assert "wp_feature_df = games_data[wp_features].copy()" in body
        assert "wp_raw_probs = wp_model.predict_proba(wp_feature_df)[:, 1]" in body
        assert "if wp_preprocessing is not None:" in body

    @requires_deployed_artifacts
    def test_the_deployed_wp_artifact_is_served_on_that_raw_path(
        self, deployed_artifacts: dict[str, dict], baseline_frame: pd.DataFrame
    ) -> None:
        """The BEHAVIOURAL half: it is not merely that the branch exists, it is taken."""
        wp_artifact = deployed_artifacts["wp"]
        assert wp_artifact["preprocessing"] is None
        assert wp_artifact["artifact_dir"].name == "wp_20260824_113325"

        raw_features = baseline_frame[wp_artifact["feature_list"]]
        expected_raw = wp_artifact["model"].predict_proba(raw_features)[:, 1]

        pipeline = NFLPredictionPipeline(
            wp_artifact=wp_artifact,
            ats_artifact=deployed_artifacts["ats"],
            ou_artifact=deployed_artifacts["ou"],
        )
        predictions = pipeline.predict_games(baseline_frame)
        served_calibrated = np.array([p.wp_home_probability for p in predictions])
        served_uncalibrated = np.asarray(
            wp_artifact["calibrator"].predict(expected_raw)
        )

        np.testing.assert_array_equal(served_calibrated, served_uncalibrated)

    def test_the_record_says_this_phase_does_not_fix_it_and_names_wave_15(self) -> None:
        record = WP_TRAINED_SCALED_SERVED_RAW

        assert record["artifact_id"] == "wp_20260824_113325"
        assert len(record["source_sites"]) >= 3
        assert record["predates_phase_331"] is True
        assert record["fixed_in_this_phase"] is False
        assert "Wave 15" in str(record["resolved_by"])
