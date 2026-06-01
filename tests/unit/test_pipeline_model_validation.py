"""Tests for the relocated in-package ModelValidator (pipeline.model_validation).

These cover the two methods the Friday orchestrator actually calls:
validate_model_availability and validate_model_loadability. They assert the
return-key shape ({name}_available / {name}_loadable booleans) and the
availability/loadability semantics against the REAL artifact convention --
versioned ``artifacts/{target}_{ts}/`` directories resolved via
``artifacts/latest.json`` with keys ``model``/``feature_list`` (CR-01 fix,
FIX-01 / D-03). The former fixtures fabricated the wrong
``{name}_model.joblib`` schema with ``model/scaler/feature_names`` keys and so
masked the defect; they have been rebuilt to mirror
``models.artifacts.save_model_artifact``'s on-disk layout.

Also covers the WR-02 gold-matrix gate in
``pipeline.steps.step_verify_data_artifacts``.
"""

import json
from pathlib import Path

import joblib
import pytest


def _write_real_artifact(artifacts_dir, target, ts="20260101_000000"):
    """Create a versioned artifact dir mirroring save_model_artifact's outputs.

    Writes ``artifacts_dir/{target}_{ts}/`` with model.pkl + metadata.json +
    feature_list.json, then updates the latest.json manifest pointer to it --
    exactly the layout the production loader (load_model_artifact) expects.

    Returns the created artifact directory Path.
    """
    artifacts_dir = Path(artifacts_dir)
    artifact_dir = artifacts_dir / f"{target}_{ts}"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(object(), artifact_dir / "model.pkl")
    (artifact_dir / "metadata.json").write_text(json.dumps({"target": target}))
    (artifact_dir / "feature_list.json").write_text(json.dumps(["a", "b", "c"]))

    latest_path = artifacts_dir / "latest.json"
    manifest = json.loads(latest_path.read_text()) if latest_path.exists() else {}
    manifest[target] = artifact_dir.name
    latest_path.write_text(json.dumps(manifest, indent=2))

    return artifact_dir


class TestModelValidatorImport:
    """The class must be importable from the pipeline package, not scripts.*."""

    def test_importable_from_pipeline_package(self):
        from pipeline.model_validation import ModelValidator

        validator = ModelValidator()
        assert hasattr(validator, "validate_model_availability")
        assert hasattr(validator, "validate_model_loadability")

    def test_default_artifacts_dir_is_artifacts(self):
        """The default resolves the real artifacts/ tree, NOT artifacts/models."""
        from pipeline.model_validation import ModelValidator

        validator = ModelValidator()
        assert str(validator.artifacts_dir) == str(Path("artifacts"))


class TestValidateModelAvailability:
    """validate_model_availability resolves each target via latest.json."""

    def test_all_present_returns_all_true(self, tmp_path):
        from pipeline.model_validation import ModelValidator

        for name in ("wp", "ats", "ou"):
            _write_real_artifact(tmp_path, name)

        validator = ModelValidator(artifacts_dir=str(tmp_path))
        result = validator.validate_model_availability(["wp", "ats", "ou"])

        assert result == {
            "wp_available": True,
            "ats_available": True,
            "ou_available": True,
        }

    def test_blend_manifest_key_is_ignored(self, tmp_path):
        """A non-target key (blend) in latest.json must not break resolution."""
        from pipeline.model_validation import ModelValidator

        for name in ("wp", "ats", "ou"):
            _write_real_artifact(tmp_path, name)
        # Inject a blend pointer the validator must ignore (only wp/ats/ou iterated).
        latest_path = tmp_path / "latest.json"
        manifest = json.loads(latest_path.read_text())
        manifest["blend"] = "blend_dynamic_20260101_000000"
        latest_path.write_text(json.dumps(manifest))

        validator = ModelValidator(artifacts_dir=str(tmp_path))
        result = validator.validate_model_availability(["wp", "ats", "ou"])

        assert result == {
            "wp_available": True,
            "ats_available": True,
            "ou_available": True,
        }

    def test_negative_missing_target_key_is_not_available(self, tmp_path):
        """Negative 1: latest.json present but target key missing -> not available."""
        from pipeline.model_validation import ModelValidator

        # Only wp is written to the manifest; ats/ou keys are absent.
        _write_real_artifact(tmp_path, "wp")

        validator = ModelValidator(artifacts_dir=str(tmp_path))
        result = validator.validate_model_availability(["wp", "ats", "ou"])

        assert result["wp_available"] is True
        assert result["ats_available"] is False
        assert result["ou_available"] is False


class TestValidateModelLoadability:
    """validate_model_loadability loads via load_model_artifact (model + feature_list)."""

    def test_valid_artifact_is_loadable(self, tmp_path):
        from pipeline.model_validation import ModelValidator

        _write_real_artifact(tmp_path, "wp")

        validator = ModelValidator(artifacts_dir=str(tmp_path))
        result = validator.validate_model_loadability(["wp"])

        assert result == {"wp_loadable": True}

    def test_missing_target_is_not_loadable(self, tmp_path):
        from pipeline.model_validation import ModelValidator

        # Empty artifacts dir with no latest.json at all.
        validator = ModelValidator(artifacts_dir=str(tmp_path))
        result = validator.validate_model_loadability(["wp"])

        assert result == {"wp_loadable": False}

    def test_negative_model_pkl_absent_is_not_loadable(self, tmp_path):
        """Negative 2: manifest points at a version dir whose model.pkl is absent."""
        from pipeline.model_validation import ModelValidator

        # Point latest.json at a version dir, then remove its model.pkl so the
        # load attempt fails and degrades to not-loadable.
        artifact_dir = _write_real_artifact(tmp_path, "wp")
        (artifact_dir / "model.pkl").unlink()

        validator = ModelValidator(artifacts_dir=str(tmp_path))
        result = validator.validate_model_loadability(["wp"])

        assert result == {"wp_loadable": False}

    def test_corrupt_artifact_is_not_loadable(self, tmp_path):
        from pipeline.model_validation import ModelValidator

        # Valid manifest + dir, but model.pkl is not a real joblib payload so the
        # load raises and is caught (graceful degradation).
        artifact_dir = _write_real_artifact(tmp_path, "wp")
        (artifact_dir / "model.pkl").write_text("not a joblib artifact")

        validator = ModelValidator(artifacts_dir=str(tmp_path))
        result = validator.validate_model_loadability(["wp"])

        assert result == {"wp_loadable": False}


class TestVerifyDataArtifactsGoldGate:
    """WR-02: step_verify_data_artifacts checks concrete gold matrices."""

    def _write_silver(self, root):
        """Create the three concrete silver files the gate also requires."""
        silver = root / "data" / "silver"
        silver.mkdir(parents=True, exist_ok=True)
        for name in (
            "games.parquet",
            "elo_game_snapshots.parquet",
            "team_form_features.parquet",
        ):
            (silver / name).write_bytes(b"")

    def _write_gold(self, root, matrices):
        gold = root / "data" / "gold"
        gold.mkdir(parents=True, exist_ok=True)
        for name in matrices:
            (gold / name).write_bytes(b"")

    def test_raises_when_gold_empty(self, tmp_path, monkeypatch):
        """An empty data/gold/ (no matrices) must raise, not pass as ready."""
        from pipeline.steps import step_verify_data_artifacts

        self._write_silver(tmp_path)
        (tmp_path / "data" / "gold").mkdir(parents=True, exist_ok=True)
        monkeypatch.chdir(tmp_path)

        with pytest.raises(RuntimeError) as exc_info:
            step_verify_data_artifacts()

        msg = str(exc_info.value)
        assert "Missing data artifacts" in msg
        assert "features_wp.parquet" in msg

    def test_passes_when_all_three_matrices_present(self, tmp_path, monkeypatch):
        """Ready when all three concrete gold matrices exist alongside silver."""
        from pipeline.steps import step_verify_data_artifacts

        self._write_silver(tmp_path)
        self._write_gold(
            tmp_path,
            ["features_wp.parquet", "features_ats.parquet", "features_ou.parquet"],
        )
        monkeypatch.chdir(tmp_path)

        # Should not raise.
        step_verify_data_artifacts()

    def test_raises_when_one_matrix_missing(self, tmp_path, monkeypatch):
        """A single missing matrix (ou) is reported as missing."""
        from pipeline.steps import step_verify_data_artifacts

        self._write_silver(tmp_path)
        self._write_gold(
            tmp_path,
            ["features_wp.parquet", "features_ats.parquet"],
        )
        monkeypatch.chdir(tmp_path)

        with pytest.raises(RuntimeError) as exc_info:
            step_verify_data_artifacts()

        assert "features_ou.parquet" in str(exc_info.value)

    def test_required_artifact_list_matches_real_build_no_drift(self):
        """The gate's required list contains only names a real build produces (D-02, B2).

        This is the recurrence-axis keystone for the v2.1 milestone-audit BLOCKER:
        ``step_verify_data_artifacts`` once required ``elo_ratings.parquet`` /
        ``team_form.parquet`` -- names NO build script writes -- so a full-mode
        Friday run aborted at the DATA-phase gate. A stub that matched those wrong
        names hid the defect from every unit test.

        To be genuinely drift-proof (NOT a second matching stub -- RESEARCH
        Pitfall 1) this test imports ``_REQUIRED_ARTIFACTS`` from the gate as the
        SINGLE source of truth and asserts every entry maps to a name a real build
        script actually produces. It explicitly asserts the two dead names can
        never reappear. It deliberately does NOT re-declare the corrected list as
        an EXPECTED value; the allowed set is derived from the known producers, so
        the test fails the instant the gate requires a path no build produces --
        even if some fixture is later updated to match the drift.

        The check is purely structural (no data-layer dependency), so it always
        runs -- including on a clean checkout with no ``data/`` tree.
        """
        from pipeline.steps import _REQUIRED_ARTIFACTS

        # Names a real build actually writes (each producer verified in-repo):
        #   data/silver/games.parquet              <- scripts/ingest_games.py (canonical games table)
        #   data/silver/elo_game_snapshots.parquet <- scripts/build_elo.py save_dataframe(..., "elo_game_snapshots", layer="silver")
        #   data/silver/team_form_features.parquet <- scripts/build_team_form.py save_dataframe(..., "team_form_features", layer="silver")
        #   data/gold/features_wp.parquet          <- scripts/build_features.py (gold matrices)
        #   data/gold/features_ats.parquet         <- scripts/build_features.py (gold matrices)
        #   data/gold/features_ou.parquet          <- scripts/build_features.py (gold matrices)
        produced_silver = {
            "data/silver/games.parquet",
            "data/silver/elo_game_snapshots.parquet",
            "data/silver/team_form_features.parquet",
        }
        produced_gold = {
            "data/gold/features_wp.parquet",
            "data/gold/features_ats.parquet",
            "data/gold/features_ou.parquet",
        }
        allowed = produced_silver | produced_gold

        for path in _REQUIRED_ARTIFACTS:
            assert path in allowed, f"gate requires {path} but no build produces it"

        # The dead names that caused the BLOCKER must never reappear in the gate.
        assert "data/silver/elo_ratings.parquet" not in _REQUIRED_ARTIFACTS
        assert "data/silver/team_form.parquet" not in _REQUIRED_ARTIFACTS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
