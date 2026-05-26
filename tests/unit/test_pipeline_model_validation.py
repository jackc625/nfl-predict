"""Tests for the relocated in-package ModelValidator (pipeline.model_validation).

These cover the two methods the Friday orchestrator actually calls:
validate_model_availability and validate_model_loadability. They assert the
return-key shape ({name}_available / {name}_loadable booleans) and the
availability/loadability semantics against joblib artifact files.
"""

import joblib
import pytest


def _write_valid_model(path):
    """Write a joblib artifact that has the three required components."""
    joblib.dump(
        {
            "model": object(),
            "scaler": object(),
            "feature_names": ["a", "b", "c"],
        },
        path,
    )


def _write_incomplete_model(path):
    """Write a joblib artifact missing a required component (no scaler)."""
    joblib.dump(
        {
            "model": object(),
            "feature_names": ["a", "b", "c"],
        },
        path,
    )


class TestModelValidatorImport:
    """The class must be importable from the pipeline package, not scripts.*."""

    def test_importable_from_pipeline_package(self):
        from pipeline.model_validation import ModelValidator

        validator = ModelValidator()
        assert hasattr(validator, "validate_model_availability")
        assert hasattr(validator, "validate_model_loadability")

    def test_default_models_path_is_artifacts_models(self):
        from pipeline.model_validation import ModelValidator

        validator = ModelValidator()
        assert str(validator.models_path) == str(
            __import__("pathlib").Path("artifacts/models")
        )


class TestValidateModelAvailability:
    """validate_model_availability checks {name}_model.joblib existence."""

    def test_all_present_returns_all_true(self, tmp_path):
        from pipeline.model_validation import ModelValidator

        for name in ("wp", "ats", "ou"):
            _write_valid_model(tmp_path / f"{name}_model.joblib")

        validator = ModelValidator(models_path=str(tmp_path))
        result = validator.validate_model_availability(["wp", "ats", "ou"])

        assert result == {
            "wp_available": True,
            "ats_available": True,
            "ou_available": True,
        }

    def test_missing_model_returns_false_for_that_key(self, tmp_path):
        from pipeline.model_validation import ModelValidator

        _write_valid_model(tmp_path / "wp_model.joblib")
        # ats and ou intentionally absent

        validator = ModelValidator(models_path=str(tmp_path))
        result = validator.validate_model_availability(["wp", "ats", "ou"])

        assert result["wp_available"] is True
        assert result["ats_available"] is False
        assert result["ou_available"] is False


class TestValidateModelLoadability:
    """validate_model_loadability checks the joblib loads with required keys."""

    def test_valid_artifact_is_loadable(self, tmp_path):
        from pipeline.model_validation import ModelValidator

        _write_valid_model(tmp_path / "wp_model.joblib")

        validator = ModelValidator(models_path=str(tmp_path))
        result = validator.validate_model_loadability(["wp"])

        assert result == {"wp_loadable": True}

    def test_missing_file_is_not_loadable(self, tmp_path):
        from pipeline.model_validation import ModelValidator

        validator = ModelValidator(models_path=str(tmp_path))
        result = validator.validate_model_loadability(["wp"])

        assert result == {"wp_loadable": False}

    def test_incomplete_artifact_is_not_loadable(self, tmp_path):
        from pipeline.model_validation import ModelValidator

        _write_incomplete_model(tmp_path / "wp_model.joblib")

        validator = ModelValidator(models_path=str(tmp_path))
        result = validator.validate_model_loadability(["wp"])

        assert result == {"wp_loadable": False}

    def test_corrupt_artifact_is_not_loadable(self, tmp_path):
        from pipeline.model_validation import ModelValidator

        # Not a valid joblib file -- load should raise and be caught.
        (tmp_path / "wp_model.joblib").write_text("not a joblib artifact")

        validator = ModelValidator(models_path=str(tmp_path))
        result = validator.validate_model_loadability(["wp"])

        assert result == {"wp_loadable": False}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
