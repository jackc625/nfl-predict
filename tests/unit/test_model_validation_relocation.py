"""Regression guard for the validate-models rewire (Plan 19-01).

The Friday orchestrator's ``step_validate_models`` used to lazy-import
``ModelValidator`` from ``scripts.validate_models`` -- an untracked file on the
CLEAN-02 delete list. Plan 19-01 relocated the class into the ``pipeline``
package and repointed the import. These static checks fail if a future edit
reintroduces a dependency on ``scripts.validate_models``, which would re-break
the live pipeline once the script is deleted (Plan 03).

Intentionally dependency-free: no model artifacts, no joblib loads.
"""

import inspect

import pipeline.steps


def test_model_validator_importable_from_pipeline_package():
    """The relocated ModelValidator must live in the pipeline package."""
    from pipeline.model_validation import ModelValidator

    validator = ModelValidator()
    assert hasattr(validator, "validate_model_availability")
    assert hasattr(validator, "validate_model_loadability")


def test_steps_module_does_not_import_scripts_validate_models():
    """pipeline.steps must not reference scripts.validate_models anywhere."""
    source = inspect.getsource(pipeline.steps)
    assert "scripts.validate_models" not in source, (
        "pipeline.steps reintroduced a scripts.validate_models dependency; "
        "step_validate_models must import ModelValidator from "
        "pipeline.model_validation (Plan 19-01 rewire)."
    )


def test_validate_models_step_imports_from_pipeline():
    """step_validate_models must import ModelValidator from the pipeline package."""
    source = inspect.getsource(pipeline.steps.step_validate_models)
    assert "from pipeline.model_validation import ModelValidator" in source
