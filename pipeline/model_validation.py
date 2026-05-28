"""In-package model-validation utility for the Friday pipeline.

Relocated from the former model-validation script so the live orchestrator
(``pipeline/steps.py::step_validate_models``) no longer depends on an untracked
file. Only the two methods the orchestrator actually calls are kept here --
``validate_model_availability`` and ``validate_model_loadability``; the source
script's comprehensive/metadata/performance/report helpers and its CLI entry
point are intentionally NOT relocated (they had no canonical caller).

Artifact convention (CR-01 fix, FIX-01 / D-03): the deployed models are the
versioned ``artifacts/{target}_{timestamp}/`` directories named in
``artifacts/latest.json`` -- the same pointer the prediction pipeline loads via
``load_model_artifact``. The former ``artifacts/models/{name}_model.joblib``
convention with keys ``model/scaler/feature_names`` was never produced by
training; validating it caused a correctly-trained Friday run to abort at the
``critical=True`` ``step_validate_models`` gate. Resolution mirrors
``api/routes/health.py::_resolve_active_model_files`` (OPS-04, commit 78288a5)
and reuses ``models.artifacts`` (pipeline/ MAY import models/; only api/ is the
UIAP-01 stdlib-constrained layer).

Logger acquisition follows the ``pipeline/`` package convention
(``get_logger`` from ``utils.logging_config``); the source script's top-level
import shim and project-root path insertion are deliberately dropped because
packages do not need them.
"""

from pathlib import Path

from models.artifacts import get_latest_artifact_path, load_model_artifact
from utils.logging_config import get_logger

logger = get_logger(__name__)


class ModelValidator:
    """Validate trained models for production readiness.

    Resolves each target via ``artifacts/latest.json`` (the real training
    convention) rather than fixed-name stubs under ``artifacts/models/``.
    """

    def __init__(self, artifacts_dir: str = "artifacts"):
        self.artifacts_dir = Path(artifacts_dir)
        self.validation_results = {}

    def validate_model_availability(self, models: list[str]) -> dict[str, bool]:
        """Check if required models are available.

        A model is available when ``artifacts/latest.json`` names a version
        directory for the target AND that directory's ``model.pkl`` exists.
        The manifest's non-target keys (e.g. ``blend``) are ignored because
        only the requested targets are iterated.
        """
        logger.info(f"Validating availability of models: {models}")

        results = {}
        for model_name in models:
            artifact_dir = get_latest_artifact_path(
                model_name, artifacts_dir=self.artifacts_dir
            )
            model_file = artifact_dir / "model.pkl" if artifact_dir else None
            available = model_file is not None and model_file.exists()
            results[f"{model_name}_available"] = available

            if not available:
                logger.error(f"Model artifact not found for target: {model_name}")
            else:
                logger.info(f"Model found: {model_name} -> {artifact_dir.name}")

        return results

    def validate_model_loadability(self, models: list[str]) -> dict[str, bool]:
        """Check if models can be loaded successfully.

        A model is loadable when ``load_model_artifact`` returns a dict that
        carries both the ``model`` and ``feature_list`` keys (the real artifact
        contract). Any load failure degrades the result to ``False`` rather than
        propagating -- the broad catch is intentional and scoped via the
        ``BLE001`` per-file ignore (mirrors ``pipeline/health.py``).
        """
        logger.info("Validating model loadability")

        results = {}
        for model_name in models:
            try:
                artifact = load_model_artifact(
                    model_name, artifacts_dir=self.artifacts_dir
                )
                has_required = "model" in artifact and "feature_list" in artifact

                if not has_required:
                    logger.error(f"Model {model_name} missing required components")
                    results[f"{model_name}_loadable"] = False
                else:
                    logger.info(f"Model {model_name} loaded successfully")
                    results[f"{model_name}_loadable"] = True

            except Exception as e:
                logger.error(f"Error loading model {model_name}: {e}")
                results[f"{model_name}_loadable"] = False

        return results
