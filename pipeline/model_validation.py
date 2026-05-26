"""In-package model-validation utility for the Friday pipeline.

Relocated from the former model-validation script so the live orchestrator
(``pipeline/steps.py::step_validate_models``) no longer depends on an untracked
file. Only the two methods the orchestrator actually calls are kept here --
``validate_model_availability`` and ``validate_model_loadability``; the source
script's comprehensive/metadata/performance/report helpers and its CLI entry
point are intentionally NOT relocated (they had no canonical caller).

Logger acquisition follows the ``pipeline/`` package convention
(``get_logger`` from ``utils.logging_config``); the source script's top-level
import shim and project-root path insertion are deliberately dropped because
packages do not need them.
"""

from pathlib import Path

import joblib

from utils.logging_config import get_logger

logger = get_logger(__name__)


class ModelValidator:
    """Validate trained models for production readiness."""

    def __init__(self, models_path: str = "artifacts/models"):
        self.models_path = Path(models_path)
        self.validation_results = {}

    def validate_model_availability(self, models: list[str]) -> dict[str, bool]:
        """Check if required models are available."""
        logger.info(f"Validating availability of models: {models}")

        results = {}
        for model_name in models:
            model_file = self.models_path / f"{model_name}_model.joblib"
            results[f"{model_name}_available"] = model_file.exists()

            if not model_file.exists():
                logger.error(f"Model file not found: {model_file}")
            else:
                logger.info(f"Model found: {model_name}")

        return results

    def validate_model_loadability(self, models: list[str]) -> dict[str, bool]:
        """Check if models can be loaded successfully."""
        logger.info("Validating model loadability")

        results = {}
        for model_name in models:
            model_file = self.models_path / f"{model_name}_model.joblib"

            if not model_file.exists():
                results[f"{model_name}_loadable"] = False
                continue

            try:
                model_data = joblib.load(model_file)

                # Check required components
                required_components = ["model", "scaler", "feature_names"]
                has_all_components = all(
                    key in model_data for key in required_components
                )

                if not has_all_components:
                    logger.error(f"Model {model_name} missing required components")
                    results[f"{model_name}_loadable"] = False
                else:
                    logger.info(f"Model {model_name} loaded successfully")
                    results[f"{model_name}_loadable"] = True

            except Exception as e:
                logger.error(f"Error loading model {model_name}: {e}")
                results[f"{model_name}_loadable"] = False

        return results
