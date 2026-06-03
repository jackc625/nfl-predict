"""Versioned model artifact storage.

Provides:
- save_model_artifact: Save model, metadata, features, and optional calibrator
- load_model_artifact: Load artifact by target and optional version
- get_latest_artifact_path: Get path to latest artifact for a target

Artifact directory structure:
    artifacts/
        wp_20260319_180000/
            model.pkl
            metadata.json
            feature_list.json
            calibrator.pkl (optional)
        ats_20260319_180000/
            ...
        latest.json  -- manifest pointing to current production models
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib

from utils import get_logger

logger = get_logger(__name__)


def save_model_artifact(
    model: Any,
    target: str,
    metadata: dict,
    feature_list: list[str],
    calibrator: Any | None = None,
    artifacts_dir: Path = Path("artifacts"),
    best_params: dict | None = None,
    tuning_metadata: dict | None = None,
    update_latest: bool = False,
) -> Path:
    """Save a trained model with versioned directory structure.

    Creates artifacts/{target}_{timestamp}/ containing:
    - model.pkl: Serialized model
    - metadata.json: Training metadata (metrics, params, etc.)
    - feature_list.json: Ordered list of feature names
    - calibrator.pkl: Calibrator model (if provided)
    - {target}_params.json: Tuning parameters sidecar (if best_params provided)

    Updating the latest.json manifest is now OPT-IN via update_latest (D24-08).
    By default the manifest is NOT touched: writing artifacts/latest.json is a
    production deploy, and the only sanctioned production swapper is
    update_manifest (invoked by scripts/promote_models on a passing gate). A
    plain train run -- including a staging re-fit -- must never auto-swap the
    served model. Pass update_latest=True only when the caller deliberately
    wants this artifact registered as the latest for its target (e.g. a
    self-contained staging dir or a test that immediately loads the artifact
    back via load_model_artifact).

    Args:
        model: Trained model object (sklearn, xgboost, etc.).
        target: Model target type ("wp", "ats", "ou").
        metadata: Training metadata dictionary.
        feature_list: Ordered list of feature column names.
        calibrator: Optional fitted calibrator.
        artifacts_dir: Root directory for artifacts.
        best_params: Optional dict of tuned hyperparameters.
            When provided, a JSON sidecar file is saved alongside the model.
        tuning_metadata: Optional dict of tuning study metadata
            (study name, n_trials, optimization metric, etc.).
        update_latest: When True, register this artifact in latest.json for its
            target via the per-key update_manifest helper. Defaults to False so
            no train run auto-swaps production (D24-08); production swaps go
            through update_manifest / scripts/promote_models.

    Returns:
        Path to the created artifact directory.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_dir = artifacts_dir / f"{target}_{timestamp}"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = artifact_dir / "model.pkl"
    joblib.dump(model, model_path)

    # Save calibrator if provided
    if calibrator is not None:
        calibrator_path = artifact_dir / "calibrator.pkl"
        joblib.dump(calibrator, calibrator_path)

    # Save metadata
    metadata_path = artifact_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str))

    # Save feature list
    feature_list_path = artifact_dir / "feature_list.json"
    feature_list_path.write_text(json.dumps(feature_list, indent=2))

    # Save params sidecar atomically with model (per D-13)
    if best_params is not None:
        params_data = {
            "best_params": best_params,
            "tuning_metadata": tuning_metadata or {},
        }
        params_path = artifact_dir / f"{target}_params.json"
        params_path.write_text(json.dumps(params_data, indent=2, default=str))

    # Update latest.json manifest only when explicitly requested (D24-08).
    # A bare save no longer auto-swaps production; the sole production swapper
    # is update_manifest (called by scripts/promote_models on a passing gate).
    if update_latest:
        update_manifest(target, artifact_dir.name, artifacts_dir)

    logger.info(
        "Saved model artifact",
        target=target,
        artifact_dir=str(artifact_dir),
        n_features=len(feature_list),
        has_calibrator=calibrator is not None,
        has_params=best_params is not None,
        update_latest=update_latest,
    )

    return artifact_dir


def update_manifest(
    target: str,
    version: str,
    artifacts_dir: Path = Path("artifacts"),
) -> None:
    """Register an artifact version as the latest for a target (sole swapper).

    This is the ONLY sanctioned writer of production artifacts/latest.json
    (D24-08). It performs a per-key update on the existing manifest --
    manifest[target] = version -- so every other key survives untouched. In
    particular the "blend" pointer and any non-promoted target keep their
    current value (Pitfall 4: never rewrite the whole manifest from a subset of
    passing targets). If latest.json does not yet exist, an empty manifest is
    created.

    Args:
        target: Model target type ("wp", "ats", "ou").
        version: Artifact directory name to point this target at
            (e.g. "wp_20260327_114739").
        artifacts_dir: Root directory containing latest.json.
    """
    latest_path = artifacts_dir / "latest.json"
    manifest = json.loads(latest_path.read_text()) if latest_path.exists() else {}
    manifest[target] = version
    latest_path.write_text(json.dumps(manifest, indent=2))


def load_model_artifact(
    target: str,
    version: str | None = None,
    artifacts_dir: Path = Path("artifacts"),
) -> dict[str, Any]:
    """Load a model artifact by target and optional version.

    If version is None, loads the latest artifact for the target
    from the latest.json manifest.

    Args:
        target: Model target type ("wp", "ats", "ou").
        version: Specific artifact directory name. If None, uses latest.
        artifacts_dir: Root directory for artifacts.

    Returns:
        Dict with keys: model, metadata, feature_list, calibrator (or None),
        artifact_dir.

    Raises:
        FileNotFoundError: If artifact directory or latest.json not found.
        KeyError: If target not found in latest.json manifest.
    """
    if version is None:
        latest_path = artifacts_dir / "latest.json"
        if not latest_path.exists():
            msg = f"latest.json not found in {artifacts_dir}"
            raise FileNotFoundError(msg)
        manifest = json.loads(latest_path.read_text())
        if target not in manifest:
            msg = f"Target '{target}' not found in latest.json manifest"
            raise KeyError(msg)
        version = manifest[target]

    artifact_dir = artifacts_dir / version

    if not artifact_dir.exists():
        msg = f"Artifact directory not found: {artifact_dir}"
        raise FileNotFoundError(msg)

    # Load model
    model = joblib.load(artifact_dir / "model.pkl")

    # Load metadata
    metadata = json.loads((artifact_dir / "metadata.json").read_text())

    # Load feature list
    feature_list = json.loads((artifact_dir / "feature_list.json").read_text())

    # Load calibrator if it exists
    calibrator_path = artifact_dir / "calibrator.pkl"
    calibrator = joblib.load(calibrator_path) if calibrator_path.exists() else None

    # Load params sidecar if it exists
    params_path = artifact_dir / f"{target}_params.json"
    params = json.loads(params_path.read_text()) if params_path.exists() else None

    logger.info(
        "Loaded model artifact",
        target=target,
        version=version,
        n_features=len(feature_list),
        has_calibrator=calibrator is not None,
        has_params=params is not None,
    )

    return {
        "model": model,
        "metadata": metadata,
        "feature_list": feature_list,
        "calibrator": calibrator,
        "params": params,
        "artifact_dir": artifact_dir,
    }


def get_latest_artifact_path(
    target: str,
    artifacts_dir: Path = Path("artifacts"),
) -> Path | None:
    """Get the path to the latest artifact directory for a target.

    Args:
        target: Model target type ("wp", "ats", "ou").
        artifacts_dir: Root directory for artifacts.

    Returns:
        Path to the latest artifact directory, or None if not found.
    """
    latest_path = artifacts_dir / "latest.json"
    if not latest_path.exists():
        return None

    manifest = json.loads(latest_path.read_text())
    if target not in manifest:
        return None

    return artifacts_dir / manifest[target]
