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
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib

from utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# THE PERSISTED-PREPROCESSING CONTRACT (D33.1-R1, owner-ratified 2026-09-12,
# attributed to SPEC R6; Plan 33.1-10 Ruling T2).
#
# Until this existed, this module had NO preprocessing parameter, returned none, and
# the string `scaler` appeared nowhere in it. That absence WAS the contract gap: an
# estimator fitted on standardised features and served on unstandardised ones produces
# a PLAUSIBLE WRONG ANSWER rather than an error, so nothing fails and nobody looks.
# `wp_20260824_113325` is in exactly that state in production today.
#
# The contract binds artifacts saved UNDER it only (D33.1-R2). An artifact carrying
# neither of the two names below is served exactly as it was before -- byte-identical,
# zero change to current predictions -- and serving never REFUSES a legacy artifact.
# ---------------------------------------------------------------------------

#: The filename the fitted preprocessing object is written to, when one is given.
PREPROCESSING_FILENAME: str = "preprocessing.pkl"

#: The metadata key carrying the ATS / O/U converter parameters as JSON.
#:
#: JSON rather than a second pickle, deliberately: three fields describe those
#: converters completely and reconstruct them exactly, a JSON record is readable a year
#: from now by a human auditing why a cover probability was what it was, and it does not
#: widen the `joblib.load` deserialisation surface.
CONVERTER_PARAMS_METADATA_KEY: str = "converter_params"


def _atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    """Write a JSON manifest atomically (temp file in the same dir, then os.replace).

    ``Path.write_text`` truncates the target and then writes; an interruption (crash, disk
    full, Ctrl-C) between truncate and flush leaves a 0-byte or partial file. For
    ``latest.json`` -- the ONLY production model-swap surface -- a partial write takes
    production down (every consumer fails to parse it) with no backup. Serializing to a temp
    file in the SAME directory and ``os.replace``-ing it over the target is an atomic rename on
    the same filesystem, so a reader sees either the old complete manifest or the new complete
    manifest, never a truncated one (WR-02).

    Args:
        path: Destination JSON path (overwritten atomically).
        data: The manifest dict to serialize.
    """
    payload = json.dumps(data, indent=2)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(payload)
        # Atomic rename on the same filesystem (Path.replace -> os.replace under the hood).
        Path(tmp).replace(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def save_model_artifact(
    model: Any,
    target: str,
    metadata: dict,
    feature_list: list[str],
    calibrator: Any | None = None,
    preprocessing: Any | None = None,
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
    - preprocessing.pkl: Fitted preprocessing object (if provided)
    - {target}_params.json: Tuning parameters sidecar (if best_params provided)

    WHY ``preprocessing`` EXISTS, which is the load-bearing part (D33.1-R1,
    owner-ratified 2026-09-12, attributed to SPEC R6). A WP estimator fitted on
    STANDARDISED features and served on UNSTANDARDISED ones produces a plausible WRONG
    answer rather than an error: nothing raises, nothing is logged, and the served
    probability is simply not the one the model was fitted to give. No amount of
    convention prevents that, because a convention is a thing a future edit can forget.
    The only structural defence is that the transform and the estimator are ONE PERSISTED
    OBJECT -- for WP the value passed here IS the four-step Pipeline whose final step is
    the estimator, so there is no way to load the estimator without its scaler.

    The deployed ``wp_20260824_113325`` predates this and is in exactly the state
    described above. It is NAMED rather than repaired (D33.1-R2): repairing it without a
    re-fit would change live predictions, which this phase's SPEC forbids. Phase 33's
    Wave 15 resolves it by shipping a replacement under this contract.

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
        preprocessing: Optional fitted preprocessing object, persisted BESIDE the model
            so the serving path can apply the same transform the model was fitted under.
            Omitting it reproduces the pre-D33.1-R1 directory exactly, file for file.
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

    # Save preprocessing if provided (D33.1-R1). The same OPTIONAL-FILE mechanism the
    # calibrator already uses, deliberately: the contract gains one key rather than a new
    # mechanism, and an artifact saved without it is byte-identical in file set to what
    # this function produced before the parameter existed.
    if preprocessing is not None:
        joblib.dump(preprocessing, artifact_dir / PREPROCESSING_FILENAME)

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
        has_preprocessing=preprocessing is not None,
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
    # Atomic write: latest.json is the sole production swap surface, so a partial write
    # (crash/disk-full mid-write) must never corrupt it (WR-02).
    _atomic_write_json(latest_path, manifest)


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
        preprocessing (or None), params (or None), artifact_dir.

        ``preprocessing`` is None for every artifact saved before D33.1-R1, and that is
        the LEGACY case rather than an error: serving falls through to the pre-contract
        path for it and never refuses it (D33.1-R2).

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

    # Load preprocessing if it exists (D33.1-R1). Absent is the LEGACY case, not an
    # error: every artifact saved before this contract has no such file, and returning
    # None is what lets the serving path keep those byte-identical (D33.1-R2).
    preprocessing_path = artifact_dir / PREPROCESSING_FILENAME
    preprocessing = (
        joblib.load(preprocessing_path) if preprocessing_path.exists() else None
    )

    # Load params sidecar if it exists
    params_path = artifact_dir / f"{target}_params.json"
    params = json.loads(params_path.read_text()) if params_path.exists() else None

    logger.info(
        "Loaded model artifact",
        target=target,
        version=version,
        n_features=len(feature_list),
        has_calibrator=calibrator is not None,
        has_preprocessing=preprocessing is not None,
        has_params=params is not None,
    )

    return {
        "model": model,
        "metadata": metadata,
        "feature_list": feature_list,
        "calibrator": calibrator,
        "preprocessing": preprocessing,
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
