"""Versioned model artifact storage.

Provides:
- save_model_artifact: Save model, metadata, features, and optional calibrator
- load_model_artifact: Load artifact by target and optional version
- get_latest_artifact_path: Get path to latest artifact for a target
- update_manifest: The PER-TARGET production swapper (one manifest key per call)
- replace_manifest: The BATCHED production swapper (all four keys, one write, SPEC R13)
- ARTIFACT_VALIDATORS: The per-kind loadability registry replace_manifest validates through

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
from collections.abc import Callable, Mapping
from dataclasses import dataclass
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

#: Metadata key recording that ``model.pkl`` and ``preprocessing.pkl`` were serialized
#: from the SAME object (code review WR-13).
#:
#: For WP under D33.1-R1 ``self.preprocessing = model`` -- one four-step Pipeline whose
#: third step is the scaler and whose fourth is the estimator -- so both files hold the
#: same estimator. On load they became two DISTINCT deserialized objects with nothing
#: asserting they agree, and the serving paths split: ``predict_games`` uses
#: ``preprocessing`` for WP (prediction_pipeline.py:794-798) while backtest scoring, the
#: deploy gate and promote_models all use ``model``. Today both answer identically; the
#: risk is a future path that writes or repairs one file and not the other, after which
#: the two serving routes diverge with NO error -- the silent divergence D33.1-R1 exists
#: to make structurally impossible.
#:
#: When this key is True, ``load_model_artifact`` returns ONE object under both names.
#: It is FALSE, and both files are loaded independently, for a genuinely SPLIT artifact
#: (a bare estimator in ``model.pkl`` with its transform beside it), which is a supported
#: and different shape. Artifacts saved before this key exists carry no flag and take the
#: legacy path unchanged, so nothing on disk today changes behaviour.
PREPROCESSING_IS_MODEL_METADATA_KEY: str = "preprocessing_is_model"

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
            # WR-13: flush the application buffer AND the OS buffer before the rename.
            # Without this the atomicity guarantee held against a crashed PROCESS but not
            # against power loss: os.replace could land while the temp file's contents
            # were still only in the page cache, leaving a renamed but empty latest.json.
            f.flush()
            os.fsync(f.fileno())
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
    production deploy, and the only sanctioned production swappers are
    update_manifest (invoked by scripts/promote_models on a passing gate) and
    its batched sibling replace_manifest (the four-artifact swap, SPEC R13). A
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

    # Save metadata.
    #
    # WR-13: the flag is recorded rather than the second write being skipped. Both files
    # still exist, so the artifact FILE SET is unchanged and D33.1-R1's "inseparable on
    # the way to disk" property is untouched -- what changes is that the LOAD path can
    # now tell a one-object artifact from a genuinely split one, instead of silently
    # producing two copies that nothing compares. A copy of the caller's dict is written
    # so recording this cannot mutate a trainer's own metadata.
    metadata_path = artifact_dir / "metadata.json"
    metadata_to_write = dict(metadata)
    metadata_to_write[PREPROCESSING_IS_MODEL_METADATA_KEY] = (
        preprocessing is not None and preprocessing is model
    )
    metadata_path.write_text(json.dumps(metadata_to_write, indent=2, default=str))

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
    """Register an artifact version as the latest for a target (per-target swapper).

    This is the sanctioned PER-TARGET writer of production artifacts/latest.json
    (D24-08); its batched sibling, :func:`replace_manifest`, installs a whole
    four-artifact bundle in one write (SPEC R13, Plan 33.2-25) and is the only
    other writer. It performs a per-key update on the existing manifest --
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
    if metadata.get(PREPROCESSING_IS_MODEL_METADATA_KEY):
        # ONE object under both names (WR-13). The artifact's own metadata records that
        # these two files were serialized from the same estimator, so deserializing the
        # second one would produce a duplicate that the two serving routes -- predict_games
        # reads `preprocessing` for WP, everything else reads `model` -- could later
        # diverge across with no error. Repairing one file alone is not a supported
        # operation, and this flag is what makes that explicit rather than implicit.
        preprocessing = model
    else:
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


# ---------------------------------------------------------------------------
# THE BATCHED FOUR-POINTER SWAP (Plan 33.2-25, SPEC R13)
#
# ``update_manifest`` above is per-target BY DESIGN: it writes one key and leaves every
# other key untouched, which is exactly right for the gate path, where only the targets
# that passed may move. Installing a whole corrected bundle through it, though, takes FOUR
# writes -- and a failure between any two of them leaves ``latest.json`` naming a mix of
# old and new models, which SPEC R13 forbids ("written only once all four exist, never a
# mix"). ``replace_manifest`` is the batched writer that sits BESIDE it, not instead of it.
# ---------------------------------------------------------------------------

#: The four manifest keys a production bundle carries, in manifest order (SPEC R13).
BUNDLE_MANIFEST_KEYS: tuple[str, ...] = ("wp", "ats", "ou", "blend")

#: The three of them that name a MODEL artifact (the fourth names the blend).
_MODEL_MANIFEST_KEYS: tuple[str, ...] = ("wp", "ats", "ou")


class ArtifactProvenanceError(Exception):
    """An artifact loads, but what it records about itself cannot be installed.

    Raised by an :data:`ARTIFACT_VALIDATORS` entry when a model records a different target
    than the manifest key it is mapped to, when an artifact records no gold generation, or
    when a blend carries no provenance or no converter binding.
    """


class ArtifactBundleInvalidError(Exception):
    """A proposed production bundle failed validation, so ``latest.json`` was not written.

    Carries EVERY failure found, keyed by the manifest key (or by the name of the
    cross-artifact check) that failed, so one run names every defect rather than the first.

    WHY ``Exception`` AND NOT ``ValueError`` / ``KeyError``. Several call sites in this
    repository catch those types and degrade quietly (see
    ``models.blending.MarketProbabilityUnavailable``); a refused production swap degraded
    into "carry on" is the silent half-swap this refusal exists to prevent.

    Attributes:
        failures: ``{key: reason}`` for every failing key or check.
    """

    def __init__(self, failures: Mapping[str, str]) -> None:
        self.failures: dict[str, str] = dict(failures)
        lines = "\n".join(
            f"  - {key}: {reason}" for key, reason in self.failures.items()
        )
        super().__init__(
            "refusing the production swap: artifacts/latest.json was NOT written because "
            f"{len(self.failures)} check(s) failed:\n{lines}"
        )


@dataclass(frozen=True)
class ArtifactFacts:
    """What one validated artifact records about itself, read through its serving loader.

    Attributes:
        key: The manifest key the artifact was validated under.
        version: The artifact directory name.
        gold_generation_digest: The gold generation it was fitted on -- from a model's
            ``metadata.json``, or from the blend's ``blend_weights.json`` provenance.
        recorded_target: A model's recorded target (None for the blend).
        source_artifact_ids: The blend's ``{wp, ats, ou}`` source models (None for a model).
        market_probability_artifact_id: The converter the blend is bound to (None for a
            model).
    """

    key: str
    version: str
    gold_generation_digest: str
    recorded_target: str | None = None
    source_artifact_ids: dict[str, str] | None = None
    market_probability_artifact_id: str | None = None


#: ``(version, artifacts_dir) -> ArtifactFacts``. A validator LOADS the artifact through the
#: loader the serving path uses and raises if it cannot, so a directory that exists but does
#: not load is a failure, never a pass.
ArtifactValidator = Callable[[str, Path], ArtifactFacts]


def _model_validator(target: str) -> ArtifactValidator:
    """The validator for one model key: it must LOAD, and record this target and a gold.

    ``load_model_artifact`` is the serving path's own loader, so a directory holding a
    ``metadata.json`` but no ``model.pkl`` or ``feature_list.json`` fails here exactly as it
    would fail in production -- existence is not loadability.
    """

    def validate_model(version: str, artifacts_dir: Path) -> ArtifactFacts:
        # The metadata key is read from its ONE declaration, beside the code that writes it,
        # so the writer and this reader cannot drift apart by a typo. Imported lazily: the
        # training module is heavy and this module is imported by lightweight readers.
        from models.train import GOLD_GENERATION_DIGEST_METADATA_KEY

        loaded = load_model_artifact(
            target, version=version, artifacts_dir=artifacts_dir
        )
        metadata = loaded["metadata"]
        recorded_target = metadata.get("target")
        if recorded_target != target:
            msg = (
                f"{version!r} records target {recorded_target!r} but is mapped to "
                f"{target!r}: installing it would serve one target's model as another's."
            )
            raise ArtifactProvenanceError(msg)
        digest = metadata.get(GOLD_GENERATION_DIGEST_METADATA_KEY)
        if not digest:
            msg = (
                f"{version!r} records no {GOLD_GENERATION_DIGEST_METADATA_KEY!r} in its "
                "metadata.json: an artifact that cannot name the gold it was fitted on "
                "cannot be checked against the rest of the bundle."
            )
            raise ArtifactProvenanceError(msg)
        return ArtifactFacts(
            key=target,
            version=version,
            gold_generation_digest=str(digest),
            recorded_target=str(recorded_target),
        )

    return validate_model


def _validate_blend_artifact(version: str, artifacts_dir: Path) -> ArtifactFacts:
    """The blend validator: it must LOAD through ``MarketBlender.from_artifacts``.

    A blend directory holds ONLY ``blend_weights.json`` (``save_blend_artifacts`` writes
    nothing else and ``from_artifacts`` opens nothing else), so its provenance is read from
    that one payload -- never from a ``metadata.json`` invented for a validator to open,
    which would give the blend two files that could disagree about its own provenance.

    ``from_artifacts`` already refuses the retired ``dynamic`` section
    (``RetiredDynamicBlendError``, D33.2-10) and a converter binding whose directory is
    absent or whose slope disagrees (``MarketProbabilityBindingError``). What it tolerates
    for backwards compatibility -- a payload with no provenance, or no binding at all -- is
    refused HERE: neither can be installed as the blend that serves.
    """
    # Imported lazily: the blender pulls in pandas and scipy, and this module is imported by
    # lightweight readers that never swap anything.
    from models.blending import MarketBlender

    blender = MarketBlender.from_artifacts(artifacts_dir, version=version)
    provenance = blender.provenance
    if provenance is None or not provenance.gold_generation_digest:
        msg = (
            f"{version!r} carries no provenance in its blend_weights.json, so it cannot name "
            "the gold or the source models it was tuned on."
        )
        raise ArtifactProvenanceError(msg)
    if blender.market_probability_artifact_id is None:
        msg = (
            f"{version!r} is bound to no converter, so it could never convert a pre-lock "
            "spread and could never serve win probability."
        )
        raise ArtifactProvenanceError(msg)
    return ArtifactFacts(
        key="blend",
        version=version,
        gold_generation_digest=provenance.gold_generation_digest,
        source_artifact_ids=dict(provenance.source_artifact_ids),
        market_probability_artifact_id=blender.market_probability_artifact_id,
    )


#: ONE per-kind loadability registry, keyed by manifest key. A model artifact and a blend
#: artifact do not share a required-file set (``model.pkl`` + ``metadata.json`` +
#: ``feature_list.json`` against ``blend_weights.json`` alone), which is why a single
#: "has a metadata.json" rule could not validate a four-artifact bundle -- it would have
#: rejected every blend (reviews round ``f924749``, Codex HIGH).
ARTIFACT_VALIDATORS: dict[str, ArtifactValidator] = {
    "wp": _model_validator("wp"),
    "ats": _model_validator("ats"),
    "ou": _model_validator("ou"),
    "blend": _validate_blend_artifact,
}


def _unsafe_version_reason(version: object) -> str | None:
    """Why *version* is not a plain directory name under the artifacts root, or None."""
    if not isinstance(version, str) or not version:
        return f"{version!r} is not a non-empty artifact directory name"
    if version in (".", "..") or Path(version).name != version:
        return (
            f"{version!r} is not a plain directory name: a manifest pointer names a "
            "directory directly under the artifacts root and may not reach outside it"
        )
    return None


def _add_failure(failures: dict[str, str], key: str, reason: str) -> None:
    """Record *reason* under *key*, keeping any reason already recorded there."""
    failures[key] = f"{failures[key]}; {reason}" if key in failures else reason


def _cross_check_bundle(
    mapping: Mapping[str, str],
    facts: Mapping[str, ArtifactFacts],
    artifacts_dir: Path,
) -> dict[str, str]:
    """The checks no single artifact can make about itself. Returns ``{check: reason}``."""
    failures: dict[str, str] = {}

    digests = {key: fact.gold_generation_digest for key, fact in facts.items()}
    if len(set(digests.values())) != 1:
        recorded = ", ".join(f"{key}={digest}" for key, digest in digests.items())
        _add_failure(
            failures,
            "gold_generation_digest",
            "the bundle's artifacts were fitted on DIFFERENT golds "
            f"({recorded}); four artifacts on two golds is the mixed manifest R13 forbids, "
            "one level down",
        )

    blend = facts["blend"]
    expected_sources = {key: mapping[key] for key in _MODEL_MANIFEST_KEYS}
    if blend.source_artifact_ids != expected_sources:
        _add_failure(
            failures,
            "blend",
            f"{blend.version!r} was tuned on source models {blend.source_artifact_ids}, "
            f"but this bundle installs {expected_sources}: the blend would serve beside "
            "models whose predictions it never saw",
        )

    converter_id = blend.market_probability_artifact_id
    if converter_id is None or not (artifacts_dir / converter_id).is_dir():
        _add_failure(
            failures,
            "blend",
            f"its converter {converter_id!r} does not resolve to a directory under "
            f"{artifacts_dir}",
        )

    return failures


def replace_manifest(
    mapping: Mapping[str, str],
    artifacts_dir: Path = Path("artifacts"),
) -> None:
    """Install a whole four-artifact bundle into ``latest.json`` in ONE atomic write.

    WHY THIS EXISTS. ``update_manifest`` is per-target by design, so four pointers take
    four writes, and a failure between them leaves ``latest.json`` naming a mix of old and
    new artifacts -- which SPEC R13 forbids. This validates the WHOLE bundle first and only
    then writes once.

    WHY ``scripts/promote_models --promote`` IS NOT REUSED. It runs the per-target deploy
    gate, and R13 removes the gate for this swap by owner ruling: the corrected artifacts
    replace today's UNCONDITIONALLY, and no pre-correction model or gate baseline is used
    as a comparator.

    The order is the point of the function:

    1. Every key must have a registered validator and every one of the four bundle keys
       must be present. Every value is validated through :data:`ARTIFACT_VALIDATORS`, which
       LOADS it through the serving path's own loader. Every failure is collected.
    2. The collected facts are cross-checked: all four gold generation digests EQUAL, the
       blend's source models EQUAL to the three model ids in this same mapping, and the
       blend's converter resolving to a directory.
    3. Only then is the existing manifest read, the mapping applied over it -- keys NOT in
       the mapping survive with their current values, the whole-manifest-rewrite-from-a-
       subset failure ``update_manifest``'s docstring warns about (Pitfall 4) -- and the
       result written ONCE through :func:`_atomic_write_json`, so a crash leaves either the
       old manifest or the new one, never a partial file.

    Args:
        mapping: ``{manifest_key: artifact_directory_name}`` for all four bundle keys.
        artifacts_dir: Root directory containing the artifacts and ``latest.json``.

    Raises:
        ArtifactBundleInvalidError: naming EVERY failing key or check. ``latest.json`` is
            not touched.
    """
    failures: dict[str, str] = {}

    for key in mapping:
        if key not in ARTIFACT_VALIDATORS:
            _add_failure(
                failures,
                key,
                f"no validator is registered for manifest key {key!r} (registered: "
                f"{sorted(ARTIFACT_VALIDATORS)}), so it cannot be proven loadable",
            )
    for key in BUNDLE_MANIFEST_KEYS:
        if key not in mapping:
            _add_failure(
                failures,
                key,
                "missing from the mapping: latest.json is written only once ALL FOUR "
                "artifacts exist (SPEC R13)",
            )

    facts: dict[str, ArtifactFacts] = {}
    for key, version in mapping.items():
        validator = ARTIFACT_VALIDATORS.get(key)
        if validator is None:
            continue
        unsafe = _unsafe_version_reason(version)
        if unsafe is not None:
            _add_failure(failures, key, unsafe)
            continue
        try:
            facts[key] = validator(version, artifacts_dir)
        # Deliberately broad, and NOT a swallow: a loader can fail in many ways (a missing
        # file, a bad pickle, a malformed payload, a refused binding), every one of them is
        # recorded here by type and message, and all of them are re-raised together below.
        # Narrowing this would turn an unlisted failure into a first-only crash that hides
        # the other keys' defects.
        except Exception as exc:  # noqa: BLE001 - every failure is COLLECTED and re-raised
            _add_failure(
                failures,
                key,
                f"{version!r} did not validate: {type(exc).__name__}: {exc}",
            )

    if failures:
        raise ArtifactBundleInvalidError(failures)

    cross_failures = _cross_check_bundle(mapping, facts, artifacts_dir)
    if cross_failures:
        raise ArtifactBundleInvalidError(cross_failures)

    latest_path = artifacts_dir / "latest.json"
    manifest = json.loads(latest_path.read_text()) if latest_path.exists() else {}
    manifest.update(mapping)
    _atomic_write_json(latest_path, manifest)

    logger.info(
        "Replaced the production bundle in latest.json",
        mapping=dict(mapping),
        gold_generation_digest=facts["blend"].gold_generation_digest,
        untouched_keys=sorted(set(manifest) - set(mapping)),
    )


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
