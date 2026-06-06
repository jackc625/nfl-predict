"""Wave-4 rollback + post-promotion manifest-validation suite (Phase 25, D25-17, WR-02).

The activation (Plan 25-03) swapped WP + ATS into ``artifacts/latest.json`` and retained v1.0 OU.
D25-17 requires the swap be REVERSIBLE: ``artifacts/latest.json`` must restore to the recorded
pre-swap version mapping via the SAME sole-swapper (``models.artifacts.update_manifest``), as a pure
per-key manifest restore (the prior artifact dirs are never deleted). This suite proves:

  1. ROLLBACK (D25-17): seed a tmp production manifest with the recorded pre-swap mapping, simulate
     the WP+ATS swap, then restore each swapped key via ``update_manifest``. Assert PARSED-EQUALITY
     (``json.loads`` of the restored manifest == the pre-swap dict) -- the contract-correct check.
     ``_atomic_write_json``'s stable insertion-order serialization makes the seed->swap->restore
     round-trip byte-stable in PRACTICE, but parsed-equality is what the rollback CONTRACT
     guarantees, so it is the primary assertion (avoids a fragile failure if serialization
     formatting ever changes -- Codex MEDIUM). The ``blend`` key and the retained OU key are
     preserved by the per-key write.

  2. SHA256 (D25-17): pair the parsed-equality with a stdlib ``hashlib`` sha256 equality check of the
     restored manifest against the RECORDED seed manifest (the one place byte-identity is asserted,
     and asserted against the SPECIFIC recorded artifact, exactly as Codex recommends). Never
     hand-roll a hash (RESEARCH Security V6 -- stdlib hashlib only).

  3. ATOMICITY (WR-02): monkeypatch ``Path.replace`` to simulate a mid-rename crash and assert the
     restore is atomic -- a crash leaves the manifest byte-unchanged with no stray temp file (the
     ``_atomic_write_json`` except-branch cleanup; mirrors
     ``test_deploy_gate.py::test_update_manifest_atomic_write_preserves_on_failure``).

  4. POST-PROMOTION MANIFEST VALIDATION (Codex suggestion): given a deployed manifest mapping, assert
     every promoted target key points to an EXISTING artifact dir containing the required
     ``metadata.json`` + model file(s), and every retained target key equals its pre-swap value. Run
     against a tmp fixture manifest (hermetic) so it is a durable STRUCTURAL guard, not a one-off on
     committed state.

Hermetic contract (T-24-20 / the Phase-23.1 tmp_path lesson): every manifest write goes to
``tmp_path``; a module-scoped guard snapshots the committed ``artifacts/latest.json`` bytes and
asserts them unchanged (the committed manifest is gitignored, so byte-identity is the meaningful
hermetic assertion).

UIAP-01 (T-25-04-uiap) is confirmed intact by running the existing API import-guard
(``tests/api/test_import_guard.py``) alongside this suite in the plan verification command -- the API
reads only the cache; no request-path model inference. No API code change is needed (D25-13).

The recorded pre-swap + post-swap mappings come from ARMED-RUN-LOG.md (Sections 1.2 and 3.5).

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from models.artifacts import _atomic_write_json, update_manifest

# Repo root resolved from this file: tests/integration/test_rollback.py -> repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]

# Production manifest the suite must NEVER mutate (hermetic guard, mirrors test_promote_models.py).
_PROD_LATEST = REPO_ROOT / "artifacts" / "latest.json"

# The RECORDED pre-swap version mapping (ARMED-RUN-LOG.md Section 1.2 -- the D25-17 rollback target).
# sha256 of the on-disk pre-swap latest.json was 14f9093e...2beb; here we seed an equivalent mapping
# into tmp_path and record ITS sha256 at seed time (the recorded-artifact anchor Codex recommends).
_PRE_SWAP_MAPPING = {
    "wp": "wp_20260327_114739",
    "ats": "ats_20260326_163724",
    "ou": "ou_20260326_163930",
    "blend": "blend_dynamic_20260526_194510",
}

# The POST-SWAP deployed mapping (ARMED-RUN-LOG.md Section 3.5 -- WP+ATS activated, OU retained,
# blend rewritten). WP/ATS are PROMOTED keys; OU is a RETAINED key (== its pre-swap value).
_POST_SWAP_MAPPING = {
    "wp": "wp_20260605_215552",
    "ats": "ats_20260605_220128",
    "ou": "ou_20260326_163930",
    "blend": "blend_dynamic_20260606_020635",
}

# The targets that ACTUALLY swapped in the activation (the keys a rollback restores).
_PROMOTED_TARGETS = ("wp", "ats")
# The target retained on v1.0 (its key must equal its pre-swap value before AND after a rollback).
_RETAINED_TARGET = "ou"


# ---------------------------------------------------------------------------
# Hermetic guard: snapshot the committed manifest, assert it is never mutated
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def _committed_manifest_unchanged() -> Any:
    """Snapshot artifacts/latest.json bytes before the module and assert unchanged after.

    The real production manifest is gitignored, so a git-status check is vacuous; byte-identity of
    the on-disk file across the whole module run is the meaningful hermetic assertion (T-24-20). All
    tests write only to tmp_path, so this snapshot must be identical at teardown.
    """
    before = _PROD_LATEST.read_bytes() if _PROD_LATEST.exists() else None
    yield
    after = _PROD_LATEST.read_bytes() if _PROD_LATEST.exists() else None
    assert before == after, (
        "a test mutated the committed artifacts/latest.json; all writes must go to tmp_path"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    """Return the hex sha256 of a file's bytes (stdlib hashlib only -- RESEARCH Security V6)."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seed_artifact_dir(
    artifacts_dir: Path, version: str, *, with_model: bool = True
) -> Path:
    """Create a minimal valid artifact dir (metadata.json + model.pkl) under artifacts_dir.

    A 'valid' deployed artifact dir (the post-promotion manifest validation contract) must contain
    a metadata.json and at least one model file. ``with_model=False`` seeds an INCOMPLETE dir
    (metadata only, no model) to prove the validator rejects it.
    """
    art = artifacts_dir / version
    art.mkdir(parents=True, exist_ok=True)
    (art / "metadata.json").write_text(json.dumps({"version": version}, indent=2))
    if with_model:
        (art / "model.pkl").write_bytes(b"stub-model")
    return art


def _validate_deployed_manifest(
    artifacts_dir: Path,
    manifest: dict[str, str],
    promoted: tuple[str, ...],
    pre_swap: dict[str, str],
    retained: tuple[str, ...],
) -> list[str]:
    """Validate a post-promotion manifest; return a list of human-readable violations (empty == ok).

    Contract (Codex suggestion):
      * Every PROMOTED target key -> an existing artifact dir containing metadata.json + a model file.
      * Every RETAINED target key == its pre-swap value (the retained target did not move).
    """
    violations: list[str] = []
    for target in promoted:
        version = manifest.get(target)
        if version is None:
            violations.append(f"promoted target {target!r} missing from manifest")
            continue
        art = artifacts_dir / version
        if not art.is_dir():
            violations.append(
                f"promoted {target} -> {version} dir does not exist at {art}"
            )
            continue
        if not (art / "metadata.json").exists():
            violations.append(f"promoted {target} -> {version} missing metadata.json")
        model_files = list(art.glob("model.pkl")) + list(art.glob("*.pkl"))
        if not model_files:
            violations.append(
                f"promoted {target} -> {version} has no model file (*.pkl)"
            )
    for target in retained:
        if manifest.get(target) != pre_swap.get(target):
            violations.append(
                f"retained target {target!r} changed: manifest {manifest.get(target)!r} != "
                f"pre-swap {pre_swap.get(target)!r}"
            )
    return violations


# ---------------------------------------------------------------------------
# D25-17: the manifest restores to the recorded pre-swap mapping (parsed-equality + sha256)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_rollback_restores_pre_swap_mapping(tmp_path: Path) -> None:
    """D25-17: a swapped manifest restores to the recorded pre-swap mapping via update_manifest.

    Seeds the recorded pre-swap mapping into a tmp manifest, records its sha256, simulates the WP+ATS
    activation swap, then restores each swapped key via the SAME sole-swapper. Asserts:
      * PARSED-EQUALITY: json.loads(restored) == the pre-swap dict (the contract-correct check).
      * SHA256: the restored manifest's sha256 == the recorded seed sha256 (byte-identity asserted
        against the SPECIFIC recorded artifact -- Codex MEDIUM).
      * The blend key + the retained OU key survive the per-key swaps.
    """
    art = tmp_path / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    latest = art / "latest.json"

    # Seed the recorded pre-swap mapping and record its sha256 (the rollback target + anchor).
    _atomic_write_json(latest, dict(_PRE_SWAP_MAPPING))
    seed_sha = _sha256(latest)
    seed_blend = json.loads(latest.read_text())["blend"]

    # Simulate the activation swap (WP + ATS to their post-swap versions; OU retained, blend left).
    for target in _PROMOTED_TARGETS:
        update_manifest(target, _POST_SWAP_MAPPING[target], art)
    swapped = json.loads(latest.read_text())
    assert swapped["wp"] == _POST_SWAP_MAPPING["wp"], (
        "pre-condition: WP must have swapped"
    )
    assert swapped["ats"] == _POST_SWAP_MAPPING["ats"], (
        "pre-condition: ATS must have swapped"
    )
    assert swapped["ou"] == _PRE_SWAP_MAPPING["ou"], (
        "OU must be retained through the swap"
    )
    assert swapped["blend"] == seed_blend, "blend must survive the per-key swap"

    # ROLLBACK: restore each swapped key to its recorded pre-swap version (the one-command restore).
    for target in _PROMOTED_TARGETS:
        update_manifest(target, _PRE_SWAP_MAPPING[target], art)

    restored = json.loads(latest.read_text())

    # PARSED-EQUALITY (the contract): the restored manifest equals the pre-swap mapping dict.
    assert restored == _PRE_SWAP_MAPPING, (
        f"rollback did not restore the pre-swap mapping: {restored} != {_PRE_SWAP_MAPPING}"
    )
    # The blend + retained OU keys are intact (never rewritten by the per-key restore).
    assert restored["blend"] == seed_blend, "blend key must survive the rollback"
    assert restored[_RETAINED_TARGET] == _PRE_SWAP_MAPPING[_RETAINED_TARGET], (
        "retained OU key must equal its pre-swap value after rollback"
    )

    # SHA256 (byte-identity against the recorded seed artifact -- the recorded-artifact anchor).
    assert _sha256(latest) == seed_sha, (
        "restored manifest sha256 does not match the recorded pre-swap seed sha256 "
        f"({_sha256(latest)} != {seed_sha})"
    )


# ---------------------------------------------------------------------------
# WR-02: the rollback write is atomic -- a mid-rename crash leaves the manifest byte-unchanged
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_rollback_write_is_atomic_on_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """WR-02: a simulated mid-rename crash during a rollback leaves the manifest byte-unchanged.

    Mirrors test_deploy_gate.py::test_update_manifest_atomic_write_preserves_on_failure but in the
    ROLLBACK direction: seed the post-swap manifest, monkeypatch Path.replace to raise mid-rename,
    attempt the restore, and assert the manifest is byte-unchanged with NO stray temp file (the
    _atomic_write_json except-branch cleanup). Without atomicity, an interrupted rollback could
    truncate the sole production swap surface.
    """
    art = tmp_path / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    latest = art / "latest.json"

    # Seed the post-swap (deployed) manifest -- the state a rollback would restore FROM.
    _atomic_write_json(latest, dict(_POST_SWAP_MAPPING))
    before_bytes = latest.read_bytes()

    # Simulate a crash mid-rename (the atomic landing step). The temp file is flushed by this point.
    def _boom(*args: object, **kwargs: object) -> None:
        raise OSError("simulated crash mid-rename")

    monkeypatch.setattr(Path, "replace", _boom)

    # The rollback write must propagate the OSError (not swallow it).
    with pytest.raises(OSError, match="simulated crash mid-rename"):
        update_manifest("wp", _PRE_SWAP_MAPPING["wp"], art)

    # The manifest is byte-unchanged (the failed rename never landed the new content).
    assert latest.read_bytes() == before_bytes, (
        "latest.json was modified despite the rollback rename failing -- atomic write is not safe"
    )
    # No stray temp file remains (the except branch unlinked it before re-raising).
    remaining = sorted(p.name for p in art.iterdir())
    assert remaining == ["latest.json"], (
        f"stray temp file(s) left after a failed rollback rename: {remaining}"
    )


# ---------------------------------------------------------------------------
# Codex suggestion: post-promotion manifest validation (every promoted key -> complete artifact dir)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_post_promotion_manifest_validation_passes_on_complete_dirs(
    tmp_path: Path,
) -> None:
    """A post-swap manifest is VALID: every promoted key -> a complete dir; the retained key holds.

    Seeds complete artifact dirs (metadata.json + model.pkl) for the post-swap WP + ATS versions and
    the retained OU version, then validates the post-swap manifest. Expects zero violations: each
    promoted key resolves to an existing dir with metadata + a model file, and the retained OU key
    equals its pre-swap value.
    """
    art = tmp_path / "artifacts"
    art.mkdir(parents=True, exist_ok=True)

    # Seed a complete dir for every version the post-swap manifest references.
    for target in ("wp", "ats", _RETAINED_TARGET):
        _seed_artifact_dir(art, _POST_SWAP_MAPPING[target], with_model=True)
    _atomic_write_json(art / "latest.json", dict(_POST_SWAP_MAPPING))

    violations = _validate_deployed_manifest(
        art,
        _POST_SWAP_MAPPING,
        promoted=_PROMOTED_TARGETS,
        pre_swap=_PRE_SWAP_MAPPING,
        retained=(_RETAINED_TARGET,),
    )
    assert violations == [], (
        f"a structurally-valid post-promotion manifest reported violations: {violations}"
    )


@pytest.mark.integration
def test_post_promotion_manifest_validation_rejects_missing_and_incomplete(
    tmp_path: Path,
) -> None:
    """The validator REJECTS a promoted key pointing at a missing OR incomplete artifact dir.

    Proves the guard is not hollow: a promoted WP key whose dir is entirely absent, and a promoted
    ATS key whose dir exists but lacks a model file, both produce violations. A retained OU key that
    drifted from its pre-swap value also produces a violation.
    """
    art = tmp_path / "artifacts"
    art.mkdir(parents=True, exist_ok=True)

    # WP: do NOT seed any dir (missing). ATS: seed metadata-only (incomplete, no model file).
    _seed_artifact_dir(art, _POST_SWAP_MAPPING["ats"], with_model=False)
    # OU retained key drifted to a different version (must be flagged).
    drifted = {
        "wp": _POST_SWAP_MAPPING["wp"],  # dir missing
        "ats": _POST_SWAP_MAPPING["ats"],  # dir incomplete (no model)
        "ou": "ou_SOME_OTHER_VERSION",  # retained key drifted
        "blend": _POST_SWAP_MAPPING["blend"],
    }
    _atomic_write_json(art / "latest.json", drifted)

    violations = _validate_deployed_manifest(
        art,
        drifted,
        promoted=_PROMOTED_TARGETS,
        pre_swap=_PRE_SWAP_MAPPING,
        retained=(_RETAINED_TARGET,),
    )
    joined = "\n".join(violations)
    assert any("wp" in v and "does not exist" in v for v in violations), (
        f"a missing promoted WP dir must be flagged; got:\n{joined}"
    )
    assert any("ats" in v and "no model file" in v for v in violations), (
        f"an incomplete promoted ATS dir (no model) must be flagged; got:\n{joined}"
    )
    assert any("ou" in v and "changed" in v for v in violations), (
        f"a drifted retained OU key must be flagged; got:\n{joined}"
    )
