"""Wave-4 integration suite for the staged-promotion rail (Phase 24, plan 24-05).

Proves the load-bearing SAFETY behaviors of ``scripts/promote_models.py`` end-to-end
WITHOUT real training, using the CORRECT seams (review concerns #1 and #2):

  * The forced-FAIL / PASS control monkeypatches the SHARED bundle builder
    ``scripts.promote_models.deploy_gate.build_candidate_bundle`` -- the function whose
    output actually drives ``evaluate_target`` (and the name promote binds at call time,
    which is identity-equal to ``models.deploy_gate.build_candidate_bundle``). It does NOT
    monkeypatch ``score_deployed_artifacts`` to set CLV: ``build_candidate_bundle`` calls
    ``compute_clv_for_predictions`` and RECOMPUTES the CLV column, so a pre-filled CLV frame
    from a monkeypatched scorer would be silently overwritten -- a hollow proof. A
    ``score_deployed_artifacts`` stub IS installed, but only to satisfy STEP 2 (so no real
    staged artifact is loaded); the FORCED gate verdict comes from the bundle-builder seam.

  * The swap / blend-preservation test SEEDS stub staging dirs (``wp_test``, ``ats_test``,
    ``ou_test``) plus a staging ``latest.json`` so STEP 1b resolves a real ``staged_version``
    and STEP 4's production ``update_manifest`` path GENUINELY executes (not a no-op) for
    passing targets. The test then asserts the passing target key actually CHANGED to its
    staged_version (proving a real swap) while the ``blend`` key is byte-preserved.

Hermetic contract (T-24-20 / the Phase-23.1 tmp_path lesson): every artifact/manifest write
is redirected to ``tmp_path``. No test touches the committed ``artifacts/latest.json``; a
module-scoped guard snapshots its bytes before any test and asserts them unchanged afterwards.

The gold-gated freshness anchor (D24-07) is the ONLY test that needs canonical gold; it is
skip-guarded exactly like ``tests/integration/test_diag_diagnosis.py`` so the suite is
offseason-safe and the FAIL/swap/dry-run tests are fully gold-independent.

Test-name -> 24-VALIDATION.md command map (one comment per test cites its command):
  test_fail_leaves_latest_unchanged   -> ::test_fail_leaves_latest_unchanged   (ACTV-02)
  test_fail_exits_nonzero             -> ::test_fail_exits_nonzero             (ACTV-02)
  test_dry_run_zero_swaps             -> ::test_dry_run_zero_swaps             (ACTV-03)
  test_swap_preserves_blend_key       -> ::test_swap_preserves_blend_key       (ACTV-03)
  test_dry_run_prints_2x2             -> ::test_dry_run_prints_2x2             (D24-11a)
  test_frozen_baseline_matches_rescore-> ::test_frozen_baseline_matches_rescore (D24-07, gold-gated)

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest

import scripts.promote_models as promote
from models import deploy_gate

if TYPE_CHECKING:
    from collections.abc import Callable

# Repo root resolved from this file: tests/integration/test_promote_models.py -> repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]

# Production manifest the suite must NEVER mutate (hermetic guard).
_PROD_LATEST = REPO_ROOT / "artifacts" / "latest.json"

# Gold presence skip-guard for the freshness anchor (mirrors test_diag_diagnosis.py:44/68-69).
_GOLD_WP_PATH = REPO_ROOT / "data" / "gold" / "features_wp.parquet"

# Freshness-anchor tolerance (the A2 anchor; same 5e-3 band as test_diag_diagnosis.py).
_FRESHNESS_TOL = 5e-3

# The seeded stub staging dir names -- the checkable contract (review concern #2). STEP 1b
# globs {staging_dir}/{target}_* and STEP 4 swaps production[target] = staged_version[target],
# so production should end up pointing at exactly these names for passing targets.
_STUB_DIRS = {"wp": "wp_test", "ats": "ats_test", "ou": "ou_test"}

# A realistic production manifest shape for the tmp copy when the real one is absent. The
# "blend" key is the one Pitfall-4 must preserve across a partial-pass swap.
_FALLBACK_MANIFEST = {
    "wp": "wp_20260327_114739",
    "ats": "ats_20260326_163724",
    "ou": "ou_20260326_163930",
    "blend": "blend_dynamic_20260526_194510",
}


# ---------------------------------------------------------------------------
# Hermetic guard: snapshot the committed manifest, assert it is never mutated
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module", autouse=True)
def _committed_manifest_unchanged() -> Any:
    """Snapshot artifacts/latest.json bytes before the module and assert unchanged after.

    The real production manifest is gitignored (artifacts/ is in .gitignore), so a
    ``git status`` check is vacuous; the meaningful hermetic assertion is byte-identity of the
    on-disk file across the whole module run (T-24-20). All tests write only to tmp_path, so
    this snapshot must be identical at teardown.
    """
    before = _PROD_LATEST.read_bytes() if _PROD_LATEST.exists() else None
    yield
    after = _PROD_LATEST.read_bytes() if _PROD_LATEST.exists() else None
    assert before == after, (
        "a test mutated the committed artifacts/latest.json; all writes must go to tmp_path"
    )


# ---------------------------------------------------------------------------
# Fixtures: a self-contained tmp production dir + seeded stub staging dir
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_artifacts(tmp_path: Path) -> Path:
    """A throwaway production artifacts dir with a real latest.json (copied or synthesized).

    Copies the committed artifacts/latest.json into tmp_path/"artifacts"/"latest.json" when it
    exists (so the real update_manifest writes against a faithful manifest shape, never the
    committed one); otherwise synthesizes the canonical {wp,ats,ou,blend} shape. This fixture is
    self-contained -- it does NOT require any staged dir to exist.

    Returns:
        The tmp production artifacts dir (contains latest.json).
    """
    art = tmp_path / "artifacts"
    art.mkdir(parents=True, exist_ok=True)
    dest = art / "latest.json"
    if _PROD_LATEST.exists():
        shutil.copyfile(_PROD_LATEST, dest)
    else:
        dest.write_text(json.dumps(_FALLBACK_MANIFEST, indent=2))
    return art


@pytest.fixture
def tmp_stage(tmp_path: Path) -> Path:
    """Seed stub staging dirs + a staging latest.json so STEP 1b/STEP 4 run the REAL path.

    Creates EXACTLY ``wp_test``, ``ats_test``, ``ou_test`` under a tmp staging root (each a real
    directory with a placeholder model.pkl -- STEP 2 scoring is monkeypatched away, so no real
    artifact load happens) AND writes a staging ``latest.json`` mapping wp/ats/ou to them. With
    ``--skip-train`` set, ``_resolve_staged_version`` globs ``{stage}/{target}_*`` and finds these
    dirs, so ``staged_version`` resolves and STEP 4's production ``update_manifest`` swap actually
    executes for passing targets (review concern #2 -- not a no-op).

    Returns:
        The tmp staging root (contains wp_test/ats_test/ou_test + latest.json).
    """
    stage = tmp_path / "staging"
    stage.mkdir(parents=True, exist_ok=True)
    for stub in _STUB_DIRS.values():
        stub_dir = stage / stub
        stub_dir.mkdir(parents=True, exist_ok=True)
        # A placeholder file is sufficient; scoring is stubbed so it is never loaded.
        (stub_dir / "model.pkl").write_bytes(b"stub")
    (stage / "latest.json").write_text(json.dumps(_STUB_DIRS, indent=2))
    return stage


# ---------------------------------------------------------------------------
# Forced-bundle control (the CORRECT seam) + hermetic STEP-2 / data stubs
# ---------------------------------------------------------------------------


def _negative_bundle(target: str) -> dict[str, Any]:
    """A deterministically gate-FAILING bundle for every target.

    Under the committed floor_mode=non_regression (D25-01) the gate consumes the paired
    candidate-minus-baseline DELTA. To force a FAIL the delta is a constant -0.5 over 280 games
    (significantly WORSE than v1.0): clv_significance reports mean<0 with a tiny p, so the pooled
    AND per-season non-regression floor FAIL. clv_values carries the same array (so the
    absolute-vs-zero verdict computation is well-defined) and the secondary metrics are set
    clearly WORSE than the frozen v1.0 baseline so even a hypothetical floor pass could not
    rescue the target. (The legacy absolute floor also fails on this clv_values.)
    """
    neg = np.full(280, -0.5)
    pooled = deploy_gate.clv_significance(neg)
    per_season = {
        s: deploy_gate.clv_significance(neg) for s in (2021, 2022, 2023, 2024)
    }
    per_season_delta = {s: np.full(280, -0.5) for s in (2021, 2022, 2023, 2024)}
    # Raw per-season candidate CLV arrays (WR-04): build_candidate_bundle ALWAYS populates
    # per_season_clv_values, so the forced bundle must too -- otherwise _absolute_verdict falls
    # back to per_season and the synthetic fixture silently diverges from the real bundle shape.
    per_season_clv_values = dict.fromkeys((2021, 2022, 2023, 2024), neg)
    bundle: dict[str, Any] = {
        "clv_values": neg,
        # floor_mode=non_regression delta keys: a significantly-negative delta -> FAIL. (These
        # mirror the keys Plan 25-02 populates from the real merge-on-game_id pairing; here they
        # are forced synthetically to drive the hermetic gate verdict.)
        "baseline_clv_values": np.zeros(280),
        "clv_delta_values": neg,
        "per_season_clv_delta_values": per_season_delta,
        "mean": pooled["mean"],
        "t": pooled["t"],
        "p": pooled["p"],
        "n": pooled["n"],
        "per_season": per_season,
        "per_season_clv_values": per_season_clv_values,
    }
    if target == "wp":
        # Clearly regressing vs frozen v1.0 (accuracy down, ECE/Brier up).
        bundle["accuracy"] = 0.50
        bundle["ece"] = 0.50
        bundle["brier_score"] = 0.50
    else:
        bundle["mae"] = 99.0
    return bundle


def _passing_bundle(target: str) -> dict[str, Any]:
    """A deterministically gate-PASSING bundle (not worse than v1.0, at-or-better secondary).

    Under the committed floor_mode=non_regression (D25-01) the gate consumes the paired
    candidate-minus-baseline DELTA. To force a PASS the delta is pinned to an EXACT zero-mean
    array (candidate ~= v1.0 baseline) so clv_significance reports mean 0 / large p -> the pooled
    AND per-season non-regression floor PASS. Secondary metrics are at-or-better than the frozen
    v1.0 baseline so the secondary/calibration gates pass.
    """
    rng = np.random.default_rng(24)
    pos = rng.normal(0.05, 0.2, 280)
    pooled = deploy_gate.clv_significance(pos)
    # Raw per-season candidate CLV arrays (WR-04): capture each season's raw array so the
    # per_season significance bundles AND per_season_clv_values describe the SAME population,
    # matching the build_candidate_bundle contract (it always populates per_season_clv_values).
    per_season_clv_values = {
        s: rng.normal(0.05, 0.2, 280) for s in (2021, 2022, 2023, 2024)
    }
    per_season = {
        s: deploy_gate.clv_significance(per_season_clv_values[s])
        for s in (2021, 2022, 2023, 2024)
    }

    def _zero_mean(n: int) -> np.ndarray:
        arr = rng.normal(0.0, 0.2, n)
        return arr - float(np.mean(arr))  # exact mean 0 -> deterministically not-worse

    delta = _zero_mean(280)
    per_season_delta = {s: _zero_mean(280) for s in (2021, 2022, 2023, 2024)}
    bundle: dict[str, Any] = {
        "clv_values": pos,
        "baseline_clv_values": pos - delta,
        "clv_delta_values": delta,
        "per_season_clv_delta_values": per_season_delta,
        "mean": pooled["mean"],
        "t": pooled["t"],
        "p": pooled["p"],
        "n": pooled["n"],
        "per_season": per_season,
        "per_season_clv_values": per_season_clv_values,
    }
    if target == "wp":
        # At-or-better than frozen v1.0 (accuracy >= baseline, ECE/Brier <= baseline).
        bundle["accuracy"] = 0.99
        bundle["ece"] = 0.0
        bundle["brier_score"] = 0.0
    else:
        bundle["mae"] = 0.0
    return bundle


def _install_hermetic_stubs(
    monkeypatch: pytest.MonkeyPatch,
    bundle_factory: Callable[[str], dict[str, Any]],
) -> None:
    """Wire promote_models for a hermetic run: stub STEP-2 data/scoring, force the bundle.

    Stubs (so NO canonical gold/odds/artifact is needed):
      * ``scripts.promote_models.score_deployed_artifacts`` -> a trivial 1-row frame (STEP 2
        only; its CLV is irrelevant -- it is recomputed inside build_candidate_bundle, which is
        itself replaced below).
      * ``BacktestEngine._load_closing_odds`` -> a trivial odds frame (avoids the silver parquet
        read).
      * ``scripts.promote_models._load_gold_holdout`` -> a trivial gold frame (avoids the
        gold parquet read via engine._load_features).

    Forces the gate verdict via the LOAD-BEARING seam:
      * ``scripts.promote_models.deploy_gate.build_candidate_bundle`` -> ``bundle_factory``
        (identity-equal to models.deploy_gate.build_candidate_bundle; the name promote binds).

    Also stubs ``_resolve_staged_version`` to return the canonical stub dir name for each
    target (``wp`` -> ``wp_test`` ...). The real resolver sorts by the embedded
    ``{YYYYMMDD}_{HHMMSS}`` timestamp and would raise on the contract-mandated ``wp_test`` names
    (no parseable stamp); the timestamp-sort logic is exercised by promote's own unit path, while
    this suite's load-bearing concern is the STEP-4 production swap. The seeded stub dirs +
    staging latest.json still exist on disk, so the resolved name maps to a real staged dir and
    STEP 4's production ``update_manifest`` genuinely executes for passing targets (not a no-op).
    """
    trivial_scored = pd.DataFrame(
        {
            "game_id": ["G0"],
            "season": [2021],
            "week": [1],
            "model_prob": [0.5],
            "actual": [1],
        }
    )
    trivial_odds = pd.DataFrame(
        {"game_id": ["G0"], "ml_home": [-110], "ml_away": [-110]}
    )

    monkeypatch.setattr(
        promote,
        "score_deployed_artifacts",
        lambda target, gold_df=None, artifacts_dir=None: trivial_scored.copy(),
    )
    monkeypatch.setattr(
        promote.BacktestEngine,
        "_load_closing_odds",
        lambda self: trivial_odds.copy(),
    )
    monkeypatch.setattr(
        promote,
        "_load_gold_holdout",
        lambda target, engine: trivial_scored.copy(),
    )
    # Resolve each target to its seeded stub dir name (the contract-mandated wp_test/ats_test/
    # ou_test). The real resolver sorts by an embedded timestamp those names lack; STEP 4's real
    # production update_manifest still runs against the resolved (on-disk) stub dir.
    monkeypatch.setattr(
        promote,
        "_resolve_staged_version",
        lambda target, staging_dir, *, skip_train: _STUB_DIRS[target],
    )
    # THE correct seam (review concern #1): force the candidate bundle the gate consumes.
    monkeypatch.setattr(
        promote.deploy_gate,
        "build_candidate_bundle",
        lambda target, scored_df, odds_df, cfg: bundle_factory(target),
    )


# ---------------------------------------------------------------------------
# ACTV-02: forced-FAIL leaves production latest.json byte-AND-mtime-unchanged + non-zero exit
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_fail_leaves_latest_unchanged(
    tmp_artifacts: Path, tmp_stage: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """24-VALIDATION: pytest .../test_promote_models.py::test_fail_leaves_latest_unchanged.

    With build_candidate_bundle forced significantly-negative for every target, a
    ``--promote`` run leaves the tmp production latest.json BYTE-identical AND mtime-identical
    (the FAIL never even rewrites the file) and exits non-zero (ACTV-02 / T-24-16, T-24-17).
    """
    _install_hermetic_stubs(monkeypatch, _negative_bundle)

    latest = tmp_artifacts / "latest.json"
    before = latest.read_bytes()
    before_mtime = latest.stat().st_mtime

    rc = promote.main(
        [
            "--promote",
            "--artifacts-dir",
            str(tmp_artifacts),
            "--staging-dir",
            str(tmp_stage),
            "--skip-train",
        ]
    )

    after = latest.read_bytes()
    after_mtime = latest.stat().st_mtime

    assert rc != 0, (
        "a forced-FAIL --promote run must exit non-zero (hard block observable)"
    )
    assert before == after, (
        "FAIL must leave production latest.json byte-identical (zero swaps)"
    )
    assert before_mtime == after_mtime, (
        "FAIL must not even rewrite production latest.json (mtime-identical)"
    )


@pytest.mark.integration
def test_fail_exits_nonzero(
    tmp_artifacts: Path, tmp_stage: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """24-VALIDATION: pytest .../test_promote_models.py::test_fail_exits_nonzero.

    The forced-FAIL run returns a non-zero exit code so CI/automation observes the hard block
    (ACTV-02 / T-24-17). Asserted both with and without --promote (the exit code is gate-driven
    even in dry-run, per D24-10).
    """
    _install_hermetic_stubs(monkeypatch, _negative_bundle)

    rc_promote = promote.main(
        [
            "--promote",
            "--artifacts-dir",
            str(tmp_artifacts),
            "--staging-dir",
            str(tmp_stage),
            "--skip-train",
        ]
    )
    assert rc_promote != 0, "forced-FAIL --promote run must exit non-zero"

    rc_dry = promote.main(
        [
            "--artifacts-dir",
            str(tmp_artifacts),
            "--staging-dir",
            str(tmp_stage),
            "--skip-train",
        ]
    )
    assert rc_dry != 0, (
        "forced-FAIL dry-run must ALSO exit non-zero (gate-driven, D24-10)"
    )


# ---------------------------------------------------------------------------
# WR-06: --promote --skip-train against an empty staging dir exits non-zero
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_promote_skip_train_empty_staging_exits_nonzero(
    tmp_artifacts: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """24-VALIDATION: pytest .../test_promote_models.py::test_promote_skip_train_empty_staging_exits_nonzero.

    WR-06: a ``--promote --skip-train`` run against a staging dir with NO candidate dirs
    resolves zero staged versions. Rather than silently exiting 0 (a no-op that masks an
    operator error), the armed promote must exit non-zero and leave production untouched.
    Uses the REAL _resolve_staged_version (skip_train=True returns None on the empty glob).
    """
    # Avoid the silver/gold reads in case any code path is reached; the run should short-circuit
    # at "nothing to gate" before STEP 2, but stub defensively.
    monkeypatch.setattr(
        promote,
        "score_deployed_artifacts",
        lambda target, gold_df=None, artifacts_dir=None: pd.DataFrame(),
    )
    empty_stage = tmp_path / "empty_staging"
    empty_stage.mkdir(parents=True, exist_ok=True)

    latest = tmp_artifacts / "latest.json"
    before = latest.read_bytes()

    rc = promote.main(
        [
            "--promote",
            "--artifacts-dir",
            str(tmp_artifacts),
            "--staging-dir",
            str(empty_stage),
            "--skip-train",
        ]
    )

    assert rc != 0, (
        "--promote --skip-train with no staged candidates must exit non-zero (WR-06)"
    )
    assert latest.read_bytes() == before, (
        "an empty-staging promote must not touch production latest.json"
    )


# ---------------------------------------------------------------------------
# ACTV-03: a dry / no-pass run produces ZERO production swaps
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_dry_run_zero_swaps(
    tmp_artifacts: Path, tmp_stage: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """24-VALIDATION: pytest .../test_promote_models.py::test_dry_run_zero_swaps.

    A bare run (NO --promote) leaves production latest.json byte-unchanged regardless of the
    gate verdict -- proven here with a PASSING forced bundle so the only thing preventing a
    swap is the missing --promote flag (ACTV-03 / T-24-18). mtime is also unchanged (the dry
    path never writes production).
    """
    _install_hermetic_stubs(monkeypatch, _passing_bundle)

    latest = tmp_artifacts / "latest.json"
    before = latest.read_bytes()
    before_mtime = latest.stat().st_mtime

    rc = promote.main(
        [
            "--artifacts-dir",
            str(tmp_artifacts),
            "--staging-dir",
            str(tmp_stage),
            "--skip-train",
        ]
    )

    after = latest.read_bytes()
    after_mtime = latest.stat().st_mtime

    assert rc == 0, "an all-pass dry-run should exit zero (no failing targets)"
    assert before == after, "a dry-run (no --promote) must swap nothing in production"
    assert before_mtime == after_mtime, (
        "a dry-run must not rewrite production latest.json"
    )


# ---------------------------------------------------------------------------
# ACTV-03: a partial-pass swap preserves the blend key AND performs a REAL swap
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_swap_preserves_blend_key(
    tmp_artifacts: Path, tmp_stage: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """24-VALIDATION: pytest .../test_promote_models.py::test_swap_preserves_blend_key.

    With WP forced to PASS and ATS/OU forced to FAIL, a ``--promote`` run (against the seeded
    stub staging dirs so staged_version resolves and the real update_manifest executes):
      * preserves the manifest "blend" key byte-for-byte (Pitfall 4 / T-24-18),
      * swaps ONLY the passing target -- production["wp"] now equals its staged_version
        ("wp_test"), proving a REAL, non-no-op swap occurred,
      * leaves the failing targets' production keys unchanged.
    """

    def _partial(target: str) -> dict[str, Any]:
        return _passing_bundle(target) if target == "wp" else _negative_bundle(target)

    _install_hermetic_stubs(monkeypatch, _partial)

    latest = tmp_artifacts / "latest.json"
    before_manifest = json.loads(latest.read_text())
    before_blend = before_manifest.get("blend")
    before_ats = before_manifest.get("ats")
    before_ou = before_manifest.get("ou")

    rc = promote.main(
        [
            "--promote",
            "--artifacts-dir",
            str(tmp_artifacts),
            "--staging-dir",
            str(tmp_stage),
            "--skip-train",
        ]
    )

    after_manifest = json.loads(latest.read_text())

    # Any-fail present (ATS/OU) so the run exits non-zero even though WP swapped.
    assert rc != 0, "ATS/OU failed the gate, so the run must exit non-zero"
    # Blend pointer preserved byte-for-byte (the sole-swapper never rewrites it).
    assert after_manifest.get("blend") == before_blend, (
        "the conditional swap must preserve the blend key (Pitfall 4)"
    )
    # The passing target ACTUALLY swapped to its staged_version (a real swap, not a no-op).
    assert after_manifest.get("wp") == _STUB_DIRS["wp"], (
        f"WP passed the gate, so production['wp'] must become its staged_version "
        f"{_STUB_DIRS['wp']!r}; got {after_manifest.get('wp')!r}"
    )
    assert after_manifest.get("wp") != before_manifest.get("wp"), (
        "WP must have changed from its pre-swap value (proves a real, non-no-op swap)"
    )
    # Failing targets unchanged in production.
    assert after_manifest.get("ats") == before_ats, "failing ATS must not swap"
    assert after_manifest.get("ou") == before_ou, "failing OU must not swap"


@pytest.mark.integration
def test_swap_copies_passing_artifact_into_production(
    tmp_artifacts: Path, tmp_stage: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """25-03 REGRESSION: a passing target's artifact dir must be RESOLVABLE in production.

    Bug (Phase 25 armed run): STEP 4 rewrote production ``latest.json`` to point at the staged
    version but NEVER copied the staged artifact dir into the production dir. With the real run's
    distinct ``--staging-dir artifacts_staging`` vs ``--artifacts-dir artifacts``, the deployed
    version then resolved to ``artifacts/{version}`` -- a dir that only existed under staging --
    so ``load_model_artifact(target)`` raised ``FileNotFoundError``: production was broken.

    This asserts the fix: after a ``--promote`` swap (WP pass, ATS/OU fail) with a staging dir
    DISTINCT from the production dir, the passing target's staged artifact dir is copied into the
    production dir (so the manifest pointer resolves) and the copy is byte-identical to the
    gate-scored staging artifact (D25-06). The failing targets' dirs are NOT copied.
    """

    def _partial(target: str) -> dict[str, Any]:
        return _passing_bundle(target) if target == "wp" else _negative_bundle(target)

    _install_hermetic_stubs(monkeypatch, _partial)

    # Seed a recognizable payload in the staged WP dir so we can assert byte-identity post-copy.
    staged_wp = tmp_stage / _STUB_DIRS["wp"]
    (staged_wp / "model.pkl").write_bytes(b"wp-model-bytes")
    (staged_wp / "metadata.json").write_text('{"target": "wp"}')

    # Sanity: production and staging are DISTINCT roots (the bug's trigger condition).
    assert tmp_artifacts.resolve() != tmp_stage.resolve()

    rc = promote.main(
        [
            "--promote",
            "--artifacts-dir",
            str(tmp_artifacts),
            "--staging-dir",
            str(tmp_stage),
            "--skip-train",
        ]
    )
    assert rc != 0, "ATS/OU failed the gate, so the run must exit non-zero"

    # The passing WP target's artifact dir is now PRESENT in production (the manifest pointer
    # resolves) -- this is the assertion the pre-fix code failed.
    prod_wp = tmp_artifacts / _STUB_DIRS["wp"]
    assert prod_wp.is_dir(), (
        f"the passing WP staged dir must be copied into production at {prod_wp} so the "
        "manifest pointer resolves (Phase 25 armed-run bug)"
    )
    # The production copy is byte-identical to the gate-scored staging artifact (D25-06).
    assert (prod_wp / "model.pkl").read_bytes() == b"wp-model-bytes", (
        "the promoted artifact must be byte-identical to the gate-scored staging artifact"
    )
    assert (prod_wp / "metadata.json").read_text() == '{"target": "wp"}'

    # The failing targets were NOT copied into production (only passing targets promote).
    assert not (tmp_artifacts / _STUB_DIRS["ats"]).exists(), (
        "failing ATS must not be copied"
    )
    assert not (tmp_artifacts / _STUB_DIRS["ou"]).exists(), (
        "failing OU must not be copied"
    )


# ---------------------------------------------------------------------------
# SPEC R5 idempotency: a SECOND armed run on the same staged artifacts is a no-op
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_second_armed_run_is_a_noop(
    tmp_artifacts: Path, tmp_stage: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SPEC R5 idempotency: re-running the armed promotion changes nothing.

    Two INDEPENDENT mechanisms make this true, and naming both is the point of this
    docstring:

      1. ``_promote_artifact_dir``'s destination-exists early return -- when
         ``artifacts_dir/{version}`` is already present (the re-promote of an identical
         version stamp), it returns before ``shutil.copytree`` and leaves the in-place
         production artifact untouched.
      2. ``update_manifest``'s per-key write is idempotent -- writing the same target the
         same version reproduces the same manifest bytes.

    The test does NOT infer the property from those two mechanisms; it ASSERTS the property
    directly, which is what SPEC R5's idempotency acceptance criterion asks for. The
    distinction matters because the two mechanisms can be individually correct and the
    property still break -- e.g. a later ``dirs_exist_ok=True`` on the copy would keep both
    docstrings honest while silently re-copying over production on every re-run.

    The no-re-copy assertion is made with a SENTINEL rather than an mtime: after the first
    run, the production copy's payload is overwritten with a marker. A second run that
    re-copies would restore the staged bytes and erase the marker. An mtime comparison would
    not catch a same-second re-copy on a coarse-resolution filesystem; the marker cannot be
    restored by accident.
    """

    def _partial(target: str) -> dict[str, Any]:
        return _passing_bundle(target) if target == "wp" else _negative_bundle(target)

    _install_hermetic_stubs(monkeypatch, _partial)

    staged_wp = tmp_stage / _STUB_DIRS["wp"]
    (staged_wp / "model.pkl").write_bytes(b"wp-model-bytes")

    argv = [
        "--promote",
        "--artifacts-dir",
        str(tmp_artifacts),
        "--staging-dir",
        str(tmp_stage),
        "--skip-train",
    ]

    rc_first = promote.main(argv)
    latest = tmp_artifacts / "latest.json"
    manifest_after_first = latest.read_bytes()
    prod_wp = tmp_artifacts / _STUB_DIRS["wp"]
    assert prod_wp.is_dir(), (
        "the first armed run must have promoted the passing WP target"
    )

    # SENTINEL: if the second run re-copies the staged dir, this marker is erased.
    sentinel = b"do-not-overwrite-me"
    (prod_wp / "model.pkl").write_bytes(sentinel)

    rc_second = promote.main(argv)

    assert rc_second == rc_first, (
        "the second armed run must reach the same gate verdict as the first "
        f"(first={rc_first}, second={rc_second})"
    )
    # THE property, asserted directly: the manifest bytes did not move.
    assert latest.read_bytes() == manifest_after_first, (
        "a second armed run against the same staged artifacts must leave "
        "artifacts/latest.json BYTE-identical (SPEC R5 idempotency)"
    )
    # THE property, second half: the already-present production dir was not copied again.
    assert (prod_wp / "model.pkl").read_bytes() == sentinel, (
        "the second armed run must not re-copy the staged artifact over the in-place "
        "production one -- _promote_artifact_dir returns early when the destination exists"
    )


# ---------------------------------------------------------------------------
# D24-11a: the dry-run prints the per-target 2x2 readout (the acceptance artifact)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_dry_run_prints_2x2(
    tmp_artifacts: Path,
    tmp_stage: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """24-VALIDATION: pytest .../test_promote_models.py::test_dry_run_prints_2x2.

    A dry run prints a per-target PASS/FAIL table with the candidate-vs-frozen-v1.0 CLV
    (before/after) -- the D24-11a acceptance readout. Uses a partial verdict (WP pass, ATS/OU
    fail) so BOTH [PASS] and [FAIL] tokens appear, and asserts each target token plus a
    candidate-vs-frozen CLV indication is present.
    """

    def _partial(target: str) -> dict[str, Any]:
        return _passing_bundle(target) if target == "wp" else _negative_bundle(target)

    _install_hermetic_stubs(monkeypatch, _partial)

    promote.main(
        [
            "--artifacts-dir",
            str(tmp_artifacts),
            "--staging-dir",
            str(tmp_stage),
            "--skip-train",
        ]
    )

    out = capsys.readouterr().out

    # Every target appears in the readout.
    for token in ("WP", "ATS", "OU"):
        assert token in out, f"2x2 readout must mention target {token}; got:\n{out}"
    # Both verdicts present (partial pass).
    assert "[PASS]" in out, f"readout must show a [PASS] verdict; got:\n{out}"
    assert "[FAIL]" in out, f"readout must show a [FAIL] verdict; got:\n{out}"
    # A candidate-vs-frozen CLV indication (the before/after column).
    assert "Pooled CLV" in out, (
        f"readout must show pooled CLV candidate vs frozen; got:\n{out}"
    )
    assert "candidate" in out and "frozen v1.0" in out, (
        f"readout must contrast candidate vs frozen v1.0; got:\n{out}"
    )


# ---------------------------------------------------------------------------
# D24-07: the frozen gate.toml baseline matches a fresh diagnose re-score (gold-gated)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(
    not _GOLD_WP_PATH.exists(),
    reason=f"Canonical gold not present at {_GOLD_WP_PATH}",
)
def test_frozen_baseline_matches_rescore() -> None:
    """24-VALIDATION: pytest .../test_promote_models.py::test_frozen_baseline_matches_rescore.

    The frozen ``config/gate.toml`` baseline (WP pooled accuracy + headline CLV) matches a
    fresh ``diagnose.run_diagnosis(run_backtest_half=False)`` production-half re-score of the
    DEPLOYED incumbent artifacts within tolerance -- proving the frozen judge has not drifted
    from the deployed artifacts (D24-07 / T-24-19). Post-Phase-25 (D25-11) the deployed WP/ATS
    incumbents are the activated re-fits and OU is retained v1.0, so this re-scores the re-fit
    WP and matches the re-frozen baseline; it is NO LONGER a v1.0 re-score. Skips cleanly when
    canonical gold is absent.
    """
    from backtest.diagnose import run_diagnosis
    from backtest.engine import BacktestEngine

    cfg = deploy_gate.load_gate_config(REPO_ROOT / "config" / "gate.toml")
    wp_pooled = cfg["baseline"]["wp"]["pooled"]
    frozen_accuracy = wp_pooled["accuracy"]
    frozen_clv_mean = wp_pooled["mean"]

    # Re-score the deployed incumbent production half on canonical 2021-2024 gold (LOAD +
    # predict). Post-D25-11 the WP incumbent is the activated re-fit, not v1.0.
    engine = BacktestEngine()
    gold: dict[str, pd.DataFrame] = {}
    for target in ("wp", "ats", "ou"):
        df = engine._load_features(target)
        in_holdout = (df["season"] >= 2021) & (df["season"] <= 2024)
        holdout_df: pd.DataFrame = df.loc[in_holdout].copy()
        gold[target] = holdout_df
    odds = engine._load_closing_odds()

    diag = run_diagnosis(gold=gold, odds=odds, run_backtest_half=False)
    prod_wp = diag["production"]["wp"]

    rescored_accuracy = prod_wp["pooled_accuracy"]
    rescored_clv_mean = prod_wp["clv_significance_raw"]["mean"]

    assert abs(rescored_accuracy - frozen_accuracy) < _FRESHNESS_TOL, (
        f"frozen WP accuracy {frozen_accuracy} drifted from a fresh re-score "
        f"{rescored_accuracy} (baseline no longer matches the deployed artifacts)"
    )
    assert abs(rescored_clv_mean - frozen_clv_mean) < _FRESHNESS_TOL, (
        f"frozen WP pooled CLV mean {frozen_clv_mean} drifted from a fresh re-score "
        f"{rescored_clv_mean} (baseline no longer matches the deployed artifacts)"
    )


@pytest.mark.integration
@pytest.mark.skipif(
    not _GOLD_WP_PATH.exists(),
    reason=f"Canonical gold not present at {_GOLD_WP_PATH}",
)
def test_frozen_baseline_matches_rescore_all_fields() -> None:
    """30-09 (T-30-38): EVERY frozen [baseline.*] field matches a fresh generator re-score.

    Why this exists alongside ``test_frozen_baseline_matches_rescore``: that test asserts the
    two WP freshness ANCHORS (pooled accuracy + pooled CLV mean). The Phase-30 threat register
    claims the freshness test "independently re-verifies all pooled and per-season values
    against a fresh re-score" -- which was NOT true of the anchor test. A hand-edit to any ATS
    or O/U value, or to any per-season mean/t/p, passed the anchor test untouched, so the
    prohibition "MUST NOT hand-edit the frozen [baseline.*] values -- generator block-paste
    only" rested on discipline rather than on a check. This closes that: it re-runs
    ``scripts.freeze_gate_baseline.compute_baseline`` -- the SAME generator whose printed block
    is the only sanctioned way to write those values -- and compares the committed config to it
    field by field.

    Tolerances: floats use ``_FRESHNESS_TOL`` (the SAME recomputation band the drift tripwire
    and the anchor test use -- no new tolerance is introduced, SPEC R5). Per-season sample
    sizes are compared EXACTLY, matching ``_drift_tripwire``'s reasoning that an integer
    population count cannot drift by float noise.

    A failure means one of two things and the message says which to check first: either the
    committed block was hand-edited, or the deployed artifacts / gold moved under it and the
    baseline needs a generator re-freeze.
    """
    from scripts.freeze_gate_baseline import compute_baseline

    cfg = deploy_gate.load_gate_config(REPO_ROOT / "config" / "gate.toml")
    frozen = cfg["baseline"]
    fresh = compute_baseline(artifacts_dir=REPO_ROOT / "artifacts")

    remediation = (
        "Either a [baseline.*] value was hand-edited (forbidden -- generator block-paste only, "
        "D24-07), or the deployed artifacts/gold moved and the baseline needs a re-freeze via "
        "`python -m scripts.freeze_gate_baseline`. Do NOT nudge the number to make this pass."
    )

    compared = 0
    for target in ("wp", "ats", "ou"):
        frozen_pooled = frozen[target]["pooled"]
        fresh_pooled = fresh[target]["pooled"]
        assert set(frozen_pooled) == set(fresh_pooled), (
            f"{target} pooled baseline KEYS differ from the generator's: committed "
            f"{sorted(frozen_pooled)} vs generated {sorted(fresh_pooled)}. {remediation}"
        )
        for key, frozen_value in frozen_pooled.items():
            fresh_value = fresh_pooled[key]
            assert abs(float(frozen_value) - float(fresh_value)) < _FRESHNESS_TOL, (
                f"frozen baseline.{target}.pooled.{key} = {frozen_value} does not match a "
                f"fresh re-score {fresh_value} (tol {_FRESHNESS_TOL}). {remediation}"
            )
            compared += 1

        for season in (2021, 2022, 2023, 2024):
            frozen_season = frozen[target]["season"][season]
            fresh_season = fresh[target]["season"][int(season)]
            # Sample size EXACT: a population-count change is hard drift, not float noise.
            assert int(frozen_season["n"]) == int(fresh_season["n"]), (
                f"frozen baseline.{target}.season.{season}.n = {frozen_season['n']} does not "
                f"match a fresh re-score {fresh_season['n']}; the holdout population changed. "
                f"{remediation}"
            )
            compared += 1
            for key in ("mean", "t", "p"):
                assert (
                    abs(float(frozen_season[key]) - float(fresh_season[key]))
                    < _FRESHNESS_TOL
                ), (
                    f"frozen baseline.{target}.season.{season}.{key} = {frozen_season[key]} "
                    f"does not match a fresh re-score {fresh_season[key]} "
                    f"(tol {_FRESHNESS_TOL}). {remediation}"
                )
                compared += 1

    # Guard the guard: if the loop silently compared nothing, the test would be vacuous.
    pooled_fields = 8 + 6 + 6  # wp (incl. accuracy/ece/brier) + ats (mae) + ou (mae)
    season_fields = 3 * 4 * 4  # 3 targets x 4 holdout seasons x (mean, t, p, n)
    expected_fields = pooled_fields + season_fields
    assert compared == expected_fields, (
        f"expected to compare {expected_fields} frozen fields, compared {compared} -- the "
        "baseline schema changed and this test is no longer covering all of it"
    )


# ---------------------------------------------------------------------------
# D25-15 (Plan 25-02): the paired baseline re-score + merge-on-game_id pairing
# ---------------------------------------------------------------------------


def _paired_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a candidate + baseline per-game CLV frame pair sharing game_ids across seasons.

    Each frame carries ``game_id``, ``season``, and the WP CLV column (``probability_clv``). The
    candidate has a small positive offset over the baseline so the paired delta is positive (a
    non-regression PASS), and the two frames intersect on every game_id so the merge keeps full n.
    """
    rng = np.random.default_rng(2502)
    rows = []
    # DERIVED from deploy_gate.HOLDOUT_SEASONS (Plan 33.1-09 Task 3). These were the
    # hand-written seasons (2021, 2022, 2023, 2024). _populate_paired_delta_keys rebuilds
    # its per-season dicts from the LIVE holdout, so a synthetic frame carrying different
    # seasons produced empty per-season arrays and a KeyError on read-back -- a fixture
    # silently declaring the partition. What this test is about is the merge-on-game_id
    # pairing and the delta invariant, neither of which depends on WHICH seasons they are.
    for season in deploy_gate.HOLDOUT_SEASONS:
        for i in range(50):
            rows.append((f"{season}_G{i}", int(season)))
    game_ids = [r[0] for r in rows]
    seasons = [r[1] for r in rows]
    n = len(rows)
    base_clv = rng.normal(-0.05, 0.1, n)
    cand_clv = base_clv + 0.01  # candidate slightly better -> positive paired delta
    candidate_valid = pd.DataFrame(
        {"game_id": game_ids, "season": seasons, "probability_clv": cand_clv}
    )
    baseline_valid = pd.DataFrame(
        {"game_id": game_ids, "season": seasons, "probability_clv": base_clv}
    )
    return candidate_valid, baseline_valid


@pytest.mark.integration
def test_paired_delta_keys_populated_on_game_id() -> None:
    """Plan 25-02: _populate_paired_delta_keys merges on game_id and keeps the Plan 25-01 invariant.

    The pinned keys (clv_values / baseline_clv_values / clv_delta_values + per-season equivalents)
    are populated from a genuine per-game pairing; the internal-consistency invariant
    clv_delta_values == clv_values - baseline_clv_values holds element-wise, and the per-season
    arrays carry the integer holdout seasons.
    """
    candidate_valid, baseline_valid = _paired_frames()
    seasons = tuple(int(season) for season in deploy_gate.HOLDOUT_SEASONS)
    candidate: dict[str, Any] = {
        "clv_values": None,
        "baseline_clv_values": None,
        "clv_delta_values": None,
        "per_season_clv_values": dict.fromkeys(seasons, None),
        "per_season_baseline_clv_values": dict.fromkeys(seasons, None),
        "per_season_clv_delta_values": dict.fromkeys(seasons, None),
        "per_season": {},
    }

    promote._populate_paired_delta_keys(
        "wp", candidate, candidate_valid, baseline_valid
    )

    cand = candidate["clv_values"]
    base = candidate["baseline_clv_values"]
    delta = candidate["clv_delta_values"]
    assert cand is not None and base is not None and delta is not None
    # Equal n on the merged intersection (T-25-02-pairing).
    assert len(cand) == len(base) == len(delta) == 50 * len(seasons)
    # Plan 25-01 internal-consistency invariant holds element-wise.
    assert np.allclose(delta, cand - base)
    # The positive offset means the paired delta is positive on average (candidate not worse).
    assert float(np.mean(delta)) > 0
    # Per-season delta keys carry the integer LIVE holdout seasons with equal n. Derived,
    # so the assertion is about the pairing rather than about which seasons the partition
    # happens to name today.
    assert set(candidate["per_season_clv_delta_values"]) == set(seasons)
    for season in seasons:
        season_delta = candidate["per_season_clv_delta_values"][season]
        assert season_delta is not None
        assert len(season_delta) == 50


@pytest.mark.integration
def test_drift_tripwire_aborts_on_baseline_drift() -> None:
    """Plan 25-02: the gate-time drift tripwire HARD-aborts when re-scored v1.0 drifts from config.

    A re-scored v1.0 baseline frame whose pooled WP CLV mean is far from the frozen
    config/gate.toml value must raise (the deployed artifacts drifted; the paired test would use a
    wrong baseline). The error names the recomputation tolerance so the abort is unambiguous.
    """
    cfg = deploy_gate.load_gate_config(REPO_ROOT / "config" / "gate.toml")
    # A baseline frame whose pooled mean is +0.5 -- nowhere near the frozen WP pooled mean
    # (~-0.0567), so the pooled-mean drift check must fire well outside the 5e-3 tolerance.
    n_per_season = 50
    rows_gid = []
    rows_season = []
    for season in (2021, 2022, 2023, 2024):
        for i in range(n_per_season):
            rows_gid.append(f"{season}_G{i}")
            rows_season.append(season)
    drifted = pd.DataFrame(
        {
            "game_id": rows_gid,
            "season": rows_season,
            "probability_clv": np.full(len(rows_gid), 0.5),
        }
    )
    with pytest.raises(ValueError, match="Drift tripwire ABORT"):
        promote._drift_tripwire("wp", drifted, cfg)


@pytest.mark.integration
def test_drift_tripwire_aborts_on_wrong_clv_column() -> None:
    """Plan 25-02: the drift tripwire aborts if the re-scored frame lacks the frozen CLV column.

    Codex MEDIUM: the CLV column identity is part of the drift check. A baseline frame missing the
    target's CLV_COLUMN_FOR column means the baseline was measured on a different metric than the
    gate reads -- a hard abort, not a tolerance question.
    """
    cfg = deploy_gate.load_gate_config(REPO_ROOT / "config" / "gate.toml")
    wrong_col = pd.DataFrame(
        {"game_id": ["2021_G0"], "season": [2021], "not_the_clv_column": [0.0]}
    )
    with pytest.raises(ValueError, match="lacks the frozen CLV column"):
        promote._drift_tripwire("wp", wrong_col, cfg)


@pytest.mark.integration
def test_drift_tripwire_aborts_on_empty_baseline_frame() -> None:
    """WR-01: a zero-row re-scored baseline frame is the most extreme drift -> loudest abort.

    np.mean([]) is nan and `abs(nan - frozen) > tol` evaluates False, so without the empty-frame
    guard the pooled-mean check silently no-ops on a degenerate baseline. The guard must HARD-abort
    on a re-score that produced ZERO has_closing_odds rows (the baseline the gate pairs against does
    not exist for the target), not rely solely on the downstream per-season exact-n check.
    """
    cfg = deploy_gate.load_gate_config(REPO_ROOT / "config" / "gate.toml")
    empty = pd.DataFrame(
        {
            "game_id": pd.Series([], dtype="object"),
            "season": pd.Series([], dtype="int64"),
            "probability_clv": pd.Series([], dtype="float64"),
        }
    )
    with pytest.raises(ValueError, match="ZERO"):
        promote._drift_tripwire("wp", empty, cfg)


@pytest.mark.integration
def test_missing_dir_guard_actionable_error(tmp_path: Path) -> None:
    """Plan 25-02: a missing production artifacts dir yields a NAMED-path actionable error.

    Gemini consensus #2: a missing/empty production artifacts dir must produce a clear error that
    names the missing path and points at the clean-checkout bootstrap remedy -- NOT an opaque
    load_model_artifact failure. Asserted for both an absent dir and an absent target subdir.
    """
    # (a) Absent production artifacts dir entirely.
    absent = tmp_path / "no_such_artifacts"
    with pytest.raises(FileNotFoundError, match="Production artifacts dir not found"):
        promote._assert_artifacts_dir_present("wp", absent, "wp_20260327_114739")

    # (b) Dir present but the target's version subdir / metadata is missing.
    present = tmp_path / "artifacts"
    present.mkdir(parents=True, exist_ok=True)
    with pytest.raises(FileNotFoundError, match="missing at"):
        promote._assert_artifacts_dir_present("wp", present, "wp_does_not_exist")

    # The bootstrap remedy is named so the error is actionable (Plan 25-05 / DIAGNOSIS-NOTES.md).
    try:
        promote._assert_artifacts_dir_present("wp", absent, "wp_x")
    except FileNotFoundError as exc:
        assert "bootstrap" in str(exc).lower(), (
            "the missing-dir error must point at the clean-checkout bootstrap remedy"
        )


@pytest.mark.integration
def test_non_regression_forced_pass_and_fail_hermetic(
    tmp_artifacts: Path, tmp_stage: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Plan 25-02: forced PASS/FAIL under floor_mode=non_regression via the bundle seam (hermetic).

    A bundle whose paired delta is ~zero/positive PASSES the non-regression gate (exit 0 on a
    bare dry-run); a bundle whose paired delta is significantly negative FAILS (non-zero exit) and
    leaves production latest.json byte-unchanged. This re-asserts the non_regression forced
    verdicts under the names the Plan 25-02 verify command filters on (-k non_regression).
    """
    # Forced PASS: a passing bundle -> exit 0 (no failing targets), production untouched (dry-run).
    _install_hermetic_stubs(monkeypatch, _passing_bundle)
    latest = tmp_artifacts / "latest.json"
    before = latest.read_bytes()
    rc_pass = promote.main(
        [
            "--artifacts-dir",
            str(tmp_artifacts),
            "--staging-dir",
            str(tmp_stage),
            "--skip-train",
        ]
    )
    assert rc_pass == 0, "an all-pass non_regression dry-run must exit 0"
    assert latest.read_bytes() == before, "dry-run must not touch production"

    # Forced FAIL: a negative-delta bundle -> non-zero exit, production byte-unchanged.
    _install_hermetic_stubs(monkeypatch, _negative_bundle)
    rc_fail = promote.main(
        [
            "--artifacts-dir",
            str(tmp_artifacts),
            "--staging-dir",
            str(tmp_stage),
            "--skip-train",
        ]
    )
    assert rc_fail != 0, (
        "a significantly-negative non_regression delta must FAIL the gate"
    )
    assert latest.read_bytes() == before, (
        "a forced-FAIL run must leave production latest.json byte-unchanged"
    )
