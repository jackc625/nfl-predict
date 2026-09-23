"""The blend re-validation left production alone (Phase 30, Plan 30-12, T-30-07 / T-30-14).

THE HAZARD, WHICH WAS NOT HYPOTHETICAL
---------------------------------------
``backtest.tune.run_comparison`` was a DIAGNOSTIC that rewrote the production swap surface by
default. Its final step called ``MarketBlender.save_blend_artifacts``, which at the time read,
mutated and atomically rewrote ``artifacts/latest.json`` UNCONDITIONALLY. Its ``baselines_dir``
parameter defaulted to ``data/baselines/v2.0``, so its comparison report and gating JSON landed
under the data tree.

WHAT CHANGED IN PLAN 33.2-24 (D33.2-10). ``run_comparison`` -- the fixed-versus-DYNAMIC
comparator this module's live tests describe -- is DELETED with the week-varying blend, and
``save_blend_artifacts`` gained ``update_latest: bool = False`` (mirroring
``save_model_artifact``'s D24-08 flag): the manifest is rewritten only when a caller says so.
The narrative below is kept because the three LIVE tests still witness what the real Plan 30-12
run did, against Phase-30 anchors this plan does not touch. The hermetic pair now runs over a
FIXED-WEIGHT, converter-bound blender, the negative control passes ``update_latest=True``
explicitly, and a new case asserts the DEFAULT does not rewrite.

Run as documented it would therefore break TWO Phase-30 constraints at once:

  * ``artifacts/latest.json`` MUST NOT be swapped through any surface other than
    ``scripts/promote_models.py --promote``; and
  * nothing MUST be written under ``data/`` outside the one sanctioned fingerprinted rebuild.

This is not a new defect. It is what the Phase-25 blend re-validation actually did -- the two files
``data/baselines/v2.0/comparison_dynamic_vs_static.md`` and
``data/baselines/v2.0/gating_dynamic_vs_static.json`` still carry that run's timestamps, and
ACTIVATION-READOUT.md's section title already discloses the blend-pointer rewrite. Phase 30's
constraints are stricter, so Plan 30-12 ran the comparison against a throwaway ``copytree`` of the
production artifacts directory with its report directory under ``outputs/``.

WHAT THIS MODULE PROVES, AND WHY IT IS NOT VACUOUS
---------------------------------------------------
A test that only asserts "production did not change" passes just as happily when the rewrite never
happened at all -- a broken comparison, an early return, a gating outcome where no target passed.
So the redirection is proven from BOTH sides:

  * ``test_blend_save_pointed_at_a_copy_leaves_the_original_untouched`` -- the copy's manifest blend
    key DID move and a new blend artifact dir appeared IN THE COPY, while the original is
    byte-identical. The rewrite is real AND contained.
  * ``test_blend_save_pointed_at_the_original_rewrites_it`` -- the NEGATIVE control. The same call
    aimed at the original, with ``update_latest=True``, moves the original. Without this, the
    first test could pass because ``save_blend_artifacts`` had quietly stopped writing manifests
    at all.
  * ``test_blend_save_by_default_leaves_the_manifest_alone`` -- the Plan 33.2-24 default: a save
    that does not ask to move the manifest leaves it byte-identical.

Those three are fully hermetic (``tmp_path`` fixtures, no repo state) so they never skip and they
survive a fresh checkout. Three further tests assert the LIVE outcome of the real Plan 30-12 run
against the git-tracked anchors in ``tests/phase30_state.py``; ``artifacts/``, ``artifacts_staging/``,
``outputs/`` and ``data/`` are all gitignored, so those anchors are the only durable record.

NOTE on the SHAPE of the data-tree assertion: the pinned digest covers every relative path AND every
file's sha256, not just the directory list. The Phase-25 run did not create a directory under
``data/baselines`` -- it OVERWROTE two files inside the existing ``v2.0`` directory. A
directory-only check would have called that clean.

Hermetic contract (T-24-20): every write in this module goes to ``tmp_path``; the live tests are
read-only. The recorded-manifest + sha256 comparison shape is copied from
``tests/integration/test_rollback.py``.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest

from models.blending import (
    BlendConfig,
    BlendProvenance,
    BlendWeights,
    EdgeThresholds,
    MarketBlender,
    TuningResult,
)
from tests.phase30_state import (
    DATA_BASELINES_TREE_SHA256,
    DEPLOYED_BLEND_VERSION,
    MANIFEST_SHA256_AFTER,
)

# Repo root resolved from this file: tests/integration/test_blend_revalidation_isolation.py.
REPO_ROOT = Path(__file__).resolve().parents[2]

# The production swap surface the re-validation must NOT have touched.
_PROD_ARTIFACTS = REPO_ROOT / "artifacts"
_PROD_LATEST = _PROD_ARTIFACTS / "latest.json"

# The throwaway copy the comparison was actually pointed at (Plan 30-12 Task 3).
_STAGING_COPY = REPO_ROOT / "artifacts_staging" / "blend_revalidation"

# The data tree the comparison's default report directory would have landed in.
_DATA_BASELINES = REPO_ROOT / "data" / "baselines"

# The redirected report directory, under the outputs tree instead.
_REPORT_DIR = REPO_ROOT / "outputs" / "blend_revalidation"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    """Return the hex sha256 of a file's bytes (stdlib hashlib only)."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_sha256(root: Path) -> str:
    """Digest an entire tree: every relative path plus every file's content hash, sorted.

    Catches a new FILE as well as a new DIRECTORY, which is the distinction that matters for the
    data-tree assertion -- the Phase-25 run overwrote files inside an existing directory rather
    than creating one.
    """
    digest = hashlib.sha256()
    for path in sorted(
        root.rglob("*"), key=lambda p: str(p.relative_to(root)).replace("\\", "/")
    ):
        rel = str(path.relative_to(root)).replace("\\", "/")
        if path.is_dir():
            digest.update(f"D:{rel}\n".encode())
        else:
            digest.update(f"F:{rel}:{_sha256(path)}\n".encode())
    return digest.hexdigest()


_CONVERTER_ID = "market_probability_20990101_000000"
_CONVERTER_SLOPE = 0.15
_SEEDED_BLEND = "blend_20990101_000000"


def _seed_artifacts_tree(root: Path, blend_version: str) -> Path:
    """Create a minimal artifacts tree: one fixed-weight blend, its converter, a latest.json.

    Mirrors the real artifact shape closely enough for ``MarketBlender.from_artifacts``: a
    ``blend_weights.json`` carrying fixed weights and the converter binding, beside the
    ``market_probability_*`` directory that binding names.
    """
    root.mkdir(parents=True, exist_ok=True)
    converter_dir = root / _CONVERTER_ID
    converter_dir.mkdir(parents=True, exist_ok=True)
    (converter_dir / "metadata.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "slope_beta": _CONVERTER_SLOPE,
                "walk_forward_slopes": {"2021": 0.14},
                "training_seasons": [2020, 2021],
                "n_games": 10,
                "input_digest": "0" * 64,
                "fitted_at": "2099-01-01T00:00:00+00:00",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    blend_dir = root / blend_version
    blend_dir.mkdir(parents=True, exist_ok=True)
    (blend_dir / "blend_weights.json").write_text(
        json.dumps(
            {
                "blender_version": "3.0",
                "weights": {"wp": 0.59, "ats": 0.55, "ou": 0.60},
                "market_probability_artifact_id": _CONVERTER_ID,
                "market_probability_slope_beta": _CONVERTER_SLOPE,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (root / "latest.json").write_text(
        json.dumps(
            {
                "wp": "wp_stub",
                "ats": "ats_stub",
                "ou": "ou_stub",
                "blend": blend_version,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return blend_dir


def _tuning_result() -> TuningResult:
    """The TuningResult shape the one fixed-weight fit hands to ``save_blend_artifacts``."""
    return TuningResult(
        weights=BlendWeights(
            wp_model_weight=0.59, ats_model_weight=0.55, ou_model_weight=0.60
        ),
        objective_by_target={
            "wp": "log_loss",
            "ats": "mean_absolute_error",
            "ou": "mean_absolute_error",
        },
        loss_by_target={"wp": 0.6, "ats": 10.0, "ou": 10.0},
        market_only_loss_by_target={"wp": 0.61, "ats": 10.1, "ou": 10.1},
        model_only_loss_by_target={"wp": 0.62, "ats": 10.2, "ou": 10.2},
        grid_by_target={},
        seasons_by_target={"wp": [2021], "ats": [2020, 2021], "ou": [2020, 2021]},
        n_games={"wp": 10, "ats": 10, "ou": 10},
        season_best_weight_by_target={},
    )


def _provenance() -> BlendProvenance:
    return BlendProvenance(
        gold_generation_digest="a" * 64,
        source_artifact_ids={"wp": "wp_stub", "ats": "ats_stub", "ou": "ou_stub"},
        tuning_corpus_rows=10,
        excluded_counts={"no_prelock_line": 0, "no_prior_fold_converter": 0},
        thread_limit=1,
    )


def _blender() -> MarketBlender:
    """A fixed-weight blender with a converter bound -- the only shape a blend now has."""
    return MarketBlender(
        config=BlendConfig(
            weights=BlendWeights(
                wp_model_weight=0.59, ats_model_weight=0.55, ou_model_weight=0.60
            ),
            edge_thresholds=EdgeThresholds(),
        ),
        market_probability_artifact_id=_CONVERTER_ID,
        market_probability_slope_beta=_CONVERTER_SLOPE,
    )


# ---------------------------------------------------------------------------
# Hermetic: the redirection contains a rewrite that genuinely happens
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_blend_save_pointed_at_a_copy_leaves_the_original_untouched(
    tmp_path: Path,
) -> None:
    """T-30-07: the blend save aimed at a COPY moves the copy and leaves the original byte-identical.

    This is the exact indirection Plan 30-12 applied to the real re-validation: copytree the whole
    production artifacts directory -- manifest AND deployed model dirs -- and point the comparison
    at the copy. Both halves are asserted:

      * the COPY's manifest blend key CHANGED and a new blend artifact dir appeared inside the copy
        (so the rewrite is REAL, not merely absent), and
      * the ORIGINAL manifest is byte-identical by sha256 and gained no new blend directory.
    """
    original = tmp_path / "artifacts"
    _seed_artifacts_tree(original, _SEEDED_BLEND)
    original_sha = _sha256(original / "latest.json")
    original_dirs = sorted(p.name for p in original.iterdir() if p.is_dir())

    copy = tmp_path / "artifacts_staging" / "blend_revalidation"
    copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(original, copy)
    assert _sha256(copy / "latest.json") == original_sha, (
        "pre-condition: the copy must start byte-identical to the original"
    )

    # A manifest-moving save, aimed at the COPY. It asks to move the manifest explicitly --
    # since Plan 33.2-24 the default does not -- so the containment below is proven against a
    # rewrite that genuinely happens.
    artifact_dir = _blender().save_blend_artifacts(
        _tuning_result(), copy, provenance=_provenance(), update_latest=True
    )

    # The rewrite is REAL and it landed in the copy.
    copy_manifest = json.loads((copy / "latest.json").read_text())
    assert copy_manifest["blend"] == artifact_dir.name, (
        "the copy's manifest blend key was not repointed at the newly saved artifact -- the "
        "rewrite did not happen, so the isolation assertion below would be vacuous"
    )
    assert copy_manifest["blend"] != _SEEDED_BLEND, (
        "the copy's manifest blend key did not move"
    )
    assert artifact_dir.parent == copy, (
        f"the new blend artifact dir landed at {artifact_dir}, outside the copy {copy}"
    )
    # The non-blend keys are preserved by the per-key write.
    for key in ("wp", "ats", "ou"):
        assert (
            copy_manifest[key]
            == json.loads((original / "latest.json").read_text())[key]
        ), f"the blend save disturbed the {key} pointer in the copy"

    # The ORIGINAL is untouched.
    assert _sha256(original / "latest.json") == original_sha, (
        "the blend save rewrote the ORIGINAL manifest despite being pointed at a copy -- the "
        "redirection did not contain the production swap (T-30-07)"
    )
    assert sorted(p.name for p in original.iterdir() if p.is_dir()) == original_dirs, (
        "a new blend artifact directory appeared beside the ORIGINAL manifest"
    )


@pytest.mark.integration
def test_blend_save_pointed_at_the_original_rewrites_it(tmp_path: Path) -> None:
    """NEGATIVE CONTROL: without the redirection, the same call moves the tree it is aimed at.

    This is what makes the test above non-vacuous. If ``save_blend_artifacts`` ever stopped writing
    manifests -- an early return, a swallowed exception, a refactor that dropped the manifest write
    -- the isolation assertion would still pass while proving nothing. Here the same call is aimed
    at the tree directly and MUST move it, which is precisely the production swap the real run had
    to be redirected away from.

    Plan 33.2-24: the rewrite is now OPT-IN, so this control passes ``update_latest=True``
    explicitly. Before, it asserted the rewrite happened BY DEFAULT -- which is exactly the
    behaviour that plan removed; the next test asserts the new default.
    """
    target = tmp_path / "artifacts"
    _seed_artifacts_tree(target, _SEEDED_BLEND)
    before_sha = _sha256(target / "latest.json")

    artifact_dir = _blender().save_blend_artifacts(
        _tuning_result(), target, provenance=_provenance(), update_latest=True
    )

    manifest = json.loads((target / "latest.json").read_text())
    assert manifest["blend"] == artifact_dir.name
    assert _sha256(target / "latest.json") != before_sha, (
        "save_blend_artifacts aimed straight at an artifacts tree did NOT rewrite its manifest; "
        "the isolation test above is therefore not proving containment of anything"
    )
    assert artifact_dir.parent == target


@pytest.mark.integration
def test_blend_save_by_default_leaves_the_manifest_alone(tmp_path: Path) -> None:
    """Plan 33.2-24: a save that does not ASK to move the manifest leaves it byte-identical.

    The new blend directory is still written -- the save is real -- but the production swap
    surface does not move. That is what lets the one fixed-weight fit write a CANDIDATE without
    deploying it; the swap is Plan 33.2-25's.
    """
    target = tmp_path / "artifacts"
    _seed_artifacts_tree(target, _SEEDED_BLEND)
    before_sha = _sha256(target / "latest.json")

    artifact_dir = _blender().save_blend_artifacts(
        _tuning_result(), target, provenance=_provenance()
    )

    assert artifact_dir.is_dir() and artifact_dir.parent == target
    assert _sha256(target / "latest.json") == before_sha, (
        "save_blend_artifacts rewrote latest.json without update_latest=True"
    )
    loaded = MarketBlender.from_artifacts(target, version=artifact_dir.name)
    assert loaded.config.weights == _blender().config.weights


# ---------------------------------------------------------------------------
# Live: the real Plan 30-12 re-validation left production alone
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_live_production_manifest_is_the_recorded_post_promotion_manifest() -> None:
    """The live ``artifacts/latest.json`` is byte-identical to the recorded post-promotion manifest.

    ``MANIFEST_SHA256_AFTER`` was pinned by Plan 30-11 at the moment of the phase's ONE authorised
    production write. The blend re-validation ran after that, through a tool that rewrites this
    exact file by default. If it had been run as documented, this digest would have moved.

    The blend pointer is asserted separately from the digest so a manifest re-serialization that
    preserved the pointer, and a pointer change that happened to preserve the digest, are told
    apart rather than conflated.
    """
    if not _PROD_LATEST.exists():
        pytest.skip(f"production manifest not present at {_PROD_LATEST}")

    manifest = json.loads(_PROD_LATEST.read_text())
    assert manifest["blend"] == DEPLOYED_BLEND_VERSION, (
        f"the deployed blend pointer moved: {manifest['blend']} != {DEPLOYED_BLEND_VERSION}. The "
        "blend re-validation swapped production outside scripts/promote_models.py (T-30-07)"
    )
    assert _sha256(_PROD_LATEST) == MANIFEST_SHA256_AFTER, (
        "artifacts/latest.json is not byte-identical to the manifest Plan 30-11 recorded after the "
        "authorised swap; something rewrote the sole production swap surface afterwards"
    )


@pytest.mark.integration
def test_live_revalidation_copy_carries_the_redirected_rewrite() -> None:
    """The real re-validation's blend-pointer rewrite landed in the COPY, not in production.

    The live counterpart of the hermetic pair above: the copy the comparison was pointed at carries
    a DIFFERENT blend pointer from production (so the rewrite genuinely happened and was
    redirected), while its wp/ats/ou pointers still match production (so the copy is otherwise the
    production state the comparison was supposed to be measuring), and the newly written blend
    artifact directory exists under the copy and NOT under ``artifacts/``.
    """
    copy_manifest_path = _STAGING_COPY / "latest.json"
    if not copy_manifest_path.exists() or not _PROD_LATEST.exists():
        pytest.skip(
            f"the Plan 30-12 re-validation copy is not present at {_STAGING_COPY} "
            "(gitignored runtime state)"
        )

    copy_manifest = json.loads(copy_manifest_path.read_text())
    prod_manifest = json.loads(_PROD_LATEST.read_text())

    assert copy_manifest["blend"] != prod_manifest["blend"], (
        "the copy's blend pointer equals production's -- the comparison's blend-artifact save "
        "never ran, so 'production is unchanged' proves nothing about redirection"
    )
    for key in ("wp", "ats", "ou"):
        assert copy_manifest[key] == prod_manifest[key], (
            f"the re-validation copy's {key} pointer diverged from production's; the comparison "
            "was not measuring the deployed models"
        )

    assert (_STAGING_COPY / copy_manifest["blend"]).is_dir(), (
        f"the copy's blend pointer {copy_manifest['blend']} does not resolve to a directory in "
        f"{_STAGING_COPY}"
    )
    assert not (_PROD_ARTIFACTS / copy_manifest["blend"]).exists(), (
        f"the re-validation's new blend artifact {copy_manifest['blend']} appeared under "
        f"{_PROD_ARTIFACTS} -- it was written into production, not into the copy"
    )


@pytest.mark.integration
def test_live_data_baselines_tree_is_unchanged_by_the_revalidation() -> None:
    """T-30-14: nothing new appeared under the data baselines tree, and nothing there was rewritten.

    ``run_comparison``'s ``baselines_dir`` defaults to ``data/baselines/v2.0``; Plan 30-12 redirected
    it to ``outputs/blend_revalidation``. The pinned digest covers every relative path AND every
    file's content hash, so it catches an overwritten file as well as a new directory -- which is
    the shape that matters, because the Phase-25 run overwrote two files inside an existing
    directory rather than creating one.
    """
    if not _DATA_BASELINES.is_dir():
        pytest.skip(f"data baselines tree not present at {_DATA_BASELINES}")

    assert _tree_sha256(_DATA_BASELINES) == DATA_BASELINES_TREE_SHA256, (
        "the data/baselines tree changed across the blend re-validation -- the comparison's report "
        "directory was not redirected under outputs/, violating the no-writes-under-data "
        "prohibition (T-30-14)"
    )
    # And the redirected report actually landed where it was sent.
    if _REPORT_DIR.is_dir():
        assert (_REPORT_DIR / "comparison_dynamic_vs_static.md").exists(), (
            f"the comparison report is missing from the redirected directory {_REPORT_DIR}"
        )
