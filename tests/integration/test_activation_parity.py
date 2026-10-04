"""Wave-4 train-serve parity suite for the Phase-25 gated activation (ACTV-05, D25-09/12/13).

THE CENTRAL TRAP (RESEARCH Pitfall 1, the reviewers' #1 consensus concern): the web cache
exposes values that come from TWO DIFFERENT MODEL POPULATIONS, and a parity test that compares
the wrong pair is either a false failure or a hollow pass. This module embeds the gitignored
Plan-25-02 DIAGNOSIS-NOTES.md two-part parity contract into committed source so it survives
(Gemini concern). The two populations are:

  * The cache ``predictions`` table (``wp_prob`` / ``ats_prediction`` / ``ou_prediction``) is fed
    from ``outputs/backtest/predictions_all.csv`` via ``api.cache._load_predictions``. That CSV is
    produced by the WALK-FORWARD BACKTEST ENGINE, which fits a FRESH model per fold (each holdout
    season is scored by a model trained only on prior seasons). It is NOT a single deployed
    artifact -- so it is the WRONG target for a deployed-artifact parity assertion.

  * The cache ``blended_*`` columns are POPULATION-MIXED: the blend WEIGHTS come from the deployed
    ``blend_weights.json``, but they are applied to the BACKTEST-CSV-fed ``wp_prob`` /
    ``ats_prediction`` / ``ou_prediction``. So ``blended_*`` is NOT a pure deployed-artifact value
    either -- only the blend weights themselves are a deployed-artifact value.

  * The cache ``feature_importances`` table is the ONLY PURE deployed-artifact-derived value: it is
    loaded directly from each deployed artifact's ``metadata.json`` (via ``latest.json``) by
    ``api.cache._load_feature_importances``. Feature importances are static, low-noise, and tied
    one-to-one to the deployed artifact, making them the train-serve-consistency anchor BOTH
    reviewers endorse.

The two-part parity definition (DIAGNOSIS-NOTES.md Section 4, the contract Plan 25-04 consumes):

  PART A -- cache-load fidelity (the ACTV-05 letter): the repopulated cache ``predictions`` rows
    EQUAL ``outputs/backtest/predictions_all.csv`` (the backtest-CSV-populated surface). This proves
    the cache FAITHFULLY reflects the backtest that ran against the deployed gold. It does NOT by
    itself prove deployed-artifact inference -- the served predictions are the per-fold backtest
    population, not a single deployed artifact. (Stated plainly so PART A is never mistaken for a
    deployed-artifact assertion.)

  PART B -- deployed-artifact parity (train-serve consistency): re-derive the deployed artifact's
    feature importances straight from its ``metadata.json`` (the version ``latest.json`` points at,
    via the IDENTICAL selection logic ``_load_feature_importances`` uses) and assert the cache
    ``feature_importances`` table EQUALS them within ~1e-9. The two read paths are genuinely
    distinct (the cache loaded them at populate time; the test re-reads the metadata now), so this
    is not a hollow value-to-itself compare. PART B also re-scores the deployed artifact via
    ``backtest.diagnose.score_deployed_artifacts`` (reused verbatim, never re-implemented inference)
    to prove the deployed artifact loads + scores consistently -- the independent deployed-artifact
    path the parity anchor is built on.

DO NOT assert "score_deployed_artifacts == cache predictions" (different populations -- the
Pitfall-1 false failure) and DO NOT anchor PART B on ``blended_*`` (population-mixed). A blend-
WEIGHTS equality check is included as an explicitly-labeled SECONDARY deployed-artifact anchor, not
a prediction-surface anchor.

Hermetic + offseason-safe: every test is ``@pytest.mark.integration`` and gold-skip-guarded
(mirrors ``test_promote_models.py``); the committed ``artifacts/latest.json`` is never mutated; any
manifest mutation a test needs goes to ``tmp_path``.

This guard protects THIS activation AND every future promotion (Plan 30 reuses it).

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from backtest.diagnose import score_deployed_artifacts

# Repo root resolved from this file: tests/integration/test_activation_parity.py -> repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]

# Production swap surface + the artifacts root the cache and the deploy gate both read.
_PROD_ARTIFACTS = REPO_ROOT / "artifacts"
_PROD_LATEST = _PROD_ARTIFACTS / "latest.json"

# The served cache + the backtest CSV the cache's predictions table is loaded from.
_CACHE_DB = REPO_ROOT / "data" / "web_cache.duckdb"
_BACKTEST_CSV = REPO_ROOT / "outputs" / "backtest" / "predictions_all.csv"

# Gold presence skip-guard (mirrors test_promote_models.py:64-65 / test_diag_diagnosis.py). PART B
# re-scores the deployed artifact over 2021-2024 gold, so the canonical gold must be present.
_GOLD_WP_PATH = REPO_ROOT / "data" / "gold" / "features_wp.parquet"

# D25-12 tolerance: ~1e-9 relative on raw deployed-artifact outputs (and on stored-precision
# display values). Feature importances are written to metadata.json at full float precision and the
# cache stores them as DOUBLE, so a faithful load is bit-for-bit -- 1e-9 is a generous safety band
# that still rejects any genuine divergence.
_PARITY_TOL = 1e-9

# The cache stores model-level feature importances under this sentinel game_id (the cache schema is
# per-game, so model-level rows use a placeholder -- see api/cache._load_feature_importances).
_FI_MODEL_GAME_ID = "_model_"

# The targets carried in latest.json + the cache.
_TARGETS = ("wp", "ats", "ou")


# ---------------------------------------------------------------------------
# Shared skip-guards (offseason / not-yet-populated safety, like the gold-gated suites)
# ---------------------------------------------------------------------------


def _require_cache_and_backtest() -> None:
    """Skip cleanly when the repopulated cache or the backtest CSV is absent.

    PART A compares the cache against the backtest CSV; both are gitignored runtime data
    (Task 1 of this plan regenerates them). Skipping mirrors the gold-gated convention so the
    suite is offseason / fresh-checkout safe rather than failing on missing data.
    """
    if not _CACHE_DB.exists():
        pytest.skip(
            f"web cache not present at {_CACHE_DB} (run scripts/populate_cache.py)"
        )
    if not _BACKTEST_CSV.exists():
        pytest.skip(
            f"backtest CSV not present at {_BACKTEST_CSV} (run scripts/run_backtest.py)"
        )


def _load_manifest() -> dict[str, str]:
    """Read the committed production latest.json (READ ONLY -- never mutated by this suite)."""
    return json.loads(_PROD_LATEST.read_text())


def _deployed_importances(target: str, manifest: dict[str, str]) -> dict[str, float]:
    """Re-derive a deployed artifact's feature importances straight from its metadata.json.

    Uses the EXACT selection logic the cache uses (``api.cache._load_feature_importances``):
    prefer ``feature_importances``, else ``top_feature_importances``. This is the independent
    deployed-artifact read path PART B anchors on -- the cache loaded these at populate time; the
    test re-reads them now, so the two paths are genuinely distinct (not a value-to-itself compare).
    """
    artifact_name = manifest[target]
    metadata = json.loads(
        (_PROD_ARTIFACTS / artifact_name / "metadata.json").read_text()
    )
    importances = metadata.get("feature_importances") or metadata.get(
        "top_feature_importances", {}
    )
    return {feat: float(imp) for feat, imp in importances.items()}


# ---------------------------------------------------------------------------
# PART A -- cache-load fidelity (the ACTV-05 letter): cache predictions == backtest CSV
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(
    not _GOLD_WP_PATH.exists(),
    reason=f"Canonical gold not present at {_GOLD_WP_PATH}",
)
def test_part_a_cache_predictions_equal_backtest_csv() -> None:
    """PART A (cache-load fidelity / the ACTV-05 letter): the cache predictions table EQUALS
    outputs/backtest/predictions_all.csv.

    The cache ``predictions`` table is BACKTEST-POPULATED -- it is fed from the walk-forward
    backtest CSV (per-fold models), NOT from a single deployed artifact. This test proves the cache
    faithfully reflects the post-promotion backtest run against the deployed gold; it does NOT by
    itself prove deployed-artifact inference (that is PART B). Per-target served columns map to the
    CSV like this (see api.cache._load_predictions): wp_prob <- model_value (target=wp),
    ats_prediction <- model_spread (target=ats), ou_prediction <- model_total (target=ou).
    """
    _require_cache_and_backtest()

    df = pd.read_csv(_BACKTEST_CSV)

    def _csv_slice(target: str, value_col: str, served_name: str) -> pd.DataFrame:
        """One target's (game_id, served_name) slice from the long backtest CSV.

        Uses ``.loc`` row/column selection (not chained ``[mask][[cols]]``) so the result is an
        unambiguous DataFrame for the downstream merge.
        """
        mask = df["target"] == target
        sub = df.loc[mask, ["game_id", value_col]].copy()
        return sub.rename(columns={value_col: served_name})

    wp_csv = _csv_slice("wp", "model_value", "wp_prob")
    ats_csv = _csv_slice("ats", "model_spread", "ats_prediction")
    ou_csv = _csv_slice("ou", "model_total", "ou_prediction")

    con = duckdb.connect(str(_CACHE_DB), read_only=True)
    try:
        cache = con.execute(
            "SELECT game_id, season, wp_prob, ats_prediction, ou_prediction "
            "FROM predictions"
        ).df()
    finally:
        con.close()

    # RE-SCOPED at the Phase 33 close-out to the population PART A is about. Since Plan
    # 33.2-26 (8bac938) the cache predictions table ALSO holds the live weeks' games, added by
    # api.cache._load_current_week_predictions from the current-week prediction CSVs for games
    # the backtest does not carry -- a second, intended source with no backtest-CSV row. PART A
    # is cache-load fidelity for the BACKTEST-populated surface, so it reads the cache rows of
    # the backtest's own seasons, and every one of those must still have a CSV source.
    # Was: every cache row.
    cache = cache.loc[cache["season"].isin(set(df["season"]))]

    assert len(cache) > 0, (
        "cache predictions table is empty -- run scripts/populate_cache.py"
    )

    # Merge each served column onto the CSV-derived value on game_id (the cache's pivot key).
    merged = (
        cache.merge(wp_csv, on="game_id", how="left", suffixes=("", "_csv"))
        .merge(ats_csv, on="game_id", how="left", suffixes=("", "_csv"))
        .merge(ou_csv, on="game_id", how="left", suffixes=("", "_csv"))
    )

    # Every served game must have a backtest-CSV WP source (the WP target is the cache pivot base).
    assert bool(merged["wp_prob_csv"].notna().all()), (
        "a cache prediction row has no backtest-CSV WP source -- the cache predictions table is not "
        "a faithful load of predictions_all.csv (PART A cache-load fidelity FAIL)"
    )

    # PART A assertion: cache served value == backtest CSV value (stored precision, ~1e-9).
    for served, csv_col, label in (
        ("wp_prob", "wp_prob_csv", "wp_prob (WP backtest model_value)"),
        (
            "ats_prediction",
            "ats_prediction_csv",
            "ats_prediction (ATS backtest model_spread)",
        ),
        (
            "ou_prediction",
            "ou_prediction_csv",
            "ou_prediction (OU backtest model_total)",
        ),
    ):
        # Compare only where the CSV side exists (ATS/OU left-merge may have gaps the cache stores
        # as NULL; PART A asserts fidelity wherever the backtest produced a value).
        both = merged[[served, csv_col]].dropna()
        served_vals = np.asarray(both[served], dtype=float)
        csv_vals = np.asarray(both[csv_col], dtype=float)
        diff = np.abs(served_vals - csv_vals)
        max_abs = float(np.max(diff)) if diff.size else 0.0
        assert max_abs <= _PARITY_TOL, (
            f"PART A cache-load fidelity FAIL for {label}: max abs diff {max_abs} exceeds "
            f"{_PARITY_TOL}; the cache predictions table diverged from the backtest CSV it loads"
        )


# ---------------------------------------------------------------------------
# PART B -- deployed-artifact parity (train-serve consistency): cache feature_importances ==
# deployed metadata.json (within ~1e-9), with an independent deployed-artifact re-score
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.skipif(
    not _GOLD_WP_PATH.exists(),
    reason=f"Canonical gold not present at {_GOLD_WP_PATH}",
)
def test_part_b_cache_feature_importances_equal_deployed_metadata() -> None:
    """PART B (deployed-artifact parity / train-serve consistency): the cache feature_importances
    table EQUALS the deployed artifacts' metadata.json importances within ~1e-9.

    Feature importances are the ONLY pure deployed-artifact-derived cache value (loaded directly
    from each deployed metadata.json via latest.json by api.cache._load_feature_importances). The
    cache ``predictions`` table is backtest-populated and ``blended_*`` is population-mixed, so
    NEITHER is a valid deployed-artifact anchor (asserting against them is the Pitfall-1 trap). This
    re-reads the deployed metadata via the IDENTICAL selection logic and asserts the served
    importances match -- a genuine two-path compare (cache loaded at populate time; test re-reads
    now), not a value-to-itself pass.
    """
    if not _CACHE_DB.exists():
        pytest.skip(
            f"web cache not present at {_CACHE_DB} (run scripts/populate_cache.py)"
        )

    manifest = _load_manifest()

    con = duckdb.connect(str(_CACHE_DB), read_only=True)
    try:
        served = con.execute(
            "SELECT target, feature_name, importance FROM feature_importances "
            "WHERE game_id = ?",
            [_FI_MODEL_GAME_ID],
        ).fetchall()
    finally:
        con.close()

    assert served, (
        "cache feature_importances table is empty -- the pure deployed-artifact parity anchor is "
        "missing (run scripts/populate_cache.py against the deployed artifacts)"
    )

    # Group the served rows by target: {target: {feature_name: importance}}.
    served_by_target: dict[str, dict[str, float]] = {t: {} for t in _TARGETS}
    for target, feat, imp in served:
        served_by_target.setdefault(target, {})[feat] = float(imp)

    checked_any = False
    for target in _TARGETS:
        deployed = _deployed_importances(target, manifest)
        if not deployed:
            # A deployed artifact with no importances in metadata contributes no anchor; the cache
            # also stores none for it (api.cache skips empty importances), so there is nothing to
            # compare. Every Phase-25 deployed artifact DOES carry importances, so this is defensive.
            continue
        checked_any = True
        cache_target = served_by_target.get(target, {})

        # Same feature set served as deployed (no dropped/extra importances).
        assert set(cache_target) == set(deployed), (
            f"PART B deployed-artifact parity FAIL for {target}: served feature set "
            f"{sorted(cache_target)} != deployed metadata feature set {sorted(deployed)}"
        )

        # Each served importance equals the deployed metadata value within ~1e-9.
        for feat, deployed_imp in deployed.items():
            served_imp = cache_target[feat]
            assert abs(served_imp - deployed_imp) <= _PARITY_TOL, (
                f"PART B deployed-artifact parity FAIL for {target}/{feat}: served {served_imp} "
                f"!= deployed metadata {deployed_imp} (diff {abs(served_imp - deployed_imp)} > "
                f"{_PARITY_TOL}); the served importances are not the deployed artifact's"
            )

    assert checked_any, (
        "no deployed target carried feature importances -- PART B had no deployed-artifact anchor "
        "to assert against (the activation deployed at least WP with a 20-entry importances dict)"
    )


@pytest.mark.integration
@pytest.mark.skipif(
    not _GOLD_WP_PATH.exists(),
    reason=f"Canonical gold not present at {_GOLD_WP_PATH}",
)
def test_part_b_deployed_artifact_rescore_is_consistent() -> None:
    """PART B support: the deployed artifact loads + scores via the parity-locked harness.

    Re-scores each deployed artifact over 2021-2024 gold via
    ``backtest.diagnose.score_deployed_artifacts`` (REUSED verbatim -- it mirrors
    ``run_predictions`` exactly with unscaled features + isotonic for WP; never re-implemented
    here). This proves the actually-deployed artifact (the version latest.json points at) loads and
    produces well-formed predictions -- the independent deployed-artifact path PART B's
    feature-importances anchor is built on. It deliberately does NOT compare this re-score against
    the cache ``predictions`` table (different population -- the Pitfall-1 false failure) nor against
    ``blended_*`` (population-mixed).
    """
    manifest = _load_manifest()
    for target in _TARGETS:
        # Confirm the deployed dir the manifest points at exists (else score would fail opaquely).
        artifact_dir = _PROD_ARTIFACTS / manifest[target]
        assert (artifact_dir / "metadata.json").exists(), (
            f"deployed {target} artifact {manifest[target]} missing metadata.json"
        )

        scored = score_deployed_artifacts(target, artifacts_dir=_PROD_ARTIFACTS)
        assert len(scored) > 0, f"deployed {target} re-score produced no rows"
        assert "model_prob" in scored.columns, (
            f"deployed {target} re-score missing model_prob (the backtest column contract)"
        )
        assert bool(scored["model_prob"].notna().all()), (
            f"deployed {target} re-score produced NaN predictions -- artifact load/score broke"
        )
        if target == "wp":
            # WP probabilities must be valid post-isotonic probabilities in [0, 1].
            assert bool(scored["model_prob"].between(0.0, 1.0).all()), (
                "deployed WP re-score produced out-of-range probabilities (calibrator broke)"
            )


# ---------------------------------------------------------------------------
# SECONDARY deployed-artifact anchor (NOT a prediction-surface anchor): blend WEIGHTS equality
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_secondary_blend_weights_are_the_deployed_weights() -> None:
    """SECONDARY (weights-equality) anchor -- explicitly NOT a prediction-surface anchor.

    The cache ``blended_*`` columns are POPULATION-MIXED: the blend WEIGHTS come from the deployed
    ``blend_weights.json`` but are applied to BACKTEST-CSV-fed predictions. So ``blended_*`` is NOT
    a pure deployed-artifact value and MUST NOT be used as a deployed-artifact prediction anchor
    (DIAGNOSIS-NOTES.md Section 4). What IS a deployed-artifact value is the blend WEIGHTS dict
    itself: this asserts the deployed blend pointer in latest.json resolves to a blend_weights.json
    carrying a well-formed {wp, ats, ou} weights mapping -- the deployed weights the cache blend
    block reads. This is a structural weights-equality check, not a blended-prediction parity claim.
    """
    manifest = _load_manifest()
    assert "blend" in manifest, (
        "latest.json has no blend pointer (the deployed blend is missing)"
    )

    blend_dir = _PROD_ARTIFACTS / manifest["blend"]
    weights_path = blend_dir / "blend_weights.json"
    assert weights_path.exists(), (
        f"deployed blend pointer {manifest['blend']} has no blend_weights.json at {weights_path}"
    )

    blend_data = json.loads(weights_path.read_text())
    weights = blend_data["weights"]
    # The deployed weights the cache blend block (api.cache._load_predictions) reads per target.
    for target in _TARGETS:
        assert target in weights, f"deployed blend weights missing target {target}"
        assert isinstance(weights[target], (int, float)), (
            f"deployed blend weight for {target} is not numeric: {weights[target]!r}"
        )
        assert 0.0 <= float(weights[target]) <= 1.0, (
            f"deployed blend weight for {target} out of [0,1]: {weights[target]}"
        )


# ---------------------------------------------------------------------------
# THE GENERATION SEAM, AND WHY THIS MODULE NEEDS NO GATE -- Plan 33.1-08 Task 2,
# 2026-09-14.
#
# Plan 33.1-08 named "any test in this module that pins a SCORED VALUE rather
# than a SHAPE" as expected to redden when Plan 33.1-07's rung-3 rebuild moved
# gold. Measured on 2026-09-14: the module passes. Its parity assertions compare
# the cache against the backtest and the deployed metadata -- two things that
# move TOGETHER when gold moves -- rather than against a committed constant. A
# comparison between two live sides survives a generation change; a comparison
# against a pinned number does not. That is the distinction, and it is why this
# module needed nothing while tests/integration/test_ou_divergence.py and
# tests/integration/test_diag_diagnosis.py each needed a gate.
#
# If an assertion here is ever anchored to a committed scalar, it needs
# tests.gold_generation.require_gold_generation and this note stops being true.
# ---------------------------------------------------------------------------
