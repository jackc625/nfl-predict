"""The SOLE production-swap entry point for per-target model promotion (Phase 24).

This orchestrator is the only sanctioned path that may swap a candidate model into
production ``artifacts/latest.json``. It is dry-run by DEFAULT: a bare
``python -m scripts.promote_models`` trains candidates to ``artifacts_staging/``, scores
them, runs the frozen significance-tested deploy gate, prints the per-target 2x2 readout
(candidate vs frozen v1.0), and swaps NOTHING in production. ``--promote`` is REQUIRED for
any production write, and even then only PASSING targets are swapped (per-key, via
``models.artifacts.update_manifest``, preserving the ``blend`` pointer and any non-promoted
target). A gate FAIL produces zero production swaps AND a non-zero exit code so automation/CI
observes the hard block.

Why a hard block (the D-17 precedent): in v2.0 an ungated swap shipped a regression -- the
deployed v1.0 pre-Elo WP/ATS models were retained only because per-target gating caught the
regression after the fact. v3.0 deliberately re-fits (lifting the v2.1 D-01 boundary), so this
gate is the safety rail: nothing ships that regresses. The frozen thresholds + baseline live in
the git-tracked ``config/gate.toml``; this script never loosens them.

The flow (D24-08/09/10/11):
  STEP 1   staging straight re-fit (subprocess: python -m models.train --no-tune, no Optuna)
           into ``artifacts_staging/`` -- never production.
  STEP 1b  write the STAGING manifest ``artifacts_staging/latest.json`` (the subprocess train
           no longer auto-writes it; Plan 24-01 made BaseTrainer.save() pass update_latest=False)
           via update_manifest(artifacts_dir=staging_dir). D24-09 ALLOWS this staging manifest;
           D24-08 protects only production ``artifacts/latest.json``. The staged_version map is
           resolved DETERMINISTICALLY here (newest by the dir-name timestamp, not filesystem
           mtime) and reused in STEP 4 -- the staging manifest must exist before STEP 2 because
           score_deployed_artifacts resolves the staged artifact via that manifest.
  STEP 2   score the staged candidates on canonical 2021-2024 gold
           (diagnose.score_deployed_artifacts, artifacts_dir=staging_dir).
  STEP 3   build candidate bundles through the SHARED deploy_gate.build_candidate_bundle (the one
           bundle shape promote/retrain/tests all use) and run deploy_gate.evaluate_target against
           the frozen gate.toml baseline; print the 2x2 readout.
  STEP 4   on --promote, swap ONLY passing targets into production via update_manifest
           (artifacts_dir=artifacts_dir); a FAIL never swaps and exits non-zero.

This script deliberately does NOT wire into the Friday orchestrator (out of scope).

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from backtest.diagnose import CLV_COLUMN_FOR, clv_significance, score_deployed_artifacts
from backtest.engine import BacktestEngine
from models import deploy_gate
from models.artifacts import update_manifest
from models.clv import compute_clv_for_predictions
from utils import get_logger

if TYPE_CHECKING:
    import pandas as pd

logger = get_logger(__name__)

# Targets promoted by this orchestrator (the production swap surface; "blend" is never
# touched by a re-fit -- update_manifest's per-key write preserves it).
_TARGETS: tuple[str, ...] = ("wp", "ats", "ou")

# The frozen walk-forward holdout window (matches BacktestConfig + diagnose + gate.toml).
# Staged candidates are scored on this window so the candidate metrics line up with the
# frozen baseline window in config/gate.toml.
_HOLDOUT_FIRST_SEASON = 2021
_HOLDOUT_LAST_SEASON = 2024

# D25-15 gate-time drift tripwire: RECOMPUTATION tolerance for the re-scored v1.0 aggregates
# vs the frozen config/gate.toml baseline. This is NOT a loosening of D25-15's "exact match"
# SEMANTIC intent -- the baseline MUST be the same artifacts, the same gold, and the same CLV
# column; drift in ANY of those is a hard abort, not a tolerance question. The tolerance covers
# only float/library recomputation jitter: a deterministic re-score on the same artifacts should
# reproduce the frozen numbers to within this band. The value mirrors the freshness band used by
# tests/integration/test_promote_models.py::test_frozen_baseline_matches_rescore (_FRESHNESS_TOL)
# and the test_diag_diagnosis anchor tolerance, so the gate-time abort uses the same recomputation
# noise band the committed freshness test already trusts.
_FRESHNESS_TOL = 5e-3


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Command-line arguments. None for sys.argv.

    Returns:
        Parsed arguments namespace with promote, artifacts_dir, staging_dir, skip_train.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Staged per-target model promotion. Dry-run by default (swaps nothing); "
            "--promote performs the conditional per-target production swap."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help=(
            "Perform the conditional per-target production swap "
            "(default: dry-run, swaps nothing)"
        ),
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Production artifacts dir -- the swap target (default: artifacts)",
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=Path("artifacts_staging"),
        help="Staging artifacts dir for the candidate re-fit (default: artifacts_staging)",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help=(
            "Reuse existing staging artifacts instead of re-training "
            "(keeps the gate path testable without a long train)"
        ),
    )
    return parser.parse_args(argv)


def _parse_version_timestamp(version_dir_name: str) -> datetime:
    """Parse the trailing ``{YYYYMMDD}_{HHMMSS}`` timestamp from an artifact dir name.

    Versioned artifact dirs are named ``{target}_{YYYYMMDD}_{HHMMSS}`` (e.g.
    ``wp_20260327_114739``). Staged-version selection sorts by THIS embedded timestamp, never
    by filesystem modification time: filesystem mtimes can tie (a fast re-fit writes all three
    targets in the same second) and a stale dir's mtime can mislead. Parsing the name is
    deterministic (review concern #10 / T-24-17).

    Args:
        version_dir_name: An artifact directory name like ``wp_20260327_114739``.

    Returns:
        The parsed ``datetime`` (used only as a sort key).

    Raises:
        ValueError: If the name does not end in a parseable ``YYYYMMDD_HHMMSS`` stamp.
    """
    # The last two underscore-separated fields are the date and time stamp; the target prefix
    # itself may contain no underscore, but splitting from the right is robust regardless.
    parts = version_dir_name.rsplit("_", 2)
    if len(parts) < 3:
        msg = (
            f"Cannot parse timestamp from artifact dir name '{version_dir_name}': "
            "expected '{target}_{YYYYMMDD}_{HHMMSS}'."
        )
        raise ValueError(msg)
    stamp = f"{parts[-2]}_{parts[-1]}"
    return datetime.strptime(stamp, "%Y%m%d_%H%M%S")  # noqa: DTZ007 (naive sort key only)


def _resolve_staged_version(
    target: str,
    staging_dir: Path,
    *,
    skip_train: bool,
) -> str | None:
    """Pick the newest staged candidate dir for a target by its DIR-NAME timestamp.

    Globs ``{staging_dir}/{target}_*`` and returns the dir whose embedded
    ``{YYYYMMDD}_{HHMMSS}`` stamp is the max (deterministic; not filesystem mtime -- review
    concern #10). The empty-glob CONTRACT is two-branch (review concern #2):

      * ``skip_train=False`` (the real path just ran a staging train): a target with no
        matching dir is a HARD ERROR -- the train was supposed to produce it.
      * ``skip_train=True`` (the test/reuse path): a missing dir is a documented SKIP -- the
        target gets no staged_version entry, is excluded from the gate loop, and can never be
        indexed in STEP 4 (so no KeyError).

    Args:
        target: One of "wp", "ats", "ou".
        staging_dir: The staging artifacts root.
        skip_train: Whether the reuse path is active (changes the empty-glob disposition).

    Returns:
        The chosen version dir name (e.g. ``wp_20260327_114739``), or None when skip_train is
        set and no candidate dir exists for the target.

    Raises:
        RuntimeError: When skip_train is False and the staging train produced no candidate dir
            for the target.
    """
    candidates = [p for p in staging_dir.glob(f"{target}_*") if p.is_dir()]
    if not candidates:
        if skip_train:
            logger.warning(
                "No staged candidate dir for target under --skip-train; excluding it",
                target=target,
                staging_dir=str(staging_dir),
            )
            return None
        msg = f"staging train did not produce a {target} candidate dir under {staging_dir}"
        raise RuntimeError(msg)
    newest = max(candidates, key=lambda p: _parse_version_timestamp(p.name))
    return newest.name


def _clear_staging_dir(staging_dir: Path) -> None:
    """Delete stale candidate dirs and any stale latest.json before a fresh staging train.

    Clearing guarantees only the freshly-trained candidates exist, so the deterministic
    newest-by-name resolution can never select a stale dir (review concern #10 / T-24-17).
    Called ONLY on the real (non --skip-train) path; --skip-train must keep the supplied
    staging dirs intact.

    Args:
        staging_dir: The staging artifacts root to clear (created if absent).
    """
    if not staging_dir.exists():
        staging_dir.mkdir(parents=True, exist_ok=True)
        return

    import shutil

    for target in _TARGETS:
        for stale in staging_dir.glob(f"{target}_*"):
            if stale.is_dir():
                shutil.rmtree(stale)
    stale_manifest = staging_dir / "latest.json"
    if stale_manifest.exists():
        stale_manifest.unlink()


def _promote_artifact_dir(
    target: str,
    version: str,
    staging_dir: Path,
    artifacts_dir: Path,
) -> None:
    """Copy a passing target's staged artifact dir into production (byte-identical, D25-06).

    ``update_manifest`` only rewrites the per-target POINTER in production ``latest.json``; it
    does NOT relocate the artifact files. The candidate was trained into -- and gate-scored from
    -- ``staging_dir/{version}`` (STEP 1/STEP 2 both use ``artifacts_dir=staging_dir``). When the
    staging dir differs from the production dir (the real run uses ``artifacts_staging`` vs
    ``artifacts``), pointing production ``latest.json`` at ``{version}`` without copying the dir
    leaves the deployed artifact UNRESOLVABLE: ``load_model_artifact(target)`` resolves
    ``artifacts/{version}`` and raises ``FileNotFoundError`` because the files only exist under
    staging. This copies the GATE-SCORED staging dir verbatim into production BEFORE the manifest
    swap, so the deployed artifact is byte-identical to the one the gate scored (D25-06) and is
    resolvable by every consumer.

    The copy is verbatim (``shutil.copytree``) -- no re-train, no re-serialize, so byte-identity is
    preserved. When staging and production are the SAME dir (e.g. ``--skip-train`` against the
    production dir, or a test that stages directly into production) the artifact is already in
    place and this is a no-op. A pre-existing production dir at the same version (re-promote of an
    identical version) is left untouched -- the version stamp makes a collision astronomically
    unlikely, and overwriting an in-place artifact would be the no-op staging==production case.

    Args:
        target: One of "wp", "ats", "ou" (for the log line).
        version: The staged artifact dir name to promote (e.g. ``wp_20260605_215552``).
        staging_dir: The staging artifacts root the candidate was trained/scored from.
        artifacts_dir: The production artifacts root (the swap target).

    Raises:
        FileNotFoundError: If the staged artifact dir does not exist (the gate scored it, so its
            absence here is an integrity failure, not a routine miss).
    """
    src = staging_dir / version
    dst = artifacts_dir / version
    if src.resolve() == dst.resolve():
        # staging == production: the gate-scored artifact is already in place (no-op).
        return
    if not src.is_dir():
        msg = (
            f"Staged artifact dir for '{target}' not found at '{src}'; cannot promote it into "
            f"production. The gate scored this exact dir, so its absence is an integrity failure."
        )
        raise FileNotFoundError(msg)
    if dst.exists():
        # Same version already present in production (re-promote of an identical stamp) -- the
        # gate-scored artifact is effectively in place; leave it untouched.
        return
    import shutil

    # Verbatim copy preserves byte-identity with the gate-scored staging artifact (D25-06).
    shutil.copytree(src, dst)


def _warn_skip_train_staleness(staging_dir: Path, *, promote: bool) -> None:
    """Print a loud staleness warning for each staged candidate under --skip-train.

    --skip-train reuses whatever already exists in ``staging_dir``; there is no guarantee those
    artifacts correspond to the current code/gold (review concern WR-06). _clear_staging_dir
    only runs on the real train path, so a stale dir from a previous (possibly buggy or
    pre-gold-rebuild) run can be scored and -- under ``--promote`` -- swapped into production.

    This makes the risk LOUD and visible: it reports each resolvable staged dir with its
    embedded ``{YYYYMMDD}_{HHMMSS}`` timestamp so the operator can see exactly how old the
    artifacts are. The strongest emphasis is reserved for ``--promote --skip-train`` (the
    production foot-gun). A full freshness sentinel (gold hash + git SHA) is intentionally out
    of scope here; this is the documented "loud warning + staged timestamp" mitigation.

    Args:
        staging_dir: The staging artifacts root being reused.
        promote: Whether the production swap is armed (raises the warning severity).
    """
    severity = "WARNING (production swap armed)" if promote else "NOTICE"
    print(
        f"  [{severity}] --skip-train reuses existing staging artifacts WITHOUT verifying "
        "they match the current code/gold."
    )
    for target in _TARGETS:
        candidates = [p for p in staging_dir.glob(f"{target}_*") if p.is_dir()]
        if not candidates:
            print(f"    {target}: (no staged dir found)")
            continue
        for cand in sorted(candidates, key=lambda p: p.name):
            try:
                stamp = _parse_version_timestamp(cand.name).isoformat(sep=" ")
            except ValueError:
                stamp = "unparseable timestamp"
            print(f"    {target}: {cand.name} (staged {stamp})")
    if promote:
        print(
            "    Confirm these staged artifacts are fresh before trusting the swap; re-run "
            "without --skip-train to train fresh candidates."
        )


def _load_gold_holdout(target: str, engine: BacktestEngine) -> pd.DataFrame:
    """Load the 2021-2024 gold holdout for a target via the canonical engine loader.

    Reuses ``BacktestEngine._load_features`` (season-filtered, canonical team mapping) then
    restricts to the 2021-2024 holdout so the staged candidate is scored on the SAME window the
    frozen gate.toml baseline was frozen on.

    Args:
        target: One of "wp", "ats", "ou".
        engine: A constructed BacktestEngine (loader-only use here).

    Returns:
        The 2021-2024 gold frame for the target.
    """
    df = engine._load_features(target)
    in_holdout = (df["season"] >= _HOLDOUT_FIRST_SEASON) & (
        df["season"] <= _HOLDOUT_LAST_SEASON
    )
    holdout: pd.DataFrame = df.loc[in_holdout].copy()
    return holdout


def _baseline_bundle(target: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """Flatten the frozen gate.toml baseline for a target into the evaluate_target shape.

    ``config/gate.toml`` stores the baseline NESTED (``[baseline.<t>.pooled]`` +
    ``[baseline.<t>.season.YYYY]``), but ``deploy_gate.evaluate_target`` reads a FLAT baseline
    bundle (``accuracy``/``ece``/``brier_score`` for WP, ``mae`` for ATS/OU). This adapts the
    nested config table to that flat shape, mapping the TOML ``brier`` key to the
    ``brier_score`` key evaluate_target reads. This does NOT build a candidate bundle -- the
    candidate side always flows through ``build_candidate_bundle`` (the single bundle seam);
    this only shapes the FROZEN baseline side for the comparison.

    Args:
        target: One of "wp", "ats", "ou".
        cfg: The loaded gate config (from ``deploy_gate.load_gate_config``).

    Returns:
        The flat baseline bundle evaluate_target consumes. Empty when the baseline table is
        unpopulated (Wave-1-tolerated shape); evaluate_target then records the secondary
        comparison as skipped.
    """
    target_cfg = cfg.get("baseline", {}).get(target, {})
    pooled = target_cfg.get("pooled", {})
    bundle: dict[str, Any] = {
        "mean": pooled.get("mean"),
        "t": pooled.get("t"),
        "p": pooled.get("p"),
    }
    if target == "wp":
        bundle["accuracy"] = pooled.get("accuracy")
        bundle["ece"] = pooled.get("ece")
        # gate.toml stores the calibration metric as "brier"; evaluate_target reads
        # "brier_score" (matching build_candidate_bundle's WP key).
        bundle["brier_score"] = pooled.get("brier")
    else:
        bundle["mae"] = pooled.get("mae")
    return bundle


def _assert_artifacts_dir_present(
    target: str, artifacts_dir: Path, version: str
) -> None:
    """Raise a clear, actionable error if a target's production artifact dir is missing.

    The paired baseline re-score (D25-15) loads the DEPLOYED v1.0 artifacts from the production
    artifacts dir; ``score_deployed_artifacts`` -> ``load_model_artifact`` would otherwise raise
    an OPAQUE failure if the dir (or its metadata) is absent. The deployed v1.0 dirs are required
    BOTH for the paired re-score AND as the rollback target (D25-17), so a missing dir is a
    deploy-blocking integrity problem, not a transient: it must surface a named-path error that
    points at the clean-checkout bootstrap remedy (documented in DIAGNOSIS-NOTES.md / RUNBOOK by
    Plan 25-05) rather than an opaque load failure (Gemini consensus concern #2).

    Codex no-artifact-deletion guard: this check intentionally runs BEFORE the re-score so a
    missing production dir cannot be silently treated as "nothing to re-score". NEVER delete the
    v1.0 artifact dirs -- they are the paired baseline AND the rollback target.

    Args:
        target: One of "wp", "ats", "ou".
        artifacts_dir: The production artifacts dir (the swap target + baseline re-score source).
        version: The production version dir name for this target (from latest.json).

    Raises:
        FileNotFoundError: If the production artifacts dir, the target's version subdir, or its
            metadata.json is missing/empty -- with a named-path message and the bootstrap remedy.
    """
    if not artifacts_dir.exists():
        msg = (
            f"Production artifacts dir not found at '{artifacts_dir}'. The paired baseline "
            f"re-score (and rollback) needs the DEPLOYED v1.0 artifacts here. Do NOT delete the "
            "v1.0 artifact dirs. On a fresh checkout, bootstrap the first artifacts/latest.json "
            "per the clean-checkout bootstrap step (see DIAGNOSIS-NOTES.md / RUNBOOK, Plan 25-05)."
        )
        raise FileNotFoundError(msg)

    version_dir = artifacts_dir / version
    metadata = version_dir / "metadata.json"
    if not version_dir.is_dir() or not metadata.exists():
        msg = (
            f"Production artifact for target '{target}' missing at '{version_dir}' "
            f"(expected metadata at '{metadata}'). The paired baseline re-score (and rollback) "
            "needs the DEPLOYED v1.0 artifact for this target. Do NOT delete the v1.0 artifact "
            "dirs. On a fresh checkout, bootstrap the first artifacts/latest.json per the "
            "clean-checkout bootstrap step (see DIAGNOSIS-NOTES.md / RUNBOOK, Plan 25-05)."
        )
        raise FileNotFoundError(msg)


def _production_versions(artifacts_dir: Path) -> dict[str, str]:
    """Read the per-target version pointers from the production ``latest.json`` manifest.

    Used by the missing-dir guard to resolve which v1.0 artifact subdir each target points at,
    so the guard can name the exact expected path.

    Args:
        artifacts_dir: The production artifacts dir containing ``latest.json``.

    Returns:
        ``{target: version_dir_name}`` for the gated targets present in the manifest.

    Raises:
        FileNotFoundError: If ``latest.json`` itself is absent (a missing production manifest is
            the clean-checkout bootstrap gap; surface it with the bootstrap remedy).
    """
    import json

    manifest = artifacts_dir / "latest.json"
    if not manifest.exists():
        msg = (
            f"Production manifest not found at '{manifest}'. The paired baseline re-score needs "
            "the DEPLOYED v1.0 artifacts the manifest points at. On a fresh checkout, bootstrap "
            "the first artifacts/latest.json per the clean-checkout bootstrap step (see "
            "DIAGNOSIS-NOTES.md / RUNBOOK, Plan 25-05)."
        )
        raise FileNotFoundError(msg)
    with manifest.open(encoding="utf-8") as f:
        return json.load(f)


def _score_baseline_clv(
    target: str,
    gold_df: pd.DataFrame,
    odds_df: pd.DataFrame,
    artifacts_dir: Path,
) -> pd.DataFrame:
    """Re-score the DEPLOYED v1.0 artifacts and compute per-game baseline CLV (the paired side).

    Mirrors the candidate-side scoring (STEP 2) but against the PRODUCTION artifacts dir: scores
    the deployed v1.0 artifact on the SAME gold the candidate was scored on, then computes the
    per-game CLV (``probability_clv`` for WP, ``line_clv`` for ATS/OU) via the same
    ``compute_clv_for_predictions`` path ``build_candidate_bundle`` uses internally. The returned
    frame carries ``game_id`` + ``season`` + the target's CLV column, restricted to games with
    closing odds -- the per-game baseline the candidate is paired against (D25-15).

    Args:
        target: One of "wp", "ats", "ou".
        gold_df: The SAME 2021-2024 gold holdout frame the candidate was scored on.
        odds_df: Normalized closing odds.
        artifacts_dir: The PRODUCTION artifacts dir (the deployed v1.0 swap surface).

    Returns:
        A ``has_closing_odds``-filtered frame with ``game_id``, ``season``, and the target's CLV
        column.
    """
    scored = score_deployed_artifacts(
        target, gold_df=gold_df, artifacts_dir=artifacts_dir
    )
    clv_df = compute_clv_for_predictions(scored, odds_df, target)
    valid = clv_df.loc[clv_df["has_closing_odds"]]
    col = CLV_COLUMN_FOR[target]
    return valid[["game_id", "season", col]].copy()


def _candidate_clv_frame(
    target: str,
    scored_df: pd.DataFrame,
    odds_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute the candidate per-game CLV frame the SAME way ``build_candidate_bundle`` does.

    ``build_candidate_bundle`` internally drops pre-existing CLV/odds columns, recomputes CLV via
    ``compute_clv_for_predictions``, and filters to ``has_closing_odds``; it does NOT expose the
    per-game frame. This re-derives that exact frame (``game_id`` + ``season`` + the target's CLV
    column) so the paired delta is built on the SAME population the bundle's pooled metrics used --
    no second scoring pass, just the same recompute on the already-scored frame.

    Args:
        target: One of "wp", "ats", "ou".
        scored_df: The candidate scored frame from ``score_deployed_artifacts`` (staging dir).
        odds_df: Normalized closing odds.

    Returns:
        A ``has_closing_odds``-filtered frame with ``game_id``, ``season``, and the target's CLV
        column.
    """
    pre_drop = [c for c in deploy_gate._CLV_ODDS_COLS if c in scored_df.columns]
    base = scored_df.drop(columns=pre_drop) if pre_drop else scored_df
    clv_df = compute_clv_for_predictions(base, odds_df, target)
    valid = clv_df.loc[clv_df["has_closing_odds"]]
    col = CLV_COLUMN_FOR[target]
    return valid[["game_id", "season", col]].copy()


def _drift_tripwire(
    target: str,
    baseline_valid: pd.DataFrame,
    cfg: dict[str, Any],
) -> None:
    """HARD-assert the re-scored v1.0 aggregates equal config/gate.toml BEFORE the paired test.

    D25-15 anchor (T-25-02-drift): the paired non-regression delta is only meaningful if the
    baseline side is the SAME frozen judge ``config/gate.toml`` describes. This re-derives the
    re-scored v1.0 pooled AND per-season CLV aggregates from ``baseline_valid`` and HARD-asserts
    they match the committed ``[baseline.<target>.*]`` values on ALL frozen fields, raising on ANY
    mismatch so the gate aborts before forming a delta against a drifted baseline (Codex MEDIUM:
    compare all fields, not only the means).

    Fields compared:
      * the CLV column identity -- ``CLV_COLUMN_FOR[target]`` is the column the re-score and the
        frozen baseline were both measured on; a different column means a different metric;
      * the pooled CLV mean;
      * each per-season CLV mean AND its per-season sample size ``n`` (the frozen block carries
        per-season ``n``; a different population is a hard drift signal).

    The numeric comparison uses ``_FRESHNESS_TOL`` (5e-3), DOCUMENTED at its definition as a
    RECOMPUTATION-noise tolerance -- NOT a loosening of D25-15's "exact match" SEMANTIC intent
    (same artifacts, same gold, same column). The sample-size comparison is EXACT (an integer
    population count cannot drift by float noise).

    Args:
        target: One of "wp", "ats", "ou".
        baseline_valid: The re-scored v1.0 per-game frame (``game_id``, ``season``, CLV column),
            ``has_closing_odds``-filtered, from ``_score_baseline_clv``.
        cfg: The loaded gate config (with int-normalized baseline season keys).

    Raises:
        ValueError: If the re-scored v1.0 frame is degenerate (zero has_closing_odds rows, the
            most extreme drift -- WR-01) or its aggregates drift from the frozen config on ANY
            field (CLV column identity, pooled mean, a per-season mean, or a per-season sample
            size).
    """
    frozen = cfg.get("baseline", {}).get(target, {})
    pooled_frozen = frozen.get("pooled", {})
    col = CLV_COLUMN_FOR[target]

    # Column-identity check: the frozen baseline and this re-score must be on the SAME CLV column.
    if col not in baseline_valid.columns:
        msg = (
            f"Drift tripwire ABORT for '{target}': the re-scored v1.0 frame lacks the frozen CLV "
            f"column '{col}' (CLV_COLUMN_FOR[{target!r}]). The baseline was measured on a "
            "different column than the gate now reads -- the paired test would compare two metrics."
        )
        raise ValueError(msg)

    # Pooled mean: re-derive from the re-scored v1.0 CLV and compare to the frozen pooled mean.
    # Empty-frame guard (WR-01): a zero-row re-score is the MOST extreme drift (the baseline the
    # gate pairs against does not exist for this target), so it must be the LOUDEST abort, not a
    # silent pass. np.mean([]) is nan and `abs(nan - frozen) > tol` evaluates False, so without
    # this guard the pooled check would no-op on a degenerate baseline and rely solely on the
    # downstream per-season exact-n check to catch it.
    clv_arr = baseline_valid[col].to_numpy()
    if clv_arr.size == 0:
        msg = (
            f"Drift tripwire ABORT for '{target}': re-scored v1.0 frame has ZERO "
            f"has_closing_odds rows on column '{col}'; the baseline is degenerate and "
            "cannot be compared to the frozen config. Re-freeze the baseline (Plan 25-05) "
            "or restore the deployed artifacts."
        )
        raise ValueError(msg)
    pooled_mean = float(np.mean(clv_arr))
    frozen_pooled_mean = pooled_frozen.get("mean")
    if frozen_pooled_mean is not None:
        drift = abs(pooled_mean - float(frozen_pooled_mean))
        if drift > _FRESHNESS_TOL:
            msg = (
                f"Drift tripwire ABORT for '{target}': re-scored v1.0 pooled CLV mean "
                f"{pooled_mean:.8f} drifted from the frozen config {float(frozen_pooled_mean):.8f} "
                f"by {drift:.3e} (> {_FRESHNESS_TOL:.0e} recomputation tolerance). The deployed "
                "artifacts no longer match config/gate.toml; the paired test would use a wrong "
                "baseline. Re-freeze the baseline (Plan 25-05) or restore the deployed artifacts."
            )
            raise ValueError(msg)

    # Per-season means AND sample sizes (Codex MEDIUM: compare per-season fields, not only pooled).
    season_frozen = frozen.get("season", {})
    for season in deploy_gate.HOLDOUT_SEASONS:
        season_block = season_frozen.get(int(season))
        if not season_block:
            continue
        slice_clv = baseline_valid.loc[
            baseline_valid["season"] == season, col
        ].to_numpy()
        # Sample size is an EXACT integer comparison -- a population-count change is a hard drift
        # signal that no float-noise tolerance should absorb.
        frozen_n = season_block.get("n")
        if frozen_n is not None and len(slice_clv) != int(frozen_n):
            msg = (
                f"Drift tripwire ABORT for '{target}' season {season}: re-scored v1.0 sample size "
                f"{len(slice_clv)} != frozen config n={int(frozen_n)}. The re-score covers a "
                "different game population than the frozen baseline; the paired test would compare "
                "mismatched populations. Re-freeze the baseline (Plan 25-05) or restore artifacts."
            )
            raise ValueError(msg)
        frozen_season_mean = season_block.get("mean")
        if frozen_season_mean is not None and len(slice_clv):
            season_mean = float(np.mean(slice_clv))
            drift = abs(season_mean - float(frozen_season_mean))
            if drift > _FRESHNESS_TOL:
                msg = (
                    f"Drift tripwire ABORT for '{target}' season {season}: re-scored v1.0 CLV mean "
                    f"{season_mean:.8f} drifted from the frozen config "
                    f"{float(frozen_season_mean):.8f} by {drift:.3e} "
                    f"(> {_FRESHNESS_TOL:.0e} recomputation tolerance). The deployed artifacts no "
                    "longer match config/gate.toml; re-freeze (Plan 25-05) or restore artifacts."
                )
                raise ValueError(msg)


def _populate_paired_delta_keys(
    target: str,
    candidate: dict[str, Any],
    candidate_valid: pd.DataFrame,
    baseline_valid: pd.DataFrame,
) -> None:
    """Populate the Plan 25-01 PINNED non-regression delta keys via merge-on-game_id pairing.

    The Plan 25-01 ``build_candidate_bundle`` ships ``baseline_clv_values`` / ``clv_delta_values``
    (and the per-season equivalents) as ``None`` placeholders; this populates them IN PLACE from a
    genuine per-game pairing (D25-15). The candidate per-game CLV and the re-scored v1.0 per-game
    CLV are merged on ``game_id`` so the delta is paired on the INTERSECTION of games both sides
    scored with closing odds (T-25-02-pairing: equal n on the merged set, not a flattened
    aggregate). The internal-consistency invariant Plan 25-01 asserts is preserved by construction:
    ``clv_delta_values == clv_values - baseline_clv_values`` element-wise on the merged order.

    Args:
        target: One of "wp", "ats", "ou".
        candidate: The candidate bundle from ``build_candidate_bundle`` (mutated in place).
        candidate_valid: The candidate per-game frame (``game_id``, ``season``, CLV column),
            ``has_closing_odds``-filtered.
        baseline_valid: The re-scored v1.0 per-game frame, ``has_closing_odds``-filtered.
    """
    col = CLV_COLUMN_FOR[target]
    paired = candidate_valid.merge(
        baseline_valid,
        on="game_id",
        how="inner",
        suffixes=("_cand", "_base"),
    )
    cand_clv = paired[f"{col}_cand"].to_numpy()
    base_clv = paired[f"{col}_base"].to_numpy()
    delta = cand_clv - base_clv

    # Pooled paired arrays (aligned by game_id on the intersection both sides scored).
    candidate["clv_values"] = cand_clv
    candidate["baseline_clv_values"] = base_clv
    candidate["clv_delta_values"] = delta
    pooled = clv_significance(cand_clv)
    candidate["mean"] = pooled["mean"]
    candidate["t"] = pooled["t"]
    candidate["p"] = pooled["p"]
    candidate["n"] = pooled["n"]

    # Per-season paired arrays (the merged frame carries season from BOTH sides; they are equal on
    # the intersection, so the candidate-side season suffix is the per-season slice key).
    season_col = "season_cand"
    per_season_baseline: dict[int, Any] = {}
    per_season_delta: dict[int, Any] = {}
    per_season_cand: dict[int, Any] = {}
    for season in deploy_gate.HOLDOUT_SEASONS:
        mask = paired[season_col] == season
        per_season_cand[int(season)] = paired.loc[mask, f"{col}_cand"].to_numpy()
        per_season_baseline[int(season)] = paired.loc[mask, f"{col}_base"].to_numpy()
        per_season_delta[int(season)] = (
            paired.loc[mask, f"{col}_cand"].to_numpy()
            - paired.loc[mask, f"{col}_base"].to_numpy()
        )
    candidate["per_season_clv_values"] = per_season_cand
    candidate["per_season_baseline_clv_values"] = per_season_baseline
    candidate["per_season_clv_delta_values"] = per_season_delta
    # Refresh the per-season raw-candidate significance bundles to the paired (intersection)
    # population so the absolute-vs-zero per-season verdict matches the paired set.
    candidate["per_season"] = {
        int(season): clv_significance(per_season_cand[int(season)])
        for season in deploy_gate.HOLDOUT_SEASONS
    }


def _fmt(value: Any) -> str:
    """Format a metric for the readout (4dp float, or 'N/A' for None)."""
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _print_target_readout(
    target: str,
    result: dict[str, Any],
) -> None:
    """Print the per-target 2x2 readout (candidate vs frozen v1.0) for one target.

    Reuses the ``[PASS]``/``[FAIL]`` idiom from ``scripts.retrain_models.print_gating_summary``.
    Prints the pooled CLV mean/p, the secondary metric (WP accuracy + ECE/Brier, ATS/OU MAE),
    the per-season CLV deltas as an INFORMATIONAL readout (Open-Q-A: per-season secondary is a
    readout, not a blocker), and every gate reason.

    Args:
        target: One of "wp", "ats", "ou".
        result: The evaluate_target result dict for the target.
    """
    candidate = result.get("candidate", {})
    baseline = result.get("baseline", {})
    status = "[PASS]" if result["passed"] else "[FAIL]"

    print(f"\n{target.upper()} -- {status}")
    cand_mean = candidate.get("mean")
    cand_p = candidate.get("p")
    base_mean = baseline.get("mean")
    print(
        f"  Pooled CLV: candidate mean={_fmt(cand_mean)} p={_fmt(cand_p)} "
        f"| frozen v1.0 mean={_fmt(base_mean)}"
    )

    if target == "wp":
        print(
            f"  Accuracy: candidate={_fmt(candidate.get('accuracy'))} "
            f"| v1.0={_fmt(baseline.get('accuracy'))}"
        )
        print(
            f"  ECE: candidate={_fmt(candidate.get('ece'))} "
            f"| v1.0={_fmt(baseline.get('ece'))}    "
            f"Brier: candidate={_fmt(candidate.get('brier_score'))} "
            f"| v1.0={_fmt(baseline.get('brier_score'))}"
        )
    else:
        print(
            f"  MAE: candidate={_fmt(candidate.get('mae'))} "
            f"| v1.0={_fmt(baseline.get('mae'))}"
        )

    # Per-season CLV deltas: informational readout only (Open-Q-A; never blocks here).
    per_season = result.get("per_season", {})
    if per_season:
        print("  Per-season CLV (readout only, not a secondary blocker):")
        for season in sorted(per_season):
            season_sig = per_season[season]
            print(
                f"    {season}: mean={_fmt(season_sig.get('mean'))} "
                f"p={_fmt(season_sig.get('p'))} n={season_sig.get('n')}"
            )

    for reason in result["reasons"]:
        print(f"  - {reason}")

    if result["passed"]:
        print(f"  >> Ship candidate {target.upper()}")
    else:
        print(f"  >> Keep v1.0 {target.upper()} (gate FAIL)")


def main(argv: list[str] | None = None) -> int:
    """Run the staged-promotion orchestrator and return a gate-driven exit code.

    Dry-run by default (swaps nothing); --promote performs the conditional per-target
    production swap. Returns 1 if ANY gated target fails the deploy gate, else 0 -- the hard
    block is observable to CI even in dry-run (D24-10).

    Args:
        argv: Command-line arguments. None for sys.argv.

    Returns:
        Exit code: 1 if any target failed the gate, else 0.
    """
    args = parse_args(argv)

    print("=" * 70)
    print("NFL Prediction System -- Staged Model Promotion")
    print("=" * 70)
    mode = (
        "PROMOTE (will swap passing targets)" if args.promote else "DRY-RUN (no swap)"
    )
    print(f"  Mode: {mode}")
    print(f"  Production artifacts dir: {args.artifacts_dir}")
    print(f"  Staging artifacts dir:    {args.staging_dir}")
    print(f"  Skip train (reuse staging): {args.skip_train}")
    print("=" * 70)

    # -- STEP 1: staging straight re-fit (no Optuna), into the staging dir only --
    print("\n[Step 1/4] Staging re-fit (straight, no Optuna) -> staging dir...")
    if args.skip_train:
        print("  --skip-train set: reusing existing staging artifacts.")
        # WR-06: --skip-train scores whatever already exists with no freshness check; make the
        # staleness LOUD and print each staged dir's embedded timestamp (strongest emphasis when
        # --promote is also set, the production foot-gun).
        _warn_skip_train_staleness(args.staging_dir, promote=args.promote)
    else:
        # Clear stale dirs first so newest-by-name resolution cannot pick a stale candidate.
        _clear_staging_dir(args.staging_dir)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "models.train",
                "--target",
                "all",
                "--artifacts-dir",
                str(args.staging_dir),
                "--no-tune",
            ],
            check=True,
        )
        print("  Staging re-fit complete.")

    # -- STEP 1b: write the STAGING manifest + deterministic staged_version resolution --
    # The subprocess train wrote NO artifacts_staging/latest.json (BaseTrainer.save() passes
    # update_latest=False, Plan 24-01). score_deployed_artifacts in STEP 2 resolves the staged
    # artifact via that manifest, so it MUST exist first. This writes the STAGING manifest ONLY
    # (artifacts_dir=args.staging_dir) -- it NEVER touches production (D24-08/09).
    print("\n[Step 1b/4] Writing staging manifest + resolving staged versions...")
    staged_version: dict[str, str] = {}
    for target in _TARGETS:
        chosen = _resolve_staged_version(
            target, args.staging_dir, skip_train=args.skip_train
        )
        if chosen is None:
            continue  # --skip-train: target excluded from the gate loop and any swap.
        staged_version[target] = chosen
        update_manifest(target, chosen, artifacts_dir=args.staging_dir)
        print(f"  {target}: {chosen}")
    if not staged_version:
        print("  No staged candidates resolved; nothing to gate.")
        # WR-06: under an armed --promote, finding zero staged candidates is an operator
        # error (e.g. --skip-train against an empty/wrong staging dir), not a clean no-op --
        # exit non-zero so automation/CI observes that the requested promotion did nothing.
        if args.promote:
            print(
                "  --promote was requested but no staged candidates exist to promote; "
                "exiting non-zero."
            )
            return 1
        return 0

    # -- STEP 2: score the staged candidates on canonical 2021-2024 gold --
    print("\n[Step 2/4] Scoring staged candidates on 2021-2024 gold...")
    engine = BacktestEngine()
    odds_df = engine._load_closing_odds()
    scored: dict[str, pd.DataFrame] = {}
    gold_holdout: dict[str, pd.DataFrame] = {}
    for target in staged_version:
        gold_holdout[target] = _load_gold_holdout(target, engine)
        scored[target] = score_deployed_artifacts(
            target, gold_df=gold_holdout[target], artifacts_dir=args.staging_dir
        )
        print(f"  {target}: scored {len(scored[target])} games")

    # -- STEP 3: build bundles via the SHARED builder, pair vs the re-scored v1.0 baseline, --
    # -- then run the gate; print the 2x2 readout. --
    # D25-15: the non-regression floor needs a PAIRED per-game CLV delta, so the deployed v1.0
    # artifacts are re-scored on the SAME gold as each candidate (the baseline side), guarded by
    # a missing-artifacts-dir check and a gate-time drift tripwire, then merged on game_id with the
    # candidate per-game CLV to populate the Plan 25-01 pinned delta keys. The whole baseline
    # pipeline runs ONLY when build_candidate_bundle shipped clv_delta_values as the None
    # placeholder (the real path); a forced-verdict bundle (the hermetic integration seam) ships
    # the delta keys pre-populated and is left untouched, so the gate's downstream consumers stay
    # testable without canonical gold or the deployed v1.0 dirs.
    print(
        "\n[Step 3/4] Building candidate bundles, pairing vs the re-scored v1.0 baseline, "
        "running the deploy gate..."
    )
    cfg = deploy_gate.load_gate_config()
    deploy_gate.validate_gate_config(cfg)

    gate_results: dict[str, dict[str, Any]] = {}
    for target in staged_version:
        candidate = deploy_gate.build_candidate_bundle(
            target, scored[target], odds_df, cfg
        )
        if candidate.get("clv_delta_values") is None:
            # Real path: re-score the deployed v1.0 baseline, guard a missing dir, abort on drift,
            # then pair on game_id to populate the pinned non-regression delta keys (D25-15).
            prod_versions = _production_versions(args.artifacts_dir)
            prod_version = prod_versions.get(target)
            if prod_version is None:
                msg = (
                    f"Production manifest has no '{target}' pointer; cannot re-score the v1.0 "
                    "baseline for the paired non-regression delta. Restore the manifest or "
                    "bootstrap it (see DIAGNOSIS-NOTES.md / RUNBOOK, Plan 25-05)."
                )
                raise KeyError(msg)
            _assert_artifacts_dir_present(target, args.artifacts_dir, prod_version)

            # Reuse the gold frame loaded for the candidate side (same window, same target) so the
            # baseline is paired on the SAME gold without a redundant parquet read.
            baseline_valid = _score_baseline_clv(
                target, gold_holdout[target], odds_df, args.artifacts_dir
            )
            # Gate-time drift tripwire: HARD-abort if the re-scored v1.0 drifted from gate.toml on
            # ANY frozen field (CLV column, pooled mean, per-season means + sample sizes).
            _drift_tripwire(target, baseline_valid, cfg)
            print(
                f"  {target}: baseline re-scored {len(baseline_valid)} games "
                "(drift tripwire PASS)"
            )
            candidate_valid = _candidate_clv_frame(target, scored[target], odds_df)
            _populate_paired_delta_keys(
                target, candidate, candidate_valid, baseline_valid
            )
        baseline = _baseline_bundle(target, cfg)
        gate_results[target] = deploy_gate.evaluate_target(
            target, candidate, baseline, cfg
        )

    print("\n" + "=" * 70)
    print("PER-TARGET DEPLOY GATE -- candidate vs frozen v1.0 (2x2 readout)")
    print("=" * 70)
    for target in staged_version:
        _print_target_readout(target, gate_results[target])
    print("\n" + "=" * 70)

    passing = [t for t in staged_version if gate_results[t]["passed"]]
    failing = [t for t in staged_version if not gate_results[t]["passed"]]
    print(f"SUMMARY: {len(passing)}/{len(staged_version)} gated targets pass")
    print(f"  Pass: {', '.join(t.upper() for t in passing) or 'none'}")
    print(f"  Fail: {', '.join(t.upper() for t in failing) or 'none'}")
    print("=" * 70)

    # -- STEP 4: conditional per-target production swap (only on --promote, only passing) --
    print("\n[Step 4/4] Production swap...")
    any_fail = bool(failing)
    if args.promote:
        for target in passing:
            # Copy the GATE-SCORED staged artifact dir into production FIRST so the manifest
            # pointer resolves to a real, byte-identical artifact (D25-06). update_manifest only
            # rewrites the pointer; without this copy the deployed version would be unresolvable
            # when staging_dir != artifacts_dir (the real run uses artifacts_staging vs artifacts).
            _promote_artifact_dir(
                target,
                staged_version[target],
                args.staging_dir,
                args.artifacts_dir,
            )
            # The SOLE production swapper: per-key update preserves blend + failing targets.
            update_manifest(
                target, staged_version[target], artifacts_dir=args.artifacts_dir
            )
            print(f"  Swapped production {target.upper()} -> {staged_version[target]}")
        if not passing:
            print("  No passing targets; production unchanged.")
        if any_fail:
            print(
                "  FAIL -> zero swaps for failing targets; non-zero exit "
                f"({', '.join(t.upper() for t in failing)})."
            )
    else:
        print("  Dry-run: no production swap performed (--promote required).")

    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
