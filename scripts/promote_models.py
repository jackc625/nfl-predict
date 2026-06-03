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

from backtest.diagnose import score_deployed_artifacts
from backtest.engine import BacktestEngine
from models import deploy_gate
from models.artifacts import update_manifest
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
    for target in staged_version:
        gold_df = _load_gold_holdout(target, engine)
        scored[target] = score_deployed_artifacts(
            target, gold_df=gold_df, artifacts_dir=args.staging_dir
        )
        print(f"  {target}: scored {len(scored[target])} games")

    # -- STEP 3: build bundles via the SHARED builder + run the gate; print the 2x2 readout --
    print("\n[Step 3/4] Building candidate bundles + running the deploy gate...")
    cfg = deploy_gate.load_gate_config()
    deploy_gate.validate_gate_config(cfg)

    gate_results: dict[str, dict[str, Any]] = {}
    for target in staged_version:
        candidate = deploy_gate.build_candidate_bundle(
            target, scored[target], odds_df, cfg
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
