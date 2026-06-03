"""Retrain all models with Optuna-tuned hyperparameters and compare to the frozen baseline.

Runs:
1. Optuna hyperparameter tuning for WP, ATS, O/U (100 trials each)
2. Full backtest with retrained models
3. v2.0 baseline capture at data/baselines/v2.0/ (human-readable comparison artifacts)
4. Comparison template fill with deltas
5. Per-target gating decision via the SHARED models.deploy_gate

ENFORCING ROLE (Phase 24, D24-13 + review concern #7): this script is the Optuna-tuned
COMPARISON harness. Its per-target gate decision now delegates to the ONE shared
``models.deploy_gate`` -- the SAME ``build_candidate_bundle`` + ``evaluate_target`` that
``scripts/promote_models.py`` uses -- judging the backtest candidate against the FROZEN
``config/gate.toml`` baseline. There is exactly one gate implementation and one candidate
bundle shape; the legacy ``headline_clv`` gate path and the gitignored ``data/baselines/``
read for the GATE DECISION are gone (the ``data/baselines/`` read survives only for the
human-readable comparison TEMPLATE, never for the deploy decision). ``main()`` returns
non-zero on a gate FAIL, agreeing with ``promote_models.py`` that the gate is a hard block.

Note on the candidate scored frame: the backtest engine fits fresh per-fold models and
already produces a scored candidate frame (``BacktestResults.all_predictions[target]`` --
game_id/season/model_prob/actual + merged odds), so that frame is fed directly to the shared
``build_candidate_bundle`` (the engine IS this harness's scorer, the analog of promote's
staged-artifact scoring). ``score_deployed_artifacts`` is NOT used here because retrain has
no single deployed-artifact dir -- ``data/baselines/v2.0`` holds metrics/predictions, not
model.pkl artifacts; the deployed-artifact gate path is ``promote_models.py``'s.

Usage:
    python scripts/retrain_models.py
    python scripts/retrain_models.py --n-trials 50  # fewer trials for quick test
    python scripts/retrain_models.py --output-dir data/baselines/v2.0

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from backtest.engine import BacktestConfig, BacktestEngine, BacktestResults
from models import deploy_gate
from scripts.capture_baseline import BaselineCapture
from scripts.promote_models import _baseline_bundle
from utils import get_logger

if TYPE_CHECKING:
    import pandas as pd

logger = get_logger(__name__)

# Default trial count per D-02
_DEFAULT_N_TRIALS = 100


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Command-line arguments. None for sys.argv.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Retrain all models with Optuna-tuned hyperparameters",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=_DEFAULT_N_TRIALS,
        help=f"Number of Optuna trials per target (default: {_DEFAULT_N_TRIALS})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/baselines/v2.0"),
        help="Output directory for v2.0 baseline files (default: data/baselines/v2.0)",
    )
    return parser.parse_args(argv)


def _set_n_trials_on_trainers(n_trials: int) -> int:
    """Patch the default n_trials in BaseTrainer.tune_hyperparameters.

    The BacktestEngine creates fresh trainer instances and calls
    train_and_evaluate(), which internally calls tune_hyperparameters()
    with a default of 100 trials. To override this from the CLI,
    we monkey-patch the default argument.

    Args:
        n_trials: Number of trials to set.

    Returns:
        Previous default value.
    """
    from models.trainers.base import BaseTrainer

    # Get the current default for tune_hyperparameters
    original_method = BaseTrainer.tune_hyperparameters

    def patched_tune(
        self,
        X_train,
        y_train,
        n_trials_arg: int = n_trials,
        season_week_df=None,
    ):
        return original_method(
            self,
            X_train,
            y_train,
            n_trials=n_trials_arg,
            season_week_df=season_week_df,
        )

    BaseTrainer.tune_hyperparameters = patched_tune
    return n_trials


def run_retraining(n_trials: int) -> BacktestResults:
    """Run backtest with Optuna-tuned hyperparameters.

    The BacktestEngine creates fresh trainer instances per season.
    Each trainer's tune_hyperparameters() delegates to OptunaTuner
    which uses SQLite-backed studies for resumability.

    Args:
        n_trials: Number of Optuna trials per target.

    Returns:
        BacktestResults with metrics from retrained models.
    """
    logger.info(
        "Starting retraining pipeline",
        n_trials=n_trials,
    )

    # Patch n_trials on trainers
    _set_n_trials_on_trainers(n_trials)

    config = BacktestConfig(
        holdout_seasons=[2021, 2022, 2023, 2024],
        first_data_season=2018,
        targets=["wp", "ats", "ou"],
        max_backtest_season=2024,
    )

    engine = BacktestEngine(config)
    results = engine.run()

    logger.info(
        "Retraining complete",
        headline_clv=results.headline_clv,
    )

    return results


def capture_v2_baseline(output_dir: Path) -> Path:
    """Capture v2.0 baseline using BaselineCapture.

    Since Optuna studies are SQLite-backed with load_if_exists=True,
    re-running the backtest will find completed trials and skip
    optimization, making this fast.

    Args:
        output_dir: Directory for v2.0 baseline files.

    Returns:
        Path to the output directory.
    """
    logger.info("Capturing v2.0 baseline", output_dir=str(output_dir))

    config = BacktestConfig(
        holdout_seasons=[2021, 2022, 2023, 2024],
        first_data_season=2018,
        targets=["wp", "ats", "ou"],
        max_backtest_season=2024,
    )

    capture = BaselineCapture(output_dir=output_dir, config=config)
    return capture.run()


def load_baseline_metrics(version_dir: Path) -> dict[str, dict]:
    """Load metrics JSON files for all targets from a baseline directory.

    Args:
        version_dir: Path to baseline version directory (e.g., data/baselines/v1.0).

    Returns:
        Dict mapping target name to metrics dict.

    Raises:
        FileNotFoundError: If any metrics file is missing.
    """
    metrics = {}
    for target in ["wp", "ats", "ou"]:
        metrics_path = version_dir / f"metrics_{target}.json"
        if not metrics_path.exists():
            msg = f"Metrics file not found: {metrics_path}"
            raise FileNotFoundError(msg)
        metrics[target] = json.loads(metrics_path.read_text())
    return metrics


def _format_delta(v2_val: float, v1_val: float) -> str:
    """Format a delta value with sign prefix.

    Args:
        v2_val: v2.0 metric value.
        v1_val: v1.0 metric value.

    Returns:
        Formatted string like "+0.0123" or "-0.0456".
    """
    delta = v2_val - v1_val
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.4f}"


def _extract_avg_metric(metrics: dict, metric_key: str) -> float | None:
    """Extract average of a metric across seasons from baseline metrics.

    Args:
        metrics: Baseline metrics dict for one target.
        metric_key: Key to look up in per-season results.

    Returns:
        Average value, or None if metric not available.
    """
    values = []
    for season_data in metrics.get("per_season_results", []):
        season_metrics = season_data.get("metrics", {})
        if metric_key in season_metrics:
            values.append(season_metrics[metric_key])
    return sum(values) / len(values) if values else None


def fill_comparison_template(
    v1_dir: Path,
    v2_dir: Path,
) -> str:
    """Fill the comparison template with v2.0 metrics and deltas.

    Reads v1.0 and v2.0 metrics, computes deltas, and generates
    a filled comparison table.

    Args:
        v1_dir: Path to v1.0 baseline directory.
        v2_dir: Path to v2.0 baseline directory.

    Returns:
        Filled comparison template as markdown string.
    """
    v1 = load_baseline_metrics(v1_dir)
    v2 = load_baseline_metrics(v2_dir)

    lines = [
        "# Baseline Comparison: v1.0 vs v2.0",
        "",
        "| Metric | WP v1.0 | WP v2.0 | WP Delta | ATS v1.0 | ATS v2.0 | ATS Delta | OU v1.0 | OU v2.0 | OU Delta |",
        "|--------|---------|---------|----------|----------|----------|-----------|---------|---------|----------|",
    ]

    # Row: Headline CLV
    wp_clv_v1 = v1["wp"]["headline_clv"]
    wp_clv_v2 = v2["wp"]["headline_clv"]
    ats_clv_v1 = v1["ats"]["headline_clv"]
    ats_clv_v2 = v2["ats"]["headline_clv"]
    ou_clv_v1 = v1["ou"]["headline_clv"]
    ou_clv_v2 = v2["ou"]["headline_clv"]

    lines.append(
        f"| Headline CLV | {wp_clv_v1} | {wp_clv_v2} | {_format_delta(wp_clv_v2, wp_clv_v1)} "
        f"| {ats_clv_v1} | {ats_clv_v2} | {_format_delta(ats_clv_v2, ats_clv_v1)} "
        f"| {ou_clv_v1} | {ou_clv_v2} | {_format_delta(ou_clv_v2, ou_clv_v1)} |"
    )

    # Per-season CLV rows are intentionally omitted (WR-03): the baseline metrics JSON carries
    # only the headline (pooled) CLV per target, not per-season CLV, so these rows could only
    # ever render "N/A". The significance-tested per-season CLV floor lives in the deploy gate
    # (deploy_gate.per_season_clv); this comparison template reports the headline CLV row above.

    # Row: Average MAE
    for metric_label, metric_key in [("Avg MAE", "mae"), ("Hit Rate", "accuracy")]:
        cells = []
        for target in ["wp", "ats", "ou"]:
            v1_val = _extract_avg_metric(v1[target], metric_key)
            v2_val = _extract_avg_metric(v2[target], metric_key)
            v1_str = f"{v1_val:.4f}" if v1_val is not None else "N/A"
            v2_str = f"{v2_val:.4f}" if v2_val is not None else "N/A"
            if v1_val is not None and v2_val is not None:
                delta_str = _format_delta(v2_val, v1_val)
            else:
                delta_str = "N/A"
            cells.append(f"{v1_str} | {v2_str} | {delta_str}")
        lines.append(f"| {metric_label} | {' | '.join(cells)} |")

    # Row: Odds Coverage
    cells = []
    for target in ["wp", "ats", "ou"]:
        v1_cov = v1[target].get("odds_coverage", {}).get(f"{target}_with_odds", "N/A")
        v2_cov = v2[target].get("odds_coverage", {}).get(f"{target}_with_odds", "N/A")
        cells.append(f"{v1_cov} | {v2_cov} | ---")
    lines.append(f"| Odds Coverage | {' | '.join(cells)} |")

    lines.append("")
    lines.append("*Generated by `scripts/retrain_models.py`*")
    lines.append("")

    return "\n".join(lines)


def gate_targets(
    results: BacktestResults,
    closing_odds_df: pd.DataFrame,
) -> dict[str, dict]:
    """Gate the backtest candidate per target via the SHARED models.deploy_gate.

    Delegates the per-target deploy decision to the ONE shared gate (D24-13, review concern
    #7): for each target it builds the candidate bundle through
    ``deploy_gate.build_candidate_bundle`` (the SAME builder ``promote_models`` uses) on the
    engine's scored candidate frame, reads the FROZEN ``config/gate.toml`` baseline via
    ``deploy_gate.load_gate_config`` (never the gitignored per-version baseline dirs), and
    calls ``deploy_gate.evaluate_target``. The legacy mean-CLV gate path is gone -- the gate
    judges on the per-target significance-tested CLV array (``probability_clv`` for WP,
    ``line_clv`` for ATS/OU). See the module docstring for the enforcing-vs-comparison role.

    Args:
        results: The backtest results carrying ``all_predictions[target]`` -- the scored
            candidate frame (game_id/season/model_prob/actual + merged odds) the shared bundle
            builder consumes; the builder drops the pre-merged CLV/odds columns and recomputes
            CLV, exactly as it does for promote's staged frame.
        closing_odds_df: Normalized closing odds (the engine loader's output).

    Returns:
        Dict mapping target to the ``evaluate_target`` result (keeps the
        ``{passed, reasons, v1_metrics, v2_metrics, ...}`` shape ``print_gating_summary``
        renders).
    """
    cfg = deploy_gate.load_gate_config()
    deploy_gate.validate_gate_config(cfg)

    gating: dict[str, dict] = {}
    for target in ("wp", "ats", "ou"):
        scored_df = results.all_predictions.get(target)
        if scored_df is None or scored_df.empty:
            gating[target] = {
                "passed": False,
                "reasons": [
                    f"No candidate predictions for {target} (empty backtest frame)"
                ],
                "v1_metrics": {},
                "v2_metrics": {},
            }
            continue

        candidate = deploy_gate.build_candidate_bundle(
            target, scored_df, closing_odds_df, cfg
        )
        baseline = _baseline_bundle(target, cfg)
        gating[target] = deploy_gate.evaluate_target(target, candidate, baseline, cfg)

    return gating


def _summarize_metrics(target: str, bundle: dict) -> str:
    """Render the scalar gate metrics for one side (baseline or candidate) of a target.

    Pulls only the human-readable scalars from a bundle -- pooled CLV mean/p plus the
    secondary metric (WP accuracy/ECE/Brier, ATS/OU MAE) -- skipping the raw ``clv_values``
    array and the ``per_season`` sub-bundles so the summary line stays compact.

    Args:
        target: One of "wp", "ats", "ou".
        bundle: A candidate or baseline bundle (evaluate_target's v2_metrics / v1_metrics).

    Returns:
        A compact, single-line metric summary string.
    """

    def _fmt(value: object) -> str:
        if value is None:
            return "N/A"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    parts = [f"clv_mean={_fmt(bundle.get('mean'))}", f"clv_p={_fmt(bundle.get('p'))}"]
    if target == "wp":
        parts.append(f"acc={_fmt(bundle.get('accuracy'))}")
        parts.append(f"ece={_fmt(bundle.get('ece'))}")
        parts.append(f"brier={_fmt(bundle.get('brier_score'))}")
    else:
        parts.append(f"mae={_fmt(bundle.get('mae'))}")
    return ", ".join(parts)


def print_gating_summary(gating_results: dict[str, dict]) -> None:
    """Print a formatted summary of per-target gating decisions.

    Args:
        gating_results: Output from gate_targets().
    """
    print("\n" + "=" * 70)
    print("PER-TARGET GATING RESULTS")
    print("=" * 70)

    for target in ["wp", "ats", "ou"]:
        result = gating_results[target]
        status_icon = "[PASS]" if result["passed"] else "[FAIL]"

        print(f"\n{target.upper()} -- {status_icon}")
        # v1_metrics is the frozen baseline bundle; v2_metrics is the candidate bundle
        # (evaluate_target aliases). Print readable scalars rather than the raw CLV array.
        baseline = result.get("v1_metrics", {})
        candidate = result.get("v2_metrics", {})
        print(f"  v1.0 (frozen): {_summarize_metrics(target, baseline)}")
        print(f"  candidate:     {_summarize_metrics(target, candidate)}")
        for reason in result["reasons"]:
            print(f"  - {reason}")

        if not result["passed"]:
            print(f"  >> WARNING: Keep v1.0 model for {target.upper()}")
        else:
            print(f"  >> Ship candidate model for {target.upper()}")

    print("\n" + "=" * 70)

    # Overall summary
    passing = [t for t, r in gating_results.items() if r["passed"]]
    failing = [t for t, r in gating_results.items() if not r["passed"]]

    if failing:
        print(f"SUMMARY: {len(passing)}/3 targets pass gating")
        print(f"  Ship v2.0: {', '.join(t.upper() for t in passing) or 'none'}")
        print(f"  Keep v1.0: {', '.join(t.upper() for t in failing)}")
    else:
        print("SUMMARY: All 3 targets pass gating -- ship v2.0 for all")

    print("=" * 70)


def main(argv: list[str] | None = None) -> int:
    """Main entry point for the retraining pipeline.

    Args:
        argv: Command-line arguments. None for sys.argv.

    Returns:
        Exit code (0 for success).
    """
    args = parse_args(argv)
    v1_dir = Path("data/baselines/v1.0")
    v2_dir = args.output_dir

    print("=" * 70)
    print("NFL Prediction System -- Model Retraining Pipeline")
    print("=" * 70)
    print(f"  Optuna trials per target: {args.n_trials}")
    print(f"  Output directory: {v2_dir}")
    print(f"  v1.0 baseline: {v1_dir}")
    print("=" * 70)

    # Verify v1.0 baseline exists
    if not v1_dir.exists():
        print(f"ERROR: v1.0 baseline not found at {v1_dir}")
        print("Run: python scripts/capture_baseline.py --version v1.0")
        return 1

    start_time = time.monotonic()

    # Step 1: Run retraining with Optuna tuning
    print("\n[Step 1/4] Running backtest with Optuna-tuned hyperparameters...")
    results = run_retraining(args.n_trials)
    retrain_elapsed = time.monotonic() - start_time
    print(f"  Retraining complete in {retrain_elapsed:.1f}s")
    print(f"  Headline CLV: {results.headline_clv}")

    # Step 2: Capture v2.0 baseline
    print("\n[Step 2/4] Capturing v2.0 baseline...")
    capture_start = time.monotonic()
    capture_v2_baseline(v2_dir)
    capture_elapsed = time.monotonic() - capture_start
    print(f"  v2.0 baseline captured in {capture_elapsed:.1f}s")

    # Step 3: Fill comparison template
    print("\n[Step 3/4] Filling comparison template...")
    comparison = fill_comparison_template(v1_dir, v2_dir)

    # Write filled template to v1.0 directory (overwrites the original)
    template_path = v1_dir / "comparison_template.md"
    template_path.write_text(comparison)
    print(f"  Template written to {template_path}")
    print()
    print(comparison)

    # Step 4: Per-target gating via the SHARED deploy_gate (build_candidate_bundle +
    # evaluate_target against the FROZEN config/gate.toml baseline; no headline_clv,
    # no data/baselines/ read for the decision).
    print("\n[Step 4/4] Applying per-target gating (shared deploy_gate)...")
    closing_odds_df = BacktestEngine()._load_closing_odds()
    gating_results = gate_targets(results, closing_odds_df)
    print_gating_summary(gating_results)

    # Save gating results as JSON for programmatic access (default=str handles numpy/arrays).
    gating_path = v2_dir / "gating_results.json"
    gating_path.write_text(json.dumps(gating_results, indent=2, default=str))
    print(f"\nGating results saved to {gating_path}")

    total_elapsed = time.monotonic() - start_time
    print(f"\nTotal pipeline time: {total_elapsed:.1f}s")

    # Hard block: agree with promote_models.py that a gate FAIL is a non-zero exit (D24-10).
    any_fail = any(not r["passed"] for r in gating_results.values())
    if any_fail:
        failing = [t for t, r in gating_results.items() if not r["passed"]]
        print(
            f"\nGate FAIL for: {', '.join(t.upper() for t in failing)} -- exiting non-zero."
        )
    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
