"""Retrain all models with Optuna-tuned hyperparameters and compare to v1.0 baseline.

Runs:
1. Optuna hyperparameter tuning for WP, ATS, O/U (100 trials each)
2. Full backtest with retrained models
3. v2.0 baseline capture at data/baselines/v2.0/
4. Comparison template fill with deltas
5. Per-target gating decision

Usage:
    python scripts/retrain_models.py
    python scripts/retrain_models.py --n-trials 50  # fewer trials for quick test
    python scripts/retrain_models.py --output-dir data/baselines/v2.0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from backtest.engine import BacktestConfig, BacktestEngine, BacktestResults
from scripts.capture_baseline import BaselineCapture
from utils import get_logger

logger = get_logger(__name__)

# Gating thresholds (per D-15 research recommendations)
# WP: CLV must not regress > 0.005, accuracy must not drop > 1%
_WP_CLV_REGRESSION_THRESHOLD = 0.005
_WP_ACCURACY_DROP_THRESHOLD = 0.01

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


def _extract_avg_metric(
    metrics: dict, metric_key: str
) -> float | None:
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

    # Rows: Per-season CLV
    for season in [2021, 2022, 2023, 2024]:
        cells = []
        for target in ["wp", "ats", "ou"]:
            v1_season_clv = _get_season_clv(v1[target], season)
            v2_season_clv = _get_season_clv(v2[target], season)
            v1_str = f"{v1_season_clv:.4f}" if v1_season_clv is not None else "N/A"
            v2_str = f"{v2_season_clv:.4f}" if v2_season_clv is not None else "N/A"
            if v1_season_clv is not None and v2_season_clv is not None:
                delta_str = _format_delta(v2_season_clv, v1_season_clv)
            else:
                delta_str = "N/A"
            cells.append(f"{v1_str} | {v2_str} | {delta_str}")
        lines.append(f"| CLV {season} | {' | '.join(cells)} |")

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


def _get_season_clv(
    metrics: dict, season: int
) -> float | None:
    """Extract per-season CLV from baseline metrics.

    The baseline capture stores per-season results with metrics, but CLV
    is computed at the aggregate level. We look for it in the per-season
    results if available.

    Note: v1.0 baseline comparison_template.md has per-season CLV values
    that were computed from the CLV DataFrame during capture. The metrics
    JSON may not have per-season CLV directly. We approximate from the
    structure available.

    Args:
        metrics: Target metrics dict from baseline JSON.
        season: Season to look up.

    Returns:
        Season CLV value, or None if not available.
    """
    # The comparison_template generated during capture has per-season CLV
    # but the metrics JSON has only headline CLV. Return None to indicate
    # per-season CLV should be sourced from the comparison template
    # or recomputed from predictions.
    return None


def gate_targets(
    v1_dir: Path,
    v2_dir: Path,
) -> dict[str, dict]:
    """Apply per-target gating logic to determine which models ship.

    Gating rules (per D-15):
    - WP: Gate on CLV (must not regress > 0.005) AND accuracy (must not drop > 1%)
    - ATS: Gate on MAE (must not increase) AND CLV (must not decrease)
    - O/U: Gate on MAE (must not increase) AND CLV (must not decrease)

    Args:
        v1_dir: Path to v1.0 baseline directory.
        v2_dir: Path to v2.0 baseline directory.

    Returns:
        Dict mapping target to gating result with keys:
        - passed: bool
        - reasons: list of str explaining the decision
        - v1_metrics: dict of v1.0 key metrics
        - v2_metrics: dict of v2.0 key metrics
    """
    v1 = load_baseline_metrics(v1_dir)
    v2 = load_baseline_metrics(v2_dir)

    results = {}

    # WP gating: CLV and accuracy
    wp_result = _gate_wp(v1["wp"], v2["wp"])
    results["wp"] = wp_result

    # ATS gating: MAE and CLV
    ats_result = _gate_regression_target(v1["ats"], v2["ats"], "ats")
    results["ats"] = ats_result

    # O/U gating: MAE and CLV
    ou_result = _gate_regression_target(v1["ou"], v2["ou"], "ou")
    results["ou"] = ou_result

    return results


def _gate_wp(v1_metrics: dict, v2_metrics: dict) -> dict:
    """Apply WP-specific gating logic.

    WP gates on:
    1. CLV must not regress by more than 0.005
    2. Accuracy must not drop by more than 1%

    Args:
        v1_metrics: v1.0 WP metrics.
        v2_metrics: v2.0 WP metrics.

    Returns:
        Gating result dict.
    """
    reasons = []
    passed = True

    v1_clv = v1_metrics["headline_clv"]
    v2_clv = v2_metrics["headline_clv"]
    clv_delta = v2_clv - v1_clv

    # CLV gate: regression must not exceed threshold
    if clv_delta < -_WP_CLV_REGRESSION_THRESHOLD:
        passed = False
        reasons.append(
            f"CLV regressed by {abs(clv_delta):.4f} "
            f"(threshold: {_WP_CLV_REGRESSION_THRESHOLD})"
        )
    else:
        reasons.append(
            f"CLV delta: {clv_delta:+.4f} "
            f"(within threshold of {_WP_CLV_REGRESSION_THRESHOLD})"
        )

    # Accuracy gate
    v1_accuracy = _extract_avg_metric(v1_metrics, "accuracy")
    v2_accuracy = _extract_avg_metric(v2_metrics, "accuracy")

    if v1_accuracy is not None and v2_accuracy is not None:
        accuracy_delta = v2_accuracy - v1_accuracy
        if accuracy_delta < -_WP_ACCURACY_DROP_THRESHOLD:
            passed = False
            reasons.append(
                f"Accuracy dropped by {abs(accuracy_delta):.4f} "
                f"(threshold: {_WP_ACCURACY_DROP_THRESHOLD})"
            )
        else:
            reasons.append(
                f"Accuracy delta: {accuracy_delta:+.4f} "
                f"(within threshold of {_WP_ACCURACY_DROP_THRESHOLD})"
            )
    else:
        reasons.append("Accuracy comparison skipped (metric not available)")

    return {
        "passed": passed,
        "reasons": reasons,
        "v1_metrics": {
            "headline_clv": v1_clv,
            "avg_accuracy": v1_accuracy,
        },
        "v2_metrics": {
            "headline_clv": v2_clv,
            "avg_accuracy": v2_accuracy,
        },
    }


def _gate_regression_target(
    v1_metrics: dict, v2_metrics: dict, target: str
) -> dict:
    """Apply regression target gating logic (ATS, O/U).

    Gates on:
    1. MAE must not increase (lower is better)
    2. CLV must not decrease (higher is better)

    Args:
        v1_metrics: v1.0 target metrics.
        v2_metrics: v2.0 target metrics.
        target: Target name ("ats" or "ou").

    Returns:
        Gating result dict.
    """
    reasons = []
    passed = True

    # CLV gate
    v1_clv = v1_metrics["headline_clv"]
    v2_clv = v2_metrics["headline_clv"]
    clv_delta = v2_clv - v1_clv

    if clv_delta < 0:
        passed = False
        reasons.append(
            f"CLV decreased: {v1_clv:.4f} -> {v2_clv:.4f} "
            f"(delta: {clv_delta:+.4f})"
        )
    else:
        reasons.append(
            f"CLV improved or stable: {v1_clv:.4f} -> {v2_clv:.4f} "
            f"(delta: {clv_delta:+.4f})"
        )

    # MAE gate (average across seasons)
    v1_mae = _extract_avg_metric(v1_metrics, "mae")
    v2_mae = _extract_avg_metric(v2_metrics, "mae")

    if v1_mae is not None and v2_mae is not None:
        mae_delta = v2_mae - v1_mae
        if mae_delta > 0:
            passed = False
            reasons.append(
                f"MAE increased: {v1_mae:.4f} -> {v2_mae:.4f} "
                f"(delta: {mae_delta:+.4f})"
            )
        else:
            reasons.append(
                f"MAE improved or stable: {v1_mae:.4f} -> {v2_mae:.4f} "
                f"(delta: {mae_delta:+.4f})"
            )
    else:
        reasons.append("MAE comparison skipped (metric not available)")

    return {
        "passed": passed,
        "reasons": reasons,
        "v1_metrics": {
            "headline_clv": v1_clv,
            "avg_mae": v1_mae,
        },
        "v2_metrics": {
            "headline_clv": v2_clv,
            "avg_mae": v2_mae,
        },
    }


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
        print(f"  v1.0: {result['v1_metrics']}")
        print(f"  v2.0: {result['v2_metrics']}")
        for reason in result["reasons"]:
            print(f"  - {reason}")

        if not result["passed"]:
            print(f"  >> WARNING: Keep v1.0 model for {target.upper()}")
        else:
            print(f"  >> Ship v2.0 model for {target.upper()}")

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

    # Step 4: Per-target gating
    print("\n[Step 4/4] Applying per-target gating...")
    gating_results = gate_targets(v1_dir, v2_dir)
    print_gating_summary(gating_results)

    # Save gating results as JSON for programmatic access
    gating_path = v2_dir / "gating_results.json"
    gating_path.write_text(json.dumps(gating_results, indent=2, default=str))
    print(f"\nGating results saved to {gating_path}")

    total_elapsed = time.monotonic() - start_time
    print(f"\nTotal pipeline time: {total_elapsed:.1f}s")

    return 0


if __name__ == "__main__":
    sys.exit(main())
