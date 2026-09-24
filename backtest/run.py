"""Backtest entry point. Run via: python -m backtest.run

Orchestrates the full backtest pipeline:
1. Run walk-forward backtest engine across holdout seasons
2. Run betting simulation with flat-stake and Kelly strategies
3. Generate interactive HTML report with Plotly charts
4. Export CSV files (predictions, season metrics, betting simulation)
5. Export JSON summary with headline metrics

Usage:
    python -m backtest.run
    python -m backtest.run --seasons 2023,2024 --targets wp,ats
    python -m backtest.run --output-dir custom/output
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from backtest.engine import BacktestConfig, BacktestEngine, BacktestResults
from backtest.report import BacktestReporter
from backtest.simulation import BettingSimulator, SimulationResults
from utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# CSV Export
# ---------------------------------------------------------------------------


def export_csv(
    backtest_results: BacktestResults,
    simulation_results: SimulationResults,
    output_dir: Path,
) -> list[Path]:
    """Export backtest data to CSV files.

    Produces three CSV files:
    - predictions_all.csv: All predictions across targets and seasons.
    - season_metrics.csv: Per-(season, target) metrics.
    - betting_simulation.csv: Individual bet records from simulation.

    Args:
        backtest_results: Complete backtest results from the engine.
        simulation_results: Complete simulation results.
        output_dir: Directory to write CSV files.

    Returns:
        List of paths to the written CSV files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # 1. predictions_all.csv
    all_preds: list[pd.DataFrame] = []
    for target in backtest_results.config.targets:
        for sr in backtest_results.season_results:
            if target not in sr.target_results:
                continue
            tr = sr.target_results[target]
            if tr.predictions_df.empty:
                continue

            df = tr.predictions_df.copy()
            df["target"] = target

            # Normalize column names for export
            rename_map: dict[str, str] = {}
            if "model_prob" in df.columns:
                rename_map["model_prob"] = "model_value"
            elif "model_spread" in df.columns:
                rename_map["model_spread"] = "model_value"
            elif "model_total" in df.columns:
                rename_map["model_total"] = "model_value"

            if "actual" in df.columns:
                rename_map["actual"] = "actual_value"

            df = df.rename(columns=rename_map)

            # Add prediction_correct column if possible
            if "model_value" in df.columns and "actual_value" in df.columns:
                if target == "wp":
                    df["prediction_correct"] = (
                        (df["model_value"] > 0.5).astype(int)
                        == df["actual_value"].astype(int)
                    ).astype(int)
                else:
                    df["prediction_correct"] = pd.NA

            # Include CLV columns if present
            clv_cols = ["probability_clv", "line_clv"]
            for col in clv_cols:
                if col not in df.columns:
                    df[col] = pd.NA

            all_preds.append(df)

    if all_preds:
        predictions_df = pd.concat(all_preds, ignore_index=True)
        predictions_path = output_dir / "predictions_all.csv"
        predictions_df.to_csv(predictions_path, index=False)
        written.append(predictions_path)
        logger.info(
            "Exported predictions CSV",
            path=str(predictions_path),
            rows=len(predictions_df),
        )

    # 2. season_metrics.csv
    metrics_rows: list[dict[str, Any]] = []
    for sr in backtest_results.season_results:
        for target, tr in sr.target_results.items():
            row: dict[str, Any] = {
                "season": sr.season,
                "target": target,
                "n_predictions": len(tr.predictions_df),
                "n_features": len(tr.feature_names),
            }
            row.update(tr.metrics)
            metrics_rows.append(row)

    if metrics_rows:
        metrics_df = pd.DataFrame(metrics_rows)
        metrics_path = output_dir / "season_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        written.append(metrics_path)
        logger.info(
            "Exported season metrics CSV", path=str(metrics_path), rows=len(metrics_df)
        )

    # 3. betting_simulation.csv
    if simulation_results.bet_records:
        bet_rows: list[dict[str, Any]] = []
        for br in simulation_results.bet_records:
            bet_rows.append(
                {
                    "game_id": br.game_id,
                    "season": br.season,
                    "week": br.week,
                    "target": br.target,
                    "bet_side": br.bet_side,
                    "model_value": br.model_value,
                    "market_value": br.market_value,
                    "edge": br.edge,
                    "slipped_line": br.slipped_line,
                    "odds": br.odds,
                    "flat_stake": br.flat_stake,
                    "kelly_stake": br.kelly_stake,
                    "outcome": br.outcome,
                    "payout_flat": br.payout_flat,
                    "payout_kelly": br.payout_kelly,
                }
            )
        bets_df = pd.DataFrame(bet_rows)
        bets_path = output_dir / "betting_simulation.csv"
        bets_df.to_csv(bets_path, index=False)
        written.append(bets_path)
        logger.info(
            "Exported betting simulation CSV", path=str(bets_path), rows=len(bets_df)
        )

    return written


# ---------------------------------------------------------------------------
# JSON Summary Export
# ---------------------------------------------------------------------------


def export_summary_json(
    backtest_results: BacktestResults,
    simulation_results: SimulationResults,
    output_dir: Path,
    baseline_results: BacktestResults | None = None,
) -> Path:
    """Export a machine-readable JSON summary of backtest results.

    Contains headline CLV, per-season summaries, simulation ROI,
    odds coverage stats, config metadata, and blending info when applicable.

    Args:
        backtest_results: Complete backtest results from the engine.
        simulation_results: Complete simulation results.
        output_dir: Directory to write the JSON file.
        baseline_results: Unblended baseline results for delta computation.

    Returns:
        Path to the written JSON file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-season summary
    per_season: list[dict[str, Any]] = []
    for sr in backtest_results.season_results:
        season_data: dict[str, Any] = {"season": sr.season, "targets": {}}
        for target, tr in sr.target_results.items():
            season_data["targets"][target] = {
                "n_predictions": len(tr.predictions_df),
                "metrics": {
                    k: float(v) if isinstance(v, (int, float)) else v
                    for k, v in tr.metrics.items()
                },
            }
        per_season.append(season_data)

    summary: dict[str, Any] = {
        "headline_clv": {k: float(v) for k, v in backtest_results.headline_clv.items()},
        "per_season": per_season,
        "simulation": {
            "flat_stake": {
                "total_bets": simulation_results.flat_stake.total_bets,
                "win_rate": float(simulation_results.flat_stake.win_rate),
                "roi": float(simulation_results.flat_stake.roi),
                "net_profit": float(simulation_results.flat_stake.net_profit),
                "final_bankroll": float(simulation_results.flat_stake.final_bankroll),
                "max_drawdown_pct": float(
                    simulation_results.flat_stake.max_drawdown_pct
                ),
            },
            "kelly": {
                "total_bets": simulation_results.kelly.total_bets,
                "win_rate": float(simulation_results.kelly.win_rate),
                "roi": float(simulation_results.kelly.roi),
                "net_profit": float(simulation_results.kelly.net_profit),
                "final_bankroll": float(simulation_results.kelly.final_bankroll),
                "max_drawdown_pct": float(simulation_results.kelly.max_drawdown_pct),
            },
        },
        "odds_coverage": backtest_results.odds_coverage,
        "config": {
            "holdout_seasons": backtest_results.config.holdout_seasons,
            "targets": backtest_results.config.targets,
            "first_data_season": backtest_results.config.first_data_season,
        },
        "generated_at": datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    # Add blending info if this is a blended backtest
    if backtest_results.is_blended:
        blend_delta_data: dict[str, Any] = {}
        if baseline_results is not None:
            for target in backtest_results.headline_clv:
                blended_clv = backtest_results.headline_clv[target]
                baseline_clv = baseline_results.headline_clv.get(target, 0.0)
                blend_delta_data[target] = {
                    "blended_clv": blended_clv,
                    "baseline_clv": baseline_clv,
                    "delta": blended_clv - baseline_clv,
                    "improved": blended_clv > baseline_clv,
                }
        summary["blending"] = {
            "is_blended": True,
            "blend_delta": blend_delta_data,
        }

    json_path = output_dir / "metrics_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    logger.info("Exported summary JSON", path=str(json_path))
    return json_path


# ---------------------------------------------------------------------------
# Main Orchestration
# ---------------------------------------------------------------------------


def run_backtest(
    seasons: list[int] | None = None,
    targets: list[str] | None = None,
    output_dir: str = "outputs/backtest",
    blend: bool = False,
    blend_artifacts_dir: str = "artifacts",
) -> None:
    """Run the complete backtest pipeline: engine + simulation + report + export.

    When blend=True, runs two backtests: blended (primary) and unblended
    (baseline), then computes and prints improvement delta per target.

    Args:
        seasons: Holdout seasons to evaluate. Defaults to [2021,2022,2023,2024].
        targets: Model targets. Defaults to ["wp","ats","ou"].
        output_dir: Output directory for reports and exports.
        blend: Whether to apply market blending to predictions.
        blend_artifacts_dir: Directory containing blend weight artifacts.
    """
    if seasons is None:
        seasons = [2021, 2022, 2023, 2024]
    if targets is None:
        targets = ["wp", "ats", "ou"]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load blend config if requested
    blend_config = None
    blender = None
    if blend:
        from models.blending import MarketBlender

        try:
            blender = MarketBlender.from_artifacts(Path(blend_artifacts_dir))
        except (KeyError, FileNotFoundError) as exc:
            print("ERROR: Blend artifacts not found.")
            print()
            print("  Blend weight artifacts must be generated before running --blend.")
            print("  Run this command first:")
            print()
            print("    python -m backtest.tune")
            print()
            print(f"  (Details: {exc})")
            raise SystemExit(1) from exc

        blend_config = blender.config
        print(
            f"Applying market blending with weights: "
            f"WP={blend_config.weights.wp_model_weight:.2f}, "
            f"ATS={blend_config.weights.ats_model_weight:.2f}, "
            f"O/U={blend_config.weights.ou_model_weight:.2f}"
        )
        print()

    # Step 1: Run walk-forward backtest
    print(f"Running walk-forward backtest for seasons: {seasons}")
    print(f"Targets: {[t.upper() for t in targets]}")
    if blend:
        print("Mode: BLENDED (market reversion applied)")
    print()

    config = BacktestConfig(
        holdout_seasons=seasons,
        targets=targets,
        blend_config=blend_config,
        # A33.2-review WR-05: the loaded blender travels with its converter binding;
        # passing only its config left WP with no converter and the run raised.
        blender=blender,
        blend_artifacts_dir=Path(blend_artifacts_dir),
    )
    engine = BacktestEngine(config=config)
    results = engine.run()

    print("Backtest engine complete.")
    for target, clv in results.headline_clv.items():
        print(f"  {target.upper()} Headline CLV: {clv:+.4f}")
    print()

    # Step 1b: Run unblended baseline if blending is active
    unblended_results: BacktestResults | None = None
    if blend:
        print("Running unblended baseline for comparison...")
        baseline_config = BacktestConfig(
            holdout_seasons=seasons,
            targets=targets,
            blend_config=None,
        )
        baseline_engine = BacktestEngine(config=baseline_config)
        unblended_results = baseline_engine.run()

        # Print improvement delta
        print()
        print("  Blend Improvement Delta:")
        for target in targets:
            blended_clv = results.headline_clv.get(target, 0.0)
            baseline_clv = unblended_results.headline_clv.get(target, 0.0)
            delta = blended_clv - baseline_clv
            direction = "+" if delta >= 0 else ""
            print(
                f"    {target.upper()}: {blended_clv:+.4f} vs {baseline_clv:+.4f} "
                f"(delta: {direction}{delta:.4f})"
            )
        print()

    # Step 2: Run betting simulation
    print("Running betting simulation...")
    closing_odds_df = engine._load_closing_odds()
    simulator = BettingSimulator()
    sim_results = simulator.simulate(results, closing_odds_df)

    print(f"  Flat-stake ROI: {sim_results.flat_stake.roi:+.2%}")
    print(f"  Kelly ROI:      {sim_results.kelly.roi:+.2%}")
    print(f"  Total bets:     {sim_results.flat_stake.total_bets}")
    print()

    # Step 3: Generate HTML report
    print("Generating HTML report...")
    reporter = BacktestReporter(output_dir=output_path)
    report_path = reporter.generate(
        results,
        sim_results,
        baseline_results=unblended_results if blend else None,
    )
    print(f"  HTML report: {report_path}")

    # Step 4: Export CSVs
    print("Exporting CSV files...")
    csv_paths = export_csv(results, sim_results, output_path)
    for path in csv_paths:
        print(f"  CSV: {path}")

    # Step 5: Export JSON summary
    print("Exporting JSON summary...")
    json_path = export_summary_json(
        results,
        sim_results,
        output_path,
        baseline_results=unblended_results if blend else None,
    )
    print(f"  JSON: {json_path}")

    # Final summary
    print()
    print("=" * 60)
    if blend:
        print("  Blended Backtest Complete")
    else:
        print("  Backtest Complete")
    print("=" * 60)
    print()
    for target, clv in results.headline_clv.items():
        sign = "+" if clv > 0 else ""
        print(f"  {target.upper()} Headline CLV: {sign}{clv:.4f}")
    print(f"  Flat-Stake ROI:  {sim_results.flat_stake.roi:+.2%}")
    print(f"  Kelly ROI:       {sim_results.kelly.roi:+.2%}")

    if blend and unblended_results:
        print()
        print("  Improvement vs Unblended Baseline:")
        for target in targets:
            blended_clv = results.headline_clv.get(target, 0.0)
            baseline_clv = unblended_results.headline_clv.get(target, 0.0)
            delta = blended_clv - baseline_clv
            status = (
                "IMPROVED" if delta > 0 else "DEGRADED" if delta < 0 else "UNCHANGED"
            )
            print(f"    {target.upper()}: delta {delta:+.4f} ({status})")

    print()
    print("  Output files:")
    print(f"    {report_path}")
    for path in csv_paths:
        print(f"    {path}")
    print(f"    {json_path}")
    print()
    print("=" * 60)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry point for running the backtest pipeline.

    Usage:
        python -m backtest.run
        python -m backtest.run --seasons 2023,2024
        python -m backtest.run --targets wp,ats --output-dir custom/path
        python -m backtest.run --blend
        python -m backtest.run --blend --blend-artifacts-dir custom/artifacts
    """
    parser = argparse.ArgumentParser(
        description="Run walk-forward backtest with HTML report and CSV exports.",
        prog="python -m backtest.run",
    )
    parser.add_argument(
        "--seasons",
        type=str,
        default="2021,2022,2023,2024",
        help="Comma-separated holdout seasons (default: 2021,2022,2023,2024)",
    )
    parser.add_argument(
        "--targets",
        type=str,
        default="wp,ats,ou",
        help="Comma-separated targets (default: wp,ats,ou)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/backtest",
        help="Output directory for reports (default: outputs/backtest)",
    )
    parser.add_argument(
        "--blend",
        action="store_true",
        default=False,
        help="Apply market blending to predictions using tuned weights from artifacts.",
    )
    parser.add_argument(
        "--blend-artifacts-dir",
        type=str,
        default="artifacts",
        help="Directory containing blend weight artifacts (default: artifacts/)",
    )

    args = parser.parse_args()

    seasons = [int(s.strip()) for s in args.seasons.split(",")]
    targets = [t.strip() for t in args.targets.split(",")]

    run_backtest(
        seasons=seasons,
        targets=targets,
        output_dir=args.output_dir,
        blend=args.blend,
        blend_artifacts_dir=args.blend_artifacts_dir,
    )


if __name__ == "__main__":
    main()
