"""Capture v1.0 model performance as a regression baseline.

Runs BacktestEngine to collect full backtest results, then serializes:
- Per-target metrics JSON (headline CLV, per-season breakdowns, odds coverage)
- Per-target predictions Parquet (all per-game predictions with outcomes)
- Feature metadata JSON (column names, row counts, numeric column stats)
- Comparison template markdown (v1.0 vs v2.0 side-by-side table)

This script is standalone and does NOT snapshot model artifacts.
Models are versioned via git.

Usage:
    python scripts/capture_baseline.py --version v1.0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from backtest.engine import BacktestConfig, BacktestEngine, BacktestResults


class BaselineCapture:
    """Captures backtest results and feature metadata as a baseline snapshot.

    Produces 8 files in the output directory:
    - metrics_{wp,ats,ou}.json -- per-target headline CLV and breakdowns
    - predictions_{wp,ats,ou}.parquet -- per-game predictions with outcomes
    - feature_metadata.json -- gold feature matrix column stats
    - comparison_template.md -- v1.0 vs v2.0 comparison table

    Args:
        output_dir: Directory to write baseline files. Created if needed.
        config: BacktestConfig to use. Defaults to standard config.
    """

    TARGETS = ["wp", "ats", "ou"]

    def __init__(
        self,
        output_dir: Path | None = None,
        config: BacktestConfig | None = None,
    ) -> None:
        self.output_dir = output_dir or Path("data/baselines/v1.0")
        self.config = config or BacktestConfig()

    def run(self) -> Path:
        """Execute the baseline capture.

        Returns:
            Path to the output directory containing all baseline files.
        """
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Run backtest
        engine = BacktestEngine(self.config)
        results = engine.run()

        # Save per-target metrics
        for target in self.TARGETS:
            self._save_metrics(target, results)

        # Save per-target predictions
        for target in self.TARGETS:
            self._save_predictions(target, results)

        # Save feature metadata
        self._save_feature_metadata()

        # Generate comparison template
        self._save_comparison_template(results)

        # Print summary
        self._print_summary(results)

        return self.output_dir

    def _save_metrics(self, target: str, results: BacktestResults) -> None:
        """Save per-target metrics as JSON.

        Each JSON contains:
        - headline_clv: float
        - per_season_results: list of dicts with season and metrics
        - odds_coverage: dict of coverage counts
        """
        headline_clv = results.headline_clv.get(target, 0.0)

        per_season_results = []
        for sr in results.season_results:
            tr = sr.target_results.get(target)
            if tr is not None:
                per_season_results.append(
                    {
                        "season": tr.season,
                        "metrics": _sanitize_for_json(tr.metrics),
                    }
                )

        # Filter odds coverage keys for this target
        target_odds = {
            k: v for k, v in results.odds_coverage.items() if k.startswith(target)
        }

        metrics_data = {
            "headline_clv": round(float(headline_clv), 4),
            "per_season_results": per_season_results,
            "odds_coverage": target_odds,
        }

        output_path = self.output_dir / f"metrics_{target}.json"
        output_path.write_text(json.dumps(metrics_data, indent=2))

    def _save_predictions(self, target: str, results: BacktestResults) -> None:
        """Save per-target predictions as Parquet."""
        preds_df = results.all_predictions.get(target, pd.DataFrame())

        if preds_df.empty:
            # Create minimal empty DataFrame with required columns
            preds_df = pd.DataFrame(
                columns=[
                    "game_id",
                    "season",
                    "week",
                    "home_team",
                    "away_team",
                    "predicted_prob",
                    "actual_outcome",
                ]
            )

        output_path = self.output_dir / f"predictions_{target}.parquet"
        preds_df.to_parquet(output_path, index=False)

    def _save_feature_metadata(self) -> None:
        """Save gold feature matrix metadata as JSON.

        For each target, records:
        - columns: list of column names
        - row_count: number of rows
        - column_stats: dict of numeric columns with mean, std, nulls
        """
        metadata: dict[str, dict] = {}

        for target in self.TARGETS:
            gold_path = f"data/gold/features_{target}.parquet"
            try:
                df = pd.read_parquet(gold_path)
            except FileNotFoundError:
                metadata[target] = {
                    "columns": [],
                    "row_count": 0,
                    "column_stats": {},
                }
                continue

            # Compute per-numeric-column stats
            column_stats: dict[str, dict] = {}
            numeric_cols = df.select_dtypes(include="number").columns
            for col in numeric_cols:
                column_stats[col] = {
                    "mean": round(float(df[col].mean()), 4),
                    "std": round(float(df[col].std()), 4),
                    "nulls": int(df[col].isna().sum()),
                }

            metadata[target] = {
                "columns": list(df.columns),
                "row_count": len(df),
                "column_stats": column_stats,
            }

        output_path = self.output_dir / "feature_metadata.json"
        output_path.write_text(json.dumps(metadata, indent=2))

    def _save_comparison_template(self, results: BacktestResults) -> None:
        """Generate a markdown comparison template.

        Creates a table with v1.0 values filled in and v2.0 placeholders.
        """
        lines = [
            "# Baseline Comparison: v1.0 vs v2.0",
            "",
            "| Metric | WP v1.0 | WP v2.0 | ATS v1.0 | ATS v2.0 | OU v1.0 | OU v2.0 |",
            "|--------|---------|---------|----------|----------|---------|---------|",
        ]

        # Row: Headline CLV
        wp_clv = round(results.headline_clv.get("wp", 0.0), 4)
        ats_clv = round(results.headline_clv.get("ats", 0.0), 4)
        ou_clv = round(results.headline_clv.get("ou", 0.0), 4)
        lines.append(
            f"| Headline CLV | {wp_clv} | --- | {ats_clv} | --- | {ou_clv} | --- |"
        )

        # Row: Per-season CLV -- extract from season results.
        #
        # DERIVED from the results being reported (review WR-14). This was the literal
        # [2021, 2022, 2023, 2024]. Once the backtest holdout moved with the committed
        # season rule, those four rows would have rendered "N/A" across the board while
        # the seasons actually measured went unreported -- a table that silently
        # describes a window the run did not use.
        measured_seasons = sorted({int(sr.season) for sr in results.season_results})
        for season in measured_seasons:
            season_clvs = {}
            for target in self.TARGETS:
                # Find this season's CLV from season_results
                season_clv = "N/A"
                for sr in results.season_results:
                    if sr.season == season:
                        tr = sr.target_results.get(target)
                        if (
                            tr is not None
                            and tr.clv_df is not None
                            and not tr.clv_df.empty
                        ):
                            if "probability_clv" in tr.clv_df.columns:
                                valid = tr.clv_df[
                                    tr.clv_df.get("has_closing_odds", True) == True  # noqa: E712
                                ]
                                if not valid.empty:
                                    season_clv = str(
                                        round(float(valid["probability_clv"].mean()), 4)
                                    )
                season_clvs[target] = season_clv
            lines.append(
                f"| CLV {season} | {season_clvs['wp']} | --- | {season_clvs['ats']} | --- | {season_clvs['ou']} | --- |"
            )

        # Rows: Brier Score, Hit Rate, Odds Coverage
        # Extract aggregate metrics where available
        for metric_name, metric_key in [
            ("Brier Score", "brier_score"),
            ("Hit Rate", "accuracy"),
        ]:
            values = {}
            for target in self.TARGETS:
                val = "N/A"
                # Average across seasons
                season_vals = []
                for sr in results.season_results:
                    tr = sr.target_results.get(target)
                    if tr is not None and metric_key in tr.metrics:
                        season_vals.append(tr.metrics[metric_key])
                if season_vals:
                    val = str(round(sum(season_vals) / len(season_vals), 4))
                values[target] = val
            lines.append(
                f"| {metric_name} | {values['wp']} | --- | {values['ats']} | --- | {values['ou']} | --- |"
            )

        # Odds Coverage
        wp_cov = results.odds_coverage.get("wp_with_odds", "N/A")
        ats_cov = results.odds_coverage.get("ats_with_odds", "N/A")
        ou_cov = results.odds_coverage.get("ou_with_odds", "N/A")
        lines.append(
            f"| Odds Coverage | {wp_cov} | --- | {ats_cov} | --- | {ou_cov} | --- |"
        )

        lines.append("")
        lines.append("*Generated by `scripts/capture_baseline.py`*")
        lines.append("")

        output_path = self.output_dir / "comparison_template.md"
        output_path.write_text("\n".join(lines))

    def _print_summary(self, results: BacktestResults) -> None:
        """Print a summary of what was saved."""
        print("\n=== Baseline Capture Complete ===")
        print(f"Output directory: {self.output_dir}")
        print(f"Targets: {', '.join(self.TARGETS)}")
        print()
        print("Headline CLV:")
        for target, clv in results.headline_clv.items():
            print(f"  {target}: {clv:.4f}")
        print()
        print("Files saved:")
        for f in sorted(self.output_dir.iterdir()):
            size_kb = f.stat().st_size / 1024
            print(f"  {f.name} ({size_kb:.1f} KB)")


def capture_baseline(version: str = "v1.0") -> Path:
    """Convenience function to capture a baseline snapshot.

    Args:
        version: Version label for the baseline (e.g., "v1.0", "v2.0").

    Returns:
        Path to the output directory containing all baseline files.
    """
    output_dir = Path(f"data/baselines/{version}")
    capture = BaselineCapture(output_dir=output_dir)
    return capture.run()


def _sanitize_for_json(obj: dict) -> dict:
    """Convert numpy/pandas types to JSON-serializable Python types."""
    import numpy as np

    sanitized = {}
    for k, v in obj.items():
        if isinstance(v, (np.integer,)):
            sanitized[k] = int(v)
        elif isinstance(v, (np.floating,)):
            sanitized[k] = round(float(v), 4)
        elif isinstance(v, np.ndarray):
            sanitized[k] = v.tolist()
        elif isinstance(v, dict):
            sanitized[k] = _sanitize_for_json(v)
        elif isinstance(v, list):
            sanitized[k] = [
                _sanitize_for_json(item) if isinstance(item, dict) else item
                for item in v
            ]
        else:
            sanitized[k] = v
    return sanitized


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Capture model performance baseline snapshot",
    )
    parser.add_argument(
        "--version",
        default="v1.0",
        help="Version label for the baseline (default: v1.0)",
    )
    args = parser.parse_args()
    capture_baseline(version=args.version)
