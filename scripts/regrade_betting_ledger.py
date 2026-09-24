"""Regrade the published /betting ledger with the current simulator (step 33.2-26d).

Owner ruling 2026-09-23: rebuild the betting page's past results with the corrected ATS cover
rule (commit ee3052f) and keep the old file for the record.

Why this does not re-run ``python -m backtest.run --blend``: that command re-trains every fold on
TODAY's gold (rebuilt 2026-09-22) and blends with TODAY's artifacts, so it would move the WP and
O/U rows too. The inputs the 2026-08-24 run handed the simulator are still on disk, though:

- the blended model value and closing line of every bet it placed (the frozen ledger itself), and
- the actual outcome and the closing prices of every game (``predictions_all.csv``, same run).

Games that run did not bet never reach the ledger (inside the no-bet band, or no price), and the
spread no-bet band is symmetric under the sign fix, so feeding back only the bet games reproduces
the exact input the simulator needs. The WP and O/U rows coming out byte-identical is the check.

Step 33.2-26e also rewrites ``metrics_summary.json`` through ``backtest.run.export_summary_json``.
Its only simulator-derived block is ``simulation`` (the flat-stake and Kelly totals across all
three targets); every other field is read back from the frozen copy of the same run's file. The
pre-fix simulator fed these inputs reproduces that frozen flat-stake block exactly. The Kelly
block moves too: the frozen ledger staked Kelly on 348 spread bets, which the current simulator
stakes at 0 (D31-04), exactly as in the regraded ledger; the 144 WP Kelly bets are unchanged.
``season_metrics.csv`` holds only model-fit metrics (no cover grading) and is not touched.
``backtest_report.html`` needs the full engine results (per-game CLV, calibration, weekly charts)
that only a re-run produces, so it is left as the 2026-08-24 run wrote it.

Usage:
    uv run python scripts/regrade_betting_ledger.py
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from backtest.run import export_csv, export_summary_json
from backtest.simulation import BettingSimulator

OUTPUT_DIR = Path("outputs/backtest")
FROZEN_LEDGER = OUTPUT_DIR / "betting_simulation.pre_ats_sign_fix.csv"
PREDICTIONS = OUTPUT_DIR / "predictions_all.csv"
FROZEN_SUMMARY = OUTPUT_DIR / "metrics_summary.pre_ats_sign_fix.json"

# The column the simulator reads the model's number from, per target.
MODEL_COLUMN = {"wp": "model_prob", "ats": "model_spread", "ou": "model_total"}
ODDS_COLUMNS = ["ml_home", "ml_away", "spread", "total", "has_closing_odds"]


def rebuild_inputs(
    ledger: pd.DataFrame, predictions: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Rebuild the simulator's per-target input frames from the frozen run's own outputs."""
    inputs: dict[str, pd.DataFrame] = {}
    for target, model_column in MODEL_COLUMN.items():
        bets = ledger.loc[ledger["target"] == target]
        model_value = bets["model_value"].to_numpy()
        if target == "wp":
            # The ledger stores P(side bet); the simulator reads P(home win).
            is_home = bets["bet_side"].to_numpy() == "home"
            model_value = np.where(is_home, model_value, 1.0 - model_value)
        frame = pd.DataFrame(
            {
                "game_id": bets["game_id"],
                "season": bets["season"],
                "week": bets["week"],
                model_column: model_value,
            }
        )
        outcomes = predictions.loc[
            predictions["target"] == target, ["game_id", "actual_value", *ODDS_COLUMNS]
        ].rename(columns={"actual_value": "actual"})
        frame = frame.merge(outcomes, on="game_id", how="left", validate="one_to_one")
        if frame["actual"].isna().to_numpy().any():
            raise ValueError(f"{target}: a ledger game has no outcome in {PREDICTIONS}")
        inputs[target] = frame
    return inputs


def summary_inputs(summary: dict) -> tuple[SimpleNamespace, SimpleNamespace | None]:
    """Rebuild what ``export_summary_json`` reads from the engine, from the frozen summary itself.

    Returns the backtest results and the unblended baseline (None when the run was not blended).
    """
    season_results = [
        SimpleNamespace(
            season=season["season"],
            target_results={
                target: SimpleNamespace(
                    # The writer reads only len() of the prediction frame.
                    predictions_df=range(result["n_predictions"]),
                    metrics=result["metrics"],
                )
                for target, result in season["targets"].items()
            },
        )
        for season in summary["per_season"]
    ]
    blending = summary.get("blending", {})
    results = SimpleNamespace(
        headline_clv=summary["headline_clv"],
        season_results=season_results,
        odds_coverage=summary["odds_coverage"],
        config=SimpleNamespace(**summary["config"]),
        is_blended=blending.get("is_blended", False),
    )
    blend_delta = blending.get("blend_delta")
    baseline = None
    if blend_delta:
        baseline = SimpleNamespace(
            headline_clv={
                target: row["baseline_clv"] for target, row in blend_delta.items()
            }
        )
    return results, baseline


def main() -> None:
    for frozen in (FROZEN_LEDGER, FROZEN_SUMMARY):
        if not frozen.exists():
            raise FileNotFoundError(f"frozen copy missing: {frozen}")
    # round_trip: the default CSV float parser can be one ulp off, which would move rows
    # that the fix does not touch. (pandas accepts it; the pandas-stubs Literal omits it.)
    ledger = pd.read_csv(FROZEN_LEDGER, float_precision="round_trip")  # pyright: ignore[reportCallIssue, reportArgumentType]
    predictions = pd.read_csv(PREDICTIONS, float_precision="round_trip")  # pyright: ignore[reportCallIssue, reportArgumentType]

    backtest_results = SimpleNamespace(
        all_predictions=rebuild_inputs(ledger, predictions),
        # Empty on purpose: export_csv then writes ONLY betting_simulation.csv and leaves the
        # frozen run's predictions_all.csv / season_metrics.csv untouched.
        config=SimpleNamespace(targets=[]),
        season_results=[],
    )
    simulation = BettingSimulator().simulate(backtest_results, pd.DataFrame())  # type: ignore[arg-type]
    for path in export_csv(backtest_results, simulation, OUTPUT_DIR):  # type: ignore[arg-type]
        print(f"wrote {path}")

    summary_results, baseline = summary_inputs(
        json.loads(FROZEN_SUMMARY.read_text(encoding="utf-8"))
    )
    summary_path = export_summary_json(
        summary_results,  # type: ignore[arg-type]
        simulation,
        OUTPUT_DIR,
        baseline_results=baseline,  # type: ignore[arg-type]
    )
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
