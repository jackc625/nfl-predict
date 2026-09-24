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

Usage:
    uv run python scripts/regrade_betting_ledger.py
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from backtest.run import export_csv
from backtest.simulation import BettingSimulator

OUTPUT_DIR = Path("outputs/backtest")
FROZEN_LEDGER = OUTPUT_DIR / "betting_simulation.pre_ats_sign_fix.csv"
PREDICTIONS = OUTPUT_DIR / "predictions_all.csv"

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


def main() -> None:
    if not FROZEN_LEDGER.exists():
        raise FileNotFoundError(f"frozen ledger missing: {FROZEN_LEDGER}")
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


if __name__ == "__main__":
    main()
