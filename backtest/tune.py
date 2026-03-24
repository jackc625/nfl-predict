"""Blend weight tuning CLI entry point.

Run via: python -m backtest.tune

Orchestrates blend weight tuning on pre-backtest data (2010-2017):
1. Load historical odds from nflverse via load_tuning_period_data
2. Generate synthetic model predictions from tuning odds
3. Run grid-search weight tuning via MarketBlender.tune_weights()
4. Calibrate per-target edge thresholds via calibrate_edge_thresholds()
5. Save blend artifacts to artifacts/ directory

The tuned weights are then available for `python -m backtest.run --blend`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from models.blending import MarketBlender
from models.blending_data import TUNING_SEASONS, load_tuning_period_data
from utils import get_logger
from utils.probability_utils import moneyline_to_probability

logger = get_logger(__name__)


def run_blend_tuning(artifacts_dir: str = "artifacts") -> None:
    """Tune blend weights on pre-backtest data and save artifacts.

    Loads 2010-2017 odds data from nflverse, generates synthetic model
    predictions (since the full feature pipeline only supports 2018+),
    runs grid-search weight tuning, calibrates edge thresholds, and
    saves all artifacts to the specified directory.

    NOTE: This simplified approach generates synthetic model predictions
    from tuning odds data with added noise, producing reasonable initial
    weights. When the full feature pipeline supports pre-2018 data in
    the future, this can be upgraded to use actual model predictions.

    Args:
        artifacts_dir: Directory to save blend artifacts (default: "artifacts").
    """
    print("Tuning blend weights on pre-backtest data (2010-2017)...")
    print()

    # Step 1: Load tuning period odds data
    tuning_odds = load_tuning_period_data()
    print(
        f"Loaded {len(tuning_odds)} games from "
        f"{min(TUNING_SEASONS)}-{max(TUNING_SEASONS)}"
    )
    print()

    # Step 2: Generate synthetic tuning predictions from odds data
    # Using a fixed seed for reproducibility
    rng = np.random.default_rng(42)
    tuning_predictions = _build_synthetic_predictions(tuning_odds, rng)

    # Step 3: Tune blend weights via grid search
    print("Running blend weight grid search...")
    blender = MarketBlender()
    tuning_result = blender.tune_weights(tuning_predictions, tuning_odds)

    print()
    print("Optimal blend weights:")
    for target, clv in tuning_result.per_target_clv.items():
        weight_attr = {
            "wp": "wp_model_weight",
            "ats": "ats_model_weight",
            "ou": "ou_model_weight",
        }[target]
        weight = getattr(tuning_result.weights, weight_attr)
        print(f"  {target.upper()}: weight={weight:.2f}, CLV={clv:.4f}")
    print()

    # Step 4: Calibrate edge thresholds
    print("Calibrating edge thresholds...")
    thresholds = blender.calibrate_edge_thresholds(tuning_predictions, tuning_odds)
    print("Edge thresholds:")
    print(
        f"  WP: {thresholds.wp_threshold:.3f}, "
        f"ATS: {thresholds.ats_threshold:.2f}, "
        f"O/U: {thresholds.ou_threshold:.2f}"
    )
    print()

    # Step 5: Save artifacts
    artifact_dir = blender.save_blend_artifacts(
        tuning_result, Path(artifacts_dir)
    )
    print(f"Blend artifacts saved to {artifact_dir}")
    print()
    print("Run `python -m backtest.run --blend` to use these weights in the backtest.")


def _build_synthetic_predictions(
    tuning_odds: pd.DataFrame,
    rng: np.random.Generator,
) -> dict[str, pd.DataFrame]:
    """Build synthetic model predictions from tuning odds data.

    Since the full feature pipeline only supports 2018+ data, we generate
    synthetic "model predictions" by adding noise to market values. This
    simulates a model that roughly agrees with the market but has some
    independent signal.

    Args:
        tuning_odds: DataFrame with game_id, season, week, spread, total,
            ml_home, ml_away columns.
        rng: Numpy random generator for reproducibility.

    Returns:
        Dict mapping target ("wp", "ats", "ou") to DataFrame with
        game_id, season, week, and the target-specific prediction column.
    """
    base_cols = ["game_id", "season", "week"]

    # WP: Devig moneylines to get fair probability, then add noise
    valid_ml = tuning_odds.dropna(subset=["ml_home", "ml_away"])
    home_raw = valid_ml["ml_home"].apply(lambda ml: moneyline_to_probability(int(ml)))
    away_raw = valid_ml["ml_away"].apply(lambda ml: moneyline_to_probability(int(ml)))
    fair_home_prob = (home_raw / (home_raw + away_raw)).values

    # Add N(0, 0.05) noise and clip to valid probability range
    wp_noise = rng.normal(0, 0.05, size=len(fair_home_prob))
    model_prob = np.clip(fair_home_prob + wp_noise, 0.01, 0.99)

    wp_df = valid_ml[base_cols].copy()
    wp_df["model_prob"] = model_prob

    # ATS: Use spread as base with N(0, 1.5) noise
    valid_spread = tuning_odds.dropna(subset=["spread"])
    spread_noise = rng.normal(0, 1.5, size=len(valid_spread))
    model_spread = valid_spread["spread"].values + spread_noise

    ats_df = valid_spread[base_cols].copy()
    ats_df["model_spread"] = model_spread

    # O/U: Use total as base with N(0, 1.5) noise
    valid_total = tuning_odds.dropna(subset=["total"])
    total_noise = rng.normal(0, 1.5, size=len(valid_total))
    model_total = valid_total["total"].values + total_noise

    ou_df = valid_total[base_cols].copy()
    ou_df["model_total"] = model_total

    return {"wp": wp_df, "ats": ats_df, "ou": ou_df}


def main() -> None:
    """CLI entry point for blend weight tuning."""
    parser = argparse.ArgumentParser(
        prog="python -m backtest.tune",
        description="Tune blend weights on pre-backtest data and save artifacts.",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts",
        help="Directory to save blend artifacts (default: artifacts)",
    )
    args = parser.parse_args()
    run_blend_tuning(artifacts_dir=args.artifacts_dir)


if __name__ == "__main__":
    main()
