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
    artifact_dir = blender.save_blend_artifacts(tuning_result, Path(artifacts_dir))
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
    model_spread = np.asarray(valid_spread["spread"].values) + spread_noise

    ats_df = valid_spread[base_cols].copy()
    ats_df["model_spread"] = model_spread

    # O/U: Use total as base with N(0, 1.5) noise
    valid_total = tuning_odds.dropna(subset=["total"])
    total_noise = rng.normal(0, 1.5, size=len(valid_total))
    model_total = np.asarray(valid_total["total"].values) + total_noise

    ou_df = valid_total[base_cols].copy()
    ou_df["model_total"] = model_total

    result: dict[str, pd.DataFrame] = {"wp": wp_df, "ats": ats_df, "ou": ou_df}
    return result


def build_dynamic_synthetic_predictions(
    tuning_odds: pd.DataFrame,
    noise_profile: dict[str, pd.DataFrame],
    rng: np.random.Generator,
) -> dict[str, pd.DataFrame]:
    """Build synthetic predictions using backtest-derived noise profiles.

    Per D-14: Instead of uniform noise, applies per-week noise from the
    noise profile extracted from actual backtest results. This produces
    more realistic synthetic predictions for sigmoid parameter tuning.

    IMPORTANT: The rng parameter MUST be a seeded np.random.Generator
    for reproducibility. Callers should create it as np.random.default_rng(42)
    and pass it in. This function does NOT create its own RNG.

    Args:
        tuning_odds: DataFrame with game_id, season, week, spread, total,
            ml_home, ml_away columns.
        noise_profile: Dict from extract_noise_profile() mapping target
            to DataFrame with week, mean, std, count columns.
        rng: Numpy random generator for reproducibility. Must be externally
            seeded for deterministic results.

    Returns:
        Dict mapping target ("wp", "ats", "ou") to DataFrame with
        game_id, season, week, and the target-specific prediction column.
    """
    base_cols = ["game_id", "season", "week"]

    # -- WP: Apply per-week noise from WP noise profile --
    wp_profile = noise_profile["wp"].set_index("week")
    wp_fallback_mean = float(wp_profile["mean"].mean())
    wp_fallback_std = float(wp_profile["std"].mean())

    valid_ml = tuning_odds.dropna(subset=["ml_home", "ml_away"]).copy()
    home_raw = valid_ml["ml_home"].apply(lambda ml: moneyline_to_probability(int(ml)))
    away_raw = valid_ml["ml_away"].apply(lambda ml: moneyline_to_probability(int(ml)))
    fair_home_prob = (home_raw / (home_raw + away_raw)).values

    # Per-row noise via sorted iteration for deterministic ordering
    wp_noise = np.zeros(len(fair_home_prob))
    for i, (_, row) in enumerate(valid_ml.iterrows()):
        week = int(row["week"])
        if week in wp_profile.index:
            mean = float(wp_profile.at[week, "mean"])
            std = float(wp_profile.at[week, "std"])
        else:
            mean, std = wp_fallback_mean, wp_fallback_std
        wp_noise[i] = rng.normal(mean, max(std, 0.01))

    model_prob = np.clip(fair_home_prob + wp_noise, 0.01, 0.99)
    wp_df = valid_ml[base_cols].copy()
    wp_df["model_prob"] = model_prob

    # -- ATS: Apply per-week noise from ATS noise profile --
    ats_profile = noise_profile["ats"].set_index("week")
    ats_fallback_mean = float(ats_profile["mean"].mean())
    ats_fallback_std = float(ats_profile["std"].mean())

    valid_spread = tuning_odds.dropna(subset=["spread"]).copy()
    ats_noise = np.zeros(len(valid_spread))
    for i, (_, row) in enumerate(valid_spread.iterrows()):
        week = int(row["week"])
        if week in ats_profile.index:
            mean = float(ats_profile.at[week, "mean"])
            std = float(ats_profile.at[week, "std"])
        else:
            mean, std = ats_fallback_mean, ats_fallback_std
        ats_noise[i] = rng.normal(mean, max(std, 0.01))

    model_spread = (
        np.asarray(valid_spread["spread"].values, dtype=np.float64) + ats_noise
    )
    ats_df = valid_spread[base_cols].copy()
    ats_df["model_spread"] = model_spread

    # -- O/U: Apply per-week noise from O/U noise profile --
    ou_profile = noise_profile["ou"].set_index("week")
    ou_fallback_mean = float(ou_profile["mean"].mean())
    ou_fallback_std = float(ou_profile["std"].mean())

    valid_total = tuning_odds.dropna(subset=["total"]).copy()
    ou_noise = np.zeros(len(valid_total))
    for i, (_, row) in enumerate(valid_total.iterrows()):
        week = int(row["week"])
        if week in ou_profile.index:
            mean = float(ou_profile.at[week, "mean"])
            std = float(ou_profile.at[week, "std"])
        else:
            mean, std = ou_fallback_mean, ou_fallback_std
        ou_noise[i] = rng.normal(mean, max(std, 0.01))

    model_total = np.asarray(valid_total["total"].values, dtype=np.float64) + ou_noise
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
