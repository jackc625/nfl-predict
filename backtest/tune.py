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
from collections.abc import Callable
from pathlib import Path

import numpy as np
import optuna
import pandas as pd

from models.blending import (
    BlendConfig,
    BlendWeights,
    DynamicBlendWeights,
    MarketBlender,
    SigmoidParams,
    TuningResult,
)
from models.blending_data import (
    TUNING_SEASONS,
    extract_noise_profile,
    load_tuning_period_data,
)
from models.clv import compute_line_clv, compute_probability_clv
from models.tuning import OptunaTuner
from models.tuning import TuningResult as OptunaTuningResult
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


def create_sigmoid_objective(
    target: str,
    tuning_predictions_df: pd.DataFrame,
    tuning_odds: pd.DataFrame,
    max_week: int = 17,
) -> Callable[[optuna.Trial], float]:
    """Create an Optuna objective function for sigmoid parameter tuning.

    CLV Objective Definition (addresses review concern about underspecification):
    - For each trial, constructs DynamicBlendWeights with trial's midpoint/steepness
    - Iterates over all weeks in the frozen tuning data
    - For each week: blends model predictions with market using the sigmoid weight
    - Computes per-game CLV: probability_clv for WP, line_clv for ATS/O/U
    - Returns MEAN per-game CLV across ALL games with valid odds
    - No edge threshold filtering during tuning (thresholds calibrated AFTER)
    - This means the optimizer maximizes average prediction quality, not betting returns

    Per D-10: Each target gets its own study with independent parameters.
    Per D-11: midpoint in [3/max_week, 14/max_week], steepness in [0.1, 1.5].

    The tuning_predictions_df and tuning_odds are merged ONCE at closure
    creation time. All trials operate on this frozen dataset (addresses
    review concern about per-trial data regeneration).

    Args:
        target: One of "wp", "ats", "ou".
        tuning_predictions_df: DataFrame with game_id, season, week, and model column.
        tuning_odds: DataFrame with game_id, spread, total, ml_home, ml_away.
        max_week: Max week for the tuning era (17 for 2010-2017 per D-05).

    Returns:
        Callable accepting an optuna.Trial, returning mean CLV (to maximize).
    """
    # Merge predictions with odds ONCE upfront (frozen dataset for all trials)
    merged = tuning_predictions_df.merge(
        tuning_odds, on="game_id", how="inner", suffixes=("", "_odds")
    )
    # Resolve potential column conflicts from merge
    if "season_odds" in merged.columns:
        merged = merged.drop(columns=["season_odds"])
    if "week_odds" in merged.columns:
        merged = merged.drop(columns=["week_odds"])

    n_games_in_dataset = len(merged)
    logger.info(
        "Sigmoid objective created",
        target=target,
        n_games=n_games_in_dataset,
        max_week=max_week,
    )

    def objective(trial: optuna.Trial) -> float:
        midpoint = trial.suggest_float("midpoint", 3 / max_week, 14 / max_week)
        steepness = trial.suggest_float("steepness", 0.1, 1.5)

        # Build DynamicBlendWeights with this trial's params for the target
        params = SigmoidParams(midpoint=midpoint, steepness=steepness)
        neutral = SigmoidParams(midpoint=0.5, steepness=1.0)
        kwargs = {"wp": neutral, "ats": neutral, "ou": neutral}
        kwargs[target] = params
        dynamic_weights = DynamicBlendWeights(**kwargs)

        blender = MarketBlender(dynamic_weights=dynamic_weights)

        total_clv = 0.0
        n_games = 0

        for week_num in sorted(merged["week"].unique()):
            week_mask = merged["week"] == week_num
            week_data = merged[week_mask]
            if week_data.empty:
                continue

            season = int(week_data["season"].iloc[0])

            if target == "wp":
                valid = week_data.dropna(subset=["ml_home", "ml_away", "model_prob"])
                if valid.empty:
                    continue
                home_raw = valid["ml_home"].apply(
                    lambda ml: moneyline_to_probability(int(ml))
                )
                away_raw = valid["ml_away"].apply(
                    lambda ml: moneyline_to_probability(int(ml))
                )
                fair_home = (home_raw / (home_raw + away_raw)).values
                model_prob = valid["model_prob"].values

                blended = blender.blend_wp(
                    np.asarray(model_prob, dtype=np.float64),
                    np.asarray(fair_home, dtype=np.float64),
                    week=int(week_num),
                    season=season,
                )

                for i, idx in enumerate(valid.index):
                    result = compute_probability_clv(
                        model_prob=float(blended[i]),
                        closing_ml_home=float(valid.at[idx, "ml_home"]),
                        closing_ml_away=float(valid.at[idx, "ml_away"]),
                        side="home",
                    )
                    total_clv += result["probability_clv"]
                    n_games += 1

            elif target == "ats":
                valid = week_data.dropna(subset=["spread", "model_spread"])
                if valid.empty:
                    continue
                model_spread = valid["model_spread"].values
                market_spread = valid["spread"].values

                blended = blender.blend_ats(
                    np.asarray(model_spread, dtype=np.float64),
                    np.asarray(market_spread, dtype=np.float64),
                    week=int(week_num),
                    season=season,
                )

                for i in range(len(blended)):
                    clv = compute_line_clv(
                        model_value=float(blended[i]),
                        closing_value=float(market_spread[i]),
                        direction="spread",
                    )
                    total_clv += clv
                    n_games += 1

            elif target == "ou":
                valid = week_data.dropna(subset=["total", "model_total"])
                if valid.empty:
                    continue
                model_total = valid["model_total"].values
                market_total = valid["total"].values

                blended = blender.blend_ou(
                    np.asarray(model_total, dtype=np.float64),
                    np.asarray(market_total, dtype=np.float64),
                    week=int(week_num),
                    season=season,
                )

                for i in range(len(blended)):
                    clv = compute_line_clv(
                        model_value=float(blended[i]),
                        closing_value=float(market_total[i]),
                        direction="total",
                    )
                    total_clv += clv
                    n_games += 1

        return total_clv / max(n_games, 1)

    return objective


def run_dynamic_blend_tuning(
    n_trials: int = 100,
    artifacts_dir: str = "artifacts",
    baselines_dir: str | None = None,
    rng_seed: int = 42,
) -> dict:
    """Run Optuna sigmoid parameter tuning for dynamic blend weights.

    Pipeline stages (addresses review concern about bundled responsibilities):
    1. prepare_inputs: Extract noise profile, load odds, generate frozen synthetic data
    2. tune_sigmoid_params: Run 3 independent Optuna studies
    3. calibrate_dynamic_thresholds: Recalibrate edge thresholds with dynamic weights
    4. write_dynamic_artifacts: Save artifacts with dynamic section

    Per D-09, D-10: Creates 3 independent Optuna studies
    (blend_sigmoid_wp, blend_sigmoid_ats, blend_sigmoid_ou),
    each optimizing midpoint and steepness to maximize CLV.

    Per D-14: Uses backtest-derived noise profiles for realistic
    synthetic predictions on 2010-2017 tuning data.

    Args:
        n_trials: Number of Optuna trials per target (default 100 per D-10).
        artifacts_dir: Directory to save blend artifacts.
        baselines_dir: Path to baselines for noise profile extraction.
            Defaults to data/baselines/v2.0.
        rng_seed: Seed for numpy RNG (default 42). Ensures deterministic
            synthetic data generation.

    Returns:
        Dict with keys:
        - "tuning_results": dict mapping target to OptunaTuningResult
        - "dynamic_weights": DynamicBlendWeights with best sigmoid params
        - "artifact_dir": Path to saved artifacts
        - "static_fallback_weights": BlendWeights (sigmoid at week fraction 0.5)
        - "metadata": dict with rng_seed, n_trials, noise_profile_source,
          search_ranges, bet_counts (guardrail metric)
    """
    # ---- Stage 1: Prepare inputs ----
    print("Stage 1: Preparing tuning inputs...")

    baselines_path = Path(baselines_dir) if baselines_dir else None
    noise_profile = extract_noise_profile(baselines_dir=baselines_path)
    noise_source = str(baselines_path or "data/baselines/v2.0")
    print(f"  Extracted noise profile from {noise_source}")

    tuning_odds = load_tuning_period_data()
    print(
        f"  Loaded {len(tuning_odds)} tuning games "
        f"from {min(TUNING_SEASONS)}-{max(TUNING_SEASONS)}"
    )

    # Generate ONE frozen synthetic dataset for all trials (addresses review concern)
    rng = np.random.default_rng(rng_seed)
    tuning_predictions = build_dynamic_synthetic_predictions(
        tuning_odds, noise_profile, rng
    )
    print(f"  Generated frozen synthetic predictions (rng_seed={rng_seed})")
    print()

    # ---- Stage 2: Tune sigmoid parameters ----
    print("Stage 2: Tuning sigmoid parameters...")

    # Per D-05: tuning era 2010-2017 is all <= 2020, so max_week = 17
    max_week = 17
    search_ranges = {
        "midpoint": [3 / max_week, 14 / max_week],
        "steepness": [0.1, 1.5],
    }
    study_names = {
        "wp": "blend_sigmoid_wp",
        "ats": "blend_sigmoid_ats",
        "ou": "blend_sigmoid_ou",
    }

    tuning_results: dict[str, OptunaTuningResult] = {}
    bet_counts: dict[str, int] = {}

    for target in ("wp", "ats", "ou"):
        print(f"  Tuning sigmoid for {target.upper()} ({n_trials} trials)...")

        objective = create_sigmoid_objective(
            target=target,
            tuning_predictions_df=tuning_predictions[target],
            tuning_odds=tuning_odds,
            max_week=max_week,
        )

        tuner = OptunaTuner(
            study_name=study_names[target],
            direction="maximize",  # Maximize mean CLV
            n_trials=n_trials,
        )
        result = tuner.optimize(objective)
        tuning_results[target] = result
        bet_counts[target] = len(tuning_predictions[target])

        print(
            f"    {target.upper()}: midpoint={result.best_params['midpoint']:.4f}, "
            f"steepness={result.best_params['steepness']:.4f}, "
            f"best_CLV={result.best_value:.6f}"
        )
    print()

    # ---- Stage 3: Build DynamicBlendWeights and calibrate thresholds ----
    print("Stage 3: Building weights and calibrating thresholds...")

    dynamic_weights = DynamicBlendWeights(
        wp=SigmoidParams(
            midpoint=tuning_results["wp"].best_params["midpoint"],
            steepness=tuning_results["wp"].best_params["steepness"],
        ),
        ats=SigmoidParams(
            midpoint=tuning_results["ats"].best_params["midpoint"],
            steepness=tuning_results["ats"].best_params["steepness"],
        ),
        ou=SigmoidParams(
            midpoint=tuning_results["ou"].best_params["midpoint"],
            steepness=tuning_results["ou"].best_params["steepness"],
        ),
    )

    # Compute static-equivalent weights (sigmoid at week fraction 0.5) per D-24
    mid_week = max(1, int(max_week * 0.5))
    static_fallback = BlendWeights(
        wp_model_weight=dynamic_weights.get_weight("wp", week=mid_week, season=2015),
        ats_model_weight=dynamic_weights.get_weight("ats", week=mid_week, season=2015),
        ou_model_weight=dynamic_weights.get_weight("ou", week=mid_week, season=2015),
    )

    blender = MarketBlender(
        config=BlendConfig(weights=static_fallback),
        dynamic_weights=dynamic_weights,
    )

    # Calibrate edge thresholds with dynamic weights (D-21)
    print("  Calibrating edge thresholds with dynamic sigmoid schedule...")
    thresholds = blender.calibrate_edge_thresholds(tuning_predictions, tuning_odds)
    print(
        f"    WP: {thresholds.wp_threshold:.3f}, "
        f"ATS: {thresholds.ats_threshold:.2f}, "
        f"O/U: {thresholds.ou_threshold:.2f}"
    )
    print()

    # ---- Stage 4: Write artifacts ----
    print("Stage 4: Writing artifacts...")

    combined_clv = {t: r.best_value for t, r in tuning_results.items()}
    combined_n_games = {t: len(tuning_predictions[t]) for t in tuning_predictions}
    tuning_result_for_save = TuningResult(
        weights=static_fallback,
        per_target_clv=combined_clv,
        per_target_grid={},  # Grid not applicable for Optuna
        tuning_seasons=sorted(TUNING_SEASONS),
        n_games=combined_n_games,
    )

    artifact_dir = blender.save_blend_artifacts(
        tuning_result_for_save, Path(artifacts_dir)
    )
    print(f"  Dynamic blend artifacts saved to {artifact_dir}")
    print()

    metadata = {
        "rng_seed": rng_seed,
        "n_trials": n_trials,
        "noise_profile_source": noise_source,
        "search_ranges": search_ranges,
        "bet_counts": bet_counts,
        "study_names": study_names,
        "max_week": max_week,
    }

    return {
        "tuning_results": tuning_results,
        "dynamic_weights": dynamic_weights,
        "artifact_dir": artifact_dir,
        "static_fallback_weights": static_fallback,
        "metadata": metadata,
    }


def _build_cli_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for blend weight tuning.

    Separated from main() for testability of CLI flags.

    Returns:
        Configured ArgumentParser with --artifacts-dir, --dynamic,
        --n-trials, --baselines-dir, and --rng-seed flags.
    """
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
    parser.add_argument(
        "--dynamic",
        action="store_true",
        default=False,
        help="Use Optuna sigmoid tuning instead of grid search (per D-12)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=100,
        help="Number of Optuna trials per target (default: 100)",
    )
    parser.add_argument(
        "--baselines-dir",
        type=str,
        default=None,
        help="Path to baselines for noise profile extraction",
    )
    parser.add_argument(
        "--rng-seed",
        type=int,
        default=42,
        help="RNG seed for deterministic synthetic data generation (default: 42)",
    )
    return parser


def main() -> None:
    """CLI entry point for blend weight tuning."""
    parser = _build_cli_parser()
    args = parser.parse_args()

    if args.dynamic:
        run_dynamic_blend_tuning(
            n_trials=args.n_trials,
            artifacts_dir=args.artifacts_dir,
            baselines_dir=args.baselines_dir,
            rng_seed=args.rng_seed,
        )
    else:
        run_blend_tuning(artifacts_dir=args.artifacts_dir)


if __name__ == "__main__":
    main()
