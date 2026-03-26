"""Unified training entry point for WP, ATS, and O/U models.

Single command to train all models:
    python -m models.train --target all

Individual targets:
    python -m models.train --target wp
    python -m models.train --target ats
    python -m models.train --target ou

This module addresses MODL-08 (baseline comparison against market closing line)
and integrates all prior work (Plans 01-03) into a single runnable pipeline.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, log_loss

from models.temporal import TemporalSplitConfig
from models.trainers.ats_trainer import ATSTrainer
from models.trainers.ou_trainer import OUTrainer
from models.trainers.wp_trainer import WPTrainer
from utils import get_logger
from utils.probability_utils import moneyline_to_probability

logger = get_logger(__name__)

# Valid target choices
_VALID_TARGETS = ("wp", "ats", "ou", "all")


def compute_market_baseline(
    games_df: pd.DataFrame,
    closing_odds_df: pd.DataFrame,
    target: str,
) -> dict[str, float]:
    """Compute market-only baseline metrics for comparison.

    Treats closing line implied probabilities as a "model" and evaluates
    with the same metrics as our trained models. This answers the question:
    "Is our model better than just following the market?"

    For WP: Devig closing moneylines to get fair home-win probability,
    then compute accuracy, Brier score, and log loss.

    For ATS: The market spread implies 50/50 probability by definition.
    Report how often the home team actually covers the closing spread.

    For O/U: The market total implies 50/50 probability by definition.
    Report how often the game actually goes over the closing total.

    Args:
        games_df: Games DataFrame with game_id, home_score, away_score.
        closing_odds_df: Closing odds with game_id, ml_home, ml_away,
            spread, total.
        target: One of "wp", "ats", "ou".

    Returns:
        Dict with keys: market_accuracy, market_brier (WP only),
        market_logloss (WP only), n_games, n_excluded.
    """
    # Merge games with closing odds on game_id
    merged = games_df.merge(
        closing_odds_df, on="game_id", how="inner", suffixes=("", "_odds")
    )

    if target == "wp":
        return _compute_wp_baseline(merged, len(games_df))
    if target == "ats":
        return _compute_ats_baseline(merged, len(games_df))
    if target == "ou":
        return _compute_ou_baseline(merged, len(games_df))
    msg = f"Unknown target: {target}. Must be one of: wp, ats, ou"
    raise ValueError(msg)


def _compute_wp_baseline(
    merged: pd.DataFrame,
    total_games: int,
) -> dict[str, float]:
    """Compute WP market baseline using devigged closing moneylines.

    Devigging removes the bookmaker's margin (vig/juice) so that
    implied probabilities sum to 1.0 instead of > 1.0.

    Args:
        merged: Merged games + odds DataFrame.
        total_games: Total games before merge (for exclusion count).

    Returns:
        Dict with market_accuracy, market_brier, market_logloss, n_games, n_excluded.
    """
    # Exclude rows with missing moneylines
    valid = merged.dropna(subset=["ml_home", "ml_away"])
    n_excluded = total_games - len(valid)

    if len(valid) == 0:
        return {
            "market_accuracy": 0.0,
            "market_brier": 1.0,
            "market_logloss": float("inf"),
            "n_games": 0,
            "n_excluded": n_excluded,
        }

    # Convert moneylines to raw implied probabilities
    home_raw = valid["ml_home"].apply(lambda x: moneyline_to_probability(int(x)))
    away_raw = valid["ml_away"].apply(lambda x: moneyline_to_probability(int(x)))

    # Devig: proportional method (divide by overround)
    total_raw = home_raw + away_raw
    fair_prob_home = home_raw / total_raw

    # Actual outcomes
    actual_home_win = (valid["home_score"] > valid["away_score"]).astype(int)

    # Compute metrics
    market_accuracy = float(
        accuracy_score(actual_home_win, (fair_prob_home > 0.5).astype(int))
    )
    market_brier = float(brier_score_loss(actual_home_win, fair_prob_home))
    market_logloss = float(
        log_loss(actual_home_win, np.clip(fair_prob_home.values, 0.01, 0.99))
    )

    return {
        "market_accuracy": market_accuracy,
        "market_brier": market_brier,
        "market_logloss": market_logloss,
        "n_games": len(valid),
        "n_excluded": n_excluded,
    }


def _compute_ats_baseline(
    merged: pd.DataFrame,
    total_games: int,
) -> dict[str, float]:
    """Compute ATS market baseline.

    The market spread is set to equalize action, meaning the implied
    probability of either side covering is ~50%. The baseline reports
    how often the home team actually covers the closing spread.

    Home covers when: (home_score - away_score) + spread > 0
    (where spread is negative for home favorite).

    Args:
        merged: Merged games + odds DataFrame.
        total_games: Total games before merge (for exclusion count).

    Returns:
        Dict with market_accuracy, n_games, n_excluded.
    """
    valid = merged.dropna(subset=["spread"])
    n_excluded = total_games - len(valid)

    if len(valid) == 0:
        return {
            "market_accuracy": 0.5,
            "n_games": 0,
            "n_excluded": n_excluded,
        }

    actual_margin = valid["home_score"] - valid["away_score"]
    # Home covers if actual margin + spread > 0
    # Spread convention: negative means home is favored
    actual_cover = ((actual_margin + valid["spread"]) > 0).astype(int)

    # Market baseline accuracy: how often does the closing spread
    # correctly predict the cover side? Since spread implies 50/50,
    # the baseline is just the empirical cover rate.
    market_accuracy = float(actual_cover.mean())

    return {
        "market_accuracy": market_accuracy,
        "n_games": len(valid),
        "n_excluded": n_excluded,
    }


def _compute_ou_baseline(
    merged: pd.DataFrame,
    total_games: int,
) -> dict[str, float]:
    """Compute O/U market baseline.

    The market total is set to equalize action, meaning the implied
    probability of over/under is ~50%. The baseline reports how often
    the game actually goes over the closing total.

    Args:
        merged: Merged games + odds DataFrame.
        total_games: Total games before merge (for exclusion count).

    Returns:
        Dict with market_accuracy, n_games, n_excluded.
    """
    valid = merged.dropna(subset=["total"])
    n_excluded = total_games - len(valid)

    if len(valid) == 0:
        return {
            "market_accuracy": 0.5,
            "n_games": 0,
            "n_excluded": n_excluded,
        }

    actual_total = valid["home_score"] + valid["away_score"]
    actual_over = (actual_total > valid["total"]).astype(int)

    # Market baseline accuracy: how often does the game go over?
    # Since the market implies 50/50, this is the empirical over rate.
    market_accuracy = float(actual_over.mean())

    return {
        "market_accuracy": market_accuracy,
        "n_games": len(valid),
        "n_excluded": n_excluded,
    }


def train_target(
    target: str,
    features_df: pd.DataFrame,
    closing_odds_df: pd.DataFrame | None = None,
    config: TemporalSplitConfig | None = None,
    artifacts_dir: Path = Path("artifacts"),
) -> dict[str, Any]:
    """Train a single model target and optionally compute market baseline.

    Instantiates the appropriate trainer, runs training with walk-forward
    evaluation, saves artifacts, and computes market baseline if odds available.

    Args:
        target: One of "wp", "ats", "ou".
        features_df: Feature matrix with ID columns, features, and target column.
        closing_odds_df: Optional closing odds for CLV and market baseline.
        config: Temporal split configuration. Defaults to default split.
        artifacts_dir: Root directory for saving model artifacts.

    Returns:
        Dict with keys: model_metrics, market_baseline, artifact_path.
    """
    # Instantiate the appropriate trainer
    trainers = {
        "wp": WPTrainer,
        "ats": ATSTrainer,
        "ou": OUTrainer,
    }

    if target not in trainers:
        msg = f"Unknown target: {target}. Must be one of: {list(trainers.keys())}"
        raise ValueError(msg)

    trainer_class = trainers[target]
    trainer = trainer_class(config=config)

    logger.info("Starting training", target=target)

    # Train and evaluate
    model_metrics = trainer.train_and_evaluate(features_df, closing_odds_df)

    # Save artifacts
    artifact_path = trainer.save(artifacts_dir)
    logger.info("Artifacts saved", target=target, path=str(artifact_path))

    # Compute market baseline if closing odds available
    market_baseline = None
    if closing_odds_df is not None:
        # Build a games_df from features for market baseline
        games_cols = ["game_id", "home_score", "away_score"]
        if all(col in features_df.columns for col in games_cols):
            games_df = features_df[games_cols].copy()
        elif "game_id" in features_df.columns:
            games_df = features_df[["game_id"]].copy()
            if "home_score" in features_df.columns:
                games_df["home_score"] = features_df["home_score"]
            if "away_score" in features_df.columns:
                games_df["away_score"] = features_df["away_score"]
        else:
            # Try using the index as game_id
            games_df = features_df.reset_index()
            if "game_id" not in games_df.columns:
                games_df = games_df.rename(columns={"index": "game_id"})

        if "home_score" in games_df.columns and "away_score" in games_df.columns:
            market_baseline = compute_market_baseline(
                games_df, closing_odds_df, target=target
            )
            logger.info(
                "Market baseline computed",
                target=target,
                baseline=market_baseline,
            )

    return {
        "model_metrics": model_metrics,
        "market_baseline": market_baseline,
        "artifact_path": str(artifact_path),
    }


def print_summary(results: dict[str, dict]) -> None:
    """Print a clean summary table of training results to stdout.

    Shows per-target metrics including accuracy, Brier score, ECE, CLV,
    and market baseline accuracy. Followed by per-season breakdown.

    Args:
        results: Dict mapping target name to training results dict.
    """
    print()
    print("=" * 70)
    print("  Training Summary")
    print("=" * 70)
    print()

    # Header row
    header = (
        f"{'Target':<8} {'Seasons':<10} {'Accuracy':<10} "
        f"{'Brier':<8} {'ECE':<8} {'CLV':<10} {'Market Acc':<10}"
    )
    print(header)
    print("-" * 70)

    for target, result in results.items():
        model = result.get("model_metrics", {})
        baseline = result.get("market_baseline")
        metadata = model.get("metadata", {})

        # Extract season range from season_results
        season_results = model.get("season_results", [])
        if season_results:
            seasons = [s["season"] for s in season_results]
            season_range = f"{min(seasons)}-{max(seasons) % 100:02d}"
        else:
            season_range = "-"

        # Extract metrics based on target type
        if target == "wp":
            # WP has accuracy from season metrics
            accuracies = [s.get("accuracy", 0) for s in season_results]
            avg_accuracy = np.mean(accuracies) if accuracies else 0
            brier_str = f"{metadata.get('ece', '-'):.3f}" if "ece" in metadata else "-"
            ece_str = f"{metadata.get('ece', '-'):.3f}" if "ece" in metadata else "-"
        else:
            # ATS/OU have MAE from season metrics
            avg_accuracy = 0.0
            brier_str = "-"
            ece_str = "-"

        # CLV
        clv_summary = metadata.get("clv_summary", {})
        mean_clv = clv_summary.get("mean_clv")
        clv_str = f"{mean_clv:+.4f}" if mean_clv is not None else "-"

        # Market baseline
        market_acc = baseline.get("market_accuracy") if baseline else None
        market_str = f"{market_acc:.3f}" if market_acc is not None else "-"

        # Format accuracy
        acc_str = f"{avg_accuracy:.3f}" if avg_accuracy > 0 else "-"

        row = (
            f"{target.upper():<8} {season_range:<10} {acc_str:<10} "
            f"{brier_str:<8} {ece_str:<8} {clv_str:<10} {market_str:<10}"
        )
        print(row)

    # Per-season breakdown
    print()
    print("-" * 70)
    print("  Per-Season Breakdown")
    print("-" * 70)
    print()

    for target, result in results.items():
        model = result.get("model_metrics", {})
        season_results = model.get("season_results", [])

        if not season_results:
            continue

        print(f"  {target.upper()}:")
        for sr in season_results:
            season = sr.get("season", "?")
            n_games = sr.get("n_games", 0)

            if target == "wp":
                acc = sr.get("accuracy", 0)
                print(f"    {season}: {n_games} games, accuracy={acc:.3f}")
            else:
                mae = sr.get("mae", 0)
                rmse = sr.get("rmse", 0)
                print(f"    {season}: {n_games} games, MAE={mae:.2f}, RMSE={rmse:.2f}")

        print()

    print("=" * 70)
    print()


def main() -> None:
    """Main entry point for unified model training.

    Usage:
        python -m models.train --target all
        python -m models.train --target wp
        python -m models.train --target ats --artifacts-dir custom/path
    """
    parser = argparse.ArgumentParser(
        description="Train NFL prediction models with walk-forward temporal validation.",
        prog="python -m models.train",
    )
    parser.add_argument(
        "--target",
        choices=list(_VALID_TARGETS),
        default="all",
        help="Which model(s) to train. Default: all",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Directory for saving model artifacts. Default: artifacts/",
    )
    parser.add_argument(
        "--no-clv",
        action="store_true",
        help="Skip CLV computation (train without closing odds).",
    )
    parser.add_argument(
        "--config-train-seasons",
        type=str,
        default="2018,2019",
        help="Comma-separated training seasons. Default: 2018,2019",
    )
    parser.add_argument(
        "--config-hp-val-seasons",
        type=str,
        default="2020",
        help="Comma-separated HP validation seasons. Default: 2020",
    )
    parser.add_argument(
        "--config-holdout-seasons",
        type=str,
        default="2021,2022,2023,2024",
        help="Comma-separated holdout seasons. Default: 2021,2022,2023,2024",
    )

    args = parser.parse_args()

    # Parse season lists
    train_seasons = [int(s.strip()) for s in args.config_train_seasons.split(",")]
    hp_val_seasons = [int(s.strip()) for s in args.config_hp_val_seasons.split(",")]
    holdout_seasons = [int(s.strip()) for s in args.config_holdout_seasons.split(",")]

    config = TemporalSplitConfig(
        train_seasons=train_seasons,
        hp_val_seasons=hp_val_seasons,
        holdout_seasons=holdout_seasons,
    )

    # Determine targets to train
    if args.target == "all":
        targets = ["wp", "ats", "ou"]
    else:
        targets = [args.target]

    # Load closing odds (unless --no-clv)
    closing_odds_df = None
    if not args.no_clv:
        odds_path = Path("data/silver/odds_snapshot.parquet")
        if odds_path.exists():
            closing_odds_df = pd.read_parquet(odds_path)
            logger.info(
                "Loaded closing odds",
                n_games=len(closing_odds_df),
                path=str(odds_path),
            )
        else:
            logger.warning(
                "Closing odds not found, skipping CLV",
                path=str(odds_path),
            )

    # Train each target
    all_results: dict[str, dict] = {}

    for target in targets:
        # Load feature matrix
        features_path = Path(f"data/gold/features_{target}.parquet")
        if not features_path.exists():
            logger.error(
                "Feature matrix not found",
                target=target,
                path=str(features_path),
            )
            print(
                f"ERROR: Feature matrix not found at {features_path}. "
                f"Run the feature build pipeline first.",
                file=sys.stderr,
            )
            continue

        features_df = pd.read_parquet(features_path)
        logger.info(
            "Loaded features",
            target=target,
            n_rows=len(features_df),
            n_cols=len(features_df.columns),
        )

        result = train_target(
            target=target,
            features_df=features_df,
            closing_odds_df=closing_odds_df,
            config=config,
            artifacts_dir=args.artifacts_dir,
        )
        all_results[target] = result

    # Print summary
    if all_results:
        print_summary(all_results)
    else:
        print("No models were trained. Check feature matrix availability.")


if __name__ == "__main__":
    main()
