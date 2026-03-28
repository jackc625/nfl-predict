#!/usr/bin/env python3
"""Current week prediction generation script.

Loads trained model artifacts, runs predictions against gold feature matrices,
optionally applies market blending, and writes output CSVs.

Usage:
    uv run python scripts/generate_current_week_predictions.py --season 2024 --week 1
    make predict SEASON=2024 WEEK=1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.artifacts import load_model_artifact
from models.blending import MarketBlender
from utils import get_logger
from utils.probability_utils import moneyline_to_probability

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Gold feature loading
# ---------------------------------------------------------------------------


def load_gold_features(
    target: str,
    season: int,
    week: int,
) -> pd.DataFrame:
    """Load and filter gold feature matrix for a target/season/week.

    Args:
        target: One of "wp", "ats", "ou".
        season: NFL season year.
        week: NFL week number.

    Returns:
        Filtered DataFrame for the requested season/week.

    Raises:
        SystemExit: If no rows match the requested season/week.
    """
    path = Path(f"data/gold/features_{target}.parquet")
    if not path.exists():
        logger.error("Gold features file not found", path=str(path))
        sys.exit(1)

    df = pd.read_parquet(path)
    filtered = df[(df["season"] == season) & (df["week"] == week)].copy()

    if filtered.empty:
        logger.error(
            f"No games found for {season} Week {week} in gold features",
            target=target,
            path=str(path),
        )
        sys.exit(1)

    logger.info(
        "Loaded gold features",
        target=target,
        season=season,
        week=week,
        n_games=len(filtered),
    )
    return filtered


# ---------------------------------------------------------------------------
# Market data loading
# ---------------------------------------------------------------------------


def load_market_data(game_ids: list[str]) -> pd.DataFrame:
    """Load market data from silver odds snapshot, filtered to given game_ids.

    Returns DataFrame with columns: game_id, spread, total, ml_home, ml_away.
    Returns empty DataFrame if odds file not found.
    """
    odds_path = Path("data/silver/odds_snapshot.parquet")
    if not odds_path.exists():
        logger.warning(
            "Odds snapshot not found, skipping market data", path=str(odds_path)
        )
        return pd.DataFrame(
            columns=["game_id", "spread", "total", "ml_home", "ml_away"]
        )

    odds_df = pd.read_parquet(odds_path)

    # Normalize team abbreviations in odds game_ids (e.g. LAR -> LA)
    # to match canonical game_ids from gold features
    from utils.team_data import normalize_team_abbreviation

    def _normalize_game_id(gid: str) -> str:
        parts = gid.split("_")
        if len(parts) >= 3:
            matchup = parts[2]  # e.g. "LAR@DET"
            sep = "@" if "@" in matchup else "_"
            teams = matchup.split(sep)
            if len(teams) == 2:
                try:
                    normalized = sep.join(normalize_team_abbreviation(t) for t in teams)
                    return "_".join([*parts[:2], normalized, *parts[3:]])
                except (ValueError, KeyError):
                    pass
        return gid

    odds_df["game_id"] = odds_df["game_id"].apply(_normalize_game_id)
    filtered = odds_df[odds_df["game_id"].isin(game_ids)].copy()

    # Keep only the columns we need, deduplicate by game_id (take first row per game)
    cols_needed = ["game_id", "spread", "total", "ml_home", "ml_away"]
    available_cols = [c for c in cols_needed if c in filtered.columns]
    filtered = filtered[available_cols].drop_duplicates(
        subset=["game_id"], keep="first"
    )

    logger.info(
        "Loaded market data",
        n_matched=len(filtered),
        n_requested=len(game_ids),
    )
    return filtered


# ---------------------------------------------------------------------------
# Prediction logic
# ---------------------------------------------------------------------------


def run_predictions(
    artifacts_dir: Path,
    season: int,
    week: int,
) -> dict[str, pd.DataFrame]:
    """Run model predictions for all three targets.

    Returns:
        Dict mapping target -> DataFrame with game_id and prediction columns.
    """
    results: dict[str, pd.DataFrame] = {}

    for target in ("wp", "ats", "ou"):
        # Load artifact
        artifact = load_model_artifact(target, artifacts_dir=artifacts_dir)
        model = artifact["model"]
        feature_list = artifact["feature_list"]
        calibrator = artifact["calibrator"]

        # Load gold features
        gold_df = load_gold_features(target, season, week)

        # Ensure all required features exist in the gold DataFrame
        missing_features = [f for f in feature_list if f not in gold_df.columns]
        if missing_features:
            logger.error(
                "Missing features in gold matrix",
                target=target,
                missing=missing_features[:10],
            )
            sys.exit(1)

        X = gold_df[feature_list]

        if target == "wp":
            raw_probs = model.predict_proba(X)[:, 1]
            if calibrator is not None:
                wp_prob = calibrator.predict(raw_probs)
                logger.info(
                    "Applied isotonic calibration to WP predictions",
                    n_games=len(wp_prob),
                )
            else:
                wp_prob = raw_probs
            result_df = gold_df[["game_id"]].copy()
            result_df["wp_prob"] = wp_prob
        elif target == "ats":
            ats_prediction = model.predict(X)
            result_df = gold_df[["game_id"]].copy()
            result_df["ats_prediction"] = ats_prediction
        else:  # ou
            ou_prediction = model.predict(X)
            result_df = gold_df[["game_id"]].copy()
            result_df["ou_prediction"] = ou_prediction

        results[target] = result_df
        logger.info(
            "Generated predictions",
            target=target,
            n_games=len(result_df),
        )

    return results


# ---------------------------------------------------------------------------
# Edge and confidence computation
# ---------------------------------------------------------------------------


def compute_confidence(edge: float) -> str:
    """Compute confidence level from edge magnitude."""
    abs_edge = abs(edge)
    if abs_edge > 0.05:
        return "high"
    if abs_edge > 0.02:
        return "medium"
    return "low"


def compute_edges(
    predictions: pd.DataFrame,
    market: pd.DataFrame,
) -> pd.DataFrame:
    """Compute edges between model predictions and market lines.

    Mutates and returns the predictions DataFrame with edge columns added.
    """
    merged = predictions.merge(market, on="game_id", how="left")

    # WP edge: model prob - implied market prob
    if "ml_home" in merged.columns and "ml_away" in merged.columns:
        valid_ml = merged["ml_home"].notna() & merged["ml_away"].notna()
        merged.loc[valid_ml, "wp_edge"] = merged.loc[valid_ml].apply(
            lambda row: (
                row["wp_prob"]
                - moneyline_to_probability(int(row["ml_home"]))
                / (
                    moneyline_to_probability(int(row["ml_home"]))
                    + moneyline_to_probability(int(row["ml_away"]))
                )
            ),
            axis=1,
        )
    if "wp_edge" not in merged.columns:
        merged["wp_edge"] = np.nan

    # ATS edge: (model_spread - (-market_spread)) / abs(market_spread)
    if "spread" in merged.columns:
        valid_spread = merged["spread"].notna() & (merged["spread"] != 0)
        merged.loc[valid_spread, "ats_edge"] = merged.loc[valid_spread].apply(
            lambda row: (row["ats_prediction"] - (-row["spread"])) / abs(row["spread"]),
            axis=1,
        )
        # Where spread == 0, edge is 0
        zero_spread = merged["spread"].notna() & (merged["spread"] == 0)
        merged.loc[zero_spread, "ats_edge"] = 0.0
    if "ats_edge" not in merged.columns:
        merged["ats_edge"] = np.nan

    # O/U edge: (model_total - market_total) / market_total
    if "total" in merged.columns:
        valid_total = merged["total"].notna() & (merged["total"] != 0)
        merged.loc[valid_total, "ou_edge"] = merged.loc[valid_total].apply(
            lambda row: (row["ou_prediction"] - row["total"]) / row["total"],
            axis=1,
        )
        zero_total = merged["total"].notna() & (merged["total"] == 0)
        merged.loc[zero_total, "ou_edge"] = 0.0
    if "ou_edge" not in merged.columns:
        merged["ou_edge"] = np.nan

    # Confidence levels
    merged["wp_confidence"] = merged["wp_edge"].apply(
        lambda e: compute_confidence(e) if pd.notna(e) else "low"
    )
    merged["ats_confidence"] = merged["ats_edge"].apply(
        lambda e: compute_confidence(e) if pd.notna(e) else "low"
    )
    merged["ou_confidence"] = merged["ou_edge"].apply(
        lambda e: compute_confidence(e) if pd.notna(e) else "low"
    )

    return merged


# ---------------------------------------------------------------------------
# Blending
# ---------------------------------------------------------------------------


def apply_blending(
    predictions: pd.DataFrame,
    market: pd.DataFrame,
    artifacts_dir: Path,
    no_blend: bool,
) -> pd.DataFrame:
    """Apply market blending if blend artifacts exist.

    Adds blended_wp, blended_ats, blended_ou columns. Sets them to NaN
    when blending is not applied.
    """
    predictions["blended_wp"] = np.nan
    predictions["blended_ats"] = np.nan
    predictions["blended_ou"] = np.nan

    if no_blend:
        logger.info("Blending skipped (--no-blend flag)")
        return predictions

    try:
        blender = MarketBlender.from_artifacts(artifacts_dir)
        has_blend = True
        logger.info(
            "Blend artifacts found, applying market blending",
            wp_weight=blender.config.weights.wp_model_weight,
            ats_weight=blender.config.weights.ats_model_weight,
            ou_weight=blender.config.weights.ou_model_weight,
        )
    except (KeyError, FileNotFoundError) as exc:
        has_blend = False
        logger.info(
            "No blend artifacts found, using raw model predictions", reason=str(exc)
        )

    if not has_blend:
        return predictions

    # Merge with market data for blending
    merged = predictions.merge(market, on="game_id", how="left", suffixes=("", "_mkt"))

    # WP blending: need fair market probability from moneylines
    ml_cols_present = "ml_home" in merged.columns and "ml_away" in merged.columns
    if ml_cols_present:
        valid_ml = merged["ml_home"].notna() & merged["ml_away"].notna()
        if valid_ml.any():
            home_raw = merged.loc[valid_ml, "ml_home"].apply(
                lambda ml: moneyline_to_probability(int(ml))
            )
            away_raw = merged.loc[valid_ml, "ml_away"].apply(
                lambda ml: moneyline_to_probability(int(ml))
            )
            market_probs = (home_raw / (home_raw + away_raw)).values
            model_probs = merged.loc[valid_ml, "wp_prob"].values

            blended_wp = blender.blend_wp(
                np.asarray(model_probs, dtype=np.float64),
                np.asarray(market_probs, dtype=np.float64),
            )
            predictions.loc[valid_ml, "blended_wp"] = blended_wp

    # ATS blending
    if "spread" in merged.columns:
        valid_spread = merged["spread"].notna()
        if valid_spread.any():
            model_spreads = merged.loc[valid_spread, "ats_prediction"].values
            market_spreads = merged.loc[valid_spread, "spread"].values
            blended_ats = blender.blend_ats(
                np.asarray(model_spreads, dtype=np.float64),
                np.asarray(market_spreads, dtype=np.float64),
            )
            predictions.loc[valid_spread, "blended_ats"] = blended_ats

    # O/U blending
    if "total" in merged.columns:
        valid_total = merged["total"].notna()
        if valid_total.any():
            model_totals = merged.loc[valid_total, "ou_prediction"].values
            market_totals = merged.loc[valid_total, "total"].values
            blended_ou = blender.blend_ou(
                np.asarray(model_totals, dtype=np.float64),
                np.asarray(market_totals, dtype=np.float64),
            )
            predictions.loc[valid_total, "blended_ou"] = blended_ou

    n_blended = predictions["blended_wp"].notna().sum()
    logger.info("Applied market blending", n_blended=int(n_blended))

    return predictions


# ---------------------------------------------------------------------------
# Game context
# ---------------------------------------------------------------------------


def build_game_context(
    game_ids: list[str],
    season: int,
    week: int,
) -> pd.DataFrame:
    """Build game context DataFrame from silver games and gold features.

    Includes team info, venue info, and selected feature values.
    """
    # Load silver games for team/venue info
    games_path = Path("data/silver/games.parquet")
    if not games_path.exists():
        logger.warning("Silver games not found, game context will be sparse")
        return pd.DataFrame({"game_id": game_ids})

    games_df = pd.read_parquet(games_path)
    games_filtered = games_df[games_df["game_id"].isin(game_ids)].copy()

    context_cols = ["game_id", "home_team", "away_team"]
    optional_cols = ["kickoff_et", "venue", "venue_roof"]
    for col in optional_cols:
        if col in games_filtered.columns:
            context_cols.append(col)

    context = games_filtered[context_cols].copy()

    # Extract feature values from gold WP features
    wp_path = Path("data/gold/features_wp.parquet")
    if wp_path.exists():
        wp_df = pd.read_parquet(wp_path)
        wp_filtered = wp_df[
            (wp_df["season"] == season) & (wp_df["week"] == week)
        ].copy()

        feature_cols_to_extract = [
            "home_elo",
            "away_elo",
            "is_divisional",
            "weather_severity_score",
            "wind_mph",
            "venue_outdoor",
        ]
        available_feature_cols = [
            c for c in feature_cols_to_extract if c in wp_filtered.columns
        ]

        if available_feature_cols:
            feature_subset = wp_filtered[["game_id", *available_feature_cols]]
            context = context.merge(feature_subset, on="game_id", how="left")

    return context


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------


def write_predictions(
    predictions: pd.DataFrame,
    season: int,
    week: int,
    output_dir: Path,
) -> Path:
    """Write prediction CSV to disk.

    Returns the path to the written file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    output_cols = [
        "game_id",
        "season",
        "week",
        "home_team",
        "away_team",
        "wp_prob",
        "ats_prediction",
        "ou_prediction",
        "wp_confidence",
        "ats_confidence",
        "ou_confidence",
        "market_spread",
        "market_total",
        "market_ml_home",
        "market_ml_away",
        "wp_edge",
        "ats_edge",
        "ou_edge",
        "blended_wp",
        "blended_ats",
        "blended_ou",
    ]

    # Only include columns that exist
    available_cols = [c for c in output_cols if c in predictions.columns]
    out_df = predictions[available_cols]

    out_path = output_dir / f"predictions_{season}_week{week}.csv"
    out_df.to_csv(out_path, index=False)
    logger.info("Wrote predictions CSV", path=str(out_path), n_rows=len(out_df))
    return out_path


def write_game_context(
    context: pd.DataFrame,
    season: int,
    week: int,
    output_dir: Path,
) -> Path:
    """Write game context CSV to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"game_context_{season}_week{week}.csv"
    context.to_csv(out_path, index=False)
    logger.info("Wrote game context CSV", path=str(out_path), n_rows=len(context))
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for prediction generation."""
    parser = argparse.ArgumentParser(
        description="Generate NFL predictions for a specific season/week using trained model artifacts."
    )
    parser.add_argument("--season", type=int, required=True, help="NFL season year")
    parser.add_argument("--week", type=int, required=True, help="NFL week number")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/predictions",
        help="Output directory for prediction files (default: outputs/predictions)",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts",
        help="Root directory for model artifacts (default: artifacts)",
    )
    parser.add_argument(
        "--no-blend",
        action="store_true",
        help="Skip market blending even if blend artifacts exist",
    )

    args = parser.parse_args()
    artifacts_dir = Path(args.artifacts_dir)
    output_dir = Path(args.output_dir)
    season = args.season
    week = args.week

    logger.info(
        "Starting prediction generation",
        season=season,
        week=week,
        artifacts_dir=str(artifacts_dir),
        output_dir=str(output_dir),
    )

    # -----------------------------------------------------------------------
    # 1. Run model predictions for all three targets
    # -----------------------------------------------------------------------
    prediction_results = run_predictions(artifacts_dir, season, week)

    # -----------------------------------------------------------------------
    # 2. Merge predictions into a single DataFrame
    # -----------------------------------------------------------------------
    wp_df = prediction_results["wp"]  # game_id, wp_prob
    ats_df = prediction_results["ats"]  # game_id, ats_prediction
    ou_df = prediction_results["ou"]  # game_id, ou_prediction

    combined = wp_df.merge(ats_df, on="game_id", how="outer")
    combined = combined.merge(ou_df, on="game_id", how="outer")

    # Add season and week
    combined["season"] = season
    combined["week"] = week

    # -----------------------------------------------------------------------
    # 3. Load market data
    # -----------------------------------------------------------------------
    game_ids = combined["game_id"].tolist()
    market_df = load_market_data(game_ids)

    # -----------------------------------------------------------------------
    # 4. Join game info from silver games (home_team, away_team)
    # -----------------------------------------------------------------------
    games_path = Path("data/silver/games.parquet")
    if games_path.exists():
        games = pd.read_parquet(games_path)
        games_filtered = games[games["game_id"].isin(game_ids)][
            ["game_id", "home_team", "away_team"]
        ].drop_duplicates(subset=["game_id"])
        combined = combined.merge(games_filtered, on="game_id", how="left")

    # -----------------------------------------------------------------------
    # 5. Compute edges
    # -----------------------------------------------------------------------
    combined = compute_edges(combined, market_df)

    # Rename market columns for output clarity
    if "spread" in combined.columns:
        combined.rename(columns={"spread": "market_spread"}, inplace=True)
    if "total" in combined.columns:
        combined.rename(columns={"total": "market_total"}, inplace=True)
    if "ml_home" in combined.columns:
        combined.rename(columns={"ml_home": "market_ml_home"}, inplace=True)
    if "ml_away" in combined.columns:
        combined.rename(columns={"ml_away": "market_ml_away"}, inplace=True)

    # -----------------------------------------------------------------------
    # 6. Apply market blending
    # -----------------------------------------------------------------------
    combined = apply_blending(combined, market_df, artifacts_dir, args.no_blend)

    # -----------------------------------------------------------------------
    # 7. Write outputs
    # -----------------------------------------------------------------------
    pred_path = write_predictions(combined, season, week, output_dir)
    context_df = build_game_context(game_ids, season, week)
    context_path = write_game_context(context_df, season, week, output_dir)

    # -----------------------------------------------------------------------
    # 8. Summary
    # -----------------------------------------------------------------------
    n_blended = int(combined["blended_wp"].notna().sum())
    logger.info(
        "Prediction generation complete",
        n_games=len(combined),
        blending_applied=n_blended > 0,
        n_blended=n_blended,
        predictions_file=str(pred_path),
        context_file=str(context_path),
    )


if __name__ == "__main__":
    main()
