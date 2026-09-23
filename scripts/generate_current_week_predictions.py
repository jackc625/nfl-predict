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
from typing import Any

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backtest.ou_divergence import dedupe_odds_by_book_preference
from models.artifacts import load_model_artifact
from models.blending import (
    MarketBlender,
    MarketProbabilityUnavailable,
    home_fav_margin_from_prelock_spread,
)
from models.market_probability import market_home_win_probability
from utils import get_logger
from utils.edge_tier import edge_tier
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
        raise FileNotFoundError(f"Gold features file not found: {path}")

    df = pd.read_parquet(path)
    filtered = df[(df["season"] == season) & (df["week"] == week)].copy()

    if filtered.empty:
        raise ValueError(
            f"No games found for {season} Week {week} in gold features "
            f"(target={target}, path={path})"
        )

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
    from utils.exceptions import DataValidationError
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
                except (ValueError, KeyError, DataValidationError):
                    # Best-effort: an unmappable odds row (e.g. sample/test data)
                    # is left as-is so it simply fails to match a real game_id.
                    pass
        return gid

    odds_df["game_id"] = odds_df["game_id"].apply(_normalize_game_id)
    filtered = odds_df[odds_df["game_id"].isin(game_ids)].copy()

    # One row per game, choosing the BOOK by name rather than by parquet row order (WR-08).
    # ``keep="first"`` picked whichever row appeared first in the file, so appending a second
    # book's row for a game silently changed which book's price the published prediction was
    # struck at. The dedupe runs BEFORE the column projection because it reads ``sportsbook``.
    filtered = dedupe_odds_by_book_preference(filtered)
    cols_needed = ["game_id", "spread", "total", "ml_home", "ml_away"]
    available_cols = [c for c in cols_needed if c in filtered.columns]
    filtered = filtered[available_cols]

    logger.info(
        "Loaded market data",
        n_matched=len(filtered),
        n_requested=len(game_ids),
    )
    return filtered


# ---------------------------------------------------------------------------
# Prediction logic
# ---------------------------------------------------------------------------

# The prediction column each target's model writes, in ``run_predictions`` order.
_PREDICTION_COLUMN: dict[str, str] = {
    "wp": "wp_prob",
    "ats": "ats_prediction",
    "ou": "ou_prediction",
}

# The published prediction CSV's columns, in order. An all-skipped day writes exactly this
# header with no rows, so downstream readers see a well-formed empty week, not a malformed file.
PREDICTION_OUTPUT_COLUMNS: list[str] = [
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


def _without_excluded(
    gold_df: pd.DataFrame, excluded_game_ids: frozenset[str]
) -> pd.DataFrame:
    """*gold_df* without the games a live run dropped (D33.2-05). Same object when none."""
    if not excluded_game_ids:
        return gold_df
    keep = ~gold_df["game_id"].astype(str).isin(sorted(excluded_game_ids))
    return gold_df.loc[keep].copy()


def run_predictions(
    artifacts_dir: Path,
    season: int,
    week: int,
    *,
    excluded_game_ids: frozenset[str] = frozenset(),
) -> dict[str, pd.DataFrame]:
    """Run model predictions for all three targets.

    Args:
        artifacts_dir: Root directory for model artifacts.
        season: NFL season year.
        week: NFL week number.
        excluded_game_ids: Games a live run dropped under the live-skip rule (D33.2-05). Their
            gold rows are removed BEFORE any model scores anything, so no prediction is ever
            COMPUTED for them -- not computed and then filtered. The week-emptiness refusal in
            ``load_gold_features`` still runs first, so "no game scheduled" keeps refusing while
            "every game excluded" returns empty frames.

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

        # Load gold features, then drop the excluded games before anything is scored.
        gold_df = _without_excluded(
            load_gold_features(target, season, week), excluded_game_ids
        )
        if gold_df.empty:
            # Every game of the week was excluded: an honestly all-skipped day, not an
            # unscheduled one (that refused above). Nothing is handed to the model.
            results[target] = pd.DataFrame(
                columns=pd.Index(["game_id", _PREDICTION_COLUMN[target]])
            )
            logger.info("Every game excluded; nothing scored", target=target)
            continue

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

    # ATS edge: model home margin MINUS market home margin, in POINTS.
    #
    # THE UNIT IS POINTS (R13 / D33-05, Plan 33-10). There is NO DENOMINATOR. This used to be
    # ``(ats_prediction - spread) / abs(spread)`` -- an unbounded ratio whose divisor is a point
    # count that approaches zero -- so a three-point disagreement on a half-point line scored 6.0,
    # and ``utils.edge_tier`` bands it against the same 0.05 / 0.02 pair it applies to a WP
    # probability. The pre-repair band distribution is recorded in
    # ``tests/phase33_state.ATS_BAND_SHARES_BEFORE``.
    #
    # THE ZERO BRANCH IS GONE (D33-31, owner ruling). A pick-em used to be forced to an edge of
    # 0.0, which was right while the edge was a ratio -- dividing by a zero line is undefined --
    # and wrong the moment it became a point margin. A pick'em is a REAL line, not a missing one,
    # so a model with the home team by three against it disagrees with the market by exactly
    # three points. ONE branch survives: no stored line means NO edge (NaN), because there is no
    # disagreement to measure.
    #
    # THE SIGN CONVENTION, unchanged and still the reason the arithmetic is this simple.
    # DEF-31-01 measured it on 2026-09-04, the owner ruled on it, and it was reproduced here:
    # corr(spread, ml_home) = -0.9506 over all 2140 stored rows, mean spread +9.19 when the home
    # side is a big favourite versus -8.49 when the away side is, and corr(spread, realized home
    # margin) = +0.4517. The stored ``spread`` is the nflverse ``spread_line``, POSITIVE when the
    # home team is favored, on the SAME home-margin scale ``models/trainers/ats_trainer.py``
    # regresses (``_get_target_column`` returns ``home_margin``). So a POSITIVE ``ats_edge`` means
    # the model expects the home team to beat the line. (Before plan 31-17 this line negated the
    # stored spread and so computed model PLUS market: a model agreeing exactly with the market
    # scored the largest possible edge.)
    #
    # TWO COPIES, ONE RULE. This is a LIVE DISPLAY VALUE on the current-week page, and
    # ``api/cache.py`` derives its OWN ``ats_edge`` from ``outputs/backtest/predictions_all.csv``
    # and never reads this file -- which is exactly why a defect once survived in one copy after
    # being fixed in the other. Both now compute the same point difference in the same vectorized
    # shape, and ``tests/unit/test_current_week_ats_edge.py`` runs one committed table
    # (``tests.phase33_state.ATS_EDGE_PARITY_CASES``) through BOTH and asserts they agree
    # value-for-value. Mind the column name: this copy reads ``spread``, the cache reads
    # ``market_spread``.
    if "spread" in merged.columns:
        spread = merged["spread"]
        merged["ats_edge"] = (merged["ats_prediction"] - spread).where(spread.notna())
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

    # The EDGE BAND from the ONE shared source (D31-23, utils/edge_tier.py). The retired local
    # ``compute_confidence`` was a byte-equivalent twin of the one in ``api/cache.py``; the absent
    # case is now answered inside the helper rather than by a guard at each call site, so the two
    # call sites cannot answer it differently. The column names keep their historical
    # ``*_confidence`` spelling: renaming them would change this CSV's header, which is published.
    #
    # THE TARGET IS REQUIRED (CLEAN-01, D33-22). Each of the three edges above is in a DIFFERENT
    # unit -- a probability delta, POINTS, and a fraction of the total -- and the frozen
    # ``(high, medium)`` pair differs accordingly. ``target`` rides pandas' forwarded keyword, so
    # the unit is named at the call site; omitting it is a TypeError rather than a silent WP band
    # applied to a point margin. The CSV header is unchanged: the band moved, the column did not.
    merged["wp_confidence"] = merged["wp_edge"].apply(edge_tier, target="wp")
    merged["ats_confidence"] = merged["ats_edge"].apply(edge_tier, target="ats")
    merged["ou_confidence"] = merged["ou_edge"].apply(edge_tier, target="ou")

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
    """Apply market blending if a blend artifact is deployed.

    Adds blended_wp, blended_ats, blended_ou columns, NaN where blending does not apply.
    ONE fixed weight per target (D33.2-10): the week-varying schedule is retired, so neither
    the season nor the week reaches the blender any more.

    THE WP MARKET SIDE IS THE SPREAD (D33.2-09, LOCKED). It is the pre-lock spread converted
    through the converter slope BOUND on the loaded blender -- the slope
    ``MarketBlender.from_artifacts`` already cross-checked against the named converter
    directory, so the converter is NOT re-resolved here. This is SERVING, so the final serving
    slope is the right one; the out-of-fold rule binds history only. A game with a moneyline
    but no spread gets NO blended WP: the moneyline is not a fallback yardstick. It is still
    carried in the market frame for pricing, settlement and CLV (D33.2-11).

    Raises:
        RetiredDynamicBlendError: when the deployed blend is the retired dynamic incumbent.
        MarketProbabilityUnavailable: when the deployed blend has no converter bound.
    """
    predictions["blended_wp"] = np.nan
    predictions["blended_ats"] = np.nan
    predictions["blended_ou"] = np.nan

    if no_blend:
        logger.info("Blending skipped (--no-blend flag)")
        return predictions

    try:
        blender = MarketBlender.from_artifacts(artifacts_dir)
        logger.info(
            "Blend artifacts found, applying market blending",
            wp_weight=blender.config.weights.wp_model_weight,
            ats_weight=blender.config.weights.ats_model_weight,
            ou_weight=blender.config.weights.ou_model_weight,
        )
    except (KeyError, FileNotFoundError) as exc:
        # ONLY "no blend deployed" is caught. ``RetiredDynamicBlendError`` (a ValueError) is
        # DELIBERATELY left to propagate: the retired dynamic incumbent must fail this run
        # loudly rather than fall back to unblended predictions nobody chose (Plan 33.2-24).
        logger.info(
            "No blend artifacts found, using raw model predictions", reason=str(exc)
        )
        return predictions

    merged = predictions.merge(market, on="game_id", how="left", suffixes=("", "_mkt"))

    # WP: the spread through the BOUND converter.
    if "spread" in merged.columns:
        valid_wp = merged["spread"].notna() & merged["wp_prob"].notna()
        if valid_wp.any():
            if blender.market_probability_slope_beta is None:
                msg = (
                    "the deployed blend has no converter bound, so the pre-lock spread "
                    "cannot be converted into the WP market probability. Refusing rather "
                    "than publishing unblended WP as if it were blended."
                )
                raise MarketProbabilityUnavailable(msg)
            market_probs = market_home_win_probability(
                home_fav_margin_from_prelock_spread(
                    merged.loc[valid_wp, "spread"].to_numpy(dtype=float)
                ),
                blender.market_probability_slope_beta,
            )
            predictions.loc[valid_wp, "blended_wp"] = blender.blend_wp(
                merged.loc[valid_wp, "wp_prob"].to_numpy(dtype=np.float64),
                np.asarray(market_probs, dtype=np.float64),
            )

    # ATS blending
    if "spread" in merged.columns:
        valid_spread = merged["spread"].notna()
        if valid_spread.any():
            predictions.loc[valid_spread, "blended_ats"] = blender.blend_ats(
                merged.loc[valid_spread, "ats_prediction"].to_numpy(dtype=np.float64),
                merged.loc[valid_spread, "spread"].to_numpy(dtype=np.float64),
            )

    # O/U blending
    if "total" in merged.columns:
        valid_total = merged["total"].notna()
        if valid_total.any():
            predictions.loc[valid_total, "blended_ou"] = blender.blend_ou(
                merged.loc[valid_total, "ou_prediction"].to_numpy(dtype=np.float64),
                merged.loc[valid_total, "total"].to_numpy(dtype=np.float64),
            )

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

    # Only include columns that exist
    available_cols = [c for c in PREDICTION_OUTPUT_COLUMNS if c in predictions.columns]
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


def generate_and_write(
    season: int,
    week: int,
    output_dir: Path = Path("outputs/predictions"),
    artifacts_dir: Path = Path("artifacts"),
    no_blend: bool = False,
    *,
    excluded_game_ids: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Generate predictions for a season/week and write the output files.

    Runs all three model targets, computes edges against market lines, applies
    market blending when a blend artifact is present, and writes the predictions
    and game-context CSVs. This is the importable core shared by the CLI
    ``main()`` and the Friday pipeline's ``step_generate_predictions``.

    Args:
        season: NFL season year.
        week: NFL week number.
        output_dir: Directory for the prediction/context CSV files.
        artifacts_dir: Root directory for model and blend artifacts.
        no_blend: Skip market blending even if a blend artifact exists.
        excluded_game_ids: Games the live run dropped under the live-skip rule (D33.2-05,
            Plan 33.2-03). They are removed BEFORE scoring, so no prediction row is ever
            computed for them. Keyword-only and EMPTY by default, so the CLI and every existing
            caller produce exactly the output they did before.

    Returns:
        Summary dict with ``n_games``, ``n_excluded`` (how many of the week's games were
        dropped), ``n_blended``, ``predictions_path``, and ``context_path``.

    Raises:
        FileNotFoundError: A gold feature matrix is missing for a target.
        ValueError: No games found for the requested season/week.
        KeyError: A model's required feature is absent from the gold matrix.
    """
    logger.info(
        "Starting prediction generation",
        season=season,
        week=week,
        artifacts_dir=str(artifacts_dir),
        output_dir=str(output_dir),
    )

    # 1. Run model predictions for all three targets, excluded games dropped before scoring
    n_excluded = _count_excluded_in_week(season, week, excluded_game_ids)
    prediction_results = run_predictions(
        artifacts_dir, season, week, excluded_game_ids=excluded_game_ids
    )

    # 2. Merge predictions into a single DataFrame
    wp_df = prediction_results["wp"]  # game_id, wp_prob
    ats_df = prediction_results["ats"]  # game_id, ats_prediction
    ou_df = prediction_results["ou"]  # game_id, ou_prediction

    combined = wp_df.merge(ats_df, on="game_id", how="outer")
    combined = combined.merge(ou_df, on="game_id", how="outer")
    combined["season"] = season
    combined["week"] = week

    if combined.empty:
        return _write_all_skipped_week(season, week, output_dir, n_excluded)

    # 3. Load market data
    game_ids = combined["game_id"].tolist()
    market_df = load_market_data(game_ids)

    # 4. Join game info from silver games (home_team, away_team)
    games_path = Path("data/silver/games.parquet")
    if games_path.exists():
        games = pd.read_parquet(games_path)
        games_filtered = games[games["game_id"].isin(game_ids)][
            ["game_id", "home_team", "away_team"]
        ].drop_duplicates(subset=["game_id"])
        combined = combined.merge(games_filtered, on="game_id", how="left")

    # 5. Compute edges, then rename market columns for output clarity
    combined = compute_edges(combined, market_df)
    if "spread" in combined.columns:
        combined.rename(columns={"spread": "market_spread"}, inplace=True)
    if "total" in combined.columns:
        combined.rename(columns={"total": "market_total"}, inplace=True)
    if "ml_home" in combined.columns:
        combined.rename(columns={"ml_home": "market_ml_home"}, inplace=True)
    if "ml_away" in combined.columns:
        combined.rename(columns={"ml_away": "market_ml_away"}, inplace=True)

    # 6. Apply market blending (the deployed fixed-weight blend artifact)
    combined = apply_blending(combined, market_df, artifacts_dir, no_blend)

    # 7. Write outputs
    pred_path = write_predictions(combined, season, week, output_dir)
    context_df = build_game_context(game_ids, season, week)
    context_path = write_game_context(context_df, season, week, output_dir)

    n_blended = int(combined["blended_wp"].notna().sum())
    logger.info(
        "Prediction generation complete",
        n_games=len(combined),
        blending_applied=n_blended > 0,
        n_blended=n_blended,
        predictions_file=str(pred_path),
        context_file=str(context_path),
    )
    return {
        "n_games": len(combined),
        "n_excluded": n_excluded,
        "n_blended": n_blended,
        "predictions_path": pred_path,
        "context_path": context_path,
    }


def _count_excluded_in_week(
    season: int, week: int, excluded_game_ids: frozenset[str]
) -> int:
    """How many of the week's WP gold games are in *excluded_game_ids*. 0 when none excluded."""
    if not excluded_game_ids:
        return 0
    path = Path("data/gold/features_wp.parquet")
    if not path.exists():
        return 0
    week_ids = pd.read_parquet(path, columns=["game_id", "season", "week"])
    in_week = week_ids.loc[(week_ids["season"] == season) & (week_ids["week"] == week)]
    return int(in_week["game_id"].astype(str).isin(sorted(excluded_game_ids)).sum())


def _write_all_skipped_week(
    season: int, week: int, output_dir: Path, n_excluded: int
) -> dict[str, Any]:
    """Write the well-formed EMPTY outputs of a day on which every game was excluded.

    SPEC R3: an all-caught day finishes with skips and ZERO predictions -- never "no games".
    The prediction CSV carries the full published header and no rows, so the currency and
    validation steps read a well-formed empty week rather than a malformed file, and the
    game-context CSV likewise carries its key column only.
    """
    empty = pd.DataFrame(columns=pd.Index(PREDICTION_OUTPUT_COLUMNS))
    pred_path = write_predictions(empty, season, week, output_dir)
    context_path = write_game_context(
        pd.DataFrame(columns=pd.Index(["game_id"])), season, week, output_dir
    )
    logger.info(
        "Every game excluded; wrote an empty prediction set",
        season=season,
        week=week,
        n_excluded=n_excluded,
    )
    return {
        "n_games": 0,
        "n_excluded": n_excluded,
        "n_blended": 0,
        "predictions_path": pred_path,
        "context_path": context_path,
    }


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

    try:
        generate_and_write(
            season=args.season,
            week=args.week,
            output_dir=Path(args.output_dir),
            artifacts_dir=Path(args.artifacts_dir),
            no_blend=args.no_blend,
        )
    except (FileNotFoundError, ValueError, KeyError) as exc:
        logger.error("Prediction generation failed", error=str(exc))
        sys.exit(1)


if __name__ == "__main__":
    main()
