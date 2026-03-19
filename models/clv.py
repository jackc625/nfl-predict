"""Closing Line Value (CLV) computation module.

Provides:
- compute_probability_clv: Single-game probability-based CLV with devigging
- compute_line_clv: Single-game line-based CLV for ATS/O/U
- compute_clv_for_predictions: Batch CLV computation across a DataFrame

CLV is the primary evaluation metric for model quality. Positive CLV means
the model identified value before the market closed, which is the strongest
indicator of long-term edge.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils import get_logger
from utils.probability_utils import moneyline_to_probability

logger = get_logger(__name__)


def compute_probability_clv(
    model_prob: float,
    closing_ml_home: float,
    closing_ml_away: float,
    side: str = "home",
) -> dict[str, float]:
    """Compute probability-based CLV for a single game.

    Devigs closing moneylines to obtain fair closing probabilities,
    then computes CLV = model_prob - fair_closing_prob.

    Args:
        model_prob: Model's probability for the given side.
        closing_ml_home: Closing moneyline for the home team.
        closing_ml_away: Closing moneyline for the away team.
        side: Which side the model prediction is for ("home" or "away").

    Returns:
        Dict with probability_clv, fair_closing_prob, model_prob, vig_pct.
    """
    # Compute raw (vigged) implied probabilities
    home_raw = moneyline_to_probability(int(closing_ml_home))
    away_raw = moneyline_to_probability(int(closing_ml_away))
    total_raw = home_raw + away_raw

    # Devig: proportional method (divide by overround)
    fair_home = home_raw / total_raw
    fair_away = away_raw / total_raw

    if side == "home":
        fair_closing = fair_home
    else:
        fair_closing = fair_away

    # CLV = model probability minus fair closing probability
    clv = model_prob - fair_closing

    return {
        "probability_clv": clv,
        "fair_closing_prob": fair_closing,
        "model_prob": model_prob,
        "vig_pct": total_raw - 1.0,
    }


def compute_line_clv(
    model_value: float,
    closing_value: float,
    direction: str = "spread",
) -> float:
    """Compute line-based CLV for a single game.

    For spread: CLV = closing_spread - model_spread
        Positive = market moved toward our number (we had value)
    For total: CLV = model_total - closing_total
        Positive = model predicted higher and market moved up

    Args:
        model_value: Model's predicted value (margin or total).
        closing_value: Closing line value (spread or total).
        direction: "spread" or "total".

    Returns:
        Line-based CLV value.
    """
    if direction == "spread":
        # Spread CLV: closing_spread - model_spread
        # If model predicted -7 and market closed at -3.5,
        # CLV = -3.5 - (-7) = 3.5 (market moved toward us)
        return closing_value - model_value
    if direction == "total":
        # Total CLV: model_total - closing_total
        # If model predicted 48 and market closed at 45,
        # CLV = 48 - 45 = 3 (model was ahead of the market)
        return model_value - closing_value
    msg = f"Unknown direction: {direction}. Must be 'spread' or 'total'."
    raise ValueError(msg)


def compute_clv_for_predictions(
    predictions_df: pd.DataFrame,
    closing_odds_df: pd.DataFrame,
    target: str,
    merge_on: str = "game_id",
) -> pd.DataFrame:
    """Compute CLV for a batch of predictions.

    Merges predictions with closing odds, computes probability-based CLV
    (and line-based CLV where applicable), and flags games without odds.

    Args:
        predictions_df: DataFrame with model predictions. Expected columns
            vary by target:
            - wp: model_prob
            - ats: model_prob (cover probability) and/or model_spread
            - ou: model_prob (over probability) and/or model_total
        closing_odds_df: DataFrame with closing odds. Expected columns:
            game_id, ml_home, ml_away, and optionally spread, total.
        target: One of "wp", "ats", "ou".
        merge_on: Column to merge on.

    Returns:
        Augmented DataFrame with probability_clv, fair_closing_prob,
        has_closing_odds, and line_clv (where applicable).
    """
    # Merge predictions with closing odds
    merged = predictions_df.merge(closing_odds_df, on=merge_on, how="left")

    # Flag which games have valid closing odds
    merged["has_closing_odds"] = merged["ml_home"].notna() & merged["ml_away"].notna()

    n_missing = (~merged["has_closing_odds"]).sum()
    if n_missing > 0:
        logger.warning(
            "Games excluded from CLV computation due to missing closing odds",
            n_missing=int(n_missing),
            n_total=len(merged),
        )

    # Initialize CLV columns with NaN
    merged["probability_clv"] = np.nan
    merged["fair_closing_prob"] = np.nan

    # Compute CLV only for games with valid closing odds
    valid_mask = merged["has_closing_odds"]

    if target == "wp":
        _compute_wp_clv(merged, valid_mask)
    elif target == "ats":
        _compute_ats_clv(merged, valid_mask)
    elif target == "ou":
        _compute_ou_clv(merged, valid_mask)
    else:
        msg = f"Unknown target: {target}. Must be 'wp', 'ats', or 'ou'."
        raise ValueError(msg)

    return merged


def _compute_wp_clv(df: pd.DataFrame, valid_mask: pd.Series) -> None:
    """Compute probability CLV for WP target (in-place)."""
    for idx in df.index[valid_mask]:
        row = df.loc[idx]
        result = compute_probability_clv(
            model_prob=row["model_prob"],
            closing_ml_home=row["ml_home"],
            closing_ml_away=row["ml_away"],
            side="home",
        )
        df.at[idx, "probability_clv"] = result["probability_clv"]
        df.at[idx, "fair_closing_prob"] = result["fair_closing_prob"]


def _compute_ats_clv(df: pd.DataFrame, valid_mask: pd.Series) -> None:
    """Compute probability and line CLV for ATS target (in-place)."""
    # Probability CLV from cover probability vs spread-implied probs
    if "model_prob" in df.columns:
        _compute_wp_clv(df, valid_mask)

    # Line CLV from model spread vs closing spread
    if "model_spread" in df.columns and "spread" in df.columns:
        df["line_clv"] = np.nan
        for idx in df.index[valid_mask]:
            row = df.loc[idx]
            if pd.notna(row.get("spread")):
                df.at[idx, "line_clv"] = compute_line_clv(
                    model_value=row["model_spread"],
                    closing_value=row["spread"],
                    direction="spread",
                )


def _compute_ou_clv(df: pd.DataFrame, valid_mask: pd.Series) -> None:
    """Compute probability and line CLV for O/U target (in-place)."""
    # Probability CLV from over probability
    if "model_prob" in df.columns:
        _compute_wp_clv(df, valid_mask)

    # Line CLV from model total vs closing total
    if "model_total" in df.columns and "total" in df.columns:
        df["line_clv"] = np.nan
        for idx in df.index[valid_mask]:
            row = df.loc[idx]
            if pd.notna(row.get("total")):
                df.at[idx, "line_clv"] = compute_line_clv(
                    model_value=row["model_total"],
                    closing_value=row["total"],
                    direction="total",
                )
