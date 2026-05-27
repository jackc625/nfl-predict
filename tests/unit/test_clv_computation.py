"""Tests for CLV computation module (MODL-05).

Verifies:
- Probability-based CLV with proper devigging
- Line-based CLV for ATS/O/U
- Batch CLV computation with missing odds handling
"""

import numpy as np
import pandas as pd
import pytest

from models.clv import (
    compute_clv_for_predictions,
    compute_line_clv,
    compute_probability_clv,
)
from utils.probability_utils import moneyline_to_probability

# ---------------------------------------------------------------------------
# Test 1: Probability CLV with positive edge
# ---------------------------------------------------------------------------


def test_probability_clv_positive_edge():
    """Model prob 0.60 vs closing ML -150/+130 yields positive CLV.

    Fair home prob after devig: 150/(150+100) / (150/(150+100) + 100/(130+100))
    = 0.6 / (0.6 + 0.4348) = 0.6 / 1.0348 ~= 0.5799
    CLV = 0.60 - 0.5799 ~= 0.0201 (positive edge)
    """
    result = compute_probability_clv(
        model_prob=0.60,
        closing_ml_home=-150,
        closing_ml_away=130,
        side="home",
    )

    assert result["probability_clv"] > 0, "Expected positive CLV"
    assert abs(result["probability_clv"] - 0.020) < 0.01, (
        f"CLV should be ~0.020, got {result['probability_clv']:.4f}"
    )
    assert result["model_prob"] == 0.60
    assert 0 < result["fair_closing_prob"] < 1


# ---------------------------------------------------------------------------
# Test 2: Probability CLV with no edge
# ---------------------------------------------------------------------------


def test_probability_clv_no_edge():
    """Model prob equals fair closing prob returns CLV ~0.0."""
    # -150 / +130: fair home ~0.5799
    home_raw = moneyline_to_probability(-150)
    away_raw = moneyline_to_probability(130)
    fair_home = home_raw / (home_raw + away_raw)

    result = compute_probability_clv(
        model_prob=fair_home,
        closing_ml_home=-150,
        closing_ml_away=130,
        side="home",
    )

    assert abs(result["probability_clv"]) < 1e-10, (
        f"CLV should be ~0.0, got {result['probability_clv']}"
    )


# ---------------------------------------------------------------------------
# Test 3: Devig removes vig
# ---------------------------------------------------------------------------


def test_devig_removes_vig():
    """Raw ML -110/-110 sum to > 1.0, fair probs sum to exactly 1.0."""
    home_raw = moneyline_to_probability(-110)
    away_raw = moneyline_to_probability(-110)

    # Raw probs should sum to > 1.0 (vig)
    assert home_raw + away_raw > 1.0, "Raw probs should include vig"

    # Compute CLV -- internally devigs
    result = compute_probability_clv(
        model_prob=0.50,
        closing_ml_home=-110,
        closing_ml_away=-110,
        side="home",
    )

    # Fair closing prob should be 0.5 for -110/-110
    assert abs(result["fair_closing_prob"] - 0.5) < 0.001, (
        f"Fair prob for -110/-110 should be 0.5, got {result['fair_closing_prob']}"
    )

    # Vig percentage should be reported
    assert result["vig_pct"] > 0


# ---------------------------------------------------------------------------
# Test 4: CLV excludes missing odds
# ---------------------------------------------------------------------------


def test_clv_excludes_missing_odds():
    """Games with NaN closing odds are excluded from CLV computation."""
    predictions_df = pd.DataFrame(
        {
            "game_id": ["g1", "g2", "g3"],
            "model_prob": [0.60, 0.55, 0.70],
        }
    )

    closing_odds_df = pd.DataFrame(
        {
            "game_id": ["g1", "g2", "g3"],
            "ml_home": [-150.0, np.nan, -200.0],
            "ml_away": [130.0, np.nan, 170.0],
        }
    )

    result = compute_clv_for_predictions(
        predictions_df=predictions_df,
        closing_odds_df=closing_odds_df,
        target="wp",
    )

    # g2 should be excluded (NaN odds)
    assert result["has_closing_odds"].sum() == 2
    assert (
        result.loc[result["game_id"] == "g2", "has_closing_odds"].iloc[0] is np.False_
    )


# ---------------------------------------------------------------------------
# Test 5: Line CLV for ATS
# ---------------------------------------------------------------------------


def test_line_clv_ats():
    """For ATS, line-based CLV = closing_spread - model_spread."""
    # Model predicts margin of -3.5, closing spread is -7.0
    # CLV = closing_spread - model_spread = (-7.0) - (-3.5) = -3.5
    # Negative because market moved away from our prediction
    clv = compute_line_clv(
        model_value=-3.5,
        closing_value=-7.0,
        direction="spread",
    )

    assert clv == pytest.approx(-3.5), f"Expected line CLV of -3.5, got {clv}"

    # Model predicts -7, closing at -3.5 (market moved toward us)
    clv_positive = compute_line_clv(
        model_value=-7.0,
        closing_value=-3.5,
        direction="spread",
    )
    assert clv_positive == pytest.approx(3.5)


# ---------------------------------------------------------------------------
# Test 6: Batch CLV computation
# ---------------------------------------------------------------------------


def test_compute_clv_for_predictions_batch():
    """Batch function returns DataFrame with probability_clv column."""
    predictions_df = pd.DataFrame(
        {
            "game_id": ["g1", "g2"],
            "model_prob": [0.60, 0.55],
        }
    )

    closing_odds_df = pd.DataFrame(
        {
            "game_id": ["g1", "g2"],
            "ml_home": [-150.0, -110.0],
            "ml_away": [130.0, -110.0],
        }
    )

    result = compute_clv_for_predictions(
        predictions_df=predictions_df,
        closing_odds_df=closing_odds_df,
        target="wp",
    )

    assert "probability_clv" in result.columns
    assert "fair_closing_prob" in result.columns
    assert "has_closing_odds" in result.columns
    assert len(result) == 2
    assert result["has_closing_odds"].all()
