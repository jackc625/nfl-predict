"""DEF-31-03: the LIVE current-week ATS edge is a difference, not a sum (Phase 31, plan 31-17).

WHAT WAS WRONG
--------------
``scripts/generate_current_week_predictions.compute_edges`` computed the current-week ATS edge as::

    (ats_prediction - (-spread)) / abs(spread)

which is ``(model + market) / |market|``. The negation assumed a line convention that DEF-31-01
MEASURED to be the opposite of the stored one, and the owner ruled against on 2026-09-04. The
consequence is not subtle: under a sum, a model that agrees EXACTLY with the market scores the
LARGEST POSSIBLE edge, and a model that disagrees most scores zero. This is a live display value on
the current-week page -- what the reader is shown -- which is why the register flagged it for
re-check once DEF-31-01 was ruled on.

THE MEASUREMENT, REPRODUCED INDEPENDENTLY BEFORE THE FIX (2026-09-06)
---------------------------------------------------------------------
Over all 2140 rows of ``data/silver/odds_snapshot.parquet`` and their realized results:

    corr(spread, ml_home)                     -0.9506
    mean spread | home is a big favourite     +9.19   (ml_home <= -250, n=565)
    mean spread | away is a big favourite     -8.49   (ml_away <= -250, n=263)
    corr(spread, realized home margin)        +0.4517

So the stored ``spread`` is the nflverse ``spread_line``, POSITIVE when the home team is favored.
And ``ats_prediction`` is a predicted home MARGIN: ``models/trainers/ats_trainer.py``'s
``_get_target_column`` returns ``home_margin``, which ``scripts/build_features.py`` defines as
``home_score - away_score``. The two are on the SAME scale, so the model's disagreement with the
market is their DIFFERENCE.

EVERY TEST BELOW FAILS UNDER THE OLD EXPRESSION. The agreement case is the sharpest: it expects
0.0 and the old code returns +2.0.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from scripts.generate_current_week_predictions import compute_edges


def _predictions(ats_prediction: float) -> pd.DataFrame:
    """One current-week prediction row carrying every column ``compute_edges`` reads."""
    return pd.DataFrame(
        [
            {
                "game_id": "2025_W01_DAL@PHI",
                "wp_prob": 0.60,
                "ats_prediction": ats_prediction,
                "ou_prediction": 44.0,
            }
        ]
    )


def _market(spread: float) -> pd.DataFrame:
    """The stored market row. ``spread`` is the nflverse ``spread_line`` (positive = home favored)."""
    return pd.DataFrame(
        [
            {
                "game_id": "2025_W01_DAL@PHI",
                "spread": spread,
                "total": 47.5,
                "ml_home": -425.0,
                "ml_away": 330.0,
            }
        ]
    )


def _ats_edge(ats_prediction: float, spread: float) -> float:
    merged = compute_edges(_predictions(ats_prediction), _market(spread))
    return float(merged.loc[0, "ats_edge"])


def test_a_model_that_agrees_with_the_market_has_no_ats_edge() -> None:
    """The sharpest case. Old expression: +2.0 -- the largest edge the formula can produce."""
    assert _ats_edge(ats_prediction=8.5, spread=8.5) == pytest.approx(0.0)


def test_a_model_that_agrees_with_a_road_favourite_line_has_no_ats_edge() -> None:
    """The same claim with the sign flipped, so a symmetric bug cannot pass one and fail the other.

    Old expression: -2.0.
    """
    assert _ats_edge(ats_prediction=-6.0, spread=-6.0) == pytest.approx(0.0)


def test_a_model_more_bullish_on_the_home_side_than_the_market_has_a_positive_edge() -> (
    None
):
    """Model expects home by 12, market by 8.5: a +3.5-point disagreement toward home.

    Old expression: (12 + 8.5) / 8.5 = +2.412 -- a number with no relation to the disagreement.
    """
    assert _ats_edge(ats_prediction=12.0, spread=8.5) == pytest.approx(3.5 / 8.5)


def test_a_model_more_bearish_on_the_home_side_than_the_market_has_a_negative_edge() -> (
    None
):
    """Model expects home by 4, market by 8.5: a -4.5-point disagreement away from home.

    Old expression: (4 + 8.5) / 8.5 = +1.471 -- POSITIVE, so the old code reported a bet ON the
    home side for a model that thinks the home side is overvalued. That inversion is the defect's
    practical consequence on the page.
    """
    assert _ats_edge(ats_prediction=4.0, spread=8.5) == pytest.approx(-4.5 / 8.5)


def test_the_edge_is_normalized_by_the_magnitude_of_the_market_line() -> None:
    """The same point-difference against a shorter line is a proportionally larger edge."""
    wide = _ats_edge(ats_prediction=13.0, spread=10.0)
    narrow = _ats_edge(ats_prediction=6.0, spread=3.0)
    assert wide == pytest.approx(3.0 / 10.0)
    assert narrow == pytest.approx(3.0 / 3.0)
    assert narrow > wide


def test_a_pick_em_line_yields_a_zero_edge_rather_than_a_division_by_zero() -> None:
    """Pre-existing behaviour, pinned so the fix did not disturb it."""
    merged = compute_edges(_predictions(3.0), _market(0.0))
    assert float(merged.loc[0, "ats_edge"]) == pytest.approx(0.0)


def test_an_absent_market_line_leaves_the_edge_absent_and_the_band_low() -> None:
    """No line, no disagreement to measure -- and the band answers ``low``, not a made-up number."""
    market = _market(8.5)
    market.loc[0, "spread"] = np.nan
    merged = compute_edges(_predictions(12.0), market)
    assert pd.isna(merged.loc[0, "ats_edge"])
    assert merged.loc[0, "ats_confidence"] == "low"
