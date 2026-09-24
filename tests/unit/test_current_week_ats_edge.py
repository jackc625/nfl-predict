"""R13 / D33-05 / D33-31: the CURRENT-WEEK ATS edge is a POINT DIFFERENCE, and the two copies agree.

WHAT WAS WRONG, IN TWO GENERATIONS
----------------------------------
GENERATION 1 (DEF-31-03, repaired in Phase 31, plan 31-17).
``scripts.generate_current_week_predictions.compute_edges`` computed::

    (ats_prediction - (-spread)) / abs(spread)

which is ``(model + market) / |market|``. The negation assumed a line convention DEF-31-01
MEASURED to be the opposite of the stored one, and the owner ruled against on 2026-09-04. Under a
sum, a model that agrees EXACTLY with the market scores the LARGEST POSSIBLE edge.

THE MEASUREMENT, REPRODUCED INDEPENDENTLY BEFORE THAT FIX (2026-09-06). Over all 2140 rows of
``data/silver/odds_snapshot.parquet`` and their realized results::

    corr(spread, ml_home)                     -0.9506
    mean spread | home is a big favourite     +9.19   (ml_home <= -250, n=565)
    mean spread | away is a big favourite     -8.49   (ml_away <= -250, n=263)
    corr(spread, realized home margin)        +0.4517

So the stored ``spread`` is the nflverse ``spread_line``, POSITIVE when the home team is favored,
and ``ats_prediction`` is a predicted home MARGIN (``models/trainers/ats_trainer.py``'s
``_get_target_column`` returns ``home_margin``). Same scale; the disagreement is the DIFFERENCE.
That reasoning is unchanged and is the reason the repair below is arithmetically simple.

GENERATION 2 (R13, repaired by Plan 33-10 -- THIS module's subject). What survived was still a
RATIO, ``(ats_prediction - spread) / |spread|``, whose divisor is a point count that approaches
zero. A three-point disagreement on a half-point line scored 6.0, and ``utils.edge_tier`` bands it
against the same ``0.05 / 0.02`` pair it applies to a WP probability.

``ats_edge`` is now ``ats_prediction - spread``, in POINTS, with exactly ONE surviving branch: no
stored line means NO edge. Per D33-31 the ``spread == 0`` branch that forced a pick-em to 0.0 is
GONE from BOTH copies.

THE CROSS-COPY PARITY TABLE
---------------------------
``api/cache.py`` derives its own ``ats_edge`` from ``outputs/backtest/predictions_all.csv`` and
NEVER reads the current-week file, so nothing in the running system would notice if one copy were
repaired and the other left as a ratio -- which is precisely what happened under DEF-31-03. The
table lives in ``tests.phase33_state.ATS_EDGE_PARITY_CASES``, committed once, and every row is run
through BOTH callables and compared value-for-value, with the absent-spread row compared through
explicit NaN handling rather than by an equality that would skip it.

PIN RECONCILIATION. Four of the seven pre-existing cases MOVED their expected value and one
(``..._is_normalized_by_the_magnitude_of_the_market_line``) had its CLAIM INVERTED, because
normalization by the line is the defect R13 names. Each carries an inline comment with the old
value, the new value and the reason. Nothing was silently rewritten.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from scripts.generate_current_week_predictions import compute_edges
from tests.phase33_state import ATS_EDGE_PARITY_CASES


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


def _market(spread: float | None) -> pd.DataFrame:
    """The stored market row. ``spread`` is the nflverse ``spread_line`` (positive = home favored)."""
    return pd.DataFrame(
        [
            {
                "game_id": "2025_W01_DAL@PHI",
                "spread": np.nan if spread is None else spread,
                "total": 47.5,
                "ml_home": -425.0,
                "ml_away": 330.0,
            }
        ]
    )


def _ats_edge(ats_prediction: float, spread: float | None) -> float:
    merged = compute_edges(_predictions(ats_prediction), _market(spread))
    return float(merged.loc[0, "ats_edge"])


# ---------------------------------------------------------------------------
# The pre-existing pins, reconciled case by case.
# ---------------------------------------------------------------------------


def test_a_model_that_agrees_with_the_market_has_no_ats_edge() -> None:
    """The sharpest case. UNCHANGED: 0.0 under the ratio and under the point margin alike.

    Under the original DEF-31-03 expression this returned +2.0.
    """
    assert _ats_edge(ats_prediction=8.5, spread=8.5) == pytest.approx(0.0)


def test_a_model_that_agrees_with_a_road_favourite_line_has_no_ats_edge() -> None:
    """The same claim with the sign flipped. UNCHANGED. (DEF-31-03 expression: -2.0.)"""
    assert _ats_edge(ats_prediction=-6.0, spread=-6.0) == pytest.approx(0.0)


def test_a_model_more_bullish_on_the_home_side_than_the_market_has_a_positive_edge() -> (
    None
):
    """Model expects home by 12, market by 8.5: a +3.5-point disagreement toward home.

    MOVED PIN. Old expected value ``3.5 / 8.5 = 0.411765``; new expected value ``3.5``. REASON:
    R13 / D33-05 put the edge on a fixed POINTS scale, so it IS the disagreement rather than the
    disagreement divided by the line. The claim the pin protects -- that a more bullish model
    carries a positive edge -- is untouched.
    """
    assert _ats_edge(ats_prediction=12.0, spread=8.5) == pytest.approx(3.5)


def test_a_model_more_bearish_on_the_home_side_than_the_market_has_a_negative_edge() -> (
    None
):
    """Model expects home by 4, market by 8.5: a -4.5-point disagreement away from home.

    MOVED PIN. Old expected value ``-4.5 / 8.5 = -0.529412``; new expected value ``-4.5``.
    REASON: the same points-scale redefinition. The SIGN is what this pin was written to protect
    (the DEF-31-03 expression returned +1.471 here, reporting a bet ON a home side the model
    thinks is overvalued) and the sign is unchanged.
    """
    assert _ats_edge(ats_prediction=4.0, spread=8.5) == pytest.approx(-4.5)


def test_the_edge_is_no_longer_normalized_by_the_magnitude_of_the_market_line() -> None:
    """INVERTED PIN -- the only one whose CLAIM reversed, and deliberately so.

    Old name: ``test_the_edge_is_normalized_by_the_magnitude_of_the_market_line``. Old
    expectations: ``wide == 3.0 / 10.0``, ``narrow == 3.0 / 3.0``, ``narrow > wide``. New
    expectations: both are ``3.0`` and they are EQUAL.

    REASON: normalization by the line IS the defect R13 names. The old pin faithfully asserted
    the behaviour of its generation -- "the same point-difference against a shorter line is a
    proportionally larger edge" -- and that behaviour is what turned a 3.0-point disagreement on
    a half-point line into an edge of 6.0. The pin is kept, inverted and annotated rather than
    deleted, so the reversal is visible in the record instead of vanishing from it.
    """
    wide = _ats_edge(ats_prediction=13.0, spread=10.0)
    narrow = _ats_edge(ats_prediction=6.0, spread=3.0)

    assert wide == pytest.approx(3.0)
    assert narrow == pytest.approx(3.0)
    assert narrow == pytest.approx(wide)


def test_a_pick_em_line_carries_the_full_point_disagreement() -> None:
    """D33-31, the owner ruling, in the second copy.

    MOVED PIN. Old name ``..._yields_a_zero_edge_rather_than_a_division_by_zero``; old expected
    value ``0.0``; new expected value ``3.0``. REASON: the forced zero was right while the edge
    was a ratio and wrong the moment it became a point margin -- a model with the home team by
    three against a pick'em line disagrees with the market by exactly three points. A pick'em is
    a REAL line, not a missing one. Both copies dropped the branch together.
    """
    merged = compute_edges(_predictions(3.0), _market(0.0))
    assert float(merged.loc[0, "ats_edge"]) == pytest.approx(3.0)


def test_an_absent_market_line_leaves_the_edge_absent_and_no_band() -> None:
    """The ONLY surviving special branch: no line, no edge, and NO band.

    No line, no disagreement to measure. The band used to answer ``low`` here, which published a
    small measured edge for a game that has none (33.2 review C2 CR-04 = B WR-11); it is now
    absent. It must stay distinguishable from the pick-em case above, which is a real line at
    zero and does carry a band.
    """
    market = _market(8.5)
    market.loc[0, "spread"] = np.nan
    merged = compute_edges(_predictions(12.0), market)
    assert pd.isna(merged.loc[0, "ats_edge"])
    assert merged.loc[0, "ats_confidence"] is None


# ---------------------------------------------------------------------------
# The cross-copy parity table -- one committed list, both callables.
# ---------------------------------------------------------------------------


def _cache_ats_edge(
    tmp_path, spread: float | None, ats_prediction: float
) -> float | None:
    """Drive the OTHER copy through its own test module's helper, not a reimplementation."""
    from tests.unit.test_cache_ats_edge import _ats_edge as cache_ats_edge

    return cache_ats_edge(tmp_path, ats_prediction=ats_prediction, spread=spread)


@pytest.mark.parametrize(
    ("market_spread", "ats_prediction", "expected"),
    ATS_EDGE_PARITY_CASES,
    ids=[
        f"spread={'none' if case[0] is None else case[0]}_pred={case[1]}"
        for case in ATS_EDGE_PARITY_CASES
    ],
)
def test_both_copies_of_the_formula_agree_value_for_value(
    tmp_path,
    market_spread: float | None,
    ats_prediction: float,
    expected: float | None,
) -> None:
    """THE drift finding, as a table: two artifacts from one codebase, one ATS edge.

    The absent-spread row is compared through explicit NaN handling. ``nan == nan`` is False and
    ``pytest.approx(nan) == nan`` is False by default, so a row left to a bare equality would
    have been reported as a DISAGREEMENT; a row left to a truthiness check would have been
    silently skipped. Neither is acceptable for the one branch this repair preserves.
    """
    from_csv = compute_edges(_predictions(ats_prediction), _market(market_spread)).loc[
        0, "ats_edge"
    ]
    from_cache = _cache_ats_edge(tmp_path, market_spread, ats_prediction)

    csv_absent = pd.isna(from_csv)
    cache_absent = from_cache is None or pd.isna(from_cache)

    if expected is None:
        assert csv_absent, (
            f"spread={market_spread!r} has no stored line, so the current-week copy must report "
            f"NO edge rather than {from_csv!r}"
        )
        assert cache_absent, (
            f"spread={market_spread!r} has no stored line, so the cache copy must report NO edge "
            f"rather than {from_cache!r}"
        )
        return

    assert not csv_absent, (
        f"the current-week copy lost the edge for spread={market_spread!r}"
    )
    assert not cache_absent, (
        f"the cache copy lost the edge for spread={market_spread!r}"
    )

    assert float(from_csv) == pytest.approx(expected)
    assert float(from_cache) == pytest.approx(expected)
    assert float(from_cache) == pytest.approx(float(from_csv)), (
        f"the cache says {from_cache} and the current-week CSV says {from_csv} for "
        f"spread={market_spread!r}, ats_prediction={ats_prediction!r}. One copy of a duplicated "
        "formula was repaired and the other was not."
    )


def test_the_parity_table_carries_a_pick_em_with_a_non_zero_prediction() -> None:
    """Non-vacuity for D33-31: without this row the table passes under the OLD forced zero."""
    pickem_non_zero = [
        case
        for case in ATS_EDGE_PARITY_CASES
        if case[0] == 0.0 and case[1] not in (0, 0.0)
    ]
    assert pickem_non_zero, "the parity table has no pick-em with a non-zero prediction"
    assert all(case[2] not in (0, 0.0) for case in pickem_non_zero), (
        "a pick-em with a non-zero prediction expects a non-zero edge under D33-31; an expected "
        "0.0 here would encode the pre-ruling behaviour"
    )


def test_the_parity_table_covers_every_shape_the_repair_changes() -> None:
    """Positive, negative, half-point, pick-em, absent, large favourite and large underdog."""
    spreads = [case[0] for case in ATS_EDGE_PARITY_CASES]
    assert len(ATS_EDGE_PARITY_CASES) >= 7
    assert any(s is not None and s > 0 for s in spreads)
    assert any(s is not None and s < 0 for s in spreads)
    assert sum(1 for s in spreads if s is not None and abs(s) == 0.5) >= 2
    assert any(s == 0.0 for s in spreads)
    assert any(s is None for s in spreads)
    assert any(s is not None and s <= -10.0 for s in spreads)
    assert any(s is not None and s >= 10.0 for s in spreads)


# ---------------------------------------------------------------------------
# The two copies must be STRUCTURALLY comparable, not merely numerically lucky.
# ---------------------------------------------------------------------------


def _ats_block(source: str) -> str:
    """The ATS slice of ``compute_edges``, from its own comment to the O/U comment.

    SCOPED DELIBERATELY. ``compute_edges`` also computes ``wp_edge`` and ``ou_edge``, both of
    which legitimately keep a row-wise ``.apply(lambda row: ...)`` and, in the O/U case, a
    zero-total branch. A whole-function scan for those strings would report this plan's subject
    as unrepaired while looking at code this plan does not touch.
    """
    start = source.index("# ATS edge:")
    end = source.index("# O/U edge:", start)
    return source[start:end]


def test_the_current_week_copy_computes_the_ats_edge_vectorized() -> None:
    """Both copies must have the SAME SHAPE, so a reader comparing them sees one rule twice."""
    block = _ats_block(inspect.getsource(compute_edges))

    assert "lambda row" not in block, (
        "the ATS edge is still computed row-wise; the cache copy is a vectorized Series "
        "expression and the two must be structurally comparable"
    )
    assert "/ abs(row[" not in block and "abs(row[" not in block
    assert '"spread"' in block, "this copy reads `spread`, not `market_spread`"
    assert "notna()" in block, "the absent-spread branch must survive"


def test_the_current_week_copy_has_no_forced_pick_em_zero() -> None:
    """D33-31 removed the branch from BOTH copies; one repaired alone is the drift, not the fix."""
    block = _ats_block(inspect.getsource(compute_edges))

    assert "= 0.0" not in block, (
        "a forced zero survives in the ATS block of compute_edges; under D33-31 a pick-em is a "
        "real line and its edge is the plain disagreement"
    )
    assert "zero_spread" not in block
