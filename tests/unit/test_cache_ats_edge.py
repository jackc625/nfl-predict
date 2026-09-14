"""R13 / D33-05 / D33-31: the CACHE's ATS edge is a POINT DIFFERENCE, not an unbounded ratio.

WHAT WAS WRONG, IN TWO GENERATIONS
----------------------------------
GENERATION 1 (WR-02, repaired in Phase 31). ``api.cache._load_predictions`` computed::

    (ats_prediction - (-market_spread)) / market_spread.abs().clip(lower=0.5)

which is ``(model + market) / |market|``: a SUM, not a difference. A model agreeing EXACTLY with
the market scored the largest edge the formula can produce. That negation is gone, and the
``.clip(lower=0.5)`` denominator floor went with it.

GENERATION 2 (R13, repaired by Plan 33-10 -- THIS module's subject). What survived the first
repair was still a RATIO::

    (ats_prediction - market_spread) / |market_spread|

whose denominator is a point count that approaches zero. On a half-point line a three-point
disagreement became an edge of 6.0, and ``utils.edge_tier`` applies the SAME ``0.05 / 0.02``
threshold pair to it that it applies to a WP probability -- which is how the ATS band landed
"high" on 92.64% of the 1,087 gate-holdout rows that carry a computable edge. The band is
rendered and sorted on by ``/`` and ``/betting``, so it is a published figure.

WHAT IT IS NOW
--------------
``ats_edge = ats_prediction - market_spread``, on a FIXED POINTS SCALE, with exactly ONE
surviving branch: a game with NO STORED SPREAD has NO edge (NULL), because there is no
disagreement to measure. Per D33-31 the old ``market_spread == 0`` branch -- which forced a
pick-em to an edge of 0.0 -- is GONE. It was defensible while the edge was a ratio (division by a
zero line is undefined); once the edge is a point margin it is a live defect, because a model
predicting the home team by three against a pick'em line disagrees with the market by exactly
three points.

THE SIGN INVARIANT, stated once and asserted below: ``market_spread`` is the nflverse
``spread_line``, POSITIVE when the home team is favored; ``ats_prediction`` is a predicted HOME
MARGIN per the DEF-31-01 ruling. So a POSITIVE ``ats_edge`` means the model expects the home team
to beat the line.

WHY THE HEADLINE CASE IS CONSTRUCTED
------------------------------------
0 of the 1,139 real ATS rows in ``outputs/backtest/predictions_all.csv`` carry ``|spread| <= 0.5``,
so R13's half-point example cannot be found in the data. It is BUILT, by Plan 33-02's
``tests.fixtures.season_2026.build_half_point_ats_frame``.

PIN RECONCILIATION. Five of the seven pre-existing cases in this module kept their claim and MOVED
their expected value under the redefinition; each carries an inline comment naming the old value,
the new value and the reason. No expectation was silently rewritten -- a regression pin whose
expected value is edited to match new code has stopped being evidence.

The test drives the REAL ``_load_predictions`` over fixture CSV/parquet inputs in ``tmp_path``.
Nothing under ``data/``, ``outputs/`` or ``artifacts/`` is read or written.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import pytest

from api.cache import CACHE_SCHEMA, _load_predictions
from tests.fixtures.season_2026 import (
    HALF_POINT_ATS_ROWS,
    build_half_point_ats_frame,
)
from tests.phase33_state import (
    ATS_BAND_SHARES_BEFORE,
    ATS_BAND_SHARES_BEFORE_DENOMINATOR,
    ATS_EDGE_SCALE,
)

_GAME_ID = "2025_W01_DAL@PHI"

# The expected POINTS-scale edge for every row of Plan 33-02's constructed frame, by game_id.
# Written out rather than computed as ``ats_prediction - market_spread``, because a test that
# recomputes the formula it is checking proves nothing.
_FIXTURE_EXPECTED_EDGE: dict[str, float | None] = {
    "2026_W19_HALFPOINT_HOME_FAVOURITE": -3.0,  # -3.5 - (-0.5)
    "2026_W19_HALFPOINT_HOME_UNDERDOG": 2.0,  # 2.5 - 0.5
    "2026_W19_PICKEM": -4.0,  # -4.0 - 0.0, the case D33-31 turns on
    "2026_W19_LARGE_FAVOURITE": 6.5,  # -7.0 - (-13.5)
    "2026_W19_LARGE_UNDERDOG": 3.5,  # 14.5 - 11.0
    "2026_W19_NO_STORED_SPREAD": None,  # no line, no disagreement to measure
}


def _write_inputs(tmp_path: Path, ats_prediction: float, spread: float | None) -> None:
    outputs = tmp_path / "outputs"
    silver = tmp_path / "silver"
    outputs.mkdir(parents=True, exist_ok=True)
    silver.mkdir(parents=True, exist_ok=True)

    shared = {
        "game_id": _GAME_ID,
        "season": 2025,
        "ml_home": -425.0,
        "ml_away": 330.0,
        "spread": spread,
        "total": 47.5,
        "probability_clv": 0.02,
    }
    pd.DataFrame(
        [
            {**shared, "target": "wp", "model_value": 0.60},
            {**shared, "target": "ats", "model_spread": ats_prediction},
            {**shared, "target": "ou", "model_total": 44.0},
        ]
    ).to_csv(outputs / "predictions_all.csv", index=False)

    pd.DataFrame(
        [
            {
                "game_id": _GAME_ID,
                "season": 2025,
                "week": 1,
                "home_team": "PHI",
                "away_team": "DAL",
                "home_score": None,
                "away_score": None,
                "kickoff_et": pd.Timestamp("2025-09-04 20:20:00", tz="UTC"),
            }
        ]
    ).to_parquet(silver / "games.parquet")


def _ats_edge(tmp_path: Path, ats_prediction: float, spread: float | None) -> float:
    _write_inputs(tmp_path, ats_prediction, spread)
    conn = duckdb.connect(":memory:")
    try:
        for statement in CACHE_SCHEMA.strip().split(";"):
            stmt = statement.strip()
            if stmt:
                conn.execute(stmt)
        _load_predictions(conn, tmp_path / "outputs", tmp_path / "silver")
        row = conn.execute(
            "SELECT ats_edge, ats_confidence FROM predictions"
        ).fetchone()
    finally:
        conn.close()
    assert row is not None
    return row[0]


# ---------------------------------------------------------------------------
# The pre-existing pins, reconciled case by case.
# ---------------------------------------------------------------------------


def test_a_model_that_agrees_with_the_market_has_no_cached_ats_edge(
    tmp_path: Path,
) -> None:
    """The sharpest case. UNCHANGED by the redefinition: 0.0 under both the ratio and the margin.

    Under the original DEF-31-03 defect this returned +2.0 and was banded "high" on a game of
    perfect agreement.
    """
    assert _ats_edge(tmp_path, ats_prediction=8.5, spread=8.5) == pytest.approx(0.0)


def test_a_model_that_agrees_with_a_road_favourite_line_has_no_cached_ats_edge(
    tmp_path: Path,
) -> None:
    """The same claim with the sign flipped. UNCHANGED: 0.0 under both definitions."""
    assert _ats_edge(tmp_path, ats_prediction=-6.0, spread=-6.0) == pytest.approx(0.0)


def test_a_more_bullish_model_has_a_positive_cached_edge(tmp_path: Path) -> None:
    """Model home by 12 against an 8.5 line: a +3.5-point disagreement toward home.

    MOVED PIN. Old expected value ``3.5 / 8.5 = 0.411765``; new expected value ``3.5``. REASON:
    R13 / D33-05 redefines the edge on a fixed POINTS scale, so the value is the disagreement
    itself rather than the disagreement divided by the line. The CLAIM -- a more bullish model
    carries a positive edge -- is unchanged, which is why the pin moved rather than being deleted.
    """
    assert _ats_edge(tmp_path, ats_prediction=12.0, spread=8.5) == pytest.approx(3.5)


def test_a_more_bearish_model_has_a_negative_cached_edge(tmp_path: Path) -> None:
    """The inversion, stated: the original DEF-31-03 code showed a POSITIVE home edge here.

    MOVED PIN. Old expected value ``-4.5 / 8.5 = -0.529412``; new expected value ``-4.5``.
    REASON: the same points-scale redefinition. The sign, which is the thing this pin was written
    to protect, is unchanged.
    """
    assert _ats_edge(tmp_path, ats_prediction=4.0, spread=8.5) == pytest.approx(-4.5)


def test_a_pick_em_line_carries_the_full_point_disagreement(tmp_path: Path) -> None:
    """D33-31, the owner ruling this plan turns on.

    MOVED PIN, and the only one whose CLAIM moved rather than just its value. Old expected value
    ``0.0`` (and the test was named ``..._yields_a_zero_edge_rather_than_a_division_by_zero``);
    new expected value ``3.0``. REASON: the forced zero was correct while the edge was a RATIO --
    dividing by a zero line is undefined -- but this plan makes the edge a POINT MARGIN, and a
    model predicting the home team by three against a pick'em line disagrees with the market by
    exactly three points. Keeping the branch would have shipped a points-scale edge that still
    reported zero disagreement on every pick'em game. The owner ruled it out; it is not an
    executor's reinterpretation.
    """
    assert _ats_edge(tmp_path, ats_prediction=3.0, spread=0.0) == pytest.approx(3.0)


def test_a_pick_em_line_carries_a_negative_disagreement_too(tmp_path: Path) -> None:
    """The second half of the pick-em claim, because a forced zero passes a non-negative check.

    A single ``!= 0`` assertion on a positive prediction could be satisfied by a sign bug. This
    row predicts the AWAY side against the same pick'em line and expects the negative margin.
    """
    assert _ats_edge(tmp_path, ats_prediction=-2.5, spread=0.0) == pytest.approx(-2.5)


def test_a_pick_em_with_a_zero_prediction_is_zero_by_arithmetic_not_by_branch(
    tmp_path: Path,
) -> None:
    """Zero is still the answer here -- but it is now the SUBTRACTION's answer, not a branch's."""
    assert _ats_edge(tmp_path, ats_prediction=0.0, spread=0.0) == pytest.approx(0.0)


def test_a_half_point_line_is_not_multiplied_by_six(tmp_path: Path) -> None:
    """R13's headline case, on a CONSTRUCTED row: 0 of 1,139 real rows carry ``|spread| <= 0.5``.

    MOVED PIN. Old expected value ``1.5 / 0.5 = 3.0``; new expected value ``1.5``. REASON: the
    old pin asserted the ratio's "true denominator" after the ``.clip(lower=0.5)`` floor was
    removed -- correct for its generation, and exactly the multiplication R13 names. A 1.5-point
    disagreement is a 1.5-point edge whatever the line is.
    """
    assert _ats_edge(tmp_path, ats_prediction=2.0, spread=0.5) == pytest.approx(1.5)


def test_an_absent_market_line_leaves_the_cached_edge_absent(tmp_path: Path) -> None:
    """UNCHANGED pin, and the ONLY surviving special branch.

    No line, no disagreement to measure. An edge of 0.0 here would be a made-up number, and it
    must stay DISTINGUISHABLE from the pick-em case above, which is a real line at zero.
    """
    assert _ats_edge(tmp_path, ats_prediction=12.0, spread=None) is None


# ---------------------------------------------------------------------------
# What the redefinition adds.
# ---------------------------------------------------------------------------


def test_the_edge_no_longer_depends_on_the_magnitude_of_the_line(
    tmp_path: Path,
) -> None:
    """THE R13 claim as one assertion: the same disagreement scores the same edge on any line.

    A 3.0-point disagreement against a 0.5-point line and against a 7.0-point line. Under the
    ratio these were 6.0 and 0.4285714 -- a fourteen-fold spread for identical disagreement, and
    the 6.0 is the "six times" in R13's headline.
    """
    half_point = _ats_edge(tmp_path, ats_prediction=3.5, spread=0.5)
    seven_point = _ats_edge(tmp_path, ats_prediction=10.0, spread=7.0)

    assert half_point == pytest.approx(3.0)
    assert seven_point == pytest.approx(3.0)
    assert half_point == pytest.approx(seven_point)


def test_a_positive_cached_edge_means_the_model_expects_home_to_beat_the_line(
    tmp_path: Path,
) -> None:
    """The SIGN INVARIANT, asserted on a case whose sign is recorded in the assertion itself.

    ``market_spread`` is the nflverse ``spread_line``, POSITIVE when the home team is favored;
    ``ats_prediction`` is a predicted HOME MARGIN per DEF-31-01. The market has the home team by
    3; the model has them by 7; so the model expects home to BEAT the line and the edge is +4.0.
    """
    assert _ats_edge(tmp_path, ats_prediction=7.0, spread=3.0) == pytest.approx(4.0)

    # And the mirror, so a global sign flip cannot pass the assertion above.
    assert _ats_edge(tmp_path, ats_prediction=1.0, spread=3.0) == pytest.approx(-2.0)


@pytest.mark.parametrize(
    "row",
    HALF_POINT_ATS_ROWS,
    ids=[str(row["game_id"]) for row in HALF_POINT_ATS_ROWS],
)
def test_every_constructed_case_reaches_the_cache_with_its_expected_edge(
    tmp_path: Path,
    row: dict[str, object],
) -> None:
    """All five shapes -- positive, negative, half-point, zero and missing -- plus the two sizes.

    Driven from Plan 33-02's committed fixture so this module and
    ``tests/unit/test_current_week_ats_edge.py`` assert the same constructed rows rather than each
    inventing their own.
    """
    game_id = str(row["game_id"])
    expected = _FIXTURE_EXPECTED_EDGE[game_id]
    spread = row["market_spread"]
    actual = _ats_edge(
        tmp_path,
        ats_prediction=float(row["ats_prediction"]),  # type: ignore[arg-type]
        spread=None if spread is None else float(spread),  # type: ignore[arg-type]
    )

    if expected is None:
        assert actual is None, (
            f"{game_id} has no stored spread, so it must carry NO edge rather than an edge of "
            f"zero; got {actual!r}"
        )
    else:
        assert actual == pytest.approx(expected), f"{game_id}: {row['case']}"


def test_the_constructed_frame_actually_carries_a_half_point_line() -> None:
    """Non-vacuity: the fixture the headline case rests on must contain the headline shape."""
    frame = build_half_point_ats_frame()
    assert int((frame["market_spread"].abs() == 0.5).sum()) >= 2
    assert int((frame["market_spread"] == 0.0).sum()) >= 1
    assert int(frame["market_spread"].isna().sum()) >= 1


# ---------------------------------------------------------------------------
# The recorded scale and the "before" half of the label-movement table.
# ---------------------------------------------------------------------------


def test_the_recorded_edge_scale_is_points() -> None:
    """Plan 33-16 freezes ATS thresholds against THIS edge; the unit has to be on the record."""
    assert ATS_EDGE_SCALE == "points"


def test_the_pre_repair_band_shares_are_recorded_for_all_three_targets() -> None:
    """The "before" half of the label-movement table Plan 33-16 must publish.

    Re-derived in Plan 33-10 Task 1 rather than copied forward. The ATS row is the finding: 92.64
    per cent of the population banded "high" under the ratio.
    """
    assert set(ATS_BAND_SHARES_BEFORE) == {"wp", "ats", "ou"}
    for target, shares in ATS_BAND_SHARES_BEFORE.items():
        assert set(shares) == {"low", "medium", "high"}, target
        assert sum(shares.values()) == pytest.approx(1.0, abs=5e-4), target

    assert ATS_BAND_SHARES_BEFORE["ats"]["high"] > 0.9
    assert ATS_BAND_SHARES_BEFORE_DENOMINATOR == 1087


# ---------------------------------------------------------------------------
# The two copies.
# ---------------------------------------------------------------------------


def test_the_cache_and_the_current_week_csv_agree_on_the_same_row(
    tmp_path: Path,
) -> None:
    """THE drift finding, as one assertion: the two copies of the formula must not disagree.

    MOVED PIN (value only). Old expected value ``-4.5 / 8.5 = -0.529412`` on both sides; new
    expected value ``-4.5`` on both sides. REASON: the points-scale redefinition lands in BOTH
    copies in this plan (D33-22, D33-31). The claim -- that two artifacts published from one
    codebase must not carry different ATS edges for the same game -- is untouched, and the fuller
    cross-copy table lives in ``tests/unit/test_current_week_ats_edge.py``.
    """
    from scripts.generate_current_week_predictions import compute_edges

    predictions = pd.DataFrame(
        [
            {
                "game_id": _GAME_ID,
                "wp_prob": 0.60,
                "ats_prediction": 4.0,
                "ou_prediction": 44.0,
            }
        ]
    )
    market = pd.DataFrame(
        [
            {
                "game_id": _GAME_ID,
                "spread": 8.5,
                "total": 47.5,
                "ml_home": -425.0,
                "ml_away": 330.0,
            }
        ]
    )
    from_csv = float(compute_edges(predictions, market).loc[0, "ats_edge"])
    from_cache = _ats_edge(tmp_path, ats_prediction=4.0, spread=8.5)

    assert from_cache == pytest.approx(-4.5)
    assert from_cache == pytest.approx(from_csv), (
        f"the cache says {from_cache} and the current-week CSV says {from_csv} for the same "
        "game. One copy of a duplicated formula was repaired and the other was not."
    )
