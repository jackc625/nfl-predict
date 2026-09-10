"""WR-02: the CACHE's ATS edge is a difference, not a sum -- the copy the dashboard renders.

WHAT WAS WRONG
--------------
``api.cache._load_predictions`` computed::

    (ats_prediction - (-market_spread)) / market_spread.abs().clip(lower=0.5)

which is ``(model + market) / |market|``: a SUM, not a difference. The identical expression was
corrected in ``scripts/generate_current_week_predictions.compute_edges`` under DEF-31-03, and that
fix's own comment records that ``api/cache.py`` "derives its own ``ats_edge`` ... and never reads
this file" -- true, and exactly why the defect survived in the copy that feeds the UI. This value
drives ``ats_edge``, ``ats_confidence``, the ``/`` page's ``sort=edge`` ordering and both export
endpoints.

Under the old expression a model agreeing EXACTLY with the market scored the largest edge the
formula can produce (3.0 against a 3.0 line gave +2.0, which ``edge_tier_series`` bands "high"),
and a model that thought the home side OVERVALUED was shown a POSITIVE home edge.

The expectations below are deliberately the SAME arithmetic
``tests/unit/test_current_week_ats_edge.py`` pins for the sibling copy, so the two can no longer
silently contradict each other -- which is the finding.

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

_GAME_ID = "2025_W01_DAL@PHI"


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


def test_a_model_that_agrees_with_the_market_has_no_cached_ats_edge(
    tmp_path: Path,
) -> None:
    """The sharpest case. Old expression: +2.0, banded "high" on a game of perfect agreement."""
    assert _ats_edge(tmp_path, ats_prediction=8.5, spread=8.5) == pytest.approx(0.0)


def test_a_model_that_agrees_with_a_road_favourite_line_has_no_cached_ats_edge(
    tmp_path: Path,
) -> None:
    """The same claim with the sign flipped, so a symmetric bug cannot pass one and fail the other."""
    assert _ats_edge(tmp_path, ats_prediction=-6.0, spread=-6.0) == pytest.approx(0.0)


def test_a_more_bullish_model_has_a_positive_cached_edge(tmp_path: Path) -> None:
    """Model home by 12 against a 8.5 line: a +3.5-point disagreement toward home."""
    assert _ats_edge(tmp_path, ats_prediction=12.0, spread=8.5) == pytest.approx(
        3.5 / 8.5
    )


def test_a_more_bearish_model_has_a_negative_cached_edge(tmp_path: Path) -> None:
    """The inversion, stated: the old code showed a POSITIVE home edge for this row."""
    assert _ats_edge(tmp_path, ats_prediction=4.0, spread=8.5) == pytest.approx(
        -4.5 / 8.5
    )


def test_a_pick_em_line_yields_a_zero_edge_rather_than_a_division_by_zero(
    tmp_path: Path,
) -> None:
    """The old ``.clip(lower=0.5)`` denominator floor turned a pick-em into ``model / 0.5``."""
    assert _ats_edge(tmp_path, ats_prediction=3.0, spread=0.0) == pytest.approx(0.0)


def test_a_half_point_line_is_not_silently_rescaled(tmp_path: Path) -> None:
    """``.clip(lower=0.5)`` was a no-op at 0.5 but hid the shape; assert the true denominator."""
    assert _ats_edge(tmp_path, ats_prediction=2.0, spread=0.5) == pytest.approx(
        1.5 / 0.5
    )


def test_an_absent_market_line_leaves_the_cached_edge_absent(tmp_path: Path) -> None:
    """No line, no disagreement to measure. An edge of 0.0 here would be a made-up number."""
    assert _ats_edge(tmp_path, ats_prediction=12.0, spread=None) is None


def test_the_cache_and_the_current_week_csv_agree_on_the_same_row(
    tmp_path: Path,
) -> None:
    """THE finding, as one assertion: the two copies of the formula must not disagree.

    Two artifacts published from one codebase carried different ATS edges for the same game, and
    the one on the website was the wrong one.
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

    assert from_cache == pytest.approx(from_csv), (
        f"the cache says {from_cache} and the current-week CSV says {from_csv} for the same "
        "game. DEF-31-03 was fixed on one side of a duplicated formula and not the other."
    )
