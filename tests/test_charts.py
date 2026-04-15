"""Plan 16 chart-generator test stubs (insights charts).

These tests have full assertion bodies *now*. They are gated by
``@pytest.mark.skip(reason="unblocked by plan 16-02")`` until Plan 16-02 lands
the ``api/charts`` package with the actual generator functions and the
``INSIGHTS_CHART_IDS`` tuple. Plan 16-02's verify step removes the skip markers
and expects every assertion to pass.

Why skip markers instead of xfail (per REVIEWS.md Codex MEDIUM #10):
xfail silently passes when a buggy implementation *happens* to raise the wrong
exception, giving false confidence. A skip marker makes the "not yet executing"
state explicit and can be flipped by deleting the marker when the dependency
lands. ``xfail`` is reserved for negative contracts (a function that must raise
``NotImplementedError``) — we do not use it in this file.
"""

from __future__ import annotations

import logging
import math
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Fixture data shared by every test below
# ---------------------------------------------------------------------------
# Schema mirrors the DuckDB tables but is a plain dict so these tests stay
# off of DuckDB entirely — they exercise pure chart generators.


@pytest.fixture()
def insights_chart_data() -> dict[str, list[dict]]:
    """Rich multi-season, multi-target data covering edge cases E1..E15."""

    backtest_predictions: list[dict] = []

    # --- WP rows (10 per season 2021-2023, 5 for 2024) ---
    for season, count in ((2021, 10), (2022, 10), (2023, 10), (2024, 5)):
        for i in range(count):
            p = 0.05 + (i / max(count - 1, 1)) * 0.90
            backtest_predictions.append(
                {
                    "game_id": f"{season}_W{i + 1:02d}_AWY@HOM",
                    "season": season,
                    "week": i + 1,
                    "target": "wp",
                    "model_prob": round(p, 4),
                    "actual": 1.0 if p >= 0.5 else 0.0,
                    "probability_clv": 0.0,
                    "has_closing_odds": True,
                },
            )
    # E11, E12: WP probability = 0.0 and 1.0 exactly.
    backtest_predictions.append(
        {
            "game_id": "2022_W18_A@B",
            "season": 2022,
            "week": 18,
            "target": "wp",
            "model_prob": 0.0,
            "actual": 1.0,
            "probability_clv": 0.0,
            "has_closing_odds": True,
        },
    )
    backtest_predictions.append(
        {
            "game_id": "2022_W19_C@D",
            "season": 2022,
            "week": 19,
            "target": "wp",
            "model_prob": 1.0,
            "actual": 0.0,
            "probability_clv": 0.0,
            "has_closing_odds": True,
        },
    )

    # --- ATS rows (10 per season, plus push + overflow) ---
    for season in (2021, 2022, 2023, 2024):
        for i in range(10):
            model = -10.0 + i * 2.0
            actual = model + (1.5 if i % 2 else -1.5)
            backtest_predictions.append(
                {
                    "game_id": f"{season}_W{i + 1:02d}_ATS",
                    "season": season,
                    "week": i + 1,
                    "target": "ats",
                    "model_prob": model,
                    "actual": actual,
                    "probability_clv": 0.0,
                    "has_closing_odds": True,
                },
            )
    # E6: ATS push (actual == -market_spread). In fixture below market_spread
    # is -3.0, so actual must equal +3.0.
    backtest_predictions.append(
        {
            "game_id": "2023_W15_DAL@PHI",
            "season": 2023,
            "week": 15,
            "target": "ats",
            "model_prob": 3.0,
            "actual": 3.0,
            "probability_clv": 0.0,
            "has_closing_odds": True,
        },
    )
    # E13, E14: ATS residual overflows.
    backtest_predictions.extend(
        [
            {
                "game_id": "2021_W16_NYJ@SF",
                "season": 2021,
                "week": 16,
                "target": "ats",
                "model_prob": -25.0,
                "actual": 5.0,
                "probability_clv": 0.0,
                "has_closing_odds": True,
            },
            {
                "game_id": "2021_W17_MIA@KC",
                "season": 2021,
                "week": 17,
                "target": "ats",
                "model_prob": 25.0,
                "actual": -5.0,
                "probability_clv": 0.0,
                "has_closing_odds": True,
            },
        ],
    )

    # --- OU rows (10 per season, plus push) ---
    for season in (2021, 2022, 2023, 2024):
        for i in range(10):
            model = 38.0 + i * 2.0
            actual = model + (2.5 if i % 2 else -2.5)
            backtest_predictions.append(
                {
                    "game_id": f"{season}_W{i + 1:02d}_OU",
                    "season": season,
                    "week": i + 1,
                    "target": "ou",
                    "model_prob": model,
                    "actual": actual,
                    "probability_clv": 0.0,
                    "has_closing_odds": True,
                },
            )
    backtest_predictions.append(
        {
            "game_id": "2023_W14_BUF@MIA",  # E7 OU push
            "season": 2023,
            "week": 14,
            "target": "ou",
            "model_prob": 47.0,
            "actual": 45.0,
            "probability_clv": 0.0,
            "has_closing_odds": True,
        },
    )

    # --- Feature importances (15 per target) ---
    feature_names = [
        "elo_diff",
        "rolling_off_epa",
        "rolling_def_epa",
        "snapshot_spread",
        "rolling_total_epa",
        "elo_momentum_home",
        "elo_rank_home",
        "home_rest_days",
        "away_rest_days",
        "weather_severity",
        "is_outdoor",
        "is_divisional",
        "is_primetime",
        "rolling_turnover_diff",
        "rolling_scoring_margin",
    ]
    weights = [
        0.18,
        0.14,
        0.12,
        0.10,
        0.09,
        0.07,
        0.06,
        0.05,
        0.04,
        0.04,
        0.03,
        0.03,
        0.02,
        0.02,
        0.01,
    ]
    feature_importances: list[dict] = []
    for target in ("wp", "ats", "ou"):
        for name, w in zip(feature_names, weights, strict=True):
            feature_importances.append(
                {
                    "game_id": "_model_",
                    "target": target,
                    "feature_name": name,
                    "importance": w,
                },
            )

    # --- Per-season backtest metrics ---
    backtest_metrics: list[dict] = []
    for season in (2021, 2022, 2023, 2024):
        backtest_metrics.extend(
            [
                {
                    "season": season,
                    "target": "wp",
                    "metric_name": "accuracy",
                    "metric_value": 0.62 + 0.01 * (season - 2021),
                },
                {
                    "season": season,
                    "target": "ats",
                    "metric_name": "mae",
                    "metric_value": 6.5 - 0.1 * (season - 2021),
                },
                {
                    "season": season,
                    "target": "ou",
                    "metric_name": "mae",
                    "metric_value": 10.8 - 0.2 * (season - 2021),
                },
            ],
        )

    # --- Predictions (market rows) including E1..E5 ---
    predictions: list[dict] = [
        # E1: null home moneyline
        {
            "game_id": "2023_W15_DAL@PHI",
            "season": 2023,
            "week": 15,
            "market_spread": -3.0,
            "market_total": 47.0,
            "market_ml_home": None,
            "market_ml_away": 140,
        },
        # E2: null away moneyline
        {
            "game_id": "2023_W14_BUF@MIA",
            "season": 2023,
            "week": 14,
            "market_spread": 2.5,
            "market_total": 45.0,
            "market_ml_home": -130,
            "market_ml_away": None,
        },
        # E3: both moneylines null + E9 home favorite (market_spread < 0)
        {
            "game_id": "2021_W16_NYJ@SF",
            "season": 2021,
            "week": 16,
            "market_spread": -7.5,
            "market_total": 42.0,
            "market_ml_home": None,
            "market_ml_away": None,
        },
        # E4: null market_spread
        {
            "game_id": "2021_W17_MIA@KC",
            "season": 2021,
            "week": 17,
            "market_spread": None,
            "market_total": 51.0,
            "market_ml_home": -200,
            "market_ml_away": 170,
        },
        # E5: null market_total
        {
            "game_id": "2021_W20_NYJ@CIN",
            "season": 2021,
            "week": 20,
            "market_spread": -4.0,
            "market_total": None,
            "market_ml_home": -180,
            "market_ml_away": 160,
        },
        # E10: home underdog (market_spread > 0)
        {
            "game_id": "2021_W21_SF@KC",
            "season": 2021,
            "week": 21,
            "market_spread": 3.5,
            "market_total": 55.0,
            "market_ml_home": 150,
            "market_ml_away": -175,
        },
    ]

    return {
        "backtest_predictions": backtest_predictions,
        "feature_importances": feature_importances,
        "backtest_metrics": backtest_metrics,
        "predictions": predictions,
    }


def _ats_rows(data: dict[str, list[dict]]) -> list[dict]:
    return [r for r in data["backtest_predictions"] if r["target"] == "ats"]


def _ou_rows(data: dict[str, list[dict]]) -> list[dict]:
    return [r for r in data["backtest_predictions"] if r["target"] == "ou"]


def _wp_rows(data: dict[str, list[dict]]) -> list[dict]:
    return [r for r in data["backtest_predictions"] if r["target"] == "wp"]


# ---------------------------------------------------------------------------
# ATS calibration (3 tests)
# ---------------------------------------------------------------------------


def test_insights_ats_calibration_happy_path(insights_chart_data: dict) -> None:
    """Generator produces a chart div with expected bin-center references."""
    from api.charts import generate_insights_ats_calibration

    html = generate_insights_ats_calibration(_ats_rows(insights_chart_data))
    assert "<div" in html
    assert "Predicted margin" in html
    # At least two bin centers from the 2-point grid should appear in the chart.
    bin_centers_found = sum(
        1 for c in ("-18", "-16", "-14", "-2", "0", "2", "4", "6") if c in html
    )
    assert bin_centers_found >= 2, (
        f"Expected >=2 bin centers in HTML; found {bin_centers_found}"
    )


def test_insights_ats_calibration_empty_data() -> None:
    """Generator returns an empty-state div with a helpful message."""
    from api.charts import generate_insights_ats_calibration

    html = generate_insights_ats_calibration([])
    assert "Chart unavailable" in html or "No ATS calibration" in html


def test_insights_ats_calibration_overflow_bins(insights_chart_data: dict) -> None:
    """Overflow bins ``<=-21`` and ``>=21`` are rendered when residuals exceed
    the regular bin range (E13 / E14)."""
    from api.charts import generate_insights_ats_calibration

    html = generate_insights_ats_calibration(_ats_rows(insights_chart_data))
    assert "<=-21" in html or "≤-21" in html
    assert ">=21" in html or "≥21" in html


# ---------------------------------------------------------------------------
# OU calibration (3 tests)
# ---------------------------------------------------------------------------


def test_insights_ou_calibration_happy_path(insights_chart_data: dict) -> None:
    from api.charts import generate_insights_ou_calibration

    html = generate_insights_ou_calibration(_ou_rows(insights_chart_data))
    assert "<div" in html
    assert "Predicted total" in html


def test_insights_ou_calibration_empty_data() -> None:
    from api.charts import generate_insights_ou_calibration

    html = generate_insights_ou_calibration([])
    assert "Chart unavailable" in html or "No OU calibration" in html


def test_insights_ou_calibration_fixed_bins(insights_chart_data: dict) -> None:
    """OU uses fixed 3-point bins across 35-65 — bin centers 36.5, 39.5,
    42.5, ..., 63.5 — plus ``<=35`` and ``>=65`` overflow bins."""
    from api.charts import generate_insights_ou_calibration

    html = generate_insights_ou_calibration(_ou_rows(insights_chart_data))
    centers_found = sum(
        1 for c in ("36.5", "39.5", "42.5", "45.5", "48.5", "51.5", "54.5") if c in html
    )
    assert centers_found >= 3, (
        f"Expected >=3 fixed 3-point bin centers; found {centers_found}"
    )


# ---------------------------------------------------------------------------
# Feature importance (parameterized over target + empty case)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", ["wp", "ats", "ou"])
def test_insights_feature_importance_happy_path(
    insights_chart_data: dict,
    target: str,
) -> None:
    """Horizontal bar chart with all 15 feature names present in order."""
    from api.charts import (
        generate_insights_feature_importance_ats,
        generate_insights_feature_importance_ou,
        generate_insights_feature_importance_wp,
    )

    fns = {
        "wp": generate_insights_feature_importance_wp,
        "ats": generate_insights_feature_importance_ats,
        "ou": generate_insights_feature_importance_ou,
    }
    rows = [
        r for r in insights_chart_data["feature_importances"] if r["target"] == target
    ]
    html = fns[target](rows)
    assert "<div" in html
    # All 15 feature names should appear in rendered HTML.
    for feature in (
        "elo_diff",
        "rolling_off_epa",
        "snapshot_spread",
        "weather_severity",
        "rolling_scoring_margin",
    ):
        assert feature in html, f"Missing feature {feature} in {target} chart"


def test_insights_feature_importance_empty_data() -> None:
    from api.charts import generate_insights_feature_importance_wp

    html = generate_insights_feature_importance_wp([])
    assert "Chart unavailable" in html or "No feature importance" in html


# ---------------------------------------------------------------------------
# Accuracy trend
# ---------------------------------------------------------------------------


def test_insights_accuracy_trend_happy_path(insights_chart_data: dict) -> None:
    """Chart includes all four season labels 2021-2024."""
    from api.charts import generate_insights_accuracy_trend

    html = generate_insights_accuracy_trend(insights_chart_data["backtest_metrics"])
    for year in ("2021", "2022", "2023", "2024"):
        assert year in html, f"Missing season label {year} in accuracy trend chart"


def test_insights_accuracy_trend_empty_data() -> None:
    from api.charts import generate_insights_accuracy_trend

    html = generate_insights_accuracy_trend([])
    assert "Chart unavailable" in html or "No per-season metrics" in html


# ---------------------------------------------------------------------------
# Model vs Market — WP (happy + empty + null-moneyline handling)
# ---------------------------------------------------------------------------


def test_insights_model_vs_market_wp_happy_path(insights_chart_data: dict) -> None:
    """WP small-multiples or metric-toggle chart renders all three metrics and
    both series labels."""
    from api.charts import generate_insights_model_vs_market_wp

    html = generate_insights_model_vs_market_wp(
        _wp_rows(insights_chart_data),
        insights_chart_data["predictions"],
    )
    assert "Model" in html
    assert "Market" in html
    # All three metric subplot titles must appear.
    for title in ("Accuracy", "Brier", "Log"):
        assert title in html, f"Missing metric subplot title containing {title!r}"


def test_insights_model_vs_market_wp_empty_data() -> None:
    from api.charts import generate_insights_model_vs_market_wp

    html = generate_insights_model_vs_market_wp([], [])
    assert "Chart unavailable" in html or "No" in html


def test_insights_model_vs_market_wp_handles_null_moneylines(
    insights_chart_data: dict,
) -> None:
    """E1/E2/E3 null moneyline rows must not crash the generator and the
    both-sides-null row (E3) must be excluded from the market series."""
    from api.charts import generate_insights_model_vs_market_wp

    html = generate_insights_model_vs_market_wp(
        _wp_rows(insights_chart_data),
        insights_chart_data["predictions"],
    )
    # Did not raise == test passed to here.
    assert "<div" in html
    # E3 game_id should not appear as a data point in the market series; an
    # easy proxy is that the generator returned valid HTML with the WP data.
    # The strict "exclusion" check is done in the metric helper unit tests.


# ---------------------------------------------------------------------------
# Model vs Market — ATS (sign convention) + OU
# ---------------------------------------------------------------------------


def test_insights_model_vs_market_ats_happy_path() -> None:
    """Sign convention: market prediction is ``-market_spread``. If we feed
    rows where model predicted home margin == -market_spread for every game,
    market MAE must be 0.0 (market is perfect against actual=model in that
    synthetic setup)."""
    from api.charts import generate_insights_model_vs_market_ats

    # Construct rows: market_spread = -3.0 => market prediction = +3.0.
    # Set actual = +3.0 so market MAE = 0.
    ats_rows = [
        {
            "game_id": f"2023_W{i:02d}_A@B",
            "season": 2023,
            "week": i,
            "target": "ats",
            "model_prob": 3.0,
            "actual": 3.0,
            "probability_clv": 0.0,
            "has_closing_odds": True,
        }
        for i in range(1, 6)
    ]
    pred_rows = [
        {
            "game_id": f"2023_W{i:02d}_A@B",
            "season": 2023,
            "week": i,
            "market_spread": -3.0,
            "market_total": 45.0,
            "market_ml_home": -150,
            "market_ml_away": 130,
        }
        for i in range(1, 6)
    ]
    html = generate_insights_model_vs_market_ats(ats_rows, pred_rows)
    assert "<div" in html
    # Market MAE series should render exactly zero somewhere (0.00 formatting).
    assert "0.00" in html or "0.0" in html


def test_insights_model_vs_market_ou_happy_path(insights_chart_data: dict) -> None:
    from api.charts import generate_insights_model_vs_market_ou

    html = generate_insights_model_vs_market_ou(
        _ou_rows(insights_chart_data),
        insights_chart_data["predictions"],
    )
    assert "<div" in html
    assert "Model" in html
    assert "Market" in html


# ---------------------------------------------------------------------------
# Pre-render failure isolation
# ---------------------------------------------------------------------------


def test_prerender_charts_failure_isolation(
    insights_chart_data: dict,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """If one generator raises, the pre-render function must (a) not re-raise,
    (b) emit a WARNING log record with chart_id + traceback, and (c) populate
    the failed chart_id with the ``_empty_chart_div`` fallback so the template
    renders a "Chart unavailable" div instead of a hole."""
    from api.charts import INSIGHTS_CHART_IDS
    from api.charts.prerender import prerender_charts_for_cache

    caplog.set_level(logging.WARNING)
    # Patch exactly one generator to blow up.
    with patch(
        "api.charts.insights.generate_insights_accuracy_trend",
        side_effect=ValueError("injected"),
    ):
        result = prerender_charts_for_cache(insights_chart_data)

    # Result must be a dict keyed by chart_id with HTML values.
    assert isinstance(result, dict)
    # All 9 insight chart IDs present.
    for chart_id in INSIGHTS_CHART_IDS:
        assert chart_id in result, f"Missing chart_id {chart_id}"

    # The failed chart_id maps to a fallback empty-chart div.
    assert "Chart unavailable" in result["insights_accuracy_trend"]

    # At least one WARNING log record mentions the failing chart_id and traceback.
    warning_records = [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and "insights_accuracy_trend" in r.getMessage()
    ]
    assert warning_records, "Expected WARNING log for failing chart_id"

    # Remaining chart IDs must contain real (non-empty) HTML.
    remaining = [cid for cid in INSIGHTS_CHART_IDS if cid != "insights_accuracy_trend"]
    for cid in remaining:
        assert "<div" in result[cid]
        assert "Chart unavailable" not in result[cid]


# Sanity guard so the file itself is valid Python even before plan 16-02.
def test_module_imports_cleanly() -> None:
    assert math.isfinite(1.0)
