"""Colour and layout rules every dashboard chart must follow under the dark theme (Task 18).

Each generator is fed small, non-empty inputs, so every one draws a real figure. The data
and layout are then read back out of the Plotly.newPlot(...) call in the HTML -- exactly
what the browser receives from the cache.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from api.charts import theme

# ---------------------------------------------------------------------------
# Small deterministic inputs (shapes mirror the cache tables)
# ---------------------------------------------------------------------------

_GAME_IDS = [f"2023_W{i:02d}_A@B" for i in range(1, 21)]

_WP_ROWS: list[dict[str, Any]] = [
    {
        "game_id": gid,
        "season": 2023,
        "week": i + 1,
        "target": "wp",
        "model_prob": round(0.05 + i * 0.045, 3),
        "actual": 1.0 if i % 3 else 0.0,
        "probability_clv": 0.01 * ((i % 5) - 2),
        "has_closing_odds": True,
    }
    for i, gid in enumerate(_GAME_IDS)
]

_ATS_ROWS: list[dict[str, Any]] = [
    {
        "game_id": gid,
        "season": 2023,
        "week": i + 1,
        "target": "ats",
        "model_prob": -10.0 + i,
        "actual": -10.0 + i + (1.5 if i % 2 else -1.5),
        "probability_clv": 0.0,
        "has_closing_odds": True,
    }
    for i, gid in enumerate(_GAME_IDS)
] + [
    {
        "game_id": "2023_W21_C@D",
        "season": 2023,
        "week": 21,
        "target": "ats",
        "model_prob": -25.0,
        "actual": 5.0,
        "probability_clv": 0.0,
        "has_closing_odds": True,
    },
    {
        "game_id": "2023_W22_E@F",
        "season": 2023,
        "week": 22,
        "target": "ats",
        "model_prob": 25.0,
        "actual": -5.0,
        "probability_clv": 0.0,
        "has_closing_odds": True,
    },
]

_OU_ROWS: list[dict[str, Any]] = [
    {
        "game_id": gid,
        "season": 2023,
        "week": i + 1,
        "target": "ou",
        "model_prob": 38.0 + i,
        "actual": 38.0 + i + (2.5 if i % 2 else -2.5),
        "probability_clv": 0.0,
        "has_closing_odds": True,
    }
    for i, gid in enumerate(_GAME_IDS)
]

_MARKET: list[dict[str, Any]] = [
    {
        "game_id": gid,
        "season": 2023,
        "week": i + 1,
        "market_spread": -3.0 + (i % 4),
        "market_total": 44.5 + (i % 3),
        "market_ml_home": -150 if i % 2 else 130,
        "market_ml_away": 130 if i % 2 else -150,
    }
    for i, gid in enumerate(_GAME_IDS)
]

_METRICS: list[dict[str, Any]] = [
    row
    for season in (2021, 2022, 2023)
    for row in (
        {
            "season": season,
            "target": "wp",
            "metric_name": "accuracy",
            "metric_value": 0.62 + 0.01 * (season - 2021),
        },
        {
            "season": season,
            "target": "wp",
            "metric_name": "brier_score",
            "metric_value": 0.22 - 0.002 * (season - 2021),
        },
        {
            "season": season,
            "target": "ats",
            "metric_name": "mae",
            "metric_value": 10.4 - 0.1 * (season - 2021),
        },
        {
            "season": season,
            "target": "ou",
            "metric_name": "mae",
            "metric_value": 10.8 - 0.2 * (season - 2021),
        },
    )
]

_IMPORTANCES: list[dict[str, Any]] = [
    {"game_id": "_model_", "target": target, "feature_name": name, "importance": weight}
    for target in ("wp", "ats", "ou")
    for name, weight in (("elo_diff", 0.4), ("hfa_used", 0.3), ("is_divisional", 0.2))
]

_EQUITY: list[dict[str, Any]] = [
    {"strategy": strategy, "bet_index": i, "bankroll": 10000.0 + step * i}
    for strategy, step in (("flat_stake", 40.0), ("kelly", -25.0))
    for i in range(5)
]


def _bet(
    game_id: str, target: str, edge: float, outcome: bool | None, payout: float
) -> dict[str, Any]:
    return {
        "game_id": game_id,
        "season": int(game_id[:4]),
        "week": int(game_id[6:8]),
        "target": target,
        "bet_side": "home",
        "model_value": 0.5,
        "market_value": 0.5,
        "edge": edge,
        "slipped_line": None if target == "wp" else -3.0,
        "odds": -110.0,
        "flat_stake": 100.0,
        "kelly_stake": 80.0,
        "outcome": outcome,
        "payout_flat": payout,
        "payout_kelly": payout * 0.8,
    }


_BETS: list[dict[str, Any]] = [
    _bet("2022_W01_A@B", "wp", 0.07, True, 66.7),
    _bet("2022_W02_C@D", "wp", 0.06, False, -100.0),
    _bet("2023_W03_E@F", "ats", 2.5, True, 90.9),
    _bet("2023_W04_G@H", "ats", 1.5, False, -100.0),
    _bet("2023_W05_I@J", "ats", 1.2, None, 0.0),
    _bet("2024_W06_K@L", "ou", 2.5, True, 90.9),
    _bet("2024_W07_M@N", "ou", 3.0, False, -100.0),
]

_SEASON_ROWS: list[dict[str, Any]] = [
    {
        "game_id": f"2024_W{wk:02d}_A@B",
        "season": 2024,
        "week": wk,
        "status": "completed",
        "home_score": 27 if wk % 2 else 17,
        "away_score": 20 if wk % 2 else 24,
        "wp_prob": 0.62 if wk % 2 else 0.41,
        "ats_prediction": -4.0 if wk % 2 else 2.0,
        "ou_prediction": 50.0 if wk % 2 else 40.0,
        "market_spread": -3.0,
        "market_total": 45.0,
    }
    for wk in range(1, 6)
]

# Charts whose series ARE realised outcomes; only these may use WIN_COLOR / LOSS_COLOR.
_OUTCOME_CHARTS = frozenset({"betting_edge_hist_ats"})

# Colours the light theme and the old palettes used. None may survive on the dark panel.
_LEGACY_LITERALS = (
    '#999"',
    '#ccc"',
    '#333"',
    '#666"',
    "#667eea",
    "#f093fb",
    "#16a34a",
    "#dc2626",
    "#95a5a6",
    "#4fd1c5",
    "#E5ECF6",
    "#E5E7EB",
    "#FFFFFF",
)


def _render_all() -> dict[str, str]:
    from api.charts import (
        generate_betting_edge_hist_ats,
        generate_betting_equity_chart,
        generate_betting_equity_mini_ats,
        generate_betting_roi_type,
        generate_dashboard_calibration_chart,
        generate_dashboard_clv_chart,
        generate_dashboard_equity_chart,
        generate_dashboard_heatmap,
        generate_insights_accuracy_trend,
        generate_insights_ats_calibration,
        generate_insights_feature_importance_wp,
        generate_insights_model_vs_market_ats,
        generate_insights_model_vs_market_ou,
        generate_insights_model_vs_market_wp,
        generate_insights_ou_calibration,
        generate_season_cumulative,
        generate_season_weekly,
    )

    return {
        "calibration": generate_dashboard_calibration_chart(_WP_ROWS),
        "clv": generate_dashboard_clv_chart(_WP_ROWS),
        "heatmap": generate_dashboard_heatmap(_METRICS),
        "equity": generate_dashboard_equity_chart(_EQUITY),
        "ats_calibration": generate_insights_ats_calibration(_ATS_ROWS),
        "ou_calibration": generate_insights_ou_calibration(_OU_ROWS),
        "feature_importance_wp": generate_insights_feature_importance_wp(_IMPORTANCES),
        "accuracy_trend": generate_insights_accuracy_trend(_METRICS),
        "mvm_wp": generate_insights_model_vs_market_wp(_WP_ROWS, _MARKET),
        "mvm_ats": generate_insights_model_vs_market_ats(_ATS_ROWS, _MARKET),
        "mvm_ou": generate_insights_model_vs_market_ou(_OU_ROWS, _MARKET),
        "betting_equity": generate_betting_equity_chart(_BETS),
        "betting_equity_mini_ats": generate_betting_equity_mini_ats(_BETS),
        "betting_roi_type": generate_betting_roi_type(_BETS),
        "betting_edge_hist_ats": generate_betting_edge_hist_ats(_BETS),
        "season_cumulative": generate_season_cumulative(_SEASON_ROWS),
        "season_weekly": generate_season_weekly(_SEASON_ROWS),
    }


@pytest.fixture(scope="module")
def charts() -> dict[str, str]:
    rendered = _render_all()
    # Every input above is non-empty, so every generator must draw a real figure. An
    # empty-state div here would make the colour checks below pass vacuously.
    empty = [name for name, html in rendered.items() if "Plotly.newPlot(" not in html]
    assert not empty, f"generators fell back to the empty state: {empty}"
    return rendered


def _plot_args(html: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return (data, layout) from the Plotly.newPlot(id, data, layout, config) call."""
    decoder = json.JSONDecoder()
    body = html[html.index("Plotly.newPlot(") + len("Plotly.newPlot(") :]
    _div_id, end = decoder.raw_decode(body, body.index('"'))
    data, end = decoder.raw_decode(body, body.index("[", end))
    layout, _ = decoder.raw_decode(body, body.index("{", end))
    return data, layout


def _layout(html: str) -> dict[str, Any]:
    return _plot_args(html)[1]


def _data(html: str) -> list[dict[str, Any]]:
    return _plot_args(html)[0]


def test_every_chart_is_transparent_on_the_panel(charts: dict[str, str]) -> None:
    for name, html in charts.items():
        layout = _layout(html)
        assert layout["paper_bgcolor"] == theme.TRANSPARENT, name
        assert layout["plot_bgcolor"] == theme.TRANSPARENT, name


def test_no_light_theme_or_legacy_palette_literal_survives(
    charts: dict[str, str],
) -> None:
    leaks = {
        name: [lit for lit in _LEGACY_LITERALS if lit.lower() in html.lower()]
        for name, html in charts.items()
    }
    assert not {name: found for name, found in leaks.items() if found}


def test_green_and_red_appear_only_on_realised_win_loss_charts(
    charts: dict[str, str],
) -> None:
    for name, html in charts.items():
        if name in _OUTCOME_CHARTS:
            assert theme.WIN_COLOR in html, name
            assert theme.LOSS_COLOR in html, name
        else:
            assert theme.WIN_COLOR not in html, (
                f"{name} paints a non-outcome series green"
            )
            assert theme.LOSS_COLOR not in html, (
                f"{name} paints a non-outcome series red"
            )


def test_market_is_dashed_muted_beside_the_solid_bet_type_colour(
    charts: dict[str, str],
) -> None:
    for name, target in (("mvm_ats", "ats"), ("mvm_ou", "ou"), ("mvm_wp", "wp")):
        traces = _data(charts[name])
        market = next(t for t in traces if t.get("name") == "Market")
        model = next(t for t in traces if t.get("name") == "Model")
        assert market["line"]["color"] == theme.MUTED, name
        assert market["line"]["dash"] == "dash", name
        assert model["line"]["color"] == theme.TARGET_COLORS[target], name
        assert "dash" not in model["line"], name


def test_reference_lines_use_the_theme_reference_colour(charts: dict[str, str]) -> None:
    for name in (
        "clv",
        "equity",
        "betting_equity",
        "betting_roi_type",
        "season_cumulative",
    ):
        shapes = _layout(charts[name]).get("shapes", [])
        assert any(s["line"]["color"] == theme.REFERENCE_LINE for s in shapes), name


def test_stacked_calibration_rows_have_room_for_titles(charts: dict[str, str]) -> None:
    for name in ("ats_calibration", "ou_calibration"):
        layout = _layout(charts[name])
        gap = layout["yaxis"]["domain"][0] - layout["yaxis2"]["domain"][1]
        assert gap >= 0.25, f"{name}: rows only {gap:.2f} apart"
        assert layout["height"] == 480, name


def test_legends_never_sit_on_a_title(charts: dict[str, str]) -> None:
    for name in (
        "equity",
        "betting_equity",
        "betting_roi_type",
        "mvm_ats",
        "season_cumulative",
    ):
        assert _layout(charts[name])["legend"]["y"] >= 1.0, name
    # The CLV legend's three long entries wrap to three rows on a 390px phone, and a top legend
    # grows upward into the title (redesign Task 19 screenshot review), so it sits underneath.
    for name in ("calibration", "accuracy_trend", "mvm_wp", "season_weekly", "clv"):
        assert _layout(charts[name])["legend"]["y"] < 0, name


def test_titles_keep_their_meaning(charts: dict[str, str]) -> None:
    assert _layout(charts["ats_calibration"])["title"]["text"] == (
        "ATS \u2014 Predicted margin vs actual"
    )
    assert "ECE = " in _layout(charts["calibration"])["title"]["text"]


def test_heatmap_is_monochrome_and_keeps_each_traces_pre_existing_direction(
    charts: dict[str, str],
) -> None:
    heatmaps = [t for t in _data(charts["heatmap"]) if t.get("type") == "heatmap"]
    # Targets render in sorted order: ats, ou, wp. The scale is reversed exactly where
    # RdYlGn_r was (the ats and ou traces) -- the pre-existing per-trace direction. This does
    # not make brighter "better" on every column: the wp trace's error columns (Brier, for
    # one) are not reversed.
    assert [t.get("reversescale", False) for t in heatmaps] == [True, True, False]
    for trace in heatmaps:
        assert trace["colorscale"] == theme.HEATMAP_SCALE


def test_a_missing_heatmap_metric_leaves_its_cell_empty() -> None:
    """A season with no value for a metric draws no cell, only its "N/A" label.

    A 0.0 stand-in drew as the BRIGHTEST cell on the reversed ATS and O/U traces and stretched
    their scale down to zero, dimming every real value beside it.
    """
    from api.charts import generate_dashboard_heatmap

    metrics = [
        m for m in _METRICS if not (m["target"] == "ats" and m["season"] == 2022)
    ]
    heatmaps = [
        t
        for t in _data(generate_dashboard_heatmap(metrics))
        if t.get("type") == "heatmap"
    ]
    ats = heatmaps[0]  # targets render in sorted order: ats, ou, wp
    assert ats["y"] == ["2021", "2022", "2023"]
    assert ats["z"][1] == [None], "the missing 2022 ATS value is not an empty cell"
    assert ats["text"][1] == ["N/A"]
    assert None not in ats["z"][0] + ats["z"][2], "a measured value went missing"


def test_edge_histogram_bars_sit_side_by_side_and_keep_their_outcome_colours(
    charts: dict[str, str],
) -> None:
    html = charts["betting_edge_hist_ats"]
    # Overlaid translucent green and red blend into an orange that reads as a third outcome.
    assert _layout(html)["barmode"] == "group"
    by_name = {t["name"]: t for t in _data(html)}
    assert by_name["Win"]["marker"]["color"] == theme.WIN_COLOR
    assert by_name["Loss"]["marker"]["color"] == theme.LOSS_COLOR
