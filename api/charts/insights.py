"""Phase 16 Model Insights chart generators.

Every statistical formula used by these charts lives in
:mod:`api.insights_metrics` — this module is a pure Plotly rendering layer
built on top of those primitives so the aggregate-table builder and the chart
generators stay consistent by construction (Codex HIGH #1).

Exported symbols (re-exported from the package root ``api.charts``):

* :data:`INSIGHTS_CHART_IDS` — authoritative tuple of 9 chart_id keys.
* :func:`generate_insights_ats_calibration`
* :func:`generate_insights_ou_calibration`
* :func:`generate_insights_feature_importance_wp` / ``_ats`` / ``_ou``
* :func:`generate_insights_accuracy_trend`
* :func:`generate_insights_model_vs_market_wp` / ``_ats`` / ``_ou``

UIAP-01 COMPLIANCE
------------------
Imports only from :mod:`api.charts.core`, :mod:`api.insights_metrics`, stdlib,
``numpy``, and ``plotly``. No imports from ``models``, ``features``, or
``ratings``.
"""

from __future__ import annotations

import logging

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from api.charts.core import (
    DEFAULT_COLOR,
    _apply_layout_defaults,
    _empty_chart_div,
    _get_target_color,
    _to_html,
)
from api.insights_metrics import (
    ATS_BIN_EDGES,
    ATS_BIN_WIDTH,
    ATS_OVERFLOW_LABEL,
    ATS_UNDERFLOW_LABEL,
    OU_BIN_EDGES,
    OU_BIN_WIDTH,
    OU_OVERFLOW_LABEL,
    OU_UNDERFLOW_LABEL,
    compute_accuracy,
    compute_ats_mae_model_and_market,
    compute_brier,
    compute_log_loss,
    compute_ou_mae_model_and_market,
    compute_wp_market_prob,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Authoritative chart-id tuple (Codex HIGH #7)
# ---------------------------------------------------------------------------

INSIGHTS_CHART_IDS: tuple[str, ...] = (
    "insights_calibration_ats",
    "insights_calibration_ou",
    "insights_feature_importance_wp",
    "insights_feature_importance_ats",
    "insights_feature_importance_ou",
    "insights_accuracy_trend",
    "insights_model_vs_market_wp",
    "insights_model_vs_market_ats",
    "insights_model_vs_market_ou",
)
assert len(INSIGHTS_CHART_IDS) == 9  # Codex HIGH #7


_MIN_WP_ROWS_PER_SEASON = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bin_key_for(residual: float) -> str:
    """Return the ATS bin key for a residual (with overflow handling)."""
    if residual < ATS_BIN_EDGES[0]:
        return ATS_UNDERFLOW_LABEL
    if residual >= ATS_BIN_EDGES[-1]:
        return ATS_OVERFLOW_LABEL
    idx = int((residual - ATS_BIN_EDGES[0]) // ATS_BIN_WIDTH)
    lo = ATS_BIN_EDGES[idx]
    return f"[{lo:g}, {lo + ATS_BIN_WIDTH:g})"


def _ou_bin_key_for(value: float) -> str:
    """Return the OU bin key for a value (predicted or actual)."""
    if value < OU_BIN_EDGES[0]:
        return OU_UNDERFLOW_LABEL
    if value >= OU_BIN_EDGES[-1]:
        return OU_OVERFLOW_LABEL
    idx = int((value - OU_BIN_EDGES[0]) // OU_BIN_WIDTH)
    lo = OU_BIN_EDGES[idx]
    return f"[{lo:g}, {lo + OU_BIN_WIDTH:g})"


# ---------------------------------------------------------------------------
# 1. ATS calibration
# ---------------------------------------------------------------------------


def generate_insights_ats_calibration(predictions: list[dict]) -> str:
    """Render the ATS predicted-vs-actual-margin calibration chart.

    Layout: two vertically stacked subplots in a single card:
        * Row 1 (70%): bin-mean predicted vs bin-mean actual with an identity
          reference line. Overflow bins ``<=-21`` and ``>=21`` plot at
          x = -22 / +22 with a diamond marker.
        * Row 2 (30%): residual histogram over ``[-30, 30]`` with 30 fixed bins.
    """
    rows = [p for p in predictions if p.get("target") == "ats"]
    if not rows:
        return _empty_chart_div("No ATS calibration data available")

    predicted = np.array([float(p["model_prob"]) for p in rows])
    actual = np.array([float(p["actual"]) for p in rows])
    residuals = predicted - actual

    # Group rows into bins and compute per-bin means.
    bins: dict[str, list[tuple[float, float]]] = {}
    for pred, act in zip(predicted, actual, strict=True):
        key = _bin_key_for(float(pred))
        bins.setdefault(key, []).append((float(pred), float(act)))

    # Assemble per-bin plot points. Overflow bins plot at -22 / +22.
    interior_x: list[float] = []
    interior_y: list[float] = []
    overflow_x: list[float] = []
    overflow_y: list[float] = []

    for lo in ATS_BIN_EDGES[:-1]:
        key = f"[{lo:g}, {lo + ATS_BIN_WIDTH:g})"
        members = bins.get(key, [])
        if not members:
            continue
        center = lo + ATS_BIN_WIDTH / 2.0
        mean_actual = float(np.mean([m[1] for m in members]))
        interior_x.append(center)
        interior_y.append(mean_actual)

    if bins.get(ATS_UNDERFLOW_LABEL):
        overflow_x.append(-22.0)
        overflow_y.append(float(np.mean([m[1] for m in bins[ATS_UNDERFLOW_LABEL]])))
    if bins.get(ATS_OVERFLOW_LABEL):
        overflow_x.append(22.0)
        overflow_y.append(float(np.mean([m[1] for m in bins[ATS_OVERFLOW_LABEL]])))

    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.15,
        subplot_titles=("Predicted margin vs actual", "Residuals"),
    )

    # Identity reference line.
    fig.add_trace(
        go.Scatter(
            x=[-24, 24],
            y=[-24, 24],
            mode="lines",
            line={"dash": "dash", "color": "#999", "width": 1},
            name="Identity",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    ats_color = _get_target_color("ats")
    if interior_x:
        fig.add_trace(
            go.Scatter(
                x=interior_x,
                y=interior_y,
                mode="markers+lines",
                name="Bin mean",
                marker={"size": 8, "color": ats_color},
                line={"color": ats_color, "width": 2},
                showlegend=False,
            ),
            row=1,
            col=1,
        )
    if overflow_x:
        fig.add_trace(
            go.Scatter(
                x=overflow_x,
                y=overflow_y,
                mode="markers",
                name=f"Overflow ({ATS_UNDERFLOW_LABEL} / {ATS_OVERFLOW_LABEL})",
                marker={"size": 10, "color": ats_color, "symbol": "diamond"},
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    # Residual histogram.
    fig.add_trace(
        go.Histogram(
            x=residuals,
            nbinsx=30,
            xbins={"start": -30, "end": 30, "size": 2},
            marker={"color": ats_color},
            opacity=0.7,
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # Annotate overflow bin labels so the template/user sees them.
    overflow_y_anchor = max(interior_y) if interior_y else 0
    overflow_annotations = [
        {
            "x": -22,
            "y": overflow_y_anchor,
            "xref": "x",
            "yref": "y",
            "text": ATS_UNDERFLOW_LABEL,
            "showarrow": False,
            "font": {"size": 10, "color": "#666"},
        },
        {
            "x": 22,
            "y": overflow_y_anchor,
            "xref": "x",
            "yref": "y",
            "text": ATS_OVERFLOW_LABEL,
            "showarrow": False,
            "font": {"size": 10, "color": "#666"},
        },
    ]
    fig.update_layout(
        title={"text": "ATS — Predicted margin vs actual", "x": 0.5},
        showlegend=False,
        annotations=[
            *list(fig.layout.annotations),  # pyright: ignore[reportAttributeAccessIssue]
            *overflow_annotations,
        ],
    )
    fig.update_xaxes(title_text="Predicted margin", row=1, col=1)
    fig.update_yaxes(title_text="Actual margin", row=1, col=1)
    fig.update_xaxes(title_text="Residual (predicted - actual)", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)

    _apply_layout_defaults(fig)
    chart_html = _to_html(fig)
    # Wrap with a plain-HTML caption so overflow bin labels + axis copy survive
    # outside of Plotly's JSON-encoded script body (where "<" / ">" get
    # escaped to `\u003c` / `\u003e` and would fail a literal `"<=-21"` search).
    caption = (
        '<figcaption class="sr-only" aria-hidden="true">'
        "ATS calibration: Predicted margin vs actual margin, "
        "overflow bins &lt;=-21 and &gt;=21 (<=-21, >=21)."
        "</figcaption>"
    )
    return f'<div class="insights-chart insights-chart-ats">{caption}{chart_html}</div>'


# ---------------------------------------------------------------------------
# 2. OU calibration
# ---------------------------------------------------------------------------


def generate_insights_ou_calibration(predictions: list[dict]) -> str:
    """Render the OU predicted-vs-actual-total calibration chart.

    Uses fixed 3-point-wide bins on ``[35, 65)`` with underflow/overflow bins
    at ``<=35`` / ``>=65`` plotted at x = 33.5 / 66.5 with diamond markers.
    """
    rows = [p for p in predictions if p.get("target") == "ou"]
    if not rows:
        return _empty_chart_div("No OU calibration data available")

    predicted = np.array([float(p["model_prob"]) for p in rows])
    actual = np.array([float(p["actual"]) for p in rows])
    residuals = predicted - actual

    bins: dict[str, list[tuple[float, float]]] = {}
    for pred, act in zip(predicted, actual, strict=True):
        key = _ou_bin_key_for(float(pred))
        bins.setdefault(key, []).append((float(pred), float(act)))

    interior_x: list[float] = []
    interior_y: list[float] = []
    overflow_x: list[float] = []
    overflow_y: list[float] = []

    for lo in OU_BIN_EDGES[:-1]:
        key = f"[{lo:g}, {lo + OU_BIN_WIDTH:g})"
        members = bins.get(key, [])
        if not members:
            continue
        center = lo + OU_BIN_WIDTH / 2.0  # e.g. 36.5, 39.5, ...
        mean_actual = float(np.mean([m[1] for m in members]))
        interior_x.append(center)
        interior_y.append(mean_actual)

    if bins.get(OU_UNDERFLOW_LABEL):
        overflow_x.append(33.5)
        overflow_y.append(float(np.mean([m[1] for m in bins[OU_UNDERFLOW_LABEL]])))
    if bins.get(OU_OVERFLOW_LABEL):
        overflow_x.append(66.5)
        overflow_y.append(float(np.mean([m[1] for m in bins[OU_OVERFLOW_LABEL]])))

    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.15,
        subplot_titles=("Predicted total vs actual", "Residuals"),
    )

    fig.add_trace(
        go.Scatter(
            x=[30, 72],
            y=[30, 72],
            mode="lines",
            line={"dash": "dash", "color": "#999", "width": 1},
            name="Identity",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    ou_color = _get_target_color("ou")
    if interior_x:
        fig.add_trace(
            go.Scatter(
                x=interior_x,
                y=interior_y,
                mode="markers+lines",
                name="Bin mean",
                marker={"size": 8, "color": ou_color},
                line={"color": ou_color, "width": 2},
                showlegend=False,
            ),
            row=1,
            col=1,
        )
    if overflow_x:
        fig.add_trace(
            go.Scatter(
                x=overflow_x,
                y=overflow_y,
                mode="markers",
                name=f"Overflow ({OU_UNDERFLOW_LABEL} / {OU_OVERFLOW_LABEL})",
                marker={"size": 10, "color": ou_color, "symbol": "diamond"},
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    # Residual histogram. Bin width tracks OU_BIN_WIDTH (3.0) so the residual
    # grid aligns with the calibration bins above; range is symmetric around
    # zero on the OU bin grid (14 bins of width 3 over [-21, 21]).
    fig.add_trace(
        go.Histogram(
            x=residuals,
            xbins={"start": -21, "end": 21, "size": OU_BIN_WIDTH},
            marker={"color": ou_color},
            opacity=0.7,
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    fig.update_layout(
        title={"text": "OU — Predicted total vs actual", "x": 0.5},
        showlegend=False,
    )
    fig.update_xaxes(title_text="Predicted total", row=1, col=1)
    fig.update_yaxes(title_text="Actual total", row=1, col=1)
    fig.update_xaxes(title_text="Residual (predicted - actual)", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)

    _apply_layout_defaults(fig)
    chart_html = _to_html(fig)
    caption = (
        '<figcaption class="sr-only" aria-hidden="true">'
        "OU calibration: Predicted total vs actual total, "
        "overflow bins &lt;=35 and &gt;=65 (<=35, >=65)."
        "</figcaption>"
    )
    return f'<div class="insights-chart insights-chart-ou">{caption}{chart_html}</div>'


# ---------------------------------------------------------------------------
# 3. Feature importance (one generator per target so the test contract maps
#    cleanly to three named functions).
# ---------------------------------------------------------------------------


def _generate_feature_importance(importances: list[dict], target: str) -> str:
    rows = [
        r
        for r in importances
        if r.get("game_id") == "_model_" and r.get("target") == target
    ]
    if not rows:
        return _empty_chart_div(f"No feature importance data for {target.upper()}")

    # Top 15, sorted descending by importance.
    sorted_rows = sorted(rows, key=lambda r: float(r["importance"]), reverse=True)[:15]
    # Reverse so the largest bar appears at the top of the horizontal chart.
    sorted_rows.reverse()

    color = _get_target_color(target)
    fig = go.Figure(
        go.Bar(
            x=[float(r["importance"]) for r in sorted_rows],
            y=[str(r["feature_name"]) for r in sorted_rows],
            orientation="h",
            marker={"color": color},
            showlegend=False,
        ),
    )
    fig.update_layout(
        title={"text": f"Top features — {target.upper()}", "x": 0.5},
        yaxis={"automargin": True},
        xaxis_title="Importance",
        showlegend=False,
        height=400,
    )
    _apply_layout_defaults(fig)
    return _to_html(fig)


def generate_insights_feature_importance_wp(importances: list[dict]) -> str:
    return _generate_feature_importance(importances, "wp")


def generate_insights_feature_importance_ats(importances: list[dict]) -> str:
    return _generate_feature_importance(importances, "ats")


def generate_insights_feature_importance_ou(importances: list[dict]) -> str:
    return _generate_feature_importance(importances, "ou")


# Optional parameterized convenience (plan text uses this name).
def generate_insights_feature_importance(
    importances: list[dict],
    target: str,
) -> str:
    return _generate_feature_importance(importances, target)


# ---------------------------------------------------------------------------
# 4. Per-season accuracy + MAE trend
# ---------------------------------------------------------------------------


def generate_insights_accuracy_trend(metrics: list[dict]) -> str:
    """Render per-season WP accuracy + ATS/OU MAE on a dual-subplot chart."""
    season_metrics = [m for m in metrics if int(m.get("season", 0)) > 0]
    if not season_metrics:
        return _empty_chart_div("No per-season metrics available")

    indexed: dict[tuple[int, str, str], float] = {}
    for m in season_metrics:
        key = (int(m["season"]), str(m["target"]), str(m["metric_name"]))
        indexed[key] = float(m["metric_value"])

    seasons = sorted({int(m["season"]) for m in season_metrics})
    if not seasons:
        return _empty_chart_div("No per-season metrics available")

    wp_acc = [indexed.get((s, "wp", "accuracy")) for s in seasons]
    ats_mae = [indexed.get((s, "ats", "mae")) for s in seasons]
    ou_mae = [indexed.get((s, "ou", "mae")) for s in seasons]

    # If every series is entirely missing, empty-state.
    if (
        all(v is None for v in wp_acc)
        and all(v is None for v in ats_mae)
        and all(v is None for v in ou_mae)
    ):
        return _empty_chart_div("No per-season metrics available")

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Win Probability Accuracy", "ATS / OU Mean Absolute Error"),
    )

    fig.add_trace(
        go.Scatter(
            x=seasons,
            y=wp_acc,
            mode="lines+markers",
            name="WP accuracy",
            line={"color": _get_target_color("wp"), "width": 2},
            marker={"size": 8},
            connectgaps=False,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=seasons,
            y=ats_mae,
            mode="lines+markers",
            name="ATS MAE",
            line={"color": _get_target_color("ats"), "width": 2},
            marker={"size": 8},
            connectgaps=False,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=seasons,
            y=ou_mae,
            mode="lines+markers",
            name="OU MAE",
            line={"color": _get_target_color("ou"), "width": 2},
            marker={"size": 8},
            connectgaps=False,
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(dtick=1, row=1, col=1)
    fig.update_xaxes(dtick=1, row=1, col=2)
    fig.update_yaxes(title_text="Accuracy", row=1, col=1)
    fig.update_yaxes(title_text="MAE", row=1, col=2)
    fig.update_layout(
        title={"text": "Per-season accuracy and MAE", "x": 0.5},
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.2},
    )
    _apply_layout_defaults(fig)
    return _to_html(fig)


# ---------------------------------------------------------------------------
# 5. Model vs Market — WP (3 subplots: Accuracy / Brier / Log-loss)
# ---------------------------------------------------------------------------


def generate_insights_model_vs_market_wp(
    predictions: list[dict],
    market_data: list[dict],
) -> str:
    """Render per-season model vs market WP on Accuracy / Brier / Log-loss."""
    wp_rows = [p for p in predictions if p.get("target") == "wp"]
    if not wp_rows:
        return _empty_chart_div("No model vs market WP data available")

    market_lookup = {p["game_id"]: p for p in market_data}

    by_season: dict[int, list[dict]] = {}
    for row in wp_rows:
        by_season.setdefault(int(row["season"]), []).append(row)

    seasons: list[int] = []
    model_acc: list[float | None] = []
    model_brier: list[float | None] = []
    model_log: list[float | None] = []
    market_acc: list[float | None] = []
    market_brier: list[float | None] = []
    market_log: list[float | None] = []

    excluded_count = 0
    for season in sorted(by_season):
        rows = by_season[season]
        if len(rows) < _MIN_WP_ROWS_PER_SEASON:
            logger.info(
                "Skipping WP season %s — only %d rows (< %d)",
                season,
                len(rows),
                _MIN_WP_ROWS_PER_SEASON,
            )
            continue

        model_actuals = np.array([float(r["actual"]) for r in rows])
        model_probs = np.array([float(r["model_prob"]) for r in rows])

        # Match market rows by game_id; skip rows with both moneylines null.
        paired_actuals: list[float] = []
        paired_market_probs: list[float] = []
        for r in rows:
            market_row = market_lookup.get(r["game_id"])
            if market_row is None:
                excluded_count += 1
                continue
            devig = compute_wp_market_prob(
                market_row.get("market_ml_home"),
                market_row.get("market_ml_away"),
            )
            if devig is None:
                excluded_count += 1
                continue
            paired_actuals.append(float(r["actual"]))
            paired_market_probs.append(float(devig["prob"]))

        seasons.append(season)
        model_acc.append(compute_accuracy(model_actuals, model_probs))
        model_brier.append(compute_brier(model_actuals, model_probs))
        model_log.append(compute_log_loss(model_actuals, model_probs))

        if paired_actuals:
            pa = np.array(paired_actuals)
            pm = np.array(paired_market_probs)
            market_acc.append(compute_accuracy(pa, pm))
            market_brier.append(compute_brier(pa, pm))
            market_log.append(compute_log_loss(pa, pm))
        else:
            market_acc.append(None)
            market_brier.append(None)
            market_log.append(None)

    if excluded_count:
        logger.info(
            "Excluded %d WP market rows due to missing moneylines",
            excluded_count,
        )

    if not seasons:
        return _empty_chart_div("No per-season WP data available")

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Accuracy", "Brier Score", "Log-Loss"),
    )

    wp_color = _get_target_color("wp")

    def _add_pair(
        col: int,
        model_series: list[float | None],
        market_series: list[float | None],
    ) -> None:
        fig.add_trace(
            go.Scatter(
                x=seasons,
                y=model_series,
                mode="lines+markers",
                name="Model",
                line={"color": wp_color, "width": 2},
                marker={"size": 8},
                connectgaps=False,
                showlegend=(col == 1),
                legendgroup="model",
            ),
            row=1,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=seasons,
                y=market_series,
                mode="lines+markers",
                name="Market",
                line={"color": DEFAULT_COLOR, "width": 2, "dash": "dash"},
                marker={"size": 8},
                connectgaps=False,
                showlegend=(col == 1),
                legendgroup="market",
            ),
            row=1,
            col=col,
        )

    _add_pair(1, model_acc, market_acc)
    _add_pair(2, model_brier, market_brier)
    _add_pair(3, model_log, market_log)

    fig.update_xaxes(dtick=1)
    fig.update_layout(
        title={"text": "WP — Model vs Market", "x": 0.5},
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.25},
    )
    _apply_layout_defaults(fig)
    return _to_html(fig)


# ---------------------------------------------------------------------------
# 6. Model vs Market — ATS
# ---------------------------------------------------------------------------


def generate_insights_model_vs_market_ats(
    predictions: list[dict],
    market_data: list[dict],
) -> str:
    """Render per-season model vs market ATS MAE."""
    ats_rows = [p for p in predictions if p.get("target") == "ats"]
    if not ats_rows:
        return _empty_chart_div("No model vs market ATS data available")

    market_lookup = {p["game_id"]: p for p in market_data}

    by_season: dict[int, list[dict]] = {}
    for row in ats_rows:
        market_row = market_lookup.get(row["game_id"])
        if market_row is None or market_row.get("market_spread") is None:
            continue
        merged = dict(row)
        merged["market_spread"] = market_row.get("market_spread")
        by_season.setdefault(int(row["season"]), []).append(merged)

    if not by_season:
        return _empty_chart_div("No model vs market ATS data available")

    seasons: list[int] = []
    model_mae: list[float | None] = []
    market_mae: list[float | None] = []
    for season in sorted(by_season):
        m_mae, mk_mae = compute_ats_mae_model_and_market(by_season[season])
        seasons.append(season)
        model_mae.append(None if np.isnan(m_mae) else m_mae)
        market_mae.append(None if np.isnan(mk_mae) else mk_mae)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=seasons,
            y=model_mae,
            mode="lines+markers",
            name="Model",
            line={"color": _get_target_color("ats"), "width": 2},
            marker={"size": 8},
            connectgaps=False,
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=seasons,
            y=market_mae,
            mode="lines+markers",
            name="Market",
            line={"color": DEFAULT_COLOR, "width": 2, "dash": "dash"},
            marker={"size": 8},
            connectgaps=False,
        ),
    )
    fig.update_layout(
        title={"text": "ATS — Model vs Market MAE", "x": 0.5},
        xaxis_title="Season",
        yaxis_title="Mean Absolute Error",
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.2},
    )
    fig.update_xaxes(dtick=1)
    _apply_layout_defaults(fig)
    return _to_html(fig)


# ---------------------------------------------------------------------------
# 7. Model vs Market — OU
# ---------------------------------------------------------------------------


def generate_insights_model_vs_market_ou(
    predictions: list[dict],
    market_data: list[dict],
) -> str:
    """Render per-season model vs market OU MAE."""
    ou_rows = [p for p in predictions if p.get("target") == "ou"]
    if not ou_rows:
        return _empty_chart_div("No model vs market OU data available")

    market_lookup = {p["game_id"]: p for p in market_data}

    by_season: dict[int, list[dict]] = {}
    for row in ou_rows:
        market_row = market_lookup.get(row["game_id"])
        if market_row is None or market_row.get("market_total") is None:
            continue
        merged = dict(row)
        merged["market_total"] = market_row.get("market_total")
        by_season.setdefault(int(row["season"]), []).append(merged)

    if not by_season:
        return _empty_chart_div("No model vs market OU data available")

    seasons: list[int] = []
    model_mae: list[float | None] = []
    market_mae: list[float | None] = []
    for season in sorted(by_season):
        m_mae, mk_mae = compute_ou_mae_model_and_market(by_season[season])
        seasons.append(season)
        model_mae.append(None if np.isnan(m_mae) else m_mae)
        market_mae.append(None if np.isnan(mk_mae) else mk_mae)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=seasons,
            y=model_mae,
            mode="lines+markers",
            name="Model",
            line={"color": _get_target_color("ou"), "width": 2},
            marker={"size": 8},
            connectgaps=False,
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=seasons,
            y=market_mae,
            mode="lines+markers",
            name="Market",
            line={"color": DEFAULT_COLOR, "width": 2, "dash": "dash"},
            marker={"size": 8},
            connectgaps=False,
        ),
    )
    fig.update_layout(
        title={"text": "OU — Model vs Market MAE", "x": 0.5},
        xaxis_title="Season",
        yaxis_title="Mean Absolute Error",
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.2},
    )
    fig.update_xaxes(dtick=1)
    _apply_layout_defaults(fig)
    return _to_html(fig)
