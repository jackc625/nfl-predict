"""Dashboard chart generation module.

Adapts the Phase 6 backtest Plotly chart types for the web dashboard.
Works with plain dicts (from DuckDB) rather than BacktestResults objects,
maintaining UIAP-01 compliance (no model class imports).

Chart types:
- Calibration reliability diagram (WP target)
- Cumulative CLV over time (all targets)
- Season comparison heatmap (metrics by season/target)
- Equity curve (flat-stake vs Kelly)

Each function returns an HTML div string via plotly.io.to_html with
include_plotlyjs=False (Plotly CDN loaded in base.html).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from sklearn.calibration import calibration_curve

from backtest.report import SEASON_COLORS, TARGET_COLORS

if TYPE_CHECKING:
    from api.services import DataService

# ---------------------------------------------------------------------------
# Layout defaults (per UI-SPEC CHART_LAYOUT_DEFAULTS)
# ---------------------------------------------------------------------------

CHART_LAYOUT_DEFAULTS: dict[str, Any] = {
    "font_family": "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
    "font_size": 13,
    "plot_bgcolor": "#FFFFFF",
    "paper_bgcolor": "#FFFFFF",
    "margin": {"l": 48, "r": 16, "t": 48, "b": 48},
    "gridcolor": "#E5E7EB",
}

PLOTLY_CONFIG = {"responsive": True, "displayModeBar": False}

DEFAULT_COLOR = "#95a5a6"


def _apply_layout_defaults(fig: go.Figure) -> None:
    """Apply the UI-SPEC layout defaults to a Plotly figure."""
    defaults = CHART_LAYOUT_DEFAULTS
    fig.update_layout(
        font_family=defaults["font_family"],
        font_size=defaults["font_size"],
        plot_bgcolor=defaults["plot_bgcolor"],
        paper_bgcolor=defaults["paper_bgcolor"],
        margin=defaults["margin"],
    )
    fig.update_xaxes(gridcolor=defaults["gridcolor"])
    fig.update_yaxes(gridcolor=defaults["gridcolor"])


def _to_html(fig: go.Figure) -> str:
    """Convert a Plotly figure to an HTML div string (no Plotly JS included)."""
    return pio.to_html(
        fig,
        full_html=False,
        include_plotlyjs=False,
        config=PLOTLY_CONFIG,
    )


def _get_season_color(season: int) -> str:
    """Return a consistent color for a given season."""
    return SEASON_COLORS.get(season, DEFAULT_COLOR)


def _get_target_color(target: str) -> str:
    """Return a consistent color for a given target."""
    return TARGET_COLORS.get(target, DEFAULT_COLOR)


def _empty_chart_div(message: str) -> str:
    """Return an HTML div indicating no chart data is available."""
    return (
        '<div class="flex items-center justify-center h-full text-sm text-gray-500">'
        f"<p>{message}</p></div>"
    )


# ---------------------------------------------------------------------------
# Chart 1: Calibration Reliability Diagram
# ---------------------------------------------------------------------------


def generate_dashboard_calibration_chart(predictions: list[dict]) -> str:
    """Generate WP model calibration reliability diagram.

    Takes backtest_predictions (target=wp), groups by season, and generates
    the same visual as backtest/report.py but from plain dicts.

    Args:
        predictions: List of backtest prediction dicts with at least
            season, model_prob, actual keys (target=wp rows only).

    Returns:
        HTML div string of the Plotly chart.
    """
    # Filter to WP predictions only
    wp_preds = [p for p in predictions if p.get("target") == "wp"]
    if not wp_preds:
        return _empty_chart_div("No WP calibration data available")

    fig = go.Figure()

    # Diagonal reference line (perfect calibration)
    fig.add_trace(go.Scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        line={"dash": "dash", "color": "#999", "width": 1},
        name="Perfect Calibration",
        showlegend=True,
    ))

    # Group by season
    seasons: dict[int, list[dict]] = {}
    for p in wp_preds:
        s = int(p["season"])
        seasons.setdefault(s, []).append(p)

    all_y_true: list[np.ndarray] = []
    all_y_prob: list[np.ndarray] = []

    for season in sorted(seasons):
        season_preds = seasons[season]
        y_true = np.array([float(p["actual"]) for p in season_preds])
        y_prob = np.array([float(p["model_prob"]) for p in season_preds])

        if len(y_true) < 10:
            continue

        all_y_true.append(y_true)
        all_y_prob.append(y_prob)

        fraction_pos, mean_pred = calibration_curve(
            y_true, y_prob, n_bins=10, strategy="uniform"
        )

        fig.add_trace(go.Scatter(
            x=mean_pred,
            y=fraction_pos,
            mode="lines+markers",
            name=str(season),
            line={"color": _get_season_color(season), "width": 2},
            marker={"size": 6},
        ))

    # Overall calibration curve
    ece_text = "ECE = N/A"
    if all_y_true:
        combined_true = np.concatenate(all_y_true)
        combined_prob = np.concatenate(all_y_prob)

        fraction_pos, mean_pred = calibration_curve(
            combined_true, combined_prob, n_bins=10, strategy="uniform"
        )

        fig.add_trace(go.Scatter(
            x=mean_pred,
            y=fraction_pos,
            mode="lines+markers",
            name="Overall",
            line={"color": "#333", "width": 3},
            marker={"size": 8},
        ))

        # Compute ECE
        from backtest.metrics import _compute_ece

        ece_value = _compute_ece(combined_true, combined_prob, n_bins=10)
        ece_text = f"ECE = {ece_value:.4f}"

    fig.update_layout(
        title={
            "text": f"WP Calibration<br><sub>{ece_text}</sub>",
            "x": 0.5,
        },
        xaxis_title="Mean Predicted Probability",
        yaxis_title="Observed Frequency",
        xaxis={"range": [0, 1]},
        yaxis={"range": [0, 1]},
        legend={"x": 0.02, "y": 0.98},
    )
    _apply_layout_defaults(fig)

    return _to_html(fig)


# ---------------------------------------------------------------------------
# Chart 2: Cumulative CLV
# ---------------------------------------------------------------------------


def generate_dashboard_clv_chart(predictions: list[dict]) -> str:
    """Generate cumulative CLV chart over time for all targets.

    Args:
        predictions: List of backtest prediction dicts with at least
            season, week, target, probability_clv keys.

    Returns:
        HTML div string of the Plotly chart.
    """
    if not predictions:
        return _empty_chart_div("No CLV data available")

    fig = go.Figure()

    # Group by target
    targets: dict[str, list[dict]] = {}
    for p in predictions:
        t = p.get("target", "")
        targets.setdefault(t, []).append(p)

    has_data = False
    for target in sorted(targets):
        preds = targets[target]

        # Filter to valid CLV values
        valid_preds = [
            p for p in preds
            if p.get("probability_clv") is not None
            and p.get("has_closing_odds") is not False
        ]

        if not valid_preds:
            continue

        # Sort chronologically
        valid_preds.sort(key=lambda p: (p.get("season", 0), p.get("week", 0)))

        clv_values = np.array([float(p["probability_clv"]) for p in valid_preds])
        cumulative_clv = np.cumsum(clv_values) / np.arange(1, len(clv_values) + 1)

        x_vals = list(range(len(cumulative_clv)))
        mean_clv = float(clv_values.mean())

        fig.add_trace(go.Scatter(
            x=x_vals,
            y=cumulative_clv.tolist(),
            mode="lines",
            name=f"{target.upper()} (Mean: {mean_clv:+.4f})",
            line={"color": _get_target_color(target), "width": 2},
        ))

        # Season boundary markers
        for i in range(1, len(valid_preds)):
            if valid_preds[i].get("season") != valid_preds[i - 1].get("season"):
                fig.add_vline(
                    x=i,
                    line_dash="dot",
                    line_color="#ccc",
                    line_width=1,
                )

        has_data = True

    if not has_data:
        return _empty_chart_div("No CLV data available")

    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="#999", line_width=1)

    fig.update_layout(
        title={"text": "Cumulative CLV Over Time", "x": 0.5},
        xaxis_title="Game Index (Chronological)",
        yaxis_title="Cumulative Mean CLV",
        legend={"x": 0.02, "y": 0.98},
    )
    _apply_layout_defaults(fig)

    return _to_html(fig)


# ---------------------------------------------------------------------------
# Chart 3: Season Comparison Heatmap
# ---------------------------------------------------------------------------


def generate_dashboard_heatmap(metrics: list[dict]) -> str:
    """Generate season comparison heatmap across targets and metrics.

    Args:
        metrics: List of backtest metric dicts with at least
            season, target, metric_name, metric_value keys.

    Returns:
        HTML div string of the Plotly chart.
    """
    if not metrics:
        return _empty_chart_div("No season metrics available")

    # Filter out overall/aggregate rows (season=0)
    season_metrics = [m for m in metrics if m.get("season", 0) != 0]
    if not season_metrics:
        return _empty_chart_div("No per-season metrics available")

    # Discover unique targets and seasons
    targets = sorted({m["target"] for m in season_metrics})
    all_seasons = sorted({int(m["season"]) for m in season_metrics})

    if not targets or not all_seasons:
        return _empty_chart_div("No season metrics available")

    n_targets = len(targets)

    fig = make_subplots(
        rows=1,
        cols=n_targets,
        subplot_titles=[t.upper() for t in targets],
        horizontal_spacing=0.08,
    )

    for col_idx, target in enumerate(targets, 1):
        target_metrics = [m for m in season_metrics if m["target"] == target]

        # Discover metric names for this target
        metric_names = sorted({m["metric_name"] for m in target_metrics})
        # Prefer a meaningful subset if too many
        preferred = {
            "wp": ["accuracy", "brier_score", "ece", "log_loss"],
            "ats": ["mae", "rmse"],
            "ou": ["mae", "rmse"],
        }
        if target in preferred:
            available_preferred = [
                m for m in preferred[target] if m in metric_names
            ]
            if available_preferred:
                metric_names = available_preferred

        if not metric_names:
            continue

        # Build z-values matrix: rows=seasons, cols=metrics
        z_values: list[list[float]] = []
        text_values: list[list[str]] = []

        for season in all_seasons:
            row_z: list[float] = []
            row_text: list[str] = []
            for metric in metric_names:
                matching = [
                    m for m in target_metrics
                    if int(m["season"]) == season and m["metric_name"] == metric
                ]
                if matching:
                    val = float(matching[0]["metric_value"])
                    row_z.append(val)
                    if metric == "accuracy":
                        row_text.append(f"{val:.1%}")
                    elif metric in ("brier_score", "ece", "log_loss"):
                        row_text.append(f"{val:.4f}")
                    else:
                        row_text.append(f"{val:.2f}")
                else:
                    row_z.append(0.0)
                    row_text.append("N/A")

            z_values.append(row_z)
            text_values.append(row_text)

        # Color scale: for error metrics, reverse (green = low)
        colorscale = "RdYlGn"
        if target in ("ats", "ou"):
            colorscale = "RdYlGn_r"

        fig.add_trace(
            go.Heatmap(
                z=z_values,
                x=metric_names,
                y=[str(s) for s in all_seasons],
                text=text_values,
                texttemplate="%{text}",
                colorscale=colorscale,
                showscale=(col_idx == n_targets),
                hovertemplate=(
                    "Season: %{y}<br>"
                    "Metric: %{x}<br>"
                    "Value: %{text}<extra></extra>"
                ),
            ),
            row=1,
            col=col_idx,
        )

    fig.update_layout(
        title={"text": "Season Comparison Heatmap", "x": 0.5},
    )
    _apply_layout_defaults(fig)

    return _to_html(fig)


# ---------------------------------------------------------------------------
# Chart 4: Equity Curves
# ---------------------------------------------------------------------------


def generate_dashboard_equity_chart(equity_data: list[dict]) -> str:
    """Generate betting simulation equity curves (flat-stake vs Kelly).

    Args:
        equity_data: List of equity curve dicts with at least
            strategy, bet_index, bankroll keys.

    Returns:
        HTML div string of the Plotly chart.
    """
    if not equity_data:
        return _empty_chart_div("No equity curve data available")

    fig = go.Figure()

    # Group by strategy
    strategies: dict[str, list[dict]] = {}
    for row in equity_data:
        s = row.get("strategy", "unknown")
        strategies.setdefault(s, []).append(row)

    strategy_colors = {
        "flat_stake": "#667eea",
        "kelly": "#f093fb",
    }

    for strategy in sorted(strategies):
        rows = sorted(strategies[strategy], key=lambda r: r.get("bet_index", 0))
        x_vals = [r.get("bet_index", i) for i, r in enumerate(rows)]
        y_vals = [float(r.get("bankroll", 0)) for r in rows]

        color = strategy_colors.get(strategy, DEFAULT_COLOR)
        display_name = strategy.replace("_", " ").title()

        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="lines",
            name=display_name,
            line={"color": color, "width": 2},
        ))

    # Break-even reference line at starting bankroll (first value)
    if equity_data:
        first_bankroll = float(equity_data[0].get("bankroll", 10000))
        fig.add_hline(
            y=first_bankroll,
            line_dash="dash",
            line_color="#999",
            line_width=1,
            annotation_text="Starting Bankroll",
            annotation_position="bottom right",
        )

    fig.update_layout(
        title={"text": "Betting Simulation: Flat-Stake vs Kelly", "x": 0.5},
        xaxis_title="Bet Number",
        yaxis_title="Bankroll ($)",
        legend={"x": 0.02, "y": 0.98},
        yaxis_tickformat="$,.0f",
    )
    _apply_layout_defaults(fig)

    return _to_html(fig)


# ---------------------------------------------------------------------------
# Pre-render for cache population
# ---------------------------------------------------------------------------


def prerender_charts_for_cache(service: DataService) -> dict[str, str]:
    """Generate all 4 dashboard charts and return their HTML divs.

    Called during cache population to store pre-rendered charts in the
    chart_cache DuckDB table.

    Args:
        service: DataService instance pointing to the web cache.

    Returns:
        Dict mapping chart_id to HTML div string:
        {"calibration": html, "clv": html, "heatmap": html, "equity": html}
    """
    charts: dict[str, str] = {}

    # Calibration + CLV use backtest predictions
    predictions = service.get_backtest_predictions()

    charts["calibration"] = generate_dashboard_calibration_chart(predictions)
    charts["clv"] = generate_dashboard_clv_chart(predictions)

    # Heatmap uses backtest metrics
    metrics = service.get_backtest_metrics()
    charts["heatmap"] = generate_dashboard_heatmap(metrics)

    # Equity uses equity curve data
    equity_data = service.get_equity_curve()
    charts["equity"] = generate_dashboard_equity_chart(equity_data)

    return charts


def prerender_charts_from_conn(conn: Any) -> dict[str, str]:
    """Generate all 4 dashboard charts directly from a DuckDB connection.

    Used during cache population when the database is already open in write
    mode (can't open a second read-only connection to the same file).

    Args:
        conn: Active DuckDB connection with populated tables.

    Returns:
        Dict mapping chart_id to HTML div string.
    """
    charts: dict[str, str] = {}

    # Query backtest_predictions for calibration + CLV charts
    predictions = [
        dict(zip([d[0] for d in conn.description], row))
        for row in conn.execute("SELECT * FROM backtest_predictions").fetchall()
    ] if conn.execute(
        "SELECT COUNT(*) FROM backtest_predictions"
    ).fetchone()[0] > 0 else []

    charts["calibration"] = generate_dashboard_calibration_chart(predictions)
    charts["clv"] = generate_dashboard_clv_chart(predictions)

    # Query backtest_metrics for heatmap
    metrics = [
        dict(zip([d[0] for d in conn.description], row))
        for row in conn.execute("SELECT * FROM backtest_metrics").fetchall()
    ] if conn.execute(
        "SELECT COUNT(*) FROM backtest_metrics"
    ).fetchone()[0] > 0 else []

    charts["heatmap"] = generate_dashboard_heatmap(metrics)

    # Query equity_curve for equity chart
    equity_data = [
        dict(zip([d[0] for d in conn.description], row))
        for row in conn.execute(
            "SELECT * FROM equity_curve ORDER BY strategy, bet_index"
        ).fetchall()
    ] if conn.execute(
        "SELECT COUNT(*) FROM equity_curve"
    ).fetchone()[0] > 0 else []

    charts["equity"] = generate_dashboard_equity_chart(equity_data)

    return charts
