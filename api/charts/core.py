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
include_plotlyjs=False (Plotly CDN loaded in base.html). Every figure gets the dark
dashboard theme from api.charts.theme through _apply_layout_defaults.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from sklearn.calibration import calibration_curve

from api.charts.theme import (
    BOUNDARY_LINE,
    FG,
    HEATMAP_SCALE,
    MUTED,
    REFERENCE_LINE,
    SEASON_COLORS,
    STRATEGY_COLORS,
    TARGET_COLORS,
    LegendPosition,
    apply_dark_theme,
)

# ---------------------------------------------------------------------------
# Layout defaults
# ---------------------------------------------------------------------------

PLOTLY_CONFIG = {"responsive": True, "displayModeBar": False}

# A series with no palette entry falls back to the theme's muted grey, never a light grey
# that disappears on the dark panel.
DEFAULT_COLOR = MUTED


def _apply_layout_defaults(
    fig: go.Figure, *, legend_position: LegendPosition = "top"
) -> None:
    """Apply the shared dark chart theme (api.charts.theme) to a Plotly figure.

    Every generator calls this LAST, after its own layout. The theme's backgrounds, fonts,
    gridlines, hover label and legend placement therefore win over any per-chart leftovers.
    Pass legend_position="bottom" when subplot titles or a long legend already occupy the
    top edge.
    """
    apply_dark_theme(fig, legend_position=legend_position)


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
    """Return an HTML div indicating no chart data is available.

    The classes are dark-theme tokens. web/static/input.css scans api/charts, so they are
    compiled even though they appear only in this Python string.
    """
    return (
        '<div class="flex min-h-[220px] items-center justify-center rounded-sm '
        'border border-dashed border-line px-4 text-center text-sm text-muted">'
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
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line={"dash": "dash", "color": REFERENCE_LINE, "width": 1},
            name="Perfect Calibration",
            showlegend=True,
        )
    )

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

        fig.add_trace(
            go.Scatter(
                x=mean_pred,
                y=fraction_pos,
                mode="lines+markers",
                name=str(season),
                line={"color": _get_season_color(season), "width": 2},
                marker={"size": 6},
            )
        )

    # Overall calibration curve
    ece_text = "ECE = N/A"
    if all_y_true:
        combined_true = np.concatenate(all_y_true)
        combined_prob = np.concatenate(all_y_prob)

        fraction_pos, mean_pred = calibration_curve(
            combined_true, combined_prob, n_bins=10, strategy="uniform"
        )

        fig.add_trace(
            go.Scatter(
                x=mean_pred,
                y=fraction_pos,
                mode="lines+markers",
                name="Overall",
                line={"color": FG, "width": 3},
                marker={"size": 8},
            )
        )

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
    )
    # Up to seven entries (reference, seasons, overall) would wrap into the title on top.
    _apply_layout_defaults(fig, legend_position="bottom")

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

        # Filter to valid CLV values. Require has_closing_odds is explicitly
        # True so rows where the closing-line capture status is unknown
        # (None / NULL) do not pollute the cumulative CLV mean. Mirrors the
        # contract in api/routes/pages.py::_compute_summary.
        valid_preds = [
            p
            for p in preds
            if p.get("probability_clv") is not None
            and p.get("has_closing_odds") is True
        ]

        if not valid_preds:
            continue

        # Sort chronologically
        valid_preds.sort(key=lambda p: (p.get("season", 0), p.get("week", 0)))

        clv_values = np.array([float(p["probability_clv"]) for p in valid_preds])
        cumulative_clv = np.cumsum(clv_values) / np.arange(1, len(clv_values) + 1)

        x_vals = list(range(len(cumulative_clv)))
        mean_clv = float(clv_values.mean())

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=cumulative_clv.tolist(),
                mode="lines",
                name=f"{target.upper()} (Mean: {mean_clv:+.4f})",
                line={"color": _get_target_color(target), "width": 2},
            )
        )

        # Season boundary markers
        for i in range(1, len(valid_preds)):
            if valid_preds[i].get("season") != valid_preds[i - 1].get("season"):
                fig.add_vline(
                    x=i,
                    line_dash="dot",
                    line_color=BOUNDARY_LINE,
                    line_width=1,
                )

        has_data = True

    if not has_data:
        return _empty_chart_div("No CLV data available")

    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color=REFERENCE_LINE, line_width=1)

    fig.update_layout(
        title={"text": "Cumulative CLV Over Time", "x": 0.5},
        xaxis_title="Game Index (Chronological)",
        yaxis_title="Cumulative Mean CLV",
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
            available_preferred = [m for m in preferred[target] if m in metric_names]
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
                    m
                    for m in target_metrics
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

        # One monochrome scale: brighter = a higher value. On this site green/red mean a
        # realised win/loss, and a metric heatmap is neither. Each trace keeps the direction
        # it already had: the ats and ou traces are reversed, exactly where the old
        # red-yellow-green scale was reversed, and the wp trace is not.
        fig.add_trace(
            go.Heatmap(
                z=z_values,
                x=metric_names,
                y=[str(s) for s in all_seasons],
                text=text_values,
                texttemplate="%{text}",
                textfont={"color": FG},
                colorscale=HEATMAP_SCALE,
                reversescale=target in ("ats", "ou"),
                xgap=2,
                ygap=2,
                colorbar={"outlinewidth": 0},
                showscale=(col_idx == n_targets),
                hovertemplate=(
                    "Season: %{y}<br>Metric: %{x}<br>Value: %{text}<extra></extra>"
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

    for strategy in sorted(strategies):
        rows = sorted(strategies[strategy], key=lambda r: r.get("bet_index", 0))
        x_vals = [r.get("bet_index", i) for i, r in enumerate(rows)]
        y_vals = [float(r.get("bankroll", 0)) for r in rows]

        color = STRATEGY_COLORS.get(strategy, DEFAULT_COLOR)
        display_name = strategy.replace("_", " ").title()

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=y_vals,
                mode="lines",
                name=display_name,
                line={"color": color, "width": 2},
            )
        )

    # Break-even reference line at starting bankroll (first value)
    if equity_data:
        first_bankroll = float(equity_data[0].get("bankroll", 10000))
        fig.add_hline(
            y=first_bankroll,
            line_dash="dash",
            line_color=REFERENCE_LINE,
            line_width=1,
            annotation_text="Starting Bankroll",
            annotation_position="bottom right",
        )

    fig.update_layout(
        title={"text": "Betting Simulation: Flat-Stake vs Kelly", "x": 0.5},
        xaxis_title="Bet Number",
        yaxis_title="Bankroll ($)",
        yaxis_tickformat="$,.0f",
    )
    _apply_layout_defaults(fig)

    return _to_html(fig)
