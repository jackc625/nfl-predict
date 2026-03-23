"""Backtest report generator with Plotly charts and Jinja2 HTML assembly.

Produces a single-page interactive HTML dashboard with:
- Calibration reliability diagram (WP model)
- Cumulative CLV chart (all targets)
- Season comparison heatmap (metrics by season/target)
- Betting simulation equity curves (flat-stake vs Kelly)
- Weekly performance trends (per season/target)

Also provides the BacktestReporter class that orchestrates chart
generation and HTML template rendering.

Key design decisions:
- Plotly charts rendered as HTML divs (no full_html), loaded via CDN
- Jinja2 templates in backtest/templates/
- Output to outputs/backtest/ by default (D-10)
- COVID annotation included in methodology section (BACK-05)
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import plotly.graph_objects as go
from jinja2 import Environment, FileSystemLoader
from plotly.subplots import make_subplots
from sklearn.calibration import calibration_curve

from utils import get_logger

if TYPE_CHECKING:
    from backtest.engine import BacktestResults
    from backtest.simulation import SimulationResults

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Color palette (consistent across charts)
# ---------------------------------------------------------------------------

SEASON_COLORS: dict[int, str] = {
    2020: "#FF6B6B",
    2021: "#4ECDC4",
    2022: "#45B7D1",
    2023: "#96CEB4",
    2024: "#FFEAA7",
}
TARGET_COLORS: dict[str, str] = {
    "wp": "#667eea",
    "ats": "#f093fb",
    "ou": "#4fd1c5",
}
DEFAULT_COLOR = "#95a5a6"


def _get_season_color(season: int) -> str:
    """Return a consistent color for a given season."""
    return SEASON_COLORS.get(season, DEFAULT_COLOR)


def _get_target_color(target: str) -> str:
    """Return a consistent color for a given target."""
    return TARGET_COLORS.get(target, DEFAULT_COLOR)


# ---------------------------------------------------------------------------
# Chart 1: Calibration Reliability Diagram
# ---------------------------------------------------------------------------


def generate_calibration_chart(backtest_results: BacktestResults) -> go.Figure:
    """Generate WP model calibration reliability diagram.

    Plots observed frequency vs mean predicted probability for the WP
    target, with per-season traces and an overall trace. Includes
    a perfect-calibration diagonal and ECE annotation.

    Args:
        backtest_results: Complete backtest results from the engine.

    Returns:
        Plotly Figure with calibration curves.
    """
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

    # Collect all WP predictions for the overall curve
    all_y_true: list[np.ndarray] = []
    all_y_prob: list[np.ndarray] = []

    # Per-season calibration curves
    for sr in backtest_results.season_results:
        if "wp" not in sr.target_results:
            continue

        tr = sr.target_results["wp"]
        preds_df = tr.predictions_df

        if preds_df.empty:
            continue

        # Extract true outcomes and predicted probabilities
        if "actual" in preds_df.columns and "model_prob" in preds_df.columns:
            y_true = preds_df["actual"].values.astype(float)
            y_prob = preds_df["model_prob"].values.astype(float)
        else:
            continue

        if len(y_true) < 10:
            continue

        all_y_true.append(y_true)
        all_y_prob.append(y_prob)

        # sklearn calibration_curve
        fraction_pos, mean_pred = calibration_curve(
            y_true, y_prob, n_bins=10, strategy="uniform"
        )

        fig.add_trace(go.Scatter(
            x=mean_pred,
            y=fraction_pos,
            mode="lines+markers",
            name=f"{sr.season}",
            line={"color": _get_season_color(sr.season), "width": 2},
            marker={"size": 6},
        ))

    # Overall calibration curve (all seasons combined)
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

        # Compute ECE for annotation
        from backtest.metrics import _compute_ece

        ece_value = _compute_ece(combined_true, combined_prob, n_bins=10)
        ece_text = f"ECE = {ece_value:.4f}"
    else:
        ece_text = "ECE = N/A (no WP predictions)"

    fig.update_layout(
        title={
            "text": f"WP Model Calibration (Reliability Diagram)<br>"
                    f"<sub>{ece_text}</sub>",
            "x": 0.5,
        },
        xaxis_title="Mean Predicted Probability",
        yaxis_title="Observed Frequency",
        xaxis={"range": [0, 1]},
        yaxis={"range": [0, 1]},
        template="plotly_white",
        legend={"x": 0.02, "y": 0.98},
        width=800,
        height=600,
    )

    return fig


# ---------------------------------------------------------------------------
# Chart 2: Cumulative CLV
# ---------------------------------------------------------------------------


def generate_clv_chart(backtest_results: BacktestResults) -> go.Figure:
    """Generate cumulative CLV chart over time for all targets.

    Plots cumulative mean CLV as a line over games, sorted
    chronologically. Vertical dashed lines mark season boundaries.

    Args:
        backtest_results: Complete backtest results from the engine.

    Returns:
        Plotly Figure with cumulative CLV curves per target.
    """
    fig = go.Figure()

    for target in backtest_results.config.targets:
        clv_df = backtest_results.all_clv.get(target)
        if clv_df is None or clv_df.empty:
            continue

        if "probability_clv" not in clv_df.columns:
            continue

        # Sort chronologically
        sort_cols = []
        if "season" in clv_df.columns:
            sort_cols.append("season")
        if "week" in clv_df.columns:
            sort_cols.append("week")

        if sort_cols:
            clv_sorted = clv_df.sort_values(sort_cols).reset_index(drop=True)
        else:
            clv_sorted = clv_df.reset_index(drop=True)

        # Filter to valid CLV values
        if "has_closing_odds" in clv_sorted.columns:
            clv_sorted = clv_sorted[clv_sorted["has_closing_odds"] == True].reset_index(drop=True)  # noqa: E712

        clv_values = clv_sorted["probability_clv"].values
        cumulative_clv = np.cumsum(clv_values) / np.arange(1, len(clv_values) + 1)

        # Use game index as x-axis
        x_vals = list(range(len(cumulative_clv)))

        headline = backtest_results.headline_clv.get(target)
        headline_str = f" (Mean: {headline:+.4f})" if headline is not None else ""

        fig.add_trace(go.Scatter(
            x=x_vals,
            y=cumulative_clv.tolist(),
            mode="lines",
            name=f"{target.upper()}{headline_str}",
            line={"color": _get_target_color(target), "width": 2},
        ))

        # Add season boundary markers
        if "season" in clv_sorted.columns:
            seasons = clv_sorted["season"].values
            for i in range(1, len(seasons)):
                if seasons[i] != seasons[i - 1]:
                    fig.add_vline(
                        x=i,
                        line_dash="dot",
                        line_color="#ccc",
                        line_width=1,
                    )

    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="#999", line_width=1)

    fig.update_layout(
        title={"text": "Cumulative CLV Over Time", "x": 0.5},
        xaxis_title="Game Index (Chronological)",
        yaxis_title="Cumulative Mean CLV",
        template="plotly_white",
        legend={"x": 0.02, "y": 0.98},
        width=1000,
        height=500,
    )

    return fig


# ---------------------------------------------------------------------------
# Chart 3: Season Comparison Heatmap
# ---------------------------------------------------------------------------


def generate_season_heatmap(backtest_results: BacktestResults) -> go.Figure:
    """Generate season comparison heatmap across targets and metrics.

    Creates a heatmap per target showing metrics (rows = seasons,
    columns = metrics). Uses color coding: green = good, red = bad.

    Args:
        backtest_results: Complete backtest results from the engine.

    Returns:
        Plotly Figure with season heatmap subplots.
    """
    targets = backtest_results.config.targets
    n_targets = len(targets)

    fig = make_subplots(
        rows=1,
        cols=n_targets,
        subplot_titles=[t.upper() for t in targets],
        horizontal_spacing=0.08,
    )

    for col_idx, target in enumerate(targets, 1):
        seasons: list[int] = []
        metric_names: list[str] = []
        z_values: list[list[float]] = []
        text_values: list[list[str]] = []

        # Determine metrics based on target type
        if target == "wp":
            metric_names = ["accuracy", "brier_score", "ece", "log_loss"]
        elif target == "ats":
            metric_names = ["mae", "rmse"]
        else:  # ou
            metric_names = ["mae", "rmse"]

        for sr in backtest_results.season_results:
            if target not in sr.target_results:
                continue

            tr = sr.target_results[target]
            seasons.append(sr.season)

            row_values: list[float] = []
            row_text: list[str] = []

            for metric in metric_names:
                val = tr.metrics.get(metric)
                if val is not None:
                    row_values.append(float(val))
                    if metric == "accuracy":
                        row_text.append(f"{val:.1%}")
                    elif metric in ("brier_score", "ece", "log_loss"):
                        row_text.append(f"{val:.4f}")
                    else:
                        row_text.append(f"{val:.2f}")
                else:
                    row_values.append(0.0)
                    row_text.append("N/A")

            z_values.append(row_values)
            text_values.append(row_text)

        if not seasons:
            continue

        # Color scale: for error metrics (lower = better), use reversed scale
        colorscale = "RdYlGn"
        if target in ("ats", "ou"):
            colorscale = "RdYlGn_r"  # Reversed: green = low error

        fig.add_trace(
            go.Heatmap(
                z=z_values,
                x=metric_names,
                y=[str(s) for s in seasons],
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
        template="plotly_white",
        width=350 * max(n_targets, 1),
        height=400,
    )

    return fig


# ---------------------------------------------------------------------------
# Chart 4: Equity Curve
# ---------------------------------------------------------------------------


def generate_equity_chart(simulation_results: SimulationResults) -> go.Figure:
    """Generate betting simulation equity curves.

    Plots flat-stake and Kelly strategy bankroll values over time.
    Includes reference line at starting bankroll and max drawdown
    annotations.

    Args:
        simulation_results: Complete simulation results from BettingSimulator.

    Returns:
        Plotly Figure with equity curves.
    """
    fig = go.Figure()

    starting_bankroll = simulation_results.config.starting_bankroll

    # Flat-stake equity curve
    flat = simulation_results.flat_stake
    fig.add_trace(go.Scatter(
        x=list(range(len(flat.equity_curve))),
        y=flat.equity_curve,
        mode="lines",
        name=f"Flat Stake (ROI: {flat.roi:+.1%})",
        line={"color": "#667eea", "width": 2},
    ))

    # Kelly equity curve
    kelly = simulation_results.kelly
    fig.add_trace(go.Scatter(
        x=list(range(len(kelly.equity_curve))),
        y=kelly.equity_curve,
        mode="lines",
        name=f"Kelly (ROI: {kelly.roi:+.1%})",
        line={"color": "#f093fb", "width": 2},
    ))

    # Break-even reference line
    fig.add_hline(
        y=starting_bankroll,
        line_dash="dash",
        line_color="#999",
        line_width=1,
        annotation_text="Starting Bankroll",
        annotation_position="bottom right",
    )

    # Max drawdown annotations
    if flat.equity_curve:
        flat_peak = starting_bankroll
        flat_dd_idx = 0
        flat_dd_val = 0.0
        for i, val in enumerate(flat.equity_curve):
            flat_peak = max(flat_peak, val)
            dd = flat_peak - val
            if dd > flat_dd_val:
                flat_dd_val = dd
                flat_dd_idx = i

        if flat_dd_val > 0:
            fig.add_annotation(
                x=flat_dd_idx,
                y=flat.equity_curve[flat_dd_idx],
                text=f"Max DD: ${flat_dd_val:,.0f}",
                showarrow=True,
                arrowhead=2,
                font={"size": 10, "color": "#667eea"},
            )

    fig.update_layout(
        title={"text": "Betting Simulation: Flat-Stake vs Kelly", "x": 0.5},
        xaxis_title="Bet Number",
        yaxis_title="Bankroll ($)",
        template="plotly_white",
        legend={"x": 0.02, "y": 0.98},
        width=1000,
        height=500,
        yaxis_tickformat="$,.0f",
    )

    return fig


# ---------------------------------------------------------------------------
# Chart 5: Weekly Performance
# ---------------------------------------------------------------------------


def generate_weekly_performance_chart(
    backtest_results: BacktestResults,
    target: str,
    season: int,
) -> go.Figure:
    """Generate weekly performance trends for a given target and season.

    Plots week-by-week accuracy (bars) with cumulative CLV overlay (line).

    Args:
        backtest_results: Complete backtest results.
        target: Model target (wp/ats/ou).
        season: Season year to plot.

    Returns:
        Plotly Figure with weekly performance.
    """
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Find the season result
    season_result = None
    for sr in backtest_results.season_results:
        if sr.season == season and target in sr.target_results:
            season_result = sr
            break

    if season_result is None:
        fig.update_layout(title=f"No data for {target.upper()} {season}")
        return fig

    tr = season_result.target_results[target]
    preds_df = tr.predictions_df

    if preds_df.empty or "week" not in preds_df.columns:
        fig.update_layout(title=f"No weekly data for {target.upper()} {season}")
        return fig

    # Group by week
    weeks = sorted(preds_df["week"].unique())

    weekly_accuracy: list[float] = []
    weekly_clv: list[float] = []

    for week in weeks:
        week_data = preds_df[preds_df["week"] == week]

        # Accuracy: for WP, use model_prob > 0.5 vs actual
        if target == "wp" and "model_prob" in week_data.columns and "actual" in week_data.columns:
            correct = ((week_data["model_prob"] > 0.5).astype(int) == week_data["actual"].astype(int))
            weekly_accuracy.append(float(correct.mean()))
        else:
            weekly_accuracy.append(0.5)

        # CLV
        if "probability_clv" in week_data.columns:
            valid_clv = week_data["probability_clv"].dropna()
            weekly_clv.append(float(valid_clv.mean()) if len(valid_clv) > 0 else 0.0)
        else:
            weekly_clv.append(0.0)

    # Normalize weeks to season progress
    x_labels = [f"W{w}" for w in weeks]

    # Bar chart for weekly accuracy
    fig.add_trace(
        go.Bar(
            x=x_labels,
            y=weekly_accuracy,
            name="Weekly Accuracy",
            marker_color=_get_target_color(target),
            opacity=0.6,
        ),
        secondary_y=False,
    )

    # Line for cumulative CLV
    cumulative_clv = np.cumsum(weekly_clv) / np.arange(1, len(weekly_clv) + 1)
    fig.add_trace(
        go.Scatter(
            x=x_labels,
            y=cumulative_clv.tolist(),
            mode="lines+markers",
            name="Cumulative CLV",
            line={"color": "#333", "width": 2},
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title={"text": f"{target.upper()} Weekly Performance - {season}", "x": 0.5},
        template="plotly_white",
        width=900,
        height=400,
    )
    fig.update_yaxes(title_text="Accuracy", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative CLV", secondary_y=True)

    return fig


# ---------------------------------------------------------------------------
# BacktestReporter
# ---------------------------------------------------------------------------


class BacktestReporter:
    """Orchestrates chart generation and HTML report assembly.

    Takes BacktestResults and SimulationResults, generates all Plotly
    charts, builds a metrics context, and renders the Jinja2 template
    into a single-page interactive HTML dashboard.

    Usage::

        reporter = BacktestReporter(output_dir="outputs/backtest")
        report_path = reporter.generate(backtest_results, sim_results)
        print(f"Report at: {report_path}")
    """

    def __init__(self, output_dir: Path | str = "outputs/backtest") -> None:
        """Initialize the reporter with output directory and Jinja2 env.

        Args:
            output_dir: Directory for writing output files.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Jinja2 environment pointing to backtest/templates/
        templates_dir = Path(__file__).parent / "templates"
        self.env = Environment(
            loader=FileSystemLoader(str(templates_dir)),
            autoescape=False,
        )

    def _build_metrics_context(
        self,
        backtest_results: BacktestResults,
        simulation_results: SimulationResults,
    ) -> dict[str, Any]:
        """Build the template context dict with all metrics.

        Organizes headline CLV, per-season metrics, per-target summaries,
        simulation stats, COVID annotation, and era info for Jinja2.

        Args:
            backtest_results: Complete backtest results.
            simulation_results: Complete simulation results.

        Returns:
            Dict ready for Jinja2 template rendering.
        """
        # Per-season breakdown
        per_season: list[dict[str, Any]] = []
        for sr in backtest_results.season_results:
            season_data: dict[str, Any] = {
                "season": sr.season,
                "targets": {},
            }
            for target, tr in sr.target_results.items():
                season_data["targets"][target] = {
                    "metrics": tr.metrics,
                    "n_predictions": len(tr.predictions_df),
                    "feature_count": len(tr.feature_names),
                }
            per_season.append(season_data)

        # Per-target aggregated metrics
        per_target: dict[str, dict[str, Any]] = {}
        for target in backtest_results.config.targets:
            clv_df = backtest_results.all_clv.get(target)
            n_predictions = 0
            for sr in backtest_results.season_results:
                if target in sr.target_results:
                    n_predictions += len(sr.target_results[target].predictions_df)

            per_target[target] = {
                "headline_clv": backtest_results.headline_clv.get(target),
                "n_predictions": n_predictions,
                "has_clv": clv_df is not None and not clv_df.empty,
            }

        # Simulation summary
        flat = simulation_results.flat_stake
        kelly = simulation_results.kelly
        simulation_summary = {
            "flat_stake": {
                "total_bets": flat.total_bets,
                "winning_bets": flat.winning_bets,
                "losing_bets": flat.losing_bets,
                "push_bets": flat.push_bets,
                "win_rate": flat.win_rate,
                "total_wagered": flat.total_wagered,
                "net_profit": flat.net_profit,
                "roi": flat.roi,
                "final_bankroll": flat.final_bankroll,
                "max_drawdown": flat.max_drawdown,
                "max_drawdown_pct": flat.max_drawdown_pct,
            },
            "kelly": {
                "total_bets": kelly.total_bets,
                "winning_bets": kelly.winning_bets,
                "losing_bets": kelly.losing_bets,
                "push_bets": kelly.push_bets,
                "win_rate": kelly.win_rate,
                "total_wagered": kelly.total_wagered,
                "net_profit": kelly.net_profit,
                "roi": kelly.roi,
                "final_bankroll": kelly.final_bankroll,
                "max_drawdown": kelly.max_drawdown,
                "max_drawdown_pct": kelly.max_drawdown_pct,
            },
            "starting_bankroll": simulation_results.config.starting_bankroll,
            "by_target": simulation_results.by_target,
            "by_season": simulation_results.by_season,
        }

        return {
            "headline_clv": backtest_results.headline_clv,
            "per_season": per_season,
            "per_target": per_target,
            "simulation": simulation_summary,
            "covid": backtest_results.covid_annotation,
            "era": backtest_results.era_info,
            "odds_coverage": backtest_results.odds_coverage,
            "config": {
                "holdout_seasons": backtest_results.config.holdout_seasons,
                "targets": backtest_results.config.targets,
                "first_data_season": backtest_results.config.first_data_season,
            },
            "generated_at": datetime.now(tz=UTC).strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            ),
        }

    def _generate_charts(
        self,
        backtest_results: BacktestResults,
        simulation_results: SimulationResults,
    ) -> dict[str, str]:
        """Generate all Plotly charts and convert to HTML div strings.

        Args:
            backtest_results: Complete backtest results.
            simulation_results: Complete simulation results.

        Returns:
            Dict mapping chart name to HTML div string.
        """
        charts: dict[str, str] = {}

        # Calibration chart
        try:
            cal_fig = generate_calibration_chart(backtest_results)
            charts["calibration"] = cal_fig.to_html(
                full_html=False, include_plotlyjs=False, div_id="chart-calibration"
            )
        except (ValueError, KeyError, TypeError, AttributeError):
            logger.warning("Failed to generate calibration chart", exc_info=True)
            charts["calibration"] = "<p>Calibration chart unavailable</p>"

        # Cumulative CLV chart
        try:
            clv_fig = generate_clv_chart(backtest_results)
            charts["clv_cumulative"] = clv_fig.to_html(
                full_html=False, include_plotlyjs=False, div_id="chart-clv"
            )
        except (ValueError, KeyError, TypeError, AttributeError):
            logger.warning("Failed to generate CLV chart", exc_info=True)
            charts["clv_cumulative"] = "<p>CLV chart unavailable</p>"

        # Season heatmap
        try:
            heatmap_fig = generate_season_heatmap(backtest_results)
            charts["season_heatmap"] = heatmap_fig.to_html(
                full_html=False, include_plotlyjs=False, div_id="chart-heatmap"
            )
        except (ValueError, KeyError, TypeError, AttributeError):
            logger.warning("Failed to generate season heatmap", exc_info=True)
            charts["season_heatmap"] = "<p>Season heatmap unavailable</p>"

        # Equity curve
        try:
            equity_fig = generate_equity_chart(simulation_results)
            charts["equity_curve"] = equity_fig.to_html(
                full_html=False, include_plotlyjs=False, div_id="chart-equity"
            )
        except (ValueError, KeyError, TypeError, AttributeError):
            logger.warning("Failed to generate equity chart", exc_info=True)
            charts["equity_curve"] = "<p>Equity chart unavailable</p>"

        return charts

    def generate(
        self,
        backtest_results: BacktestResults,
        simulation_results: SimulationResults,
    ) -> Path:
        """Generate the full backtest HTML report.

        Builds metrics context, generates all charts, renders the Jinja2
        template, and writes the HTML file.

        Args:
            backtest_results: Complete backtest results from the engine.
            simulation_results: Complete simulation results from BettingSimulator.

        Returns:
            Path to the generated HTML report file.
        """
        logger.info("Generating backtest report", output_dir=str(self.output_dir))

        # Build context and charts
        metrics = self._build_metrics_context(backtest_results, simulation_results)
        charts = self._generate_charts(backtest_results, simulation_results)

        # Render template
        template = self.env.get_template("backtest_report.html")
        html_content = template.render(
            charts=charts,
            metrics=metrics,
        )

        # Write output
        output_path = self.output_dir / "backtest_report.html"
        output_path.write_text(html_content, encoding="utf-8")

        logger.info("Report generated", path=str(output_path))
        return output_path
