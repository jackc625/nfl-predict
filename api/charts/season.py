"""Phase 18 Season Tracking chart generators.

Every statistical formula used by these charts lives in
:mod:`api.season_metrics` — this module is a pure Plotly rendering layer built
on top of those primitives so the per-season KPI JSON-blob builder
(:mod:`api.charts.prerender`) and the chart generators stay consistent by
construction (the single-source rule the Phase 16/17 twins established for
:mod:`api.charts.insights` / :mod:`api.insights_metrics` and
:mod:`api.charts.betting` / :mod:`api.betting_metrics`).

Exported symbols (re-exported from the package root ``api.charts``):

* :func:`generate_season_cumulative` — one running-hit-rate line per target
  (WP/ATS/OU) on a shared 0-100% axis with the 50% coin-flip and 52.4%
  -110-breakeven reference lines (DASH-07).
* :func:`generate_season_weekly` — per-week hit-rate markers per target PLUS a
  rolling-average overlay line in the same target color (DASH-08).
* :func:`compute_season_kpis`, :func:`compute_cumulative_series`,
  :func:`compute_weekly_series`, :func:`compute_weekly_records` — thin re-exports of the
  :mod:`api.season_metrics` functions so the prerender loop can reach all
  season math through this one module (the single import surface
  ``unittest.mock.patch`` targets and calls).

NAMING / ID SHAPE (CONTEXT D-12 / D-13, RESEARCH Pitfall 1)
-----------------------------------------------------------
:data:`_SEASON_CHART_BASES` is a FIXED bases tuple
(``season_cumulative`` / ``season_weekly`` / ``season_kpis``). Unlike the
betting layer (a fixed Cartesian product over exactly two scopes forever), the
season dimension is DATA-DEPENDENT: the full ``<base>_<season>`` id set is
derived at runtime from ``SELECT DISTINCT season FROM predictions`` in
:mod:`api.charts.prerender`. NO literal year appears here — ingesting 2025
produces ``season_*_2025`` automatically with zero code change (D-01).

UIAP-01 COMPLIANCE
------------------
Imports only from :mod:`api.charts.core`, :mod:`api.season_metrics`, stdlib, and
``plotly``. No imports from ``models``, ``features``, or ``ratings``. Palettes
come through :mod:`api.charts.core` (which sources ``backtest.report`` — the
only permitted ``backtest.*`` import in the chart layer), so the three target
lines use ``_get_target_color`` rather than fresh hex literals (D-13).

PLOTLY CDN COMPATIBILITY (RESEARCH Pitfall: 6.6.0 server / 2.35.2 client)
-------------------------------------------------------------------------
Only figure constructs the shipped Phase 16 / 17 / dashboard charts already use
are used here (``go.Scatter`` lines+markers / ``add_hline`` / ``update_layout``)
so the server-side Plotly output renders against the pinned CDN client.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import plotly.graph_objects as go

from api.charts.core import (
    _apply_layout_defaults,
    _empty_chart_div,
    _get_target_color,
    _to_html,
)
from api.season_metrics import (
    BREAKEVEN_WIN_RATE,
    compute_cumulative_series,
    compute_season_kpis,
    compute_weekly_records,
    compute_weekly_series,
)

logger = logging.getLogger(__name__)

# Re-export the season-metrics functions through this module so the prerender
# loop can reach every season computation via ``api.charts.season`` (the single
# import surface the orchestration patches and calls). ``BREAKEVEN_WIN_RATE`` is
# imported for the 52.4% reference line and to keep the metrics contract visible
# from the chart layer.
__all__ = [
    "compute_cumulative_series",
    "compute_season_kpis",
    "compute_weekly_records",
    "compute_weekly_series",
    "generate_season_cumulative",
    "generate_season_weekly",
]


# ---------------------------------------------------------------------------
# Fixed chart bases (NOT a Cartesian product over hardcoded years — Pitfall 1)
# ---------------------------------------------------------------------------
# The full id set (``<base>_<season>``) is derived at runtime from the seasons
# present in ``predictions``; see api/charts/prerender.py. This tuple only names
# the three per-season bases, analogous to ``_BETTING_CHART_BASES`` but WITHOUT
# binding any year.
_SEASON_CHART_BASES: tuple[str, ...] = (
    "season_cumulative",
    "season_weekly",
    "season_kpis",
)

# Canonical target order for the three hit-rate series (matches
# api.season_metrics._TARGETS).
_TARGETS: tuple[str, ...] = ("wp", "ats", "ou")

# Human-readable legend labels per target (axis/legend copy, not a new palette).
_TARGET_LABEL: dict[str, str] = {
    "wp": "Winner",
    "ats": "Spread",
    "ou": "Totals",
}

# Empty-state copy shared by every generator (matches the _safe_render fallback
# + the test contract: "Chart unavailable" OR "No season").
_EMPTY_MSG = "No season data available"

# 52.4% -110 breakeven as a percentage (BREAKEVEN_WIN_RATE is the 0..1 fraction).
_BREAKEVEN_PCT: float = BREAKEVEN_WIN_RATE * 100.0


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _add_reference_lines(fig: go.Figure) -> None:
    """Add the 50% coin-flip and 52.4% -110-breakeven dashed reference lines.

    The idiom is copied from ``api/charts/core.py`` /
    ``api/charts/betting.py`` (dashed ``#999`` ``add_hline`` with an annotation);
    ``#999`` is the only inline color literal, matching the core reference-line
    contract (D-06). The 52.4% line is exact for spread/totals; moneyline (WP)
    odds vary, so the caption near the chart frames it as a reference.
    """
    fig.add_hline(
        y=50,
        line_dash="dash",
        line_color="#999",
        line_width=1,
        annotation_text="50% coin flip",
        annotation_position="bottom right",
    )
    fig.add_hline(
        y=_BREAKEVEN_PCT,
        line_dash="dash",
        line_color="#999",
        line_width=1,
        annotation_text="-110 breakeven (~52.4%)",
        annotation_position="top right",
    )


# ---------------------------------------------------------------------------
# 1. Cumulative running hit-rate (DASH-07) — one line per target, [0,100] axis
# ---------------------------------------------------------------------------


def generate_season_cumulative(rows: Sequence[dict]) -> str:
    """Render the cumulative season-to-date hit-rate chart (one line per target).

    Three ``go.Scatter(mode="lines")`` traces (Winner / Spread / Totals), each
    colored by :func:`_get_target_color`, plotted against the chronological week
    index within the season on a shared ``yaxis`` range ``[0, 100]``. The 50% +
    52.4% reference lines are always drawn (D-06 / D-07 §1). ALL hit-rate math
    is delegated to :func:`compute_cumulative_series` — there are no formulas in
    this layer.

    The running cumulative mean IS the DASH-07 trend line (where the season has
    settled so far). A target with no decided games contributes no trace.
    """
    if not rows:
        return _empty_chart_div(_EMPTY_MSG)

    series = compute_cumulative_series(rows)

    fig = go.Figure()
    has_data = False
    for target in _TARGETS:
        target_series = series.get(target, {})
        values = target_series.get("values", [])
        if not values:
            continue
        # X axis = chronological week index within the season (one point per
        # decided game, in week order). Using a 1-based index keeps the
        # trajectory monotonic in "games played" while the weeks list stays
        # available for hover context.
        weeks = target_series.get("weeks", [])
        x_vals = list(range(1, len(values) + 1))
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=values,
                mode="lines",
                name=_TARGET_LABEL.get(target, target.upper()),
                line={"color": _get_target_color(target), "width": 2},
                customdata=weeks,
                hovertemplate=(
                    f"{_TARGET_LABEL.get(target, target.upper())}"
                    "<br>Game %{x} (Week %{customdata})"
                    "<br>Hit rate: %{y:.1f}%<extra></extra>"
                ),
            ),
        )
        has_data = True

    if not has_data:
        return _empty_chart_div(_EMPTY_MSG)

    _add_reference_lines(fig)

    fig.update_layout(
        title={"text": "Cumulative Hit Rate", "x": 0.5},
        xaxis_title="Games Played (chronological by week)",
        yaxis_title="Hit Rate (%)",
        yaxis={"range": [0, 100]},
        legend={"x": 0.02, "y": 0.98},
    )
    _apply_layout_defaults(fig)
    return _to_html(fig)


# ---------------------------------------------------------------------------
# 2. Weekly hit-rate + rolling-average overlay (DASH-08)
# ---------------------------------------------------------------------------


def generate_season_weekly(rows: Sequence[dict]) -> str:
    """Render the per-week hit-rate chart with a rolling-average overlay.

    For each target: a raw per-week ``go.Scatter(mode="markers")`` mark PLUS a
    rolling-average ``go.Scatter(mode="lines", line={"dash": "dot"})`` overlay in
    the SAME target color (reduced opacity) so the overlay reads as a smoothing
    of the same series, not a fourth target (UI-SPEC D-07 §2). Shared ``yaxis``
    range ``[0, 100]`` with the 50% + 52.4% reference lines. ALL hit-rate +
    rolling math is delegated to :func:`compute_weekly_series`.

    Markers (not bars) are chosen so three overlaid targets plus their rolling
    overlays do not crowd into an unreadable grouped-bar cluster on the thin
    ~14-16-game weekly samples (D-07 discretion). The rolling window is owned by
    :mod:`api.season_metrics` (3 weeks).
    """
    if not rows:
        return _empty_chart_div(_EMPTY_MSG)

    series = compute_weekly_series(rows)

    fig = go.Figure()
    has_data = False
    for target in _TARGETS:
        target_series = series.get(target, {})
        weeks = target_series.get("weeks", [])
        values = target_series.get("values", [])
        rolling = target_series.get("rolling", [])
        if not weeks or not values:
            continue
        color = _get_target_color(target)
        label = _TARGET_LABEL.get(target, target.upper())

        # Raw per-week hit-rate markers.
        fig.add_trace(
            go.Scatter(
                x=weeks,
                y=values,
                mode="markers",
                name=label,
                marker={"color": color, "size": 8},
                hovertemplate=(
                    f"{label}<br>Week %{{x}}<br>Hit rate: %{{y:.1f}}%<extra></extra>"
                ),
            ),
        )
        # Rolling-average overlay (same color, dashed, reduced opacity).
        if rolling:
            fig.add_trace(
                go.Scatter(
                    x=weeks,
                    y=rolling,
                    mode="lines",
                    name=f"{label} (3-wk avg)",
                    line={"color": color, "width": 2, "dash": "dot"},
                    opacity=0.6,
                    hovertemplate=(
                        f"{label} 3-wk avg<br>Week %{{x}}<br>%{{y:.1f}}%<extra></extra>"
                    ),
                ),
            )
        has_data = True

    if not has_data:
        return _empty_chart_div(_EMPTY_MSG)

    _add_reference_lines(fig)

    fig.update_layout(
        title={"text": "Weekly Hit Rate", "x": 0.5},
        xaxis_title="Week",
        yaxis_title="Hit Rate (%)",
        yaxis={"range": [0, 100]},
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.25},
    )
    _apply_layout_defaults(fig)
    return _to_html(fig)
