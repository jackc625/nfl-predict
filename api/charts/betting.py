"""Phase 17 Betting Dashboard chart generators.

Every statistical formula used by these charts lives in
:mod:`api.betting_metrics` — this module is a pure Plotly rendering layer built
on top of those primitives so the per-scope KPI/ROI-table JSON-blob builder
(:mod:`api.charts.prerender`) and the chart generators stay consistent by
construction (the single-source rule the Phase 16 twin established for
:mod:`api.charts.insights` / :mod:`api.insights_metrics`).

Exported symbols (re-exported from the package root ``api.charts``):

* :data:`BETTING_CHART_IDS` — authoritative tuple of 24 chart_id keys
  (12 chart bases x 2 scope variants ``all`` / ``recommended``).
* :func:`filter_scope`, :func:`compute_kpis`, :func:`compute_roi_table` —
  thin re-exports of the :mod:`api.betting_metrics` functions so the prerender
  loop can reach all betting math through this one module.
* :func:`generate_betting_equity_chart` — main flat/Kelly chronological equity.
* :func:`generate_betting_equity_mini_wp` / ``_ats`` / ``_ou`` — per-type minis.
* :func:`generate_betting_roi_type` / ``_season`` / ``_bucket`` — grouped ROI bars.
* :func:`generate_betting_edge_hist_wp` / ``_ats`` / ``_ou`` — outcome-colored
  edge histograms.

UIAP-01 COMPLIANCE
------------------
Imports only from :mod:`api.charts.core`, :mod:`api.betting_metrics`, stdlib,
``numpy``, and ``plotly``. No imports from ``models``, ``features``, or
``ratings``, and the ``BettingSimulator`` class is never named. Palettes come
from :mod:`api.charts.theme` (the dark dashboard theme), so no ``backtest.*``
module is imported here.

PLOTLY CDN COMPATIBILITY (RESEARCH Pitfall 5)
---------------------------------------------
Only figure constructs the shipped Phase 16 / dashboard charts already use are
used here (``go.Scatter`` / ``go.Bar`` / ``go.Histogram`` / ``add_vline`` /
``add_hline`` / ``barmode`` group+overlay) so the server-side Plotly 6.3.1
output renders against the pinned CDN client (2.35.2).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import plotly.graph_objects as go

from api.betting_metrics import (
    BREAKEVEN_WIN_RATE,
    STARTING_BANKROLL,
    _num,
    compute_kpis,
    compute_roi_table,
    edge_bucket_for,
    filter_scope,
)
from api.charts.core import (
    _apply_layout_defaults,
    _empty_chart_div,
    _get_target_color,
    _to_html,
)

# Palette: the dark dashboard theme's names, used directly (no local alias to drift). Flat and
# Kelly are staking strategies, not outcomes, so they take STRATEGY_COLORS; WIN_COLOR and
# LOSS_COLOR appear only on the edge histograms' realised win/loss series.
from api.charts.theme import (
    BOUNDARY_LINE,
    LOSS_COLOR,
    REFERENCE_LINE,
    STRATEGY_COLORS,
    WIN_COLOR,
)

logger = logging.getLogger(__name__)

# Re-export the betting-metrics functions through this module so the prerender
# loop can reach every betting computation via ``api.charts.betting`` (the
# single import surface the orchestration patches and calls). ``BREAKEVEN_WIN_RATE``
# is imported for the win-rate honesty reference and to keep the metrics
# contract visible from the chart layer.
__all__ = [
    "BETTING_CHART_IDS",
    "compute_kpis",
    "compute_roi_table",
    "filter_scope",
    "generate_betting_edge_hist_ats",
    "generate_betting_edge_hist_ou",
    "generate_betting_edge_hist_wp",
    "generate_betting_equity_chart",
    "generate_betting_equity_mini_ats",
    "generate_betting_equity_mini_ou",
    "generate_betting_equity_mini_wp",
    "generate_betting_roi_bucket",
    "generate_betting_roi_season",
    "generate_betting_roi_type",
]

_ = BREAKEVEN_WIN_RATE  # honesty reference constant (kept in the contract surface)


# ---------------------------------------------------------------------------
# Authoritative chart-id tuple (must match tests/api/conftest.py BETTING_CHART_IDS)
# ---------------------------------------------------------------------------
# Naming convention: ``betting_<chart>_<scope>`` for scope in ("all",
# "recommended"). 12 chart bases x 2 scopes = 24 ids (D-20). The two JSON-blob
# bases (kpis, roi_table) carry decodable JSON rather than chart HTML; their ids
# are part of this tuple so the prerender self-check covers them too.

_BETTING_CHART_BASES: tuple[str, ...] = (
    "betting_equity",
    "betting_equity_mini_wp",
    "betting_equity_mini_ats",
    "betting_equity_mini_ou",
    "betting_roi_type",
    "betting_roi_season",
    "betting_roi_bucket",
    "betting_edge_hist_wp",
    "betting_edge_hist_ats",
    "betting_edge_hist_ou",
    "betting_kpis",
    "betting_roi_table",
)
BETTING_SCOPES: tuple[str, ...] = ("all", "recommended")
BETTING_CHART_IDS: tuple[str, ...] = tuple(
    f"{base}_{scope}" for base in _BETTING_CHART_BASES for scope in BETTING_SCOPES
)
assert len(BETTING_CHART_IDS) == 24  # 12 chart bases x 2 scopes


# Empty-state copy shared by every generator (matches _safe_render fallback +
# the test contract: "Chart unavailable" OR "No betting").
_EMPTY_MSG = "No betting data available"

# Display order of edge buckets along the ROI-by-bucket x-axis.
_BUCKET_ORDER: tuple[str, str, str] = ("small", "medium", "big")


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _bets_for(rows: Sequence[dict], target: str) -> list[dict]:
    """Return only the rows for a given bet *target* (wp / ats / ou)."""
    return [r for r in rows if r.get("target") == target]


def _sorted_chronologically(rows: Sequence[dict]) -> list[dict]:
    """Sort rows by (season, week, game_id, target).

    NOTE (RESEARCH Pitfall 4): the CSV carries no per-bet kickoff timestamp, so
    within a (season, week) the ordering is a stable secondary sort on
    (game_id, target), NOT true chronology. D-07 explicitly accepts season->week
    ordering as sufficient; the true per-bet date axis is deferred.
    """
    return sorted(
        rows,
        key=lambda r: (
            r.get("season") or 0,
            r.get("week") or 0,
            str(r.get("game_id") or ""),
            str(r.get("target") or ""),
        ),
    )


def _cumulative_equity(rows: Sequence[dict], payout_col: str) -> list[float]:
    """Return the cumulative bankroll series starting at STARTING_BANKROLL.

    Each row's ``payout_col`` (net profit/loss, already net of stake in the CSV)
    is added in the supplied order. The returned list has one point per row.
    """
    equity = STARTING_BANKROLL
    series: list[float] = []
    for r in rows:
        equity += _num(r.get(payout_col))
        series.append(equity)
    return series


def _add_season_boundary_vlines(fig: go.Figure, ordered_rows: Sequence[dict]) -> None:
    """Add dotted season-boundary vlines (idiom from generate_dashboard_clv_chart).

    A vline is drawn at every index where the season changes from the previous
    row, mirroring ``api/charts/core.py:267-283``.
    """
    for i in range(1, len(ordered_rows)):
        if ordered_rows[i].get("season") != ordered_rows[i - 1].get("season"):
            fig.add_vline(x=i, line_dash="dot", line_color=BOUNDARY_LINE, line_width=1)


def _add_starting_bankroll_hline(fig: go.Figure) -> None:
    """Add the dashed $10,000 starting-bankroll reference line (D-07 / D-18)."""
    fig.add_hline(
        y=STARTING_BANKROLL,
        line_dash="dash",
        line_color=REFERENCE_LINE,
        line_width=1,
        annotation_text="Starting Bankroll",
        annotation_position="bottom right",
    )


# ---------------------------------------------------------------------------
# 1. Main equity chart (D-07) — flat vs Kelly, chronological, season boundaries
# ---------------------------------------------------------------------------


def generate_betting_equity_chart(rows: Sequence[dict]) -> str:
    """Render the main cumulative-bankroll equity chart (flat-stake vs Kelly).

    Two ``go.Scatter`` lines (Flat / Kelly in the theme's strategy colours) over a chronological
    season->week index, with dotted season-boundary vlines and a dashed $10,000
    "Starting Bankroll" reference line. X-axis ordering follows
    :func:`_sorted_chronologically` (within-week order is not true chronology;
    RESEARCH Pitfall 4 / D-07).
    """
    if not rows:
        return _empty_chart_div(_EMPTY_MSG)

    ordered = _sorted_chronologically(rows)
    flat_equity = _cumulative_equity(ordered, "payout_flat")
    kelly_equity = _cumulative_equity(ordered, "payout_kelly")
    x_vals = list(range(len(ordered)))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=flat_equity,
            mode="lines",
            name="Flat",
            line={"color": STRATEGY_COLORS["flat_stake"], "width": 2},
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=kelly_equity,
            mode="lines",
            name="Kelly",
            line={"color": STRATEGY_COLORS["kelly"], "width": 2},
        ),
    )

    _add_season_boundary_vlines(fig, ordered)
    _add_starting_bankroll_hline(fig)

    fig.update_layout(
        title={"text": "Equity Curve: Flat-Stake vs Kelly", "x": 0.5},
        xaxis_title="Bet Index (chronological by season then week)",
        yaxis_title="Bankroll ($)",
        yaxis_tickformat="$,.0f",
    )
    _apply_layout_defaults(fig)
    return _to_html(fig)


# ---------------------------------------------------------------------------
# 2. Per-type mini equity charts (D-08) — flat-stake sufficient
# ---------------------------------------------------------------------------


def _generate_equity_mini(rows: Sequence[dict], target: str) -> str:
    """Render a per-type mini equity chart (flat-stake) colored by target.

    Flat-stake is sufficient for the minis (D-08); the line is colored by the
    target's palette entry via :func:`_get_target_color`.
    """
    type_rows = _bets_for(rows, target)
    if not type_rows:
        return _empty_chart_div(f"No betting data for {target.upper()}")

    ordered = _sorted_chronologically(type_rows)
    flat_equity = _cumulative_equity(ordered, "payout_flat")
    x_vals = list(range(len(ordered)))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=flat_equity,
            mode="lines",
            name="Flat",
            line={"color": _get_target_color(target), "width": 2},
        ),
    )

    _add_season_boundary_vlines(fig, ordered)
    _add_starting_bankroll_hline(fig)

    fig.update_layout(
        title={"text": f"{target.upper()} Equity (Flat)", "x": 0.5},
        xaxis_title="Bet Index",
        yaxis_title="Bankroll ($)",
        yaxis_tickformat="$,.0f",
        showlegend=False,
        height=320,
    )
    _apply_layout_defaults(fig)
    return _to_html(fig)


def generate_betting_equity_mini_wp(rows: Sequence[dict]) -> str:
    return _generate_equity_mini(rows, "wp")


def generate_betting_equity_mini_ats(rows: Sequence[dict]) -> str:
    return _generate_equity_mini(rows, "ats")


def generate_betting_equity_mini_ou(rows: Sequence[dict]) -> str:
    return _generate_equity_mini(rows, "ou")


# ---------------------------------------------------------------------------
# 3. ROI grouped-bar charts (D-10 / D-11 / D-18) — flat vs Kelly + 0% baseline
# ---------------------------------------------------------------------------


def _roi_grouped_bar(
    categories: list[str],
    flat_rois: list[float],
    kelly_rois: list[float],
    *,
    title: str,
    xaxis_title: str,
) -> str:
    """Build a grouped flat-vs-Kelly ROI bar chart with a 0% baseline.

    Two ``go.Bar`` traces (Flat / Kelly in the theme's strategy colours),
    ``barmode="group"``, and a dashed ``REFERENCE_LINE`` ``add_hline`` at y=0 (the 0%
    ROI baseline, D-18).
    """
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            name="Flat",
            x=categories,
            y=flat_rois,
            marker_color=STRATEGY_COLORS["flat_stake"],
        ),
    )
    fig.add_trace(
        go.Bar(
            name="Kelly",
            x=categories,
            y=kelly_rois,
            marker_color=STRATEGY_COLORS["kelly"],
        ),
    )
    # 0% ROI baseline (profit above, loss below) — D-18 honesty reference.
    fig.add_hline(y=0, line_dash="dash", line_color=REFERENCE_LINE, line_width=1)
    fig.update_layout(
        title={"text": title, "x": 0.5},
        barmode="group",
        xaxis_title=xaxis_title,
        yaxis_title="ROI (%)",
    )
    _apply_layout_defaults(fig)
    return _to_html(fig)


def _roi_rows_by_kind(rows: Sequence[dict], slice_kind: str) -> list[dict]:
    """Return the :func:`compute_roi_table` rows for a single slice kind."""
    return [r for r in compute_roi_table(rows) if r.get("slice_kind") == slice_kind]


def generate_betting_roi_type(rows: Sequence[dict]) -> str:
    """ROI grouped bars by bet type (wp / ats / ou)."""
    if not rows:
        return _empty_chart_div(_EMPTY_MSG)
    type_rows = _roi_rows_by_kind(rows, "type")
    if not type_rows:
        return _empty_chart_div(_EMPTY_MSG)
    categories = [str(r["label"]).upper() for r in type_rows]
    flat_rois = [float(r["roi_flat"]) for r in type_rows]
    kelly_rois = [float(r["roi_kelly"]) for r in type_rows]
    return _roi_grouped_bar(
        categories,
        flat_rois,
        kelly_rois,
        title="ROI by Bet Type",
        xaxis_title="Bet Type",
    )


def generate_betting_roi_season(rows: Sequence[dict]) -> str:
    """ROI grouped bars by season (ascending)."""
    if not rows:
        return _empty_chart_div(_EMPTY_MSG)
    season_rows = _roi_rows_by_kind(rows, "season")
    if not season_rows:
        return _empty_chart_div(_EMPTY_MSG)
    categories = [str(r["label"]) for r in season_rows]
    flat_rois = [float(r["roi_flat"]) for r in season_rows]
    kelly_rois = [float(r["roi_kelly"]) for r in season_rows]
    return _roi_grouped_bar(
        categories,
        flat_rois,
        kelly_rois,
        title="ROI by Season",
        xaxis_title="Season",
    )


def generate_betting_roi_bucket(rows: Sequence[dict]) -> str:
    """ROI grouped bars by per-type edge bucket (small / medium / big).

    Buckets are labeled ``<target>:<bucket>`` (per-type, D-12) and ordered by
    (target, small->medium->big). The slice math comes from
    :func:`compute_roi_table` (slice_kind == "bucket"); :func:`edge_bucket_for`
    is imported as the shared bucketing source of truth.
    """
    if not rows:
        return _empty_chart_div(_EMPTY_MSG)
    bucket_rows = _roi_rows_by_kind(rows, "bucket")
    if not bucket_rows:
        return _empty_chart_div(_EMPTY_MSG)

    # compute_roi_table already emits buckets in (target, small->medium->big)
    # order; preserve it. ``edge_bucket_for`` is referenced to keep the chart
    # layer bound to the same per-type bucketing contract used to build labels.
    _ = edge_bucket_for
    categories = [str(r["label"]).upper() for r in bucket_rows]
    flat_rois = [float(r["roi_flat"]) for r in bucket_rows]
    kelly_rois = [float(r["roi_kelly"]) for r in bucket_rows]
    return _roi_grouped_bar(
        categories,
        flat_rois,
        kelly_rois,
        title="ROI by Edge Bucket",
        xaxis_title="Edge Bucket (per type)",
    )


# ---------------------------------------------------------------------------
# 4. Per-type edge histograms (D-13 / D-14) — overlaid win/loss, pushes excluded
# ---------------------------------------------------------------------------

# Per-type honest x-axis titles (D-13: never share an axis across types).
_EDGE_AXIS_TITLE: dict[str, str] = {
    "wp": "Edge (probability)",
    "ats": "Edge (points)",
    "ou": "Edge (points)",
}


def _generate_edge_hist(rows: Sequence[dict], target: str) -> str:
    """Render a per-type edge histogram, overlaid win (green) / loss (red).

    Two ``go.Histogram`` traces using identity checks (``outcome is True`` /
    ``is False``); pushes (``outcome is None``) are excluded (D-14). Rendered
    with ``barmode="overlay"`` + ``opacity=0.7`` and an honest per-type x-axis
    title (D-13 — no shared axis).
    """
    type_rows = _bets_for(rows, target)
    if not type_rows:
        return _empty_chart_div(f"No betting data for {target.upper()}")

    won = [_num(r.get("edge")) for r in type_rows if r.get("outcome") is True]
    lost = [_num(r.get("edge")) for r in type_rows if r.get("outcome") is False]

    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=won,
            name="Win",
            marker_color=WIN_COLOR,
            opacity=0.7,
        ),
    )
    fig.add_trace(
        go.Histogram(
            x=lost,
            name="Loss",
            marker_color=LOSS_COLOR,
            opacity=0.7,
        ),
    )
    fig.update_layout(
        title={"text": f"{target.upper()} Edge Distribution", "x": 0.5},
        barmode="overlay",
        xaxis_title=_EDGE_AXIS_TITLE.get(target, "Edge"),
        yaxis_title="Count",
        height=320,
    )
    _apply_layout_defaults(fig)
    return _to_html(fig)


def generate_betting_edge_hist_wp(rows: Sequence[dict]) -> str:
    return _generate_edge_hist(rows, "wp")


def generate_betting_edge_hist_ats(rows: Sequence[dict]) -> str:
    return _generate_edge_hist(rows, "ats")


def generate_betting_edge_hist_ou(rows: Sequence[dict]) -> str:
    return _generate_edge_hist(rows, "ou")
