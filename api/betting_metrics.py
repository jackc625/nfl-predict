"""Shared pure-function metrics for the Betting Dashboard page.

Single source of truth for every betting statistic used by both the chart
generators (:mod:`api.charts.betting`) and the per-scope KPI/ROI-table JSON-blob
builders (:mod:`api.charts.prerender`). Do NOT duplicate any of these formulas
elsewhere — the chart layer and the JSON-blob builder both import from here so
they cannot drift (the "single-source" rule the Phase 16 twin established for
:mod:`api.insights_metrics`).

Contents
--------
* Scope filter — :func:`filter_scope` (``recommended`` == ``kelly_stake > 0``).
* KPI scoreboard — :func:`compute_kpis` (7 scope-aware values; win-rate excludes
  pushes; max-drawdown is peak-to-trough on the cumulative equity series).
* ROI-by-slice — :func:`compute_roi_table` (per-slice formatted rows with a
  tri-state ``roi_favorable`` flag; the single-source builder both the JSON-blob
  builder and any table-shaped chart call).
* Per-type edge buckets — :func:`edge_bucket_for` (WP by ``|edge|``; ATS/OU by
  points).
* Private helpers — :func:`_max_drawdown`, plus ROI / win-rate primitives.

UIAP-01 COMPLIANCE
------------------
This module imports only stdlib and ``numpy``. No imports from ``models``,
``features``, ``ratings``, or ``api.*`` — keeping api-side formula reuse free of
circular import risk and satisfying the UIAP-01 import guard. The locked
constants below are copied *values* (with a source-citing comment), never
imports of ``backtest.simulation`` / ``utils.kelly_criterion`` / ``models.blending``.

OUTCOME PARSING (HIGH-severity pitfall)
---------------------------------------
``outcome`` is a nullable boolean: ``True`` (win), ``False`` (loss), ``None``
(push). Every classification uses identity checks — ``outcome is True`` /
``is False`` / ``is None`` — NEVER ``bool(outcome)``: ``bool("False") == True``
and ``bool(float("nan")) == True`` would both silently inflate the win rate.
The win-rate denominator is ``wins + losses`` (pushes excluded).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

# ---------------------------------------------------------------------------
# Locked constants
# ---------------------------------------------------------------------------

# Source: backtest.simulation.DEFAULT_STARTING_BANKROLL (NOT imported — UIAP-01).
# The betting simulation tracks a pretend bankroll starting at $10,000; the
# equity curve and the final-bankroll / max-drawdown KPIs are anchored here.
STARTING_BANKROLL: float = 10_000.0

# Standard -110 break-even win rate (110 / 210). Exact for spread/totals (D-18);
# moneyline (WP) odds vary, so this is presented as a reference line, not a
# per-bet truth. Defined as a copied value (NOT imported — UIAP-01).
BREAKEVEN_WIN_RATE: float = 0.524

# Per-type edge-bucket cut points (D-12). Edge means different things per bet
# type, so cuts are per target: WP edge is a probability delta (signed; bucketed
# by magnitude here), ATS/OU edge is points. Defaults are the observed-percentile
# terciles from 17-RESEARCH.md Pitfall 3 (WP by |edge|: <0.03 / 0.03-0.08 / >0.08;
# ATS terciles ~1.1 / 2.6; OU terciles ~1.3 / 2.9). Tuple is (small_max, medium_max):
# edge in [0, small_max) -> "small"; [small_max, medium_max) -> "medium";
# >= medium_max -> "big".
EDGE_BUCKET_CUTS: dict[str, tuple[float, float]] = {
    "wp": (0.03, 0.08),
    "ats": (1.1, 2.6),
    "ou": (1.3, 2.9),
}

# Bucket labels in display order (small -> medium -> big).
EDGE_BUCKET_LABELS: tuple[str, str, str] = ("small", "medium", "big")

# Neutral-zone epsilon for the tri-state ``roi_favorable`` flag. |roi| at or
# below this magnitude is treated as neutral (None) so a dead-flat slice is not
# painted green or red.
ROI_NEUTRAL_EPSILON: float = 1e-9


# ---------------------------------------------------------------------------
# Scope filter
# ---------------------------------------------------------------------------


def filter_scope(rows: Sequence[dict], scope: str) -> list[dict]:
    """Return the per-bet rows belonging to *scope*.

    * ``"recommended"`` — only rows the simulation sized with real money under
      Kelly (``kelly_stake > 0``), the sim's own genuine-edge gate (CONTEXT
      D-16 REVISED). Kelly's ``edge <= 0.02`` early-exit zeroes the stake for
      non-edge bets, so ``kelly_stake > 0`` is a faithful, already-materialized
      "recommended" signal that requires importing no betting modules.
    * ``"all"`` (or any unknown scope) — every row.

    ``kelly_stake`` may be missing or ``None``; ``(r.get("kelly_stake") or 0)``
    coerces both to 0 so they fall out of the recommended scope.
    """
    if scope == "recommended":
        return [r for r in rows if (r.get("kelly_stake") or 0) > 0]
    return list(rows)


# ---------------------------------------------------------------------------
# Win-rate / ROI primitives
# ---------------------------------------------------------------------------


def _num(v: Any) -> float:
    """Coerce a value to ``float``, mapping ``None``/non-numeric/NaN to ``0.0``.

    The ``float(x or 0.0)`` idiom this replaces guards ``None`` and literal
    ``0.0`` but NOT ``float('nan')``: a NaN is truthy, so ``(nan or 0.0)`` is
    ``nan`` and propagates through ``sum(...)``, the ROI ratio, the cumulative
    equity series, and the histogram bins — turning one bad CSV value into an
    all-NaN KPI/chart. Coercing NaN explicitly here drops the single bad value
    to ``0.0`` instead of corrupting the whole scope's metrics (WR-03).
    """
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if math.isnan(f) else f  # NaN -> 0.0 (do not let it propagate)


def _count_outcomes(rows: Sequence[dict]) -> tuple[int, int, int]:
    """Return ``(wins, losses, pushes)`` using identity checks only.

    Pushes (``outcome is None``) are counted separately and are excluded from
    the win-rate denominator by the callers. NEVER use ``bool(outcome)`` here
    (Pitfall 1).
    """
    wins = sum(1 for r in rows if r.get("outcome") is True)
    losses = sum(1 for r in rows if r.get("outcome") is False)
    pushes = sum(1 for r in rows if r.get("outcome") is None)
    return wins, losses, pushes


def _win_rate(rows: Sequence[dict]) -> float:
    """Win rate as a percentage over *decided* bets only (wins + losses).

    Pushes are excluded from the denominator. Returns ``0.0`` when there are no
    decided bets (zero-division guarded).
    """
    wins, losses, _pushes = _count_outcomes(rows)
    decided = wins + losses
    if decided == 0:
        return 0.0
    return wins / decided * 100.0


def _roi(rows: Sequence[dict], stake_col: str, payout_col: str) -> float:
    """ROI as a percentage: ``sum(payout) / sum(stake) * 100``.

    ``payout_*`` is the net profit/loss per bet (already net of stake in the
    simulation CSV), so ROI = total net / total wagered. Returns ``0.0`` when
    nothing was wagered (zero-division guarded).
    """
    wagered = sum(_num(r.get(stake_col)) for r in rows)
    net = sum(_num(r.get(payout_col)) for r in rows)
    if wagered == 0.0:
        return 0.0
    return net / wagered * 100.0


def _max_drawdown(rows: Sequence[dict], payout_col: str) -> float:
    """Peak-to-trough drawdown of the cumulative equity series, in dollars.

    Equity starts at :data:`STARTING_BANKROLL` and each bet's ``payout_col``
    (net profit/loss) is added in the row order supplied (rows are pre-sorted by
    ``(season, week)`` upstream). The drawdown is the largest peak-minus-equity
    seen along the way. An empty list or a monotonically rising equity returns
    ``0.0``.
    """
    equity = STARTING_BANKROLL
    peak = STARTING_BANKROLL
    max_dd = 0.0
    for r in rows:
        equity += _num(r.get(payout_col))
        peak = max(peak, equity)
        drawdown = peak - equity
        max_dd = max(max_dd, drawdown)
    return max_dd


# ---------------------------------------------------------------------------
# KPI scoreboard (D-06)
# ---------------------------------------------------------------------------


def compute_kpis(rows: Sequence[dict]) -> dict[str, Any]:
    """Compute the 7-card KPI scoreboard for a (scope-filtered) row list.

    Returns a dict with exactly these keys:

    * ``total_bets`` — row count (pushes counted as bets, not as wins).
    * ``win_rate`` — wins / (wins + losses) * 100; pushes excluded.
    * ``roi_flat`` — sum(payout_flat) / sum(flat_stake) * 100.
    * ``roi_kelly`` — sum(payout_kelly) / sum(kelly_stake) * 100.
    * ``net_profit_flat`` — sum(payout_flat).
    * ``final_bankroll_flat`` — STARTING_BANKROLL + net_profit_flat.
    * ``max_drawdown_flat`` — peak-to-trough on the cumulative flat-equity series.

    Empty input returns zeroed values (with ``final_bankroll_flat`` ==
    ``STARTING_BANKROLL``) and never raises.
    """
    net_profit_flat = sum(_num(r.get("payout_flat")) for r in rows)
    return {
        "total_bets": len(rows),
        "win_rate": _win_rate(rows),
        "roi_flat": _roi(rows, "flat_stake", "payout_flat"),
        "roi_kelly": _roi(rows, "kelly_stake", "payout_kelly"),
        "net_profit_flat": net_profit_flat,
        "final_bankroll_flat": STARTING_BANKROLL + net_profit_flat,
        "max_drawdown_flat": _max_drawdown(rows, "payout_flat"),
    }


# ---------------------------------------------------------------------------
# Per-type edge buckets (D-12)
# ---------------------------------------------------------------------------


def edge_bucket_for(target: str, edge: float) -> str:
    """Return the ``"small"`` / ``"medium"`` / ``"big"`` bucket for an *edge*.

    Cut points are per bet type (:data:`EDGE_BUCKET_CUTS`) because edge units
    differ: WP edge is a signed probability delta (bucketed by magnitude, so a
    -0.05 WP edge buckets the same as +0.05), while ATS/OU edge is points.
    Unknown targets fall back to the ATS cut points.

    Boundaries are left-closed / right-open: ``[0, small_max)`` -> small,
    ``[small_max, medium_max)`` -> medium, ``>= medium_max`` -> big.
    """
    small_max, medium_max = EDGE_BUCKET_CUTS.get(target, EDGE_BUCKET_CUTS["ats"])
    magnitude = abs(_num(edge))  # NaN-safe: a NaN edge buckets as 0.0 -> "small"
    if magnitude < small_max:
        return EDGE_BUCKET_LABELS[0]
    if magnitude < medium_max:
        return EDGE_BUCKET_LABELS[1]
    return EDGE_BUCKET_LABELS[2]


# ---------------------------------------------------------------------------
# ROI-by-slice table (D-10 / D-11) — single-source formatted-row builder
# ---------------------------------------------------------------------------


def _roi_favorable(roi_flat: float, bet_count: int, decided: int) -> bool | None:
    """Tri-state favorable flag for an ROI slice.

    * ``True`` if ``roi_flat`` is meaningfully positive.
    * ``False`` if meaningfully negative.
    * ``None`` (neutral) when there are no decided bets in the slice, the slice
      is empty, or ``|roi_flat|`` is within :data:`ROI_NEUTRAL_EPSILON`.

    Mirrors the ``gap_favorable`` tri-state in
    :func:`api.insights_metrics._gap_favorable` so the template's ``is sameas``
    coloring (True/False/None -> green/red/gray) works identically.
    """
    if bet_count == 0 or decided == 0:
        return None
    if abs(roi_flat) <= ROI_NEUTRAL_EPSILON:
        return None
    return roi_flat > 0.0


def _format_pct(value: float) -> str:
    """Format an ROI / win-rate percentage with a sign and two decimals."""
    return f"{value:+.2f}%"


def _format_win_rate(value: float) -> str:
    """Format a win-rate percentage (no leading sign) with one decimal."""
    return f"{value:.1f}%"


def _make_roi_row(slice_kind: str, label: str, rows: Sequence[dict]) -> dict[str, Any]:
    """Assemble one ROI-table row for a slice of bets with formatted fields.

    The returned dict carries the raw numeric values, the ``*_fmt`` display
    strings, a ``bet_count``, and the tri-state ``roi_favorable`` flag — the
    exact shape the JSON-blob builder and any table-shaped chart consume.
    """
    roi_flat = _roi(rows, "flat_stake", "payout_flat")
    roi_kelly = _roi(rows, "kelly_stake", "payout_kelly")
    win_rate = _win_rate(rows)
    wins, losses, _pushes = _count_outcomes(rows)
    decided = wins + losses
    bet_count = len(rows)
    return {
        "slice_kind": slice_kind,
        "label": label,
        "bet_count": bet_count,
        "roi_flat": roi_flat,
        "roi_kelly": roi_kelly,
        "win_rate": win_rate,
        "roi_flat_fmt": _format_pct(roi_flat),
        "roi_kelly_fmt": _format_pct(roi_kelly),
        "win_rate_fmt": _format_win_rate(win_rate),
        "roi_favorable": _roi_favorable(roi_flat, bet_count, decided),
    }


def _slice_by_key(rows: Sequence[dict], key: str) -> dict[Any, list[dict]]:
    """Group *rows* by ``row[key]``, preserving first-seen order of the keys."""
    groups: dict[Any, list[dict]] = {}
    for r in rows:
        value = r.get(key)
        groups.setdefault(value, []).append(r)
    return groups


def compute_roi_table(rows: Sequence[dict]) -> list[dict[str, Any]]:
    """Build the ROI-by-slice summary table (D-10).

    Produces per-slice formatted rows across three slice kinds:

    * ``"type"`` — by bet target (wp / ats / ou), in canonical order.
    * ``"season"`` — by season, ascending.
    * ``"bucket"`` — by per-type edge bucket, as ``"<target>:<bucket>"`` labels
      in (target, small->medium->big) order.

    Each row carries ``roi_flat`` / ``roi_kelly`` / ``win_rate`` plus their
    ``*_fmt`` strings, a ``bet_count``, and a tri-state ``roi_favorable`` flag.
    This is the single-source builder both the JSON-blob KPI/ROI accessor and
    any table-shaped chart call, so the formulas cannot drift.

    Empty input returns ``[]``.
    """
    if not rows:
        return []

    table: list[dict[str, Any]] = []

    # --- by bet type (canonical order, only types actually present) ---
    by_type = _slice_by_key(rows, "target")
    for target in ("wp", "ats", "ou"):
        if target in by_type:
            table.append(_make_roi_row("type", target, by_type[target]))

    # --- by season (ascending; skip None seasons defensively) ---
    by_season = _slice_by_key(rows, "season")
    for season in sorted(s for s in by_season if s is not None):
        table.append(_make_roi_row("season", str(season), by_season[season]))

    # --- by per-type edge bucket ((target, small->medium->big) order) ---
    for target in ("wp", "ats", "ou"):
        type_rows = by_type.get(target, [])
        if not type_rows:
            continue
        bucketed: dict[str, list[dict]] = {label: [] for label in EDGE_BUCKET_LABELS}
        for r in type_rows:
            bucket = edge_bucket_for(target, _num(r.get("edge")))
            bucketed[bucket].append(r)
        for label in EDGE_BUCKET_LABELS:
            slice_rows = bucketed[label]
            if slice_rows:
                table.append(_make_roi_row("bucket", f"{target}:{label}", slice_rows))

    return table
