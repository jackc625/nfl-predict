"""Shared pure-function metrics for the Model Insights page.

Single source of truth for every statistical primitive used by both the chart
generators (:mod:`api.charts.insights`) and the aggregate-table builder
(:mod:`api.charts.prerender`). Do NOT duplicate any of these formulas
elsewhere in the codebase.

Contents
--------
* American-odds / devig helpers — :func:`american_odds_to_prob`,
  :func:`compute_wp_market_prob`.
* WP metrics — :func:`compute_brier`, :func:`compute_log_loss` (with epsilon
  clipping per Codex HIGH #2), :func:`compute_accuracy` (with ``p >= 0.5``
  threshold per Codex HIGH #3).
* ATS / OU MAE primitives + rowwise helpers that enforce the
  ``-market_spread`` sign convention: :func:`compute_ats_mae`,
  :func:`compute_ats_mae_model_and_market`, :func:`compute_ou_mae`,
  :func:`compute_ou_mae_model_and_market`.
* Residual bin helpers — :func:`bin_ats_residuals`, :func:`bin_ou_values`
  (overflow bins retained, empty bins retained; Codex HIGH #4).
* Aggregate-table builder — :func:`build_aggregate_rows` (Codex HIGH #1).

UIAP-01 COMPLIANCE
------------------
This module imports only stdlib, ``numpy``, and ``sklearn``. No imports from
``models``, ``features``, ``ratings``, or ``api.*`` — keeping api-side formula
reuse free of circular import risk.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypedDict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    mean_absolute_error,
)

# ---------------------------------------------------------------------------
# Locked constants
# ---------------------------------------------------------------------------

LOG_LOSS_EPSILON: float = 1e-15  # Codex HIGH #2
WP_CLASS_THRESHOLD: float = 0.5  # Codex HIGH #3 — accuracy cutoff

ATS_BIN_EDGES: tuple[float, ...] = tuple(float(x) for x in range(-21, 22, 2))
ATS_BIN_WIDTH: float = 2.0
ATS_UNDERFLOW_LABEL: str = "<=-21"
ATS_OVERFLOW_LABEL: str = ">=21"

OU_BIN_EDGES: tuple[float, ...] = tuple(float(x) for x in range(35, 68, 3))
OU_BIN_WIDTH: float = 3.0
OU_UNDERFLOW_LABEL: str = "<=35"
OU_OVERFLOW_LABEL: str = ">=65"

# Neutral-zone threshold for aggregate-table "gap_favorable" disposition.
# |gap| <= this magnitude is treated as neutral (Codex MEDIUM #13).
GAP_NEUTRAL_EPSILON: float = 0.005


class MarketProbResult(TypedDict):
    """Return shape for :func:`compute_wp_market_prob`."""

    prob: float
    devig_method: Literal["standard", "implied"]


# ---------------------------------------------------------------------------
# American odds -> implied probability
# ---------------------------------------------------------------------------


def american_odds_to_prob(ml: int | float) -> float:
    """Convert an American moneyline to its implied probability.

    Negative odds: ``p = -ml / (-ml + 100)`` (e.g. -150 -> 0.60).
    Positive odds: ``p = 100 / (ml + 100)`` (e.g. +130 -> 100 / 230).

    Raises:
        ValueError: if *ml* is zero (invalid moneyline).
    """
    ml_f = float(ml)
    if ml_f == 0.0:
        raise ValueError("moneyline of 0 is invalid")
    if ml_f < 0:
        return -ml_f / (-ml_f + 100.0)
    return 100.0 / (ml_f + 100.0)


# Plan-text alias so future callers that copied the plan docstring still work.
american_odds_to_implied_prob = american_odds_to_prob


# ---------------------------------------------------------------------------
# Devig (Codex HIGH #5)
# ---------------------------------------------------------------------------


def compute_wp_market_prob(
    ml_home: int | float | None,
    ml_away: int | float | None,
) -> MarketProbResult | None:
    """Compute the devigged home-team WP from American moneylines.

    Behaviour:
        * Both sides present: multiplicative devig ->
          ``p_home_fair = p_home_implied / (p_home_implied + p_away_implied)``
          with ``devig_method = "standard"``.
        * Exactly one side present: return the implied probability of that side
          (when only *ml_home* is present) or ``1 - implied(ml_away)`` when
          only *ml_away* is present. ``devig_method = "implied"``.
        * Neither side present: return ``None``. Caller should exclude the row
          from market-comparison statistics and log the excluded count.
    """
    if ml_home is None and ml_away is None:
        return None
    if ml_home is None:
        p_away = american_odds_to_prob(ml_away)  # type: ignore[arg-type]
        return {"prob": 1.0 - p_away, "devig_method": "implied"}
    if ml_away is None:
        p_home = american_odds_to_prob(ml_home)
        return {"prob": p_home, "devig_method": "implied"}
    p_home = american_odds_to_prob(ml_home)
    p_away = american_odds_to_prob(ml_away)
    total = p_home + p_away
    if total <= 0:
        return None
    return {"prob": p_home / total, "devig_method": "standard"}


# ---------------------------------------------------------------------------
# WP metrics
# ---------------------------------------------------------------------------


def _to_float_array(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return a 1-D float64 numpy array from any numeric sequence."""
    return np.asarray(values, dtype=np.float64)


def compute_brier(
    y_true: Sequence[float] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
) -> float:
    """Brier score: ``mean((y_prob - y_true) ** 2)``."""
    return float(brier_score_loss(_to_float_array(y_true), _to_float_array(y_prob)))


def compute_log_loss(
    y_true: Sequence[float] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
) -> float:
    """Log-loss with probabilities clipped to ``[LOG_LOSS_EPSILON, 1 - eps]``.

    Without clipping, probabilities at exactly 0 or 1 produce ``-inf`` (Codex
    HIGH #2). Computed in closed form rather than via sklearn so the
    float64-representation drift between ``1 - eps`` and ``eps`` (which makes
    ``1 - (1 - eps)`` not equal to ``eps``) does not leak into results: when a
    prediction would be clipped to ``1 - eps`` for a negative label, the
    loss-side term is evaluated directly against ``-log(eps)``.
    """
    y_true_arr = _to_float_array(y_true)
    y_prob_arr = _to_float_array(y_prob)
    n = y_prob_arr.size
    if n == 0:
        return 0.0
    # Clip predictions into ``[eps, 1 - eps]`` so log() stays finite.
    p_pos = np.clip(y_prob_arr, LOG_LOSS_EPSILON, 1.0 - LOG_LOSS_EPSILON)
    # Compute the "other side" of the clip directly off LOG_LOSS_EPSILON so
    # perfectly-confident wrong predictions resolve to exactly ``-log(eps)``
    # regardless of float rounding in ``1 - (1 - eps)``.
    unclipped = y_prob_arr
    p_neg = np.where(
        unclipped >= 1.0 - LOG_LOSS_EPSILON,
        LOG_LOSS_EPSILON,
        np.where(
            unclipped <= LOG_LOSS_EPSILON,
            1.0 - LOG_LOSS_EPSILON,
            1.0 - p_pos,
        ),
    )
    losses = -(y_true_arr * np.log(p_pos) + (1.0 - y_true_arr) * np.log(p_neg))
    return float(np.mean(losses))


def compute_accuracy(
    y_true: Sequence[float] | np.ndarray,
    y_prob: Sequence[float] | np.ndarray,
) -> float:
    """WP accuracy using the fixed threshold ``p >= 0.5`` (Codex HIGH #3)."""
    y_true_arr = _to_float_array(y_true).astype(int)
    y_prob_arr = _to_float_array(y_prob)
    y_pred = (y_prob_arr >= WP_CLASS_THRESHOLD).astype(int)
    return float(accuracy_score(y_true_arr, y_pred))


# ---------------------------------------------------------------------------
# ATS / OU MAE
# ---------------------------------------------------------------------------


def compute_ats_mae(
    predicted_home_margins: Sequence[float] | np.ndarray,
    actual_home_margins: Sequence[float] | np.ndarray,
) -> float:
    """Mean absolute error for ATS. Both arrays are in home-margin space."""
    return float(
        mean_absolute_error(
            _to_float_array(actual_home_margins),
            _to_float_array(predicted_home_margins),
        )
    )


def market_implied_home_margin(market_spread: float) -> float:
    """Convert ``market_spread`` (away spread, negative => home favored) to
    the market's implied HOME margin.

    Contract: ``implied_home_margin = -market_spread``. This is the sign flip
    tested by ``tests/test_insights_metrics.py::test_compute_ats_mae_sign_convention``.
    """
    return -float(market_spread)


def compute_ou_mae(
    predicted_totals: Sequence[float] | np.ndarray,
    actual_totals: Sequence[float] | np.ndarray,
) -> float:
    """Mean absolute error for OU. Market predicted total == ``market_total``."""
    return float(
        mean_absolute_error(
            _to_float_array(actual_totals),
            _to_float_array(predicted_totals),
        )
    )


def compute_ats_mae_model_and_market(rows: Sequence[dict]) -> tuple[float, float]:
    """Return ``(model_mae, market_mae)`` for ATS rows.

    Each row carries ``model_prob`` (predicted home margin), ``actual``
    (actual home margin), and ``market_spread`` (away spread, so market's
    predicted home margin is ``-market_spread`` per the sign convention).
    Rows with ``market_spread is None`` are excluded from BOTH series so the
    comparison is over the same games (Codex HIGH #4).

    Returns ``(nan, nan)`` when no usable rows remain.
    """
    usable = [r for r in rows if r.get("market_spread") is not None]
    if not usable:
        return (float("nan"), float("nan"))
    model_preds = np.array([float(r["model_prob"]) for r in usable])
    actuals = np.array([float(r["actual"]) for r in usable])
    market_preds = np.array(
        [market_implied_home_margin(float(r["market_spread"])) for r in usable]
    )
    return (
        compute_ats_mae(model_preds, actuals),
        compute_ats_mae(market_preds, actuals),
    )


def compute_ou_mae_model_and_market(rows: Sequence[dict]) -> tuple[float, float]:
    """Return ``(model_mae, market_mae)`` for OU rows.

    Market predicted total == ``market_total`` directly (no sign flip). Rows
    with ``market_total is None`` are excluded from BOTH series.
    """
    usable = [r for r in rows if r.get("market_total") is not None]
    if not usable:
        return (float("nan"), float("nan"))
    model_preds = np.array([float(r["model_prob"]) for r in usable])
    actuals = np.array([float(r["actual"]) for r in usable])
    market_preds = np.array([float(r["market_total"]) for r in usable])
    return (
        compute_ou_mae(model_preds, actuals),
        compute_ou_mae(market_preds, actuals),
    )


# ---------------------------------------------------------------------------
# Residual binning (Codex HIGH #4)
# ---------------------------------------------------------------------------


def _format_bin_label(lo: float, width: float) -> str:
    """Return ``"[lo, lo+width)"`` with ``:g`` formatting for both ends."""
    return f"[{lo:g}, {lo + width:g})"


def _initial_bin_dict(
    edges: Sequence[float],
    width: float,
    underflow_label: str,
    overflow_label: str,
) -> dict[str, int]:
    """Return an ordered dict with every interior + overflow bin seeded to 0.

    Empty bins are RETAINED (Codex HIGH #4): the caller sees every bin key,
    not just the ones that received at least one value.
    """
    counts: dict[str, int] = {underflow_label: 0}
    for lo in edges[:-1]:
        counts[_format_bin_label(lo, width)] = 0
    counts[overflow_label] = 0
    return counts


def bin_ats_residuals(
    residuals: Sequence[float] | np.ndarray,
) -> dict[str, int]:
    """Bin ATS residuals into fixed 2-point bins with overflow labels.

    Intervals are left-closed / right-open: ``[-21, -19), [-19, -17), ...,
    [19, 21)``. Residuals ``< -21`` fall into ``ATS_UNDERFLOW_LABEL``, residuals
    ``>= 21`` fall into ``ATS_OVERFLOW_LABEL``. Every bin key is retained in the
    output (even bins with count 0).
    """
    counts = _initial_bin_dict(
        ATS_BIN_EDGES,
        ATS_BIN_WIDTH,
        ATS_UNDERFLOW_LABEL,
        ATS_OVERFLOW_LABEL,
    )
    first_edge = ATS_BIN_EDGES[0]
    last_edge = ATS_BIN_EDGES[-1]
    for r in residuals:
        rf = float(r)
        if rf < first_edge:
            counts[ATS_UNDERFLOW_LABEL] += 1
        elif rf >= last_edge:
            counts[ATS_OVERFLOW_LABEL] += 1
        else:
            idx = int((rf - first_edge) // ATS_BIN_WIDTH)
            lo = ATS_BIN_EDGES[idx]
            counts[_format_bin_label(lo, ATS_BIN_WIDTH)] += 1
    return counts


def bin_ou_values(
    values: Sequence[float] | np.ndarray,
) -> dict[str, int]:
    """Bin OU values (predicted or actual totals) into fixed 3-point bins.

    Intervals are left-closed / right-open: ``[35, 38), [38, 41), ..., [62, 65)``.
    Values ``< 35`` fall into ``OU_UNDERFLOW_LABEL``; values ``>= 65`` fall into
    ``OU_OVERFLOW_LABEL``. Every bin key is retained in the output.
    """
    counts = _initial_bin_dict(
        OU_BIN_EDGES,
        OU_BIN_WIDTH,
        OU_UNDERFLOW_LABEL,
        OU_OVERFLOW_LABEL,
    )
    first_edge = OU_BIN_EDGES[0]
    last_edge = OU_BIN_EDGES[-1]
    for v in values:
        vf = float(v)
        if vf < first_edge:
            counts[OU_UNDERFLOW_LABEL] += 1
        elif vf >= last_edge:
            counts[OU_OVERFLOW_LABEL] += 1
        else:
            idx = int((vf - first_edge) // OU_BIN_WIDTH)
            lo = OU_BIN_EDGES[idx]
            counts[_format_bin_label(lo, OU_BIN_WIDTH)] += 1
    return counts


# Plan-text alias for the OU residual binner.
bin_ou_residuals = bin_ou_values


# ---------------------------------------------------------------------------
# Aggregate-table row construction (Codex HIGH #1)
# ---------------------------------------------------------------------------


_METRIC_DIRECTIONS: dict[str, str] = {
    # True favorable direction for each metric (larger-is-better vs smaller).
    "Accuracy": "higher",
    "Brier": "lower",
    "Log-loss": "lower",
    "MAE": "lower",
}


def _format_value(metric: str, value: float | None) -> str:
    """Format a model/market cell value for display in the aggregate table."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    if metric == "Accuracy":
        return f"{value * 100:.1f}%"
    if metric in ("Brier", "Log-loss"):
        return f"{value:.4f}"
    # MAE
    return f"{value:.2f}"


def _format_gap(metric: str, gap: float | None) -> str:
    """Format the gap cell with a leading sign for non-zero values."""
    if gap is None or (isinstance(gap, float) and np.isnan(gap)):
        return "N/A"
    if metric == "Accuracy":
        return f"{gap * 100:+.1f}%"
    if metric in ("Brier", "Log-loss"):
        return f"{gap:+.4f}"
    return f"{gap:+.2f}"


def _gap_favorable(metric: str, gap: float | None) -> bool | None:
    """Return True/False/None (neutral) for the gap direction.

    * Accuracy: positive gap favors the model.
    * Brier, Log-loss, MAE: negative gap favors the model.
    * ``|gap| <= GAP_NEUTRAL_EPSILON`` => neutral (None) per Codex MEDIUM #13.
    """
    if gap is None or (isinstance(gap, float) and np.isnan(gap)):
        return None
    if abs(gap) <= GAP_NEUTRAL_EPSILON:
        return None
    direction = _METRIC_DIRECTIONS.get(metric, "lower")
    if direction == "higher":
        return gap > 0
    return gap < 0


def _make_row(
    target: str,
    metric: str,
    model: float | None,
    market: float | None,
) -> dict:
    """Assemble a single aggregate-table row with formatted fields."""
    gap: float | None
    if (
        model is None
        or market is None
        or (isinstance(model, float) and np.isnan(model))
        or (isinstance(market, float) and np.isnan(market))
    ):
        gap = None
    else:
        gap = model - market
    return {
        "target": target,
        "metric": metric,
        "model": None
        if model is None or (isinstance(model, float) and np.isnan(model))
        else float(model),
        "market": None
        if market is None or (isinstance(market, float) and np.isnan(market))
        else float(market),
        "gap": None if gap is None else float(gap),
        "gap_favorable": _gap_favorable(metric, gap),
        "model_fmt": _format_value(metric, model),
        "market_fmt": _format_value(metric, market),
        "gap_fmt": _format_gap(metric, gap),
    }


def build_aggregate_rows(
    backtest_predictions: Sequence[dict],
    market_data_by_game: dict[str, dict],
) -> list[dict]:
    """Build the 5-row Model vs Market aggregate table (Codex HIGH #1).

    Rows produced:
        1. WP Accuracy
        2. WP Brier
        3. WP Log-loss
        4. ATS MAE
        5. OU MAE

    Market computations:
        * WP market prob uses :func:`compute_wp_market_prob` and excludes rows
          with both moneylines missing.
        * ATS market prediction = ``-market_spread``; rows with
          ``market_spread is None`` excluded from both series.
        * OU market prediction = ``market_total``; rows with
          ``market_total is None`` excluded from both series.

    Args:
        backtest_predictions: list of ``backtest_predictions`` dicts across all
            targets and seasons.
        market_data_by_game: dict keyed by ``game_id`` mapping to the
            ``predictions`` table row carrying market fields.

    Returns:
        A list of 5 aggregate-row dicts — always exactly 5 entries, with
        ``None`` model/market/gap values when the underlying subset is empty.
    """
    # ---- WP rows ----
    wp_rows = [p for p in backtest_predictions if p.get("target") == "wp"]

    # Model WP series: every WP row has a model_prob + actual.
    if wp_rows:
        wp_actual = np.array([float(r["actual"]) for r in wp_rows])
        wp_model_prob = np.array([float(r["model_prob"]) for r in wp_rows])
        model_acc = compute_accuracy(wp_actual, wp_model_prob)
        model_brier = compute_brier(wp_actual, wp_model_prob)
        model_log_loss = compute_log_loss(wp_actual, wp_model_prob)
    else:
        model_acc = model_brier = model_log_loss = float("nan")

    # Market WP series: paired by game_id with devig; exclude both-None rows.
    market_wp_actuals: list[float] = []
    market_wp_probs: list[float] = []
    for row in wp_rows:
        gid = row.get("game_id")
        market_row = market_data_by_game.get(gid) if gid is not None else None
        if market_row is None:
            continue
        devig = compute_wp_market_prob(
            market_row.get("market_ml_home"),
            market_row.get("market_ml_away"),
        )
        if devig is None:
            continue
        market_wp_actuals.append(float(row["actual"]))
        market_wp_probs.append(float(devig["prob"]))

    if market_wp_actuals:
        market_actual_arr = np.array(market_wp_actuals)
        market_prob_arr = np.array(market_wp_probs)
        market_acc = compute_accuracy(market_actual_arr, market_prob_arr)
        market_brier = compute_brier(market_actual_arr, market_prob_arr)
        market_log_loss = compute_log_loss(market_actual_arr, market_prob_arr)
    else:
        market_acc = market_brier = market_log_loss = float("nan")

    # ---- ATS + OU rows ----
    ats_rows_full: list[dict] = []
    for r in backtest_predictions:
        if r.get("target") != "ats":
            continue
        gid = r.get("game_id")
        market_row = market_data_by_game.get(gid) if gid is not None else None
        merged = dict(r)
        merged["market_spread"] = (
            market_row.get("market_spread") if market_row else None
        )
        ats_rows_full.append(merged)
    ats_model_mae, ats_market_mae = compute_ats_mae_model_and_market(ats_rows_full)

    ou_rows_full: list[dict] = []
    for r in backtest_predictions:
        if r.get("target") != "ou":
            continue
        gid = r.get("game_id")
        market_row = market_data_by_game.get(gid) if gid is not None else None
        merged = dict(r)
        merged["market_total"] = market_row.get("market_total") if market_row else None
        ou_rows_full.append(merged)
    ou_model_mae, ou_market_mae = compute_ou_mae_model_and_market(ou_rows_full)

    return [
        _make_row("WP", "Accuracy", model_acc, market_acc),
        _make_row("WP", "Brier", model_brier, market_brier),
        _make_row("WP", "Log-loss", model_log_loss, market_log_loss),
        _make_row("ATS", "MAE", ats_model_mae, ats_market_mae),
        _make_row("OU", "MAE", ou_model_mae, ou_market_mae),
    ]
