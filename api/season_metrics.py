"""Shared pure-function metrics for the Season Tracking page.

Single source of truth for every per-target hit-rate statistic used by both the
chart generators (:mod:`api.charts.season`) and the per-season KPI JSON-blob
builder (:mod:`api.charts.prerender`). Do NOT duplicate any of these formulas
elsewhere — the chart layer and the JSON-blob builder both import from here so
they cannot drift (the "single-source" rule the Phase 16/17 twins established
for :mod:`api.insights_metrics` / :mod:`api.betting_metrics`).

Contents
--------
* Per-game classifiers — :func:`_wp_outcome`, :func:`_ats_outcome`,
  :func:`_ou_outcome` (return ``True`` hit / ``False`` miss / ``None`` excluded).
* Season KPI scoreboard — :func:`compute_season_kpis` (per-target hit-rate +
  decided/hit counts + a straight-up W-L record; pushes/ties excluded).
* Cumulative series — :func:`compute_cumulative_series` (running season-to-date
  hit-rate per target, count-based denominator so a push/tie never advances it).
* Weekly series — :func:`compute_weekly_series` (per-week hit-rate + a
  rolling-average overlay).

LOCKED HIT-RATE CONVENTION (CONTEXT D-05, post-research)
--------------------------------------------------------
ATS/OU use the authoritative ``BettingSimulator`` SIGN CONVENTION WITH 0.5-pt
slippage so the season page reproduces the ``/betting`` numbers exactly
(ATS ~51.4%, OU ~49.3%, WP ~67.7%). This is NOT the This-Week-banner summary
math in ``api/routes/pages.py``, which reads ``ats_prediction`` as a home
margin (sign-flipped) and yields a spurious ~78% ATS hit-rate. WP additionally
excludes tie games (``margin == 0``). Pushes (actual lands exactly on the
slipped line) are excluded from every denominator.

The sign convention is COPIED (with source-citing comments) from
``backtest/simulation.py`` — ``_determine_bet_side_ats`` (284-297),
``_determine_bet_side_ou`` (299-312), ``apply_slippage_spread`` (163-189),
``apply_slippage_total`` (192-218), ``_resolve_ats_outcome`` (324-342),
``_resolve_ou_outcome`` (344-357) — but NEVER imported (UIAP-01).

UIAP-01 COMPLIANCE
------------------
This module imports only stdlib and ``numpy``. No imports from ``models``,
``features``, ``ratings``, ``backtest``, or ``api.*`` — keeping api-side formula
reuse free of circular-import risk and satisfying the UIAP-01 import guard. The
locked constants below are copied *values* (with a source-citing comment),
never imports of ``backtest.simulation``.

NaN-SAFETY (WR-03 idiom)
------------------------
A NaN score or NaN market line must NOT propagate into a hit-rate or a series.
:func:`_num` coerces non-numeric / NaN to ``0.0``, and :func:`_score` /
:func:`_line` return ``None`` for missing-or-NaN values so the affected row
drops to the excluded path rather than corrupting the whole season's metrics.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Locked constants (copied VALUES with source-citing comments — UIAP-01)
# ---------------------------------------------------------------------------

# Standard -110 break-even win rate (110 / 210). Exact for spread/totals (D-06);
# moneyline (WP) odds vary, so this is presented as a reference line, not a
# per-game truth. Defined as a copied value (NOT imported — UIAP-01); the same
# constant lives in api/betting_metrics.py:57.
BREAKEVEN_WIN_RATE: float = 0.524

# Half-point slippage against the bettor. Source: backtest.simulation
# SLIPPAGE_POINTS (NOT imported — UIAP-01). Applied so the season hit-rate
# reproduces the /betting page: home_cover -> line - 0.5, away_cover -> line +
# 0.5; over -> line + 0.5, under -> line - 0.5 (always against the bettor).
SLIPPAGE_POINTS: float = 0.5

# Canonical target order used throughout the public surface.
_TARGETS: tuple[str, ...] = ("wp", "ats", "ou")

# Rolling-average window for the weekly overlay (weeks). D-07 leaves this to
# Claude's discretion; 3 weeks smooths the ~14-16-game weekly noise without
# lagging the "current form" read the overlay is meant to convey.
ROLLING_WINDOW: int = 3


# ---------------------------------------------------------------------------
# Numeric coercion helpers
# ---------------------------------------------------------------------------


def _num(v: Any) -> float:
    """Coerce a value to ``float``, mapping ``None``/non-numeric/NaN to ``0.0``.

    The ``float(x or 0.0)`` idiom this replaces guards ``None`` and literal
    ``0.0`` but NOT ``float('nan')``: a NaN is truthy, so ``(nan or 0.0)`` is
    ``nan`` and propagates through every downstream sum/ratio/series. Coercing
    NaN explicitly here drops the single bad value to ``0.0`` instead of
    corrupting the whole season's metrics (WR-03, copied from betting_metrics).
    """
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if math.isnan(f) else f  # NaN -> 0.0 (do not let it propagate)


def _score(v: Any) -> float | None:
    """Return a score as ``float``, or ``None`` if missing / non-numeric / NaN.

    Unlike :func:`_num` (which maps a bad value to ``0.0``), an unusable score
    must EXCLUDE the row from the denominator — a NaN/None score is not a 0-0
    game. Callers treat ``None`` as "undecided" so the row drops out cleanly.
    """
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(f) else f


def _line(v: Any) -> float | None:
    """Return a market line as ``float``, or ``None`` if missing / NaN.

    A null or NaN ``market_spread`` / ``market_total`` excludes the row from
    that target's denominator (no line to bet against). Same semantics as
    :func:`_score`.
    """
    return _score(v)


# ---------------------------------------------------------------------------
# Per-game classifiers (LOCKED simulator convention — return True/False/None)
# ---------------------------------------------------------------------------


def _wp_outcome(row: dict) -> bool | None:
    """WP hit: did the predicted straight-up winner win? Ties EXCLUDED.

    ``margin = home_score - away_score``. A tie (``margin == 0``) is a push for
    a straight-up bet and is EXCLUDED (D-05; 3 such games exist in the data).
    Otherwise the pick hits iff ``(wp_prob > 0.5) == (margin > 0)``.

    Returns ``None`` (excluded) for a tie, a missing/NaN score, or a missing
    ``wp_prob``.
    """
    home = _score(row.get("home_score"))
    away = _score(row.get("away_score"))
    if home is None or away is None:
        return None
    margin = home - away
    if margin == 0:
        return None  # tie -> push (EXCLUDE from WP denominator)
    wp_prob = row.get("wp_prob")
    if wp_prob is None:
        return None
    return (float(wp_prob) > 0.5) == (margin > 0)


def _ats_outcome(row: dict) -> bool | None:
    """ATS hit using the LOCKED simulator sign convention WITH 0.5-pt slippage.

    ``market_spread`` is the HOME-perspective closing spread. The model picks a
    side by comparing ``ats_prediction`` (== model_spread) to ``market_spread``
    (simulation.py::_determine_bet_side_ats):

    * ``ats_prediction < market_spread`` -> ``home_cover``, slipped =
      ``market_spread - 0.5``.
    * ``ats_prediction > market_spread`` -> ``away_cover``, slipped =
      ``market_spread + 0.5``.
    * equal -> no directional pick -> EXCLUDED.

    With ``actual_margin = home_score - away_score``: a push (``abs(actual_margin
    - slipped) < 1e-9``) is EXCLUDED; otherwise ``home_covers = actual_margin >
    slipped`` and the pick hits = ``home_covers`` for ``home_cover`` else ``not
    home_covers`` (simulation.py::_resolve_ats_outcome).

    Returns ``None`` (excluded) for no directional pick, a push, a missing/NaN
    score, or a missing/NaN ``market_spread`` / ``ats_prediction``.
    """
    home = _score(row.get("home_score"))
    away = _score(row.get("away_score"))
    market_spread = _line(row.get("market_spread"))
    if home is None or away is None or market_spread is None:
        return None
    ats_prediction = row.get("ats_prediction")
    if ats_prediction is None or (
        isinstance(ats_prediction, float) and math.isnan(ats_prediction)
    ):
        return None
    pred = float(ats_prediction)

    if pred < market_spread:
        slipped = market_spread - SLIPPAGE_POINTS  # home_cover (against bettor)
        side = "home_cover"
    elif pred > market_spread:
        slipped = market_spread + SLIPPAGE_POINTS  # away_cover (against bettor)
        side = "away_cover"
    else:
        return None  # no directional pick -> EXCLUDE

    actual_margin = home - away
    if abs(actual_margin - slipped) < 1e-9:
        return None  # push -> EXCLUDE
    home_covers = actual_margin > slipped
    return home_covers if side == "home_cover" else (not home_covers)


def _ou_outcome(row: dict) -> bool | None:
    """OU hit using the LOCKED simulator sign convention WITH 0.5-pt slippage.

    The model picks a side by comparing ``ou_prediction`` to ``market_total``
    (simulation.py::_determine_bet_side_ou):

    * ``ou_prediction > market_total`` -> ``over``, slipped =
      ``market_total + 0.5``.
    * ``ou_prediction < market_total`` -> ``under``, slipped =
      ``market_total - 0.5``.
    * equal -> no directional pick -> EXCLUDED.

    With ``actual_total = home_score + away_score``: a push (``abs(actual_total -
    slipped) < 1e-9``) is EXCLUDED; otherwise ``went_over = actual_total >
    slipped`` and the pick hits = ``went_over`` for ``over`` else ``not
    went_over`` (simulation.py::_resolve_ou_outcome).

    Returns ``None`` (excluded) for no directional pick, a push, a missing/NaN
    score, or a missing/NaN ``market_total`` / ``ou_prediction``.
    """
    home = _score(row.get("home_score"))
    away = _score(row.get("away_score"))
    market_total = _line(row.get("market_total"))
    if home is None or away is None or market_total is None:
        return None
    ou_prediction = row.get("ou_prediction")
    if ou_prediction is None or (
        isinstance(ou_prediction, float) and math.isnan(ou_prediction)
    ):
        return None
    pred = float(ou_prediction)

    if pred > market_total:
        slipped = market_total + SLIPPAGE_POINTS  # over (against bettor)
        side = "over"
    elif pred < market_total:
        slipped = market_total - SLIPPAGE_POINTS  # under (against bettor)
        side = "under"
    else:
        return None  # no directional pick -> EXCLUDE

    actual_total = home + away
    if abs(actual_total - slipped) < 1e-9:
        return None  # push -> EXCLUDE
    went_over = actual_total > slipped
    return went_over if side == "over" else (not went_over)


_OUTCOME_FNS = {
    "wp": _wp_outcome,
    "ats": _ats_outcome,
    "ou": _ou_outcome,
}


# ---------------------------------------------------------------------------
# Row selection + ordering helpers
# ---------------------------------------------------------------------------


def _is_completed(row: dict) -> bool:
    """A row counts only if ``status == 'completed'`` with usable scores.

    Per D-03, per-target accuracy is computed from completed games only
    (``status='completed'`` AND non-null scores). A NaN/None score fails here so
    it never reaches a denominator.
    """
    if row.get("status") != "completed":
        return False
    return _score(row.get("home_score")) is not None and (
        _score(row.get("away_score")) is not None
    )


def _completed_sorted(rows: Sequence[dict]) -> list[dict]:
    """Return completed rows sorted chronologically by ``(week,)`` within season.

    Rows are assumed to belong to one season (the prerender loop slices per
    season), so ordering by week is sufficient; ``game_id`` breaks ties for
    determinism. A missing week sorts first (0).
    """
    completed = [r for r in rows if _is_completed(r)]
    completed.sort(key=lambda r: (int(_num(r.get("week"))), str(r.get("game_id", ""))))
    return completed


def _rate(hits: int, decided: int) -> float:
    """Hit-rate as a percentage over decided games; ``0.0`` when none decided."""
    if decided == 0:
        return 0.0
    return hits / decided * 100.0


# ---------------------------------------------------------------------------
# Season KPI scoreboard (D-08)
# ---------------------------------------------------------------------------


def compute_season_kpis(rows: Sequence[dict]) -> dict[str, Any]:
    """Compute the season-to-date KPI strip for one season's prediction rows.

    Returns a dict with, per target ``t`` in (wp, ats, ou):

    * ``{t}_hit_rate`` — hits / decided * 100 (pushes/ties excluded).
    * ``{t}_decided``  — decided game count (hits + misses).
    * ``{t}_hits``     — hit count.

    plus a straight-up W-L record (correct/incorrect WP calls, ties excluded):

    * ``record_wins`` / ``record_losses`` — int counts.
    * ``record`` — ``"<wins>-<losses>"`` display string.

    Only completed games (``status='completed'`` with usable scores) contribute.
    Empty / all-excluded input returns zeroed values (record ``"0-0"``) and
    never raises.
    """
    completed = [r for r in rows if _is_completed(r)]

    kpis: dict[str, Any] = {}
    for target in _TARGETS:
        fn = _OUTCOME_FNS[target]
        hits = 0
        decided = 0
        for r in completed:
            outcome = fn(r)
            if outcome is None:
                continue  # push / tie / excluded -> not in the denominator
            decided += 1
            if outcome:
                hits += 1
        kpis[f"{target}_hit_rate"] = _rate(hits, decided)
        kpis[f"{target}_decided"] = decided
        kpis[f"{target}_hits"] = hits

    # Straight-up W-L record == WP hits / misses (ties already excluded).
    record_wins = kpis["wp_hits"]
    record_losses = kpis["wp_decided"] - kpis["wp_hits"]
    kpis["record_wins"] = record_wins
    kpis["record_losses"] = record_losses
    kpis["record"] = f"{record_wins}-{record_losses}"
    return kpis


# ---------------------------------------------------------------------------
# Cumulative running hit-rate (DASH-07) — count-based denominator
# ---------------------------------------------------------------------------


def compute_cumulative_series(rows: Sequence[dict]) -> dict[str, dict[str, list]]:
    """Per-target running season-to-date hit-rate (count-based denominator).

    For each target, walk completed games in chronological (week) order; for
    every DECIDED game append a point ``cumulative_hits / cumulative_decided *
    100``. A push/tie produces NO point and does NOT advance the denominator
    (the count-based idiom — adapted from ``api/charts/core.py:251-252`` but
    accumulating hit/decided COUNTS, not a value-mean).

    Returns ``{target: {"weeks": [...], "values": [...]}}`` where ``weeks`` are
    the weeks of the decided games (one entry per ``values`` point). Empty /
    all-excluded input yields empty lists per target.
    """
    completed = _completed_sorted(rows)

    series: dict[str, dict[str, list]] = {}
    for target in _TARGETS:
        fn = _OUTCOME_FNS[target]
        decided_hits: list[float] = []
        decided_weeks: list[int] = []
        for r in completed:
            outcome = fn(r)
            if outcome is None:
                continue  # push/tie -> does not advance the denominator
            decided_hits.append(1.0 if outcome else 0.0)
            decided_weeks.append(int(_num(r.get("week"))))

        if decided_hits:
            hits_arr = np.array(decided_hits, dtype=float)
            running = np.cumsum(hits_arr) / np.arange(1, len(hits_arr) + 1) * 100.0
            values = [float(v) for v in running]
        else:
            values = []
        series[target] = {"weeks": decided_weeks, "values": values}
    return series


# ---------------------------------------------------------------------------
# Weekly hit-rate + rolling-average overlay (DASH-08)
# ---------------------------------------------------------------------------


def _rolling_average(values: Sequence[float], window: int) -> list[float]:
    """Trailing rolling mean over up to ``window`` prior points (inclusive).

    Point ``i`` is the mean of ``values[max(0, i-window+1) : i+1]`` so early
    points use a shorter window (no NaN padding). A flat series maps to itself.
    """
    out: list[float] = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        chunk = values[lo : i + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def compute_weekly_series(
    rows: Sequence[dict],
) -> dict[str, dict[str, list]]:
    """Per-target per-week hit-rate plus a rolling-average overlay (DASH-08).

    For each target, group completed games by week (ascending) and compute the
    per-week hit-rate = week hits / week decided * 100 (pushes/ties excluded). A
    week with zero decided games for a target is omitted from that target's
    series. The ``rolling`` overlay is a trailing :data:`ROLLING_WINDOW`-week
    mean of the weekly values (UI-SPEC: "a smoothing of the same series").

    Returns ``{target: {"weeks": [...], "values": [...], "rolling": [...]}}``;
    all three lists align by index. Empty input yields empty lists per target.
    """
    completed = _completed_sorted(rows)

    series: dict[str, dict[str, list]] = {}
    for target in _TARGETS:
        fn = _OUTCOME_FNS[target]
        # Accumulate (hits, decided) per week preserving ascending week order.
        per_week: dict[int, list[int]] = {}
        for r in completed:
            outcome = fn(r)
            if outcome is None:
                continue
            week = int(_num(r.get("week")))
            bucket = per_week.setdefault(week, [0, 0])
            bucket[1] += 1  # decided
            if outcome:
                bucket[0] += 1  # hits

        weeks = sorted(per_week)
        values = [_rate(per_week[w][0], per_week[w][1]) for w in weeks]
        rolling = _rolling_average(values, ROLLING_WINDOW)
        series[target] = {"weeks": weeks, "values": values, "rolling": rolling}
    return series
