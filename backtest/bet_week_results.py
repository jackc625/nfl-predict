"""Last week's actual versus expected: a PURE per-week aggregation over the stored bet-list rows.

Phase 35, plan 35-03 (SPEC R5, NOTIF-05, UIAP-01, D-20).

WHAT THIS MODULE IS
-------------------
One row per ``(season, week, counting class)`` holding the week's live forward record: wins,
losses, pushes, the graded and still-ungraded counts, the ACTUAL return per unit staked and the
EXPECTED return per unit staked. ``/bets`` and the weekly email show the row for ``(season,
week - 1)``. Like ``backtest/bet_tracker.py`` beside it, it composes and does not re-derive: every
figure is a count or a sum over columns another module already wrote. It prices nothing, re-grades
nothing and opens no connection.

It REUSES the tracker rather than restating it. :func:`backtest.bet_tracker.prepare_tracker_rows`
drops the shadow arm, groups pre-verdict rows under their own class and overlays the in-force
corrections (D-06), and :func:`backtest.bet_tracker._flat_return_units` is the actual return's
formula, ``sum(payout_flat) / sum(flat_stake)``. A week row therefore counts exactly what the
season-to-date tracker counts for the same rows.

WHY IT IS COMPUTED HERE AND ONLY PERSISTED BY ``api/``
-----------------------------------------------------
Aggregation is computation, and computation in the request path breaks UIAP-01; the email must
read stored output, never recompute it (NOTIF-05). So the rows are built at CACHE BUILD by the one
shared reader (``forward_ledger.cache_sources``), ``api.cache.materialize_bet_week_results``
inserts them with no arithmetic, and ``api.services.DataService.get_bet_week_results`` only
selects them. As in the tracker, this module imports constants only from ``api.cache`` -- the
direction ``api/`` never reverses.

ABSENT IS NOT ZERO
------------------
A week with no graded row stores both returns as NULL ("nothing graded"), and a graded row with no
``per_bet_ev`` makes the expected return NULL ("not measured") rather than an average over the
other rows. Neither is ever stored as 0.0, which would read as a measured break-even.

Run the tests:  uv run python -m pytest tests/unit/test_bet_week_results.py -q

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from typing import Any, cast

import pandas as pd

from api.cache import (
    BET_STATUS_LIVE,
    BET_WEEK_CLASS_ORDER,
    BET_WEEK_RESULTS_COLUMNS,
    GRADING_STATUS_LOSS,
    GRADING_STATUS_PENDING,
    GRADING_STATUS_PUSH,
    GRADING_STATUS_WIN,
    PROVENANCE_FORWARD,
)
from backtest.bet_tracker import _flat_return_units, prepare_tracker_rows
from utils import get_logger

logger = get_logger(__name__)

__all__ = [
    "BET_WEEK_CLASS_ORDER",
    "aggregate_bet_weeks",
]

# The bet-list columns read here beyond the tracker's own (``prepare_tracker_rows`` checks those).
_WEEK_REQUIRED_COLUMNS: tuple[str, ...] = ("season", "week", "per_bet_ev")

# The grading statuses that mean the result is KNOWN. ``pending`` is deliberately absent.
_GRADED_STATUSES: tuple[str, ...] = (
    GRADING_STATUS_WIN,
    GRADING_STATUS_LOSS,
    GRADING_STATUS_PUSH,
)

# The two figures that are NULL when not measured; kept as object dtype so None never becomes NaN.
_RETURN_COLUMNS: tuple[str, ...] = ("actual_return_units", "expected_return_units")


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=pd.Index(BET_WEEK_RESULTS_COLUMNS))


def _expected_return_units(graded: pd.DataFrame) -> float | None:
    """``sum(per_bet_ev x flat_stake) / sum(flat_stake)`` over *graded*, or ``None``.

    ``None`` when nothing is graded, when ANY graded row carries no ``per_bet_ev`` (an average over
    the rest would be a different population from the actual return's), or when no stake was made.
    The stake handling is the actual return's own: a NULL stake counts as 0.0.
    """
    if graded.empty or bool(graded["per_bet_ev"].isna().any()):
        return None
    stakes = graded["flat_stake"].fillna(0.0).astype(float)
    total_stake = float(stakes.sum())
    if total_stake <= 0:
        return None
    return float((graded["per_bet_ev"].astype(float) * stakes).sum() / total_stake)


def _class_position(counting_class: str) -> int:
    """*counting_class*'s index in ``BET_WEEK_CLASS_ORDER``; a class outside it is refused."""
    if counting_class not in BET_WEEK_CLASS_ORDER:
        msg = (
            f"aggregate_bet_weeks: counting class {counting_class!r} is not in "
            f"BET_WEEK_CLASS_ORDER {BET_WEEK_CLASS_ORDER}; a week row cannot be ordered by a "
            "class nobody declared, and ordering it silently would misplace it."
        )
        raise ValueError(msg)
    return BET_WEEK_CLASS_ORDER.index(counting_class)


def _week_record(
    season: int, week: int, counting_class: str, rows: pd.DataFrame
) -> dict[str, Any]:
    """One stored row for the live forward *rows* of one week and counting class."""
    statuses = cast(pd.Series, rows["grading_status"])
    graded = cast(pd.DataFrame, rows[statuses.isin(_GRADED_STATUSES)])
    return {
        "season": season,
        "week": week,
        "counting_class": counting_class,
        "wins": int((statuses == GRADING_STATUS_WIN).sum()),
        "losses": int((statuses == GRADING_STATUS_LOSS).sum()),
        "pushes": int((statuses == GRADING_STATUS_PUSH).sum()),
        "bets_graded": len(graded),
        "bets_ungraded": int((statuses == GRADING_STATUS_PENDING).sum()),
        "actual_return_units": None if graded.empty else _flat_return_units(graded),
        "expected_return_units": _expected_return_units(graded),
    }


def aggregate_bet_weeks(
    bet_list_df: pd.DataFrame,
    *,
    corrections: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One actual-versus-expected row per ``(season, week, counting class)`` of live forward bets.

    The rows are first passed through :func:`backtest.bet_tracker.prepare_tracker_rows` (shadow
    rows dropped, pre-verdict rows relabelled, in-force corrections overlaid); only forward rows
    whose ``status`` is live are kept, so suppressed and backtest-replay rows never enter a week.
    The counting class is the prepared ``validation_type`` (``forward_realized`` or
    ``pre_verdict``), so no figure pools the counted and the pre-verdict classes.

    Args:
        bet_list_df: Bet-list rows carrying the tracker's required columns plus ``season``,
            ``week`` and ``per_bet_ev``.
        corrections: The in-force corrections (see :func:`prepare_tracker_rows`). ``None`` or empty
            applies none.

    Returns:
        A frame in ``api.cache.BET_WEEK_RESULTS_COLUMNS`` order, sorted by season, week and then
        ``BET_WEEK_CLASS_ORDER``, whose two return columns are object dtype with ``None`` for
        NULL. An input with no live forward row returns an empty frame with those columns.

    Raises:
        KeyError: on an absent required column.
        ValueError: on a counting class outside ``BET_WEEK_CLASS_ORDER``, or as
            :func:`prepare_tracker_rows` raises.
    """
    missing = [c for c in _WEEK_REQUIRED_COLUMNS if c not in bet_list_df.columns]
    if missing:
        msg = (
            f"aggregate_bet_weeks: bet_list frame missing required column(s) {missing}; a week "
            "row reads stored values and cannot substitute for an absent one."
        )
        raise KeyError(msg)

    prepared = prepare_tracker_rows(bet_list_df, corrections=corrections)
    live_forward = cast(
        pd.DataFrame,
        prepared[
            (prepared["provenance"] == PROVENANCE_FORWARD)
            & (prepared["status"] == BET_STATUS_LIVE)
        ],
    )
    if live_forward.empty:
        return _empty_frame()

    records = [
        _week_record(int(season), int(week), str(counting_class), rows)
        for (season, week, counting_class), rows in live_forward.groupby(
            ["season", "week", "validation_type"], sort=False, dropna=False
        )
    ]
    records.sort(
        key=lambda r: (
            r["season"],
            r["week"],
            _class_position(r["counting_class"]),
        )
    )

    frame = pd.DataFrame(records, columns=pd.Index(BET_WEEK_RESULTS_COLUMNS))
    for column in _RETURN_COLUMNS:
        frame[column] = frame[column].astype(object).where(frame[column].notna(), None)
    logger.info("Aggregated per-week actual vs expected rows", rows=len(frame))
    return frame
