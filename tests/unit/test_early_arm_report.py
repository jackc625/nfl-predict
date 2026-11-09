"""The early-week arm's record: the figures the /bets section shows (backtest/early_arm_report.py).

The record counts the early arm's placed bets and nothing else. A figure with nothing behind it is
None, so the page can say "--" instead of a zero nobody measured, and an in-force correction moves
the record the way it moves the retired system's.

Synthetic rows only; nothing is read or written.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import pytest

from api.cache import BET_LIST_COLUMNS, BET_LIST_CORRECTIONS_COLUMNS
from backtest.early_arm_report import early_arm_summary

SUMMARY_KEYS = {
    "bets",
    "graded",
    "wins",
    "losses",
    "pushes",
    "flat_return",
    "closing_bets",
    "mean_clv",
    "closing_share",
}


def _row(
    game_id: str,
    *,
    arm: str | None = "early",
    status: str = "live",
    grading_status: str = "pending",
    payout_flat: float | None = None,
    forward_clv: float | None = None,
) -> dict[str, Any]:
    """One bet-list row carrying the columns the record reads."""
    row: dict[str, Any] = dict.fromkeys(BET_LIST_COLUMNS)
    row.update(
        {
            "game_id": game_id,
            "season": 2026,
            "week": 11,
            "target": "ou",
            "arm": arm,
            "status": status,
            "provenance": "forward",
            "validation_type": "forward_realized",
            "grading_status": grading_status,
            "flat_stake": 1.0,
            "payout_flat": payout_flat,
            "forward_clv": forward_clv,
        }
    )
    return row


def _correction(
    row: dict[str, Any], *, corrected: str, payout_flat: float
) -> dict[str, Any]:
    """The in-force correction of *row*, in the columns the cache stores."""
    correction: dict[str, Any] = dict.fromkeys(BET_LIST_CORRECTIONS_COLUMNS)
    correction.update(
        {
            "game_id": row["game_id"],
            "season": row["season"],
            "week": row["week"],
            "target": row["target"],
            "arm": row["arm"],
            "original_grading_status": row["grading_status"],
            "corrected_grading_status": corrected,
            "corrected_outcome": {"win": True, "loss": False}.get(corrected),
            "corrected_payout_flat": payout_flat,
            "corrected_realized_units": payout_flat,
        }
    )
    return correction


def test_the_summary_counts_only_early_live_rows() -> None:
    rows = [
        _row("2026_W11_AAA@BBB", grading_status="win", payout_flat=0.5),
        _row("2026_W11_CCC@DDD", status="suppressed"),
        _row("2026_W11_EEE@FFF", arm="live", grading_status="win", payout_flat=0.9),
        _row("2026_W11_GGG@HHH", arm="shadow", grading_status="loss", payout_flat=-1.0),
        _row("2025_W11_III@JJJ", arm=None, grading_status="win", payout_flat=0.9),
    ]

    summary = early_arm_summary(pd.DataFrame(rows), None)

    assert set(summary) == SUMMARY_KEYS
    assert (summary["bets"], summary["graded"]) == (1, 1)
    assert (summary["wins"], summary["losses"], summary["pushes"]) == (1, 0, 0)
    assert summary["flat_return"] == pytest.approx(0.5)


def test_every_grade_is_counted_and_the_return_is_over_the_stakes() -> None:
    rows = [
        _row("2026_W11_AAA@BBB", grading_status="win", payout_flat=0.5),
        _row("2026_W11_CCC@DDD", grading_status="loss", payout_flat=-1.0),
        _row("2026_W11_EEE@FFF", grading_status="push", payout_flat=0.0),
        _row("2026_W11_GGG@HHH"),
    ]

    summary = early_arm_summary(pd.DataFrame(rows), None)

    assert (summary["bets"], summary["graded"]) == (4, 3)
    assert (summary["wins"], summary["losses"], summary["pushes"]) == (1, 1, 1)
    assert summary["flat_return"] == pytest.approx(-0.5 / 3)


def test_closing_value_is_the_mean_over_rows_that_have_one() -> None:
    rows = [
        _row(
            "2026_W11_AAA@BBB",
            grading_status="win",
            payout_flat=0.9,
            forward_clv=1.0,
        ),
        _row("2026_W11_CCC@DDD", grading_status="loss", payout_flat=-1.0),
    ]

    summary = early_arm_summary(pd.DataFrame(rows), None)

    assert summary["closing_bets"] == 1
    assert summary["mean_clv"] == pytest.approx(1.0)
    assert summary["closing_share"] == pytest.approx(0.5)


def test_a_game_not_played_yet_is_not_a_missed_close() -> None:
    """The share is of the settled bets: a game still ahead has no close to miss."""
    rows = [
        _row(
            "2026_W11_AAA@BBB",
            grading_status="win",
            payout_flat=0.9,
            forward_clv=-0.5,
        ),
        _row("2026_W12_CCC@DDD"),
    ]

    summary = early_arm_summary(pd.DataFrame(rows), None)

    assert summary["mean_clv"] == pytest.approx(-0.5)
    assert summary["closing_share"] == pytest.approx(1.0)


def test_nothing_graded_is_none_not_zero() -> None:
    summary = early_arm_summary(pd.DataFrame([_row("2026_W11_AAA@BBB")]), None)

    assert summary["bets"] == 1
    assert (summary["graded"], summary["closing_bets"]) == (0, 0)
    assert summary["flat_return"] is None and summary["mean_clv"] is None
    assert summary["closing_share"] is None


def test_settled_bets_with_no_close_are_a_measured_zero_share() -> None:
    """Every game was played and no close was read: that IS a measurement, of 0%."""
    rows = [_row("2026_W11_AAA@BBB", grading_status="loss", payout_flat=-1.0)]

    summary = early_arm_summary(pd.DataFrame(rows), None)

    assert summary["closing_share"] == pytest.approx(0.0)
    assert summary["mean_clv"] is None


@pytest.mark.parametrize(
    "frame",
    [pd.DataFrame(), pd.DataFrame(columns=pd.Index(BET_LIST_COLUMNS))],
    ids=["no_columns", "no_rows"],
)
def test_a_frame_with_no_early_rows_is_none_not_zero(frame: pd.DataFrame) -> None:
    corrections = pd.DataFrame(
        [
            _correction(
                _row("2026_W11_AAA@BBB", grading_status="win"),
                corrected="loss",
                payout_flat=-1.0,
            )
        ]
    )

    summary = early_arm_summary(frame, corrections)

    assert summary == {
        "bets": 0,
        "graded": 0,
        "wins": 0,
        "losses": 0,
        "pushes": 0,
        "flat_return": None,
        "closing_bets": 0,
        "mean_clv": None,
        "closing_share": None,
    }


def test_a_correction_changes_the_record() -> None:
    early = _row("2026_W11_AAA@BBB", grading_status="win", payout_flat=0.9)
    other = _row("2026_W11_AAA@BBB", arm="live", grading_status="win", payout_flat=0.9)
    corrections = pd.DataFrame([_correction(early, corrected="loss", payout_flat=-1.0)])

    before = early_arm_summary(pd.DataFrame([early, other]), None)
    after = early_arm_summary(pd.DataFrame([early, other]), corrections)

    assert (before["wins"], before["losses"]) == (1, 0)
    assert before["flat_return"] == pytest.approx(0.9)
    assert (after["bets"], after["graded"]) == (1, 1)
    assert (after["wins"], after["losses"]) == (0, 1)
    assert after["flat_return"] == pytest.approx(-1.0)


def test_a_correction_to_another_arm_leaves_the_record_alone() -> None:
    early = _row("2026_W11_AAA@BBB", grading_status="win", payout_flat=0.9)
    other = _row("2026_W11_AAA@BBB", arm="live", grading_status="win", payout_flat=0.9)
    corrections = pd.DataFrame([_correction(other, corrected="loss", payout_flat=-1.0)])

    summary = early_arm_summary(pd.DataFrame([early, other]), corrections)

    assert (summary["wins"], summary["losses"]) == (1, 0)
    assert summary["flat_return"] == pytest.approx(0.9)


def test_the_callers_frame_is_not_changed() -> None:
    frame = pd.DataFrame([_row("2026_W11_AAA@BBB", grading_status="win")])
    before = frame.copy()

    early_arm_summary(frame, None)

    pd.testing.assert_frame_equal(frame, before)


def test_the_summary_is_plain_json() -> None:
    """It is stored as one JSON value, so no numpy number may reach it."""
    rows = [
        _row(
            "2026_W11_AAA@BBB",
            grading_status="win",
            payout_flat=0.9,
            forward_clv=1.5,
        ),
    ]

    summary = early_arm_summary(pd.DataFrame(rows), None)

    assert json.loads(json.dumps(summary)) == summary
    assert type(summary["bets"]) is int
    assert type(summary["mean_clv"]) is float
