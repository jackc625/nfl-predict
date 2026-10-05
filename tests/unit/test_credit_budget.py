"""The free-quota credit rule and the written 2026 credit budget (Phase 34, LDGR-07, D-08).

The budget is computed from the recorded 2026 schedule (``tests/fixtures/season_2026``,
read-only): one decision board on every lock day plus one closing board per reading window, and
the worst case with the ingest step's 3x retry bound on every decision day. The test prints the
per-month table so the plan SUMMARY can copy it (Plan 34-21 publishes it in AUTOMATION.md).

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import sys
from datetime import date, datetime

import pandas as pd
import pytest

from forward_ledger.credits import (
    CREDITS_PER_BOARD,
    DECISION_CAPTURE_SAFETY,
    MONTHLY_FREE_QUOTA,
    closing_capture_allowed,
    decision_days,
    monthly_budget,
    remaining_decision_days_in_month,
)
from tests.fixtures.season_2026 import (
    CapturedScheduleUnavailableError,
    transform_captured_schedule,
)
from utils.date_utils import ET


@pytest.fixture(scope="module")
def schedule_2026() -> pd.DataFrame:
    try:
        return transform_captured_schedule()
    except CapturedScheduleUnavailableError as error:
        pytest.skip(f"the recorded 2026 schedule is absent on this checkout: {error}")


def _games(*kickoffs: datetime) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"game_id": f"g{index}", "kickoff_et": kickoff, "home_score": None}
            for index, kickoff in enumerate(kickoffs)
        ]
    )


# Three later-October games on three different days: lock days Oct 7, Oct 10 and Oct 14.
_THREE_DECISION_DAYS = _games(
    datetime(2026, 10, 8, 20, 15, tzinfo=ET),
    datetime(2026, 10, 11, 13, 0, tzinfo=ET),
    datetime(2026, 10, 11, 16, 25, tzinfo=ET),
    datetime(2026, 10, 15, 20, 15, tzinfo=ET),
)
_TODAY = date(2026, 10, 5)


def test_budget_under_quota_every_month(schedule_2026: pd.DataFrame) -> None:
    budget = monthly_budget(schedule_2026)
    header = (
        "| Month | Decision days | Decision credits | Closing windows | Closing credits "
        "| Total | Worst case (decision x3 + closing) |"
    )
    lines = [header, "|---|---|---|---|---|---|---|"]
    for month, row in budget.items():
        lines.append(
            f"| {month} | {row['decision_days']} | {row['decision_credits']} | "
            f"{row['closing_windows']} | {row['closing_credits']} | {row['total']} | "
            f"{row['worst_case_total']} |"
        )
    # Written to stdout so `pytest -rP` shows the table the plan SUMMARY copies.
    sys.stdout.write("\n" + "\n".join(lines) + "\n")

    assert budget, "the 2026 schedule produced no month at all"
    assert sum(row["decision_days"] for row in budget.values()) == len(
        decision_days(schedule_2026)
    )
    for month, row in budget.items():
        assert row["decision_credits"] == CREDITS_PER_BOARD * row["decision_days"]
        assert row["closing_credits"] == CREDITS_PER_BOARD * row["closing_windows"]
        assert row["total"] == row["decision_credits"] + row["closing_credits"]
        assert row["worst_case_total"] == (
            DECISION_CAPTURE_SAFETY * row["decision_credits"] + row["closing_credits"]
        )
        assert row["total"] < MONTHLY_FREE_QUOTA, month
        assert row["worst_case_total"] < MONTHLY_FREE_QUOTA, month


def test_low_credits_skip_closing() -> None:
    assert remaining_decision_days_in_month(_THREE_DECISION_DAYS, _TODAY) == 3
    # reserve = 3 credits x 3 safety x 3 days = 27; refuse when remaining - 3 < 27.
    assert closing_capture_allowed(20, _THREE_DECISION_DAYS, _TODAY) == (
        False,
        "credit_reserve",
    )
    assert closing_capture_allowed(29, _THREE_DECISION_DAYS, _TODAY) == (
        False,
        "credit_reserve",
    )
    assert closing_capture_allowed(30, _THREE_DECISION_DAYS, _TODAY) == (True, None)
    assert closing_capture_allowed(400, _THREE_DECISION_DAYS, _TODAY) == (True, None)


def test_unreadable_header_skips() -> None:
    assert closing_capture_allowed(None, _THREE_DECISION_DAYS, _TODAY) == (
        False,
        "credit_header_unreadable",
    )
