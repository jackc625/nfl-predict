"""Closing-reading wake windows come only from real kickoffs (Phase 34, LDGR-07, D-07).

A window opens ``READING_LEAD`` before the earliest kickoff it serves and serves every kickoff no
more than 60:00 after its start, so the one reading it takes is in-window for each of them. The
edge cases run on synthetic kickoffs and never skip; the whole-season checks run on the recorded
2026 schedule (``tests/fixtures/season_2026``, read-only) and skip by name when it is absent.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pandas as pd
import pytest
from forward_ledger.closing_windows import (
    MAX_TRIGGERS,
    READING_LEAD,
    ReadingWindow,
    reading_windows,
    windows_for_schedule,
)

from forward_ledger.closing import CLOSING_WINDOW
from tests.fixtures.season_2026 import (
    CapturedScheduleUnavailableError,
    transform_captured_schedule,
)
from utils.date_utils import ET, kickoff_wall_clock_et


def _et(year: int, month: int, day: int, hour: int, minute: int) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=ET)


@pytest.fixture(scope="module")
def schedule_2026() -> pd.DataFrame:
    try:
        return transform_captured_schedule()
    except CapturedScheduleUnavailableError as error:
        pytest.skip(f"the recorded 2026 schedule is absent on this checkout: {error}")


@pytest.fixture(scope="module")
def season_windows(schedule_2026: pd.DataFrame) -> list[ReadingWindow]:
    kickoffs = dict(
        zip(schedule_2026["game_id"], schedule_2026["kickoff_et"], strict=True)
    )
    return reading_windows(kickoffs)


def _window_of(windows: list[ReadingWindow], game_id: str) -> ReadingWindow:
    owners = [window for window in windows if game_id in window.game_ids]
    assert len(owners) == 1, f"{game_id} is served by {len(owners)} windows"
    return owners[0]


def test_shared_window_405_425() -> None:
    windows = reading_windows(
        {"late_a": _et(2026, 10, 18, 16, 5), "late_b": _et(2026, 10, 18, 16, 25)}
    )
    assert len(windows) == 1
    assert windows[0].start_utc == _et(2026, 10, 18, 15, 55)
    assert windows[0].start_utc.tzinfo is not None
    assert windows[0].game_ids == ("late_a", "late_b")


def test_split_window_mnf_doubleheader() -> None:
    windows = reading_windows(
        {"early": _et(2026, 10, 19, 19, 15), "late": _et(2026, 10, 19, 20, 15)}
    )
    assert [window.game_ids for window in windows] == [("early",), ("late",)]
    assert [window.start_utc for window in windows] == [
        _et(2026, 10, 19, 19, 5),
        _et(2026, 10, 19, 20, 5),
    ]


def test_every_kickoff_served_within_60(
    schedule_2026: pd.DataFrame, season_windows: list[ReadingWindow]
) -> None:
    for game_id, kickoff in zip(
        schedule_2026["game_id"], schedule_2026["kickoff_et"], strict=True
    ):
        window = _window_of(season_windows, game_id)
        served = window.kickoffs_utc[window.game_ids.index(game_id)]
        assert served == kickoff_wall_clock_et(kickoff)
        assert READING_LEAD <= served - window.start_utc <= CLOSING_WINDOW


def test_non_standard_slots_covered(season_windows: list[ReadingWindow]) -> None:
    slots = {
        "2026_W05_PHI@JAX": _et(2026, 10, 11, 9, 30),  # international, Sunday morning
        "2026_W15_SEA@PHI": _et(2026, 12, 19, 17, 0),  # Saturday
        "2026_W12_CHI@DET": _et(2026, 11, 26, 13, 0),  # Thanksgiving
        "2026_W12_PHI@DAL": _et(2026, 11, 26, 16, 30),
        "2026_W12_KC@BUF": _et(2026, 11, 26, 20, 20),
        "2026_W16_GB@CHI": _et(2026, 12, 25, 13, 0),  # Christmas
        "2026_W12_GB@LA": _et(2026, 11, 25, 20, 0),  # Wednesday
        "2026_W01_NE@SEA": _et(2026, 9, 9, 20, 20),  # Wednesday opener
    }
    for game_id, kickoff in slots.items():
        assert _window_of(season_windows, game_id).start_utc == kickoff - READING_LEAD

    thanksgiving = [
        window
        for window in season_windows
        if window.start_utc.astimezone(ET).date() == date(2026, 11, 26)
    ]
    assert len(thanksgiving) == 3


def test_no_window_without_a_game(season_windows: list[ReadingWindow]) -> None:
    assert reading_windows({}) == []
    for window in season_windows:
        assert window.game_ids
        assert all(
            window.start_utc < kickoff <= window.start_utc + CLOSING_WINDOW
            for kickoff in window.kickoffs_utc
        )


def test_eight_day_horizon_cap() -> None:
    now = datetime(2026, 10, 5, 21, 0, tzinfo=UTC)
    nan = float("nan")
    rows = [
        ("kicked_off", now - timedelta(minutes=30), nan),
        ("played", now - timedelta(days=1), 24.0),
        ("soon", now + timedelta(hours=5), nan),
        ("in_horizon", now + timedelta(days=7, hours=23), nan),
        ("beyond_horizon", now + timedelta(days=8, minutes=1), nan),
    ]
    games = pd.DataFrame(
        [
            {
                "game_id": game_id,
                "kickoff_et": kickoff,
                "home_score": score,
                "away_score": score,
            }
            for game_id, kickoff, score in rows
        ]
    )
    windows = windows_for_schedule(games, now, days=8)
    served = {game_id for window in windows for game_id in window.game_ids}
    assert served == {"soon", "in_horizon"}
    assert all(now < window.start_utc <= now + timedelta(days=8) for window in windows)

    crowded = pd.DataFrame(
        [
            {
                "game_id": f"g{index:02d}",
                "kickoff_et": now + timedelta(hours=2 * (index + 1)),
                "home_score": nan,
                "away_score": nan,
            }
            for index in range(80)
        ]
    )
    capped = windows_for_schedule(crowded, now, days=8)
    assert len(capped) == MAX_TRIGGERS
    assert capped == sorted(capped, key=lambda window: window.start_utc)
    assert capped[0].game_ids == ("g00",)
