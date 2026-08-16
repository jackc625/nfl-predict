"""Tests for NFL season-start derivation in utils/date_utils.py.

The season opens on the Thursday after Labor Day (the first Monday in
September). The previous implementation used a "first Thursday in September"
heuristic, which agrees with the real rule in most years but runs a week early
whenever Sept 1 falls on a Tuesday -- the first Thursday (Sept 3) then precedes
Labor Day (Sept 7).

That divergence hits 2020 and 2026. It mis-keyed ~18% of the paid
``odds_timeline`` archive (every 2020 game_id derived a week high) and would
have shifted every live 2026 week number consumed by scripts/friday_pipeline.py
and pipeline/staleness.py.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from utils.date_utils import (
    ET,
    NFL_SEASON_START_DAY_RANGE,
    get_nfl_season_start,
)

# Verified NFL kickoff Thursdays (Thursday after Labor Day).
KNOWN_KICKOFFS = {
    2018: (2018, 9, 6),
    2019: (2019, 9, 5),
    2020: (2020, 9, 10),
    2021: (2021, 9, 9),
    2022: (2022, 9, 8),
    2023: (2023, 9, 7),
    2024: (2024, 9, 5),
    2025: (2025, 9, 4),
    2026: (2026, 9, 10),
}


@pytest.mark.parametrize("season", sorted(KNOWN_KICKOFFS))
def test_matches_known_kickoff_dates(season: int) -> None:
    """Derived kickoff matches the real schedule for every recent season."""
    year, month, day = KNOWN_KICKOFFS[season]
    assert get_nfl_season_start(season) == datetime(year, month, day, tzinfo=ET)


@pytest.mark.parametrize("season", [2020, 2026])
def test_tuesday_sept_first_seasons_do_not_run_a_week_early(season: int) -> None:
    """Regression: the seasons where Sept 1 is a Tuesday.

    These are exactly the years the old "first Thursday" heuristic got wrong.
    A Sept 3 result here means the bug is back and every derived week number
    for the season is one too high.
    """
    start = get_nfl_season_start(season)
    assert datetime(season, 9, 1, tzinfo=ET).weekday() == 1, (
        "premise: Sept 1 is a Tuesday"
    )
    assert start.day != 3, f"{season} kickoff regressed to the pre-Labor-Day Thursday"
    assert start == datetime(season, 9, 10, tzinfo=ET)


@pytest.mark.parametrize("season", range(2000, 2051))
def test_kickoff_is_always_thursday_after_labor_day(season: int) -> None:
    """The Labor Day invariant holds across a wide span of seasons."""
    start = get_nfl_season_start(season)

    sept_first = datetime(season, 9, 1, tzinfo=ET)
    labor_day = sept_first + timedelta(days=(0 - sept_first.weekday()) % 7)

    assert start.weekday() == 3, "kickoff must be a Thursday"
    assert start == labor_day + timedelta(days=3)
    assert start > labor_day, "kickoff must follow Labor Day, never precede it"


@pytest.mark.parametrize("season", range(2000, 2051))
def test_kickoff_stays_inside_the_declared_day_window(season: int) -> None:
    """Kickoff always lands in the Sept 4-10 window the constant declares."""
    earliest, latest = NFL_SEASON_START_DAY_RANGE
    assert earliest <= get_nfl_season_start(season).day <= latest


def test_returns_eastern_time_aware_datetime() -> None:
    """Callers fence on ET; a naive or UTC-stamped result would shift the fence."""
    start = get_nfl_season_start(2026)
    assert start.tzinfo is not None
    assert start.tzname() in {"EDT", "EST"}
