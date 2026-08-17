"""The date_utils contracts that live automation depends on.

Two Phase-29 review findings, both pinned here against a FROZEN clock so the
assertions are about arithmetic rather than about the day the suite happens to run:

* **WR-05** -- ``get_current_nfl_week`` returned a week ONE TOO HIGH on Thursday,
  Friday, Saturday and Sunday (every game day except Monday) for every week of the
  season. Every consumer uses the value verbatim -- ``pipeline/steps.py`` passes it
  straight into ``generate_and_write(season=..., week=...)`` and names the output
  file after it -- so the Friday orchestrator was generating NEXT week's
  predictions. No test pinned the old behaviour, which is why it survived.
* **WR-04** -- ``parse_nfl_date``'s dateutil fallback called ``ET.localize(dt)``,
  the pytz API on a ``zoneinfo.ZoneInfo``, so it raised ``AttributeError`` instead
  of the ``ValueError`` its own docstring documents.
"""

from datetime import datetime, timedelta

import pytest

from utils.date_utils import (
    ET,
    NFL_REGULAR_SEASON_WEEKS,
    NFL_TOTAL_WEEKS,
    get_current_nfl_week,
    get_nfl_season_start,
    parse_nfl_date,
)


def _et(date_str: str, hour: int = 12) -> datetime:
    """A frozen ET instant on *date_str*."""
    return datetime.fromisoformat(date_str).replace(hour=hour, tzinfo=ET)


class TestCurrentNflWeekAcrossEveryWeekday:
    """The 2024 season opens Thursday 2024-09-05 (Thursday after Labor Day)."""

    # All seven weekdays across two week boundaries. Weeks transition on TUESDAY,
    # so the opening Thursday through the following Monday are all week 1.
    @pytest.mark.parametrize(
        ("date_str", "weekday", "expected_week"),
        [
            ("2024-09-05", "Thu", 1),
            ("2024-09-06", "Fri", 1),
            ("2024-09-07", "Sat", 1),
            ("2024-09-08", "Sun", 1),
            ("2024-09-09", "Mon", 1),
            ("2024-09-10", "Tue", 2),
            ("2024-09-11", "Wed", 2),
            ("2024-09-12", "Thu", 2),
            ("2024-09-15", "Sun", 2),
            ("2024-09-16", "Mon", 2),
            ("2024-09-17", "Tue", 3),
            ("2024-11-08", "Fri", 10),
        ],
    )
    def test_week_is_correct_on_every_weekday(
        self, date_str: str, weekday: str, expected_week: int
    ) -> None:
        now = _et(date_str)
        assert now.strftime("%a") == weekday, "fixture date/weekday disagree"

        season, week = get_current_nfl_week(now)

        assert (season, week) == (2024, expected_week), (
            f"{weekday} {date_str} should be week {expected_week}; the pre-WR-05 "
            f"code returned one week high on Thu/Fri/Sat/Sun"
        )

    @pytest.mark.parametrize("season", [2020, 2026])
    def test_labor_day_shifted_seasons(self, season: int) -> None:
        """2020 and 2026 are the seasons where Sept 1 falls on a Tuesday, so the
        first Thursday precedes Labor Day. Both open 2020-09-10 / 2026-09-10."""
        season_start = get_nfl_season_start(season)
        assert season_start.day == 10

        # Opening Thursday through the following Monday: week 1.
        for offset_days in range(0, 5):
            now = season_start.replace(hour=12) + timedelta(days=offset_days)
            assert get_current_nfl_week(now) == (season, 1), (
                f"{now:%a %Y-%m-%d} should be week 1 of {season}"
            )

        # The following Tuesday: week 2.
        now = season_start.replace(hour=12) + timedelta(days=5)
        assert now.strftime("%a") == "Tue"
        assert get_current_nfl_week(now) == (season, 2)

    def test_preseason_guard_is_unchanged(self) -> None:
        """Deliberately preserved. Anchoring two days earlier without this guard
        would flip the pre-season Tuesday and Wednesday from the previous season's
        week 18 to the new season's week 1 -- a behaviour change nobody asked for."""
        assert get_current_nfl_week(_et("2024-08-20")) == (
            2023,
            NFL_REGULAR_SEASON_WEEKS,
        )
        # The Tuesday and Wednesday immediately before kickoff are the edge the
        # guard exists to hold.
        assert get_current_nfl_week(_et("2024-09-03")) == (
            2023,
            NFL_REGULAR_SEASON_WEEKS,
        )
        assert get_current_nfl_week(_et("2024-09-04")) == (
            2023,
            NFL_REGULAR_SEASON_WEEKS,
        )

    def test_week_is_clamped_to_the_total_week_count(self) -> None:
        """Deep into February, the bucket arithmetic must not run past week 22."""
        assert get_current_nfl_week(_et("2025-02-20")) == (2024, NFL_TOTAL_WEEKS)

    def test_default_now_still_works_with_no_argument(self) -> None:
        """All twenty call sites pass nothing; the parameter is purely additive."""
        season, week = get_current_nfl_week()
        assert 1 <= week <= NFL_TOTAL_WEEKS
        assert 2000 < season < 2100


class TestParseNflDateFallback:
    """WR-04: the fallback path reaches its documented ValueError."""

    def test_month_name_format_returns_an_et_datetime(self) -> None:
        parsed = parse_nfl_date("Sep 5 2024")

        assert parsed.tzinfo is ET
        assert (parsed.year, parsed.month, parsed.day) == (2024, 9, 5)

    def test_unparseable_string_raises_valueerror_not_attributeerror(self) -> None:
        with pytest.raises(ValueError, match="Unable to parse date string"):
            parse_nfl_date("not a date at all")

    def test_regex_paths_are_unchanged(self) -> None:
        assert parse_nfl_date("2024-09-05").day == 5
        assert parse_nfl_date("09/05/2024").month == 9
        assert parse_nfl_date("9/5/2024").day == 5
