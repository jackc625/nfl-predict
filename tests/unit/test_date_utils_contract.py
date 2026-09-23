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

import pandas as pd
import pytest

from utils.date_utils import (
    ET,
    NFL_REGULAR_SEASON_WEEKS,
    NFL_TOTAL_WEEKS,
    ensure_utc_aware,
    get_current_nfl_week,
    get_nfl_season_start,
    kickoff_wall_clock_et,
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

    def test_the_preseason_holds_until_the_openers_lock_day(self) -> None:
        """The pre-season contract holds up to, and not including, the opener's LOCK DAY.

        Was: ``test_preseason_guard_is_unchanged``, which also pinned Wednesday
        2024-09-04 to (2023, 18). That Wednesday is the 2024 opener's lock day
        (``2024_W01_BAL@KC`` kicks off Thursday 2024-09-05 and locks Wednesday 18:00 ET,
        D33.2-01), so the old pin asserted the step-24b defect: a daily run at that lock
        resolved the PREVIOUS season and omitted the opener. The slate is now read from the
        schedule (step 24c); the days before the lock day keep the documented value.
        """
        assert get_current_nfl_week(_et("2024-08-20")) == (
            2023,
            NFL_REGULAR_SEASON_WEEKS,
        )
        # The Tuesday before a Thursday opener is still the pre-season.
        assert get_current_nfl_week(_et("2024-09-03")) == (
            2023,
            NFL_REGULAR_SEASON_WEEKS,
        )
        # The opener's lock day is week 1.
        assert get_current_nfl_week(_et("2024-09-04")) == (2024, 1)

    def test_week_is_clamped_to_the_total_week_count(self) -> None:
        """Deep into February, the bucket arithmetic must not run past week 22."""
        assert get_current_nfl_week(_et("2025-02-20")) == (2024, NFL_TOTAL_WEEKS)

    def test_default_now_still_works_with_no_argument(self) -> None:
        """All twenty call sites pass nothing; the parameter is purely additive."""
        season, week = get_current_nfl_week()
        assert 1 <= week <= NFL_TOTAL_WEEKS
        assert 2000 < season < 2100


class TestKickoffWallClockAccessor:
    """WR-06: the ONE accessor for games.kickoff_et, and its three semantics."""

    def test_a_true_utc_instant_is_converted(self) -> None:
        # A 1 PM ET kickoff on 2020-09-13 is stored as 17:00 UTC (EDT).
        assert (
            kickoff_wall_clock_et(pd.Timestamp("2020-09-13 17:00:00+00:00")).hour == 13
        )

    def test_an_et_aware_value_is_an_identity(self) -> None:
        # The dtype the DuckDB games copy carries.
        assert (
            kickoff_wall_clock_et(pd.Timestamp("2020-09-13 13:00:00-04:00")).hour == 13
        )

    def test_a_naive_value_is_localized_as_et_matching_the_writer(self) -> None:
        """Deliberately NOT ensure_utc_aware's naive-equals-UTC convention.

        The accessor has to match the WRITER -- GameSchema.validate_timestamps
        (data/schemas.py:98-102) ET-localizes a naive kickoff. Matching the other
        helper in the same module would shift every naive kickoff by four or five
        hours.
        """
        assert kickoff_wall_clock_et(pd.Timestamp("2020-11-01 20:20:00")).hour == 20

    def test_the_two_naive_conventions_really_do_differ(self) -> None:
        """Pins the difference the accessor's docstring calls load-bearing."""
        naive = datetime(2020, 11, 1, 20, 20)
        assert (
            kickoff_wall_clock_et(naive).utcoffset()
            != ensure_utc_aware(naive).utcoffset()
        )

    def test_a_night_game_never_becomes_a_phantom_friday(self) -> None:
        """The catastrophic wrong direction, pinned as a should-never-happen.

        A Sunday-night 20:20 ET kickoff is stored as 01:20 UTC the next day.
        Relabelling that instant as ET (rather than converting) would move 156
        Thursday/Sunday/Monday night games onto the following day -- and for those
        nearest the boundary, onto a phantom FRIDAY that a Friday-18:00-ET freeze
        would then fence roughly eighteen hours AFTER the real kickoff.
        """
        converted = kickoff_wall_clock_et(pd.Timestamp("2023-11-27 01:20:00+00:00"))
        assert converted.strftime("%a") == "Sun"
        assert (converted.hour, converted.minute) == (20, 20)


class TestWeekdayFamilyIsCorrectForEitherDtype:
    """N-02: the weekday family must not depend on which games copy was resolved.

    ``detect_short_week`` emits thursday_game, monday_game, saturday_game, short_week
    and game_day_of_week -- all five land in gold. Before WR-06 it read the raw cell,
    so it was correct on the ET-typed DuckDB table and WRONG for 718 of 6,499 rows on
    the UTC-typed parquet: 286 Sunday-night games read as Monday, 212 Monday-night as
    Tuesday, 153 Thursday-night as Friday. That is one ``db.table_exists()`` away from
    silently breaking 11 percent of gold, which is why it is pinned for BOTH dtypes
    rather than for whichever copy happens to be resolved today.
    """

    @staticmethod
    def _family(kickoff: pd.Timestamp) -> dict:
        from features.contextual import ContextualFeaturesCalculator

        return ContextualFeaturesCalculator().detect_short_week(
            kickoff_wall_clock_et(kickoff), 2023, 1
        )

    def test_thursday_night_reads_as_thursday_from_a_utc_typed_value(self) -> None:
        # TNF 20:15 ET on Thursday 2023-09-07 == 2023-09-08 00:15 UTC.
        family = self._family(pd.Timestamp("2023-09-08 00:15:00+00:00"))
        assert family["thursday_game"] == 1.0
        assert family["game_day_of_week"] == 3.0
        assert family["short_week"] == 1.0

    def test_thursday_night_reads_as_thursday_from_an_et_typed_value(self) -> None:
        family = self._family(pd.Timestamp("2023-09-07 20:15:00-04:00"))
        assert family["thursday_game"] == 1.0
        assert family["game_day_of_week"] == 3.0

    def test_monday_night_reads_as_monday_from_a_utc_typed_value(self) -> None:
        # MNF 20:15 ET on Monday 2023-09-11 == 2023-09-12 00:15 UTC.
        family = self._family(pd.Timestamp("2023-09-12 00:15:00+00:00"))
        assert family["monday_game"] == 1.0
        assert family["game_day_of_week"] == 0.0
        assert family["short_week"] == 1.0

    def test_monday_night_reads_as_monday_from_an_et_typed_value(self) -> None:
        family = self._family(pd.Timestamp("2023-09-11 20:15:00-04:00"))
        assert family["monday_game"] == 1.0

    def test_sunday_night_does_not_read_as_monday(self) -> None:
        """The single largest pre-fix error class: 286 SNF games read as Monday."""
        family = self._family(pd.Timestamp("2023-11-27 01:20:00+00:00"))
        assert family["monday_game"] == 0.0
        assert family["game_day_of_week"] == 6.0

    def test_saturday_night_does_not_read_as_sunday(self) -> None:
        # 20:15 ET Saturday 2023-12-16 == 2023-12-17 01:15 UTC.
        family = self._family(pd.Timestamp("2023-12-17 01:15:00+00:00"))
        assert family["saturday_game"] == 1.0
        assert family["game_day_of_week"] == 5.0

    def test_friday_afternoon_reads_as_friday(self) -> None:
        """The Black Friday shape CR-01 turns on: 15:00 ET Friday == 20:00 UTC."""
        family = self._family(pd.Timestamp("2023-11-24 20:00:00+00:00"))
        assert family["game_day_of_week"] == 4.0
        assert family["saturday_game"] == 0.0


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
