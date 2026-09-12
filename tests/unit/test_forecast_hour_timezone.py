"""The forecast hour is read in the VENUE's timezone, not in the home team's.

THE DEFECT THIS EXISTS TO CATCH
-------------------------------
A forecast is an ARRAY OF HOURS. Picking the wrong index returns a real, plausible,
internally-consistent weather reading for the wrong moment -- and there is nothing in the
value itself to say so. When the venue is on another continent, the wrong index is also
the wrong DAY.

Eight of 2026's games are international. The Maracana game is nominally a Dallas home
game; read in Dallas's zone its forecast hour is 16, and read in Rio's it is 17. The
Melbourne game is worse: read in Eastern time it is hour 20 on 2026-09-10, and read in
Melbourne it is hour 10 on 2026-09-11 -- a different hour AND a different date. This is
COLD-09's defect expressed in time instead of in space, and the fix has the same shape:
resolve the venue by `stadium_id` and use the venue's own IANA zone.

WHY THE EXPECTED VALUES ARE WRITTEN OUT
---------------------------------------
Every expectation below is a LITERAL hour and a LITERAL date. Deriving them inside the
test from the same conversion the production code performs would make the test agree with
the implementation by construction and assert nothing. They were computed once, on
2026-09-12, from the captured 2026 schedule and the owner-ratified venue records, and are
pinned here.

THE NEGATIVE CONTROL IS NOT DECORATION
--------------------------------------
`test_the_home_team_zone_gives_a_different_answer_for_the_maracana_game` proves this
module can catch the defect it exists for. Without it, every assertion here would be
"the conversion returned something", which a hardcoded Eastern conversion would also
satisfy on the twelve of twenty games that happen to sit in Eastern time.

NO NETWORK, NO WRITES. Hour selection is pure arithmetic over a kickoff instant and an
IANA zone name.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from tests import phase33_state
from tests.fixtures import season_2026

# Every `as_of_utc` below is derived from the game's OWN pinned kickoff, two days
# before it, rather than from one shared constant. The eight international games span
# September to November, so a single `as_of` could not be inside the horizon for all of
# them -- and widening the horizon to make one constant work would be tuning the
# production rule to suit a test. The kickoff is a literal from the pinned capture, so
# this is still fully injected: nothing here reads a process clock or depends on the day
# the suite runs.
AS_OF_LEAD = timedelta(days=2)

# stadium_id -> (expected local date, expected local hour, expected IANA zone).
#
# MEASURED 2026-09-12 from tests.fixtures.season_2026.transform_captured_schedule()
# and data/venues.json. The ET hour is recorded beside each one because it is the
# answer a fixed-timezone implementation would give, and the gap is the point.
#
#   stadium  venue zone             local           ET (the WRONG answer)
#   MEL00    Australia/Melbourne    2026-09-11 10   2026-09-10 20
#   RIO00    America/Sao_Paulo      2026-09-27 17   2026-09-27 16
#   LON02    Europe/London          2026-10-04 14   2026-10-04 09
#   LON00    Europe/London          2026-10-18 14   2026-10-18 09
#   PAR00    Europe/Paris           2026-10-25 14   2026-10-25 09
#   MAD01    Europe/Madrid          2026-11-08 15   2026-11-08 09
#   MUN01    Europe/Berlin          2026-11-15 15   2026-11-15 09
#   MEX00    America/Mexico_City    2026-11-22 19   2026-11-22 20
EXPECTED_FORECAST_HOUR: dict[str, tuple[str, int, str]] = {
    "MEL00": ("2026-09-11", 10, "Australia/Melbourne"),
    "RIO00": ("2026-09-27", 17, "America/Sao_Paulo"),
    "LON02": ("2026-10-04", 14, "Europe/London"),
    "LON00": ("2026-10-18", 14, "Europe/London"),
    "PAR00": ("2026-10-25", 14, "Europe/Paris"),
    "MAD01": ("2026-11-08", 15, "Europe/Madrid"),
    "MUN01": ("2026-11-15", 15, "Europe/Berlin"),
    "MEX00": ("2026-11-22", 19, "America/Mexico_City"),
}

# The ET reading of the same eight kickoffs -- what a fixed-Eastern implementation
# returns. Recorded so the "different answer" assertions compare against a stated
# number rather than merely against "not equal to something".
ET_FORECAST_HOUR: dict[str, tuple[str, int]] = {
    "MEL00": ("2026-09-10", 20),
    "RIO00": ("2026-09-27", 16),
    "LON02": ("2026-10-04", 9),
    "LON00": ("2026-10-18", 9),
    "PAR00": ("2026-10-25", 9),
    "MAD01": ("2026-11-08", 9),
    "MUN01": ("2026-11-15", 9),
    "MEX00": ("2026-11-22", 20),
}


def _require_capture() -> None:
    try:
        season_2026.load_captured_schedule()
    except season_2026.CapturedScheduleUnavailableError as exc:
        pytest.skip(str(exc))


def _as_of_for(game: dict[str, Any]) -> datetime:
    """Two days before THIS game's pinned kickoff, in UTC."""
    from utils.date_utils import kickoff_wall_clock_et

    return kickoff_wall_clock_et(game["kickoff_et"]).astimezone(UTC) - AS_OF_LEAD


def _venue_records() -> dict[str, dict[str, Any]]:
    """The eight owner-ratified international venue records, keyed by `stadium_id`.

    Built from `phase33_state.INTERNATIONAL_VENUE_FACTS` rather than from
    `data/venues.json`, so this module reads the RATIFIED record and a silent edit to
    the json surfaces here as a disagreement.
    """
    fields = (
        "stadium_id",
        "venue_id",
        "venue_name",
        "city",
        "country",
        "latitude",
        "longitude",
        "elevation_ft",
        "roof_type",
        "surface",
        "capacity",
        "climate_zone",
        "timezone",
    )
    return {
        str(row[0]): dict(zip(fields, row, strict=True))
        for row in phase33_state.INTERNATIONAL_VENUE_FACTS
    }


def _neutral_games_by_stadium() -> dict[str, dict[str, Any]]:
    frame = season_2026.transform_captured_schedule()
    neutral = frame[frame["neutral_site"]]
    return {str(row["stadium_id"]): row.to_dict() for _, row in neutral.iterrows()}


class TestTheEightInternationalVenuesResolveToTheirOwnLocalHour:
    """Value-by-value, all eight, against literals rather than a re-derivation."""

    @pytest.mark.parametrize("stadium_id", sorted(EXPECTED_FORECAST_HOUR))
    def test_the_local_date_and_hour_are_the_measured_ones(
        self, stadium_id: str
    ) -> None:
        _require_capture()
        import scripts.ingest_weather as ingest

        game = _neutral_games_by_stadium()[stadium_id]
        venue = _venue_records()[stadium_id]
        expected_date, expected_hour, expected_zone = EXPECTED_FORECAST_HOUR[stadium_id]

        selected = ingest.select_forecast_hour_for_kickoff(
            game, venue, as_of_utc=_as_of_for(game)
        )

        assert selected.timezone == expected_zone
        assert selected.local_date == expected_date
        assert selected.hour == expected_hour

    @pytest.mark.parametrize("stadium_id", sorted(EXPECTED_FORECAST_HOUR))
    def test_the_venue_answer_differs_from_the_eastern_answer(
        self, stadium_id: str
    ) -> None:
        """Every one of the eight would be read at the wrong hour, the wrong date, or
        both, by an implementation that used a fixed Eastern zone."""
        _require_capture()
        import scripts.ingest_weather as ingest

        game = _neutral_games_by_stadium()[stadium_id]
        venue = _venue_records()[stadium_id]
        selected = ingest.select_forecast_hour_for_kickoff(
            game, venue, as_of_utc=_as_of_for(game)
        )

        et_date, et_hour = ET_FORECAST_HOUR[stadium_id]
        assert (selected.local_date, selected.hour) != (et_date, et_hour), (
            f"{stadium_id} reads the same in Eastern time as in its own zone, so this "
            "case cannot distinguish a correct implementation from a hardcoded one."
        )


class TestTheNegativeControl:
    """Proof that this module can catch the defect it exists for."""

    def test_the_home_team_zone_gives_a_different_answer_for_the_maracana_game(
        self,
    ) -> None:
        """The Maracana game is nominally a DALLAS home game.

        Resolved through the home team's own zone (America/Chicago) it reads at hour 15;
        resolved through the venue it reads at hour 17 in Rio. Two hours wrong, on the
        wrong continent, with nothing in the value itself to say so.
        """
        _require_capture()
        import scripts.ingest_weather as ingest

        game = _neutral_games_by_stadium()["RIO00"]
        venue = dict(_venue_records()["RIO00"])
        assert game["home_team"] == "DAL"

        correct = ingest.select_forecast_hour_for_kickoff(
            game, venue, as_of_utc=_as_of_for(game)
        )

        # The SAME game, resolved through the home team's stadium record.
        dallas = dict(venue)
        dallas["timezone"] = "America/Chicago"
        dallas["stadium_id"] = "DAL00"
        wrong = ingest.select_forecast_hour_for_kickoff(
            game, dallas, as_of_utc=_as_of_for(game)
        )

        assert correct.timezone == "America/Sao_Paulo"
        assert correct.hour == 17
        assert wrong.hour == 15
        assert wrong.hour != correct.hour

    def test_selecting_through_the_eastern_zone_misses_melbourne_by_a_day(self) -> None:
        """The worst case: the wrong zone changes the DATE, so the request would ask
        the provider for a different day's hours entirely."""
        _require_capture()
        import scripts.ingest_weather as ingest

        game = _neutral_games_by_stadium()["MEL00"]
        venue = dict(_venue_records()["MEL00"])
        eastern = dict(venue, timezone="America/New_York")

        correct = ingest.select_forecast_hour_for_kickoff(
            game, venue, as_of_utc=_as_of_for(game)
        )
        wrong = ingest.select_forecast_hour_for_kickoff(
            game, eastern, as_of_utc=_as_of_for(game)
        )

        assert correct.local_date == "2026-09-11"
        assert wrong.local_date == "2026-09-10"
        assert correct.hour == 10
        assert wrong.hour == 20


class TestDaylightSavingIsHonouredRatherThanAssumed:
    """A US venue on each side of the 2026-11-01 transition.

    The kickoff INSTANT is held fixed and only the date moves, so the one-hour shift in
    the local reading is the offset change and nothing else. A conversion that attached
    a fixed offset instead of a zone would return the same hour twice.
    """

    HIGHMARK = {
        "stadium_id": "BUF00",
        "venue_id": "highmark_stadium",
        "latitude": 42.7738,
        "longitude": -78.787,
        "roof_type": "outdoor",
        "timezone": "America/New_York",
    }

    def _select(self, kickoff_utc: datetime):
        import scripts.ingest_weather as ingest

        game = {
            "game_id": "2026_W08_KC@BUF",
            "season": 2026,
            "week": 8,
            "home_team": "BUF",
            "away_team": "KC",
            "kickoff_et": kickoff_utc,
            "stadium_id": "BUF00",
            "neutral_site": False,
        }
        return ingest.select_forecast_hour_for_kickoff(
            game, self.HIGHMARK, as_of_utc=kickoff_utc - timedelta(days=2)
        )

    def test_before_the_transition_the_offset_is_daylight_time(self) -> None:
        selected = self._select(datetime(2026, 10, 25, 18, 0, tzinfo=UTC))
        assert selected.local_date == "2026-10-25"
        assert selected.hour == 14

    def test_after_the_transition_the_offset_is_standard_time(self) -> None:
        selected = self._select(datetime(2026, 11, 8, 18, 0, tzinfo=UTC))
        assert selected.local_date == "2026-11-08"
        assert selected.hour == 13


class TestHourSelectionCarriesTheHorizonRefusal:
    """The horizon check is INSIDE hour selection, so it cannot be skipped by a caller
    that reaches for the hour directly."""

    def test_a_beyond_horizon_kickoff_is_refused_during_hour_selection(self) -> None:
        import scripts.ingest_weather as ingest

        kickoff = datetime(2026, 12, 20, 18, 0, tzinfo=UTC)
        game = {
            "game_id": "2026_W16_KC@BUF",
            "season": 2026,
            "week": 16,
            "home_team": "BUF",
            "away_team": "KC",
            "kickoff_et": kickoff,
            "stadium_id": "BUF00",
            "neutral_site": False,
        }
        venue = dict(TestDaylightSavingIsHonouredRatherThanAssumed.HIGHMARK)

        with pytest.raises(ingest.BeyondForecastHorizonError):
            ingest.select_forecast_hour_for_kickoff(
                game,
                venue,
                as_of_utc=kickoff
                - timedelta(days=phase33_state.FORECAST_HORIZON_DAYS + 1),
            )

    def test_a_venue_without_a_timezone_is_refused_by_name(self) -> None:
        """An absent zone must RAISE, not default. Defaulting is how the Maracana game
        would silently be read in Eastern time."""
        import scripts.ingest_weather as ingest

        game = {
            "game_id": "2026_W08_KC@BUF",
            "season": 2026,
            "week": 8,
            "home_team": "BUF",
            "away_team": "KC",
            "kickoff_et": datetime(2026, 10, 25, 18, 0, tzinfo=UTC),
            "stadium_id": "BUF00",
            "neutral_site": False,
        }
        venue = {"stadium_id": "BUF00", "roof_type": "outdoor"}

        with pytest.raises(Exception, match="timezone"):
            ingest.select_forecast_hour_for_kickoff(
                game, venue, as_of_utc=datetime(2026, 10, 23, 18, 0, tzinfo=UTC)
            )
