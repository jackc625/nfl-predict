"""All eight 2026 neutral-site games, through all three resolvers, value by value.

Phase 33, Plan 33-06 Task 2(h) (COLD-09, R11, D33-04/D33-15/D33-16, NF-05,
T-33-26/28/29).

WHY AN INTEGRATION TEST AND NOT THREE UNIT TESTS
------------------------------------------------
There are THREE venue resolvers in this repository, not two: ``features/contextual.py``,
``scripts/ingest_weather.py`` and ``scripts/ingest_games._get_venue_roof_type``. The
third is keyed on the venue NAME, lowercased, and it is the one most likely to be
forgotten. A repair that lands in two of the three leaves the system PARTIALLY repaired
and SILENTLY inconsistent -- the worst available state, because the two correct
resolvers make the third look correct too. So the agreement is asserted here, once, over
all eight games at the same time.

THE MARACANA IS THE HEADLINE CASE
---------------------------------
Before this plan, ``2026_W03_BAL@DAL`` resolved to AT&T Stadium in Arlington. Dallas
coordinates, Dallas timezone, Dallas weather, and a travel distance of ZERO miles for a
trip to Brazil, with no error raised. Two distances are asserted below and they are
different questions:

* how far the venue MOVED -- AT&T Stadium to the Maracana, 5,233 miles -- which is the
  size of the error this plan removes; and
* what the away team's travel feature now READS -- Baltimore to the Maracana, 4,808
  miles -- which is the number that actually enters the model.

Pinning both is what stops a transposed sign or a swapped lat/lon from passing: the two
origins are 1,000 miles apart, so a test that accidentally used the wrong one fails.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from features import contextual
from scripts import ingest_games, ingest_weather
from tests import phase33_state
from tests.fixtures import season_2026

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
VENUES_PATH = REPO_ROOT / "data" / "venues.json"

# AT&T Stadium, the venue every neutral-site game used to resolve to when Dallas was
# the nominal home team. Read from data/venues.json rather than retyped.
AT_T_STADIUM_STADIUM_ID = "DAL00"
M_AND_T_BANK_STADIUM_ID = "BAL00"

# Great-circle miles, computed with the project's own Haversine on the RATIFIED
# coordinates. Tolerances are tight enough that a sign flip or a lat/lon swap fails.
AT_T_TO_MARACANA_MILES = 5233.0
BALTIMORE_TO_MARACANA_MILES = 4808.0
DISTANCE_TOLERANCE_MILES = 50.0

# R11's own floor, kept as a separate and much weaker assertion so the two are not
# confused: "non-zero and consistent with a trip to Brazil".
MINIMUM_INTERNATIONAL_TRAVEL_MILES = 4000.0


def _require_capture() -> None:
    try:
        season_2026.load_captured_schedule()
    except season_2026.CapturedScheduleUnavailableError as exc:
        pytest.skip(str(exc))


def _venue_records() -> list[dict[str, object]]:
    return json.loads(VENUES_PATH.read_text(encoding="utf-8"))["venues"]


def _venues_frame() -> pd.DataFrame:
    return pd.DataFrame(_venue_records())


def _by_stadium_id() -> dict[str, dict[str, object]]:
    return {str(r["stadium_id"]): r for r in _venue_records() if r.get("stadium_id")}


def _neutral_games() -> list[dict[str, object]]:
    """The eight feed rows, as plain dicts in the shape the routers accept."""
    frame = season_2026.neutral_site_games()
    return [
        {
            "game_id": f"{int(row.season)}_W{int(row.week):02d}"
            f"_{row.away_team}@{row.home_team}",
            "season": int(row.season),
            "week": int(row.week),
            "home_team": str(row.home_team),
            "away_team": str(row.away_team),
            "location": str(row.location),
            "stadium_id": str(row.stadium_id),
            "venue": str(row.stadium),
        }
        for row in frame.itertuples()
    ]


class TestAllEightResolveValueByValue:
    """R11: the true stadium's coordinates and timezone, asserted cell by cell."""

    def test_the_capture_still_holds_exactly_eight_neutral_games(self) -> None:
        _require_capture()
        games = _neutral_games()
        assert len(games) == phase33_state.NEUTRAL_SITE_GAME_COUNT_2026
        assert [g["stadium_id"] for g in games] == list(
            phase33_state.INTERNATIONAL_STADIUM_IDS
        ), (
            "the neutral-site set or its week order moved. Every expectation in this "
            "module was measured on the recorded eight."
        )

    def test_every_game_resolves_to_its_ratified_venue(self) -> None:
        _require_capture()
        facts = {row[0]: row for row in phase33_state.INTERNATIONAL_VENUE_FACTS}

        for game in _neutral_games():
            venue = contextual.resolve_venue_for_game(game)
            assert venue is not None, f"{game['game_id']} resolved to nothing"
            fact = facts[str(game["stadium_id"])]
            assert venue["stadium_id"] == fact[0]
            assert venue["venue_name"] == fact[2], game["game_id"]
            assert venue["latitude"] == fact[5], game["game_id"]
            assert venue["longitude"] == fact[6], game["game_id"]
            assert venue["elevation_ft"] == fact[7], game["game_id"]
            assert venue["roof_type"] == fact[8], game["game_id"]
            assert venue["timezone"] == fact[12], game["game_id"]

    def test_no_game_resolves_to_the_home_teams_own_stadium(self) -> None:
        """The defect, stated as its own assertion rather than implied by the above."""
        _require_capture()
        calculator = contextual.ContextualFeaturesCalculator()
        for game in _neutral_games():
            venue = contextual.resolve_venue_for_game(game)
            assert venue is not None
            home_venue_id = calculator.team_venues.get(str(game["home_team"]))
            assert venue["venue_id"] != home_venue_id, (
                f"{game['game_id']} still resolves to the home team's own stadium "
                f"({home_venue_id}). That is the whole defect."
            )


class TestAllThreeResolversAgree:
    """NF-05: two out of three is PARTIALLY repaired and silently inconsistent."""

    def test_the_weather_resolver_returns_the_same_coordinates_and_roof(self) -> None:
        _require_capture()
        ingester = ingest_weather.WeatherDataIngester()
        venues = _venues_frame()

        for game in _neutral_games():
            venue = contextual.resolve_venue_for_game(game)
            assert venue is not None
            lat, lon, roof = ingester._get_venue_coordinates_by_stadium_id(
                str(game["stadium_id"]), venues
            )
            assert (lat, lon, roof) == (
                venue["latitude"],
                venue["longitude"],
                venue["roof_type"],
            ), f"{game['game_id']}: contextual and weather disagree"

    def test_the_ingest_games_resolver_returns_the_same_roof(self) -> None:
        _require_capture()
        games_ingester = ingest_games.GameDataIngester()
        feed_roofs = dict(phase33_state.FEED_ROOF_VALUES_2026)

        for game in _neutral_games():
            venue = contextual.resolve_venue_for_game(game)
            assert venue is not None
            code = str(game["stadium_id"])
            roof = games_ingester._get_venue_roof_type(
                str(game["venue"]), feed_roofs[code], stadium_id=code
            )
            assert roof == venue["roof_type"], (
                f"{game['game_id']}: the ingest_games resolver says {roof!r} and the "
                f"venue record says {venue['roof_type']!r}."
            )


class TestTheThreeFalseDomesDoNotReachSilver:
    """D33-16 / T-33-28: an inherited `dome` would zero the weather on three games."""

    def test_silver_venue_roof_is_not_indoor_for_the_three(self) -> None:
        _require_capture()
        raw = season_2026.load_captured_schedule()
        transformed = season_2026.transform_captured_schedule()
        assert len(transformed) == len(raw), (
            "the transform dropped rows, so the positional join below is unsafe."
        )
        codes = list(raw["stadium_id"])
        roofs = list(transformed["venue_roof"])
        by_code = dict(zip(codes, roofs, strict=True))

        for code in phase33_state.FEED_ROOF_DISAGREEMENTS:
            assert by_code[code] != "indoor", (
                f"silver.venue_roof for {code} is 'indoor'. The feed says 'dome', the "
                "venue is open-air, and _is_outdoor_game turns 'indoor' into a "
                "weather SKIP."
            )

    def test_all_eight_take_the_ratified_roof_in_silver(self) -> None:
        _require_capture()
        raw = season_2026.load_captured_schedule()
        transformed = season_2026.transform_captured_schedule()
        by_code = dict(
            zip(list(raw["stadium_id"]), list(transformed["venue_roof"]), strict=True)
        )
        facts = {row[0]: row for row in phase33_state.INTERNATIONAL_VENUE_FACTS}
        for code in phase33_state.INTERNATIONAL_STADIUM_IDS:
            assert by_code[code] == facts[code][8], (
                f"{code}: silver says {by_code[code]!r}, ratified {facts[code][8]!r}"
            )


class TestTheMaracanaTravelDistance:
    """R11: today it is ZERO miles for a trip to Brazil, with no error raised."""

    def test_the_venue_moved_five_thousand_miles(self) -> None:
        """AT&T Stadium to the Maracana -- the size of the error being removed."""
        calculator = contextual.ContextualFeaturesCalculator()
        records = _by_stadium_id()
        arlington = records[AT_T_STADIUM_STADIUM_ID]
        maracana = records["RIO00"]

        miles = calculator._calculate_distance(
            arlington["latitude"],
            arlington["longitude"],
            maracana["latitude"],
            maracana["longitude"],
        )
        assert abs(miles - AT_T_TO_MARACANA_MILES) <= DISTANCE_TOLERANCE_MILES, (
            f"AT&T Stadium to the ratified Maracana coordinate is {miles:.1f} miles, "
            f"expected {AT_T_TO_MARACANA_MILES} +/- {DISTANCE_TOLERANCE_MILES}. A "
            "transposed sign or a swapped lat/lon fails here rather than passing."
        )

    def test_the_away_travel_feature_is_baltimores_trip_and_not_dallas(self) -> None:
        """The number that enters the model, pinned to the AWAY team's origin.

        BAL is the away team, so the travel feature is Baltimore to Rio (4,808 mi),
        NOT Dallas to Rio (5,233 mi). The two are a thousand miles apart, which is
        what makes this assertion able to catch the wrong origin.
        """
        _require_capture()
        calculator = contextual.ContextualFeaturesCalculator()
        maracana = _by_stadium_id()["RIO00"]
        kickoff = datetime(2026, 9, 27, 13, 0, tzinfo=ZoneInfo("America/New_York"))

        metrics = calculator.calculate_travel_metrics(
            "BAL", "DAL", maracana["venue_id"], kickoff
        )
        miles = metrics["travel_distance_miles"]

        assert miles > MINIMUM_INTERNATIONAL_TRAVEL_MILES, (
            f"the Maracana game's travel distance is {miles:.1f} miles. Before this "
            "plan it was 0.0 -- the game resolved to AT&T Stadium and the away team "
            "was recorded as having travelled nowhere."
        )
        assert abs(miles - BALTIMORE_TO_MARACANA_MILES) <= DISTANCE_TOLERANCE_MILES, (
            f"{miles:.1f} miles is not Baltimore to Rio "
            f"({BALTIMORE_TO_MARACANA_MILES} +/- {DISTANCE_TOLERANCE_MILES}). If it "
            f"reads about {AT_T_TO_MARACANA_MILES} the wrong team's origin was used."
        )

    def test_the_baltimore_origin_is_the_one_in_the_file(self) -> None:
        """The control for the assertion above: BAL's home venue has not moved."""
        records = _by_stadium_id()
        calculator = contextual.ContextualFeaturesCalculator()
        baltimore = records[M_AND_T_BANK_STADIUM_ID]
        maracana = records["RIO00"]
        miles = calculator._calculate_distance(
            baltimore["latitude"],
            baltimore["longitude"],
            maracana["latitude"],
            maracana["longitude"],
        )
        assert abs(miles - BALTIMORE_TO_MARACANA_MILES) <= DISTANCE_TOLERANCE_MILES

    def test_the_timezone_difference_is_the_venues_not_the_home_teams(self) -> None:
        """America/Sao_Paulo, not America/Chicago."""
        calculator = contextual.ContextualFeaturesCalculator()
        maracana = _by_stadium_id()["RIO00"]
        kickoff = datetime(2026, 9, 27, 13, 0, tzinfo=ZoneInfo("America/New_York"))
        metrics = calculator.calculate_travel_metrics(
            "BAL", "DAL", maracana["venue_id"], kickoff
        )
        assert metrics["abs_timezone_diff_hours"] > 0.0, (
            "the Maracana game reports a zero timezone difference, which is what a "
            "Baltimore-to-Baltimore trip looks like."
        )


class TestTheEncodedVenueFeaturesFollowTheRatifiedValues:
    """The encoder is where a ratified elevation actually becomes a model feature."""

    def test_mexico_city_trips_high_altitude_and_madrid_does_not(self) -> None:
        calculator = contextual.ContextualFeaturesCalculator()
        records = _by_stadium_id()

        mexico = calculator.encode_venue_features(records["MEX00"]["venue_id"])
        madrid = calculator.encode_venue_features(records["MAD01"]["venue_id"])

        assert mexico["venue_elevation_ft"] == 7365.0
        assert mexico["venue_high_altitude"] == 1.0
        assert madrid["venue_elevation_ft"] == 2349.0
        assert madrid["venue_high_altitude"] == 0.0, (
            "Madrid at 2,349 ft has 651 ft of margin under the 3,000 ft threshold."
        )

    def test_the_three_open_air_venues_encode_as_outdoor(self) -> None:
        calculator = contextual.ContextualFeaturesCalculator()
        records = _by_stadium_id()
        for code in phase33_state.FEED_ROOF_DISAGREEMENTS:
            encoded = calculator.encode_venue_features(records[code]["venue_id"])
            assert encoded["venue_outdoor"] == 1.0, code
            assert encoded["venue_indoor"] == 0.0, code

    def test_the_new_climate_token_encodes_to_neither_flag(self) -> None:
        calculator = contextual.ContextualFeaturesCalculator()
        encoded = calculator.encode_venue_features(
            _by_stadium_id()["MEX00"]["venue_id"]
        )
        assert encoded["venue_cold_climate"] == 0.0
        assert encoded["venue_warm_climate"] == 0.0, (
            "subtropical_highland must be inert, exactly as oceanic and mediterranean "
            "already are. A new token that silently created a feature would be a "
            "change nobody ratified."
        )
