"""`stadium_id`-keyed venue routing: the hard-fail, the case rule, and the season gate.

Phase 33, Plan 33-06 Task 2 (COLD-09, R11, D33-15, T-33-26/29).

THE DEFECT THIS ROUTES AROUND
-----------------------------
All three venue resolvers key off the home team (or, in the contextual builder, off the
venue NAME with a home-team-shaped fallback), so the 2026 Maracana game resolves to
AT&T Stadium in Arlington: Dallas coordinates, Dallas timezone, Dallas weather, and a
travel distance of zero miles for a trip to Brazil, with no error raised.

THE ROUTING RULE (D33-15), STATED ONCE
---------------------------------------
For ``season >= 2026``, a game the feed marks neutral resolves BY ``stadium_id`` and
RAISES ``UnknownStadiumError`` on a miss. Every other game -- any season before 2026,
and any non-neutral game -- resolves by ``home_team`` exactly as it does today. History
is NOT re-resolved: the 91 historical neutral-site games keep the resolution they have
always had, and that is a disclosure (``HISTORICAL_NEUTRAL_MISRESOLUTION``), not a
repair.

WHY THE UNKNOWN-ID CASE HARD-FAILS ON A NEUTRAL GAME AND ONLY WARNS ON A HOME GAME
-----------------------------------------------------------------------------------
A neutral game with an unresolvable stadium has NO defensible fallback -- the home
team's stadium is the wrong answer by construction, which is the whole defect. A HOME
game with an unrecognised ``stadium_id`` is a new or renamed home stadium; the home-team
resolution is still the documented rule and still approximately right, so it resolves
and emits a NAMED warning. Silence there would be the same defect class as the Maracana
case, only quieter.

Modules are imported as MODULES and the new names reached through them, so a missing
symbol is a failure of the test that needs it rather than a collection error that takes
the whole file down with it.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from features import contextual
from scripts import ingest_games, ingest_weather
from tests import phase33_state

REPO_ROOT = Path(__file__).resolve().parents[2]
VENUES_PATH = REPO_ROOT / "data" / "venues.json"


# The stand-in for "a stadium_id data/venues.json does not know".
#
# This used to be the literal "DAL99", which was a REAL nflverse code for Texas Stadium
# that simply had no record yet. Plan 33.1-01 (2026-09-12) added the 22 historical venue
# records, DAL99 among them, and this test silently stopped testing anything: the id
# became known, the warning correctly stopped firing, and the assertion failed. The
# lesson is that an "unknown" fixture must be unknown BY CONSTRUCTION, not by the
# accident of nobody having added it yet. ZZZ99 is not an nflverse code and never will
# be, and `test_the_unknown_id_fixture_is_genuinely_unknown` keeps that honest.
UNKNOWN_STADIUM_ID = "ZZZ99"


def _venues_frame() -> pd.DataFrame:
    return pd.DataFrame(json.loads(VENUES_PATH.read_text(encoding="utf-8"))["venues"])


def _neutral_game(
    stadium_id: str = "RIO00",
    season: int = 2026,
    game_id: str = "2026_W03_BAL@DAL",
) -> dict[str, object]:
    return {
        "game_id": game_id,
        "season": season,
        "week": 3,
        "home_team": "DAL",
        "away_team": "BAL",
        "location": "Neutral",
        "stadium_id": stadium_id,
        "venue": "Maracana Stadium",
    }


def _record_warnings(monkeypatch) -> list[tuple[str, dict]]:
    """Capture ``features.contextual``'s structlog warnings as (event, kwargs)."""
    recorded: list[tuple[str, dict]] = []
    real_logger = contextual.logger

    class _Recorder:
        def warning(self, event, **kwargs):
            recorded.append((str(event), kwargs))
            return real_logger.warning(event, **kwargs)

        def __getattr__(self, name):
            return getattr(real_logger, name)

    monkeypatch.setattr(contextual, "logger", _Recorder())
    return recorded


def _home_game(
    stadium_id: str | None = "DAL00",
    season: int = 2026,
) -> dict[str, object]:
    return {
        "game_id": "2026_W05_PHI@DAL",
        "season": season,
        "week": 5,
        "home_team": "DAL",
        "away_team": "PHI",
        "location": "Home",
        "stadium_id": stadium_id,
        "venue": "AT&T Stadium",
    }


class TestExactCaseSensitiveLookup:
    """R11 edge (encoding): the match is EXACT and CASE-SENSITIVE, no normalization."""

    def test_the_exact_code_resolves(self) -> None:
        venue = contextual._get_venue_by_stadium_id("RIO00")
        assert venue is not None, (
            "RIO00 does not resolve. The eight international records must carry the "
            "feed's exact codes."
        )
        assert venue["venue_name"] == "Maracana Stadium"

    def test_a_lowercased_code_does_not_resolve(self) -> None:
        assert contextual._get_venue_by_stadium_id("rio00") is None, (
            "a lowercased stadium_id matched. R11 requires exact, case-sensitive "
            "matching -- a normalizing lookup would quietly accept a feed value that "
            "is not the one recorded."
        )

    def test_a_padded_code_does_not_resolve(self) -> None:
        assert contextual._get_venue_by_stadium_id(" RIO00 ") is None, (
            "a whitespace-padded stadium_id matched, so the lookup is stripping. "
            "No normalization means no normalization."
        )

    def test_an_absent_code_resolves_to_none_rather_than_raising(self) -> None:
        """The LOOKUP answers None; the ROUTER decides whether that is fatal."""
        assert contextual._get_venue_by_stadium_id("ZZZ99") is None


class TestTheNeutralSiteHardFail:
    """T-33-26: an unknown id never falls back to the home team's own stadium."""

    def test_an_unknown_id_on_a_neutral_game_raises(self) -> None:
        with pytest.raises(contextual.UnknownStadiumError):
            contextual.resolve_venue_for_game(_neutral_game(stadium_id="ZZZ99"))

    def test_the_refusal_names_the_id_the_game_and_the_file(self) -> None:
        """The repo's refusal-carries-its-recovery-command convention."""
        with pytest.raises(contextual.UnknownStadiumError) as excinfo:
            contextual.resolve_venue_for_game(_neutral_game(stadium_id="ZZZ99"))
        message = str(excinfo.value)
        assert "ZZZ99" in message
        assert "2026_W03_BAL@DAL" in message
        assert "data/venues.json" in message

    def test_a_missing_id_on_a_neutral_game_raises_rather_than_defaulting(self) -> None:
        game = _neutral_game()
        game["stadium_id"] = None
        with pytest.raises(contextual.UnknownStadiumError):
            contextual.resolve_venue_for_game(game)

    def test_a_known_neutral_game_resolves_to_its_real_stadium(self) -> None:
        venue = contextual.resolve_venue_for_game(_neutral_game())
        assert venue is not None
        assert venue["stadium_id"] == "RIO00"
        assert venue["city"] == "Rio de Janeiro"
        assert venue["timezone"] == "America/Sao_Paulo", (
            "the Maracana game must derive its local clock from the VENUE's zone, "
            "not from the home team's."
        )


class TestTheSeasonGateLeavesHistoryAlone:
    """D33-15: routing binds at 2026; every earlier season is untouched."""

    def test_a_pre_2026_neutral_game_still_resolves_by_home_team(self) -> None:
        game = _neutral_game(season=2025, game_id="2025_W05_JAX@KC")
        game["home_team"] = "KC"
        game["stadium_id"] = "LON00"
        venue = contextual.resolve_venue_for_game(game)
        assert venue is not None
        assert venue["stadium_id"] == "KAN00", (
            "a 2025 neutral-site game resolved by stadium_id. History keeps its "
            "home_team resolution -- the 91 mis-resolved games are a DISCLOSURE, not "
            "a repair, and re-resolving them would move gold for three deployed "
            "models."
        )

    def test_the_routing_first_season_is_2026(self) -> None:
        assert contextual.STADIUM_ID_ROUTING_FIRST_SEASON == 2026

    def test_a_pre_2026_unknown_id_does_not_raise(self) -> None:
        """History cannot be made to hard-fail on a code it never carried."""
        game = _neutral_game(season=2019, stadium_id="ZZZ99")
        game["home_team"] = "KC"
        venue = contextual.resolve_venue_for_game(game)
        assert venue is not None
        assert venue["stadium_id"] == "KAN00"


class TestTheNonNeutralUnknownStadiumCase:
    """A new or renamed HOME stadium resolves by home_team AND says so."""

    def test_the_unknown_id_fixture_is_genuinely_unknown(self) -> None:
        """Non-vacuity: the two tests below mean nothing if the id is actually known.

        This is the control that would have caught Plan 33.1-01 turning the old DAL99
        fixture into a recognised venue, instead of leaving it to be discovered as a
        failing assertion whose cause was three files away.
        """
        known = {
            str(record["stadium_id"])
            for record in json.loads(VENUES_PATH.read_text(encoding="utf-8"))["venues"]
        }
        assert UNKNOWN_STADIUM_ID not in known, (
            f"{UNKNOWN_STADIUM_ID} is now a real record in data/venues.json, so the "
            "unknown-stadium tests below are asserting against a KNOWN id and prove "
            "nothing. Pick an id that cannot be added."
        )

    def test_an_unknown_id_on_a_home_game_resolves_by_home_team(self) -> None:
        venue = contextual.resolve_venue_for_game(
            _home_game(stadium_id=UNKNOWN_STADIUM_ID)
        )
        assert venue is not None
        assert venue["venue_id"] == "at_t_stadium", (
            "a non-neutral game must keep today's home_team resolution -- that IS "
            "the documented D33-15 rule."
        )

    def test_it_emits_a_named_warning(self, monkeypatch) -> None:
        """Silence would be the Maracana defect class, only quieter.

        The logger is captured DIRECTLY rather than through ``caplog``: this project
        logs through structlog, so a stdlib-handler assertion would pass or fail on
        the logging CONFIGURATION rather than on what the router actually recorded.
        """
        recorded = _record_warnings(monkeypatch)
        contextual.resolve_venue_for_game(_home_game(stadium_id=UNKNOWN_STADIUM_ID))

        assert any(
            UNKNOWN_STADIUM_ID in event
            or kwargs.get("stadium_id") == UNKNOWN_STADIUM_ID
            for event, kwargs in recorded
        ), (
            "an unrecognised stadium_id on a home game was absorbed without a word. "
            "A new home stadium silently inheriting the old venue's coordinates, "
            "timezone and elevation is the same defect class as the Maracana case. "
            f"Warnings seen: {[event for event, _ in recorded]}"
        )

    def test_a_recognised_home_id_warns_about_nothing(self, monkeypatch) -> None:
        recorded = _record_warnings(monkeypatch)
        venue = contextual.resolve_venue_for_game(_home_game())

        assert venue is not None
        assert venue["venue_id"] == "at_t_stadium"
        assert not [
            event for event, kwargs in recorded if kwargs.get("stadium_id") == "DAL00"
        ], "a RECOGNISED stadium_id must not warn; the warning is for the unknown case"


class TestTheRelocatedTeamCase:
    """The FEED names the stadium; an abbreviation's history does not."""

    def test_a_relocated_team_resolves_to_the_stadium_the_feed_names(self) -> None:
        """LA plays its 2026 week-1 game in Melbourne, not at SoFi.

        `LA`'s venue record is SoFi Stadium and always has been. The neutral-site
        routing must take the feed's `stadium_id` over the abbreviation's historical
        home, which is the same mechanism that protects a genuine franchise move.
        """
        game = {
            "game_id": "2026_W01_SF@LA",
            "season": 2026,
            "week": 1,
            "home_team": "LA",
            "away_team": "SF",
            "location": "Neutral",
            "stadium_id": "MEL00",
            "venue": "Melbourne Cricket Ground",
        }
        venue = contextual.resolve_venue_for_game(game)
        assert venue is not None
        assert venue["stadium_id"] == "MEL00"
        assert venue["venue_id"] != "sofi_stadium"
        assert venue["timezone"] == "Australia/Melbourne"


class TestTheNeutralSiteFactCanBeReadFromEitherSpelling:
    """The feed says `location`; silver will say `neutral_site`. Both must route."""

    def test_the_silver_spelling_routes(self) -> None:
        game = _neutral_game()
        del game["location"]
        game["neutral_site"] = True
        venue = contextual.resolve_venue_for_game(game)
        assert venue is not None
        assert venue["stadium_id"] == "RIO00"

    def test_neither_spelling_present_means_not_neutral(self) -> None:
        game = _neutral_game()
        del game["location"]
        venue = contextual.resolve_venue_for_game(game)
        assert venue is not None
        assert venue["venue_id"] == "at_t_stadium"


class TestTheWeatherResolver:
    """NF-05 resolver 2 of 3: the same rule, the same codes."""

    def test_the_weather_lookup_resolves_the_exact_code(self) -> None:
        ingester = ingest_weather.WeatherDataIngester()
        lat, lon, roof = ingester._get_venue_coordinates_by_stadium_id(
            "RIO00", _venues_frame()
        )
        facts = {row[0]: row for row in phase33_state.INTERNATIONAL_VENUE_FACTS}
        assert (lat, lon) == (facts["RIO00"][5], facts["RIO00"][6])
        assert roof == "outdoor"

    def test_the_weather_lookup_is_case_sensitive(self) -> None:
        ingester = ingest_weather.WeatherDataIngester()
        with pytest.raises(Exception, match="rio00"):
            ingester._get_venue_coordinates_by_stadium_id("rio00", _venues_frame())

    def test_the_weather_refusal_names_the_id_and_the_file(self) -> None:
        ingester = ingest_weather.WeatherDataIngester()
        with pytest.raises(Exception) as excinfo:
            ingester._get_venue_coordinates_by_stadium_id("ZZZ99", _venues_frame())
        message = str(excinfo.value)
        assert "ZZZ99" in message
        assert "data/venues.json" in message

    def test_the_normalize_team_hard_fail_survives(self) -> None:
        """The EXISTING refusal must not be weakened by the new code path.

        It fires BEFORE the venue lookup -- `normalize_team_abbreviation` refuses an
        unknown abbreviation outright -- which is why this asserts on that message
        and not on the venues.json one below.
        """
        ingester = ingest_weather.WeatherDataIngester()
        with pytest.raises(Exception, match="Unknown team abbreviation"):
            ingester._get_venue_coordinates("ZZZ", _venues_frame())

    def test_the_home_teams_refusal_survives(self) -> None:
        """A KNOWN team with no venue record still carries its recovery command."""
        ingester = ingest_weather.WeatherDataIngester()
        without_buffalo = _venues_frame()
        without_buffalo = without_buffalo[
            ~without_buffalo["home_teams"].apply(lambda teams: "BUF" in teams)
        ]
        with pytest.raises(Exception, match="home_teams"):
            ingester._get_venue_coordinates("BUF", without_buffalo)


class TestTheIngestGamesResolver:
    """NF-05 resolver 3 of 3: stadium_id is consulted BEFORE the venue NAME."""

    def test_the_three_false_domes_do_not_become_indoor(self) -> None:
        ingester = ingest_games.GameDataIngester()
        for code, feed_roof in phase33_state.FEED_ROOF_VALUES_2026:
            if code not in phase33_state.FEED_ROOF_DISAGREEMENTS:
                continue
            roof = ingester._get_venue_roof_type("", feed_roof, stadium_id=code)
            assert roof != "indoor", (
                f"{code} resolved to {roof!r} through the ingest_games resolver. The "
                "feed says 'dome' and the venue is open-air; venues.json is "
                "AUTHORITATIVE on roof for all eight (D33-16)."
            )

    def test_stadium_id_wins_over_a_wrong_venue_name(self) -> None:
        """The NAME lookup must not be able to override the code lookup."""
        ingester = ingest_games.GameDataIngester()
        roof = ingester._get_venue_roof_type("Ford Field", "dome", stadium_id="MUN01")
        assert roof == "outdoor", (
            f"resolved to {roof!r}. A row whose venue NAME happened to match an "
            "indoor US stadium must still take the stadium_id's answer."
        )

    def test_the_name_lookup_still_works_without_a_stadium_id(self) -> None:
        ingester = ingest_games.GameDataIngester()
        assert ingester._get_venue_roof_type("Ford Field", "outdoors") == "indoor"

    def test_the_nflverse_fallback_still_works(self) -> None:
        ingester = ingest_games.GameDataIngester()
        assert ingester._get_venue_roof_type("Nowhere Stadium", "closed") == "indoor"

    def test_an_unrecognised_stadium_id_falls_through_rather_than_raising(self) -> None:
        """Ingest is not the router; it must not turn a new code into a hard stop."""
        ingester = ingest_games.GameDataIngester()
        assert (
            ingester._get_venue_roof_type("Ford Field", "dome", stadium_id="ZZZ99")
            == "indoor"
        )
