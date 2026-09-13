"""`is_outdoor` comes from the GAME's own roof, not from the venue's roof_type.

WHAT IS BEING PINNED
--------------------
SPEC R4: "`is_outdoor` derives from the game's own `roof` value in the pinned feed
(`outdoors`/`open` mean weather applies, `dome`/`closed` mean it does not)."

The MAPPING was already correct before Plan 33.1-02. Only the SOURCE of the value
was wrong::

    feed roof    NFLVERSE_ROOF_MAP      _is_outdoor_game    weather applies
    "outdoors" -> "outdoor"          -> True                yes
    "open"     -> "retractable"      -> True                yes
    "closed"   -> "indoor"           -> False               no
    "dome"     -> "indoor"           -> False               no

`_resolve_venue_for_game_row` returned `venue["roof_type"]` and
`fetch_weather_for_games` branched on that, so every game at a retractable stadium
answered the same way -- including the six of eight 2016 Lucas Oil games played
with the roof SHUT. The old test asserted `is_outdoor=True` for a retractable
venue and passed for the wrong reason: it was reading a property of the BUILDING
where the requirement is about the GAME.

WHY THE FIXTURE IS DERIVED RATHER THAN TYPED
--------------------------------------------
`per_game_roof_games` reads its two rows from `data.upstream_pin.load_schedules`
and from silver. A typed fixture could describe a stadium the feed does not have,
or a roof a stadium never had, and would then prove a property of the fixture.

FOUR CONTROLS, copied from `tests/unit/test_weather_archive_quarantined.py:31-42`
---------------------------------------------------------------------------------
1. NON-VACUITY: the pair actually carries two DIFFERENT feed roofs at ONE
   stadium_id, asserted before anything is concluded from it.
2. THE ASSERTION itself: the two games differ in `is_outdoor`.
3. A PLANTED VIOLATION: branching on the VENUE's roof_type instead -- the old
   behaviour -- makes the two agree, proving the assertion discriminates.
4. NO FALSE POSITIVE: all four feed values map the way R4 says, so the rule is
   not merely "IND00 is special".

This module writes nothing. It needs no `writes_production_store` marker.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scripts.ingest_weather import NFLVERSE_ROOF_MAP

# The stadium the pair lives at, and the two feed roofs it carries in 2016.
PAIR_STADIUM_ID = "IND00"
OPEN_GAME_ID = "2016_W01_DET@IND"
CLOSED_GAME_ID = "2016_W03_LAC@IND"

# R4's rule, as a table rather than as prose, so a change to NFLVERSE_ROOF_MAP
# that broke it would have to break this too.
FEED_ROOF_MEANS_OUTDOOR: tuple[tuple[str, bool], ...] = (
    ("outdoors", True),
    ("open", True),
    ("closed", False),
    ("dome", False),
)

_OBSERVATION = {
    "temp_f": 61.0,
    "temp_c": 16.1,
    "wind_mph": 7.0,
    "wind_direction": 180.0,
    "humidity_pct": 55.0,
    "precip_prob": None,
    "precip_mm": 0.0,
    "condition": None,
    "condition_code": 1,
    "visibility_km": None,
    "dew_point_f": 45.0,
    "apparent_temp_f": 60.0,
    "snowfall_cm": 0.0,
    "wind_gusts_mph": 11.0,
    "cloud_cover_pct": 20.0,
    "weather_code": 1,
}


@pytest.fixture
def backfiller():
    """The archive path, with settings mocked. Reaches no network in this module."""
    with patch("scripts.ingest_weather.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock()
        from scripts.backfill_historical_weather import HistoricalWeatherBackfiller

        return HistoricalWeatherBackfiller()


@pytest.fixture
def venues_df():
    import json
    from pathlib import Path

    import pandas as pd

    return pd.DataFrame(json.loads(Path("data/venues.json").read_text())["venues"])


def _run(backfiller, frame, venues_df):
    with patch.object(
        backfiller, "_fetch_openmeteo_weather", AsyncMock(return_value=_OBSERVATION)
    ):
        return backfiller.fetch_weather_for_games(frame, venues_df)


class TestTheFixtureIsGenuinelyAPair:
    """CONTROL 1 -- non-vacuity. A pair that is not a pair proves nothing."""

    def test_the_two_games_sit_at_one_stadium_with_two_different_roofs(
        self, per_game_roof_games
    ):
        from scripts.backfill_historical_weather import load_pinned_game_facts

        frame = per_game_roof_games()
        facts = load_pinned_game_facts([2016]).set_index("game_id")

        assert set(frame["game_id"]) == {OPEN_GAME_ID, CLOSED_GAME_ID}
        stadium_ids = {facts.loc[gid, "stadium_id"] for gid in frame["game_id"]}
        assert stadium_ids == {PAIR_STADIUM_ID}, (
            "the whole point is ONE stadium_id with two roofs; two stadiums would "
            f"make the assertion trivially true. Got {stadium_ids}."
        )
        assert {facts.loc[gid, "roof"] for gid in frame["game_id"]} == {
            "open",
            "closed",
        }

    def test_the_venue_record_answers_identically_for_both_games(self, venues_df):
        """The VENUE cannot distinguish them -- which is why the GAME must."""
        record = venues_df[venues_df["stadium_id"] == PAIR_STADIUM_ID]
        assert len(record) == 1
        assert record.iloc[0]["roof_type"] == "retractable", (
            "if the venue record stopped saying 'retractable', the planted "
            "violation below would no longer reproduce the old behaviour and this "
            "module would be asserting against a premise that had quietly changed."
        )


class TestTheGamesOwnRoofDecides:
    """CONTROL 2 -- the assertion."""

    @pytest.mark.parametrize("with_silver_stadium_id", [False, True])
    def test_a_closed_game_and_an_open_game_at_one_stadium_differ(
        self, backfiller, venues_df, per_game_roof_games, with_silver_stadium_id
    ):
        frame = per_game_roof_games(with_silver_stadium_id=with_silver_stadium_id)
        result = _run(backfiller, frame, venues_df)

        by_id = result.set_index("game_id")["is_outdoor"]
        assert bool(by_id[OPEN_GAME_ID]) is True, (
            f"{OPEN_GAME_ID} was played with the roof OPEN; weather applies."
        )
        assert bool(by_id[CLOSED_GAME_ID]) is False, (
            f"{CLOSED_GAME_ID} was played with the roof CLOSED; weather does not."
        )
        assert bool(by_id[OPEN_GAME_ID]) != bool(by_id[CLOSED_GAME_ID])

    def test_only_the_open_game_was_fetched(
        self, backfiller, venues_df, per_game_roof_games
    ):
        """The branch decides a NETWORK CALL, not only a column.

        A closed game that still issued a request would be a real cost over 6,499
        games even if its `is_outdoor` came out right.
        """
        fetcher = AsyncMock(return_value=_OBSERVATION)
        with patch.object(backfiller, "_fetch_openmeteo_weather", fetcher):
            backfiller.fetch_weather_for_games(per_game_roof_games(), venues_df)

        assert fetcher.await_count == 1, (
            "exactly one of the two games needs an observation; "
            f"{fetcher.await_count} requests were issued."
        )


class TestThePlantedViolation:
    """CONTROL 3 -- the OLD behaviour, reproduced, must make the two agree."""

    def test_branching_on_the_venue_roof_type_collapses_the_pair(
        self, backfiller, venues_df, per_game_roof_games
    ):
        frame = per_game_roof_games()
        venue = venues_df[venues_df["stadium_id"] == PAIR_STADIUM_ID].iloc[0]

        # This IS the pre-33.1-02 decision, written out: the venue's roof_type
        # mapped through the same predicate, for every game at that stadium.
        old_answer = backfiller._is_outdoor_game(venue["roof_type"])
        collapsed = {old_answer for _ in frame["game_id"]}

        assert collapsed == {True}, (
            "the old rule answered True for BOTH games at this retractable "
            "stadium. If it no longer does, this control is not reproducing the "
            "defect and the test above is not proving what it claims."
        )

        # And the new rule does not collapse.
        result = _run(backfiller, frame, venues_df)
        assert len(set(result["is_outdoor"].astype(bool))) == 2


class TestNoFalsePositive:
    """CONTROL 4 -- the rule is R4's, not a special case for one stadium."""

    @pytest.mark.parametrize(("feed_roof", "expected"), FEED_ROOF_MEANS_OUTDOOR)
    def test_every_feed_roof_value_maps_the_way_the_requirement_says(
        self, backfiller, feed_roof, expected
    ):
        assert backfiller._per_game_roof_is_outdoor(feed_roof) is expected

    def test_the_four_values_are_the_whole_feed_vocabulary(self):
        """A fifth feed value would make the table above incomplete in silence."""
        assert {roof for roof, _ in FEED_ROOF_MEANS_OUTDOOR} == set(NFLVERSE_ROOF_MAP)

    def test_an_unknown_roof_raises_rather_than_defaulting_to_outdoor(self, backfiller):
        """`_map_nflverse_roof_type` defaults to "outdoor"; this path must not.

        A defaulted roof would fetch weather for a game played under a closed one,
        or skip the fetch for a game played in the open, and either lands in the
        corpus as a fact.
        """
        from utils.exceptions import WeatherDataError

        with pytest.raises(WeatherDataError) as excinfo:
            backfiller._per_game_roof_is_outdoor("retractable-ish")

        message = str(excinfo.value)
        assert "retractable-ish" in message
        assert all(roof in message for roof in NFLVERSE_ROOF_MAP)
