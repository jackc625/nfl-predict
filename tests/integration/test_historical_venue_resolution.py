"""All 6,499 games resolve, and none of them resolves to somebody else's stadium.

Phase 33.1, Plan 33.1-03 Task 2 (R1, D33.1-06, T-33.1-15/16/20).

WHAT IS BEING PINNED
--------------------
SPEC R1's acceptance, over the whole corpus rather than over a sample: "for all
6,499 games in 2002-2025, venue resolution returns a record". Plan 33.1-03 adds
the half R1 leaves implicit and D33.1-06 makes binding -- that the record returned
is the game's OWN stadium, for every season, with no season test and no
neutral-site test.

WHY THE POPULATION IS THE PINNED FEED AND NOT A FIXTURE
--------------------------------------------------------
A fixture can only contain games somebody thought to write down, and the failure
mode this test exists to catch is a stadium_id nobody remembered. The population
is ``data.upstream_pin.load_schedules(range(2002, 2026))`` -- the sealed two-zone
pin, 6,499 rows, zero nulls in ``stadium_id`` across 55 distinct codes. If the pin
moves, the count assertion below says so rather than the corpus quietly shrinking.

THE AGREEMENT CHECK HERE IS THE WEAK ONE, DELIBERATELY
-------------------------------------------------------
Test 6 below asserts that the contextual builder and the weather ingester return
the same ``stadium_id`` for all 6,499 games. That is worth having and it is not
what proves they share one rule: an agreement check passes identically against one
implementation and against two copies that currently coincide. The claim that they
are ONE function is proven by MUTATION in
``tests/unit/test_venue_resolver_is_shared.py``. Both exist on purpose; neither
substitutes for the other.

THIS MODULE READS AND NEVER WRITES. It touches the pinned bronze snapshot and
``data/venues.json``, both read-only, and reaches no network. It needs no
``writes_production_store`` marker.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from conf.season_partition import CORPUS_FIRST_SEASON
from data.upstream_pin import load_schedules
from features import contextual
from scripts import ingest_weather

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
VENUES_PATH = REPO_ROOT / "data" / "venues.json"

# The corpus, stated once. 2026 is excluded because the pin's live zone is
# append-only per week and this phase is history only (SPEC boundaries).
# The corpus floor, read from the one rule module (Plan 33.2-18: no floor literal outside
# conf/season_partition.py). Was: FIRST_SEASON = 2002.
FIRST_SEASON = CORPUS_FIRST_SEASON
LAST_SEASON = 2025
EXPECTED_GAMES = 6499
EXPECTED_DISTINCT_STADIUM_IDS = 55


@lru_cache(maxsize=1)
def _schedules() -> pd.DataFrame:
    """The pinned 2002-2025 feed, loaded once for the whole module."""
    return load_schedules(list(range(FIRST_SEASON, LAST_SEASON + 1)))


@lru_cache(maxsize=1)
def _venue_records() -> tuple[dict[str, Any], ...]:
    return tuple(json.loads(VENUES_PATH.read_text(encoding="utf-8"))["venues"])


@lru_cache(maxsize=1)
def _venues_frame() -> pd.DataFrame:
    return pd.DataFrame(list(_venue_records()))


def _game_rows() -> list[dict[str, Any]]:
    """Every pinned game as a plain mapping in the shape the routers accept."""
    frame = _schedules()
    return [
        {
            "game_id": str(row.game_id),
            "season": int(row.season),
            "week": int(row.week),
            "home_team": str(row.home_team),
            "away_team": str(row.away_team),
            "location": str(row.location),
            "roof": str(row.roof),
            "stadium_id": str(row.stadium_id),
        }
        for row in frame.itertuples()
    ]


class TestThePopulationIsTheWholePinnedCorpus:
    """Non-vacuity. Every assertion below is "no game was wrong"."""

    def test_the_pin_still_holds_the_recorded_number_of_games(self) -> None:
        assert len(_schedules()) == EXPECTED_GAMES, (
            "the pinned 2002-2025 corpus is no longer 6,499 games. Every count in "
            "this module and in tests.phase33_state.ROUTING_COORDINATE_DIFF was "
            "measured on that population."
        )

    def test_every_game_names_a_stadium_id(self) -> None:
        stadium_ids = _schedules()["stadium_id"]
        assert stadium_ids.notna().all(), (
            "a pinned game carries a null stadium_id. Under D33.1-06 that is an "
            "unroutable game, not a game that falls back to its home team."
        )
        assert stadium_ids.nunique() == EXPECTED_DISTINCT_STADIUM_IDS


class TestEveryGameResolvesToItsOwnStadium:
    """Test 1 -- R1's acceptance, plus the half D33.1-06 makes binding."""

    def test_no_game_raises_and_no_game_resolves_elsewhere(self) -> None:
        records = list(_venue_records())
        raised: list[str] = []
        misrouted: list[tuple[str, str, str]] = []

        for game in _game_rows():
            try:
                venue = contextual.resolve_venue_for_game(game, records)
            except contextual.UnknownStadiumError:
                raised.append(game["game_id"])
                continue
            if venue["stadium_id"] != game["stadium_id"]:
                misrouted.append(
                    (game["game_id"], game["stadium_id"], str(venue["stadium_id"]))
                )

        assert not raised, (
            f"{len(raised)} game(s) could not be resolved at all, e.g. "
            f"{raised[:5]}. Plan 33.1-01 added the 22 missing venue records so "
            "that every one of the 55 pinned codes resolves; a raise here means a "
            "record was removed or a code appeared."
        )
        assert not misrouted, (
            f"{len(misrouted)} game(s) resolved to a stadium other than their own, "
            f"e.g. {misrouted[:5]}. This lands near 1,070 if only the season half "
            "of the old two-condition gate was retired, because the 141 Oakland "
            "Coliseum games are not neutral-site."
        )

    def test_the_relocation_classes_land_on_their_own_coordinates(self) -> None:
        """The named cases, so the aggregate above cannot hide a swap.

        Each pair is a franchise whose present-day stadium is hundreds or
        thousands of miles from the one these games were played in.
        """
        records = list(_venue_records())
        by_id = {str(r["stadium_id"]): r for r in records}
        for old_code, successor_code in (
            ("OAK00", "VEG00"),
            ("SDG00", "LAX01"),
            ("STL00", "LAX01"),
            ("LAX99", "LAX01"),
            ("LAX97", "LAX01"),
            ("NYC00", "NYC01"),
        ):
            games = [g for g in _game_rows() if g["stadium_id"] == old_code]
            assert games, f"no pinned game names {old_code}"
            for game in games:
                venue = contextual.resolve_venue_for_game(game, records)
                assert (venue["latitude"], venue["longitude"]) == (
                    by_id[old_code]["latitude"],
                    by_id[old_code]["longitude"],
                ), f"{game['game_id']} did not land on {old_code}'s coordinates"
                assert (venue["latitude"], venue["longitude"]) != (
                    by_id[successor_code]["latitude"],
                    by_id[successor_code]["longitude"],
                ), (
                    f"{game['game_id']} resolved to {successor_code}'s coordinates, "
                    "which is the successor stadium this repair exists to stop."
                )


class TestBothConsumersReturnTheSameStadium:
    """Test 6 -- the WEAK agreement check, over the whole corpus.

    Kept as a separate and deliberately weaker statement than the mutation proof
    in tests/unit/test_venue_resolver_is_shared.py. What it adds is BREADTH: the
    mutation test uses one game, and this one says the shared resolver answers
    identically for all 6,499 through both entry points, including the frame
    adaptation the weather side performs on the way.
    """

    def test_the_contextual_and_weather_paths_agree_on_all_games(self) -> None:
        records = list(_venue_records())
        venues_df = _venues_frame()
        ingester = ingest_weather.WeatherDataIngester()

        disagreements: list[tuple[str, str, str]] = []
        for game in _game_rows():
            from_contextual = contextual.resolve_venue_for_game(game, records)
            from_weather = ingester._resolve_venue_record_for_game(game, venues_df)
            if str(from_contextual["stadium_id"]) != str(from_weather["stadium_id"]):
                disagreements.append(
                    (
                        game["game_id"],
                        str(from_contextual["stadium_id"]),
                        str(from_weather["stadium_id"]),
                    )
                )

        assert not disagreements, (
            f"{len(disagreements)} game(s) got different stadiums from the two "
            f"consumers, e.g. {disagreements[:5]}. They call one function; a "
            "disagreement means one of them stopped."
        )

    def test_the_weather_path_also_agrees_on_the_coordinates(self) -> None:
        """Same stadium_id with different coordinates would still be a divergence."""
        records = list(_venue_records())
        venues_df = _venues_frame()
        ingester = ingest_weather.WeatherDataIngester()

        for game in _game_rows()[::97]:  # a deterministic stride over the corpus
            from_contextual = contextual.resolve_venue_for_game(game, records)
            from_weather = ingester._resolve_venue_record_for_game(game, venues_df)
            assert (
                from_contextual["latitude"],
                from_contextual["longitude"],
                from_contextual["roof_type"],
            ) == (
                from_weather["latitude"],
                from_weather["longitude"],
                from_weather["roof_type"],
            ), game["game_id"]
