"""The venue-to-forecast-station map is derived from venues.json, frozen, and verified (Plan 33.2-11).

The frozen map must equal what the nearest-primary-airport rule derives from today's
``data/venues.json``; a venue edit that moves the nearest airport fails here and forces a
deliberate re-freeze. Every mapped station must be one probed present in both eras the backfill
reads, and no non-US venue may ever be given a stand-in station.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from config import mos_stations
from config.mos_stations import (
    CANDIDATE_AIRPORTS,
    EXCLUDED_NON_PRIMARY_AIRPORTS,
    PROBED_PRESENT_STATIONS,
    UNCOVERABLE_NON_US_STADIUM_IDS,
    VENUE_STATIONS,
    derive_venue_stations,
    station_for_venue,
)

VENUES_PATH = Path(__file__).resolve().parents[2] / "data" / "venues.json"


def _venues() -> list[dict]:
    return json.loads(VENUES_PATH.read_text(encoding="utf-8"))["venues"]


def test_exactly_37_us_venues_are_mapped() -> None:
    assert len(VENUE_STATIONS) == 37
    by_id = {v["stadium_id"]: v for v in _venues()}
    for stadium_id in VENUE_STATIONS:
        assert by_id[stadium_id]["country"] == "USA", stadium_id
        assert by_id[stadium_id]["roof_type"] in mos_stations.COVERED_ROOF_TYPES, (
            stadium_id
        )


def test_the_frozen_map_equals_the_derivation_from_venues_json() -> None:
    derived = derive_venue_stations(_venues(), VENUE_STATIONS)
    assert derived == VENUE_STATIONS


def test_every_mapped_station_was_probed_present_in_both_eras() -> None:
    used = {choice.station for choice in VENUE_STATIONS.values()}
    assert used <= PROBED_PRESENT_STATIONS
    assert "KCGX" not in used and "KGEU" not in used


def test_the_excluded_airports_are_not_candidates() -> None:
    assert set(EXCLUDED_NON_PRIMARY_AIRPORTS).isdisjoint(CANDIDATE_AIRPORTS)


def test_the_uncoverable_set_is_every_non_us_venue() -> None:
    non_us = {v["stadium_id"] for v in _venues() if v["country"] != "USA"}
    assert non_us == UNCOVERABLE_NON_US_STADIUM_IDS
    assert non_us.isdisjoint(VENUE_STATIONS)


def test_a_non_us_venue_is_refused_by_name_never_given_a_stand_in() -> None:
    with pytest.raises(KeyError, match="outside the USA"):
        station_for_venue("LON00")


def test_the_arguable_choices_follow_the_rule() -> None:
    assert station_for_venue("BOS00") == "KBOS"
    assert station_for_venue("NYC01") == "KEWR"
    assert station_for_venue("SFO01") == "KSJC"
    assert station_for_venue("SDG00") == "KSAN"
