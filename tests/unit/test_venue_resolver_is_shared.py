"""The contextual builder and the weather ingester share ONE resolver, not two.

Phase 33.1, Plan 33.1-03 Task 1 (R1, D33.1-06, Ruling I2, T-33.1-16).

WHY A MUTATION TEST AND NOT AN EQUALITY TEST
--------------------------------------------
Until this plan there were TWO independently maintained implementations of
"which stadium was this game played at": ``features.contextual``'s module-level
router and ``WeatherDataIngester._get_venue_record_by_stadium_id``. Retiring the
season gate in both made them AGREE. It did not make them ONE RULE.

An agreement test -- "resolve all 6,499 games through each and assert the answers
match" -- passes IDENTICALLY against one shared implementation and against two
copies that currently coincide. It cannot distinguish the state this task exists
to leave behind from the state it exists to create. That agreement check is still
worth having and it still runs, over the whole corpus, in
``tests/integration/test_historical_venue_resolution.py``; it is kept as a
SEPARATE and deliberately WEAKER check rather than relied on for this claim.

What only a MUTATION test can assert is causation: change the shared function and
BOTH consumers' answers change. That is a statement about the call graph, and it
is false the moment somebody re-inlines the lookup for convenience -- which is the
failure mode, since two copies that agree today diverge at the next edit to
either. Do NOT "simplify" this module into an equality check.

WHY THE MONKEYPATCH REACHES THE WEATHER MODULE AT ALL
------------------------------------------------------
``scripts/ingest_weather.py`` imports the MODULE and calls
``venue_routing.venue_record_for_stadium_id(...)``, rather than importing the
function by name. That is deliberate and it is part of what is under test here: a
``from features.contextual import venue_record_for_stadium_id`` would bind a
second name, and patching the canonical one would leave the weather call site
pointing at the original -- so this test would report "two implementations" for a
module that is in fact delegating correctly, and worse, could be "fixed" by
patching both names, which is an equality test wearing a mutation test's clothes.

FOUR CONTROLS, following tests/unit/test_weather_archive_quarantined.py:31-42
-----------------------------------------------------------------------------
1. NON-VACUITY: the two consumers return the REAL record before the patch, so the
   marker observed after it is a change rather than a coincidence.
2. THE ASSERTION: both consumers return the marker under the patch.
3. A PLANTED VIOLATION: a stand-in consumer that carries its OWN copy of the
   lookup is shown to be INVISIBLE to the same patch -- proving the assertion can
   fail, and demonstrating exactly what a re-inlined body would look like here.
4. NO FALSE POSITIVE: a source scan confirms the weather adapter delegates and
   constructs no refusal of its own, and the real refusal still fires with the
   patch removed.

This module reads only. It reaches no network and writes nothing.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from features import contextual
from scripts import ingest_weather
from utils.exceptions import WeatherDataError

REPO_ROOT = Path(__file__).resolve().parents[2]
VENUES_PATH = REPO_ROOT / "data" / "venues.json"

UNKNOWN_STADIUM_ID = "ZZZ99"

# A record no venue file will ever hold. Its values are deliberately absurd so a
# test that accidentally read the real file cannot pass by resembling it.
MARKER_RECORD: dict[str, Any] = {
    "venue_id": "marker_venue_not_in_any_file",
    "stadium_id": "MARK0",
    "venue_name": "Marker Venue",
    "latitude": -1.2345,
    "longitude": 98.7654,
    "elevation_ft": 12345,
    "roof_type": "outdoor",
    "timezone": "UTC",
}

GAME: dict[str, object] = {
    "game_id": "2015_W01_CIN@OAK",
    "season": 2015,
    "week": 1,
    "home_team": "OAK",
    "away_team": "CIN",
    "location": "Home",
    "stadium_id": "OAK00",
}


def _venue_records() -> list[dict[str, Any]]:
    return json.loads(VENUES_PATH.read_text(encoding="utf-8"))["venues"]


def _venues_frame() -> pd.DataFrame:
    return pd.DataFrame(_venue_records())


@pytest.fixture
def ingester() -> ingest_weather.WeatherDataIngester:
    return ingest_weather.WeatherDataIngester()


def _marker_resolver(stadium_id, venues=None, **kwargs) -> dict[str, Any]:
    """Stand in for the shared resolver, ignoring every argument.

    ``**kwargs`` absorbs the keyword-only ``game_id`` the router passes for its
    refusal message. A stub that refused it would fail for a reason that has
    nothing to do with the claim under test.
    """
    return MARKER_RECORD


class TestTheConsumersAnswerCorrectlyBeforeAnyPatch:
    """CONTROL 1 -- non-vacuity. The marker below must be a CHANGE."""

    def test_the_contextual_router_returns_the_real_oakland_record(self) -> None:
        venue = contextual.resolve_venue_for_game(GAME)
        assert venue["stadium_id"] == "OAK00"
        assert venue["venue_id"] != MARKER_RECORD["venue_id"]

    def test_the_weather_resolver_returns_the_real_oakland_record(
        self, ingester
    ) -> None:
        venue = ingester._resolve_venue_record_for_game(GAME, _venues_frame())
        assert venue["stadium_id"] == "OAK00"
        assert venue["venue_id"] != MARKER_RECORD["venue_id"]

    def test_the_two_agree_on_the_real_record(self, ingester) -> None:
        """The WEAK check, stated here so its weakness is on the record.

        This passes just as well against two copies. It is the whole-corpus
        version of this assertion that lives in
        tests/integration/test_historical_venue_resolution.py, and neither is
        what proves the resolver is shared.
        """
        from_contextual = contextual.resolve_venue_for_game(GAME)
        from_weather = ingester._resolve_venue_record_for_game(GAME, _venues_frame())
        assert from_contextual["stadium_id"] == from_weather["stadium_id"]


class TestPatchingTheSharedResolverMovesBothAnswers:
    """CONTROL 2 -- the assertion an equality test cannot make."""

    def test_the_contextual_router_returns_the_marker(self, monkeypatch) -> None:
        monkeypatch.setattr(contextual, "venue_record_for_stadium_id", _marker_resolver)
        assert contextual.resolve_venue_for_game(GAME) is MARKER_RECORD

    def test_the_weather_resolver_returns_the_marker(
        self, monkeypatch, ingester
    ) -> None:
        monkeypatch.setattr(contextual, "venue_record_for_stadium_id", _marker_resolver)
        resolved = ingester._resolve_venue_record_for_game(GAME, _venues_frame())
        assert resolved is MARKER_RECORD, (
            "the weather path did not go through "
            "features.contextual.venue_record_for_stadium_id. Either it carries its "
            "own copy of the lookup again, or it bound the function by name at "
            "import time -- both of which make the contextual builder and the "
            "weather ingester two rules that agree today and diverge at the next "
            f"edit to either. Got: {resolved!r}"
        )

    def test_both_consumers_return_the_same_marker_object(
        self, monkeypatch, ingester
    ) -> None:
        """The claim in one assertion: ONE function answered for BOTH."""
        monkeypatch.setattr(contextual, "venue_record_for_stadium_id", _marker_resolver)
        assert (
            contextual.resolve_venue_for_game(GAME)
            is ingester._resolve_venue_record_for_game(GAME, _venues_frame())
            is MARKER_RECORD
        )

    def test_the_class_level_router_returns_the_marker_too(self, monkeypatch) -> None:
        """The third consumer: the calculator's own method, and its venue_id."""
        calculator = contextual.ContextualFeaturesCalculator()
        monkeypatch.setattr(contextual, "venue_record_for_stadium_id", _marker_resolver)
        assert calculator._resolve_venue_id_for_game(GAME) == MARKER_RECORD["venue_id"]


class TestAPrivateCopyIsInvisibleToThePatch:
    """CONTROL 3 -- the PLANTED violation, so the assertion can be seen to fail.

    This is what ``_get_venue_record_by_stadium_id`` looked like before this plan:
    its own match, over its own records, raising its own refusal. It agrees with
    the shared resolver on every real input -- and the patch above does not move
    it at all, which is precisely the state a mutation test detects and an
    equality test does not.
    """

    @staticmethod
    def _resolver_with_a_private_copy(stadium_id: object) -> dict[str, Any]:
        for venue in _venue_records():
            if venue.get("stadium_id") == stadium_id:
                return venue
        raise LookupError(stadium_id)

    def test_the_private_copy_agrees_on_the_real_data(self) -> None:
        """It is indistinguishable from the shared resolver by equality alone."""
        assert (
            self._resolver_with_a_private_copy("OAK00")["venue_id"]
            == contextual.resolve_venue_for_game(GAME)["venue_id"]
        )

    def test_the_private_copy_ignores_the_patch(self, monkeypatch) -> None:
        monkeypatch.setattr(contextual, "venue_record_for_stadium_id", _marker_resolver)
        assert self._resolver_with_a_private_copy("OAK00") is not MARKER_RECORD, (
            "the planted private copy followed the patch, so this control is "
            "vacuous and the assertions above prove less than they claim."
        )


class TestTheRefusalStringLivesInExactlyOnePlace:
    """CONTROL 4 -- no false positive, by source scan and by behaviour."""

    def test_the_weather_adapter_calls_the_shared_name(self) -> None:
        source = [
            line
            for line in inspect.getsource(
                ingest_weather.WeatherDataIngester._get_venue_record_by_stadium_id
            ).splitlines()
            if not line.lstrip().startswith("#")
        ]
        assert any("venue_record_for_stadium_id" in line for line in source), (
            "the weather adapter no longer calls the shared resolver by name. "
            "Whatever it does instead is a second implementation."
        )

    def test_the_weather_adapter_constructs_no_refusal_of_its_own(self) -> None:
        """One missing record must not produce two different recovery texts."""
        source = [
            line
            for line in inspect.getsource(
                ingest_weather.WeatherDataIngester._get_venue_record_by_stadium_id
            ).splitlines()
            if not line.lstrip().startswith("#")
        ]
        assert not [line for line in source if "UnknownStadiumError(" in line], (
            "the weather adapter constructs its own UnknownStadiumError. The "
            "refusal text is part of the rule: two consumers that hand an operator "
            "different recovery instructions for the same missing record have "
            "already diverged."
        )

    def test_the_refusal_still_fires_and_carries_one_text(self, ingester) -> None:
        """The delegation did not turn a hard refusal into a soft miss."""
        with pytest.raises(WeatherDataError) as weather_exc:
            ingester._get_venue_record_by_stadium_id(
                UNKNOWN_STADIUM_ID, _venues_frame()
            )
        with pytest.raises(contextual.UnknownStadiumError) as contextual_exc:
            contextual.venue_record_for_stadium_id(UNKNOWN_STADIUM_ID)

        assert str(weather_exc.value) == str(contextual_exc.value), (
            "the two consumers report DIFFERENT text for the same missing record. "
            "The type may differ -- WeatherDataError is this module's contract "
            "with its callers -- but the recovery instructions may not."
        )
        assert UNKNOWN_STADIUM_ID in str(weather_exc.value)
        assert "data/venues.json" in str(weather_exc.value)

    def test_a_filtered_frame_is_honoured_rather_than_the_file(self, ingester) -> None:
        """The adapter passes the FRAME's records, not the file's.

        If the delegation had reached for data/venues.json instead of the frame it
        was handed, a caller's filtered frame would be silently ignored -- and the
        Wave-12 reconciliation tests that remove a record to prove a refusal would
        pass while asserting nothing.
        """
        frame = _venues_frame()
        without_oakland = frame[frame["stadium_id"] != "OAK00"]
        with pytest.raises(WeatherDataError, match="OAK00"):
            ingester._get_venue_record_by_stadium_id("OAK00", without_oakland)
