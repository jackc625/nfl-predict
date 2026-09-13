"""An unresolvable stadium stops the run and names the game. So does a bad id.

WHAT IS BEING PINNED
--------------------
SPEC R4, first half: "an unresolvable `stadium_id` raises and names the game".
Plan 33.1-02 Ruling D4: a malformed `game_id` is a REFUSAL, not a skip. Plan
33.1-02 Ruling D2: `stadium_id` is RECONCILED between silver and the pinned feed,
never joined blind.

All three are the same principle in three places -- a wrong answer must not arrive
QUIETLY -- and all three are about a corpus that has to be EXACT. Over a 4,847-game
paced run a silently skipped game surfaces later as a season the coverage test
calls incomplete for a reason no log line explains, and the operator's next move is
to re-run the season and spend the budget again.

WHY THE HOME-TEAM FALLBACK IS GONE, NOT MERELY UNUSED
------------------------------------------------------
For 1,082 historical games the home team's present-day stadium was built AFTER the
game was played. `_get_venue_record_by_stadium_id` raises instead, exactly and
case-sensitively, and the refusal names `data/venues.json` as the file to edit --
the repo's refusal-carries-its-recovery-command convention.

FOUR CONTROLS, copied from `tests/unit/test_weather_archive_quarantined.py:31-42`
---------------------------------------------------------------------------------
1. NON-VACUITY: the "unknown" id is unknown BY CONSTRUCTION, with a control that
   fails if it ever becomes known. (Plan 33.1-01 learned this the hard way: its
   predecessor used `DAL99`, a real nflverse code that merely had no record yet,
   and adding the 22 historical venues made the premise false while the sibling
   test kept passing.)
2. THE ASSERTIONS: the refusals fire and name what they must.
3. PLANTED VIOLATIONS: a disagreeing `stadium_id` and a malformed `game_id` are
   injected, proving each check FIRES rather than only ever finding nothing.
4. NO FALSE POSITIVE: a PRESENT id and a well-formed id do not raise.

This module writes nothing and reaches no network. It needs no
`writes_production_store` marker.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

from utils.exceptions import DataIngestionError, WeatherDataError

# NOT an nflverse code. Plan 33.1-01's `test_stadium_id_routing` used `DAL99`
# -- Texas Stadium, a real code with no record -- and this phase made it known.
# `ZZZ99` is unknown BY CONSTRUCTION, and the control below fails if that changes.
UNKNOWN_STADIUM_ID = "ZZZ99"

VENUES_JSON = "data/venues.json"

OPEN_GAME_ID = "2016_W01_DET@IND"
CLOSED_GAME_ID = "2016_W03_LAC@IND"

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
    with patch("scripts.ingest_weather.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock()
        from scripts.backfill_historical_weather import HistoricalWeatherBackfiller

        return HistoricalWeatherBackfiller()


@pytest.fixture
def venues_df():
    return pd.DataFrame(json.loads(Path(VENUES_JSON).read_text())["venues"])


class TestTheUnknownIdFixtureIsGenuinelyUnknown:
    """CONTROL 1 -- non-vacuity."""

    def test_the_unknown_stadium_id_resolves_to_no_record(self, venues_df):
        assert UNKNOWN_STADIUM_ID not in set(venues_df["stadium_id"]), (
            f"{UNKNOWN_STADIUM_ID} has become a real venue record. Every refusal "
            "test below would then be asserting against a premise that quietly "
            "became false -- which is exactly what happened to DAL99 in Plan "
            "33.1-01. Choose a new code that is not an nflverse code."
        )

    def test_the_unknown_stadium_id_is_absent_from_the_pinned_feed(self):
        from scripts.backfill_historical_weather import load_pinned_game_facts

        facts = load_pinned_game_facts([2016])
        assert UNKNOWN_STADIUM_ID not in set(facts["stadium_id"])


class TestAnUnknownStadiumIdRaisesAndNamesItself:
    """CONTROL 2 -- the assertion. CONTROL 3 -- the id is PLANTED."""

    def test_it_raises_quoting_the_id_and_the_file_to_edit(self, backfiller, venues_df):
        with pytest.raises(WeatherDataError) as excinfo:
            backfiller._get_venue_record_by_stadium_id(UNKNOWN_STADIUM_ID, venues_df)

        message = str(excinfo.value)
        assert UNKNOWN_STADIUM_ID in message, (
            "a refusal that does not quote the id leaves the reader to guess "
            "which of 6,499 games it was about."
        )
        assert VENUES_JSON in message, (
            "the repo's convention is that a refusal carries its recovery "
            "command. Naming the file to edit is that command here."
        )

    def test_it_does_not_fall_back_to_the_home_teams_stadium(
        self, backfiller, venues_df, per_game_roof_games
    ):
        """The PLANTED violation: a real game whose stadium_id is overwritten.

        The home team is IND and IND00 is a perfectly resolvable record, so a
        surviving home-team fallback would return a row instead of raising -- and
        would return the RIGHT row, which is what makes this failure mode so quiet
        for the 1,082 historical games where it returns the wrong one.
        """
        frame = per_game_roof_games(with_silver_stadium_id=True)
        frame.loc[:, "stadium_id"] = UNKNOWN_STADIUM_ID

        fetcher = AsyncMock(return_value=_OBSERVATION)
        with patch.object(backfiller, "_fetch_openmeteo_weather", fetcher):
            with pytest.raises((WeatherDataError, DataIngestionError)) as excinfo:
                backfiller.fetch_weather_for_games(frame, venues_df)

        assert UNKNOWN_STADIUM_ID in str(excinfo.value)
        assert fetcher.await_count == 0, (
            "no observation should be fetched for a game that cannot be placed."
        )


class TestNoFalsePositiveOnAPresentId:
    """CONTROL 4 -- the refusal is reachable, not universal."""

    def test_a_present_stadium_id_returns_a_record(self, backfiller, venues_df):
        record = backfiller._get_venue_record_by_stadium_id("IND00", venues_df)
        assert record["stadium_id"] == "IND00"
        assert record["timezone"]

    def test_a_well_formed_pair_of_games_produces_two_rows(
        self, backfiller, venues_df, per_game_roof_games
    ):
        with patch.object(
            backfiller, "_fetch_openmeteo_weather", AsyncMock(return_value=_OBSERVATION)
        ):
            result = backfiller.fetch_weather_for_games(
                per_game_roof_games(), venues_df
            )
        assert set(result["game_id"]) == {OPEN_GAME_ID, CLOSED_GAME_ID}


class TestTheWave12StadiumIdReconciliation:
    """Ruling D2, asserted in ALL THREE input shapes.

    Phase 33 Wave 12 adds `stadium_id` to silver `games`. It has not run. A plan
    that assumed either shape would break on the other, and the tracer is wave 2
    of an eleven-wave serial chain, so the defect would block all ten later waves.
    """

    def test_the_post_wave_12_shape_collapses_to_exactly_one_column(
        self, backfiller, per_game_roof_games
    ):
        prepared = backfiller._prepare_games_frame(
            per_game_roof_games(with_silver_stadium_id=True)
        )
        stadium_columns = [c for c in prepared.columns if "stadium_id" in c]
        assert stadium_columns == ["stadium_id"], (
            "a plain pandas merge of two frames that BOTH carry stadium_id yields "
            "stadium_id_x and stadium_id_y and NO stadium_id, and then every "
            "row['stadium_id'] downstream raises. Got: " + repr(stadium_columns)
        )
        assert set(prepared["stadium_id"]) == {"IND00"}

    def test_the_pre_wave_12_shape_takes_the_pinned_value(
        self, backfiller, per_game_roof_games
    ):
        frame = per_game_roof_games(with_silver_stadium_id=False)
        assert "stadium_id" not in frame.columns, (
            "this control needs the PRE-Wave-12 shape. If silver games has "
            "started carrying stadium_id, the fixture flag is what to change."
        )

        prepared = backfiller._prepare_games_frame(frame)
        assert [c for c in prepared.columns if "stadium_id" in c] == ["stadium_id"]
        assert set(prepared["stadium_id"]) == {"IND00"}

    def test_a_disagreement_raises_naming_the_game_and_both_values(
        self, backfiller, per_game_roof_games
    ):
        """CONTROL 3 -- a PLANTED disagreement, proving the comparison fires.

        Silver's `stadium_id` is a transform of the same pinned feed, so a
        disagreement means the pin moved or the transform is wrong. Neither side
        is allowed to win silently.
        """
        frame = per_game_roof_games(with_silver_stadium_id=True)
        frame.loc[frame["game_id"] == OPEN_GAME_ID, "stadium_id"] = "DEN00"

        with pytest.raises(DataIngestionError) as excinfo:
            backfiller._prepare_games_frame(frame)

        message = str(excinfo.value)
        assert OPEN_GAME_ID in message, "the disagreeing game must be named"
        assert "DEN00" in message, "the silver value must be shown"
        assert "IND00" in message, "the pinned value must be shown"

    def test_an_agreeing_frame_does_not_raise(self, backfiller, per_game_roof_games):
        """CONTROL 4 -- the comparison is not simply always true."""
        prepared = backfiller._prepare_games_frame(
            per_game_roof_games(with_silver_stadium_id=True)
        )
        assert len(prepared) == 2


class TestAMalformedGameIdIsARefusalNotASkip:
    """Ruling D4. CONTROL 3 -- the bad id is PLANTED into a real frame."""

    def test_it_raises_naming_the_game_before_any_request_is_issued(
        self, backfiller, venues_df, per_game_roof_games
    ):
        frame = per_game_roof_games()
        frame.loc[0, "game_id"] = "NOT-A-GAME-ID"

        fetcher = AsyncMock(return_value=_OBSERVATION)
        with patch.object(backfiller, "_fetch_openmeteo_weather", fetcher):
            with pytest.raises(WeatherDataError) as excinfo:
                backfiller.fetch_weather_for_games(frame, venues_df)

        assert "NOT-A-GAME-ID" in str(excinfo.value)
        assert fetcher.await_count == 0, (
            "the refusal must fire BEFORE the season is fetched. A refusal after "
            "140 requests has already spent budget the resume rule cannot give "
            f"back. {fetcher.await_count} request(s) were issued."
        )

    def test_the_sibling_game_produces_no_row_either(
        self, backfiller, venues_df, per_game_roof_games
    ):
        """All-or-nothing. A partial frame is the quieter form of the same hole."""
        frame = per_game_roof_games()
        frame.loc[0, "game_id"] = "NOT-A-GAME-ID"

        with patch.object(
            backfiller, "_fetch_openmeteo_weather", AsyncMock(return_value=_OBSERVATION)
        ):
            with pytest.raises(WeatherDataError):
                backfiller.fetch_weather_for_games(frame, venues_df)

    def test_a_well_formed_frame_is_not_refused(
        self, backfiller, venues_df, per_game_roof_games
    ):
        """CONTROL 4 -- the id check is reachable, not universal."""
        with patch.object(
            backfiller, "_fetch_openmeteo_weather", AsyncMock(return_value=_OBSERVATION)
        ):
            result = backfiller.fetch_weather_for_games(
                per_game_roof_games(), venues_df
            )
        assert len(result) == 2
