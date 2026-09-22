"""The dated 2026 gold-default switch: the record of the hold, and of the hold ENDING.

THE HOLD HAS ENDED (Plan 33-15 Task 4, owner ruling 2026-09-14)
---------------------------------------------------------------
The flip condition below was MET when Phase 33 Wave 15's re-fit trained the deployed
artifacts on the corrected historical weather, and the owner removed the hold:
``features.weather.WEATHER_GOLD_DEFAULT_SEASONS`` is EMPTY by ruling, and the module's own
removal comment is the record of the hold ending. Plan 33.2-13 (orchestrator-assigned)
rewrote the eight tests that still asserted the hold so they pin THAT record instead: the
set is empty, the creating ruling and the removal are both still readable in committed
source, and a 2026 build takes the ORDINARY path -- the real observation reaches gold,
exactly as a 2025 build's does. Nothing here restores 2026 to the hold. The history below
is kept because it is why the switch existed.

WHAT WAS PINNED WHILE THE HOLD STOOD
------------------------------------
D33-25, as the owner RULED it on 2026-09-14 at Plan 33-09's Task-4 blocking
checkpoint: the GOLD weather family is held at its declared default for the 2026
season behind a named, DATED switch readable in committed source, so train and
serve agree while the silver data accumulates from week one.

THE FLIP CONDITION IS PHASE 33 WAVE 15'S RE-FIT, NOT PHASE 37. The plan as
written named Phase 37. The owner overrode that at the ruling, in their own
words: "The only justification for holding 2026 weather back was that the O/U
model had never seen weather vary. Phase 33.1 fixed the historical record, so
that reason expires at the next re-fit rather than a future phase." A test that
still asserted Phase 37 would pin the superseded instruction, so the assertion
below checks for Phase 33 Wave 15 AND asserts Phase 37 is absent -- the negative
half is what stops the old wording drifting back in.

WHY THE HELD STATE IS NULL AND NOT A NUMBER
-------------------------------------------
The plan's Task-5 behaviour line says the held season "takes its historical
default". That default WAS `temp_f: 65.0` and its one-hot family, and Plan
33.1-04 DELETED it -- D33.1-07 leaves exactly one spelling of "we do not know" in
`features/weather.py`, and the SPEC's first prohibition is that no other number
may take the old default's place under any name. So the held state is the
module's own absent-observation family: every weather column NULL, with
`weather_coverage` at 0.0 saying so on the row itself. A held row is therefore
DISTINGUISHABLE from an observed one in the data, which is the whole difference
between this switch and the silent Elo imputation it exists to be the opposite
of.

THE CONTROLS
------------
1. NON-VACUITY: the silver frame fed to every held build carries a REAL, VARYING
   outdoor observation (24F, 21 mph, snowing). The A/B below builds that exact
   frame with the switch EMPTIED and asserts the real values come through -- so a
   passing hold test proves the switch overrode live data, not that there was
   nothing to override.
2. THE ASSERTIONS: every weather column NULL, coverage 0.0, applicability kept.
3. NO FALSE POSITIVE: a 2025 build is identical with the switch active and with
   it emptied, asserted frame-by-frame on both builders.
4. THE SWITCH DOES NOT REACH SILVER: the input frame is unmutated, and the live
   ingest module carries no reference to the switch at all.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import inspect
import re
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

import utils.game_lock as lock_rule
from features import weather
from features.weather import (
    WEATHER_FLAG_COLUMNS,
    WEATHER_MERGE_KEY_COLUMNS,
    WeatherFeaturesCalculator,
)

HELD_SEASON = 2026
UNHELD_SEASON = 2025
HELD_GAME_ID = "2026_W02_KC@BUF"
UNHELD_GAME_ID = "2025_W02_KC@BUF"
FORECAST_TIME = datetime(2026, 9, 11, 18, 0)
# AWARE since Plan 33.2-12: the one weather fence admits a row when its forecast time is at
# or before min(the game's lock, the build instant) and refuses a naive instant rather than
# relabelling it. The silver row below is a LIVE forecast row, fetched two days before a
# Sunday 13:00 ET kickoff in its own season, so it is admitted for either season.
AS_OF = datetime(2026, 9, 11, 20, 0, tzinfo=UTC)


def _season_instant(game_id: str, month_day_time: str) -> pd.Timestamp:
    return pd.Timestamp(f"{game_id[:4]}-{month_day_time}", tz="UTC")


# The four columns the compressed builder emits beside `game_id`.
COMPRESSED_FEATURE_COLUMNS = (
    "weather_severity_score",
    "wind_mph",
    "is_precipitation",
    "is_outdoor",
)

# Every full-builder weather column EXCEPT the two flags, which are the only two
# a held row is allowed to carry a number in.
HELD_NULL_COLUMNS = tuple(
    column
    for column in weather.WEATHER_FEATURE_COLUMNS_BY_BUILDER["full"]
    if column not in WEATHER_FLAG_COLUMNS
)


def _games(season: int, game_id: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": game_id,
                "season": season,
                "week": 2,
                "kickoff_et": _season_instant(game_id, "09-13 17:00"),
            }
        ]
    )


def _silver(game_id: str) -> pd.DataFrame:
    """A REAL, VARYING outdoor observation -- the values the switch must override.

    Deliberately extreme and deliberately nothing like the deleted mild default:
    if a held build ever let these through, `temp_f` would read 24.0 and
    `weather_severity_score` would be large, both of which the assertions below
    would catch immediately.
    """
    return pd.DataFrame(
        [
            {
                "game_id": game_id,
                "forecast_time": _season_instant(game_id, "09-11 18:00"),
                "weather_source": "forecast",
                "is_outdoor": True,
                "weather_coverage": True,
                "temp_f": 24.0,
                "wind_mph": 21.0,
                "precip_prob": 0.9,
                "precip_mm": 6.0,
                "humidity_pct": 82.0,
                "condition": "Snow",
            }
        ]
    )


def _build_full(games: pd.DataFrame, silver: pd.DataFrame) -> pd.DataFrame:
    calculator = WeatherFeaturesCalculator()
    with pytest.warns(DeprecationWarning):
        return calculator.build_weather_features(games, weather_df=silver)


def _build_compressed(games: pd.DataFrame, silver: pd.DataFrame) -> pd.DataFrame:
    calculator = WeatherFeaturesCalculator()
    return calculator.build_features(games, AS_OF, weather_df=silver)


def _is_null(value: object) -> bool:
    if value is None:
        return True
    result = pd.isna(value)
    return bool(result) if isinstance(result, bool) else False


@pytest.fixture
def switch_emptied(monkeypatch: pytest.MonkeyPatch) -> None:
    """The A/B control arm: the module with no season held.

    Patching the module constant rather than editing the source is what makes
    this a CONTROLLED comparison -- the same builder, the same silver frame, one
    variable moved.
    """
    monkeypatch.setattr(weather, "WEATHER_GOLD_DEFAULT_SEASONS", frozenset())


# The line in features/weather.py that separates the retained record of the ruling that
# CREATED the hold from the record of the hold ENDING (Plan 33-15 Task 4).
_REMOVAL_MARKER = "THE CONDITION WAS MET, AND THE HOLD IS REMOVED"


def _creating_record() -> str:
    """The committed source immediately BEFORE the removal record: the creating ruling."""
    source = inspect.getsource(weather)
    index = source.find(_REMOVAL_MARKER)
    assert index > 0, "the removal record is gone from features/weather.py"
    return source[max(0, index - 2600) : index]


def _removal_record() -> str:
    """The committed source from the removal marker to the (emptied) constant."""
    source = inspect.getsource(weather)
    start = source.find(_REMOVAL_MARKER)
    end = source.find("WEATHER_GOLD_DEFAULT_SEASONS:", start)
    assert 0 < start < end, "the removal record must precede the constant it empties"
    return source[start:end]


class TestTheSwitchRecordsTheHoldEnding:
    """The switch stays readable in committed source -- now as a hold that ENDED."""

    def test_the_held_season_set_is_empty(self):
        """Was: the switch names 2026. The owner ended the hold; the set is EMPTY."""
        assert isinstance(weather.WEATHER_GOLD_DEFAULT_SEASONS, frozenset)
        assert HELD_SEASON not in weather.WEATHER_GOLD_DEFAULT_SEASONS
        assert frozenset() == weather.WEATHER_GOLD_DEFAULT_SEASONS

    def test_no_season_is_held_and_the_hold_was_not_re_dated(self):
        """Was: no season other than 2026 is held. The bridge is removed, not moved:
        the flip condition said "REMOVED, not re-dated", so no other season took 2026's
        place."""
        assert set(weather.WEATHER_GOLD_DEFAULT_SEASONS) == set()
        assert "MET AND REMOVED" in weather.WEATHER_GOLD_DEFAULT_FLIP_CONDITION

    def test_the_flip_condition_names_phase_33_wave_15(self):
        flip = weather.WEATHER_GOLD_DEFAULT_FLIP_CONDITION
        assert flip.strip(), "an empty flip condition is a hold with no expiry"
        assert "Phase 33" in flip
        assert "Wave 15" in flip

    def test_the_flip_condition_does_not_still_name_phase_37(self):
        """The owner's 2026-09-14 ruling SUPERSEDED the plan's Phase 37 wording.

        Asserted as its own test rather than folded into the one above, because
        the failure modes differ: one catches a flip condition that never named
        the re-fit, this one catches the superseded instruction drifting back.
        """
        assert "Phase 37" not in weather.WEATHER_GOLD_DEFAULT_FLIP_CONDITION

    def test_the_creating_ruling_and_the_removal_are_both_dated_and_attributed(self):
        """Was: the comment carries the date, the measurement and the ruling.

        The creating ruling is RETAINED UNEDITED above the removal record (the module
        says so), so it is read there -- the removal record now sits between it and the
        constant, which is why the old fixed window before the constant stopped reaching
        it. The removal record must itself be dated and name the owner's ruling.
        """
        created = _creating_record()[-1200:]
        assert re.search(r"20\d\d-\d\d-\d\d", created), "the ruling is undated"
        assert "constan" in created.lower(), "the ruling cites no measurement"
        assert "owner" in created.lower(), "the ruling names no authorising owner"

        removed = _removal_record()
        assert "2026-09-14" in removed, "the removal is undated"
        assert "Plan 33-15" in removed, "the removal names no plan"
        assert "owner" in removed.lower(), "the removal names no authorising ruling"

    def test_the_retained_record_still_cites_task_3s_numbers_by_value(self):
        """Was: the comment cites Task 3's re-derived numbers by value.

        Still cited BY VALUE in the retained creating record, so the reason the hold
        existed cannot drift from `tests/phase33_state.GOLD_WEATHER_CONSTANCY_MEASUREMENT`.
        """
        created = _creating_record()
        assert "45 of 46" in created
        assert "99.7846" in created or "6,485 of 6,499" in created


class TestA2026BuildTakesTheOrdinaryPath:
    """With the hold ended, a 2026 build reads silver exactly as a 2025 build does."""

    def test_every_weather_column_carries_the_real_observation_on_a_2026_row(self):
        """Was: every weather column is NULL on a held row. The hold is gone, so the
        live 2026 observation reaches gold -- the models now promoted were trained on
        weather that varies, which is the whole reason the owner ended it."""
        frame = _build_full(_games(HELD_SEASON, HELD_GAME_ID), _silver(HELD_GAME_ID))
        row = frame.iloc[0]

        assert row["temp_f"] == 24.0
        assert row["raw_temp_f"] == 24.0
        assert row["wind_mph"] == 21.0
        assert row["weather_severity_score"] > 0.0
        nulls = [column for column in HELD_NULL_COLUMNS if _is_null(row[column])]
        assert len(nulls) < len(HELD_NULL_COLUMNS), (
            f"every weather column is NULL on a 2026 row ({nulls}): the retired hold "
            "is still in force"
        )

    def test_a_2026_row_says_weather_applies_and_was_observed(self):
        """Was: a held row says weather applies and was NOT observed. Applicability
        is unchanged; the coverage flag now says the observation is there."""
        frame = _build_full(_games(HELD_SEASON, HELD_GAME_ID), _silver(HELD_GAME_ID))
        row = frame.iloc[0]

        assert row["weather_affects_game"] == 1.0
        assert row["weather_coverage"] == 1.0, (
            "a 2026 row with a real observation must say so IN THE DATA; 0.0 here "
            "would mean the retired hold is still withholding it"
        )

    def test_the_same_silver_frame_yields_real_values_when_the_switch_is_empty(
        self, switch_emptied
    ):
        """NON-VACUITY. Without this the hold test could pass on an empty frame."""
        frame = _build_full(_games(HELD_SEASON, HELD_GAME_ID), _silver(HELD_GAME_ID))
        row = frame.iloc[0]

        assert row["temp_f"] == 24.0
        assert row["wind_mph"] == 21.0
        assert row["raw_temp_f"] == 24.0
        assert not _is_null(row["weather_severity_score"])
        assert row["weather_severity_score"] > 0.0
        assert row["weather_coverage"] == 1.0

    def test_the_compressed_builder_no_longer_holds_2026_either(self):
        """Was: the compressed builder holds 2026 too. Both builders feed gold, so
        the hold's ending must reach both, or it is half an ending."""
        frame = _build_compressed(
            _games(HELD_SEASON, HELD_GAME_ID), _silver(HELD_GAME_ID)
        )
        row = frame.iloc[0]

        assert row["wind_mph"] == 21.0
        assert row["is_precipitation"] == 1.0
        assert row["weather_severity_score"] > 0.0
        assert row["is_outdoor"] == 1.0

    def test_the_compressed_builder_yields_real_values_when_the_switch_is_empty(
        self, switch_emptied
    ):
        frame = _build_compressed(
            _games(HELD_SEASON, HELD_GAME_ID), _silver(HELD_GAME_ID)
        )
        row = frame.iloc[0]

        assert row["wind_mph"] == 21.0
        assert row["is_precipitation"] == 1.0
        assert row["weather_severity_score"] > 0.0

    def test_the_single_game_path_is_not_held_either(self):
        """Was: the hold applies when only the game_id carries the season.

        `get_features_for_game` builds a frame with `season: 0`, which is why the
        season resolver falls back to the game_id. With no season held, resolving
        2026 from the id no longer holds anything: the single-game path takes the
        ordinary fenced path. That path needs the game's kickoff to find its lock
        (Plan 33.2-12), so a frame without one is refused by name rather than built.
        """
        games = _games(HELD_SEASON, HELD_GAME_ID).assign(season=0, week=0)
        row = _build_compressed(games, _silver(HELD_GAME_ID)).iloc[0]

        assert row["wind_mph"] == 21.0
        assert row["is_outdoor"] == 1.0

        kickoff_less = pd.DataFrame([{"game_id": HELD_GAME_ID, "season": 0, "week": 0}])
        with pytest.raises(lock_rule.MissingKickoffError):
            _build_compressed(kickoff_less, _silver(HELD_GAME_ID))


class TestANonHeldSeasonIsUnchanged:
    """History must read exactly as it reads today. The switch is 2026-only."""

    def test_a_2025_full_build_is_identical_with_and_without_the_switch(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        games = _games(UNHELD_SEASON, UNHELD_GAME_ID)
        with_switch = _build_full(games, _silver(UNHELD_GAME_ID))

        monkeypatch.setattr(weather, "WEATHER_GOLD_DEFAULT_SEASONS", frozenset())
        without_switch = _build_full(games, _silver(UNHELD_GAME_ID))

        pd.testing.assert_frame_equal(with_switch, without_switch)

    def test_a_2025_compressed_build_is_identical_with_and_without_the_switch(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        games = _games(UNHELD_SEASON, UNHELD_GAME_ID)
        with_switch = _build_compressed(games, _silver(UNHELD_GAME_ID))

        monkeypatch.setattr(weather, "WEATHER_GOLD_DEFAULT_SEASONS", frozenset())
        without_switch = _build_compressed(games, _silver(UNHELD_GAME_ID))

        pd.testing.assert_frame_equal(with_switch, without_switch)

    def test_a_2025_build_still_carries_the_real_observation(self):
        """The identity test above would also pass if BOTH arms were empty."""
        row = _build_full(
            _games(UNHELD_SEASON, UNHELD_GAME_ID), _silver(UNHELD_GAME_ID)
        ).iloc[0]

        assert row["temp_f"] == 24.0
        assert row["weather_coverage"] == 1.0


class TestSilverKeepsAccumulating:
    """The switch holds 2026 OUT OF GOLD. It does not stop collecting it."""

    def test_a_held_build_does_not_mutate_the_silver_frame(self):
        silver = _silver(HELD_GAME_ID)
        before = silver.copy(deep=True)

        _build_full(_games(HELD_SEASON, HELD_GAME_ID), silver)

        pd.testing.assert_frame_equal(silver, before)

    def test_the_live_ingest_path_carries_no_reference_to_the_switch(self):
        """A structural proof, and stated as one.

        A source scan can show that the ingest module CANNOT consult the switch.
        It says nothing about what the ingest module writes -- that is proven by
        `tests/unit/test_weather_pipeline.py` driving the path and reading the
        rows back.
        """
        source = Path("scripts/ingest_weather.py").read_text(encoding="utf-8")

        assert "WEATHER_GOLD_DEFAULT" not in source
        assert "features.weather" not in source

    def test_the_switch_is_declared_in_exactly_one_place(self):
        """A module constant with one home, not a config key and not a predicate.

        RECORDED DISCRETION from Plan 33-09 Task 5: a module constant is
        greppable and its comment travels with it.
        """
        source = inspect.getsource(weather)
        assignments = re.findall(
            r"^WEATHER_GOLD_DEFAULT_SEASONS\s*:", source, flags=re.MULTILINE
        )

        assert len(assignments) == 1


class TestTheHeldFrameStillSatisfiesTheColumnContract:
    """A held build is a NORMAL build with different values, not a narrower one."""

    def test_the_full_builder_emits_its_declared_columns_on_a_held_row(self):
        frame = _build_full(_games(HELD_SEASON, HELD_GAME_ID), _silver(HELD_GAME_ID))
        emitted = set(frame.columns) - set(WEATHER_MERGE_KEY_COLUMNS)

        assert emitted == set(weather.WEATHER_FEATURE_COLUMNS_BY_BUILDER["full"])

    def test_the_compressed_builder_emits_its_declared_columns_on_a_held_row(self):
        frame = _build_compressed(
            _games(HELD_SEASON, HELD_GAME_ID), _silver(HELD_GAME_ID)
        )

        assert set(frame.columns) == {"game_id", *COMPRESSED_FEATURE_COLUMNS}
