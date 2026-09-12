"""The run's selection unit is the FREEZE INSTANT, not the calendar week.

Phase 33, Plan 33-05 Task 1 (COLD-03, D33-28, T-33-25).

WHY THE WEEK IS THE WRONG UNIT
------------------------------
A game's freeze is the Friday 6 PM Eastern instant preceding ITS OWN kickoff (D31-18), so one
Friday instant covers week N's Sunday and Monday games TOGETHER WITH week N+1's Thursday game.
MEASURED on the live 2026 capture, the 2026-09-18 22:00Z instant covers 14 week-2 Sunday games,
1 week-2 Monday game and 1 week-3 Thursday game.

A week-scoped binding refusal would therefore remove roughly EIGHTEEN Thursday games a season
from a forward measurement D40-08 defines as weeks 2-22: each one's freeze belongs to the
PREVIOUS week's Friday run, so a run scoped to week N+1 would meet it already past its freeze
and refuse it by name. That is the failure mode T-33-25 names, and scoping by instant is the
fix.

WHY A BATCH ABSTRACTION IS GENUINELY NEEDED (Codex HIGH, confirmed against live source)
--------------------------------------------------------------------------------------
``build_weekly_candidates(season, week, ...)`` is WEEK-SCOPED and returns a 2-tuple of frames.
One freeze instant spans two ``(season, week)`` groups by design, so the week-scoped function
cannot serve it and a caller cannot get there with one call.
``build_freeze_instant_candidates`` therefore groups the instant's games by ``(season, week)``,
calls the EXISTING week-scoped builder once per group, RESTRICTS each group's result to the
``game_id``s the instant actually selected, and concatenates. It is a wrapper, not a
reimplementation, and the delegation is asserted by a call-count stub rather than by reading it.

WHY THE BUILDER IS STUBBED IN THE WRAPPER TESTS, AND WHAT THAT DOES NOT CLAIM
----------------------------------------------------------------------------
``build_weekly_candidates`` scores three deployed artifacts against three gold matrices and
joins the silver odds snapshot. None of that exists for 2026 yet -- that is the cold start this
whole phase is about. The wrapper under test is a pure COMPOSITION over the builder, so a
counting stub is the correct instrument for the composition and it claims NOTHING about the
builder's own correctness, which ``tests/unit/test_weekly_bet_list.py`` and the Phase-31 suite
already cover.

Run this module:  uv run pytest tests/unit/test_selection_scoped_by_freeze_instant.py -q

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import pandas as pd
import pytest

from api.cache import PROVENANCE_FORWARD
from backtest import weekly_bet_list
from backtest.weekly_bet_list import (
    DECIDED_AT_COLUMN,
    assert_decided_at_before_freeze,
    build_freeze_instant_candidates,
    select_games_for_freeze_instant,
)
from scripts.ingest_historical_odds import (
    get_synthetic_snapshot_ts,
    require_aware_snapshot_ts,
)
from tests.fixtures.season_2026 import (
    CapturedScheduleUnavailableError,
    load_captured_schedule,
)

# MEASURED 2026-09-12 from the one freeze rule, not hand-typed: the week-2 Thursday game
# (DET @ BUF, gameday 2026-09-17) freezes SIX DAYS before the week-2 Sunday slate
# (gameday 2026-09-20). That six-day gap IS the reason a week-scoped fence loses the Thursday.
_WEEK_2_THURSDAY_GAMEDAY = "2026-09-17"
_WEEK_2_SUNDAY_GAMEDAY = "2026-09-20"
_WEEK_2_THURSDAY_FREEZE = "2026-09-11T22:00:00+00:00"
_WEEK_2_SUNDAY_FREEZE = "2026-09-18T22:00:00+00:00"

_ONE_MICROSECOND = timedelta(microseconds=1)


# ---------------------------------------------------------------------------
# The measured anchors.
# ---------------------------------------------------------------------------


class TestTheMeasuredFreezeAnchors:
    """The two instants every later assertion in this plan is expressed against."""

    def test_the_week_2_thursday_freeze_is_the_measured_instant(self) -> None:
        assert (
            get_synthetic_snapshot_ts(_WEEK_2_THURSDAY_GAMEDAY).isoformat()
            == _WEEK_2_THURSDAY_FREEZE
        )

    def test_the_week_2_sunday_freeze_is_the_measured_instant(self) -> None:
        assert (
            get_synthetic_snapshot_ts(_WEEK_2_SUNDAY_GAMEDAY).isoformat()
            == _WEEK_2_SUNDAY_FREEZE
        )

    def test_the_two_instants_are_exactly_seven_days_apart(self) -> None:
        """MEASURED SEVEN, not the six the plan's prose says (corrected in the SUMMARY).

        2026-09-11 22:00Z to 2026-09-18 22:00Z is one whole week. SIX days is the gap between
        the week-2 Thursday's own KICKOFF DATE (2026-09-17) and its freeze (2026-09-11) -- a
        different, also-true fact about the same pair of dates, and the one the prose meant.
        Both are asserted here so neither number can be quoted without its referent.
        """
        thursday_freeze = get_synthetic_snapshot_ts(_WEEK_2_THURSDAY_GAMEDAY)
        sunday_freeze = get_synthetic_snapshot_ts(_WEEK_2_SUNDAY_GAMEDAY)
        assert sunday_freeze - thursday_freeze == timedelta(days=7)

        kickoff_date = pd.Timestamp(_WEEK_2_THURSDAY_GAMEDAY).tz_localize(
            thursday_freeze.tzinfo
        )
        assert (kickoff_date - thursday_freeze).days == 5
        assert (
            kickoff_date.normalize() - pd.Timestamp(thursday_freeze).normalize()
        ).days == 6


# ---------------------------------------------------------------------------
# The real 2026 slate.
# ---------------------------------------------------------------------------


def _captured_schedule() -> pd.DataFrame:
    """The real 2026 capture as a selection schedule, or an evidence-backed skip."""
    try:
        feed = load_captured_schedule()
    except CapturedScheduleUnavailableError as exc:
        pytest.skip(str(exc))
    return feed[["game_id", "season", "week", "gameday", "weekday"]].copy()


class TestAFridayInstantSpansTwoWeeksOnTheRealSchedule:
    """D33-28 against real data: week N's Sunday/Monday PLUS week N+1's Thursday."""

    @pytest.mark.parametrize("week", [2, 3])
    def test_the_instant_carries_next_weeks_thursday_game(self, week: int) -> None:
        schedule = _captured_schedule()
        sunday = schedule[
            (schedule["week"] == week) & (schedule["weekday"] == "Sunday")
        ]
        assert not sunday.empty, f"week {week} has no Sunday game in the capture"

        instant = get_synthetic_snapshot_ts(str(sunday.iloc[0]["gameday"]))
        selected = select_games_for_freeze_instant(
            schedule, instant, now=instant - timedelta(seconds=1)
        )
        selected_ids = set(selected["game_id"])

        # Every one of week N's Sunday and Monday games is in scope.
        this_week = schedule[
            (schedule["week"] == week)
            & (schedule["weekday"].isin(["Sunday", "Monday"]))
        ]
        assert set(this_week["game_id"]) <= selected_ids

        # And so is week N+1's Thursday game -- the whole point.
        next_thursday = schedule[
            (schedule["week"] == week + 1) & (schedule["weekday"] == "Thursday")
        ]
        assert not next_thursday.empty, f"week {week + 1} has no Thursday game"
        assert set(next_thursday["game_id"]) <= selected_ids

        # The instant covers exactly TWO weeks, not one and not three.
        assert set(selected["week"]) == {week, week + 1}

    def test_the_week_2_thursday_belongs_to_week_1s_friday_run(self) -> None:
        """The DET @ BUF concession in one assertion: its freeze is a WEEK-1 instant.

        It is captured by the run scoped to 2026-09-11 22:00Z -- the Friday preceding week 1's
        Sunday slate -- not by the week-2 run. A week-scoped fence would meet it already past
        its freeze and refuse it, which is the roughly-eighteen-games-a-season loss T-33-25
        names.
        """
        schedule = _captured_schedule()
        thursday = schedule[schedule["gameday"] == _WEEK_2_THURSDAY_GAMEDAY]
        assert not thursday.empty

        instant = get_synthetic_snapshot_ts(_WEEK_2_THURSDAY_GAMEDAY)
        selected = select_games_for_freeze_instant(
            schedule, instant, now=instant - timedelta(seconds=1)
        )
        assert set(thursday["game_id"]) <= set(selected["game_id"])
        assert set(selected["week"]) == {1, 2}


# ---------------------------------------------------------------------------
# The batch wrapper over the week-scoped builder.
# ---------------------------------------------------------------------------


class _BuilderStub:
    """Counts calls and returns a whole-week frame, so a leak across instants is visible."""

    def __init__(self, schedule: pd.DataFrame) -> None:
        self._schedule = schedule
        self.calls: list[tuple[int, int]] = []

    def __call__(
        self, season: int, week: int, **_kwargs: Any
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        self.calls.append((int(season), int(week)))
        week_games = self._schedule[
            (self._schedule["season"] == season) & (self._schedule["week"] == week)
        ]
        candidates = pd.concat(
            [
                week_games.assign(target=target, model_value=float(index))
                for index, target in enumerate(("wp", "ats", "ou"))
            ],
            ignore_index=True,
        )
        return candidates, week_games.reset_index(drop=True)


@pytest.fixture
def two_week_schedule() -> pd.DataFrame:
    """Week 2's Sunday + Monday, week 2's Thursday (a DIFFERENT instant) and week 3's Thursday."""
    return pd.DataFrame(
        [
            {
                "game_id": "2026_02_DET_BUF",
                "season": 2026,
                "week": 2,
                "gameday": _WEEK_2_THURSDAY_GAMEDAY,
            },
            {
                "game_id": "2026_02_CAR_ATL",
                "season": 2026,
                "week": 2,
                "gameday": _WEEK_2_SUNDAY_GAMEDAY,
            },
            {
                "game_id": "2026_02_LV_NYJ",
                "season": 2026,
                "week": 2,
                "gameday": "2026-09-21",
            },
            {
                "game_id": "2026_03_ATL_GB",
                "season": 2026,
                "week": 3,
                "gameday": "2026-09-24",
            },
        ]
    )


@pytest.fixture
def builder_stub(
    monkeypatch: pytest.MonkeyPatch, two_week_schedule: pd.DataFrame
) -> _BuilderStub:
    stub = _BuilderStub(two_week_schedule)
    monkeypatch.setattr(weekly_bet_list, "build_weekly_candidates", stub)
    return stub


def _sunday_instant() -> datetime:
    return get_synthetic_snapshot_ts(_WEEK_2_SUNDAY_GAMEDAY)


class TestTheBatchWrapperSpansTheInstantsWeeks:
    """A cross-week instant yields candidates for BOTH groups, from the tested builder."""

    def test_the_returned_frame_carries_two_distinct_season_week_pairs(
        self, builder_stub: _BuilderStub, two_week_schedule: pd.DataFrame
    ) -> None:
        """Asserted on the PAIRS present, not on a row count -- a count can coincide."""
        instant = _sunday_instant()
        candidates, schedule = build_freeze_instant_candidates(
            instant,
            now=instant - timedelta(seconds=1),
            schedule=two_week_schedule,
        )

        pairs = set(zip(candidates["season"], candidates["week"], strict=True))
        assert pairs == {(2026, 2), (2026, 3)}
        assert set(schedule["week"]) == {2, 3}

    def test_it_delegates_to_the_week_scoped_builder_once_per_group(
        self, builder_stub: _BuilderStub, two_week_schedule: pd.DataFrame
    ) -> None:
        """The delegation claim, by call count: a reimplementation would call it zero times."""
        instant = _sunday_instant()
        build_freeze_instant_candidates(
            instant,
            now=instant - timedelta(seconds=1),
            schedule=two_week_schedule,
        )
        assert sorted(builder_stub.calls) == [(2026, 2), (2026, 3)]

    def test_a_game_on_a_different_instant_in_the_same_week_does_not_leak_in(
        self, builder_stub: _BuilderStub, two_week_schedule: pd.DataFrame
    ) -> None:
        """The restriction is load-bearing: the stub returns the WHOLE week 2 on purpose.

        ``2026_02_DET_BUF`` is a week-2 game whose freeze is a DIFFERENT instant, so a
        wrapper that merely concatenated the week-scoped output would publish it under this
        run and restate a decision made six days earlier.
        """
        instant = _sunday_instant()
        candidates, schedule = build_freeze_instant_candidates(
            instant,
            now=instant - timedelta(seconds=1),
            schedule=two_week_schedule,
        )
        assert "2026_02_DET_BUF" not in set(candidates["game_id"])
        assert "2026_02_DET_BUF" not in set(schedule["game_id"])
        assert "2026_02_CAR_ATL" in set(candidates["game_id"])

    def test_a_single_group_instant_matches_one_week_scoped_call(
        self, builder_stub: _BuilderStub, two_week_schedule: pd.DataFrame
    ) -> None:
        """The wrapper is a SUPERSET, not a behaviour change for the ordinary case.

        The week-2 Thursday's own instant covers only that one game here, so the wrapper's
        output must equal the week-scoped call restricted to it -- same columns, same values.
        """
        instant = get_synthetic_snapshot_ts(_WEEK_2_THURSDAY_GAMEDAY)
        candidates, _schedule = build_freeze_instant_candidates(
            instant,
            now=instant - timedelta(seconds=1),
            schedule=two_week_schedule,
        )
        direct, _direct_schedule = _BuilderStub(two_week_schedule)(2026, 2)
        expected = direct[direct["game_id"] == "2026_02_DET_BUF"].reset_index(drop=True)

        assert list(candidates.columns) == list(expected.columns)
        pd.testing.assert_frame_equal(
            candidates.reset_index(drop=True), expected, check_like=False
        )

    def test_an_instant_covering_no_game_is_refused_by_name(
        self, builder_stub: _BuilderStub, two_week_schedule: pd.DataFrame
    ) -> None:
        """An empty universe is refused rather than published as a run that recommended none.

        Same ruling ``_load_week_schedule`` already makes for an empty week.
        """
        instant = get_synthetic_snapshot_ts("2027-01-09")
        with pytest.raises(ValueError, match="no scheduled game"):
            build_freeze_instant_candidates(
                instant,
                now=instant - timedelta(seconds=1),
                schedule=two_week_schedule,
            )
        assert builder_stub.calls == []


# ---------------------------------------------------------------------------
# (f2) The two fences are JOINTLY SATISFIABLE.
# ---------------------------------------------------------------------------


class TestTheTwoFencesAreJointlySatisfiable:
    """A peer reviewer read R6 and R7 as contradictory. They are not, and this proves it."""

    def test_a_row_selected_a_microsecond_before_freeze_passes_the_write_assertion(
        self,
    ) -> None:
        """The two fences sit on OPPOSITE SIDES of the same boundary.

        Selection refuses ``now >= freeze``; the write asserts ``decided_at <= freeze``. So
        everything selection ADMITS satisfies the write assertion with STRICT inequality, and
        R7's equality case is an UNREACHABLE-IN-PRODUCTION boundary assertion proven on a
        constructed row rather than a normal-operation outcome.

        One test, deliberately: the emission and the write live in one sequence here because
        the disputed claim is about their conjunction, and splitting it would let each half
        pass while the conjunction went unproven.
        """
        gameday = "2026-09-24"
        freeze = get_synthetic_snapshot_ts(gameday)
        clock = freeze - _ONE_MICROSECOND

        schedule = pd.DataFrame(
            [
                {
                    "game_id": "2026_03_ATL_GB",
                    "season": 2026,
                    "week": 3,
                    "gameday": gameday,
                }
            ]
        )
        selected = select_games_for_freeze_instant(schedule, freeze, now=clock)
        assert set(selected["game_id"]) == {"2026_03_ATL_GB"}, (
            "selection refused a game one microsecond BEFORE its freeze; the fence is >=, "
            "so this row must be emitted"
        )

        row = {
            "game_id": "2026_03_ATL_GB",
            "provenance": PROVENANCE_FORWARD,
            "freeze_ts": freeze.isoformat(),
            DECIDED_AT_COLUMN: clock.isoformat(),
        }
        assert_decided_at_before_freeze(row)  # accepted: raises nothing

        decided = require_aware_snapshot_ts(row[DECIDED_AT_COLUMN])
        stored_freeze = require_aware_snapshot_ts(row["freeze_ts"])
        assert decided < stored_freeze, (
            "a row emitted before its freeze must carry decided_at_utc STRICTLY before it"
        )
        assert decided != stored_freeze
