"""The run's selection unit is the LOCK INSTANT, not the calendar week.

Phase 33, Plan 33-05 Task 1 (COLD-03, D33-28, T-33-25); re-expressed on the day-before lock by
Plan 33.2-02 (D33.2-01, D33.2-05, D33.2-18). The module keeps its original file name so its
history stays traceable; its subject is now each game's LOCK.

WHY THE WEEK IS THE WRONG UNIT
------------------------------
A game locks at 18:00 Eastern on the Eastern day before ITS OWN kickoff, so a week's games lock
on several different days: a Thursday game on the Wednesday, the Sunday slate on the Saturday,
Monday night on the Sunday. MEASURED on the live 2026 capture with the one rule: 57 distinct
lock instants across the 272-game regular season, the 2026-09-19 18:00 ET instant covering all
14 week-2 Sunday games, and no instant covering two NFL weeks. A week-scoped run would meet the
week's Thursday game long past its lock and refuse it -- roughly EIGHTEEN games a season removed
from a forward measurement D40-08 defines as weeks 2-22 (T-33-25).

WHY A BATCH ABSTRACTION IS STILL NEEDED
---------------------------------------
``build_weekly_candidates(season, week, ...)`` is WEEK-SCOPED and returns a 2-tuple of frames.
An instant's games normally sit in one week, but nothing in the rule guarantees it -- a game
moved onto another week's date (2020 had many) shares that date's lock -- so
``build_freeze_instant_candidates`` groups the instant's games by ``(season, week)``, calls the
EXISTING week-scoped builder once per group, RESTRICTS each group's result to the ``game_id``s
the instant selected, and concatenates. The delegation is asserted by a call-count stub.

WHY THE BUILDER IS STUBBED IN THE WRAPPER TESTS, AND WHAT THAT DOES NOT CLAIM
----------------------------------------------------------------------------
``build_weekly_candidates`` scores three deployed artifacts against three gold matrices and
joins the silver odds snapshot. The wrapper under test is a pure COMPOSITION over the builder,
so a counting stub is the correct instrument and it claims NOTHING about the builder's own
correctness; ``tests/unit/test_weekly_candidates_exclusion.py`` drives the REAL builder for the
exclusion claims.

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
    select_games_for_decision_instant,
)
from scripts.ingest_historical_odds import gameday_lock, require_aware_snapshot_ts
from tests.fixtures.season_2026 import (
    CapturedScheduleUnavailableError,
    load_captured_schedule,
)

# MEASURED 2026-09-21 through the one rule, not hand-typed: the week-2 Thursday game (DET @ BUF,
# gameday 2026-09-17) locks Wednesday 2026-09-16 and the week-2 Sunday slate (gameday
# 2026-09-20) locks Saturday 2026-09-19, both 18:00 EDT = 22:00 UTC.
_WEEK_2_THURSDAY_GAMEDAY = "2026-09-17"
_WEEK_2_SUNDAY_GAMEDAY = "2026-09-20"
_WEEK_2_THURSDAY_LOCK = "2026-09-16T22:00:00+00:00"
_WEEK_2_SUNDAY_LOCK = "2026-09-19T22:00:00+00:00"

# The real 2026 capture's lock-instant count, measured 2026-09-21.
_CAPTURED_2026_LOCK_INSTANTS = 57

_ONE_SECOND = timedelta(seconds=1)


def _utc_text(value: datetime) -> str:
    return require_aware_snapshot_ts(value).isoformat()


# ---------------------------------------------------------------------------
# The measured anchors.
# ---------------------------------------------------------------------------


class TestTheMeasuredLockAnchors:
    """The two instants every later assertion in this module is expressed against."""

    def test_the_week_2_thursday_lock_is_the_measured_instant(self) -> None:
        assert (
            _utc_text(gameday_lock(_WEEK_2_THURSDAY_GAMEDAY)) == _WEEK_2_THURSDAY_LOCK
        )

    def test_the_week_2_sunday_lock_is_the_measured_instant(self) -> None:
        assert _utc_text(gameday_lock(_WEEK_2_SUNDAY_GAMEDAY)) == _WEEK_2_SUNDAY_LOCK

    def test_the_two_locks_are_three_days_apart(self) -> None:
        """One week's two locks are three days apart, each the ET day before its own game.

        The retired rule put these two games' instants SEVEN days apart and the Thursday
        game's instant five days before its kickoff; the lock is 26 hours before it.
        """
        thursday = gameday_lock(_WEEK_2_THURSDAY_GAMEDAY)
        sunday = gameday_lock(_WEEK_2_SUNDAY_GAMEDAY)
        assert sunday - thursday == timedelta(days=3)
        assert (
            pd.Timestamp(_WEEK_2_THURSDAY_GAMEDAY).date() - thursday.date()
        ).days == 1


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


class TestALockInstantCoversOneCalendarDayOnTheRealSchedule:
    """D33.2-01 against real data: one instant, one Eastern gameday, whatever the week."""

    def test_the_sunday_lock_carries_the_whole_sunday_slate_and_nothing_else(
        self,
    ) -> None:
        schedule = _captured_schedule()
        sunday = schedule[(schedule["week"] == 2) & (schedule["weekday"] == "Sunday")]
        assert not sunday.empty, "week 2 has no Sunday game in the capture"

        instant = gameday_lock(_WEEK_2_SUNDAY_GAMEDAY)
        selected = select_games_for_decision_instant(
            schedule, instant, decided_at=instant
        )

        assert set(selected["game_id"]) == set(sunday["game_id"])
        assert set(selected["gameday"]) == {_WEEK_2_SUNDAY_GAMEDAY}

    def test_the_week_2_thursday_has_its_own_wednesday_lock(self) -> None:
        """The DET @ BUF case in one assertion: its lock is NOT the Sunday slate's."""
        schedule = _captured_schedule()
        thursday = schedule[schedule["gameday"] == _WEEK_2_THURSDAY_GAMEDAY]
        assert not thursday.empty

        instant = gameday_lock(_WEEK_2_THURSDAY_GAMEDAY)
        selected = select_games_for_decision_instant(
            schedule, instant, decided_at=instant
        )
        assert set(selected["game_id"]) == set(thursday["game_id"])
        assert set(selected["week"]) == {2}

    def test_every_game_belongs_to_exactly_one_instant(self) -> None:
        """The partition claim, counted: the union of the instants' selections IS the season."""
        schedule = _captured_schedule()
        instants = sorted(
            {gameday_lock(str(day)) for day in schedule["gameday"].dropna().unique()}
        )
        assert len(instants) == _CAPTURED_2026_LOCK_INSTANTS

        seen: list[str] = []
        for instant in instants:
            selected = select_games_for_decision_instant(
                schedule, instant, decided_at=instant
            )
            assert selected["gameday"].nunique() == 1, instant
            seen.extend(str(game_id) for game_id in selected["game_id"])

        assert len(seen) == len(set(seen)), "a game was selected by two instants"
        assert set(seen) == set(schedule["game_id"].astype(str))


# ---------------------------------------------------------------------------
# The batch wrapper over the week-scoped builder.
# ---------------------------------------------------------------------------


class _BuilderStub:
    """Counts calls, records the exclusion set, and returns a whole-week frame."""

    def __init__(self, schedule: pd.DataFrame) -> None:
        self._schedule = schedule
        self.calls: list[tuple[int, int]] = []
        self.excluded_seen: list[frozenset[str]] = []

    def __call__(
        self, season: int, week: int, **kwargs: Any
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        self.calls.append((int(season), int(week)))
        excluded = kwargs.get("excluded_game_ids", frozenset())
        self.excluded_seen.append(excluded)
        week_games = self._schedule[
            (self._schedule["season"] == season)
            & (self._schedule["week"] == week)
            & ~self._schedule["game_id"].isin(excluded)
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
    """Week 2's Thursday (its own lock), two week-2 Sunday games, and a week-3 game moved onto
    the same Sunday -- so one lock instant spans two (season, week) groups, the case the wrapper
    exists for.
    """
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
                "game_id": "2026_02_MIN_CHI",
                "season": 2026,
                "week": 2,
                "gameday": _WEEK_2_SUNDAY_GAMEDAY,
            },
            {
                "game_id": "2026_03_ATL_GB",
                "season": 2026,
                "week": 3,
                "gameday": _WEEK_2_SUNDAY_GAMEDAY,
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


def _sunday_lock() -> datetime:
    return gameday_lock(_WEEK_2_SUNDAY_GAMEDAY)


class TestTheBatchWrapperSpansTheInstantsWeeks:
    """A cross-week instant yields candidates for BOTH groups, from the tested builder."""

    def test_the_returned_frame_carries_two_distinct_season_week_pairs(
        self, builder_stub: _BuilderStub, two_week_schedule: pd.DataFrame
    ) -> None:
        """Asserted on the PAIRS present, not on a row count -- a count can coincide."""
        instant = _sunday_lock()
        candidates, schedule = build_freeze_instant_candidates(
            instant, decided_at=instant, schedule=two_week_schedule
        )

        pairs = set(zip(candidates["season"], candidates["week"], strict=True))
        assert pairs == {(2026, 2), (2026, 3)}
        assert set(schedule["week"]) == {2, 3}

    def test_it_delegates_to_the_week_scoped_builder_once_per_group(
        self, builder_stub: _BuilderStub, two_week_schedule: pd.DataFrame
    ) -> None:
        """The delegation claim, by call count: a reimplementation would call it zero times."""
        instant = _sunday_lock()
        build_freeze_instant_candidates(
            instant, decided_at=instant, schedule=two_week_schedule
        )
        assert sorted(builder_stub.calls) == [(2026, 2), (2026, 3)]

    def test_a_game_on_a_different_instant_in_the_same_week_does_not_leak_in(
        self, builder_stub: _BuilderStub, two_week_schedule: pd.DataFrame
    ) -> None:
        """The restriction is load-bearing: the stub returns the WHOLE week 2 on purpose.

        ``2026_02_DET_BUF`` is a week-2 game whose lock is a DIFFERENT instant, so a wrapper
        that merely concatenated the week-scoped output would publish it under this run and
        restate a decision made on another day.
        """
        instant = _sunday_lock()
        candidates, schedule = build_freeze_instant_candidates(
            instant, decided_at=instant, schedule=two_week_schedule
        )
        assert "2026_02_DET_BUF" not in set(candidates["game_id"])
        assert "2026_02_DET_BUF" not in set(schedule["game_id"])
        assert "2026_02_CAR_ATL" in set(candidates["game_id"])

    def test_a_single_group_instant_matches_one_week_scoped_call(
        self, builder_stub: _BuilderStub, two_week_schedule: pd.DataFrame
    ) -> None:
        """The wrapper is a SUPERSET, not a behaviour change for the ordinary case."""
        instant = gameday_lock(_WEEK_2_THURSDAY_GAMEDAY)
        candidates, _schedule = build_freeze_instant_candidates(
            instant, decided_at=instant, schedule=two_week_schedule
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
        """An empty universe is refused rather than published as a run that recommended none."""
        instant = gameday_lock("2027-01-09")
        with pytest.raises(ValueError, match="no scheduled game"):
            build_freeze_instant_candidates(
                instant, decided_at=instant, schedule=two_week_schedule
            )
        assert builder_stub.calls == []

    def test_a_decision_after_the_lock_is_refused_before_any_build(
        self, builder_stub: _BuilderStub, two_week_schedule: pd.DataFrame
    ) -> None:
        from backtest.weekly_bet_list import LockPassedError

        instant = _sunday_lock()
        with pytest.raises(LockPassedError):
            build_freeze_instant_candidates(
                instant, decided_at=instant + _ONE_SECOND, schedule=two_week_schedule
            )
        assert builder_stub.calls == []


class TestTheWrapperHonoursTheExclusionSet:
    """D33.2-05: an excluded game yields no candidate; an all-excluded day is not an error."""

    def test_an_excluded_game_yields_no_candidate_and_is_passed_through(
        self, builder_stub: _BuilderStub, two_week_schedule: pd.DataFrame
    ) -> None:
        instant = _sunday_lock()
        excluded = frozenset({"2026_02_CAR_ATL"})
        candidates, schedule = build_freeze_instant_candidates(
            instant,
            decided_at=instant,
            schedule=two_week_schedule,
            excluded_game_ids=excluded,
        )

        assert "2026_02_CAR_ATL" not in set(candidates["game_id"])
        assert "2026_02_CAR_ATL" not in set(schedule["game_id"])
        assert {"2026_02_MIN_CHI", "2026_03_ATL_GB"} <= set(candidates["game_id"])
        assert builder_stub.excluded_seen, "the week-scoped builder was never called"
        assert all(seen == excluded for seen in builder_stub.excluded_seen), (
            "the exclusion set was not passed through to the week-scoped builder"
        )

    def test_an_excluded_game_past_its_lock_does_not_fail_the_run(
        self, builder_stub: _BuilderStub, two_week_schedule: pd.DataFrame
    ) -> None:
        """Excluding every game at the instant, decided after the lock: empty, no raise."""
        instant = _sunday_lock()
        at_instant = {"2026_02_CAR_ATL", "2026_02_MIN_CHI", "2026_03_ATL_GB"}
        candidates, schedule = build_freeze_instant_candidates(
            instant,
            decided_at=instant + _ONE_SECOND,
            schedule=two_week_schedule,
            excluded_game_ids=frozenset(at_instant),
        )
        assert candidates.empty
        assert schedule.empty
        assert builder_stub.calls == []

    def test_all_excluded_is_empty_while_unscheduled_still_refuses(
        self, builder_stub: _BuilderStub, two_week_schedule: pd.DataFrame
    ) -> None:
        """The two empties are different facts and must stay distinguishable (SPEC R3)."""
        instant = _sunday_lock()
        at_instant = {"2026_02_CAR_ATL", "2026_02_MIN_CHI", "2026_03_ATL_GB"}
        candidates, _schedule = build_freeze_instant_candidates(
            instant,
            decided_at=instant,
            schedule=two_week_schedule,
            excluded_game_ids=frozenset(at_instant),
        )
        assert candidates.empty

        unscheduled = gameday_lock("2027-01-09")
        with pytest.raises(ValueError, match="no scheduled game"):
            build_freeze_instant_candidates(
                unscheduled,
                decided_at=unscheduled,
                schedule=two_week_schedule,
                excluded_game_ids=frozenset(at_instant),
            )


# ---------------------------------------------------------------------------
# (f2) The two fences AGREE on the boundary.
# ---------------------------------------------------------------------------


class TestTheTwoFencesAgreeOnTheBoundary:
    """Selection and the write assertion use the SAME operator, so they compose cleanly."""

    def test_a_row_decided_exactly_at_its_lock_passes_both_fences(self) -> None:
        """At-lock is admissible at selection AND accepted at write (D33.2-01).

        Under the retired fence the two sat on opposite sides of the boundary (selection
        refused ``now >= freeze`` while the write accepted ``decided_at <= freeze``), so the
        equality case was unreachable in production. Both are ``<=`` now, and the equality
        case is the ordinary latest-possible decision. One test, deliberately: the disputed
        claim is about the conjunction, and splitting it would let each half pass alone.
        """
        gameday = "2026-09-24"
        lock = gameday_lock(gameday)

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
        selected = select_games_for_decision_instant(schedule, lock, decided_at=lock)
        assert set(selected["game_id"]) == {"2026_03_ATL_GB"}, (
            "selection refused a game decided exactly AT its lock; at-lock is admissible"
        )

        row = {
            "game_id": "2026_03_ATL_GB",
            "provenance": PROVENANCE_FORWARD,
            "freeze_ts": lock.isoformat(),
            DECIDED_AT_COLUMN: lock.isoformat(),
        }
        assert_decided_at_before_freeze(row)  # accepted: raises nothing

        decided = require_aware_snapshot_ts(row[DECIDED_AT_COLUMN])
        stored_lock = require_aware_snapshot_ts(row["freeze_ts"])
        assert decided == stored_lock

    def test_a_row_decided_one_second_late_fails_both_fences(self) -> None:
        from backtest.weekly_bet_list import DecidedAfterFreezeError, LockPassedError

        gameday = "2026-09-24"
        lock = gameday_lock(gameday)
        late = lock + _ONE_SECOND
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

        with pytest.raises(LockPassedError):
            select_games_for_decision_instant(schedule, lock, decided_at=late)
        with pytest.raises(DecidedAfterFreezeError):
            assert_decided_at_before_freeze(
                {
                    "game_id": "2026_03_ATL_GB",
                    "provenance": PROVENANCE_FORWARD,
                    "freeze_ts": lock.isoformat(),
                    DECIDED_AT_COLUMN: late.isoformat(),
                }
            )
