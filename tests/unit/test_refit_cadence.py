"""The re-fit trigger reads only the frozen lists, the clock and the records (RFIT-02, R2, R10).

The date lists are checked against their own evidence and, read-only, against silver; the trigger
is exercised over the A/A slide (D-02), the end rule (D-33), catch-ups, retries and idempotency.
No test here writes a file.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import inspect
from datetime import UTC, date, datetime, timedelta
from itertools import pairwise
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from backtest.refit_preregistration_2026 import (
    AA_EVALUATION_TUESDAY_BY_WEEK,
    CADENCE_SPACING_NFL_WEEKS,
    CADENCE_TUESDAY_BY_WEEK,
    FIRST_KICKOFF_EVIDENCE_UTC,
    LAST_NFL_WEEK,
    PLAYOFF_FIRST_GAME_DATES,
    REFIT_DATE_LISTS,
)
from refit.cadence import cadence_dates, cutoff_utc, decide, on_time
from utils.current_slate import default_schedule_path

_ET = ZoneInfo("America/New_York")
_TUESDAY = 1
NO_SILVER_SKIP_MESSAGE = "silver games.parquet is absent: the silver cross-check of the cadence Tuesdays cannot run"


def _utc(text: str) -> datetime:
    return datetime.fromisoformat(text).astimezone(UTC)


def _aa(week: int, clean: bool) -> dict[str, Any]:
    return {"season": 2026, "week": week, "clean": clean}


def _event(kind: str, date_index: int, day: str, **fields: Any) -> dict[str, Any]:
    return {
        "event": kind,
        "list_week": 8,
        "date_index": date_index,
        "date": day,
        "nfl_week": 9,
        **fields,
    }


def _last_tuesday_before(day: date) -> date:
    candidate = day - timedelta(days=1)
    while candidate.weekday() != _TUESDAY:
        candidate -= timedelta(days=1)
    return candidate


def test_every_list_is_well_formed() -> None:
    for list_week, entries in REFIT_DATE_LISTS.items():
        assert entries, f"list {list_week} is empty"
        assert entries[0] == (AA_EVALUATION_TUESDAY_BY_WEEK[list_week], list_week + 1)
        days = [date.fromisoformat(day) for day, _ in entries]
        assert days == sorted(days) and len(set(days)) == len(days)
        weeks = [nfl_week for _, nfl_week in entries]
        for day, nfl_week in entries:
            assert date.fromisoformat(day).weekday() == _TUESDAY
            assert day == CADENCE_TUESDAY_BY_WEEK[nfl_week]
        assert all(b - a == CADENCE_SPACING_NFL_WEEKS for a, b in pairwise(weeks))
        assert weeks[-1] <= LAST_NFL_WEEK
        assert weeks[-1] + CADENCE_SPACING_NFL_WEEKS > LAST_NFL_WEEK


def test_every_cutoff_is_17_et_and_22_utc() -> None:
    for list_week in REFIT_DATE_LISTS:
        for cadence_date in cadence_dates(list_week):
            local = cadence_date.cutoff_utc.astimezone(_ET)
            assert (local.date(), local.hour, local.minute) == (
                cadence_date.date,
                17,
                0,
            )
            assert cadence_date.cutoff_utc.utcoffset() == timedelta(0)
            assert (cadence_date.cutoff_utc.hour, cadence_date.cutoff_utc.minute) == (
                22,
                0,
            )
            assert cadence_date.cutoff_utc == cutoff_utc(cadence_date.date)


def test_the_tuesdays_precede_each_weeks_first_game() -> None:
    for week in range(9, 19):
        first_game_et = _utc(FIRST_KICKOFF_EVIDENCE_UTC[week]).astimezone(_ET).date()
        assert (
            CADENCE_TUESDAY_BY_WEEK[week]
            == _last_tuesday_before(first_game_et).isoformat()
        )
    for week in range(19, 23):
        first_game = date.fromisoformat(PLAYOFF_FIRST_GAME_DATES[week])
        assert (
            CADENCE_TUESDAY_BY_WEEK[week]
            == _last_tuesday_before(first_game).isoformat()
        )

    schedule_path = default_schedule_path()
    if not schedule_path.is_file():
        pytest.skip(NO_SILVER_SKIP_MESSAGE)
    games = pd.read_parquet(schedule_path, columns=["season", "week", "kickoff_et"])
    season = games[(games["season"] == 2026) & games["week"].between(9, 18)]
    assert not season.empty
    for week, kickoff in season.groupby("week")["kickoff_et"].min().items():
        first_game_et = pd.Timestamp(kickoff).tz_convert(_ET).date()
        assert (
            CADENCE_TUESDAY_BY_WEEK[int(week)]
            == _last_tuesday_before(first_game_et).isoformat()
        )


def test_waiting_until_week_8_is_evaluable() -> None:
    before = decide(_utc("2026-11-03T21:59:59+00:00"), [], [])
    assert before.due is None
    assert before.aa_week_due is None
    assert before.waiting_reason is not None

    at = decide(_utc("2026-11-03T22:00:00+00:00"), [], [])
    assert at.aa_week_due == 8
    assert at.due is None


def test_clean_week_8_makes_nov_3_due() -> None:
    decision = decide(_utc("2026-11-03T22:05:00+00:00"), [_aa(8, True)], [])
    assert decision.in_force_list == 8
    assert decision.due is not None
    assert (decision.due.index, decision.due.date, decision.due.nfl_week) == (
        1,
        date(2026, 11, 3),
        9,
    )
    assert decision.due.cutoff_utc == _utc("2026-11-03T22:00:00+00:00")
    assert decision.close_not_run == ()


def test_unclean_week_8_slides_to_week_9() -> None:
    now = _utc("2026-11-10T22:05:00+00:00")
    pending = decide(now, [_aa(8, False)], [])
    assert pending.aa_week_due == 9
    assert pending.due is None

    chosen = decide(now, [_aa(8, False), _aa(9, True)], [])
    assert chosen.in_force_list == 9
    assert chosen.due is not None
    assert chosen.due.date == date(2026, 11, 10)


def test_no_clean_week_is_the_end_rule() -> None:
    records = [_aa(week, False) for week in range(8, 17)]
    for now in ("2026-12-29T22:05:00+00:00", "2027-02-10T00:00:00+00:00"):
        decision = decide(_utc(now), records, [])
        assert decision.no_list_ever is True
        assert decision.due is None
        assert decision.aa_week_due is None


def test_zero_new_games_still_due() -> None:
    assert list(inspect.signature(decide).parameters) == [
        "now_utc",
        "aa_records",
        "date_events",
    ]
    decision = decide(_utc("2026-11-03T22:00:00+00:00"), [_aa(8, True)], [])
    assert decision.due is not None
    assert decision.due.date == date(2026, 11, 3)


def test_only_the_latest_passed_date_is_due() -> None:
    decision = decide(_utc("2026-12-01T22:05:00+00:00"), [_aa(8, True)], [])
    assert decision.due is not None
    assert decision.due.date == date(2026, 12, 1)
    assert tuple(d.date for d in decision.close_not_run) == (date(2026, 11, 3),)


def test_no_run_between_dates_closes_the_older() -> None:
    decision = decide(_utc("2026-12-02T15:00:00+00:00"), [_aa(8, True)], [])
    assert decision.due is not None
    assert (decision.due.index, decision.due.date) == (2, date(2026, 12, 1))
    assert [(d.index, d.date) for d in decision.close_not_run] == [
        (1, date(2026, 11, 3))
    ]


def test_a_retry_keeps_the_cutoff() -> None:
    events = [_event("attempt", 1, "2026-11-03", outcome="refused")]
    for now in ("2026-11-04T22:05:00+00:00", "2026-11-05T22:05:00+00:00"):
        decision = decide(_utc(now), [_aa(8, True)], events)
        assert decision.due is not None
        assert decision.due.index == 1
        assert decision.due.cutoff_utc == _utc("2026-11-03T22:00:00+00:00")


def test_a_success_is_never_due_again() -> None:
    ran = [_event("closed", 1, "2026-11-03", status="ran")]
    decision = decide(_utc("2026-11-04T22:05:00+00:00"), [_aa(8, True)], ran)
    assert decision.due is None
    assert decision.close_not_run == ()

    not_run = [_event("closed", 1, "2026-11-03", status="not run")]
    decision = decide(_utc("2026-11-30T22:05:00+00:00"), [_aa(8, True)], not_run)
    assert decision.due is None
    assert decision.close_not_run == ()


def test_the_list_stays_fixed_once_used() -> None:
    events = [_event("attempt", 1, "2026-11-03", outcome="refused")]
    records = [_aa(8, False), _aa(9, True)]
    decision = decide(_utc("2026-11-10T22:05:00+00:00"), records, events)
    assert decision.in_force_list == 8
    assert decision.due is not None
    assert decision.due.date == date(2026, 11, 3)


def test_on_time_means_the_dates_own_et_day() -> None:
    first = cadence_dates(8)[0]
    assert on_time(first, date(2026, 11, 3)) is True
    assert on_time(first, date(2026, 11, 4)) is False
