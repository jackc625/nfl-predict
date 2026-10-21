"""The re-fit trigger: due, waiting or closed-as-not-run, from the calendar alone (RFIT-02, R2, R10).

WHAT IT READS, AND WHAT IT NEVER READS
--------------------------------------
:func:`decide` is a pure function of the pre-registered date lists
(``backtest.refit_preregistration_2026``), the run clock and the two record files (the A/A weeks
and the cadence-date events of ``refit.records``). It reads no game, metric, drift, CLV or
comparison value, so neither performance nor drift can ever move a re-fit date (must-NOT P1,
scanned by Plan 37-15). A cadence date with zero newly finished games is still due.

THE RULES (D-01, D-02, D-05, D-06, D-33)
----------------------------------------
* Once any cadence-date event exists, its list stays in force.
* Otherwise the A/A weeks 8..16 are scanned in order; the first record of a week is binding. A
  clean week selects its literal list; an unclean week passes to the next; a week with no record
  is due for evaluation once its evaluation Tuesday's cutoff has passed, and is waited for before.
  All of 8..16 unclean means no re-fit in 2026 (the end rule).
* With a list in force, the latest date whose cutoff has passed is due when it is still open, and
  every older open passed date is closed as ``not run``: at most one re-fit per daily run. A retry
  or a catch-up keeps the date's own cutoff; a closed date is never due again.
* Every cutoff is 17:00 America/New_York on its Tuesday, through the zone and never a fixed
  offset (DST ends Sun 2026-11-01, so every listed cutoff is 22:00 UTC).

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, time
from typing import Any
from zoneinfo import ZoneInfo

from backtest.refit_preregistration_2026 import (
    AA_EVALUATION_TUESDAY_BY_WEEK,
    AA_FIRST_WEEK,
    AA_LAST_WEEK,
    CUTOFF_LOCAL_TIME,
    CUTOFF_TIMEZONE,
    REFIT_DATE_LISTS,
    REFIT_SEASON,
)

__all__ = [
    "CadenceDate",
    "DateStatus",
    "TriggerDecision",
    "aa_evaluation_instant",
    "cadence_dates",
    "cutoff_utc",
    "date_statuses",
    "decide",
    "on_time",
]

_CUTOFF_ZONE = ZoneInfo(CUTOFF_TIMEZONE)
_OPEN = "open"


@dataclass(frozen=True)
class CadenceDate:
    """One pre-registered re-fit date of one list.

    Attributes:
        list_week: The first clean A/A week that selects this list.
        index: The date's 1-based position in its list.
        date: The cadence Tuesday.
        nfl_week: The NFL week whose first game the Tuesday precedes.
        cutoff_utc: 17:00 America/New_York on that Tuesday, in UTC.
    """

    list_week: int
    index: int
    date: date
    nfl_week: int
    cutoff_utc: datetime


@dataclass(frozen=True)
class DateStatus:
    """Where one cadence date stands in the records.

    Attributes:
        date: The cadence date.
        status: ``open``, or the status of its ``closed`` event (``ran``, ``ran late``,
            ``not run``).
        last_outcome: The outcome of its latest ``attempt``, or None before any attempt.
        attempts: How many attempts it has.
    """

    date: CadenceDate
    status: str
    last_outcome: str | None
    attempts: int


@dataclass(frozen=True)
class TriggerDecision:
    """What one daily run must do about re-fitting.

    Attributes:
        aa_week_due: The A/A week due for evaluation now, before any list is chosen.
        in_force_list: The list in force, or None before one is chosen.
        waiting_reason: Why nothing can be decided yet, when that is the case.
        no_list_ever: Every A/A week 8..16 was unclean: no re-fit runs in 2026 (D-33).
        close_not_run: Older passed dates still open, to close as ``not run``.
        due: The one date to attempt now, or None.
    """

    aa_week_due: int | None
    in_force_list: int | None
    waiting_reason: str | None
    no_list_ever: bool
    close_not_run: tuple[CadenceDate, ...]
    due: CadenceDate | None


def cutoff_utc(day: date) -> datetime:
    """17:00 America/New_York on *day*, as a UTC instant (D-05)."""
    hour, minute = (int(part) for part in CUTOFF_LOCAL_TIME.split(":"))
    local = datetime.combine(day, time(hour, minute), tzinfo=_CUTOFF_ZONE)
    return local.astimezone(UTC)


def cadence_dates(list_week: int) -> tuple[CadenceDate, ...]:
    """The literal dates of the list selected by the first clean A/A week *list_week*."""
    return tuple(
        CadenceDate(
            list_week=list_week,
            index=index,
            date=date.fromisoformat(day),
            nfl_week=nfl_week,
            cutoff_utc=cutoff_utc(date.fromisoformat(day)),
        )
        for index, (day, nfl_week) in enumerate(REFIT_DATE_LISTS[list_week], 1)
    )


def aa_evaluation_instant(week: int) -> datetime:
    """The instant A/A week *week* becomes evaluable: its evaluation Tuesday's cutoff (R7)."""
    return cutoff_utc(date.fromisoformat(AA_EVALUATION_TUESDAY_BY_WEEK[week]))


def date_statuses(
    dates: Sequence[CadenceDate], date_events: Sequence[Mapping[str, Any]]
) -> tuple[DateStatus, ...]:
    """Each date's status from its events: a ``closed`` event decides it; otherwise ``open``."""
    statuses: list[DateStatus] = []
    for cadence_date in dates:
        own = [
            event
            for event in date_events
            if event["list_week"] == cadence_date.list_week
            and event["date_index"] == cadence_date.index
        ]
        attempts = [event for event in own if event["event"] == "attempt"]
        closed = [event for event in own if event["event"] == "closed"]
        statuses.append(
            DateStatus(
                date=cadence_date,
                status=closed[0]["status"] if closed else _OPEN,
                last_outcome=attempts[-1]["outcome"] if attempts else None,
                attempts=len(attempts),
            )
        )
    return tuple(statuses)


def _decision(**fields: Any) -> TriggerDecision:
    defaults: dict[str, Any] = {
        "aa_week_due": None,
        "in_force_list": None,
        "waiting_reason": None,
        "no_list_ever": False,
        "close_not_run": (),
        "due": None,
    }
    return TriggerDecision(**{**defaults, **fields})


def decide(
    now_utc: datetime,
    aa_records: Sequence[Mapping[str, Any]],
    date_events: Sequence[Mapping[str, Any]],
) -> TriggerDecision:
    """The trigger: from the frozen lists, the clock and the records alone (see the module doc).

    Raises:
        ValueError: *now_utc* is naive.
    """
    if now_utc.tzinfo is None:
        msg = f"decide needs an aware instant, got the naive {now_utc!r}"
        raise ValueError(msg)

    if date_events:
        list_week = int(date_events[0]["list_week"])
    else:
        first_by_week: dict[int, Mapping[str, Any]] = {}
        for record in aa_records:
            if record["season"] == REFIT_SEASON:
                first_by_week.setdefault(int(record["week"]), record)
        chosen: int | None = None
        for week in range(AA_FIRST_WEEK, AA_LAST_WEEK + 1):
            record = first_by_week.get(week)
            if record is None:
                evaluable_at = aa_evaluation_instant(week)
                if now_utc >= evaluable_at:
                    return _decision(aa_week_due=week)
                reason = f"A/A week {week} is evaluable at {evaluable_at.isoformat()}"
                return _decision(waiting_reason=reason)
            if record["clean"]:
                chosen = week
                break
        if chosen is None:
            return _decision(no_list_ever=True)
        list_week = chosen

    statuses = date_statuses(cadence_dates(list_week), date_events)
    passed = [status for status in statuses if now_utc >= status.date.cutoff_utc]
    if not passed:
        return _decision(in_force_list=list_week)
    latest = passed[-1]
    close_not_run = tuple(
        status.date for status in passed[:-1] if status.status == _OPEN
    )
    due = latest.date if latest.status == _OPEN else None
    return _decision(in_force_list=list_week, close_not_run=close_not_run, due=due)


def on_time(cadence_date: CadenceDate, run_date_et: date) -> bool:
    """True when a run on the ET day *run_date_et* is on the date's own ET day."""
    return run_date_et == cadence_date.date
