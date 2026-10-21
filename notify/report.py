"""What one daily run did, as the notifier reads it (Phase 35, RESEARCH Pattern 1).

WHY A REPORT
------------
Every outcome the owner must hear about already exists in the daily run and the forward ledger, but
almost none of it is returned as data: the run's own exit code, a printed ``RUN_STATUS=`` line, side
steps that print and record but return nothing. :class:`RunReport` is the one small, mutable record
the daily run fills as it goes (Plan 35-08 wires it), so the message builders in
:mod:`notify.messages` read stored facts and never re-derive them.

THE STATUS VOCABULARY
---------------------
The constants below are plain strings on purpose: ``notify`` must not import ``pipeline`` (the
daily run imports ``notify``, never the reverse). The three orchestrator strings equal
``pipeline.steps.RunStatus`` values and the orchestrator's bare ``"degraded"``; a test pins that.
The other four are the daily run's no-prediction outcomes (``logs/daily_lock_runs.jsonl``).

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import TYPE_CHECKING, Any

if (
    TYPE_CHECKING
):  # typing only: notify never imports the ledger or pipeline at load time
    from forward_ledger.closing_schedule import TriggerRegistration
    from forward_ledger.runner import SettleOutcome
    from forward_ledger.sync import SyncOutcome
    from pipeline.site_push import PushOutcome
    from refit.hook import RefitOutcome

__all__ = [
    "STATUS_DEGRADED",
    "STATUS_FAILED",
    "STATUS_FINISHED_WITH_SKIPS",
    "STATUS_LOCK_PASSED",
    "STATUS_NO_GAMES",
    "STATUS_OFFSEASON",
    "STATUS_RUN_FAILED",
    "STATUS_SUCCESS",
    "RunReport",
]

# The daily run's no-prediction outcomes (scripts/daily_lock_pipeline.py).
STATUS_LOCK_PASSED = "lock_passed"
STATUS_OFFSEASON = "offseason"
STATUS_NO_GAMES = "no_games"
STATUS_RUN_FAILED = "run_failed"
# The orchestrator's terminal statuses (pipeline.steps.RunStatus values, plus its bare "degraded").
STATUS_SUCCESS = "success"
STATUS_FAILED = "failed"
STATUS_FINISHED_WITH_SKIPS = "finished_with_skips"
STATUS_DEGRADED = "degraded"


@dataclass
class RunReport:
    """One daily run's outcome, filled by the run as it goes. Mutable on purpose.

    Attributes:
        run_date_et: The run's ET calendar date.
        start: The instant the run started.
        lock: The slate's lock (``pipeline.daily_steps.slate_lock(run_date_et)``), supplied by the
            daily run so ``notify`` never restates the lock rule.
        run_id: The ledger run id, or None (the lock-passed refusal runs before one is bound).
        status: One of the ``STATUS_*`` constants, or None while the run is in flight.
        tomorrow_game_ids: Tomorrow's games per the recorded schedule; None when unknown.
        unpredicted_game_ids: The games left without a prediction; None when unknown.
        skipped_game_ids: The games a live skip left out (``finished_with_skips``).
        failed_steps: The names of the steps that failed (``degraded``).
        exception_class: The class name of the exception that stopped the run, if any.
        exception_text: The first line of that exception's message. Held in memory only and
            never logged: it can carry a URL or a key until :func:`notify.messages.one_line_error`
            scrubs it.
        interrupted: True when the run was stopped by Ctrl+C (D-13: nothing is sent).
        slate_season: The selected slate's season, when a slate was selected.
        slate_week: The selected slate's week, when a slate was selected.
        settle_ran: True when the ledger settle pass was attempted.
        settle: The settle pass's ``SettleOutcome``, when it returned one.
        settle_error: The class name of the exception the settle pass raised, if any.
        triggers: The closing-trigger regeneration's ``TriggerRegistration``, when it ran.
        sync_ran: True when the end-of-run ledger sync was attempted.
        sync: The sync's ``SyncOutcome``; None when it returned none or raised.
        sync_error: The class name of the exception the sync raised, if any.
        appended_rows: The forward-ledger rows this run newly appended (Plan 35-04's
            ``RecordOutcome.appended_rows``), exactly as written: both arms' rows (the notifier
            keeps the live arm only).
        shadow_holes: ``((game_id, season, week, target), reason)`` for every new live row the
            frozen arm could not twin (Phase 37, SPEC R8), copied from what the recommend step
            recorded and never recomputed; the reason is an exception class name only.
        season_window: ``in``, ``out`` or ``unknown`` for the watcher's season window (Plan 35-13).
        cache_rebuild_attempted: True when this run tried to rebuild the web cache (the populate
            step on a prediction day, or the settle-day rebuild on a no-games day), so the website
            push can refuse a rebuild that did not finish (Phase 36, RESEARCH Pitfall 10).
        cache_rebuild_completed: True when that rebuild finished.
        site_push_ran: True when the end-of-run website push was attempted (Phase 36, D-13).
        site_push: The push's ``PushOutcome`` -- exactly what its push-log record says -- when the
            routine returned one.
        site_push_error: The class name of an exception that escaped the push routine, if any.
            No push-log record could be written for that attempt, so this is its only trace.
        refit_ran: True when the re-fit side step was called (Phase 37, D-15, D-17).
        refit: The side step's ``RefitOutcome``, exactly as it returned it.
        refit_error: The class name of an exception that escaped the side step, if any.
    """

    run_date_et: date
    start: datetime
    lock: datetime
    run_id: str | None = None
    status: str | None = None
    tomorrow_game_ids: tuple[str, ...] | None = None
    unpredicted_game_ids: tuple[str, ...] | None = None
    skipped_game_ids: tuple[str, ...] = ()
    failed_steps: tuple[str, ...] = ()
    exception_class: str | None = None
    exception_text: str | None = None
    interrupted: bool = False
    slate_season: int | None = None
    slate_week: int | None = None
    settle_ran: bool = False
    settle: SettleOutcome | None = None
    settle_error: str | None = None
    triggers: TriggerRegistration | None = None
    sync_ran: bool = False
    sync: SyncOutcome | None = None
    sync_error: str | None = None
    appended_rows: list[dict[str, Any]] = field(default_factory=list)
    shadow_holes: tuple[tuple[tuple[Any, ...], str], ...] = ()
    season_window: str = "unknown"
    cache_rebuild_attempted: bool = False
    cache_rebuild_completed: bool = False
    site_push_ran: bool = False
    site_push: PushOutcome | None = None
    site_push_error: str | None = None
    refit_ran: bool = False
    refit: RefitOutcome | None = None
    refit_error: str | None = None

    def record_exception(self, exc: BaseException) -> None:
        """Note the exception that stopped the run: its class, its first line, a Ctrl+C."""
        self.exception_class = type(exc).__name__
        lines = str(exc).splitlines()
        self.exception_text = lines[0] if lines and lines[0].strip() else None
        self.interrupted = isinstance(exc, KeyboardInterrupt)
