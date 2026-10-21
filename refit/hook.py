"""The daily run's re-fit side step (Phase 37; D-15, D-17, D-37, SPEC R9, R10, RFIT-02, RFIT-06).

WHAT ONE CALL DOES
------------------
:func:`run_refit_side_step` is called once per daily run, after the day's ledger write and before
the cache rebuild, the website push and the notifications (``scripts/daily_lock_pipeline.py``). Under
the attempt lock it, in order:

1. closes a date whose install a crash interrupted after the swap (``refit.swap.pending_install``),
   without re-training (research Pitfall 2);
2. before any date list is in force, evaluates every A/A week whose evaluation instant has passed
   and has no record, in order, recording each, until the trigger stops asking (D-02);
3. closes every older open date the trigger names ``not run``;
4. attempts the one due date (:func:`attempt_date`): the code commit is captured once, the recipe
   fingerprint is checked first (P4), then the A/A proof, the schedule refresh on a run that made
   none, the D-37 build, training, the re-measure, the integrity checks and the swap.

THE TRIGGER IS CALENDAR-ONLY
----------------------------
Whether anything is due comes from ``refit.cadence.decide``: the frozen date lists, the clock and the
records. Nothing here reads a metric. A day with nothing to do takes no lock and writes no file.

EVERY OUTCOME IS A RECORD; A REFUSAL IS A RESULT
------------------------------------------------
Every attempt appends one ``attempt`` event (installed, refused with its check, or crashed with its
exception CLASS only), and every resolved date one ``closed`` event (``ran``, ``ran late``,
``not run``). A refused or crashed date stays open and is retried at the next daily run with the
same cutoff, holding the first attempt's no-final-score games (R10). The swap is gate-free: no deploy
gate is called (RFIT-06). Exception text never reaches a record (D-10).

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import contextlib
import dataclasses
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd

from backtest.neutral_hfa_cold_start_constants import EDGE_TIER_THRESHOLDS_BY_TARGET
from backtest.refit_preregistration_2026 import (
    MARKET_CONVERTER_ID,
    REFIT_RECIPE_ID,
    REFIT_SEASON,
    THREAD_LIMIT,
)
from forward_ledger.settle import load_silver_games
from forward_ledger.store import LEDGER_DIR, read_entries
from models.artifacts import ResolvedArtifacts, resolve_production_artifacts
from models.market_probability import load_market_probability_artifact
from refit import integrity, records, witness
from refit.aa import AAWeekResult, evaluate_week, record_week
from refit.cadence import CadenceDate, TriggerDecision, cadence_dates, decide, on_time
from refit.data_prep import prepare_gold
from refit.generations import (
    Generation,
    in_force_edge_thresholds,
    installed_generations,
)
from refit.integrity import IntegrityRefusal, run_integrity_checks
from refit.remeasure import (
    BetRuleNumbers,
    RemeasureRefusal,
    bet_rule_payload,
    extended_corpus,
    extended_slopes,
    measure_bet_rule,
    tune_generation_blend,
    walk_forward_predictions,
)
from refit.swap import install_generation, pending_install
from refit.training import (
    GenerationTraining,
    RefitTrainingError,
    _hold_no_score,
    train_generation,
)
from utils.file_lock import FileLockHeldError
from utils.logging_config import get_logger

if TYPE_CHECKING:
    from pipeline.daily_steps import DailySlate

logger = get_logger(__name__)

__all__ = [
    "REFIT_STATUSES",
    "RefitOutcome",
    "RefitSteps",
    "Remeasured",
    "attempt_date",
    "remeasure_generation",
    "run_refit_side_step",
]

#: What one side step reports (``REFIT= <status>``).
REFIT_STATUSES: tuple[str, ...] = (
    "waiting",
    "not_due",
    "no_refit_this_season",
    "ran",
    "ran late",
    "refused",
    "crashed",
    "busy",
)

_TARGETS: tuple[str, ...] = ("wp", "ats", "ou")
_RAN: str = "ran"
_RAN_LATE: str = "ran late"
_NOT_RUN: str = "not run"
_COUNTING_CLOSES: frozenset[str] = frozenset({_RAN, _RAN_LATE})


@dataclass(frozen=True)
class RefitOutcome:
    """What one side step did, built from everything the run produced.

    Attributes:
        status: One of :data:`REFIT_STATUSES`.
        list_week: The date list in force (its first clean A/A week), when one is.
        date_index: The attempted (or recovered) date's 1-based index in its list.
        date: That cadence date, ISO.
        nfl_week: The NFL week whose first game that date precedes.
        next_date: The following date of the list, ISO, or None when it is the last.
        check: The pre-registered check a refused attempt named.
        exception_class: The class name of a crashed attempt's exception.
        aa_results: EVERY A/A week evaluated in this run, in order (a catch-up run's earlier
            mismatch is never overwritten by a later clean week).
        recovered_close: ``(date ISO, "ran" | "ran late")`` when this run closed a date whose
            swap a crash interrupted.
        closed_not_run: The dates this run closed ``not run``, ISO.
        wrote_records: True when the run appended any record.
        after_ids: The five ids serving after an install or a recovered close.
    """

    status: str
    list_week: int | None = None
    date_index: int | None = None
    date: str | None = None
    nfl_week: int | None = None
    next_date: str | None = None
    check: str | None = None
    exception_class: str | None = None
    aa_results: tuple[AAWeekResult, ...] = ()
    recovered_close: tuple[str, str] | None = None
    closed_not_run: tuple[str, ...] = ()
    wrote_records: bool = False
    after_ids: dict[str, str | None] | None = None


@dataclass(frozen=True)
class Remeasured:
    """The re-measure step's answer: the generation's five ids and its bet-rule record's sha256.

    Attributes:
        ids: wp, ats, ou, blend and converter of the new generation.
        bet_rule_sha256: The sha256 naming the generation's bet-rule record (never a path: the
            one join of a ledger dir and a stored bet-rule path is ``refit.records``').
        numbers: The re-measured numbers, which ``remeasure_complete`` checks.
    """

    ids: dict[str, str]
    bet_rule_sha256: str
    numbers: BetRuleNumbers


def _current_thresholds(
    artifacts_dir: Path, ledger_dir: Path
) -> dict[str, tuple[float, float]]:
    """The edge-tier pairs in force: the in-force generation's, else the row-19 constants."""
    in_force = in_force_edge_thresholds(artifacts_dir, ledger_dir)
    source = EDGE_TIER_THRESHOLDS_BY_TARGET if in_force is None else in_force
    return {target: pair for target, pair in source.items() if pair is not None}


def remeasure_generation(
    training: GenerationTraining,
    *,
    games: pd.DataFrame,
    date: CadenceDate,
    gold_dir: Path,
    silver_dir: Path,
    artifacts_dir: Path,
    ledger_dir: Path,
) -> Remeasured:
    """Re-measure the generation's bet rule and blend, and write its bet-rule record (record 9).

    Walk-forward predictions -> the extended corpus -> the bet-rule numbers -> the generation's
    blend -> the record payload -> the content-addressed record. *games* is the frame the
    training read, with any held no-score game already masked (R10).
    """
    cutoff = date.cutoff_utc
    predictions = walk_forward_predictions(gold_dir, games, cutoff)
    corpus = extended_corpus(silver_dir, games, cutoff)
    slopes = extended_slopes(
        load_market_probability_artifact(MARKET_CONVERTER_ID, artifacts_dir)
    )
    numbers = measure_bet_rule(
        predictions,
        corpus,
        slopes,
        current_thresholds=_current_thresholds(artifacts_dir, ledger_dir),
    )
    blend = tune_generation_blend(
        artifacts_dir,
        silver_dir,
        gold_dir,
        artifact_ids=training.artifact_ids,
        predictions=predictions,
        corpus=corpus,
        slopes=slopes,
        gold_generation=training.gold_generation_digest,
    )
    ids = {
        **{target: training.artifact_ids[target] for target in _TARGETS},
        "blend": blend.artifact_dir.name,
        "converter": blend.converter_artifact_id,
    }
    weights = blend.result.weights
    payload = bet_rule_payload(
        numbers,
        artifact_ids=ids,
        cutoff_utc=cutoff,
        list_week=date.list_week,
        date_index=date.index,
        blend_weights={
            "wp": weights.wp_model_weight,
            "ats": weights.ats_model_weight,
            "ou": weights.ou_model_weight,
        },
    )
    _path, sha = records.write_bet_rule_record(ledger_dir, payload)
    return Remeasured(ids=ids, bet_rule_sha256=sha, numbers=numbers)


@dataclass(frozen=True)
class RefitSteps:
    """The heavy steps of one attempt: the production functions, the seam the tests inject.

    Attributes:
        prepare: The D-37 nightly build, ``(run_date_et, *, slate)``.
        train: Cutoff-scoped training (``refit.training.train_generation``'s signature).
        remeasure: The re-measure, returning :class:`Remeasured` (:func:`remeasure_generation`).
        checks: The pre-swap integrity checks (``refit.integrity.run_integrity_checks``).
        install: The one swap surface (``refit.swap.install_generation``).
    """

    prepare: Callable[..., None] = prepare_gold
    train: Callable[..., GenerationTraining] = train_generation
    remeasure: Callable[..., Remeasured] = remeasure_generation
    checks: Callable[..., None] = run_integrity_checks
    install: Callable[..., tuple[dict[str, str | None], dict[str, str | None]]] = (
        install_generation
    )


# ---------------------------------------------------------------------------
# Small readers and event builders
# ---------------------------------------------------------------------------


def _served_ids(artifacts_dir: Path) -> dict[str, str | None]:
    """The five ids ``latest.json`` serves now."""
    return dataclasses.asdict(resolve_production_artifacts(artifacts_dir))


def _event(kind: str, cadence_date: CadenceDate) -> dict[str, Any]:
    """The fields every cadence-date event of *cadence_date* carries."""
    return {
        "event": kind,
        "list_week": cadence_date.list_week,
        "date_index": cadence_date.index,
        "date": cadence_date.date.isoformat(),
        "nfl_week": cadence_date.nfl_week,
    }


def _closed_event(
    cadence_date: CadenceDate,
    status: str,
    *,
    closed_at: datetime,
    before: Mapping[str, str | None] | None,
    after: Mapping[str, str | None] | None,
) -> dict[str, Any]:
    return {
        **_event("closed", cadence_date),
        "status": status,
        "closed_at_utc": closed_at.isoformat(),
        "before_ids": None if before is None else dict(before),
        "after_ids": None if after is None else dict(after),
    }


def _own_events(
    events: Sequence[Mapping[str, Any]], cadence_date: CadenceDate
) -> list[Mapping[str, Any]]:
    return [
        event
        for event in events
        if event["list_week"] == cadence_date.list_week
        and event["date_index"] == cadence_date.index
    ]


def _held_no_score_ids(
    events: Sequence[Mapping[str, Any]], cadence_date: CadenceDate
) -> frozenset[str]:
    """The no-final-score games of the date's EARLIEST attempt that recorded them (R10)."""
    for event in _own_events(events, cadence_date):
        if event["event"] == "attempt" and event.get("no_final_score_ids") is not None:
            return frozenset(str(game_id) for game_id in event["no_final_score_ids"])
    return frozenset()


def _next_date(cadence_date: CadenceDate) -> str | None:
    """The date after *cadence_date* in its list, ISO, or None when it is the last."""
    dates = cadence_dates(cadence_date.list_week)
    if cadence_date.index < len(dates):
        return dates[cadence_date.index].date.isoformat()
    return None


def _first_date_after(list_week: int, now_utc: datetime) -> str | None:
    """The first date of the list whose cutoff is still ahead, ISO."""
    for cadence_date in cadence_dates(list_week):
        if cadence_date.cutoff_utc > now_utc:
            return cadence_date.date.isoformat()
    return None


def _date_fields(cadence_date: CadenceDate) -> dict[str, Any]:
    return {
        "list_week": cadence_date.list_week,
        "date_index": cadence_date.index,
        "date": cadence_date.date.isoformat(),
        "nfl_week": cadence_date.nfl_week,
        "next_date": _next_date(cadence_date),
    }


# ---------------------------------------------------------------------------
# One attempt
# ---------------------------------------------------------------------------


def attempt_date(
    date: CadenceDate,
    *,
    run_date_et: date,
    now_utc: datetime,
    slate: DailySlate | None,
    schedule_refreshed: bool,
    refresh_schedule: Callable[[], object] | None,
    first_refit: bool,
    ledger_dir: Path,
    artifacts_dir: Path,
    gold_dir: Path,
    silver_dir: Path,
    steps: RefitSteps,
) -> RefitOutcome:
    """Attempt the due cadence *date* once and record the attempt (and its close on success).

    The order is the record's: the code commit captured ONCE (written on the install intent and
    on the attempt alike), the recipe fingerprint first (P4: nothing is built or trained from an
    edited record), the A/A proof, the schedule refresh when this run made none (D-37 on a
    ``lock_passed`` run), the nightly build, training holding the date's first-attempt
    no-final-score games, the re-measure on the same masked games, the pre-swap checks and the
    swap. ``IntegrityRefusal``, ``RefitTrainingError`` and ``RemeasureRefusal`` are refusals,
    recorded with their check; any other ``Exception`` is a crash, recorded by class only.

    Args:
        date: The due cadence date.
        run_date_et: The daily run's ET calendar day (on time when it is the date's own day).
        now_utc: The attempt instant (tz-aware).
        slate: The day's slate on a prediction day, else None.
        schedule_refreshed: Whether this run already refreshed the schedule.
        refresh_schedule: Refreshes it when it has not been (the ``lock_passed`` run).
        first_refit: True until a date of the list has closed ``ran`` / ``ran late``.
        ledger_dir, artifacts_dir, gold_dir, silver_dir: The stores.
        steps: The heavy steps.
    """
    started = time.monotonic()
    ledger = Path(ledger_dir)
    artifacts = Path(artifacts_dir)
    commit = records.code_commit(witness.REPO_ROOT)
    events = records.read_date_events(ledger)
    held = _held_no_score_ids(events, date)
    slate_ids = slate.game_ids if slate is not None else frozenset()

    before: dict[str, str | None] | None = None
    after: dict[str, str | None] | None = None
    no_final_score_ids: list[str] | None = None
    outcome = "installed"
    check: str | None = None
    exception_class: str | None = None
    try:
        before = _served_ids(artifacts)
        integrity.check_recipe_fingerprint(witness.REPO_ROOT)
        aa_records = records.read_aa_records(ledger)
        integrity.check_aa_proof(date.list_week, aa_records, first_refit=first_refit)
        if not schedule_refreshed and refresh_schedule is not None:
            refresh_schedule()
        steps.prepare(run_date_et, slate=slate)
        training = steps.train(
            date.cutoff_utc,
            gold_dir=gold_dir,
            silver_dir=silver_dir,
            artifacts_dir=artifacts,
            thread_limit=THREAD_LIMIT,
            held_no_score_ids=held,
        )
        no_final_score_ids = sorted(training.no_final_score_ids)
        # The training's own in-memory mask (R10), so the re-measure and the checks read the
        # same games the training read.
        games = _hold_no_score(load_silver_games(silver_dir), held)
        remeasured = steps.remeasure(
            training,
            games=games,
            date=date,
            gold_dir=gold_dir,
            silver_dir=silver_dir,
            artifacts_dir=artifacts,
            ledger_dir=ledger,
        )
        generation = Generation(
            number=len(installed_generations(ledger)),
            ids=ResolvedArtifacts(**remeasured.ids),
            recipe=REFIT_RECIPE_ID,
            bet_rule_path=records.bet_rule_record_path(
                ledger, remeasured.bet_rule_sha256
            ),
            date_index=date.index,
            list_week=date.list_week,
        )
        steps.checks(
            repo_root=witness.REPO_ROOT,
            list_week=date.list_week,
            aa_records=aa_records,
            first_refit=first_refit,
            training=training,
            games=games,
            cutoff_utc=date.cutoff_utc,
            numbers=remeasured.numbers,
            generation=generation,
            season=REFIT_SEASON,
            finished_week=date.nfl_week - 1,
            slate_game_ids=slate_ids,
            artifacts_dir=artifacts,
            gold_dir=gold_dir,
            silver_dir=silver_dir,
            ledger_dir=ledger,
        )
        before, after = steps.install(
            ledger,
            artifacts,
            date=date,
            ids=remeasured.ids,
            bet_rule_sha256=remeasured.bet_rule_sha256,
            code_commit=commit,
        )
    except (IntegrityRefusal, RefitTrainingError, RemeasureRefusal) as refusal:
        outcome, check = "refused", refusal.check
    except Exception as exc:  # noqa: BLE001 - a crash is a recorded result, by class only
        outcome, exception_class = "crashed", type(exc).__name__

    records.append_date_event(
        ledger,
        {
            **_event("attempt", date),
            "cutoff_utc": date.cutoff_utc.isoformat(),
            "attempted_at_utc": now_utc.isoformat(),
            "run_date_et": run_date_et.isoformat(),
            "code_commit": commit,
            "outcome": outcome,
            "check": check,
            "exception_class": exception_class,
            "before_ids": before,
            "after_ids": after,
            "no_final_score_ids": no_final_score_ids,
        },
    )
    if outcome != "installed":
        logger.warning(
            "Re-fit attempt did not install",
            date=date.date.isoformat(),
            outcome=outcome,
            check=check,
            exception_class=exception_class,
        )
        return RefitOutcome(
            status=outcome,
            check=check,
            exception_class=exception_class,
            wrote_records=True,
            **_date_fields(date),
        )

    status = _RAN if on_time(date, run_date_et) else _RAN_LATE
    closed_at = now_utc + timedelta(seconds=time.monotonic() - started)
    records.append_date_event(
        ledger,
        _closed_event(date, status, closed_at=closed_at, before=before, after=after),
    )
    logger.info("Re-fit installed", date=date.date.isoformat(), status=status)
    return RefitOutcome(
        status=status, wrote_records=True, after_ids=after, **_date_fields(date)
    )


# ---------------------------------------------------------------------------
# The side step
# ---------------------------------------------------------------------------


def _decide(ledger: Path, now_utc: datetime) -> TriggerDecision:
    return decide(
        now_utc, records.read_aa_records(ledger), records.read_date_events(ledger)
    )


def _has_work(ledger: Path, artifacts: Path, now_utc: datetime) -> bool:
    """Whether this run has anything to record; read-only, so a quiet day takes no lock."""
    decision = _decide(ledger, now_utc)
    if (
        decision.aa_week_due is not None
        or decision.due is not None
        or decision.close_not_run
    ):
        return True
    return pending_install(ledger, artifacts) is not None


def _idle_status(decision: TriggerDecision) -> str:
    if decision.no_list_ever:
        return "no_refit_this_season"
    if decision.in_force_list is None:
        return "waiting"
    return "not_due"


def _recover_pending_install(
    ledger: Path, artifacts: Path, run_date_et: date, now_utc: datetime
) -> tuple[CadenceDate, str, dict[str, str | None]] | None:
    """Close the date whose last intent ``latest.json`` serves, with no re-training (Pitfall 2)."""
    index = pending_install(ledger, artifacts)
    if index is None:
        return None
    events = records.read_date_events(ledger)
    list_week = int(events[0]["list_week"])
    cadence_date = cadence_dates(list_week)[index - 1]
    intents = [
        event
        for event in _own_events(events, cadence_date)
        if event["event"] == "install_intent"
    ]
    served: dict[str, str | None] = dict(intents[-1]["ids"])
    status = _RAN if on_time(cadence_date, run_date_et) else _RAN_LATE
    records.append_date_event(
        ledger,
        _closed_event(
            cadence_date, status, closed_at=now_utc, before=None, after=served
        ),
    )
    logger.info(
        "Re-fit install recovered", date=cadence_date.date.isoformat(), status=status
    )
    return cadence_date, status, served


def _evaluate_due_aa_weeks(
    ledger: Path, silver_dir: Path, now_utc: datetime
) -> tuple[AAWeekResult, ...]:
    """Evaluate and record every A/A week the trigger asks for, in order (D-02)."""
    results: list[AAWeekResult] = []
    aa_records = records.read_aa_records(ledger)
    events = records.read_date_events(ledger)
    decision = decide(now_utc, aa_records, events)
    if decision.aa_week_due is None:
        return ()
    games = load_silver_games(silver_dir)
    entries = read_entries(ledger)
    commit = records.code_commit(witness.REPO_ROOT)
    while decision.aa_week_due is not None:
        result = evaluate_week(
            entries,
            games,
            REFIT_SEASON,
            decision.aa_week_due,
            committed_at=witness.committed_at(),
        )
        record_week(ledger, result, code_commit=commit)
        results.append(result)
        logger.info(
            "A/A week evaluated",
            week=result.week,
            clean=result.clean,
            reason=result.reason,
        )
        # The trigger reads only (season, week, clean): this run's record, held in memory, moves
        # it on even under a sink that persists nothing.
        aa_records = [
            *aa_records,
            {"season": result.season, "week": result.week, "clean": result.clean},
        ]
        decision = decide(now_utc, aa_records, events)
    return tuple(results)


def _first_refit(events: Sequence[Mapping[str, Any]]) -> bool:
    """True until a date has closed ``ran`` / ``ran late`` (the A/A proof binds only the first)."""
    return not any(
        event["event"] == "closed" and event["status"] in _COUNTING_CLOSES
        for event in events
    )


def _run_locked(
    *,
    run_date_et: date,
    now_utc: datetime,
    slate: DailySlate | None,
    schedule_refreshed: bool,
    refresh_schedule: Callable[[], object] | None,
    ledger: Path,
    artifacts: Path,
    gold_dir: Path,
    silver_dir: Path,
    steps: RefitSteps,
) -> RefitOutcome:
    """Everything the side step does under the attempt lock (see the module doc)."""
    recovered = _recover_pending_install(ledger, artifacts, run_date_et, now_utc)
    aa_results = _evaluate_due_aa_weeks(ledger, silver_dir, now_utc)

    decision = _decide(ledger, now_utc)
    for cadence_date in decision.close_not_run:
        records.append_date_event(
            ledger,
            _closed_event(
                cadence_date, _NOT_RUN, closed_at=now_utc, before=None, after=None
            ),
        )
    closed_not_run = tuple(d.date.isoformat() for d in decision.close_not_run)
    wrote = recovered is not None or bool(aa_results) or bool(closed_not_run)
    shared: dict[str, Any] = {
        "aa_results": aa_results,
        "recovered_close": None
        if recovered is None
        else (recovered[0].date.isoformat(), recovered[1]),
        "closed_not_run": closed_not_run,
    }

    if decision.due is not None:
        attempted = attempt_date(
            decision.due,
            run_date_et=run_date_et,
            now_utc=now_utc,
            slate=slate,
            schedule_refreshed=schedule_refreshed,
            refresh_schedule=refresh_schedule,
            first_refit=_first_refit(records.read_date_events(ledger)),
            ledger_dir=ledger,
            artifacts_dir=artifacts,
            gold_dir=gold_dir,
            silver_dir=silver_dir,
            steps=steps,
        )
        return dataclasses.replace(attempted, **shared)
    if recovered is not None:
        cadence_date, status, served = recovered
        return RefitOutcome(
            status=status,
            wrote_records=True,
            after_ids=served,
            **_date_fields(cadence_date),
            **shared,
        )
    in_force = decision.in_force_list
    return RefitOutcome(
        status=_idle_status(decision),
        list_week=in_force,
        next_date=None if in_force is None else _first_date_after(in_force, now_utc),
        wrote_records=wrote,
        **shared,
    )


def run_refit_side_step(
    *,
    run_date_et: date,
    now_utc: datetime,
    slate: DailySlate | None,
    schedule_refreshed: bool,
    refresh_schedule: Callable[[], object] | None,
    ledger_dir: Path | str = LEDGER_DIR,
    artifacts_dir: Path | str = Path("artifacts"),
    gold_dir: Path | str = Path("data/gold"),
    silver_dir: Path | str = Path("data/silver"),
    steps: RefitSteps | None = None,
) -> RefitOutcome:
    """The daily run's re-fit side step: recover, evaluate A/A, close, attempt (module doc).

    A run with nothing to record takes no lock and writes nothing. A second concurrent invocation
    finds the attempt lock held and returns ``busy`` having recorded nothing (R9).

    Args:
        run_date_et: The daily run's ET calendar day.
        now_utc: The run's instant at the re-fit point (tz-aware).
        slate: The day's slate on a prediction day, else None.
        schedule_refreshed: Whether this run refreshed the schedule (False on ``lock_passed``).
        refresh_schedule: Refreshes it before a due attempt when it was not.
        ledger_dir, artifacts_dir, gold_dir, silver_dir: The stores (production defaults).
        steps: The heavy steps; None is the production set.

    Raises:
        refit.records.RefitRecordFormatError: a record file is unreadable (never read as empty).
    """
    if now_utc.tzinfo is None:
        msg = f"the re-fit side step needs an aware instant, got the naive {now_utc!r}"
        raise ValueError(msg)
    ledger = Path(ledger_dir)
    artifacts = Path(artifacts_dir)
    if not _has_work(ledger, artifacts, now_utc):
        decision = _decide(ledger, now_utc)
        in_force = decision.in_force_list
        return RefitOutcome(
            status=_idle_status(decision),
            list_week=in_force,
            next_date=None
            if in_force is None
            else _first_date_after(in_force, now_utc),
        )
    with contextlib.ExitStack() as stack:
        try:
            stack.enter_context(records.attempt_lock(ledger))
        except FileLockHeldError:
            logger.warning("Re-fit attempt lock is held; this run records nothing")
            return RefitOutcome(status="busy")
        return _run_locked(
            run_date_et=run_date_et,
            now_utc=now_utc,
            slate=slate,
            schedule_refreshed=schedule_refreshed,
            refresh_schedule=refresh_schedule,
            ledger=ledger,
            artifacts=artifacts,
            gold_dir=Path(gold_dir),
            silver_dir=Path(silver_dir),
            steps=steps or RefitSteps(),
        )
