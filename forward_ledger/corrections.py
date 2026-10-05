"""Correction entries for scores restated after grading (Phase 34, Plan 34-09; D-05, D-06).

WHERE THE OBLIGATION COMES FROM
-------------------------------
Phase 32's live detector (``data/live_revision.py``) records a verdict on every live capture in
``config/upstream_live/<season>.json``. When a capture moved a week whose bets were already graded,
the verdict's ``event_class`` is ``live_revision_graded`` and it carries ``correction_owed: true``
with a scope ``{season, weeks, dataset, ...}`` (D32-12). The detector only RAISES the obligation;
this module discharges it. :func:`owed_correction_events` reads those verdicts.

WHAT A CORRECTION IS
--------------------
For each graded live-status row in an owed week, :func:`correction_payloads` re-grades the row
under its OWN target strategy against the corrected score, through the one re-grade path
``forward_ledger.settle.regrade_row``. Where the re-graded outcome differs from the outcome IN FORCE
it builds one correction payload (the 22 ``CORRECTION_COLUMNS_V1`` fields); applied through
``forward_ledger.store.commit_changes`` it becomes a chained correction entry. The row's own
grading half is never touched: the one-way grader stays one-way (D-05).

THE OUTCOME IN FORCE (D-05, D-06)
---------------------------------
:func:`in_force_outcomes` answers "what does the ledger say this row returned": the LATEST
correction entry for the key, else the row's own grade. A payload records three states side by
side -- the row's ``original_*`` grade (never changes), the ``prior_*`` state it replaces (the
in-force one, with the in-force correction's seq) and the ``corrected_*`` state it puts in force
-- so a win -> loss -> win history reads loss -> win on its second entry (34-REVIEWS finding 3).
The 2026 verdict and the ``/bets`` tracker count the in-force outcome.

WHY THIS IS QUIET MOST WEEKS (34-RESEARCH Pitfall 15)
-----------------------------------------------------
Upstream restates graded weeks almost every week, so an owed event usually changes no outcome.
Nothing is produced unless the outcome actually changes, and a re-run of an applied event finds
the corrected outcome already in force and produces nothing again.

A WITHDRAWN SCORE (34-RESEARCH Q3)
----------------------------------
A row whose corrected score is now missing gets no entry; it is returned as one observation for
the run log (event ``correction_label_withdrawn``), and raised to the owner only if it happens.

Pure: no I/O and no clock read -- the caller passes the manifest, the realized values and ``now``.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from api.cache import BET_STATUS_LIVE, GRADING_STATUS_PENDING
from data.revision_events import (
    CORRECTION_OWED,
    CORRECTION_OWED_SCOPE,
    RevisionEventClass,
)
from data.upstream_live import CAPTURE_VERDICT_KEY
from forward_ledger.canonical import (
    CORRECTION_COLUMNS_V1,
    ENTRY_KIND_CORRECTION,
    ENTRY_KIND_ROW,
)
from forward_ledger.schema import LEDGER_ROW_KEY
from forward_ledger.settle import regrade_row
from forward_ledger.store import LedgerEntry

__all__ = [
    "LIVE_REVISION_GRADED",
    "CorrectionResult",
    "InForce",
    "OwedEvent",
    "correction_payloads",
    "in_force_outcomes",
    "owed_correction_events",
]

# The event class that owes a correction, taken from Phase 32's frozen vocabulary, never re-typed.
LIVE_REVISION_GRADED: str = str(RevisionEventClass.LIVE_REVISION_GRADED)


@dataclass(frozen=True)
class OwedEvent:
    """One capture whose verdict owes a correction: its identity, instant and owed weeks."""

    dataset: str
    season: int
    week: int
    sequence: int
    captured_at_utc: str
    scope_weeks: tuple[int, ...]


@dataclass(frozen=True)
class InForce:
    """The outcome the ledger currently stands on for one row key.

    ``corrected`` is True when it comes from a correction entry, whose ledger seq is
    ``correction_seq``; both are False / None when the row's own grade is in force.
    """

    grading_status: str
    outcome: bool | None
    payout_flat: float | None
    realized_units: float | None
    corrected: bool
    correction_seq: int | None


@dataclass(frozen=True)
class CorrectionResult:
    """What one correction pass found: payloads to append and observations to log."""

    payloads: tuple[dict[str, Any], ...]
    observations: tuple[dict[str, Any], ...]


def _instant(text: str) -> datetime:
    """A capture's ``captured_at_utc`` as a tz-aware instant, so events order by time, not text."""
    instant = datetime.fromisoformat(text)
    if instant.tzinfo is None:
        msg = f"captured_at_utc {text!r} is naive; a detection instant must carry its offset"
        raise ValueError(msg)
    return instant


def owed_correction_events(manifest: Mapping[str, Any]) -> list[OwedEvent]:
    """Every capture in *manifest* whose verdict owes a correction, oldest first.

    A capture qualifies when its ``revision`` block has ``event_class == live_revision_graded`` and
    ``correction_owed`` is exactly ``True``; the owed weeks are its scope's ``weeks``.
    """
    events: list[OwedEvent] = []
    for dataset, block in (manifest.get("datasets") or {}).items():
        for capture in block.get("captures") or ():
            revision = capture.get(CAPTURE_VERDICT_KEY)
            if not isinstance(revision, Mapping):
                continue
            if revision.get("event_class") != LIVE_REVISION_GRADED:
                continue
            if revision.get(CORRECTION_OWED) is not True:
                continue
            scope = revision[CORRECTION_OWED_SCOPE]
            events.append(
                OwedEvent(
                    dataset=str(revision.get("dataset", dataset)),
                    season=int(scope["season"]),
                    week=int(capture["week"]),
                    sequence=int(capture["sequence"]),
                    captured_at_utc=str(capture["captured_at_utc"]),
                    scope_weeks=tuple(sorted(int(week) for week in scope["weeks"])),
                )
            )
    return sorted(
        events,
        key=lambda event: (
            _instant(event.captured_at_utc),
            event.dataset,
            event.sequence,
        ),
    )


def _key(values: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(values[name] for name in LEDGER_ROW_KEY)


def in_force_outcomes(entries: Sequence[LedgerEntry]) -> dict[tuple[Any, ...], InForce]:
    """``LEDGER_ROW_KEY`` values -> the outcome in force: the latest correction, else the row's grade."""
    in_force: dict[tuple[Any, ...], InForce] = {}
    for entry in entries:
        if entry.kind == ENTRY_KIND_ROW:
            grading = entry.grading or {}
            in_force[_key(entry.immutable)] = InForce(
                grading_status=grading.get("grading_status") or GRADING_STATUS_PENDING,
                outcome=grading.get("outcome"),
                payout_flat=grading.get("payout_flat"),
                realized_units=grading.get("realized_units"),
                corrected=False,
                correction_seq=None,
            )
        elif entry.kind == ENTRY_KIND_CORRECTION:
            values = entry.immutable
            in_force[_key(values)] = InForce(
                grading_status=values["corrected_grading_status"],
                outcome=values["corrected_outcome"],
                payout_flat=values["corrected_payout_flat"],
                realized_units=values["corrected_realized_units"],
                corrected=True,
                correction_seq=entry.seq,
            )
    return in_force


def _owed_by_week(events: Sequence[OwedEvent]) -> dict[tuple[int, int], OwedEvent]:
    """``(season, week) -> the LATEST owed event covering it`` (events arrive oldest first)."""
    covering: dict[tuple[int, int], OwedEvent] = {}
    for event in events:
        for week in event.scope_weeks:
            covering[(event.season, week)] = event
    return covering


def _source(event: OwedEvent) -> dict[str, Any]:
    return {
        "dataset": event.dataset,
        "season": event.season,
        "week": event.week,
        "sequence": event.sequence,
    }


def correction_payloads(
    entries: Sequence[LedgerEntry],
    events: Sequence[OwedEvent],
    realized: Mapping[str, Mapping[str, float]],
    strategies: Mapping[str, Any],
    now: datetime,
) -> CorrectionResult:
    """The correction payloads owed by *events*, and the observations to log. Writes nothing.

    Args:
        entries: The ledger, in file order.
        events: :func:`owed_correction_events`' output, oldest first.
        realized: ``{target -> {game_id -> realized value}}`` from the CORRECTED scores
            (``forward_ledger.settle.realized_values_from_scores``).
        strategies: ``{target -> strategy}`` (``forward_ledger.settle.live_strategies``).
        now: The correction instant, tz-aware.

    Returns:
        At most one payload per row key, each exactly the 22 ``CORRECTION_COLUMNS_V1`` fields;
        one observation ``{key, week, source}`` per in-scope row whose corrected score is missing.

    Raises:
        ValueError: *now* is naive, or a row's target has no registered strategy.
    """
    if now.tzinfo is None:
        msg = f"the correction instant {now!r} is naive; it must be tz-aware"
        raise ValueError(msg)
    corrected_at = now.astimezone(UTC).isoformat()
    covering = _owed_by_week(events)
    in_force = in_force_outcomes(entries)

    payloads: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    for entry in entries:
        if entry.kind != ENTRY_KIND_ROW:
            continue
        row = {**entry.immutable, **(entry.grading or {})}
        own_status = row.get("grading_status") or GRADING_STATUS_PENDING
        if row.get("status") != BET_STATUS_LIVE or own_status == GRADING_STATUS_PENDING:
            continue
        event = covering.get((row["season"], row["week"]))
        if event is None:
            continue

        key = _key(row)
        target = str(row["target"])
        value = realized.get(target, {}).get(str(row["game_id"]))
        if value is None:
            observations.append(
                {"key": key, "week": row["week"], "source": _source(event)}
            )
            continue
        strategy = strategies.get(target)
        if strategy is None:
            msg = (
                f"no strategy registered for target {target!r}; a correction cannot re-grade a "
                "row under a rule that is not present."
            )
            raise ValueError(msg)

        regraded = regrade_row(row, strategy, value, graded_at=now)
        prior = in_force[key]
        if regraded["grading_status"] == prior.grading_status:
            continue

        payload = {
            "game_id": row["game_id"],
            "season": row["season"],
            "week": row["week"],
            "target": row["target"],
            "arm": row["arm"],
            "original_grading_status": own_status,
            "original_payout_flat": row.get("payout_flat"),
            "prior_grading_status": prior.grading_status,
            "prior_payout_flat": prior.payout_flat,
            "prior_realized_units": prior.realized_units,
            "prior_correction_seq": prior.correction_seq,
            "corrected_grading_status": regraded["grading_status"],
            "corrected_outcome": regraded["outcome"],
            "corrected_payout_flat": regraded["payout_flat"],
            "corrected_realized_units": regraded["realized_units"],
            "realized_value": float(value),
            "source_dataset": event.dataset,
            "source_season": event.season,
            "source_week": event.week,
            "source_sequence": event.sequence,
            "detected_at_utc": event.captured_at_utc,
            "corrected_at_utc": corrected_at,
        }
        payloads.append({name: payload[name] for name in CORRECTION_COLUMNS_V1})
    return CorrectionResult(tuple(payloads), tuple(observations))
