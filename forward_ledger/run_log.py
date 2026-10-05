"""The ledger's run log: one JSON line per ledger event (Phase 34).

Every ledger event -- an append, a refused first pick, a settlement, a correction, a closing-line
write, an anchor or backup commit and push -- is recorded in ``logs/ledger_runs.jsonl`` with its
outcome, so a failed push or a refused pick is visible after the fact rather than lost in a console.
The vocabulary is CLOSED: an event outside :data:`LEDGER_EVENTS` is refused, so a typo cannot
create a second spelling of an event that a reader counting events would miss.

Records go through ``data.write_sink.append_jsonl`` (kind ``ledger_run``), so a dry run records the
intended write and persists nothing.

ONE RUN ID PER DAILY RUN (Plan 34-15, review finding 8)
-------------------------------------------------------
Every record carries ``run_id``: the id bound by :func:`bound_run_id` for the run that wrote it,
or ``None`` outside any binding (a CLI run, a test). The daily run binds ONE id around everything
it does, and its steps run in-process, so the append, the settle pass and the end-of-run sync's
anchor and backup events all name the same run -- which is what lets Plan 34-23 prove that each
appending run's pushes are its own. The id lives in a ``contextvars.ContextVar``, so a binding
never leaks past its block, even when the block raises.

NEVER LOG SECRETS. Callers pass outcomes, counts, hashes and redacted command lines -- never a key,
a token or an unredacted URL carrying one.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import contextvars
import json
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from data.write_sink import append_jsonl

__all__ = [
    "LEDGER_EVENTS",
    "LEDGER_RUN_LOG",
    "bound_run_id",
    "current_run_id",
    "new_run_id",
    "read_events",
    "record_event",
]

LEDGER_RUN_LOG: Path = Path("logs/ledger_runs.jsonl")

LEDGER_EVENTS: tuple[str, ...] = (
    "append",
    "first_pick_refusal",
    "settle",
    "correction_appended",
    "correction_label_withdrawn",
    "closing_finalized",
    "anchor_commit",
    "anchor_push",
    "backup_commit",
    "backup_push",
    "sync_refused",
    "closing_triggers",
    "closing_capture",
    "fill_recorded",
    "migration",
    "cutover",
)

_RESERVED_FIELDS: frozenset[str] = frozenset({"event", "at_utc", "run_id"})

_RUN_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "ledger_run_id", default=None
)


def new_run_id() -> str:
    """A fresh run id: 32 lowercase hex characters, unique per call."""
    return uuid.uuid4().hex


@contextmanager
def bound_run_id(run_id: str) -> Iterator[str]:
    """Bind *run_id* to every event recorded inside the block; the binding is reset on exit."""
    token = _RUN_ID.set(run_id)
    try:
        yield run_id
    finally:
        _RUN_ID.reset(token)


def current_run_id() -> str | None:
    """The run id bound by the innermost :func:`bound_run_id`, or None outside any binding."""
    return _RUN_ID.get()


def record_event(event: str, *, path: Path | None = None, **fields: Any) -> bool:
    """Append ``{"event", "at_utc", "run_id", **fields}`` to the run log. Returns True when written.

    *path* defaults to :data:`LEDGER_RUN_LOG`, read at call time.

    Raises:
        ValueError: *event* is outside :data:`LEDGER_EVENTS`, or a field would overwrite
            ``event`` / ``at_utc`` / ``run_id``.
    """
    if event not in LEDGER_EVENTS:
        msg = f"ledger event {event!r} is outside the closed vocabulary {LEDGER_EVENTS}"
        raise ValueError(msg)
    clashing = sorted(_RESERVED_FIELDS & set(fields))
    if clashing:
        msg = f"ledger event fields {clashing} are reserved by the record itself"
        raise ValueError(msg)
    record = {
        "event": event,
        "at_utc": datetime.now(tz=UTC).isoformat(),
        "run_id": current_run_id(),
        **fields,
    }
    return append_jsonl(path or LEDGER_RUN_LOG, record, kind="ledger_run")


def read_events(
    path: Path | None = None, event: str | None = None
) -> list[dict[str, Any]]:
    """Every record in the run log, oldest first; only *event*'s records when it is given.

    *path* defaults to :data:`LEDGER_RUN_LOG`, read at call time. An absent log is ``[]``: no
    event has been recorded yet.

    Raises:
        ValueError: a line is not one JSON object, named by its path and 1-based line number.
            A malformed log is never read as "no events": a skipped ``closing_capture`` record
            would turn a recorded skip reason into ``capture_missed``.
    """
    log_path = path or LEDGER_RUN_LOG
    if not log_path.exists():
        return []
    records: list[dict[str, Any]] = []
    for number, line in enumerate(
        log_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            msg = f"ledger run log {log_path.as_posix()} line {number} is not JSON: {error}"
            raise ValueError(msg) from error
        if not isinstance(record, dict):
            msg = f"ledger run log {log_path.as_posix()} line {number} is not one JSON object"
            raise ValueError(msg)
        if event is None or record.get("event") == event:
            records.append(record)
    return records
