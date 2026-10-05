"""The ledger's run log: one JSON line per ledger event (Phase 34).

Every ledger event -- an append, a refused first pick, a settlement, a correction, a closing-line
write, an anchor or backup commit and push -- is recorded in ``logs/ledger_runs.jsonl`` with its
outcome, so a failed push or a refused pick is visible after the fact rather than lost in a console.
The vocabulary is CLOSED: an event outside :data:`LEDGER_EVENTS` is refused, so a typo cannot
create a second spelling of an event that a reader counting events would miss.

Records go through ``data.write_sink.append_jsonl`` (kind ``ledger_run``), so a dry run records the
intended write and persists nothing.

NEVER LOG SECRETS. Callers pass outcomes, counts, hashes and redacted command lines -- never a key,
a token or an unredacted URL carrying one.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from data.write_sink import append_jsonl

__all__ = ["LEDGER_EVENTS", "LEDGER_RUN_LOG", "record_event"]

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

_RESERVED_FIELDS: frozenset[str] = frozenset({"event", "at_utc"})


def record_event(event: str, *, path: Path = LEDGER_RUN_LOG, **fields: Any) -> bool:
    """Append ``{"event", "at_utc", **fields}`` to the run log. Returns True when written.

    Raises:
        ValueError: *event* is outside :data:`LEDGER_EVENTS`, or a field would overwrite
            ``event`` / ``at_utc``.
    """
    if event not in LEDGER_EVENTS:
        msg = f"ledger event {event!r} is outside the closed vocabulary {LEDGER_EVENTS}"
        raise ValueError(msg)
    clashing = sorted(_RESERVED_FIELDS & set(fields))
    if clashing:
        msg = f"ledger event fields {clashing} are reserved by the record itself"
        raise ValueError(msg)
    record = {"event": event, "at_utc": datetime.now(tz=UTC).isoformat(), **fields}
    return append_jsonl(path, record, kind="ledger_run")
