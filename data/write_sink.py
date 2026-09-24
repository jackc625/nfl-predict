"""The ONE write sink every persisting primitive on the daily path consults (Plan 33.2-27 Task 2b).

WHY
---
A dry run of the daily lock-time cycle must execute the REAL collection code and write nothing.
The capture and ingest steps call write-producing functions directly and accept no execution
context (``pipeline/steps.py``), so a flag threaded through their signatures would have to reach
every call at every depth. Instead each write PRIMITIVE asks :func:`current_sink` before its first
filesystem operation:

* :class:`ProductionSink` (the default) -- the write happens, byte-for-byte as it always did;
* :class:`RecordingSink` -- the intended write is RECORDED (target, kind, rows) and NOTHING is
  persisted.

This is the mechanism ``data.upstream_live`` already uses to re-address loads "at any depth, with
their call shapes unmodified" (``as_of_capture`` / ``current_as_of``): a module-level
``ContextVar``, set by a context manager that resets its token in a ``finally``, and ONE read
function every consumer calls. Two implementations of "which sink is in force" would be two
rules, and the one that disagreed would be the one that wrote during a dry run.

THE PRIMITIVES THAT CONSULT IT
------------------------------
``data.storage``: ``save_dataframe``, ``save_bronze_snapshot``, ``upsert_silver``,
``upsert_silver_composite`` and the live accumulating writer ``append_odds_captures``;
``data.sealed_probe_log.append_probe_entry`` and ``data.upstream_live.write_live_manifest`` (the
two git-tracked config records a live capture appends to);
``pipeline.skip_log.append_skip_record``; and this module's own :func:`write_csv` and
:func:`append_jsonl`, which the daily entry point uses for its prediction files and run records.

WHAT THE SEAM DOES NOT COVER, stated rather than hidden
-------------------------------------------------------
``pipeline.steps.step_populate_web_cache`` writes the derived ``data/web_cache.duckdb`` through
``api.cache.populate_cache``, outside every primitive above. A dry run therefore STOPS BEFORE that
step; the cache is rebuilt wholesale from the durable artifacts by the next real run, so skipping
it omits a derived artifact, not a record. Any other writer that bypasses these primitives is
caught by the content-digest bracket in ``tests/data_boundary.py``, which remains the SECOND line
behind the sink.
"""

from __future__ import annotations

import contextlib
import json
from collections.abc import Iterator
from contextvars import ContextVar
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd


@dataclass(frozen=True)
class IntendedWrite:
    """One write a primitive was asked to make.

    Attributes:
        target: What would have been written (a path, or ``layer/table``).
        kind: The primitive's name (``upsert_silver``, ``append_probe_entry``, ...).
        rows: Rows in the write, where the write is a frame; ``None`` otherwise.
    """

    target: str
    kind: str
    rows: int | None = None


class ProductionSink:
    """The default sink: every write happens."""

    def authorize(self, target: str, kind: str, rows: int | None = None) -> bool:
        """Return True: the caller performs the write."""
        return True


@dataclass
class RecordingSink:
    """Records every intended write and persists NOTHING.

    Attributes:
        intended_writes: Every write a primitive was asked to make, in order.
    """

    intended_writes: list[IntendedWrite] = field(default_factory=list)

    def authorize(self, target: str, kind: str, rows: int | None = None) -> bool:
        """Record the write and return False: the caller must NOT perform it."""
        self.intended_writes.append(IntendedWrite(str(target), kind, rows))
        return False

    @property
    def production_write_count(self) -> int:
        """Writes actually performed under this sink: always zero, by construction."""
        return 0


WriteSink = ProductionSink | RecordingSink

_PRODUCTION = ProductionSink()
_SINK_VAR: ContextVar[WriteSink] = ContextVar("write_sink", default=_PRODUCTION)


def current_sink() -> WriteSink:
    """The sink in force. Every write primitive calls this, and nothing re-derives it."""
    return _SINK_VAR.get()


def write_csv(frame: pd.DataFrame, path: Path, *, kind: str) -> bool:
    """Write *frame* to *path* as CSV, through the sink. Returns True when it was written.

    The daily entry point persists through this and :func:`append_jsonl` and never calls
    ``to_csv`` or ``open`` itself, so a dry run cannot write through a side door.
    """
    if not current_sink().authorize(str(path), kind, len(frame)):
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return True


def append_jsonl(path: Path, record: dict[str, Any], *, kind: str) -> bool:
    """Append one JSON line to *path*, through the sink. Returns True when it was written."""
    if not current_sink().authorize(str(path), kind, 1):
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
    return True


@contextlib.contextmanager
def active_sink(sink: WriteSink) -> Iterator[WriteSink]:
    """Put *sink* in force for the duration of the block.

    The token is RESET in a ``finally``, so the sink never survives its block, including when
    the body raises: a failed dry run must not leave recording on for a later real run -- nor,
    worse, leave a real run silently recording.
    """
    token = _SINK_VAR.set(sink)
    try:
        yield sink
    finally:
        _SINK_VAR.reset(token)
