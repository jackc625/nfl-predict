"""The Phase 37 records: append-only JSONL files under ``<ledger>/refit/`` (D-19, SPEC R8).

WHERE THEY LIVE, AND WHY
------------------------
Every Phase 37 record lives inside the ledger directory, under :data:`REFIT_DIRNAME`, so the
Phase 34 private backup carries it and the ledger stays self-contained. The first record kind is
the shadow hole: a new live row written without a frozen shadow twin, with the reason named
(an exception class, or ``no_shadow_row``). A hole is recorded, never back-filled later.

APPEND-ONLY, THROUGH THE WRITE SINK
-----------------------------------
Each record is one JSON line appended through ``data.write_sink.append_jsonl``, so a dry run
records the intended write and persists nothing, and the test-write guard sees every write.

A MALFORMED RECORD IS NEVER READ AS "NO RECORDS"
------------------------------------------------
:func:`read_records` returns ``[]`` only for an absent file. A line that is not one JSON object
raises :class:`RefitRecordFormatError` naming the file and the 1-based line number: a skipped hole
would read as a clean week.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from data.write_sink import append_jsonl

__all__ = [
    "REFIT_DIRNAME",
    "SHADOW_HOLES_FILENAME",
    "RefitRecordFormatError",
    "append_record",
    "read_records",
    "record_shadow_holes",
    "refit_dir",
]

REFIT_DIRNAME: str = "refit"
SHADOW_HOLES_FILENAME: str = "shadow_holes.jsonl"

# The write-sink kind of a shadow-hole record.
_SHADOW_HOLE_KIND: str = "refit_shadow_hole"


class RefitRecordFormatError(Exception):
    """A Phase 37 record file holds a line that is not one JSON object (path and line named).

    Inherits bare ``Exception`` (the ``data.graded_weeks`` rule): an unreadable record must never
    be degraded into "no records".
    """


def refit_dir(ledger_dir: Path | str) -> Path:
    """The directory holding every Phase 37 record of the ledger at *ledger_dir*."""
    return Path(ledger_dir) / REFIT_DIRNAME


def append_record(
    ledger_dir: Path | str, filename: str, record: dict[str, Any], *, kind: str
) -> bool:
    """Append *record* to ``<ledger>/refit/<filename>``, stamped ``recorded_at_utc``.

    Returns:
        True when the line was written (False under a recording sink).
    """
    stamped = {**record, "recorded_at_utc": datetime.now(tz=UTC).isoformat()}
    return append_jsonl(refit_dir(ledger_dir) / filename, stamped, kind=kind)


def read_records(ledger_dir: Path | str, filename: str) -> list[dict[str, Any]]:
    """Every record in ``<ledger>/refit/<filename>``, oldest first; ``[]`` when it is absent.

    Raises:
        RefitRecordFormatError: a non-blank line is not one JSON object.
    """
    path = refit_dir(ledger_dir) / filename
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            msg = f"refit record {path.as_posix()} line {number} is not JSON: {error}"
            raise RefitRecordFormatError(msg) from error
        if not isinstance(record, dict):
            msg = f"refit record {path.as_posix()} line {number} is not one JSON object"
            raise RefitRecordFormatError(msg)
        records.append(record)
    return records


def record_shadow_holes(
    ledger_dir: Path | str,
    holes: Iterable[tuple[tuple[str, int, int, str], str]],
    *,
    decided_at: datetime,
) -> None:
    """Append one record per hole: the live row it names, the reason and the decision instant.

    Args:
        ledger_dir: The ledger directory.
        holes: ``((game_id, season, week, target), reason)`` per new live row with no shadow twin.
        decided_at: The decision instant of the run that wrote the live rows (tz-aware).
    """
    for (game_id, season, week, target), reason in holes:
        append_record(
            ledger_dir,
            SHADOW_HOLES_FILENAME,
            {
                "game_id": game_id,
                "season": season,
                "week": week,
                "target": target,
                "reason": reason,
                "decided_at_utc": decided_at.isoformat(),
            },
            kind=_SHADOW_HOLE_KIND,
        )
