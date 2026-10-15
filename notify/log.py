"""The notification log: one JSON line per message, the one place to read what reached the owner (D-11).

WHAT A RECORD IS
----------------
One record per MESSAGE, not per attempt, with exactly the keys in :data:`NOTIFICATION_FIELDS`. The
shape is CLOSED: a record with any other key is refused, ``source`` and ``outcome`` come from closed
vocabularies, and ``failure_kind`` / ``receipt_id`` must match narrow patterns. That closed shape is
what keeps a topic, a URL, a key, a recipient or a message body out of the log.

HOW IT IS WRITTEN
-----------------
The daily run and the closing-line task can append at the same moment. Each append therefore asks the
write sink first (so a dry run records the intended write and persists nothing, and the test suite's
write guard sees it), then holds an exclusive OS lock on a sibling lock file, then makes ONE write of
one complete line and flushes. Records land in send order and never interleave.

:data:`NOTIFICATION_LOG` is read at call time, so the test suite redirects it to a per-test file.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from data.write_sink import current_sink
from utils.file_lock import exclusive_file_lock

__all__ = [
    "LOCK_WAIT_SECONDS",
    "NOTIFICATION_FIELDS",
    "NOTIFICATION_LOCK_FILENAME",
    "NOTIFICATION_LOG",
    "NOTIFICATION_OUTCOMES",
    "NOTIFICATION_SOURCES",
    "append_notification",
    "read_notifications",
]

NOTIFICATION_LOG: Path = Path("logs/notifications.jsonl")
NOTIFICATION_LOCK_FILENAME = ".notifications.lock"
# The odds store waits 120 s for the same two writers; one log line needs far less.
LOCK_WAIT_SECONDS = 60.0

NOTIFICATION_FIELDS: tuple[str, ...] = (
    "run_id",
    "run_date_et",
    "source",
    "event_class",
    "priority",
    "attempts",
    "outcome",
    "receipt_id",
    "failure_kind",
    "sent_at",
)
NOTIFICATION_SOURCES: tuple[str, ...] = ("daily_run", "closing_task", "test_command")
# ``withheld``: a check-in deliberately withheld on the one D-16 watcher test day.
NOTIFICATION_OUTCOMES: tuple[str, ...] = ("delivered", "failed", "withheld")

_FAILURE_KIND = re.compile(r"^[A-Za-z0-9_]+$")
_RECEIPT_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")


def _refuse_foreign_shape(record: Mapping[str, Any]) -> None:
    """Raise ValueError, naming the key, for any record outside the closed shape."""
    extra = sorted(set(record) - set(NOTIFICATION_FIELDS))
    if extra:
        msg = f"notification record carries keys outside {NOTIFICATION_FIELDS}: {extra}"
        raise ValueError(msg)
    missing = [name for name in NOTIFICATION_FIELDS if name not in record]
    if missing:
        msg = f"notification record is missing keys: {missing}"
        raise ValueError(msg)
    if record["source"] not in NOTIFICATION_SOURCES:
        msg = f"notification record key 'source' is outside {NOTIFICATION_SOURCES}"
        raise ValueError(msg)
    if record["outcome"] not in NOTIFICATION_OUTCOMES:
        msg = f"notification record key 'outcome' is outside {NOTIFICATION_OUTCOMES}"
        raise ValueError(msg)
    failure_kind = record["failure_kind"]
    if failure_kind is not None and not (
        isinstance(failure_kind, str) and _FAILURE_KIND.fullmatch(failure_kind)
    ):
        msg = "notification record key 'failure_kind' is not a bare identifier"
        raise ValueError(msg)
    receipt_id = record["receipt_id"]
    if receipt_id is not None and not (
        isinstance(receipt_id, str) and _RECEIPT_ID.fullmatch(receipt_id)
    ):
        msg = "notification record key 'receipt_id' is not a provider message id"
        raise ValueError(msg)


def append_notification(record: Mapping[str, Any], *, path: Path | None = None) -> bool:
    """Append *record* as one line. Returns True when written, False when the sink declined.

    *path* defaults to :data:`NOTIFICATION_LOG`, read at call time.

    Raises:
        ValueError: *record* is outside the closed shape (named by its key).
        utils.file_lock.FileLockHeldError: the lock stayed held for :data:`LOCK_WAIT_SECONDS`.
    """
    _refuse_foreign_shape(record)
    log_path = path or NOTIFICATION_LOG
    if not current_sink().authorize(str(log_path), "notification", 1):
        return False
    line = json.dumps(dict(record), sort_keys=True) + "\n"
    # The lock file is opened with O_CREAT, so its directory must exist first.
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        exclusive_file_lock(
            log_path.parent / NOTIFICATION_LOCK_FILENAME, wait_seconds=LOCK_WAIT_SECONDS
        ),
        log_path.open("a", encoding="utf-8") as handle,
    ):
        handle.write(line)
        handle.flush()
    return True


def read_notifications(path: Path | None = None) -> list[dict[str, Any]]:
    """Every record in the notification log, oldest first. An absent log is ``[]``.

    *path* defaults to :data:`NOTIFICATION_LOG`, read at call time.

    Raises:
        ValueError: a line is not one JSON object, named by its path and 1-based line number. A
            malformed log is never read as "no records".
    """
    log_path = path or NOTIFICATION_LOG
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
            msg = f"notification log {log_path.as_posix()} line {number} is not JSON: {error}"
            raise ValueError(msg) from error
        if not isinstance(record, dict):
            msg = f"notification log {log_path.as_posix()} line {number} is not one JSON object"
            raise ValueError(msg)
        records.append(record)
    return records
