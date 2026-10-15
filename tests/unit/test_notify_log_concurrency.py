"""Two processes appending to the notification log at once never tear a record (SPEC R3 backstop, D-11).

The daily run and the closing-line task are separate processes that can send at the same moment.
``notify.log.append_notification`` takes an exclusive OS lock and makes ONE write per record, so
every line must parse as one complete record and none may be lost or duplicated.

Two ``spawn`` processes (the start method Windows uses) each append 200 records to ONE log file
under ``tmp_path``, released together by a barrier: without it, spawn start-up is slow enough that
the first worker finishes before the second begins and nothing is ever contended. (The records need
not interleave in the file: the waiting worker re-polls the lock every 0.25 s, so the holder may
finish a long run between polls.) Each record carries a long ``receipt_id`` so a torn or interleaved
write would be visible, and a ``failure_kind`` naming its worker and index so every append is
accounted for.
The children run outside pytest's monkeypatches and are handed the explicit path, so they write only
under ``tmp_path``.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
import multiprocessing
from collections import Counter
from pathlib import Path
from typing import Any

from notify.log import NOTIFICATION_FIELDS, append_notification

APPENDS_PER_WORKER = 200
JOIN_TIMEOUT_SECONDS = 120.0
WORKERS = {"worker_a": "daily_run", "worker_b": "closing_task"}


def _append_many(path_text: str, label: str, count: int, start: Any = None) -> None:
    """Append *count* records for *label* to the log at *path_text* (runs in a child process).

    *start*, when given, is a barrier every worker waits on, so all of them append at once.
    """
    path = Path(path_text)
    if start is not None:
        start.wait(JOIN_TIMEOUT_SECONDS)
    for index in range(count):
        append_notification(
            {
                "run_id": f"run_{label}",
                "run_date_et": "2026-10-17",
                "source": WORKERS[label],
                "event_class": "failed",
                "priority": 3,
                "attempts": 1,
                "outcome": "failed",
                # The maximum receipt length the closed shape allows: a torn write shows.
                "receipt_id": f"{label}_{index:04d}_".ljust(128, "x"),
                "failure_kind": f"{label}_{index}",
                "sent_at": "2026-10-17T21:00:00+00:00",
            },
            path=path,
        )


def test_two_processes_appending_at_once_never_tear_a_record(tmp_path: Path) -> None:
    log_path = tmp_path / "notifications.jsonl"
    context = multiprocessing.get_context("spawn")
    start = context.Barrier(len(WORKERS))
    processes = [
        context.Process(
            target=_append_many,
            args=(str(log_path), label, APPENDS_PER_WORKER, start),
        )
        for label in WORKERS
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join(JOIN_TIMEOUT_SECONDS)

    for process in processes:
        assert not process.is_alive(), "a worker did not finish within the timeout"
        assert process.exitcode == 0
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == APPENDS_PER_WORKER * len(WORKERS)
    records = [json.loads(line) for line in lines]
    for record in records:
        assert isinstance(record, dict)
        assert set(record) == set(NOTIFICATION_FIELDS)
        assert len(record["receipt_id"]) == 128
    appended = Counter(record["failure_kind"] for record in records)
    for label in WORKERS:
        expected = {f"{label}_{index}" for index in range(APPENDS_PER_WORKER)}
        assert {kind for kind in appended if kind.startswith(f"{label}_")} == expected
    assert set(appended.values()) == {1}
