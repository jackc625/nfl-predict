"""The closing-line process's one notification: a failed reading pushes at normal priority (D-13).

WHEN THIS RUNS
--------------
``scripts/capture_closing_lines.py`` calls :func:`notify_closing_failure` once for each run whose
``closing_capture`` outcome is ``failed``, and for no other outcome. It is called after the failed
event was attempted, and whether or not that event could be written: the outside watcher does not
monitor this process, so the push is its only external signal. When the event could not be
written, the push says so, because the weekly email reads the run log and will not list it.

NEVER A CHECK-IN
----------------
Only the daily run's end-of-run step checks in with the watcher (SPEC must-NOT). This module sends
one phone push through the closed router and nothing else. Nothing here suppresses, dedupes or
rate-limits a message: two failures in one night are two pushes.

This module lives apart from ``notify.end_of_run`` so the closing process never imports the daily
run's end-of-run code.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from datetime import datetime

from notify import secrets as notify_secrets
from notify.events import SendContext, send_push
from notify.messages import closing_failure_message
from notify.secrets import NotifySecrets
from notify.transport import Delivery
from utils.date_utils import ET

__all__ = ["LOAD_SECRETS", "UNLOGGED_NOTE", "notify_closing_failure"]

# Seam, read at call time: tests hand in dummy secrets so the owner's .env is never read.
LOAD_SECRETS = notify_secrets.load_secrets

UNLOGGED_NOTE = (
    "Note: this failure could not be written to logs/ledger_runs.jsonl, so the weekly email "
    "will not list it."
)


def notify_closing_failure(
    game_ids: Sequence[str] | None,
    error_type: str,
    *,
    log_written: bool = True,
    secrets: NotifySecrets | None = None,
) -> Delivery:
    """Send the one normal-priority failed push for a failed closing reading.

    Args:
        game_ids: The games the reading was for.
        error_type: The exception's class name only (a message can carry a request URL).
        log_written: False when the failed ``closing_capture`` event could not be written; the
            push then carries :data:`UNLOGGED_NOTE`.
        secrets: The notification secrets (default: :data:`LOAD_SECRETS`, read at call time).

    Returns:
        The delivery, also printed as ``NOTIFY_FAILED= <state> <title>``.
    """
    loaded = secrets if secrets is not None else LOAD_SECRETS()
    context = SendContext("closing_task", datetime.now(ET).date(), loaded)
    message = closing_failure_message(game_ids, error_type)
    if not log_written:
        message = dataclasses.replace(
            message, header_lines=(*message.header_lines, UNLOGGED_NOTE)
        )
    delivery = send_push(message, context)
    state = (
        f"delivered {delivery.receipt_id}"
        if delivery.delivered
        else f"failed {delivery.failure_kind}"
    )
    print(f"NOTIFY_FAILED= {state} {message.title}")  # noqa: T201
    return delivery
