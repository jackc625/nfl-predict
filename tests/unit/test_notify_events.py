"""The closed push router's must-NOTs (Phase 35 SPEC must-NOTs, R1/R2 edges, T-35-47).

* Only ``failed``, ``bets`` and ``test`` may be pushed: any other class is refused before a request.
* Nothing suppresses a message: two identical failures are two deliveries and two log records.
* Records land in send order.
* A delivered push whose log record could not be written is delivered but visibly unlogged.

The MockTransport, sleep-recorder and dummy-secret helpers come from
``tests/unit/test_notify_transport.py``; Plans 35-07 and 35-08 extend this file.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json

import pytest

from notify import log as notification_log
from notify.events import PRIORITY_NORMAL, PushMessage, send_push
from notify.log import read_notifications
from tests.unit.test_notify_transport import (
    DUMMY_RECEIPT_ID,
    daily_context,
    failed_push,
    install_sleep_recorder,
    install_transport,
    ntfy_ok,
)
from utils.file_lock import FileLockHeldError, exclusive_file_lock


@pytest.mark.parametrize("event_class", ["digest", "alert", "checkin"])
def test_the_push_router_refuses_an_unknown_class(
    monkeypatch: pytest.MonkeyPatch, event_class: str
) -> None:
    install_sleep_recorder(monkeypatch)
    requests = install_transport(monkeypatch, ntfy_ok())
    message = PushMessage(
        event_class=event_class,
        title="not a push class",
        header_lines=(),
        item_lines=(),
        priority=PRIORITY_NORMAL,
    )

    with pytest.raises(ValueError, match="closed vocabulary"):
        send_push(message, daily_context())

    assert len(requests) == 0
    assert read_notifications() == []


def test_two_failures_back_to_back_are_two_deliveries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_sleep_recorder(monkeypatch)
    requests = install_transport(monkeypatch, ntfy_ok())

    first = send_push(failed_push(), daily_context())
    second = send_push(failed_push(), daily_context())

    assert first.delivered
    assert second.delivered
    assert len(requests) == 2
    records = read_notifications()
    assert len(records) == 2
    assert [record["outcome"] for record in records] == ["delivered", "delivered"]


def test_records_are_appended_in_send_order(monkeypatch: pytest.MonkeyPatch) -> None:
    install_sleep_recorder(monkeypatch)
    receipts = ["receiptFirst1", "receiptSecond", "receiptThird3"]
    requests = install_transport(
        monkeypatch, *(ntfy_ok(receipt) for receipt in receipts)
    )
    titles = ["FAILED: first", "FAILED: second", "FAILED: third"]

    for title in titles:
        send_push(failed_push(title), daily_context())

    assert [json.loads(request.content)["title"] for request in requests] == titles
    assert [record["receipt_id"] for record in read_notifications()] == receipts


def test_a_delivered_push_whose_log_lock_times_out_is_delivered_but_unlogged(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    install_sleep_recorder(monkeypatch)
    install_transport(monkeypatch, ntfy_ok())

    def _lock_never_frees(*_args: object, **_kwargs: object) -> None:
        msg = "the lock is held by another writer (waited 60s)."
        raise FileLockHeldError(msg)

    monkeypatch.setattr(notification_log, "exclusive_file_lock", _lock_never_frees)

    delivery = send_push(failed_push(), daily_context())

    assert delivery.delivered is True
    assert delivery.receipt_id == DUMMY_RECEIPT_ID
    assert delivery.logged is False
    log_path = notification_log.NOTIFICATION_LOG
    assert not log_path.exists() or log_path.read_text(encoding="utf-8") == ""
    assert "NOTIFICATION_LOG= failed FileLockHeldError" in capsys.readouterr().out

    monkeypatch.setattr(notification_log, "exclusive_file_lock", exclusive_file_lock)

    recorded = send_push(failed_push(), daily_context())

    assert recorded.delivered is True
    assert recorded.logged is True
    assert len(read_notifications()) == 1
