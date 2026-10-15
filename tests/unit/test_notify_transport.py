"""The ntfy push path: a provider receipt is the only proof of delivery (Phase 35 R1, D-10..D-12).

Every request goes through ``notify.transport.CLIENT_FACTORY`` replaced by an ``httpx.MockTransport``
client, and every retry delay through a recording ``SLEEP``, so nothing leaves the machine and
nothing sleeps. Secrets are dummy ``SecretStr`` values; the owner's ``.env`` is never read.

The helpers at the top are shared with ``tests/unit/test_notify_events.py``.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import date
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from notify import log as notification_log
from notify import transport
from notify.events import PRIORITY_TOP, PushMessage, SendContext, send_push
from notify.log import NOTIFICATION_FIELDS, read_notifications
from notify.secrets import NTFY_SERVER_URL, NotifySecrets

# A 32-character dummy topic, the shape the SPEC requires of the real one.
DUMMY_TOPIC = "dummytopic0123456789abcdefghijkl"
DUMMY_PING_UUID = "00000000-1111-2222-3333-444444444444"
DUMMY_PING_URL = f"https://hc-ping.com/{DUMMY_PING_UUID}"
DUMMY_RECEIPT_ID = "aB3dE5fG7hJ9"
RUN_DATE = date(2026, 10, 17)

# One reply per request, in order: a zero-argument callable building a fresh response, or an
# exception instance the transport raises. The last reply repeats once the list runs out.
Reply = Callable[[], httpx.Response] | Exception


def dummy_secrets(**overrides: Any) -> NotifySecrets:
    """A NotifySecrets carrying dummy values only."""
    values: dict[str, Any] = {
        "ntfy_topic": SecretStr(DUMMY_TOPIC),
        "healthchecks_ping_url": SecretStr(DUMMY_PING_URL),
    }
    values.update(overrides)
    return NotifySecrets(**values)


def daily_context(secrets: NotifySecrets | None = None) -> SendContext:
    """A daily-run send context over the dummy secrets."""
    return SendContext(
        source="daily_run",
        run_date_et=RUN_DATE,
        secrets=secrets if secrets is not None else dummy_secrets(),
    )


def failed_push(title: str = "FAILED: 3 games not predicted") -> PushMessage:
    """A top-priority failed push."""
    return PushMessage(
        event_class="failed",
        title=title,
        header_lines=(
            "The daily run failed.",
            "Next action: re-run before 6:00 PM ET.",
        ),
        item_lines=("BUF@MIA", "KC@DEN", "DAL@NYG"),
        priority=PRIORITY_TOP,
    )


def ntfy_ok(receipt_id: str = DUMMY_RECEIPT_ID) -> Callable[[], httpx.Response]:
    """An ntfy success reply carrying a message id (the real reply also echoes the topic)."""
    return lambda: httpx.Response(200, json={"id": receipt_id, "topic": DUMMY_TOPIC})


def install_transport(
    monkeypatch: pytest.MonkeyPatch, *replies: Reply
) -> list[httpx.Request]:
    """Route every notify request to a MockTransport answering with *replies*; return the requests."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        reply = replies[min(len(requests), len(replies)) - 1]
        if isinstance(reply, Exception):
            raise reply
        return reply()

    monkeypatch.setattr(
        transport,
        "CLIENT_FACTORY",
        lambda: httpx.Client(transport=httpx.MockTransport(handler)),
    )
    return requests


def install_sleep_recorder(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record every retry delay the transport asks for, in order; nothing actually sleeps."""
    recorded: list[float] = []
    monkeypatch.setattr(transport, "SLEEP", recorded.append)
    return recorded


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Every retry delay the transport asked for, in order; nothing actually sleeps."""
    return install_sleep_recorder(monkeypatch)


def test_one_failed_push_is_delivered_and_logged(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    requests = install_transport(monkeypatch, ntfy_ok())

    delivery = send_push(failed_push(), daily_context())

    assert delivery.delivered
    assert delivery.attempts == 1
    assert delivery.receipt_id == DUMMY_RECEIPT_ID
    assert delivery.logged is True
    assert sleeps == []

    (request,) = requests
    assert request.method == "POST"
    assert str(request.url) == NTFY_SERVER_URL == "https://ntfy.sh/"
    assert DUMMY_TOPIC not in str(request.url)
    body = json.loads(request.content)
    assert set(body) == {"topic", "title", "message", "priority"}
    assert "tags" not in body
    assert body["topic"] == DUMMY_TOPIC
    assert body["priority"] == 5
    assert isinstance(body["priority"], int)

    lines = notification_log.NOTIFICATION_LOG.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert set(record) == set(NOTIFICATION_FIELDS)
    assert record["receipt_id"] == DUMMY_RECEIPT_ID
    assert record["event_class"] == "failed"
    assert record["priority"] == 5
    assert record["outcome"] == "delivered"
    assert record["attempts"] == 1
    assert record["source"] == "daily_run"
    assert record["run_date_et"] == RUN_DATE.isoformat()
    assert record["failure_kind"] is None
    assert DUMMY_TOPIC not in lines[0]
    assert DUMMY_PING_UUID not in lines[0]


def test_unset_topic_is_a_recorded_failure_not_a_skip(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    requests = install_transport(monkeypatch, ntfy_ok())

    delivery = send_push(failed_push(), daily_context(dummy_secrets(ntfy_topic=None)))

    assert requests == []
    assert not delivery.delivered
    assert delivery.failure_kind == "missing_secret"
    assert delivery.attempts == 1
    assert sleeps == []
    (record,) = read_notifications()
    assert record["outcome"] == "failed"
    assert record["failure_kind"] == "missing_secret"
    assert record["receipt_id"] is None
    assert record["attempts"] == 1


def test_success_without_id_is_a_failure(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    requests = install_transport(monkeypatch, lambda: httpx.Response(200, json={}))

    delivery = send_push(failed_push(), daily_context())

    assert not delivery.delivered
    assert delivery.failure_kind == "no_receipt"
    assert delivery.attempts == 1
    assert len(requests) == 1
    assert sleeps == []
    (record,) = read_notifications()
    assert record["failure_kind"] == "no_receipt"


def test_non_success_is_a_failure(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    requests = install_transport(monkeypatch, lambda: httpx.Response(400))

    delivery = send_push(failed_push(), daily_context())

    assert not delivery.delivered
    assert delivery.failure_kind == "http_400"
    assert delivery.attempts == 1
    assert len(requests) == 1
    assert sleeps == []
    (record,) = read_notifications()
    assert record["failure_kind"] == "http_400"
    assert record["outcome"] == "failed"


# ---------------------------------------------------------------------------
# D-12: only transient failures are retried, with the documented delays
# ---------------------------------------------------------------------------


def test_transient_5xx_retries_three_times_with_the_documented_delays(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    requests = install_transport(monkeypatch, lambda: httpx.Response(503))

    delivery = send_push(failed_push(), daily_context())

    assert not delivery.delivered
    assert delivery.failure_kind == "http_503"
    assert delivery.attempts == 3
    assert len(requests) == 3
    # A wait before attempts 2 and 3, and never one after the last attempt.
    assert sleeps == [10.0, 30.0]
    (record,) = read_notifications()
    assert record["attempts"] == 3
    assert record["failure_kind"] == "http_503"


def test_429_then_success_is_delivered_on_attempt_two(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    requests = install_transport(monkeypatch, lambda: httpx.Response(429), ntfy_ok())

    delivery = send_push(failed_push(), daily_context())

    assert delivery.delivered
    assert delivery.attempts == 2
    assert delivery.receipt_id == DUMMY_RECEIPT_ID
    assert len(requests) == 2
    assert sleeps == [10.0]
    (record,) = read_notifications()
    assert record["outcome"] == "delivered"
    assert record["attempts"] == 2
    assert record["receipt_id"] == DUMMY_RECEIPT_ID


def test_timeout_is_transient_and_recorded_by_class_only(
    monkeypatch: pytest.MonkeyPatch,
    sleeps: list[float],
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.DEBUG)
    requests = install_transport(
        monkeypatch,
        httpx.ReadTimeout(
            f"timed out reading {DUMMY_PING_URL} for topic {DUMMY_TOPIC}"
        ),
    )

    delivery = send_push(failed_push(), daily_context())

    assert not delivery.delivered
    assert delivery.failure_kind == "transport_ReadTimeout"
    assert delivery.attempts == 3
    assert len(requests) == 3
    assert sleeps == [10.0, 30.0]
    log_text = notification_log.NOTIFICATION_LOG.read_text(encoding="utf-8")
    captured = capsys.readouterr()
    for secret in (DUMMY_PING_URL, DUMMY_PING_UUID, DUMMY_TOPIC):
        assert secret not in log_text
        assert secret not in captured.out
        assert secret not in captured.err
        assert secret not in caplog.text


def test_permanent_4xx_fails_at_once(
    monkeypatch: pytest.MonkeyPatch, sleeps: list[float]
) -> None:
    requests = install_transport(monkeypatch, lambda: httpx.Response(401))

    delivery = send_push(failed_push(), daily_context())

    assert not delivery.delivered
    assert delivery.failure_kind == "http_401"
    assert delivery.attempts == 1
    assert len(requests) == 1
    assert sleeps == []
