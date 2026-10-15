"""The Resend email and healthchecks.io watcher channels, and their routers (Phase 35 NOTIF-02, NOTIF-04).

* An email counts as delivered only when Resend answers 2xx AND returns a non-empty id; that id is
  the receipt on the email's one notification-log record.
* Resend is called with a Bearer key, an explicit User-Agent and an Idempotency-Key; a 409 (a retry
  colliding with the in-flight idempotent request) is transient.
* The email router accepts only ``digest`` and ``test``; anything else is refused before a request.

Every request goes through ``notify.transport.CLIENT_FACTORY`` replaced by an ``httpx.MockTransport``
client and every retry delay through a recording ``SLEEP`` (the helpers come from
``tests/unit/test_notify_transport.py``). Secrets are dummy values; the owner's ``.env`` is never read.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from notify import log as notification_log
from notify.events import EmailMessage, send_email
from notify.log import NOTIFICATION_FIELDS, read_notifications
from notify.secrets import RESEND_EMAILS_URL, RESEND_SENDER, NotifySecrets
from notify.transport import USER_AGENT
from tests.unit.test_notify_transport import (
    daily_context,
    dummy_secrets,
    install_sleep_recorder,
    install_transport,
)

DUMMY_RESEND_KEY = "re_dummyKey0123456789abcdefghij"
DUMMY_RECIPIENT = "owner.dummy@example.invalid"
DUMMY_EMAIL_ID = "4ef9a417-02e9-4d39-ad75-9611e0fcc33c"


def email_secrets(**overrides: Any) -> NotifySecrets:
    """Dummy secrets carrying a Resend key and a recipient."""
    values: dict[str, Any] = {
        "resend_api_key": SecretStr(DUMMY_RESEND_KEY),
        "digest_to_address": SecretStr(DUMMY_RECIPIENT),
    }
    values.update(overrides)
    return dummy_secrets(**values)


def digest_email(
    event_class: str = "digest", idempotency_key: str = "k1"
) -> EmailMessage:
    """A Saturday digest email."""
    return EmailMessage(
        event_class=event_class,
        subject="NFL Predict: 2 bets this week",
        html="<p>Two bets this week.</p>",
        text="Two bets this week.",
        idempotency_key=idempotency_key,
    )


def resend_ok(email_id: str = DUMMY_EMAIL_ID) -> Any:
    """A Resend success reply carrying the email id."""
    return lambda: httpx.Response(200, json={"id": email_id})


# ---------------------------------------------------------------------------
# NOTIF-02: the Resend email
# ---------------------------------------------------------------------------


def test_resend_send_carries_headers_and_logs_the_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps = install_sleep_recorder(monkeypatch)
    requests = install_transport(monkeypatch, resend_ok())

    delivery = send_email(digest_email(), daily_context(email_secrets()))

    assert delivery.delivered
    assert delivery.attempts == 1
    assert delivery.receipt_id == DUMMY_EMAIL_ID
    assert delivery.logged is True
    assert sleeps == []

    (request,) = requests
    assert request.method == "POST"
    assert str(request.url) == RESEND_EMAILS_URL == "https://api.resend.com/emails"
    assert request.headers["Authorization"] == f"Bearer {DUMMY_RESEND_KEY}"
    assert request.headers["User-Agent"] == USER_AGENT == "nfl-predict-notify/1"
    assert request.headers["Idempotency-Key"] == "k1"
    body = json.loads(request.content)
    assert body == {
        "from": RESEND_SENDER,
        "to": [DUMMY_RECIPIENT],
        "subject": "NFL Predict: 2 bets this week",
        "html": "<p>Two bets this week.</p>",
        "text": "Two bets this week.",
    }
    assert "onboarding@resend.dev" in body["from"]

    lines = notification_log.NOTIFICATION_LOG.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert set(record) == set(NOTIFICATION_FIELDS)
    assert record["event_class"] == "digest"
    assert record["priority"] is None
    assert record["outcome"] == "delivered"
    assert record["receipt_id"] == DUMMY_EMAIL_ID
    assert record["attempts"] == 1
    assert DUMMY_RESEND_KEY not in lines[0]
    assert DUMMY_RECIPIENT not in lines[0]


def test_resend_409_is_transient(monkeypatch: pytest.MonkeyPatch) -> None:
    sleeps = install_sleep_recorder(monkeypatch)
    requests = install_transport(monkeypatch, lambda: httpx.Response(409), resend_ok())

    delivery = send_email(digest_email(), daily_context(email_secrets()))

    assert delivery.delivered
    assert delivery.attempts == 2
    assert delivery.receipt_id == DUMMY_EMAIL_ID
    assert sleeps == [10.0]
    assert [request.headers["Idempotency-Key"] for request in requests] == ["k1", "k1"]
    (record,) = read_notifications()
    assert record["attempts"] == 2
    assert record["outcome"] == "delivered"


def test_resend_success_without_id_is_a_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps = install_sleep_recorder(monkeypatch)
    requests = install_transport(monkeypatch, lambda: httpx.Response(200, json={}))

    delivery = send_email(digest_email(), daily_context(email_secrets()))

    assert not delivery.delivered
    assert delivery.failure_kind == "no_receipt"
    assert delivery.attempts == 1
    assert len(requests) == 1
    assert sleeps == []
    (record,) = read_notifications()
    assert record["failure_kind"] == "no_receipt"
    assert record["receipt_id"] is None


@pytest.mark.parametrize("unset", ["resend_api_key", "digest_to_address"])
def test_a_missing_key_or_recipient_is_a_recorded_failure(
    monkeypatch: pytest.MonkeyPatch, unset: str
) -> None:
    install_sleep_recorder(monkeypatch)
    requests = install_transport(monkeypatch, resend_ok())

    delivery = send_email(digest_email(), daily_context(email_secrets(**{unset: None})))

    assert requests == []
    assert not delivery.delivered
    assert delivery.failure_kind == "missing_secret"
    assert delivery.attempts == 1
    (record,) = read_notifications()
    assert record["event_class"] == "digest"
    assert record["outcome"] == "failed"
    assert record["failure_kind"] == "missing_secret"


def test_the_email_router_refuses_a_push_class(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_sleep_recorder(monkeypatch)
    requests = install_transport(monkeypatch, resend_ok())

    with pytest.raises(ValueError, match="closed vocabulary"):
        send_email(digest_email(event_class="failed"), daily_context(email_secrets()))

    assert len(requests) == 0
    assert read_notifications() == []
