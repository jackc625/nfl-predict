"""The Resend email and healthchecks.io watcher channels, and their routers (Phase 35 NOTIF-02, NOTIF-04).

* An email counts as delivered only when Resend answers 2xx AND returns a non-empty id; that id is
  the receipt on the email's one notification-log record.
* Resend is called with a Bearer key, an explicit User-Agent and an Idempotency-Key; a 409 (a retry
  colliding with the in-flight idempotent request) is transient.
* The email router accepts only ``digest`` and ``test``; anything else is refused before a request.
* A healthchecks.io ping is delivered only on HTTP 200 with the body ``OK``: ``OK (not found)`` is a
  permanent failure and ``OK (rate limited)`` a transient one (RESEARCH Pitfall 2).
* The pause POSTs ``/api/v3/checks/<uuid>/pause`` with the read-write key; the uuid comes from the
  ping URL, and a ping URL that is not the uuid form is refused without being echoed (D-01).
* The check-in, the withheld check-in and the pause each leave exactly one log record (D-11).
* The management calls return narrow dataclasses that carry no uuid and no URL (T-35-42).

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
from notify import transport
from notify.channels import (
    WatcherApiError,
    WatcherChannel,
    WatcherCheck,
    check_uuid,
    hc_get_check,
    hc_list_channels,
    hc_pause,
    hc_ping,
    hc_update_check,
)
from notify.events import (
    EmailMessage,
    record_withheld_checkin,
    send_checkin,
    send_email,
    send_pause,
)
from notify.log import NOTIFICATION_FIELDS, read_notifications
from notify.secrets import RESEND_EMAILS_URL, RESEND_SENDER, NotifySecrets
from notify.transport import USER_AGENT
from tests.unit.test_notify_transport import (
    DUMMY_PING_URL,
    DUMMY_PING_UUID,
    daily_context,
    dummy_secrets,
    install_sleep_recorder,
    install_transport,
)

DUMMY_RESEND_KEY = "re_dummyKey0123456789abcdefghij"
DUMMY_RECIPIENT = "owner.dummy@example.invalid"
DUMMY_EMAIL_ID = "4ef9a417-02e9-4d39-ad75-9611e0fcc33c"
DUMMY_HC_API_KEY = "hcw_dummyReadWriteKey0123456789ab"
DUMMY_CHANNEL_A = "aaaaaaaa-0000-4000-8000-000000000001"
DUMMY_CHANNEL_B = "bbbbbbbb-0000-4000-8000-000000000002"


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


# ---------------------------------------------------------------------------
# NOTIF-04: the healthchecks.io check-in, pause and management calls
# ---------------------------------------------------------------------------


def watcher_secrets(**overrides: Any) -> NotifySecrets:
    """Dummy secrets carrying the shared dummy ping URL and a read-write API key."""
    values: dict[str, Any] = {"healthchecks_api_key": SecretStr(DUMMY_HC_API_KEY)}
    values.update(overrides)
    return dummy_secrets(**values)


def text_reply(body: str, status: int = 200) -> Any:
    """A plain-text reply, the shape every healthchecks.io ping answer takes."""
    return lambda: httpx.Response(status, text=body)


def test_healthchecks_ok_is_delivered(monkeypatch: pytest.MonkeyPatch) -> None:
    install_sleep_recorder(monkeypatch)
    requests = install_transport(monkeypatch, text_reply("OK"))

    success = hc_ping(watcher_secrets(), body="run 2026-10-17 ok", fail=False)
    failure = hc_ping(watcher_secrets(), body="run 2026-10-17 failed", fail=True)

    for delivery in (success, failure):
        assert delivery.delivered
        assert delivery.attempts == 1
        assert delivery.receipt_id is None
    first, second = requests
    assert first.method == second.method == "POST"
    assert str(first.url) == DUMMY_PING_URL
    assert first.content == b"run 2026-10-17 ok"
    assert str(second.url) == f"{DUMMY_PING_URL}/fail"
    assert second.content == b"run 2026-10-17 failed"


def test_healthchecks_ok_not_found_is_a_permanent_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps = install_sleep_recorder(monkeypatch)
    requests = install_transport(monkeypatch, text_reply("OK (not found)"))

    delivery = hc_ping(watcher_secrets(), body="run", fail=False)

    assert not delivery.delivered
    assert delivery.failure_kind == "not_found"
    assert delivery.attempts == 1
    assert len(requests) == 1
    assert sleeps == []


def test_healthchecks_ok_rate_limited_is_transient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sleeps = install_sleep_recorder(monkeypatch)
    requests = install_transport(
        monkeypatch, text_reply("OK (rate limited)"), text_reply("OK")
    )

    delivery = hc_ping(watcher_secrets(), body="run", fail=False)

    assert delivery.delivered
    assert delivery.attempts == 2
    assert len(requests) == 2
    assert sleeps == [10.0]


def test_pause_posts_the_uuid_path_with_the_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_sleep_recorder(monkeypatch)
    requests = install_transport(monkeypatch, lambda: httpx.Response(200, json={}))

    delivery = hc_pause(watcher_secrets())

    assert delivery.delivered
    assert delivery.attempts == 1
    (request,) = requests
    assert request.method == "POST"
    assert (
        str(request.url)
        == f"https://healthchecks.io/api/v3/checks/{DUMMY_PING_UUID}/pause"
    )
    assert request.headers["X-Api-Key"] == DUMMY_HC_API_KEY


def test_check_uuid_refuses_a_slug_url() -> None:
    ping_key = "dummyPingKey0123456789"
    slug_url = f"https://hc-ping.com/{ping_key}/nfl-predict-daily"

    assert check_uuid(DUMMY_PING_URL) == DUMMY_PING_UUID
    with pytest.raises(ValueError, match=r"hc-ping\.com/<uuid>") as refused:
        check_uuid(slug_url)

    assert ping_key not in str(refused.value)
    assert "nfl-predict-daily" not in str(refused.value)


def test_each_watcher_send_appends_one_record(monkeypatch: pytest.MonkeyPatch) -> None:
    install_sleep_recorder(monkeypatch)
    requests = install_transport(monkeypatch, text_reply("OK"))
    context = daily_context(watcher_secrets())

    checkin = send_checkin(body="run 2026-10-17 ok", fail=False, context=context)

    assert checkin.delivered
    assert len(requests) == 1
    (checkin_record,) = read_notifications()
    assert checkin_record["event_class"] == "checkin"
    assert checkin_record["outcome"] == "delivered"
    assert checkin_record["priority"] is None

    record_withheld_checkin(context)

    assert len(requests) == 1
    records = read_notifications()
    assert len(records) == 2
    withheld_record = records[-1]
    assert withheld_record["event_class"] == "checkin"
    assert withheld_record["outcome"] == "withheld"
    assert withheld_record["attempts"] == 0
    assert withheld_record["receipt_id"] is None
    assert withheld_record["failure_kind"] is None

    pause_requests = install_transport(
        monkeypatch, lambda: httpx.Response(200, json={})
    )
    pause = send_pause(context)

    assert pause.delivered
    assert len(pause_requests) == 1
    records = read_notifications()
    assert len(records) == 3
    assert records[-1]["event_class"] == "pause"
    assert records[-1]["outcome"] == "delivered"
    for record in records:
        assert set(record) == set(NOTIFICATION_FIELDS)
    log_text = notification_log.NOTIFICATION_LOG.read_text(encoding="utf-8")
    for secret in (DUMMY_PING_URL, DUMMY_PING_UUID, DUMMY_HC_API_KEY):
        assert secret not in log_text


def _full_check_object() -> dict[str, Any]:
    """A v3 check as a read-write key sees it: the uuid and four secret-bearing URLs included."""
    api_check = f"https://healthchecks.io/api/v3/checks/{DUMMY_PING_UUID}"
    return {
        "name": "nfl-predict daily",
        "uuid": DUMMY_PING_UUID,
        "ping_url": DUMMY_PING_URL,
        "update_url": api_check,
        "pause_url": f"{api_check}/pause",
        "resume_url": f"{api_check}/resume",
        "schedule": "0 17 * * *",
        "tz": "America/New_York",
        "grace": 5400,
        "manual_resume": False,
        "channels": f"{DUMMY_CHANNEL_A},{DUMMY_CHANNEL_B}",
        "status": "up",
    }


def test_management_results_carry_no_uuid_or_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx.Request] = []
    check_path = f"/api/v3/checks/{DUMMY_PING_UUID}"

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.method == "GET" and request.url.path == "/api/v3/channels/":
            return httpx.Response(
                200,
                json={
                    "channels": [
                        {"id": DUMMY_CHANNEL_A, "name": "ntfy", "kind": "ntfy"},
                        {"id": DUMMY_CHANNEL_B, "name": "email", "kind": "email"},
                    ]
                },
            )
        if request.method in {"GET", "POST"} and request.url.path == check_path:
            return httpx.Response(200, json=_full_check_object())
        return httpx.Response(404)

    monkeypatch.setattr(
        transport,
        "CLIENT_FACTORY",
        lambda: httpx.Client(transport=httpx.MockTransport(handler)),
    )
    payload = {"schedule": "0 17 * * *", "grace": 5400}
    secrets = watcher_secrets()

    listed = hc_list_channels(secrets)
    fetched = hc_get_check(secrets)
    updated = hc_update_check(secrets, payload)

    assert listed == (
        WatcherChannel(id=DUMMY_CHANNEL_A, name="ntfy", kind="ntfy"),
        WatcherChannel(id=DUMMY_CHANNEL_B, name="email", kind="email"),
    )
    expected = WatcherCheck(
        schedule="0 17 * * *",
        tz="America/New_York",
        grace=5400,
        manual_resume=False,
        channel_ids=frozenset({DUMMY_CHANNEL_A, DUMMY_CHANNEL_B}),
    )
    assert fetched == expected
    assert updated == expected
    for result in (listed, *listed, fetched, updated):
        assert DUMMY_PING_UUID not in repr(result)
        assert "https://" not in repr(result)
    assert len(requests) == 3
    assert all(request.headers["X-Api-Key"] == DUMMY_HC_API_KEY for request in requests)
    assert requests[-1].method == "POST"
    assert json.loads(requests[-1].content) == payload


def test_a_management_refusal_carries_the_status_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_transport(monkeypatch, lambda: httpx.Response(404, text=DUMMY_PING_URL))

    with pytest.raises(WatcherApiError) as refused:
        hc_get_check(watcher_secrets())

    assert refused.value.status == 404
    assert DUMMY_PING_UUID not in str(refused.value)
