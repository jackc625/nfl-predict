"""Negative controls for the suite-wide notification network guard (Phase 35 must-NOT).

The suite must never make a real ntfy, Resend or healthchecks.io request: a test check-in would
make a dead laptop look alive, and a test push would reach the owner's phone. ``tests/conftest.py``
refuses those hosts at ``httpx.HTTPTransport`` for every test. A guard that never fires passes
vacuously, so these tests make it fire -- and prove it leaves ``httpx.MockTransport`` clients and
every other host alone.

No request here leaves the machine: the guard refuses before the transport opens a connection.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from notify import log as notification_log
from tests.conftest import notification_host_refusal

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DUMMY_PING_UUID = "00000000-1111-2222-3333-444444444444"


@pytest.mark.parametrize(
    "host",
    [
        "ntfy.sh",
        "api.resend.com",
        "hc-ping.com",
        "healthchecks.io",
        "www.healthchecks.io",
    ],
)
def test_the_guard_refuses_each_notification_host(host: str) -> None:
    with (
        httpx.Client() as client,
        pytest.raises(pytest.fail.Exception) as refusal,
    ):
        client.post(f"https://{host}/", json={"probe": "refused before sending"})

    assert host in str(refusal.value)


def test_the_refusal_names_the_host_only() -> None:
    with (
        httpx.Client() as client,
        pytest.raises(pytest.fail.Exception) as refusal,
    ):
        client.post(f"https://hc-ping.com/{DUMMY_PING_UUID}", content=b"summary")

    message = str(refusal.value)
    assert "hc-ping.com" in message
    assert DUMMY_PING_UUID not in message


def test_a_mock_transport_is_not_intercepted() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"id": "mocked"})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        response = client.post("https://ntfy.sh/", json={"topic": "mocked"})

    assert response.status_code == 200
    assert len(seen) == 1


def test_a_non_notification_host_is_not_refused() -> None:
    assert notification_host_refusal("example.invalid") is None
    # A host that merely ENDS with a refused name is not a subdomain of it.
    assert notification_host_refusal("notntfy.sh") is None
    assert notification_host_refusal("api.ntfy.sh") is not None


def test_the_notification_log_is_quarantined() -> None:
    log_path = Path(notification_log.NOTIFICATION_LOG).resolve()

    assert not log_path.is_relative_to(REPOSITORY_ROOT)
