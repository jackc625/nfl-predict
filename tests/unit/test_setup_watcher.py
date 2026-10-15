"""The update-only watcher setup and its field-by-field read-back (Phase 35 D-02, D-22, SPEC R7).

``scripts/setup_watcher.py`` finds the two owner-created integrations by NAME, updates the existing
check (schedule ``0 17 * * *``, tz ``America/New_York``, grace 5400, ``manual_resume`` false, both
channels), reads it back and prints one ``<field>= <want> | <got> | MATCH|MISMATCH`` line per field
and ``WATCHER_READBACK_MATCH=``. It never creates a check, refuses by name when an integration or the
check is missing, and prints no secret, uuid, channel id, URL or raw response.

Every request goes through ``notify.transport.CLIENT_FACTORY`` replaced by an ``httpx.MockTransport``
client, and the secrets are dummy values injected through the ``LOAD_SECRETS`` seam; the owner's
``.env`` is never read and healthchecks.io is never reached (the conftest guard would fail the test).

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from notify import transport
from notify.secrets import NotifySecrets
from scripts import setup_watcher
from tests.unit.test_notify_channels import (
    DUMMY_CHANNEL_A,
    DUMMY_CHANNEL_B,
    DUMMY_HC_API_KEY,
    watcher_secrets,
)
from tests.unit.test_notify_transport import DUMMY_PING_URL, DUMMY_PING_UUID

CHECK_PATH = f"/api/v3/checks/{DUMMY_PING_UUID}"
CREATE_PATH = "/api/v3/checks/"
CHANNELS_PATH = "/api/v3/channels/"

BOTH_CHANNELS = [
    {"id": DUMMY_CHANNEL_A, "name": "NFL Predict ntfy", "kind": "ntfy"},
    {"id": DUMMY_CHANNEL_B, "name": "NFL Predict email", "kind": "email"},
]


def check_body(**overrides: Any) -> dict[str, Any]:
    """A read-write check object: the read-back fields plus the uuid-bearing URLs it really carries."""
    body: dict[str, Any] = {
        "name": "NFL Predict daily run",
        "schedule": "0 17 * * *",
        "tz": "America/New_York",
        "grace": 5400,
        "manual_resume": False,
        "channels": f"{DUMMY_CHANNEL_B},{DUMMY_CHANNEL_A}",
        "uuid": DUMMY_PING_UUID,
        "ping_url": DUMMY_PING_URL,
        "update_url": f"https://healthchecks.io{CHECK_PATH}",
        "pause_url": f"https://healthchecks.io{CHECK_PATH}/pause",
    }
    body.update(overrides)
    return body


def install_api(
    monkeypatch: pytest.MonkeyPatch,
    *,
    channels: list[dict[str, str]] | None = None,
    list_status: int = 200,
    update_status: int = 200,
    check: dict[str, Any] | None = None,
) -> list[httpx.Request]:
    """A MockTransport healthchecks.io Management API; returns every request it saw."""
    requests: list[httpx.Request] = []
    listed = BOTH_CHANNELS if channels is None else channels
    answered = check_body() if check is None else check

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        path = request.url.path
        if request.method == "GET" and path == CHANNELS_PATH:
            return httpx.Response(list_status, json={"channels": listed})
        if request.method == "POST" and path == CHECK_PATH:
            return httpx.Response(update_status, json=answered)
        if request.method == "GET" and path == CHECK_PATH:
            return httpx.Response(200, json=answered)
        pytest.fail(f"unexpected request {request.method} {path}")

    monkeypatch.setattr(
        transport,
        "CLIENT_FACTORY",
        lambda: httpx.Client(transport=httpx.MockTransport(handler)),
    )
    return requests


def use_secrets(
    monkeypatch: pytest.MonkeyPatch,
    factory: Callable[[], NotifySecrets] = watcher_secrets,
) -> None:
    monkeypatch.setattr(setup_watcher, "LOAD_SECRETS", factory)


def posts(requests: list[httpx.Request]) -> list[httpx.Request]:
    return [request for request in requests if request.method == "POST"]


def test_a_matching_check_reads_back_true(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    use_secrets(monkeypatch)
    requests = install_api(monkeypatch)

    code = setup_watcher.main([])

    lines = capsys.readouterr().out.splitlines()
    assert code == 0
    rows = [
        line for line in lines if line.split("=", 1)[0] in setup_watcher.READBACK_FIELDS
    ]
    assert [row.split("=", 1)[0] for row in rows] == list(setup_watcher.READBACK_FIELDS)
    assert all(row.endswith("| MATCH") for row in rows)
    assert "WATCHER_READBACK_MATCH= True" in lines
    (update,) = posts(requests)
    assert update.url.path == CHECK_PATH
    assert update.headers["X-Api-Key"] == DUMMY_HC_API_KEY
    assert json.loads(update.content) == {
        "schedule": "0 17 * * *",
        "tz": "America/New_York",
        "grace": 5400,
        "manual_resume": False,
        "channels": f"{DUMMY_CHANNEL_A},{DUMMY_CHANNEL_B}",
    }


def test_a_drifted_field_exits_non_zero_naming_it(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    use_secrets(monkeypatch)
    install_api(monkeypatch, check=check_body(grace=3600))

    code = setup_watcher.main([])

    lines = capsys.readouterr().out.splitlines()
    assert code == 1
    assert "grace= 5400 | 3600 | MISMATCH" in lines
    assert "WATCHER_READBACK_MATCH= False" in lines


def test_a_missing_integration_is_refused_by_name(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    use_secrets(monkeypatch)
    requests = install_api(monkeypatch, channels=BOTH_CHANNELS[:1])

    code = setup_watcher.main([])

    out = capsys.readouterr().out
    assert code == 1
    refusal = next(
        line for line in out.splitlines() if "WATCHER_SETUP_REFUSED=" in line
    )
    assert "NFL Predict email" in refusal
    assert "web UI" in refusal
    assert posts(requests) == []


def test_an_unknown_check_is_refused(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    use_secrets(monkeypatch)
    requests = install_api(monkeypatch, update_status=404)

    code = setup_watcher.main([])

    out = capsys.readouterr().out
    assert code == 1
    refusal = next(
        line for line in out.splitlines() if "WATCHER_SETUP_REFUSED=" in line
    )
    assert "HEALTHCHECKS_PING_URL" in refusal
    assert "not found" in refusal
    assert "web UI" in refusal
    assert all(request.url.path != CREATE_PATH for request in posts(requests))


@pytest.mark.parametrize("status", [401, 403])
def test_a_read_only_key_is_refused(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], status: int
) -> None:
    use_secrets(monkeypatch)
    install_api(monkeypatch, list_status=status)

    code = setup_watcher.main([])

    out = capsys.readouterr().out
    assert code == 1
    refusal = next(
        line for line in out.splitlines() if "WATCHER_SETUP_REFUSED=" in line
    )
    assert "read-write" in refusal


@pytest.mark.parametrize(
    "secrets",
    [
        NotifySecrets(healthchecks_ping_url=SecretStr(DUMMY_PING_URL)),
        NotifySecrets(healthchecks_api_key=SecretStr(DUMMY_HC_API_KEY)),
        NotifySecrets(),
    ],
)
def test_missing_secrets_exit_2(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    secrets: NotifySecrets,
) -> None:
    use_secrets(monkeypatch, lambda: secrets)
    requests = install_api(monkeypatch)

    code = setup_watcher.main([])

    out = capsys.readouterr().out
    assert code == 2
    assert (
        "WATCHER_SETUP_REFUSED= HEALTHCHECKS_PING_URL and HEALTHCHECKS_API_KEY must both "
        "be set in .env" in out
    )
    assert requests == []


def test_verify_only_reads_back_without_updating(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    use_secrets(monkeypatch)
    requests = install_api(monkeypatch)

    code = setup_watcher.main(["--verify-only"])

    assert code == 0
    assert "WATCHER_READBACK_MATCH= True" in capsys.readouterr().out
    assert posts(requests) == []
    assert [request.url.path for request in requests] == [CHANNELS_PATH, CHECK_PATH]


def test_nothing_secret_is_printed(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    use_secrets(monkeypatch)
    install_api(monkeypatch, check=check_body(channels=f"{DUMMY_CHANNEL_A}"))
    assert setup_watcher.main([]) == 1
    install_api(monkeypatch, update_status=404)
    assert setup_watcher.main([]) == 1

    captured = capsys.readouterr()
    printed = captured.out + captured.err
    for secret in (
        DUMMY_PING_URL,
        DUMMY_PING_UUID,
        DUMMY_HC_API_KEY,
        DUMMY_CHANNEL_A,
        DUMMY_CHANNEL_B,
        "https://",
        "{",
    ):
        assert secret not in printed
    assert "NFL Predict ntfy" in printed
    assert "NFL Predict email" in printed
