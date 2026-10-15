"""The owner's TEST command fires every channel and never checks in (Phase 35 SPEC R9).

* One run sends a TEST failed push (priority 5), a TEST bets push (priority 3) and a TEST digest
  email (priority None) through the real sender, prints each receipt id on its own
  ``NOTIFY_TEST_*=`` line and exits 0. The secrets reach it through the real ``Settings`` ->
  ``load_secrets`` bridge, from a ``tmp_path`` ``.env``.
* Any message that is not delivered makes the command exit 1, naming the failure kind; the other
  messages still go out.
* With no secret configured, all three are recorded ``missing_secret`` failures and no request is
  made.
* No case ever reaches hc-ping.com or healthchecks.io: the TEST command makes no check-in.
* With no web cache the TEST digest still goes out, carrying the stale notice.

Every request goes through ``notify.transport.CLIENT_FACTORY`` replaced by an ``httpx.MockTransport``
that answers per host (``tests/unit/test_notify_watcher.install_hosts``; any other host fails the
test) and every retry delay through a recording ``SLEEP``. The web cache is a ``tmp_path`` DuckDB
file behind ``scripts.notify_test.CACHE_PATH``. Secrets are dummy values; the owner's ``.env`` is
never read and no real provider is called.

Run this module:  uv run python -m pytest tests/unit/test_notify_test_command.py -q

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from conf.settings import Settings
from notify import digest
from notify.log import read_notifications
from notify.secrets import NotifySecrets, load_secrets
from scripts import notify_test
from tests.unit.test_notify_channels import resend_ok
from tests.unit.test_notify_digest import build_cache, stamp_current_slate, week6_bets
from tests.unit.test_notify_transport import install_sleep_recorder, ntfy_ok
from tests.unit.test_notify_watcher import (
    CHECKIN_FUNCTIONS,
    REPO_ROOT,
    hosts,
    install_hosts,
)
from tests.unit.test_settings_notify_secrets import SECRET_FIELDS

FAILED_ID = "testFailedId01"
BETS_ID = "testBetsId02"
DIGEST_ID = "5d1c0a3e-7b8f-4c2d-9e6a-1f0b2c3d4e5f"
WATCHER_HOSTS = frozenset({"hc-ping.com", "healthchecks.io"})


@pytest.fixture(autouse=True)
def _no_sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """No retry delay in this module actually sleeps."""
    return install_sleep_recorder(monkeypatch)


@pytest.fixture
def env_secrets(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> NotifySecrets:
    """The five dummy secrets read from a tmp ``.env`` through ``Settings`` and ``load_secrets``.

    The command's ``LOAD_SECRETS`` seam is pointed at that bridge, so the real ``.env`` is never
    read. The process environment beats the env file in pydantic-settings, so each name is cleared.
    """
    for env_name in SECRET_FIELDS:
        monkeypatch.delenv(env_name, raising=False)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "".join(f"{name}={dummy}\n" for name, (_, dummy) in SECRET_FIELDS.items()),
        encoding="utf-8",
    )
    secrets = load_secrets(Settings(_env_file=env_file))
    monkeypatch.setattr(notify_test, "LOAD_SECRETS", lambda: secrets)
    return secrets


@pytest.fixture
def cache_path(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Where the command reads its web cache (a tmp file, absent until a test builds it)."""
    path = tmp_path / "web_cache.duckdb"
    monkeypatch.setattr(notify_test, "CACHE_PATH", path)
    return path


def week6_cache(path: Path) -> Path:
    """A cache whose current slate is 2026 week 6: three live bets, stamped as populated."""
    build_cache(
        path,
        bets=week6_bets(),
        stamped_at=datetime.now(UTC) - timedelta(hours=2),
    )
    stamp_current_slate(path, 2026, 6)
    return path


def _ntfy_titles(requests: list[httpx.Request]) -> list[str]:
    return [
        json.loads(request.content)["title"]
        for request in requests
        if request.url.host == "ntfy.sh"
    ]


def _sent_email(requests: list[httpx.Request]) -> dict[str, object]:
    (request,) = [r for r in requests if r.url.host == "api.resend.com"]
    return json.loads(request.content)


# ---------------------------------------------------------------------------
# Task 1 (tracer): three TEST messages, three receipts, exit 0, no check-in
# ---------------------------------------------------------------------------


def test_three_test_messages_print_three_receipts(
    monkeypatch: pytest.MonkeyPatch,
    env_secrets: NotifySecrets,
    cache_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    week6_cache(cache_path)
    requests = install_hosts(
        monkeypatch,
        {
            "ntfy.sh": (ntfy_ok(FAILED_ID), ntfy_ok(BETS_ID)),
            "api.resend.com": (resend_ok(DIGEST_ID),),
        },
    )

    assert notify_test.main([]) == 0

    out = capsys.readouterr().out
    assert f"NOTIFY_TEST_FAILED= delivered {FAILED_ID}" in out
    assert f"NOTIFY_TEST_BETS= delivered {BETS_ID}" in out
    assert f"NOTIFY_TEST_DIGEST= delivered {DIGEST_ID}" in out

    assert hosts(requests) == ["ntfy.sh", "ntfy.sh", "api.resend.com"]
    titles = _ntfy_titles(requests)
    assert all(title.startswith("TEST - ") for title in titles)
    assert titles == [
        "TEST - FAILED: notification test",
        "TEST - 3 new bets: notification test",
    ]
    bets_body = json.loads(requests[1].content)["message"]
    assert bets_body.index("DAL @ PHI") < bets_body.index("CIN @ CLE")
    sent = _sent_email(requests)
    assert str(sent["subject"]).startswith("TEST - ")
    assert sent["subject"] == "TEST - NFL Predict - 2026 week 6 - 3 bets - run clean"

    records = read_notifications()
    assert len(records) == 3
    assert all(record["source"] == "test_command" for record in records)
    assert all(record["event_class"] == "test" for record in records)
    assert [(record["receipt_id"], record["priority"]) for record in records] == [
        (FAILED_ID, 5),
        (BETS_ID, 3),
        (DIGEST_ID, None),
    ]


def test_any_failure_exits_non_zero(
    monkeypatch: pytest.MonkeyPatch,
    env_secrets: NotifySecrets,
    cache_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    week6_cache(cache_path)
    requests = install_hosts(
        monkeypatch,
        {
            "ntfy.sh": (ntfy_ok(FAILED_ID), ntfy_ok(BETS_ID)),
            "api.resend.com": (lambda: httpx.Response(400, json={"message": "bad"}),),
        },
    )

    assert notify_test.main([]) == 1

    out = capsys.readouterr().out
    assert "NOTIFY_TEST_DIGEST= failed http_400" in out
    assert f"NOTIFY_TEST_FAILED= delivered {FAILED_ID}" in out
    assert f"NOTIFY_TEST_BETS= delivered {BETS_ID}" in out
    assert hosts(requests).count("ntfy.sh") == 2


def test_missing_secrets_fail_loudly(
    monkeypatch: pytest.MonkeyPatch,
    cache_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(notify_test, "LOAD_SECRETS", NotifySecrets)
    requests = install_hosts(monkeypatch, {})

    assert notify_test.main([]) == 1

    out = capsys.readouterr().out
    for name in ("FAILED", "BETS", "DIGEST"):
        assert f"NOTIFY_TEST_{name}= failed missing_secret" in out
    assert requests == []
    assert [record["failure_kind"] for record in read_notifications()] == [
        "missing_secret"
    ] * 3


@pytest.mark.parametrize(
    "case",
    ["all_delivered", "resend_refused", "ntfy_unreachable", "no_secrets"],
)
def test_no_check_in_is_made(
    monkeypatch: pytest.MonkeyPatch,
    env_secrets: NotifySecrets,
    cache_path: Path,
    case: str,
) -> None:
    week6_cache(cache_path)
    ntfy_replies = (
        (httpx.ConnectError("unreachable"),)
        if case == "ntfy_unreachable"
        else (ntfy_ok(FAILED_ID), ntfy_ok(BETS_ID))
    )
    resend_reply = (
        (lambda: httpx.Response(422, json={"message": "no"}))
        if case == "resend_refused"
        else resend_ok(DIGEST_ID)
    )
    if case == "no_secrets":
        monkeypatch.setattr(notify_test, "LOAD_SECRETS", NotifySecrets)
    requests = install_hosts(
        monkeypatch, {"ntfy.sh": ntfy_replies, "api.resend.com": (resend_reply,)}
    )

    notify_test.main([])

    assert not set(hosts(requests)) & WATCHER_HOSTS
    assert all(record["event_class"] == "test" for record in read_notifications())

    tree = ast.parse(
        (REPO_ROOT / "scripts" / "notify_test.py").read_text(encoding="utf-8")
    )
    named = {
        alias.name.rsplit(".", 1)[-1]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import | ast.ImportFrom)
        for alias in node.names
    } | {
        node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute | ast.Name)
    }
    assert not named & CHECKIN_FUNCTIONS


def test_an_absent_cache_still_sends_a_test_digest(
    monkeypatch: pytest.MonkeyPatch,
    env_secrets: NotifySecrets,
    cache_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert not cache_path.exists()
    requests = install_hosts(
        monkeypatch,
        {
            "ntfy.sh": (ntfy_ok(FAILED_ID), ntfy_ok(BETS_ID)),
            "api.resend.com": (resend_ok(DIGEST_ID),),
        },
    )

    assert notify_test.main([]) == 0

    assert f"NOTIFY_TEST_DIGEST= delivered {DIGEST_ID}" in capsys.readouterr().out
    assert _ntfy_titles(requests)[1] == "TEST - bets unknown: notification test"
    sent = _sent_email(requests)
    assert "bets unknown" in str(sent["subject"])
    for part in (str(sent["html"]), str(sent["text"])):
        assert digest.STALE_NOTICE in part
        assert "0 bets" not in part
