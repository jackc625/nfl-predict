"""The outside watcher hears from every in-season daily run (Phase 35 NOTIF-04, SPEC R7, D-13, D-15).

* Every in-season run checks in ONCE, LAST -- after every message -- with a short ASCII run summary
  as the ping body, never a secret (D-15).
* A run whose messages were all delivered checks in normally, a delivered failed push included; a
  message that stayed undelivered, or a delivered one whose receipt never reached the notification
  log, sends the check-in to ``<ping_url>/fail`` (SPEC R8; T-35-47).

Every request goes through ``notify.transport.CLIENT_FACTORY`` replaced by an ``httpx.MockTransport``
that answers per host and records every request in send order; every retry delay goes through a
recording ``SLEEP``. Secrets are dummy values; no real provider is called. The suite-wide
``tests/conftest.py`` fixtures keep the digest step on "not a digest day".

Run this module:  uv run python -m pytest tests/unit/test_notify_watcher.py -q

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import timedelta
from typing import Any

import httpx
import pytest

from notify import log as notification_log
from notify import transport
from notify.end_of_run import notify_end_of_run
from notify.log import read_notifications
from notify.report import STATUS_RUN_FAILED, STATUS_SUCCESS, RunReport
from notify.secrets import NotifySecrets
from pipeline.daily_steps import slate_lock
from tests.unit.test_notify_channels import email_secrets, watcher_secrets
from tests.unit.test_notify_messages import live_bet
from tests.unit.test_notify_transport import (
    DUMMY_PING_URL,
    RUN_DATE,
    install_sleep_recorder,
    ntfy_ok,
)
from utils.file_lock import FileLockHeldError, exclusive_file_lock

TOMORROW = ("2026_06_CIN@CLE", "2026_06_DAL@PHI")

Reply = Callable[[], httpx.Response] | Exception


def ping_ok() -> Callable[[], httpx.Response]:
    """healthchecks.io's answer to a ping that did something: HTTP 200 with the body ``OK``."""
    return lambda: httpx.Response(200, text="OK")


def install_hosts(
    monkeypatch: pytest.MonkeyPatch, by_host: dict[str, tuple[Reply, ...]]
) -> list[httpx.Request]:
    """Answer each provider host from its own reply list; return EVERY request in send order.

    The last reply of each list repeats once it runs out. A request to a host with no list fails
    the test.
    """
    seen: list[httpx.Request] = []
    counts: dict[str, int] = dict.fromkeys(by_host, 0)

    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        seen.append(request)
        if host not in by_host:
            pytest.fail(f"unexpected request to {host}", pytrace=False)
        counts[host] += 1
        options = by_host[host]
        reply = options[min(counts[host], len(options)) - 1]
        if isinstance(reply, Exception):
            raise reply
        return reply()

    monkeypatch.setattr(
        transport,
        "CLIENT_FACTORY",
        lambda: httpx.Client(transport=httpx.MockTransport(handler)),
    )
    return seen


def hosts(requests: list[httpx.Request]) -> list[str]:
    """The host of each request, in send order."""
    return [request.url.host for request in requests]


def in_season_report(**fields: Any) -> RunReport:
    """An in-season run for RUN_DATE that started an hour before its slate's lock."""
    lock = slate_lock(RUN_DATE)
    values: dict[str, Any] = {
        "run_date_et": RUN_DATE,
        "start": lock - timedelta(hours=1),
        "lock": lock,
        "status": STATUS_SUCCESS,
        "tomorrow_game_ids": TOMORROW,
        "unpredicted_game_ids": (),
        "slate_season": 2026,
        "slate_week": 6,
        "season_window": "in",
    }
    values.update(fields)
    return RunReport(**values)


def every_secret() -> NotifySecrets:
    """Dummy values for all five notification secrets."""
    return email_secrets(healthchecks_api_key=watcher_secrets().healthchecks_api_key)


@pytest.fixture(autouse=True)
def _no_sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """No retry delay in this module actually sleeps."""
    return install_sleep_recorder(monkeypatch)


# ---------------------------------------------------------------------------
# Task 1 (tracer): an in-season run checks in once, last, with an ASCII run summary
# ---------------------------------------------------------------------------


def test_an_in_season_success_checks_in_last_with_a_summary(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    requests = install_hosts(monkeypatch, {"hc-ping.com": (ping_ok(),)})
    report = in_season_report()

    notify_end_of_run(report, secrets=watcher_secrets(), now=report.start)

    (ping,) = requests
    assert requests[-1] is ping
    assert ping.method == "POST"
    assert str(ping.url) == DUMMY_PING_URL
    body = ping.content.decode("ascii")
    assert "outcome=success" in body
    assert f"NFL Predict daily run {RUN_DATE.isoformat()}" in body
    assert "games tomorrow=2" in body
    (record,) = read_notifications()
    assert record["event_class"] == "checkin"
    assert record["outcome"] == "delivered"
    assert "WATCHER_CHECKIN= delivered" in capsys.readouterr().out


def test_a_failed_run_whose_push_was_delivered_checks_in_normally(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    requests = install_hosts(
        monkeypatch, {"ntfy.sh": (ntfy_ok(),), "hc-ping.com": (ping_ok(),)}
    )
    report = in_season_report(
        status=STATUS_RUN_FAILED,
        exception_class="RuntimeError",
        unpredicted_game_ids=TOMORROW,
    )

    outcome = notify_end_of_run(report, secrets=watcher_secrets(), now=report.start)

    assert hosts(requests) == ["ntfy.sh", "hc-ping.com"]
    assert json.loads(requests[0].content)["title"].startswith("FAILED:")
    assert str(requests[-1].url) == DUMMY_PING_URL
    assert outcome.any_undelivered is False
    body = requests[-1].content.decode("ascii")
    assert "outcome=run_failed" in body
    assert "failed pushes=1/1" in body
    assert [r["event_class"] for r in read_notifications()] == ["failed", "checkin"]
    assert "WATCHER_CHECKIN= delivered\n" in capsys.readouterr().out


def test_the_summary_carries_no_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    requests = install_hosts(
        monkeypatch, {"ntfy.sh": (ntfy_ok(),), "hc-ping.com": (ping_ok(),)}
    )
    secrets = every_secret()
    report = in_season_report(
        appended_rows=[live_bet(TOMORROW[0], per_bet_ev=0.05)],
    )

    notify_end_of_run(report, secrets=secrets, now=report.start)

    body = requests[-1].content
    assert len(body) <= 1000
    text = body.decode("ascii")
    assert "bets appended=1" in text
    assert "bets push=delivered" in text
    assert "digest=none" in text
    for secret in secrets.secret_values():
        assert secret not in text


def test_an_unlogged_receipt_fails_the_check_in(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    requests = install_hosts(
        monkeypatch, {"ntfy.sh": (ntfy_ok(),), "hc-ping.com": (ping_ok(),)}
    )
    locks_taken: list[int] = []

    def _bets_record_lock_never_frees(*args: Any, **kwargs: Any) -> Any:
        locks_taken.append(1)
        if len(locks_taken) == 1:  # the bets record; the check-in record is written
            msg = "the lock is held by another writer (waited 60s)."
            raise FileLockHeldError(msg)
        return exclusive_file_lock(*args, **kwargs)

    monkeypatch.setattr(
        notification_log, "exclusive_file_lock", _bets_record_lock_never_frees
    )
    report = in_season_report(appended_rows=[live_bet(TOMORROW[0], per_bet_ev=0.05)])

    outcome = notify_end_of_run(report, secrets=watcher_secrets(), now=report.start)

    assert (outcome.any_undelivered, outcome.any_unlogged) == (False, True)
    assert hosts(requests) == ["ntfy.sh", "hc-ping.com"]
    assert str(requests[-1].url) == f"{DUMMY_PING_URL}/fail"
    assert "unlogged receipts=1" in requests[-1].content.decode("ascii")
    assert "WATCHER_CHECKIN= delivered as failure" in capsys.readouterr().out
