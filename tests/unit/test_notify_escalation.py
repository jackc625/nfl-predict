"""When a message is lost, the watcher check-in is the failure signal (Phase 35 SPEC R8, D-13).

* ntfy down: the failed push stays undelivered after its attempts and the check-in goes to
  ``<ping_url>/fail``, so the watcher alerts through its own channels.
* Resend down: the undelivered weekly email sends one ``FAILED: weekly email`` push, then the
  check-in goes to ``/fail``.
* Sending can never change a decision: a whole daily run's ledger and bet-list bytes, and its exit
  code, are identical with every provider up, with ntfy down and with Resend down.
* Nothing is re-sent later: a clean run the next day makes only its own check-in.

Every request goes through ``notify.transport.CLIENT_FACTORY`` replaced by an ``httpx.MockTransport``
(``tests/unit/test_notify_watcher.install_hosts``); every retry delay through a recording ``SLEEP``.
Secrets are dummy values; no real provider is called and every file is under ``tmp_path``.

Run this module:  uv run python -m pytest tests/unit/test_notify_escalation.py -q

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import hashlib
import json
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

import scripts.daily_lock_pipeline as daily
from notify import digest, end_of_run
from notify.end_of_run import notify_end_of_run
from notify.log import read_notifications
from notify.report import STATUS_RUN_FAILED, STATUS_SUCCESS
from tests.unit.test_daily_lock_run import RUN_DATE as DAILY_RUN_DATE
from tests.unit.test_daily_lock_run import (
    _orchestrator_log,
    _pipeline_that,
    _schedule,
    _slate_env,
    _start,
)
from tests.unit.test_notify_channels import resend_ok, watcher_secrets
from tests.unit.test_notify_digest import (
    WEEK6,
    build_cache,
    saturday_report,
    schedule_frame,
    week6_bets,
)
from tests.unit.test_notify_messages import live_bet
from tests.unit.test_notify_transport import DUMMY_PING_URL, ntfy_ok
from tests.unit.test_notify_watcher import (
    TOMORROW,
    every_secret,
    hosts,
    in_season_report,
    install_hosts,
    ping_ok,
)
from utils.current_slate import prepare_schedule


@pytest.fixture(autouse=True)
def _no_sleeps(monkeypatch: pytest.MonkeyPatch) -> None:
    """No retry delay in this module actually sleeps."""
    from notify import transport

    monkeypatch.setattr(transport, "SLEEP", lambda _seconds: None)


def test_ntfy_down_makes_the_check_in_a_failure_signal(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    requests = install_hosts(
        monkeypatch,
        {"ntfy.sh": (lambda: httpx.Response(503),), "hc-ping.com": (ping_ok(),)},
    )
    report = in_season_report(
        status=STATUS_RUN_FAILED,
        exception_class="RuntimeError",
        unpredicted_game_ids=TOMORROW,
    )

    outcome = notify_end_of_run(report, secrets=watcher_secrets(), now=report.start)

    assert hosts(requests) == ["ntfy.sh"] * 3 + ["hc-ping.com"]
    assert outcome.any_undelivered is True
    assert str(requests[-1].url) == f"{DUMMY_PING_URL}/fail"
    assert "failed pushes=0/1" in requests[-1].content.decode("ascii")
    assert "WATCHER_CHECKIN= delivered as failure" in capsys.readouterr().out


def test_resend_down_pushes_and_fails_the_check_in(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    schedule = schedule_frame(WEEK6)
    monkeypatch.setattr(digest, "LOAD_SCHEDULE", lambda: schedule)
    cache_path = tmp_path / "web_cache.duckdb"
    monkeypatch.setattr(digest, "WEB_CACHE_PATH", cache_path)
    report = saturday_report(season_window="in")
    build_cache(
        cache_path, bets=week6_bets(), stamped_at=report.start + timedelta(minutes=12)
    )
    requests = install_hosts(
        monkeypatch,
        {
            "api.resend.com": (lambda: httpx.Response(500),),
            "ntfy.sh": (ntfy_ok(),),
            "hc-ping.com": (ping_ok(),),
        },
    )

    notify_end_of_run(
        report, secrets=every_secret(), now=report.start + timedelta(minutes=20)
    )

    assert hosts(requests) == ["api.resend.com"] * 3 + ["ntfy.sh", "hc-ping.com"]
    (push,) = [request for request in requests if request.url.host == "ntfy.sh"]
    assert json.loads(push.content)["title"] == "FAILED: weekly email"
    assert str(requests[-1].url) == f"{DUMMY_PING_URL}/fail"


# The three transports a whole daily run is driven through.
ALL_UP = {
    "ntfy.sh": (ntfy_ok(),),
    "api.resend.com": (resend_ok(),),
    "hc-ping.com": (ping_ok(),),
}
NTFY_DOWN = {**ALL_UP, "ntfy.sh": (lambda: httpx.Response(503),)}
RESEND_DOWN = {**ALL_UP, "api.resend.com": (lambda: httpx.Response(500),)}


def _decision_run_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> list[Path]:
    """A Saturday daily run (a digest day) whose recommend step writes a ledger and a bet list.

    The real notifier runs at the end over dummy secrets; returns the two decision files.
    """
    from forward_ledger import closing_schedule

    _slate_env(monkeypatch, tmp_path)
    monkeypatch.setattr(
        closing_schedule,
        "register_closing_triggers",
        lambda *_a, **_k: closing_schedule.TriggerRegistration(
            "skipped", "test_stand_in", (), False
        ),
    )
    ledger = tmp_path / "ledger" / "entries.jsonl"
    bet_list = tmp_path / "bet_list_2026_week3.csv"
    row = live_bet("2026_W03_LAC@BUF", week=3, per_bet_ev=0.05)

    def _registry(slate):
        def _recommend() -> None:
            ledger.parent.mkdir(exist_ok=True)
            ledger.write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
            bet_list.write_text(
                "game_id,target,stake_units\n2026_W03_LAC@BUF,ats,1.25\n",
                encoding="utf-8",
            )
            slate.appended_rows.append(row)

        return [SimpleNamespace(name="generate_recommendations", callable=_recommend)]

    monkeypatch.setattr(daily, "build_daily_step_registry", _registry)
    _pipeline_that(monkeypatch, _orchestrator_log(STATUS_SUCCESS))
    monkeypatch.setattr(daily, "SEASON_WINDOW", lambda _d: "in")
    monkeypatch.setattr(daily, "NOTIFY_END_OF_RUN", daily._notify_end_of_run)
    monkeypatch.setattr(end_of_run, "LOAD_SECRETS", every_secret)
    schedule = prepare_schedule(_schedule().assign(game_type="REG"))
    monkeypatch.setattr(digest, "LOAD_SCHEDULE", lambda: schedule)
    monkeypatch.setattr(digest, "WEB_CACHE_PATH", tmp_path / "absent_cache.duckdb")
    return [ledger, bet_list]


def test_decisions_are_byte_identical_with_and_without_failures(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    decision_files = _decision_run_env(monkeypatch, tmp_path)
    results = {}
    for mode, replies in (
        ("all_up", ALL_UP),
        ("ntfy_down", NTFY_DOWN),
        ("resend_down", RESEND_DOWN),
    ):
        for path in decision_files:
            path.unlink(missing_ok=True)
        requests = install_hosts(monkeypatch, replies)

        exit_code = daily.run_daily(DAILY_RUN_DATE, start=_start(), dry_run=False)

        digests = tuple(
            hashlib.sha256(path.read_bytes()).hexdigest() for path in decision_files
        )
        results[mode] = (exit_code, digests)
        checkin = requests[-1]
        assert checkin.url.host == "hc-ping.com"
        expect_fail = mode != "all_up"
        assert str(checkin.url).endswith("/fail") is expect_fail, mode

    assert results["all_up"] == results["ntfy_down"] == results["resend_down"]
    assert results["all_up"][0] == 0


def test_a_later_run_sends_nothing_for_an_earlier_day(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_hosts(
        monkeypatch,
        {
            "ntfy.sh": (lambda: httpx.Response(503),),
            "hc-ping.com": (lambda: httpx.Response(503),),
        },
    )
    day_one = in_season_report(
        status=STATUS_RUN_FAILED,
        exception_class="RuntimeError",
        unpredicted_game_ids=TOMORROW,
    )
    notify_end_of_run(day_one, secrets=watcher_secrets(), now=day_one.start)
    day_one_records = read_notifications()
    assert {record["outcome"] for record in day_one_records} == {"failed"}

    requests = install_hosts(monkeypatch, {"hc-ping.com": (ping_ok(),)})
    day_two = in_season_report(
        run_date_et=day_one.run_date_et + timedelta(days=1),
        start=day_one.start + timedelta(days=1),
        lock=day_one.lock + timedelta(days=1),
    )
    notify_end_of_run(day_two, secrets=watcher_secrets(), now=day_two.start)

    (checkin,) = requests
    assert str(checkin.url) == DUMMY_PING_URL
    records = read_notifications()
    assert records[: len(day_one_records)] == day_one_records
    (added,) = records[len(day_one_records) :]
    assert (added["event_class"], added["run_date_et"]) == (
        "checkin",
        day_two.run_date_et.isoformat(),
    )
