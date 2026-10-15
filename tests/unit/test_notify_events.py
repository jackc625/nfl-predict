"""The closed push router's must-NOTs (Phase 35 SPEC must-NOTs, R1/R2 edges, T-35-47).

* Only ``failed``, ``bets`` and ``test`` may be pushed: any other class is refused before a request.
* Nothing suppresses a message: two identical failures are two deliveries and two log records.
* Records land in send order.
* A delivered push whose log record could not be written is delivered but visibly unlogged.
* Plan 35-07: every SPEC R2 outcome maps to exactly the right number and priority of failed
  pushes, and a crash before the orchestrator reaches ntfy as one top-priority push, logged once.

The MockTransport, sleep-recorder and dummy-secret helpers come from
``tests/unit/test_notify_transport.py``; Plan 35-08 extends this file.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import timedelta
from typing import Any

import pytest

from forward_ledger.closing_schedule import TriggerRegistration
from forward_ledger.runner import SettleOutcome
from forward_ledger.sync import BACKUP_REPO_MISSING, SyncOutcome
from notify import log as notification_log
from notify.events import PRIORITY_NORMAL, PRIORITY_TOP, PushMessage, send_push
from notify.log import read_notifications
from notify.messages import RERUN_COMMAND, failed_messages
from notify.report import (
    STATUS_DEGRADED,
    STATUS_FAILED,
    STATUS_FINISHED_WITH_SKIPS,
    STATUS_LOCK_PASSED,
    STATUS_NO_GAMES,
    STATUS_OFFSEASON,
    STATUS_RUN_FAILED,
    STATUS_SUCCESS,
    RunReport,
)
from pipeline.daily_steps import slate_lock
from tests.unit.test_notify_transport import (
    DUMMY_PING_UUID,
    DUMMY_RECEIPT_ID,
    DUMMY_TOPIC,
    RUN_DATE,
    daily_context,
    dummy_secrets,
    failed_push,
    install_sleep_recorder,
    install_transport,
    ntfy_ok,
)
from utils.file_lock import FileLockHeldError, exclusive_file_lock

DUMMY_ODDS_KEY = "dummyoddsapikey0123456789abcdef"
TOMORROW = ("2026_07_DAL@PHI", "2026_07_KC@BUF")


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


# ---------------------------------------------------------------------------
# Plan 35-07: the failed pushes (SPEC R2, D-06, D-23)
# ---------------------------------------------------------------------------


def _report(**fields: Any) -> RunReport:
    """A report for RUN_DATE whose run started an hour before the slate's lock."""
    lock = slate_lock(RUN_DATE)
    values: dict[str, Any] = {
        "run_date_et": RUN_DATE,
        "start": lock - timedelta(hours=1),
        "lock": lock,
        "tomorrow_game_ids": TOMORROW,
    }
    values.update(fields)
    return RunReport(**values)


def _sync(**errors: str | None) -> SyncOutcome:
    """A sync on a no-new-row day: nothing pushed (Pitfall 6), plus any *errors* named."""
    values: dict[str, Any] = {
        "anchor_committed": False,
        "anchor_pushed": False,
        "anchor_error": None,
        "backup_committed": False,
        "backup_pushed": False,
        "backup_error": None,
        "refused_reason": None,
    }
    values.update(errors)
    return SyncOutcome(**values)


def _settle(**fields: Any) -> SettleOutcome:
    values: dict[str, Any] = {
        "changed": False,
        "graded": 0,
        "corrections": 0,
        "closing_finalized": 0,
        "observations": 0,
    }
    values.update(fields)
    return SettleOutcome(**values)


def _triggers(
    outcome: str, reason: str | None, *, readback_match: bool
) -> TriggerRegistration:
    return TriggerRegistration(outcome, reason, (), readback_match)


def _clean(**fields: Any) -> RunReport:
    """A clean run with every ledger side step clean, then *fields* on top."""
    values: dict[str, Any] = {
        "status": STATUS_SUCCESS,
        "unpredicted_game_ids": (),
        "settle_ran": True,
        "settle": _settle(),
        "triggers": _triggers("registered", None, readback_match=True),
        "sync_ran": True,
        "sync": _sync(),
    }
    values.update(fields)
    return _report(**values)


def _crashed() -> RunReport:
    report = _report(status=STATUS_RUN_FAILED, unpredicted_game_ids=TOMORROW)
    report.record_exception(RuntimeError("critical step failed: build_gold"))
    return report


def _skipped_trigger(reason: str) -> RunReport:
    return _clean(triggers=_triggers("skipped", reason, readback_match=False))


_PUSH_REFUSED = "CalledProcessError: push refused"

# One row per SPEC R2 outcome: (case id, report builder, top messages, normal messages).
R2_OUTCOMES: list[tuple[str, Callable[[], RunReport], int, int]] = [
    # Clean daily outcomes.
    ("success", _clean, 0, 0),
    ("no_games", lambda: _report(status=STATUS_NO_GAMES, tomorrow_game_ids=()), 0, 0),
    ("offseason", lambda: _report(status=STATUS_OFFSEASON, tomorrow_game_ids=()), 0, 0),
    (
        "lock_passed_zero_games",
        lambda: _report(status=STATUS_LOCK_PASSED, tomorrow_game_ids=()),
        0,
        0,
    ),
    # Top-priority daily outcomes.
    (
        "failed",
        lambda: _clean(status=STATUS_FAILED, unpredicted_game_ids=TOMORROW),
        1,
        0,
    ),
    (
        "degraded",
        lambda: _clean(status=STATUS_DEGRADED, failed_steps=("populate_web_cache",)),
        1,
        0,
    ),
    (
        "finished_with_skips",
        lambda: _clean(
            status=STATUS_FINISHED_WITH_SKIPS,
            unpredicted_game_ids=None,
            skipped_game_ids=(TOMORROW[0],),
        ),
        1,
        0,
    ),
    ("run_failed", _crashed, 1, 0),
    (
        "lock_passed_two_games",
        lambda: _report(status=STATUS_LOCK_PASSED, tomorrow_game_ids=TOMORROW),
        1,
        0,
    ),
    (
        "lock_passed_games_unknown",
        lambda: _report(status=STATUS_LOCK_PASSED, tomorrow_game_ids=None),
        1,
        0,
    ),
    # Normal-priority ledger and closing failures, each on an otherwise clean run.
    ("settle_raised", lambda: _clean(settle=None, settle_error="OSError"), 0, 1),
    (
        "closing_deferred",
        lambda: _clean(settle=_settle(closing_deferred="ValueError: run log line 12")),
        0,
        1,
    ),
    ("anchor_error", lambda: _clean(sync=_sync(anchor_error=_PUSH_REFUSED)), 0, 1),
    ("backup_error", lambda: _clean(sync=_sync(backup_error=_PUSH_REFUSED)), 0, 1),
    (
        "backup_repo_missing",
        lambda: _clean(sync=_sync(backup_error=BACKUP_REPO_MISSING)),
        0,
        1,
    ),
    (
        "sync_refused",
        lambda: _clean(sync=_sync(refused_reason="the ledger no longer extends")),
        0,
        1,
    ),
    ("sync_raised", lambda: _clean(sync=None, sync_error="RuntimeError"), 0, 1),
    (
        "triggers_failed",
        lambda: _clean(
            triggers=_triggers("failed", "error: OSError()", readback_match=False)
        ),
        0,
        1,
    ),
    (
        "triggers_skipped_not_installed",
        lambda: _skipped_trigger("closing_task_not_installed"),
        0,
        1,
    ),
    (
        "triggers_skipped_not_eastern",
        lambda: _skipped_trigger("machine_not_eastern"),
        0,
        1,
    ),
    (
        "triggers_readback_mismatch",
        lambda: _clean(
            triggers=_triggers("registered", "readback_mismatch", readback_match=False)
        ),
        0,
        1,
    ),
    # Ledger outcomes that are clean.
    (
        "triggers_skipped_task_running",
        lambda: _skipped_trigger("closing_task_running"),
        0,
        0,
    ),
    (
        "triggers_registered_readback_match",
        lambda: _clean(triggers=_triggers("registered", None, readback_match=True)),
        0,
        0,
    ),
]


def test_a_crash_before_the_orchestrator_delivers_one_top_priority_push(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_sleep_recorder(monkeypatch)
    requests = install_transport(monkeypatch, ntfy_ok())
    secrets = dummy_secrets(other_secret_values=(DUMMY_ODDS_KEY,))
    report = _report(status=STATUS_RUN_FAILED, unpredicted_game_ids=TOMORROW)
    report.record_exception(
        RuntimeError(
            f"schedule refresh exploded at https://example.com/x?apiKey={DUMMY_ODDS_KEY}"
        )
    )

    messages = failed_messages(
        report, now=report.start, secret_values=secrets.secret_values()
    )

    (message,) = messages
    assert message.priority == PRIORITY_TOP
    assert message.title == "FAILED: 2 games not predicted"
    delivery = send_push(message, daily_context(secrets))
    assert delivery.delivered
    (request,) = requests
    body = json.loads(request.content)
    assert body["priority"] == PRIORITY_TOP
    text = body["message"]
    assert "Games: DAL @ PHI, KC @ BUF" in text
    (error_line,) = [line for line in text.splitlines() if line.startswith("Error: ")]
    assert error_line.startswith(
        "Error: RuntimeError: schedule refresh exploded at https://example.com"
    )
    assert "/x" not in error_line
    assert "apiKey" not in error_line
    assert RERUN_COMMAND in text
    for secret in (DUMMY_ODDS_KEY, DUMMY_TOPIC, DUMMY_PING_UUID):
        assert secret not in text
    (record,) = read_notifications()
    assert record["event_class"] == "failed"
    assert record["priority"] == PRIORITY_TOP
    assert record["outcome"] == "delivered"


@pytest.mark.parametrize(
    ("build", "top", "normal"),
    [
        pytest.param(build, top, normal, id=case)
        for case, build, top, normal in R2_OUTCOMES
    ],
)
def test_every_r2_outcome_is_classified(
    build: Callable[[], RunReport], top: int, normal: int
) -> None:
    report = build()

    messages = failed_messages(report, now=report.start)

    assert all(message.event_class == "failed" for message in messages)
    assert sum(message.priority == PRIORITY_TOP for message in messages) == top
    assert sum(message.priority == PRIORITY_NORMAL for message in messages) == normal
    assert len(messages) == top + normal


def test_failed_and_ledger_failures_together_put_the_top_message_first() -> None:
    report = _clean(
        status=STATUS_FAILED,
        unpredicted_game_ids=TOMORROW,
        settle=None,
        settle_error="OSError",
        sync=_sync(anchor_error=_PUSH_REFUSED),
    )
    report.record_exception(RuntimeError("critical step failed"))

    messages = failed_messages(report, now=report.start)

    assert [message.priority for message in messages] == [
        PRIORITY_TOP,
        PRIORITY_NORMAL,
        PRIORITY_NORMAL,
    ]
    assert [message.title for message in messages] == [
        "FAILED: 2 games not predicted",
        "FAILED: ledger settle",
        "FAILED: ledger anchor push",
    ]
