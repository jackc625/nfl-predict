"""The closed push router's must-NOTs (Phase 35 SPEC must-NOTs, R1/R2 edges, T-35-47).

* Only ``failed``, ``bets`` and ``test`` may be pushed: any other class is refused before a request.
* Nothing suppresses a message: two identical failures are two deliveries and two log records.
* Records land in send order.
* A delivered push whose log record could not be written is delivered but visibly unlogged.
* Plan 35-07: every SPEC R2 outcome maps to exactly the right number and priority of failed
  pushes, and a crash before the orchestrator reaches ntfy as one top-priority push, logged once.
* Plan 35-07: the bets push lists exactly the newly appended live bets, in ``/bets`` order (pinned
  against ``DataService.get_bet_list``); an identical re-run sends none, a re-run adding one bet
  lists exactly it. The runner cases reuse the forward-ledger runner's tmp_path environment.
* Plan 35-08: the end of the daily run sends failed first, then bets -- two deliveries, two
  records, in that order -- and reports a delivered message whose receipt was not logged.

The MockTransport, sleep-recorder and dummy-secret helpers come from
``tests/unit/test_notify_transport.py``.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import date, timedelta
from typing import Any

import duckdb
import httpx
import pandas as pd
import pytest

from api.cache import (
    ARM_SHADOW,
    BET_LIST_COLUMNS,
    GRADING_STATUS_PENDING,
    materialize_bet_list,
)
from api.services import DataService, clear_cache
from forward_ledger.closing_schedule import TriggerRegistration
from forward_ledger.runner import SettleOutcome
from forward_ledger.sync import BACKUP_REPO_MISSING, SyncOutcome
from notify import log as notification_log
from notify.end_of_run import notify_end_of_run
from notify.events import PRIORITY_NORMAL, PRIORITY_TOP, PushMessage, send_push
from notify.log import read_notifications
from notify.messages import RERUN_COMMAND, bets_message, failed_messages
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
from tests.unit.test_forward_ledger_runner import (  # noqa: F401 -- env, run_log_path: fixtures
    DECIDED_AT,
    _Env,
    env,
    run_log_path,
)
from tests.unit.test_forward_ledger_runner import LOCK as RUNNER_LOCK
from tests.unit.test_forward_ledger_runner_appended import EXTRA_GAME, _with_extra_game
from tests.unit.test_notify_messages import live_bet
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
# The forward-ledger runner environment's slate: run date Wednesday, games Thursday.
RUNNER_SLATE_DATE = date(2026, 10, 14)


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


# ---------------------------------------------------------------------------
# Plan 35-07: the bets push (SPEC R3, D-14)
# ---------------------------------------------------------------------------


def _bets_report(rows: list[dict[str, Any]], **fields: Any) -> RunReport:
    return _report(slate_season=2026, slate_week=6, appended_rows=rows, **fields)


def _cache_order(rows: list[dict[str, Any]]) -> list[str]:
    """The matchups ``/bets`` would list for these rows, read through ``DataService``."""
    frame = pd.DataFrame(
        [
            {
                **dict.fromkeys(BET_LIST_COLUMNS),
                **{name: row[name] for name in BET_LIST_COLUMNS if name in row},
                "grading_status": GRADING_STATUS_PENDING,
            }
            for row in rows
        ]
    )
    conn = duckdb.connect(":memory:")
    try:
        materialize_bet_list(conn, frame)
        clear_cache()
        listed = DataService(conn).get_bet_list(2026, 6)
    finally:
        clear_cache()
        conn.close()
    return [str(row["game_id"]).split("_")[-1].replace("@", " @ ") for row in listed]


def test_three_new_live_bets_give_one_push_with_three_lines_in_bets_order() -> None:
    # Handed over out of order, with an EV tie the game_id tie-break has to settle.
    rows = [
        live_bet("2026_06_DAL@PHI", target="ou", bet_side="under", line=47.5),
        live_bet("2026_06_CIN@CLE", per_bet_ev=0.0377),
        live_bet("2026_06_KC@BUF", target="wp", bet_side="away", per_bet_ev=0.061),
    ]

    message = bets_message(_bets_report(rows))

    assert message is not None
    assert message.event_class == "bets"
    assert message.priority == PRIORITY_NORMAL
    assert message.title == "3 new bets: 2026 week 6, Sunday"
    assert message.header_lines == ("2026 week 6, Sunday games",)
    assert len(message.item_lines) == 3
    matchups = [line.split(" | ")[0] for line in message.item_lines]
    assert matchups == _cache_order(rows) == ["KC @ BUF", "CIN @ CLE", "DAL @ PHI"]


def test_zero_new_live_bets_give_no_push() -> None:
    shadow = live_bet("2026_06_CIN@CLE", arm=ARM_SHADOW)
    suppressed = live_bet(
        "2026_06_KC@BUF", status="suppressed", rejection_reason="below_floor"
    )

    assert bets_message(_bets_report([])) is None
    assert bets_message(_bets_report([shadow, suppressed])) is None


def _runner_report(outcome: Any) -> RunReport:
    """A report carrying the rows one runner call appended, for its 2026 week 6 slate."""
    return RunReport(
        run_date_et=RUNNER_SLATE_DATE,
        start=DECIDED_AT,
        lock=RUNNER_LOCK,
        slate_season=2026,
        slate_week=6,
        appended_rows=list(outcome.appended_rows),
    )


def test_an_identical_rerun_gives_no_push(env: _Env) -> None:  # noqa: F811
    first = bets_message(_runner_report(env.record()))

    again = bets_message(
        _runner_report(env.record(decided_at=DECIDED_AT + timedelta(minutes=12)))
    )

    assert first is not None
    assert len(first.item_lines) == 2
    assert again is None


def test_a_rerun_adding_one_bet_lists_exactly_it(env: _Env) -> None:  # noqa: F811
    env.record()

    message = bets_message(
        _runner_report(
            env.record(
                decided_at=DECIDED_AT + timedelta(minutes=5),
                bundle_builder=_with_extra_game(env),
            )
        )
    )

    assert message is not None
    assert message.title == "1 new bet: 2026 week 6, Thursday"
    (line,) = message.item_lines
    assert line.startswith(f"{EXTRA_GAME.split('_')[-1]} | Spread | ")


# ---------------------------------------------------------------------------
# Plan 35-08: the end of the daily run sends failed first, then bets (SPEC R1/R3 adjacency)
# ---------------------------------------------------------------------------


def _ping_ok() -> httpx.Response:
    """healthchecks.io's answer to a ping that did something: HTTP 200 with the body ``OK``."""
    return httpx.Response(200, text="OK")


def _skips_with_two_new_bets() -> RunReport:
    """A finished-with-skips run that also appended two live bets."""
    return _report(
        status=STATUS_FINISHED_WITH_SKIPS,
        skipped_game_ids=(TOMORROW[0],),
        slate_season=2026,
        slate_week=7,
        appended_rows=[
            live_bet("2026_07_KC@BUF", week=7, per_bet_ev=0.05),
            live_bet("2026_07_CIN@CLE", week=7, per_bet_ev=0.04),
        ],
    )


def test_failed_then_bets_are_two_deliveries_in_that_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_sleep_recorder(monkeypatch)
    requests = install_transport(
        monkeypatch,
        ntfy_ok("receiptFailed1"),
        ntfy_ok("receiptBets001"),
        _ping_ok,  # the watcher check-in that ends every run (Plan 35-13)
    )
    report = _skips_with_two_new_bets()

    outcome = notify_end_of_run(report, secrets=dummy_secrets(), now=report.start)

    *pushes, checkin = requests
    titles = [json.loads(request.content)["title"] for request in pushes]
    assert titles == ["FAILED: 1 game not predicted", "2 new bets: 2026 week 7, Sunday"]
    assert checkin.url.host == "hc-ping.com"
    assert [step for step, _delivery in outcome.deliveries] == ["failed", "bets"]
    records = read_notifications()
    assert [record["event_class"] for record in records] == [
        "failed",
        "bets",
        "checkin",
    ]
    assert [record["receipt_id"] for record in records[:2]] == [
        "receiptFailed1",
        "receiptBets001",
    ]
    assert outcome.any_undelivered is False
    assert outcome.any_unlogged is False


def test_an_unlogged_receipt_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    install_sleep_recorder(monkeypatch)
    install_transport(
        monkeypatch, ntfy_ok("receiptFailed1"), ntfy_ok("receiptBets001"), _ping_ok
    )
    locks_taken: list[int] = []

    def _bets_record_lock_never_frees(*args: Any, **kwargs: Any) -> Any:
        locks_taken.append(1)
        if len(locks_taken) == 2:  # the bets record, after the failed one
            msg = "the lock is held by another writer (waited 60s)."
            raise FileLockHeldError(msg)
        return exclusive_file_lock(*args, **kwargs)

    monkeypatch.setattr(
        notification_log, "exclusive_file_lock", _bets_record_lock_never_frees
    )
    report = _skips_with_two_new_bets()

    outcome = notify_end_of_run(report, secrets=dummy_secrets(), now=report.start)

    assert outcome.any_undelivered is False
    assert outcome.any_unlogged is True
    ((_, failed), (_, bets)) = outcome.deliveries
    assert (failed.logged, bets.logged) == (True, False)
    # The bets record is missing; the check-in that follows it is recorded (Plan 35-13).
    assert [record["event_class"] for record in read_notifications()] == [
        "failed",
        "checkin",
    ]
