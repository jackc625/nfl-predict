"""Phase 37 notification proofs: a frozen-arm hole reaches the phone as a FAILED push (SPEC R8, R14).

Every send goes through the fake transport (``install_transport``); no test reaches a real
provider, reads the owner's ``.env`` or touches a production store.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import dataclasses
from datetime import timedelta
from types import SimpleNamespace
from typing import Any

import httpx
import pytest

import scripts.daily_lock_pipeline as daily
from notify.end_of_run import notify_end_of_run
from notify.events import PRIORITY_NORMAL, PRIORITY_TOP
from notify.log import read_notifications
from notify.messages import failed_messages
from notify.report import STATUS_RUN_FAILED, STATUS_SUCCESS, RunReport
from pipeline import daily_steps
from pipeline.daily_steps import slate_lock
from tests.unit.test_daily_lock_run import RUN_DATE as DAILY_RUN_DATE
from tests.unit.test_daily_lock_run import (  # noqa: F401 -- trigger_calls: fixture
    _cut_over,
    _orchestrator_log,
    _pipeline_that,
    _recommend_env,
    _seen_reports,
    _slate,
    _slate_env,
    _start,
    trigger_calls,
)
from tests.unit.test_notify_transport import (
    RUN_DATE,
    dummy_secrets,
    install_sleep_recorder,
    install_transport,
    ntfy_ok,
)

TOMORROW = ("2026_08_DAL@PHI", "2026_08_KC@BUF")
HOLE_RECEIPT = "receiptHole001"
# Every report here started an hour before its slate's lock.
REPORT_START = slate_lock(RUN_DATE) - timedelta(hours=1)

# Three week-8 holes, keyed (game_id, season, week, target) per Plan 37-01, all one crash.
THREE_HOLES: tuple[tuple[tuple[Any, ...], str], ...] = (
    (("2026_08_DAL@PHI", 2026, 8, "wp"), "RuntimeError"),
    (("2026_08_DAL@PHI", 2026, 8, "ats"), "RuntimeError"),
    (("2026_08_KC@BUF", 2026, 8, "ou"), "RuntimeError"),
)


def _report(**fields: Any) -> RunReport:
    """A report for RUN_DATE whose run started an hour before the slate's lock."""
    values: dict[str, Any] = {
        "run_date_et": RUN_DATE,
        "start": REPORT_START,
        "lock": slate_lock(RUN_DATE),
        "tomorrow_game_ids": TOMORROW,
    }
    values.update(fields)
    return RunReport(**values)


def _clean_with_holes(**fields: Any) -> RunReport:
    """A clean run (no other failure) whose recommend step recorded THREE_HOLES."""
    values: dict[str, Any] = {
        "status": STATUS_SUCCESS,
        "unpredicted_game_ids": (),
        "slate_season": 2026,
        "slate_week": 8,
        "shadow_holes": THREE_HOLES,
    }
    values.update(fields)
    return _report(**values)


def _run_failed_and_site_push_failed(**fields: Any) -> RunReport:
    """A run that already owes two failed pushes: the run itself (top) and the website push."""
    report = _report(
        status=STATUS_RUN_FAILED,
        unpredicted_game_ids=TOMORROW,
        site_push_ran=True,
        site_push_error="OSError",
        **fields,
    )
    report.record_exception(RuntimeError("critical step failed: build_gold"))
    return report


def test_hole_push_standalone() -> None:
    for report in (_clean_with_holes(), _clean_with_holes(slate_week=None)):
        (message,) = failed_messages(report, now=report.start)

        assert message.event_class == "failed"
        assert message.title == "FAILED: frozen arm missed 3 rows (week 8)"
        assert message.priority == PRIORITY_NORMAL
        assert message.item_lines == ()
        (error_line,) = [
            line for line in message.header_lines if line.startswith("Error: ")
        ]
        assert error_line == "Error: RuntimeError"
        # The class only, never an exception message (D-10).
        assert not any("RuntimeError:" in line for line in message.header_lines)
        assert any(
            "ledger/refit/shadow_holes.jsonl" in line for line in message.header_lines
        )


def test_a_clean_run_without_holes_owes_no_failed_push() -> None:
    report = _clean_with_holes(shadow_holes=())

    assert failed_messages(report, now=report.start) == []


def test_hole_push_merges_into_first_failed() -> None:
    without = failed_messages(_run_failed_and_site_push_failed(), now=REPORT_START)
    with_holes = failed_messages(
        _run_failed_and_site_push_failed(shadow_holes=THREE_HOLES),
        now=REPORT_START,
    )
    (standalone,) = failed_messages(_clean_with_holes(), now=REPORT_START)

    assert len(without) == 2
    assert len(with_holes) == 2
    first, second = with_holes
    assert first.priority == PRIORITY_TOP
    assert first.event_class == "failed"
    assert first.title == f"{without[0].title} + frozen arm missed 3 rows"
    assert first.header_lines == (
        *without[0].header_lines,
        *standalone.header_lines,
    )
    assert first.item_lines == without[0].item_lines
    assert dataclasses.asdict(second) == dataclasses.asdict(without[1])
    assert second.title == "FAILED: website push"


def test_hole_push_sent_with_receipt(monkeypatch: pytest.MonkeyPatch) -> None:
    install_sleep_recorder(monkeypatch)
    requests = install_transport(
        monkeypatch,
        ntfy_ok(HOLE_RECEIPT),
        lambda: httpx.Response(200, text="OK"),  # the watcher check-in ending every run
    )
    report = _clean_with_holes()

    outcome = notify_end_of_run(report, secrets=dummy_secrets(), now=report.start)

    ((step, delivery),) = outcome.deliveries
    assert step == "failed"
    assert delivery.delivered
    assert delivery.receipt_id == HOLE_RECEIPT
    assert requests[0].url.host != "hc-ping.com"
    (record, *_checkin) = read_notifications()
    assert record["event_class"] == "failed"
    assert record["priority"] == PRIORITY_NORMAL
    assert record["outcome"] == "delivered"
    assert record["receipt_id"] == HOLE_RECEIPT


def test_recommend_slate_carries_holes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    trigger_calls: Any,  # noqa: F811
) -> None:
    from forward_ledger import runner

    # 1. recommend_slate copies the runner's holes onto the slate, beside its rows.
    _cut_over(monkeypatch)
    slate = _slate(30)
    _recommend_env(monkeypatch, tmp_path, slate)
    holes = (
        (("2026_W03_LAC@BUF", 2026, 3, "wp"), "RuntimeError"),
        (("2026_W03_LA@DEN", 2026, 3, "ou"), "RuntimeError"),
    )
    monkeypatch.setattr(
        runner,
        "record_forward_slate",
        lambda *_a, **_k: runner.RecordOutcome(
            2, 2, 0, (), True, None, None, shadow_holes=holes
        ),
    )

    daily_steps.recommend_slate(slate)

    assert slate.shadow_holes == list(holes)

    # 2. run_daily's report copies what the recommend step recorded on the slate.
    _cut_over(monkeypatch, on=False)
    _slate_env(monkeypatch, tmp_path)

    def _registry(selected):
        def _recommend():
            selected.shadow_holes.extend(holes)

        return [SimpleNamespace(name="generate_recommendations", callable=_recommend)]

    monkeypatch.setattr(daily, "build_daily_step_registry", _registry)
    _pipeline_that(monkeypatch, _orchestrator_log("success"))
    seen = _seen_reports(monkeypatch)

    assert daily.run_daily(DAILY_RUN_DATE, start=_start(), dry_run=False) == 0

    (report,) = seen
    assert report.shadow_holes == holes
