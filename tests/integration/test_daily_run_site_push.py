"""The website push inside the daily run (Phase 36, D-13, SPEC R5/R7, RESEARCH Pitfall 10).

WHAT THESE TESTS PROVE
----------------------
``scripts.daily_lock_pipeline.run_daily`` is driven through the light harness of
``tests/unit/test_daily_lock_run.py`` with ``SITE_PUSH`` restored to the real ``daily._push_site``
(``tests/conftest.py`` keeps it inert everywhere else):

* the push runs after the ledger sync and before the end-of-run notifier, on the main path and on
  the lock-passed branch, and its outcome lands on the run's report;
* a run ending ``failed`` or ``run_failed`` records "not pushed" instead; a dry run, a Ctrl+C and a
  refused ``--date`` neither push nor record;
* the push hears whether this run attempted a cache rebuild and whether it completed;
* a raise that escapes the push routine is caught, printed by class and noted on the report, and
  the run's result is unchanged.

``pipeline.site_push.push_site`` / ``record_not_pushed`` are recorders here unless a test says
otherwise. Nothing deploys: the conftest net refuses the Vercel CLI and every Vercel host.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from datetime import UTC, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

import scripts.daily_lock_pipeline as daily
from notify.report import RunReport
from pipeline import site_push
from pipeline.daily_steps import slate_lock
from tests.unit.test_daily_lock_run import (
    RUN_DATE,
    SLATE_IDS,
    _arrange_lock_passed,
    _crash_at_refresh,
    _cut_over,
    _no_games_env,
    _orchestrated,
    _orchestrator_log,
    _start,
    trigger_calls,  # noqa: F401 - autouse: no run reaches the real Task Scheduler
)

SUCCEEDED = site_push.PushOutcome(outcome="succeeded", publish_id="p" * 32)


class PushRecorder:
    """Stands in for ``site_push.push_site`` and ``record_not_pushed``; records every call.

    Attributes:
        order: The shared order list each call appends its name to.
        pushes: The keyword arguments of every ``push_site`` call.
        not_pushed: The keyword arguments of every ``record_not_pushed`` call.
        push_does: The outcome ``push_site`` returns, or an exception it raises.
    """

    def __init__(self, order: list[str], push_does: Any = SUCCEEDED) -> None:
        self.order = order
        self.pushes: list[dict[str, Any]] = []
        self.not_pushed: list[dict[str, Any]] = []
        self.push_does = push_does

    def push_site(self, **kwargs: Any) -> site_push.PushOutcome:
        self.order.append("push")
        self.pushes.append(kwargs)
        if isinstance(self.push_does, BaseException):
            raise self.push_does
        return self.push_does

    def record_not_pushed(self, **kwargs: Any) -> site_push.PushOutcome:
        self.order.append("not_pushed")
        self.not_pushed.append(kwargs)
        return site_push.PushOutcome(outcome="not_pushed", reason=kwargs["reason"])


def wire_the_push(
    monkeypatch: pytest.MonkeyPatch, order: list[str], push_does: Any = SUCCEEDED
) -> PushRecorder:
    """Restore the real ``SITE_PUSH`` seam over recorders of the two push-log routines."""
    recorder = PushRecorder(order, push_does)
    monkeypatch.setattr(daily, "SITE_PUSH", daily._push_site)
    monkeypatch.setattr(site_push, "push_site", recorder.push_site)
    monkeypatch.setattr(site_push, "record_not_pushed", recorder.record_not_pushed)
    return recorder


def record_the_notifier(
    monkeypatch: pytest.MonkeyPatch, order: list[str]
) -> list[RunReport]:
    """Replace the notifier seam with a recorder; return the reports it was handed."""
    seen: list[RunReport] = []

    def _notify(report: RunReport) -> None:
        order.append("notify")
        seen.append(report)

    monkeypatch.setattr(daily, "NOTIFY_END_OF_RUN", _notify)
    return seen


def refuse_both_routines(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real seam, with both push-log routines failing the test if they are reached."""

    def _refuse(*_args: Any, **_kwargs: Any) -> None:
        pytest.fail("the run reached the website push")

    monkeypatch.setattr(daily, "SITE_PUSH", daily._push_site)
    monkeypatch.setattr(site_push, "push_site", _refuse)
    monkeypatch.setattr(site_push, "record_not_pushed", _refuse)


# ---------------------------------------------------------------------------
# Task 1 (tracer): a quiet day pushes after the sync and before the notifier
# ---------------------------------------------------------------------------


def test_a_quiet_day_pushes_after_the_sync_and_before_the_notifier(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    _cut_over(monkeypatch)
    order = _no_games_env(monkeypatch, tmp_path)
    recorder = wire_the_push(monkeypatch, order)
    seen = record_the_notifier(monkeypatch, order)

    assert daily.run_daily(RUN_DATE, start=_start(), dry_run=False) == 0

    assert order[-3:] == ["sync", "push", "notify"]
    (report,) = seen
    (push,) = recorder.pushes
    assert push["source"] == "daily"
    assert push["run_id"] == report.run_id is not None
    assert (push["rebuild_attempted"], push["rebuild_completed"]) == (False, False)
    assert push["lock_wait_seconds"] == site_push.DAILY_LOCK_WAIT_SECONDS
    assert report.site_push is SUCCEEDED
    assert report.site_push_ran is True
    assert recorder.not_pushed == []


def _arrange_run_failed(monkeypatch: pytest.MonkeyPatch, tmp_path) -> int | None:
    """The schedule refresh raises: the run ends ``run_failed`` and re-raises."""
    _crash_at_refresh(monkeypatch, tmp_path)
    return None


def _arrange_failed(monkeypatch: pytest.MonkeyPatch, tmp_path) -> int | None:
    """The orchestrator returns ``failed`` without raising: the run exits 1."""
    _orchestrated(_orchestrator_log("failed", failed=["build_gold"]))(
        monkeypatch, tmp_path
    )
    return 1


@pytest.mark.parametrize(
    ("arrange", "status"),
    [
        pytest.param(_arrange_run_failed, "run_failed", id="run_failed"),
        pytest.param(_arrange_failed, "failed", id="failed"),
    ],
)
def test_failed_runs_record_not_pushed(
    monkeypatch: pytest.MonkeyPatch, tmp_path, arrange, status: str
) -> None:
    order: list[str] = []
    recorder = wire_the_push(monkeypatch, order)
    seen = record_the_notifier(monkeypatch, order)
    exit_code = arrange(monkeypatch, tmp_path)

    if exit_code is None:
        with pytest.raises(RuntimeError, match="exploded"):
            daily.run_daily(RUN_DATE, start=_start(), dry_run=False)
    else:
        assert daily.run_daily(RUN_DATE, start=_start(), dry_run=False) == exit_code

    (report,) = seen
    assert report.status == status
    assert recorder.pushes == []
    assert recorder.not_pushed == [
        {"source": "daily", "reason": status, "run_id": report.run_id}
    ]
    assert order == ["not_pushed", "notify"]


def _dry_run(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _no_games_env(monkeypatch, tmp_path)
    assert daily.run_daily(RUN_DATE, start=_start(), dry_run=True) == 0


def _ctrl_c(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    _crash_at_refresh(monkeypatch, tmp_path)

    def _interrupted(*_a: Any, **_k: Any) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(daily, "_refresh_schedule", _interrupted)
    with pytest.raises(KeyboardInterrupt):
        daily.run_daily(RUN_DATE, start=_start(), dry_run=False)


def _refused_date(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(daily, "DAILY_RUN_RECORDS", tmp_path / "daily.jsonl")
    next_day = slate_lock(RUN_DATE).astimezone(UTC) + timedelta(hours=10)
    with pytest.raises(daily.RunDateRefusedError):
        daily.run_daily(RUN_DATE, start=next_day, dry_run=False)


@pytest.mark.parametrize("run", [_dry_run, _ctrl_c, _refused_date])
def test_dry_runs_ctrl_c_and_refused_dates_never_push(
    monkeypatch: pytest.MonkeyPatch, tmp_path, run
) -> None:
    refuse_both_routines(monkeypatch)

    run(monkeypatch, tmp_path)

    assert not site_push.SITE_PUSH_LOG.exists()


def test_the_lock_passed_branch_pushes_before_its_notifier(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    order: list[str] = []
    recorder = wire_the_push(monkeypatch, order)
    seen = record_the_notifier(monkeypatch, order)
    at_the_lock = _arrange_lock_passed(monkeypatch, tmp_path)

    assert daily.run_daily(RUN_DATE, start=at_the_lock, dry_run=False) == 0

    assert order == ["push", "notify"]
    (push,) = recorder.pushes
    assert (push["source"], push["run_id"]) == ("daily", None)
    (report,) = seen
    assert report.status == "lock_passed"
    assert report.site_push is SUCCEEDED


def test_a_raising_push_is_caught_and_noted(
    monkeypatch: pytest.MonkeyPatch, tmp_path, capsys: pytest.CaptureFixture[str]
) -> None:
    order = _no_games_env(monkeypatch, tmp_path)
    # Stands in for a raise the routine cannot record itself, such as a failed record write.
    wire_the_push(monkeypatch, order, RuntimeError("push-detail https://x.invalid"))
    seen = record_the_notifier(monkeypatch, order)

    assert daily.run_daily(RUN_DATE, start=_start(), dry_run=False) == 0

    (report,) = seen
    assert report.site_push_error == "RuntimeError"
    assert report.site_push is None
    assert order.count("notify") == 1
    assert order.index("push") < order.index("notify")
    out = capsys.readouterr().out
    assert "SITE_PUSH= failed RuntimeError" in out
    assert "push-detail" not in out


def _orchestrated_with(steps: list[SimpleNamespace]):
    """A success run whose orchestrator log carries *steps* after its prediction step."""
    log = _orchestrator_log("success")
    log.steps.extend(steps)
    return _orchestrated(log, predicted=SLATE_IDS)


def _settle_day_rebuild_raises(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """A no-games day whose settle changed the ledger, and whose cache rebuild raises."""
    import pipeline.steps as steps_mod
    from forward_ledger import runner

    _cut_over(monkeypatch)
    _no_games_env(monkeypatch, tmp_path)
    monkeypatch.setattr(
        runner, "settle_ledger", lambda **_k: runner.SettleOutcome(True, 1, 0, 1, 0)
    )

    def _cache_down() -> None:
        raise OSError("the cache file is locked")

    monkeypatch.setattr(steps_mod, "step_populate_web_cache", _cache_down)


@pytest.mark.parametrize(
    ("arrange", "expected"),
    [
        pytest.param(
            _orchestrated_with(
                [SimpleNamespace(name="populate_web_cache", status="failed")]
            ),
            (True, False),
            id="populate_failed",
        ),
        pytest.param(
            _orchestrated_with(
                [SimpleNamespace(name="populate_web_cache", status="success")]
            ),
            (True, True),
            id="populate_succeeded",
        ),
        pytest.param(_orchestrated_with([]), (False, False), id="no_populate_step"),
        pytest.param(
            _settle_day_rebuild_raises, (True, False), id="settle_day_rebuild_raises"
        ),
    ],
)
def test_the_push_hears_whether_the_rebuild_completed(
    monkeypatch: pytest.MonkeyPatch, tmp_path, arrange, expected: tuple[bool, bool]
) -> None:
    order: list[str] = []
    recorder = wire_the_push(monkeypatch, order)
    record_the_notifier(monkeypatch, order)
    arrange(monkeypatch, tmp_path)

    assert daily.run_daily(RUN_DATE, start=_start(), dry_run=False) == 0

    (push,) = recorder.pushes
    assert (push["rebuild_attempted"], push["rebuild_completed"]) == expected
