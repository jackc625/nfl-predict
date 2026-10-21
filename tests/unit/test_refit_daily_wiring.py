"""The daily run calls the re-fit side step after its picks, on every branch past them (Plan 37-13).

WHAT IS PROVEN (D-15, D-17)
---------------------------
* A ``no_games`` day calls the hook after the settle pass and before the one cache rebuild and the
  no-prediction record; the rebuild runs once when the hook wrote, printing
  ``WEB_CACHE_AFTER_REFIT=``, and a settle-only change still prints ``WEB_CACHE_AFTER_SETTLE=``.
* A prediction day calls it after the pipeline (so after the ledger write) with the day's slate,
  and before the website push and the notifier; a ``lock_passed`` run calls it with a refresh.
* A dry run, and a run whose pipeline raised, never call it.
* A refused re-fit, or a hook that raises, never changes the exit code; the contract line names
  the status, or the exception class only.

Every test restores ``scripts.daily_lock_pipeline.REFIT_HOOK`` to a recording fake (the autouse
fixture in ``tests/conftest.py`` keeps it inert everywhere else); no test trains a model or touches
a production store.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

import scripts.daily_lock_pipeline as daily
from notify.report import RunReport
from refit.hook import RefitOutcome
from tests.unit.test_daily_lock_run import (
    RUN_DATE,
    SLATE_IDS,
    _arrange_lock_passed,
    _cut_over,
    _no_games_env,
    _orchestrator_log,
    _slate_env,
    _start,
    trigger_calls,  # noqa: F401 - the autouse stand-in for the real Task Scheduler
)

# A message the printed output must never carry (D-10: the class name only).
SECRET_TEXT: str = "secret token in a re-fit failure"


def recording_hook(
    order: list[str], calls: list[dict[str, Any]], outcome: RefitOutcome | None
):
    """A REFIT_HOOK that records its call (and where in the run it came) and returns *outcome*."""

    def hook(report: RunReport, **kwargs: Any) -> RefitOutcome | None:
        order.append("refit")
        calls.append({"report": report, **kwargs})
        return outcome

    return hook


def wrote(status: str = "ran") -> RefitOutcome:
    return RefitOutcome(status=status, date="2026-11-03", wrote_records=True)


def quiet() -> RefitOutcome:
    return RefitOutcome(status="not_due", wrote_records=False)


def _ledger_passes_inert(monkeypatch: pytest.MonkeyPatch, order: list[str]) -> None:
    """The settle pass and the sync record their place and touch no ledger."""
    from forward_ledger import runner

    def _settle(**_kwargs: Any) -> Any:
        order.append("settle")
        return runner.SettleOutcome(False, 0, 0, 0, 0)

    monkeypatch.setattr(runner, "settle_ledger", _settle)
    monkeypatch.setattr(
        runner, "sync_after_run", lambda **_k: order.append("sync") or None
    )


def _cache_records(monkeypatch: pytest.MonkeyPatch, order: list[str]) -> None:
    import pipeline.steps as steps_mod

    monkeypatch.setattr(
        steps_mod, "step_populate_web_cache", lambda: order.append("cache")
    )


def _seams_record(
    monkeypatch: pytest.MonkeyPatch, order: list[str], seen: list[RunReport]
) -> None:
    """The website push and the notifier record their place; the notifier keeps the report."""

    def _push(_report: RunReport) -> None:
        order.append("push")

    def _notify(report: RunReport) -> None:
        order.append("notify")
        seen.append(report)

    monkeypatch.setattr(daily, "SITE_PUSH", _push)
    monkeypatch.setattr(daily, "NOTIFY_END_OF_RUN", _notify)


def _prediction_day(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, order: list[str], outcome: Any
) -> None:
    """Tomorrow's two-game slate; the orchestrator records its place and returns *outcome*."""
    from pipeline import orchestrator

    _slate_env(monkeypatch, tmp_path, predicted=SLATE_IDS)
    _ledger_passes_inert(monkeypatch, order)
    monkeypatch.setattr(daily, "build_daily_step_registry", lambda _slate: [])

    class _Pipeline:
        def __init__(self, *, steps: Any, deadline: Any) -> None:
            self.steps = steps

        def run(self) -> Any:
            order.append("pipeline")
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

    monkeypatch.setattr(orchestrator, "FridayPipeline", _Pipeline)


# ---------------------------------------------------------------------------
# The three insertion points
# ---------------------------------------------------------------------------


def test_no_games_day_calls_the_hook_then_rebuilds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    from forward_ledger import runner

    _cut_over(monkeypatch)
    order = _no_games_env(monkeypatch, tmp_path)
    _cache_records(monkeypatch, order)
    seen: list[RunReport] = []
    monkeypatch.setattr(daily, "NOTIFY_END_OF_RUN", seen.append)
    calls: list[dict[str, Any]] = []

    # The hook wrote records: one rebuild, after the hook, before the no-prediction record.
    monkeypatch.setattr(daily, "REFIT_HOOK", recording_hook(order, calls, wrote()))
    assert daily.run_daily(RUN_DATE, start=_start(), dry_run=False) == 0
    out = capsys.readouterr().out
    assert order.count("refit") == order.count("cache") == 1
    assert (
        order.index("settle")
        < order.index("refit")
        < order.index("cache")
        < order.index("no_games")
    )
    assert "WEB_CACHE_AFTER_REFIT= rebuilt" in out
    assert "REFIT= ran date=2026-11-03 check=none" in out
    (call,) = calls
    assert (call["slate"], call["schedule_refreshed"]) == (None, True)
    assert call["run_date_et"] == RUN_DATE
    assert call["now_utc"].tzinfo is not None
    (report,) = seen
    assert report.cache_rebuild_attempted is True
    assert report.cache_rebuild_completed is True
    assert (report.refit_ran, report.refit) == (True, wrote())

    # Settle changed AND the hook wrote: still ONE rebuild, named for the re-fit.
    monkeypatch.setattr(
        runner, "settle_ledger", lambda **_k: runner.SettleOutcome(True, 1, 0, 1, 0)
    )
    order.clear()
    assert daily.run_daily(RUN_DATE, start=_start(), dry_run=False) == 0
    out = capsys.readouterr().out
    assert order.count("cache") == 1
    assert "WEB_CACHE_AFTER_REFIT= rebuilt" in out
    assert "WEB_CACHE_AFTER_SETTLE=" not in out

    # Only the settle changed: the one rebuild prints exactly what it printed before.
    monkeypatch.setattr(daily, "REFIT_HOOK", recording_hook(order, calls, quiet()))
    order.clear()
    assert daily.run_daily(RUN_DATE, start=_start(), dry_run=False) == 0
    out = capsys.readouterr().out
    assert order.count("cache") == 1
    assert order.index("refit") < order.index("cache") < order.index("no_games")
    assert "WEB_CACHE_AFTER_SETTLE= rebuilt" in out
    assert "WEB_CACHE_AFTER_REFIT=" not in out


def test_prediction_day_calls_the_hook_after_the_pipeline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    order: list[str] = []
    _prediction_day(monkeypatch, tmp_path, order, _orchestrator_log("success"))
    _cache_records(monkeypatch, order)
    seen: list[RunReport] = []
    _seams_record(monkeypatch, order, seen)
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(daily, "REFIT_HOOK", recording_hook(order, calls, wrote()))

    assert daily.run_daily(RUN_DATE, start=_start(), dry_run=False) == 0

    assert (
        order.index("pipeline")
        < order.index("refit")
        < order.index("cache")
        < order.index("push")
        < order.index("notify")
    )
    (call,) = calls
    assert call["slate"] is not None
    assert call["slate"].game_ids == frozenset(SLATE_IDS)
    assert call["schedule_refreshed"] is True
    (report,) = seen
    assert report.refit == wrote()
    assert report.cache_rebuild_attempted is True


def test_lock_passed_calls_the_hook_with_a_refresh(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    order: list[str] = []
    start = _arrange_lock_passed(monkeypatch, tmp_path)
    seen: list[RunReport] = []
    _seams_record(monkeypatch, order, seen)
    refreshed: list[tuple[date, bool]] = []

    def _refresh(run_date_et: date, *, dry_run: bool) -> int:
        refreshed.append((run_date_et, dry_run))
        return 2026

    monkeypatch.setattr(daily, "_refresh_schedule", _refresh)
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(daily, "REFIT_HOOK", recording_hook(order, calls, quiet()))

    assert daily.run_daily(RUN_DATE, start=start, dry_run=False) == 0

    assert order.index("refit") < order.index("push") < order.index("notify")
    (call,) = calls
    assert (call["slate"], call["schedule_refreshed"]) == (None, False)
    assert refreshed == [], "the lock-passed run refreshed before anything was due"
    call["refresh_schedule"]()
    assert refreshed == [(RUN_DATE, False)]
    (report,) = seen
    assert report.status == "lock_passed"
    assert report.refit == quiet()


def test_never_on_dry_run_or_a_crash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    order = _no_games_env(monkeypatch, tmp_path)
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(daily, "REFIT_HOOK", recording_hook(order, calls, wrote()))

    assert daily.run_daily(RUN_DATE, start=_start(), dry_run=True) == 0
    assert calls == [], "a dry run attempted a re-fit"

    crash_order: list[str] = []
    _prediction_day(
        monkeypatch, tmp_path, crash_order, RuntimeError("Critical step failed")
    )
    with pytest.raises(RuntimeError, match="Critical step failed"):
        daily.run_daily(RUN_DATE, start=_start(), dry_run=False)
    assert "pipeline" in crash_order
    assert calls == [], "a run that crashed before the re-fit point attempted one"


def test_a_refit_failure_never_changes_the_exit_code(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    order: list[str] = []
    _prediction_day(monkeypatch, tmp_path, order, _orchestrator_log("success"))
    _cache_records(monkeypatch, order)
    seen: list[RunReport] = []
    _seams_record(monkeypatch, order, seen)

    monkeypatch.setattr(daily, "REFIT_HOOK", lambda _report, **_kwargs: None)
    baseline = daily.run_daily(RUN_DATE, start=_start(), dry_run=False)
    capsys.readouterr()

    refused = RefitOutcome(
        status="refused", date="2026-11-03", check="scores_frame", wrote_records=True
    )
    monkeypatch.setattr(daily, "REFIT_HOOK", lambda _report, **_kwargs: refused)
    assert daily.run_daily(RUN_DATE, start=_start(), dry_run=False) == baseline
    assert (
        "REFIT= refused date=2026-11-03 check=scores_frame" in capsys.readouterr().out
    )
    assert seen[-1].refit == refused
    assert seen[-1].refit_error is None

    def _raises(_report: RunReport, **_kwargs: Any) -> RefitOutcome:
        raise RuntimeError(SECRET_TEXT)

    monkeypatch.setattr(daily, "REFIT_HOOK", _raises)
    assert daily.run_daily(RUN_DATE, start=_start(), dry_run=False) == baseline
    out = capsys.readouterr().out
    assert "REFIT= failed RuntimeError" in out
    assert SECRET_TEXT not in out
    assert (seen[-1].refit_ran, seen[-1].refit, seen[-1].refit_error) == (
        True,
        None,
        "RuntimeError",
    )
    assert seen[-1].refit_error_location is None
    assert baseline == 0

    # WR-C05: an unreadable record is named by its path and line, never by its message.
    from refit.records import RefitRecordFormatError

    def _torn(_report: RunReport, **_kwargs: Any) -> RefitOutcome:
        msg = f"refit record ledger/refit/dates.jsonl line 7 {SECRET_TEXT}"
        raise RefitRecordFormatError(msg, location="ledger/refit/dates.jsonl line 7")

    monkeypatch.setattr(daily, "REFIT_HOOK", _torn)
    assert daily.run_daily(RUN_DATE, start=_start(), dry_run=False) == baseline
    out = capsys.readouterr().out
    assert (
        "REFIT= failed RefitRecordFormatError at ledger/refit/dates.jsonl line 7" in out
    )
    assert SECRET_TEXT not in out
    assert (seen[-1].refit_error, seen[-1].refit_error_location) == (
        "RefitRecordFormatError",
        "ledger/refit/dates.jsonl line 7",
    )
