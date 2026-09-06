"""A cache-step failure DEGRADES the Friday run; it never fails a run whose predictions succeeded.

WHAT THIS PROVES, AND WHY IT DRIVES THE REAL ORCHESTRATOR
----------------------------------------------------------
SPEC R9 / D31-29 say the cache-population step is registered ``critical=False`` and that its
failure routes to the EXISTING alert path with no new alert code. Both halves are claims about the
``pipeline.orchestrator.FridayPipeline`` loop, not about the step body, so this module drives the
REAL loop through the REAL ``build_step_registry()`` -- no registry patch and no ``make_mock_step``.
Stubbing the registry is the aliasing blind spot that hid cb61042 in this repository's own history,
and it would void the proof here for the same reason: the property under test IS the registry's
``critical`` flag.

Only the step BODIES are no-op'd, and they are no-op'd by a name derived from the registry rather
than from a hand-maintained list, so a step added later cannot quietly execute for real inside this
test. The one body that is NOT a no-op is the cache step, which is replaced by a raiser.

THE CONTROL IS NOT OPTIONAL
----------------------------
``test_a_clean_run_is_success_not_degraded`` runs the identical wiring with the cache step
succeeding. Without it, a bug that reported EVERY run as degraded would pass the degraded
assertion, and "degraded" would have stopped discriminating between a broken cache and a healthy
one.

WHAT DEGRADED ACTUALLY BUYS -- STATED, NOT OVERSOLD
----------------------------------------------------
``pipeline/alert.py`` documents alerts as log-only by default and this project's record is that the
email and messaging channels are inert. So this test asserts the routing (exactly one
``alert_degraded_completion`` and no failure alert), and does NOT claim anybody is notified. The
real protection against a silently stale bet list is the ``/bets`` hard-block asserted in
``tests/api/test_bets_page.py``.

Selectors (``-k``): degraded, alert_route, still_recorded, clean_run, empty_prediction_set.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pytest

from api.cache import populate_cache
from pipeline.alert import PipelineAlertManager
from pipeline.steps import PipelinePhase, StepStatus, build_step_registry

_SEASON = 2024
_WEEK = 1

# The step this plan registers, and the adapter-name convention every registry entry follows.
_CACHE_STEP = "populate_web_cache"
_ADAPTER_PREFIX = "step_"


class _AlertSpy:
    """Records which completion alert the orchestrator routed to, and how many times."""

    def __init__(self) -> None:
        self.calls: dict[str, list[tuple[Any, ...]]] = {
            "degraded": [],
            "failure": [],
            "success": [],
        }

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        spy = self

        def _record(kind: str):
            def _inner(_self: PipelineAlertManager, *args: Any, **kwargs: Any) -> None:
                spy.calls[kind].append((args, kwargs))

            return _inner

        monkeypatch.setattr(
            PipelineAlertManager, "alert_degraded_completion", _record("degraded")
        )
        monkeypatch.setattr(
            PipelineAlertManager, "alert_pipeline_failure", _record("failure")
        )
        monkeypatch.setattr(
            PipelineAlertManager, "alert_pipeline_success", _record("success")
        )


def _wire_predictions_phase(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, cache_step_raises: bool
) -> list[str]:
    """No-op every PREDICTIONS-phase body, replace the cache body, and return the step names.

    The adapter names are DERIVED from the registry (``step_`` + entry name) and each derivation is
    asserted against the registered callable before it is patched. A step whose callable is not the
    module-level adapter of that name fails here rather than silently running its real body -- so
    this test cannot quietly start ingesting odds or rebuilding gold when a future step is added.
    """
    from pipeline import orchestrator, steps

    monkeypatch.setattr(orchestrator, "LOG_PATH", tmp_path / "friday_pipeline.json")
    monkeypatch.setattr(
        "pipeline.orchestrator.get_current_nfl_week", lambda: (_SEASON, _WEEK)
    )
    monkeypatch.setattr(
        "utils.date_utils.get_current_nfl_week", lambda: (_SEASON, _WEEK)
    )

    prediction_steps = [
        step
        for step in build_step_registry()
        if step.phase is PipelinePhase.PREDICTIONS
    ]
    assert prediction_steps, (
        "the predictions phase is empty; this test would prove nothing"
    )

    names: list[str] = []
    for step in prediction_steps:
        adapter = _ADAPTER_PREFIX + step.name
        registered = getattr(steps, adapter, None)
        assert registered is step.callable, (
            f"registry entry {step.name!r} is not the module-level adapter {adapter!r}; this "
            "test patches by derived name and would otherwise run the REAL body"
        )
        names.append(step.name)

        if step.name == _CACHE_STEP and cache_step_raises:

            def _raise() -> None:
                raise RuntimeError("cache population failed")

            monkeypatch.setattr(steps, adapter, _raise)
        else:
            monkeypatch.setattr(steps, adapter, lambda: None)

    return names


@pytest.mark.integration
class TestACacheFailureDegradesTheRunRatherThanFailingIt:
    """SPEC R9 boundary, asserted on the REAL orchestrator loop."""

    def test_a_raising_cache_step_leaves_the_run_degraded_and_not_failed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The run completes with status ``degraded``; nothing raises out of ``run()``."""
        from pipeline.orchestrator import FridayPipeline

        _wire_predictions_phase(monkeypatch, tmp_path, cache_step_raises=True)
        log = FridayPipeline(mode="predictions-only", force=True).run()

        assert log.status == "degraded", (
            f"a failing NON-CRITICAL cache step produced status {log.status!r}; "
            "'failed' would mean a downstream convenience discarded good prediction output"
        )
        assert log.error is None

        cache_entry = next(s for s in log.steps if s.name == _CACHE_STEP)
        assert cache_entry.status == StepStatus.FAILED.value
        assert cache_entry.error and "cache population failed" in cache_entry.error
        assert any(_CACHE_STEP in warning for warning in log.warnings), (
            f"the non-critical failure was not recorded as a warning: {log.warnings}"
        )

    def test_the_prediction_steps_results_are_still_recorded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every step BEFORE the cache step is still recorded as a success in the run log.

        This is the half that makes non-criticality worth anything: the prediction work is kept,
        not discarded, when the cache fails.
        """
        from pipeline.orchestrator import FridayPipeline

        expected = _wire_predictions_phase(
            monkeypatch, tmp_path, cache_step_raises=True
        )
        log = FridayPipeline(mode="predictions-only", force=True).run()

        recorded = {entry.name: entry.status for entry in log.steps}
        assert set(recorded) == set(expected), (
            f"the run did not record every predictions-phase step: {sorted(recorded)}"
        )
        for name in expected:
            if name == _CACHE_STEP:
                continue
            assert recorded[name] == StepStatus.SUCCESS.value, (
                f"{name!r} was recorded as {recorded[name]!r}; a cache failure must not discard "
                "the prediction steps' results"
            )

    def test_the_degraded_completion_alert_path_is_invoked_exactly_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The EXISTING degraded alert route fires once, and the failure route does not fire.

        No new alert code exists to test: the orchestrator already routed a degraded completion
        here before this plan. What is new is that a cache failure can now reach it.
        """
        from pipeline.orchestrator import FridayPipeline

        spy = _AlertSpy()
        spy.install(monkeypatch)
        _wire_predictions_phase(monkeypatch, tmp_path, cache_step_raises=True)

        FridayPipeline(mode="predictions-only", force=True).run()

        assert len(spy.calls["degraded"]) == 1, (
            f"expected exactly one degraded-completion alert, got {len(spy.calls['degraded'])}"
        )
        assert not spy.calls["failure"], (
            "a failure alert fired for a non-critical failure"
        )
        assert not spy.calls["success"], "a success alert fired for a degraded run"

        failed_steps = spy.calls["degraded"][0][0][0]
        assert failed_steps == [_CACHE_STEP], (
            f"the degraded alert named {failed_steps!r} as the failed steps"
        )

    def test_a_clean_run_is_success_not_degraded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The control: identical wiring with the cache step succeeding reports success.

        Without it, an always-degraded bug would satisfy the assertions above and 'degraded' would
        have stopped discriminating between a broken cache and a healthy one.
        """
        from pipeline.orchestrator import FridayPipeline

        spy = _AlertSpy()
        spy.install(monkeypatch)
        _wire_predictions_phase(monkeypatch, tmp_path, cache_step_raises=False)

        log = FridayPipeline(mode="predictions-only", force=True).run()

        assert log.status == "success"
        assert len(spy.calls["success"]) == 1
        assert not spy.calls["degraded"]


@pytest.mark.integration
class TestCachePopulationOverAnEmptyPredictionSet:
    """SPEC R9 empty: an empty input set writes a valid, queryable cache rather than raising."""

    def test_an_empty_input_set_writes_a_valid_schema_and_zero_bet_rows(
        self, tmp_path: Path
    ) -> None:
        """Every loader is missing-input tolerant, so the run produces an EMPTY but usable cache.

        This is the state a first-ever Friday run on a fresh checkout reaches, and a raise there
        would abort the step (and degrade the run) for a condition that is not an error.
        """
        db_path = tmp_path / "web_cache.duckdb"
        empty_inputs = tmp_path / "empty"
        empty_inputs.mkdir()

        populate_cache(
            db_path=db_path,
            artifacts_dir=empty_inputs,
            outputs_dir=empty_inputs,
            gold_dir=empty_inputs,
            silver_dir=empty_inputs,
        )

        assert db_path.exists()
        conn = duckdb.connect(str(db_path), read_only=True)
        try:
            assert conn.execute("SELECT COUNT(*) FROM bet_list").fetchone()[0] == 0
            assert conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 0
            # The schema is present and queryable, not merely a created file.
            columns = {row[0] for row in conn.execute("DESCRIBE bet_list").fetchall()}
            assert {"game_id", "season", "week", "target", "status"} <= columns
        finally:
            conn.close()
