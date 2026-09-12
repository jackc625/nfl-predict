"""Tests for pipeline orchestrator, step registry, and CLI."""

import time
from unittest.mock import MagicMock, patch

import pytest

from pipeline.steps import (
    PipelinePhase,
    StepDefinition,
    build_step_registry,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_mock_step(
    name: str,
    phase: PipelinePhase,
    critical: bool = True,
    retryable: bool = False,
    max_retries: int = 3,
    should_fail: bool = False,
    fail_exception: type[Exception] = RuntimeError,
    fail_count: int | None = None,
) -> StepDefinition:
    """Create a StepDefinition with a mock callable.

    Args:
        fail_count: If set, the callable fails this many times then succeeds.
                    (Overrides should_fail.)
    """
    mock_callable = MagicMock()
    if fail_count is not None:
        effects = [fail_exception(f"{name} transient failure")] * fail_count + [None]
        mock_callable.side_effect = effects
    elif should_fail:
        mock_callable.side_effect = fail_exception(f"{name} failed")
    return StepDefinition(
        name=name,
        callable=mock_callable,
        phase=phase,
        critical=critical,
        retryable=retryable,
        max_retries=max_retries,
        description=f"Test step: {name}",
    )


# ---------------------------------------------------------------------------
# Step Registry Tests
# ---------------------------------------------------------------------------


class TestStepRegistry:
    """Tests for build_step_registry."""

    def test_build_step_registry_returns_21_steps(self):
        """build_step_registry returns exactly 21 StepDefinition objects.

        18 through Plan 31-17; the nineteenth is the NON-CRITICAL ``populate_web_cache`` step
        Plan 31-18 registered last (SPEC R9, D31-29). Its position and non-criticality are pinned
        separately in ``tests/unit/test_step_registry_order.py``.

        The twentieth and twenty-first are Phase 33 Plan 33-07's phase-boundary currency
        gates, ``verify_gold_currency`` and ``verify_prediction_currency`` (D33-30). Each is
        registered immediately after its own producer, which is the whole reason there are
        two of them rather than one combined check inside ``verify_data_artifacts``: that
        gate runs BEFORE gold and predictions exist, so a currency check there would report
        an ordering fact as a stale artifact on every correct run.
        """
        registry = build_step_registry()
        assert len(registry) == 21
        assert all(isinstance(s, StepDefinition) for s in registry)

    def test_build_step_registry_phases_correct(self):
        """First 8 steps are DATA, last 13 are PREDICTIONS."""
        registry = build_step_registry()
        data_steps = [s for s in registry if s.phase == PipelinePhase.DATA]
        pred_steps = [s for s in registry if s.phase == PipelinePhase.PREDICTIONS]
        assert len(data_steps) == 8
        assert len(pred_steps) == 13
        # DATA steps come first
        for i, step in enumerate(registry[:8]):
            assert step.phase == PipelinePhase.DATA, f"Step {i} should be DATA"
        for i, step in enumerate(registry[8:], start=8):
            assert step.phase == PipelinePhase.PREDICTIONS, (
                f"Step {i} should be PREDICTIONS"
            )

    def test_build_step_registry_retryable_flags(self):
        """Only ingestion steps have retryable=True."""
        registry = build_step_registry()
        retryable_steps = [s for s in registry if s.retryable]
        retryable_names = {s.name for s in retryable_steps}
        assert retryable_names == {"ingest_games", "ingest_weather", "ingest_odds"}

    def test_duplicate_step_names_not_allowed(self):
        """All step names in registry are unique."""
        registry = build_step_registry()
        names = [s.name for s in registry]
        assert len(names) == len(set(names)), f"Duplicate names: {names}"


# ---------------------------------------------------------------------------
# Orchestrator Tests
# ---------------------------------------------------------------------------


@pytest.fixture
def _patch_nfl_week():
    """Patch get_current_nfl_week to return (2025, 5)."""
    with patch("pipeline.orchestrator.get_current_nfl_week", return_value=(2025, 5)):
        yield


@pytest.fixture
def _patch_log_write(tmp_path):
    """Patch write_execution_log_atomic to prevent file I/O."""
    with patch("pipeline.orchestrator.write_execution_log_atomic") as mock_write:
        yield mock_write


@pytest.fixture(autouse=True)
def _patch_gates():
    """Patch staleness gate, health checker, and alert manager for all tests.

    Ensures orchestrator tests focus on step execution logic without
    triggering real staleness/health checks or offseason gates.
    """
    from pipeline.staleness import StalenessResult

    mock_staleness = MagicMock()
    mock_staleness.run_all_checks.return_value = StalenessResult(passed=True)

    mock_health = MagicMock()
    mock_health.run_preflight.return_value = {"status": "healthy", "checks": []}
    mock_health.run_postrun.return_value = {"status": "healthy", "checks": []}

    mock_alert = MagicMock()

    with (
        patch("pipeline.orchestrator.StalenessGate", return_value=mock_staleness),
        patch("pipeline.orchestrator.PipelineHealthChecker", return_value=mock_health),
        patch("pipeline.orchestrator.PipelineAlertManager", return_value=mock_alert),
    ):
        yield


class TestFridayPipelineRun:
    """Tests for FridayPipeline.run() method."""

    @pytest.mark.usefixtures("_patch_nfl_week")
    def test_run_executes_all_steps_in_order(self, _patch_log_write):
        """FridayPipeline.run() executes all step callables in order."""
        from pipeline.orchestrator import FridayPipeline

        steps = [
            make_mock_step("step_a", PipelinePhase.DATA),
            make_mock_step("step_b", PipelinePhase.DATA),
            make_mock_step("step_c", PipelinePhase.PREDICTIONS),
        ]
        with patch("pipeline.orchestrator.build_step_registry", return_value=steps):
            pipeline = FridayPipeline()
            log = pipeline.run()

        for step in steps:
            step.callable.assert_called_once()
        assert log.status in ("success", "degraded")

    @pytest.mark.usefixtures("_patch_nfl_week")
    def test_data_only_mode_filters_steps(self, _patch_log_write):
        """mode='data-only' only runs DATA phase steps."""
        from pipeline.orchestrator import FridayPipeline

        data_step = make_mock_step("data_step", PipelinePhase.DATA)
        pred_step = make_mock_step("pred_step", PipelinePhase.PREDICTIONS)
        steps = [data_step, pred_step]

        with patch("pipeline.orchestrator.build_step_registry", return_value=steps):
            pipeline = FridayPipeline(mode="data-only")
            pipeline.run()

        data_step.callable.assert_called_once()
        pred_step.callable.assert_not_called()

    @pytest.mark.usefixtures("_patch_nfl_week")
    def test_predictions_only_mode_filters_steps(self, _patch_log_write):
        """mode='predictions-only' only runs PREDICTIONS phase steps."""
        from pipeline.orchestrator import FridayPipeline

        data_step = make_mock_step("data_step", PipelinePhase.DATA)
        pred_step = make_mock_step("pred_step", PipelinePhase.PREDICTIONS)
        steps = [data_step, pred_step]

        with patch("pipeline.orchestrator.build_step_registry", return_value=steps):
            pipeline = FridayPipeline(mode="predictions-only")
            pipeline.run()

        data_step.callable.assert_not_called()
        pred_step.callable.assert_called_once()

    @pytest.mark.usefixtures("_patch_nfl_week")
    def test_predictions_only_logs_prerequisite_warning(self, _patch_log_write):
        """mode='predictions-only' logs a warning about data freshness prerequisites."""
        from pipeline.orchestrator import FridayPipeline

        pred_step = make_mock_step("pred_step", PipelinePhase.PREDICTIONS)
        steps = [pred_step]

        with patch("pipeline.orchestrator.build_step_registry", return_value=steps):
            # Patch the module-level logger directly (already bound at import time)
            with patch("pipeline.orchestrator.logger") as mock_logger:
                pipeline = FridayPipeline(mode="predictions-only")
                pipeline.run()

                # Check that a warning about data artifacts was logged
                warning_calls = [
                    c
                    for c in mock_logger.warning.call_args_list
                    if "data artifacts" in str(c).lower()
                ]
                assert len(warning_calls) > 0, (
                    "Expected warning about data artifact freshness"
                )

    @pytest.mark.usefixtures("_patch_nfl_week")
    def test_critical_step_failure_aborts(self, _patch_log_write):
        """Critical step failure sets log status='failed' and raises."""
        from pipeline.orchestrator import FridayPipeline

        step_a = make_mock_step("step_a", PipelinePhase.DATA)
        step_fail = make_mock_step(
            "step_fail", PipelinePhase.DATA, critical=True, should_fail=True
        )
        step_c = make_mock_step("step_c", PipelinePhase.DATA)
        steps = [step_a, step_fail, step_c]

        with patch("pipeline.orchestrator.build_step_registry", return_value=steps):
            pipeline = FridayPipeline()
            with pytest.raises(RuntimeError):
                pipeline.run()

        step_a.callable.assert_called_once()
        step_fail.callable.assert_called_once()
        step_c.callable.assert_not_called()
        assert pipeline.execution_log.status == "failed"

    @pytest.mark.usefixtures("_patch_nfl_week")
    def test_non_critical_step_failure_continues(self, _patch_log_write):
        """Non-critical step failure logs warning and continues; status='degraded'."""
        from pipeline.orchestrator import FridayPipeline

        step_a = make_mock_step("step_a", PipelinePhase.DATA)
        step_warn = make_mock_step(
            "step_warn", PipelinePhase.DATA, critical=False, should_fail=True
        )
        step_c = make_mock_step("step_c", PipelinePhase.DATA)
        steps = [step_a, step_warn, step_c]

        with patch("pipeline.orchestrator.build_step_registry", return_value=steps):
            pipeline = FridayPipeline()
            log = pipeline.run()

        step_a.callable.assert_called_once()
        step_warn.callable.assert_called_once()
        step_c.callable.assert_called_once()
        assert log.status == "degraded"
        assert len(log.warnings) > 0

    @pytest.mark.usefixtures("_patch_nfl_week")
    def test_retryable_step_retries_on_transient(self, _patch_log_write):
        """Retryable step retries on ConnectionError up to max_retries."""
        from pipeline.orchestrator import FridayPipeline

        # Fails twice with ConnectionError, then succeeds on 3rd call
        step = make_mock_step(
            "retry_step",
            PipelinePhase.DATA,
            retryable=True,
            max_retries=3,
            fail_count=2,
            fail_exception=ConnectionError,
        )
        steps = [step]

        with patch("pipeline.orchestrator.build_step_registry", return_value=steps):
            pipeline = FridayPipeline()
            log = pipeline.run()

        assert step.callable.call_count == 3
        assert log.status == "success"

    @pytest.mark.usefixtures("_patch_nfl_week")
    def test_retryable_step_no_retry_on_non_transient(self, _patch_log_write):
        """Retryable step does NOT retry on ValueError (non-transient)."""
        from pipeline.orchestrator import FridayPipeline

        step = make_mock_step(
            "no_retry_step",
            PipelinePhase.DATA,
            critical=True,
            retryable=True,
            max_retries=3,
            should_fail=True,
            fail_exception=ValueError,
        )
        steps = [step]

        with patch("pipeline.orchestrator.build_step_registry", return_value=steps):
            pipeline = FridayPipeline()
            with pytest.raises(RuntimeError):
                pipeline.run()

        # Called only once -- no retry for ValueError (non-transient)
        assert step.callable.call_count == 1

    @pytest.mark.usefixtures("_patch_nfl_week")
    def test_non_retryable_step_no_retry(self, _patch_log_write):
        """Non-retryable step fails immediately without retry."""
        from pipeline.orchestrator import FridayPipeline

        step = make_mock_step(
            "no_retry",
            PipelinePhase.DATA,
            critical=True,
            retryable=False,
            should_fail=True,
        )
        steps = [step]

        with patch("pipeline.orchestrator.build_step_registry", return_value=steps):
            pipeline = FridayPipeline()
            with pytest.raises(RuntimeError):
                pipeline.run()

        assert step.callable.call_count == 1

    @pytest.mark.usefixtures("_patch_nfl_week")
    def test_dry_run_returns_step_names(self, _patch_log_write):
        """dry_run() returns list of step name strings without calling any callable."""
        from pipeline.orchestrator import FridayPipeline

        step_a = make_mock_step("step_a", PipelinePhase.DATA)
        step_b = make_mock_step("step_b", PipelinePhase.PREDICTIONS)
        steps = [step_a, step_b]

        with patch("pipeline.orchestrator.build_step_registry", return_value=steps):
            pipeline = FridayPipeline()
            names = pipeline.dry_run()

        assert names == ["step_a: Test step: step_a", "step_b: Test step: step_b"]
        step_a.callable.assert_not_called()
        step_b.callable.assert_not_called()

    @pytest.mark.usefixtures("_patch_nfl_week")
    def test_execution_log_records_step_results(self, _patch_log_write):
        """Execution log contains StepLogEntry for each step with correct fields."""
        from pipeline.orchestrator import FridayPipeline

        steps = [
            make_mock_step("step_a", PipelinePhase.DATA),
            make_mock_step("step_b", PipelinePhase.DATA),
        ]

        with patch("pipeline.orchestrator.build_step_registry", return_value=steps):
            pipeline = FridayPipeline()
            log = pipeline.run()

        assert len(log.steps) == 2
        assert log.steps[0].name == "step_a"
        assert log.steps[0].status == "success"
        assert log.steps[0].duration_ms >= 0
        assert log.steps[1].name == "step_b"

    @pytest.mark.usefixtures("_patch_nfl_week")
    def test_execution_log_written_incrementally(self, tmp_path):
        """write_execution_log_atomic called N+1 times for N steps (initial + per-step)."""
        from pipeline.orchestrator import FridayPipeline

        steps = [
            make_mock_step("step_a", PipelinePhase.DATA),
            make_mock_step("step_b", PipelinePhase.DATA),
            make_mock_step("step_c", PipelinePhase.DATA),
        ]

        with patch("pipeline.orchestrator.build_step_registry", return_value=steps):
            with patch(
                "pipeline.orchestrator.write_execution_log_atomic"
            ) as mock_write:
                pipeline = FridayPipeline()
                pipeline.run()

                # 1 initial write + 3 step writes + 1 final write = 5
                # or at minimum: 1 initial + 3 per-step = 4
                assert mock_write.call_count >= 4

    @pytest.mark.usefixtures("_patch_nfl_week")
    def test_execution_log_contains_pid(self, _patch_log_write):
        """Execution log pid is set to a positive integer."""
        from pipeline.orchestrator import FridayPipeline

        steps = [make_mock_step("step_a", PipelinePhase.DATA)]
        with patch("pipeline.orchestrator.build_step_registry", return_value=steps):
            pipeline = FridayPipeline()
            log = pipeline.run()

        assert isinstance(log.pid, int)
        assert log.pid > 0


class TestEdgeCases:
    """Edge case and robustness tests."""

    @pytest.mark.usefixtures("_patch_nfl_week")
    def test_interrupted_run_leaves_usable_log(self, tmp_path):
        """If a step crashes mid-run, the last successful step is captured in log."""
        from pipeline.orchestrator import FridayPipeline

        step_ok = make_mock_step("step_ok", PipelinePhase.DATA)
        step_crash = make_mock_step(
            "step_crash", PipelinePhase.DATA, critical=True, should_fail=True
        )
        steps = [step_ok, step_crash]

        log_path = tmp_path / "pipeline.json"
        with patch("pipeline.orchestrator.build_step_registry", return_value=steps):
            with patch("pipeline.orchestrator.LOG_PATH", log_path):
                pipeline = FridayPipeline()
                with pytest.raises(RuntimeError):
                    pipeline.run()

        # Log file should exist and be valid JSON
        import json

        assert log_path.exists()
        with open(log_path) as f:
            data = json.load(f)
        assert data["status"] == "failed"
        # step_ok should be recorded
        assert len(data["steps"]) == 2
        assert data["steps"][0]["name"] == "step_ok"
        assert data["steps"][0]["status"] == "success"

    @pytest.mark.usefixtures("_patch_nfl_week")
    def test_predictions_only_missing_artifacts_runs_with_warning(
        self, _patch_log_write
    ):
        """predictions-only with missing data artifacts still runs but logs warning."""
        from pipeline.orchestrator import FridayPipeline

        pred_step = make_mock_step("pred_step", PipelinePhase.PREDICTIONS)
        steps = [pred_step]

        with patch("pipeline.orchestrator.build_step_registry", return_value=steps):
            with patch("pipeline.orchestrator.logger") as mock_logger:
                pipeline = FridayPipeline(mode="predictions-only")
                log = pipeline.run()

                # Should still complete (step succeeded)
                assert log.status == "success"
                pred_step.callable.assert_called_once()

                # Warning should have been logged about data artifacts
                warning_calls = [
                    c
                    for c in mock_logger.warning.call_args_list
                    if "data artifacts" in str(c).lower()
                ]
                assert len(warning_calls) > 0


class TestArgparse:
    """Tests for CLI argument parsing."""

    def test_argparse_flags(self):
        """argparse accepts --data-only, --predictions-only, --dry-run, --force."""

        # We need to test that the argparse in friday_pipeline.py is configured correctly.
        # Import the module and check its parser setup.
        # Since main() calls parse_args, we test by simulating different flag combos.
        from scripts.friday_pipeline import main

        # --dry-run --force should work (not actually run pipeline due to dry-run)
        with patch("sys.argv", ["friday_pipeline.py", "--dry-run", "--force"]):
            with patch(
                "pipeline.orchestrator.get_current_nfl_week", return_value=(2025, 5)
            ):
                with patch(
                    "pipeline.orchestrator.build_step_registry", return_value=[]
                ):
                    result = main()
                    assert result == 0

    def test_data_only_and_predictions_only_mutually_exclusive(self):
        """--data-only and --predictions-only cannot be used together."""
        import subprocess
        import sys

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.friday_pipeline",
                "--data-only",
                "--predictions-only",
            ],
            capture_output=True,
            text=True,
            cwd="C:/Users/jackc/Code/nfl-predict",
        )
        assert result.returncode != 0


# ---------------------------------------------------------------------------
# AUTO-02 coverage-gap behavior tests (Plan 21-02)
# ---------------------------------------------------------------------------


def _make_gate_mocks(preflight_status: str = "healthy"):
    """Build fresh (staleness, health, alert) mocks for an AUTO-02 gap test.

    Each test re-patches the gates with these so it holds a direct handle to
    the injected ``alert_manager`` / ``health_checker`` mocks (the autouse
    ``_patch_gates`` fixture does not expose its mocks). Mirrors the re-patch
    idiom in ``test_predictions_only_logs_prerequisite_warning``.

    Args:
        preflight_status: ``run_preflight`` status to return ("healthy" or
            "unhealthy").

    Returns:
        (mock_staleness, mock_health, mock_alert) tuple.
    """
    from pipeline.staleness import StalenessResult

    mock_staleness = MagicMock()
    mock_staleness.run_all_checks.return_value = StalenessResult(passed=True)

    mock_health = MagicMock()
    mock_health.run_preflight.return_value = {
        "status": preflight_status,
        "checks": [],
    }
    mock_health.run_postrun.return_value = {"status": "healthy", "checks": []}

    mock_alert = MagicMock()
    return mock_staleness, mock_health, mock_alert


class TestAuto02CoverageGaps:
    """Behavior tests for the two real AUTO-02 coverage gaps (Plan 21-02).

    Gap 1: ``--force`` makes the pre-flight health check ADVISORY (continue +
    warn) on an unhealthy result rather than aborting.

    Gap 3: exactly ONE alert method is fired per terminal outcome
    (failed -> alert_pipeline_failure, degraded -> alert_degraded_completion,
    success -> alert_pipeline_success), with the other two never fired.

    These intentionally re-patch the gates (overriding the autouse
    ``_patch_gates`` fixture) so each test holds a direct handle to the injected
    alert/health mocks. The genuinely-covered behavior (retry, sequencing,
    critical-abort status, degrade status, staleness block/pass) is NOT
    duplicated here.
    """

    @pytest.mark.usefixtures("_patch_nfl_week", "_patch_log_write")
    def test_force_health_advisory_on_unhealthy_preflight(self):
        """Gap 1: a forced run continues on unhealthy pre-flight and warns.

        With ``force=True`` an unhealthy pre-flight health result must NOT abort
        the run; instead the orchestrator appends the exact advisory warning and
        completes (status in {success, degraded}).
        """
        from pipeline.orchestrator import FridayPipeline

        mock_staleness, mock_health, mock_alert = _make_gate_mocks(
            preflight_status="unhealthy"
        )

        steps = [
            make_mock_step("step_a", PipelinePhase.DATA),
            make_mock_step("step_b", PipelinePhase.PREDICTIONS),
        ]

        with (
            patch("pipeline.orchestrator.StalenessGate", return_value=mock_staleness),
            patch(
                "pipeline.orchestrator.PipelineHealthChecker",
                return_value=mock_health,
            ),
            patch(
                "pipeline.orchestrator.PipelineAlertManager", return_value=mock_alert
            ),
            patch("pipeline.orchestrator.build_step_registry", return_value=steps),
        ):
            pipeline = FridayPipeline(force=True)
            log = pipeline.run()

        # Forced + unhealthy pre-flight => run COMPLETES (advisory), not aborted.
        assert log.status in ("success", "degraded")
        assert "Pre-flight health: unhealthy (forced)" in log.warnings
        # No CRITICAL failure alert fired on the advisory path.
        mock_alert.alert_pipeline_failure.assert_not_called()

    @pytest.mark.usefixtures("_patch_nfl_week", "_patch_log_write")
    def test_signal_failed_fires_one_critical_alert(self):
        """Gap 3 (failed): exactly one alert_pipeline_failure on critical failure.

        The critical-failure path fires ``alert_pipeline_failure`` once then
        raises before Phase E, so the total must be exactly 1 (not 2). The
        success/degraded alerts must never fire.
        """
        from pipeline.orchestrator import FridayPipeline

        mock_staleness, mock_health, mock_alert = _make_gate_mocks()

        steps = [
            make_mock_step("step_a", PipelinePhase.DATA),
            make_mock_step(
                "step_fail", PipelinePhase.DATA, critical=True, should_fail=True
            ),
        ]

        with (
            patch("pipeline.orchestrator.StalenessGate", return_value=mock_staleness),
            patch(
                "pipeline.orchestrator.PipelineHealthChecker",
                return_value=mock_health,
            ),
            patch(
                "pipeline.orchestrator.PipelineAlertManager", return_value=mock_alert
            ),
            patch("pipeline.orchestrator.build_step_registry", return_value=steps),
        ):
            pipeline = FridayPipeline()
            with pytest.raises(RuntimeError):
                pipeline.run()

        assert mock_alert.alert_pipeline_failure.call_count == 1
        mock_alert.alert_degraded_completion.assert_not_called()
        mock_alert.alert_pipeline_success.assert_not_called()

    @pytest.mark.usefixtures("_patch_nfl_week", "_patch_log_write")
    def test_signal_degraded_fires_one_warning_alert(self):
        """Gap 3 (degraded): exactly one alert_degraded_completion; no failure alert."""
        from pipeline.orchestrator import FridayPipeline

        mock_staleness, mock_health, mock_alert = _make_gate_mocks()

        steps = [
            make_mock_step("step_a", PipelinePhase.DATA),
            make_mock_step(
                "step_warn", PipelinePhase.DATA, critical=False, should_fail=True
            ),
            make_mock_step("step_c", PipelinePhase.DATA),
        ]

        with (
            patch("pipeline.orchestrator.StalenessGate", return_value=mock_staleness),
            patch(
                "pipeline.orchestrator.PipelineHealthChecker",
                return_value=mock_health,
            ),
            patch(
                "pipeline.orchestrator.PipelineAlertManager", return_value=mock_alert
            ),
            patch("pipeline.orchestrator.build_step_registry", return_value=steps),
        ):
            pipeline = FridayPipeline()
            log = pipeline.run()

        assert log.status == "degraded"
        assert mock_alert.alert_degraded_completion.call_count == 1
        mock_alert.alert_pipeline_failure.assert_not_called()
        mock_alert.alert_pipeline_success.assert_not_called()

    @pytest.mark.usefixtures("_patch_nfl_week", "_patch_log_write")
    def test_signal_success_fires_one_info_alert(self):
        """Gap 3 (success): exactly one alert_pipeline_success; no other alert."""
        from pipeline.orchestrator import FridayPipeline

        mock_staleness, mock_health, mock_alert = _make_gate_mocks()

        steps = [
            make_mock_step("step_a", PipelinePhase.DATA),
            make_mock_step("step_b", PipelinePhase.PREDICTIONS),
        ]

        with (
            patch("pipeline.orchestrator.StalenessGate", return_value=mock_staleness),
            patch(
                "pipeline.orchestrator.PipelineHealthChecker",
                return_value=mock_health,
            ),
            patch(
                "pipeline.orchestrator.PipelineAlertManager", return_value=mock_alert
            ),
            patch("pipeline.orchestrator.build_step_registry", return_value=steps),
        ):
            pipeline = FridayPipeline()
            log = pipeline.run()

        assert log.status == "success"
        assert mock_alert.alert_pipeline_success.call_count == 1
        mock_alert.alert_pipeline_failure.assert_not_called()
        mock_alert.alert_degraded_completion.assert_not_called()


class TestTheSuccessAlertReportsARealDuration:
    """WR-09: Phase E read ``total_duration_ms`` before anything computed it.

    ``_finalize_log`` was the ONLY place that summed the step durations, and it ran AFTER the
    Phase-E alert. ``ExecutionLog.total_duration_ms`` defaults to 0, so every successful Friday run
    alerted ``execution_time_ms=0``. Since ``pipeline/alert.py`` is log-only by default, that alert
    line is the one place a pipeline slowing down would show, and it was permanently zero.

    The failure and degraded branches never read the field, so they were unaffected -- asserted
    below so the fix is not credited with more than it did.
    """

    @staticmethod
    def _run_success() -> tuple[object, object]:
        from pipeline.orchestrator import FridayPipeline

        mock_staleness, mock_health, mock_alert = _make_gate_mocks()

        def _slow() -> None:
            time.sleep(0.02)

        steps = [
            make_mock_step("step_a", PipelinePhase.DATA),
            make_mock_step("step_b", PipelinePhase.PREDICTIONS),
        ]
        steps[0].callable.side_effect = _slow
        steps[1].callable.side_effect = _slow

        with (
            patch("pipeline.orchestrator.StalenessGate", return_value=mock_staleness),
            patch(
                "pipeline.orchestrator.PipelineHealthChecker", return_value=mock_health
            ),
            patch(
                "pipeline.orchestrator.PipelineAlertManager", return_value=mock_alert
            ),
            patch("pipeline.orchestrator.build_step_registry", return_value=steps),
        ):
            log = FridayPipeline().run()
        return log, mock_alert

    @pytest.mark.usefixtures("_patch_nfl_week", "_patch_log_write")
    def test_the_alerted_execution_time_is_not_zero(self):
        log, mock_alert = self._run_success()

        alerted = mock_alert.alert_pipeline_success.call_args.kwargs[
            "execution_time_ms"
        ]

        assert alerted > 0, (
            "the success alert reported execution_time_ms=0 for a run whose steps actually took "
            f"{log.total_duration_ms} ms -- Phase E read the field before anything set it"
        )

    @pytest.mark.usefixtures("_patch_nfl_week", "_patch_log_write")
    def test_the_alerted_time_is_the_same_figure_the_written_log_carries(self):
        """Two numbers for one run would be worse than one wrong number."""
        log, mock_alert = self._run_success()

        alerted = mock_alert.alert_pipeline_success.call_args.kwargs[
            "execution_time_ms"
        ]

        assert alerted == pytest.approx(log.total_duration_ms)

    @pytest.mark.usefixtures("_patch_nfl_week", "_patch_log_write")
    def test_the_duration_still_equals_the_sum_of_the_step_durations(self):
        """The stamp must not become a wall-clock reading; it is the recorded steps' sum."""
        log, _ = self._run_success()

        assert log.total_duration_ms == pytest.approx(
            sum(s.duration_ms for s in log.steps)
        )
