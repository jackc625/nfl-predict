"""Tests for pipeline orchestrator, step registry, and CLI."""

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

    def test_build_step_registry_returns_18_steps(self):
        """build_step_registry returns exactly 18 StepDefinition objects."""
        registry = build_step_registry()
        assert len(registry) == 18
        assert all(isinstance(s, StepDefinition) for s in registry)

    def test_build_step_registry_phases_correct(self):
        """First 8 steps are DATA, last 10 are PREDICTIONS."""
        registry = build_step_registry()
        data_steps = [s for s in registry if s.phase == PipelinePhase.DATA]
        pred_steps = [s for s in registry if s.phase == PipelinePhase.PREDICTIONS]
        assert len(data_steps) == 8
        assert len(pred_steps) == 10
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
