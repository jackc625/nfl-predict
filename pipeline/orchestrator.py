"""Friday pipeline orchestrator.

Runs all 18 data-to-prediction steps in sequence with retry logic,
phase filtering, and structured execution logging.

Per D-09: No checkpoint-based resume. Always start fresh on re-run.
Per D-23: Uses existing get_logger() for all logging.
"""

import os
import time
from datetime import datetime
from pathlib import Path

from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from pipeline.execution_log import (
    ExecutionLog,
    StepLogEntry,
    write_execution_log_atomic,
)
from pipeline.steps import (
    TRANSIENT_EXCEPTIONS,
    PipelinePhase,
    StepDefinition,
    StepResult,
    StepStatus,
    build_step_registry,
)
from utils.date_utils import ET, get_current_nfl_week
from utils.logging_config import get_logger

# Consistent absolute path for the execution log (D-05)
LOG_PATH = Path("logs/friday_pipeline.json").resolve()

logger = get_logger(__name__)


class FridayPipeline:
    """Unified Friday pipeline orchestrator.

    Replaces both friday_data_update.py and friday_predictions_run.py
    with a single process that runs all steps via direct Python imports.
    """

    def __init__(self, force: bool = False, mode: str = "full") -> None:
        """Initialize the pipeline.

        Args:
            force: Bypass pre-flight staleness/season checks.
            mode: One of 'full', 'data-only', 'predictions-only'.
        """
        self.force = force
        self.mode = mode

        season, week = get_current_nfl_week()

        self._log = ExecutionLog(
            status="running",
            start_time=datetime.now(ET).isoformat(),
            season=season,
            week=week,
            forced=force,
            mode=mode,
            pid=os.getpid(),
        )

    # -- Public properties ---------------------------------------------------

    @property
    def execution_log(self) -> ExecutionLog:
        """Access the current execution log state."""
        return self._log

    # -- Phase filtering -----------------------------------------------------

    def _filter_steps(self, steps: list[StepDefinition]) -> list[StepDefinition]:
        """Filter steps based on the selected mode.

        Args:
            steps: Full step registry.

        Returns:
            Filtered list of steps for the current mode.
        """
        if self.mode == "data-only":
            return [s for s in steps if s.phase == PipelinePhase.DATA]
        if self.mode == "predictions-only":
            logger.warning(
                "Running predictions-only mode -- ensure data artifacts are fresh"
            )
            return [s for s in steps if s.phase == PipelinePhase.PREDICTIONS]
        return steps

    # -- Single-step execution -----------------------------------------------

    def _execute_step(self, step: StepDefinition) -> StepResult:
        """Execute a single pipeline step, with optional retry.

        Retryable steps use tenacity with exception-type filtering so that
        only transient network errors (ConnectionError, TimeoutError, OSError,
        httpx.HTTPStatusError) trigger retries. Local/validation errors
        (ValueError, TypeError, etc.) fail fast.

        Args:
            step: The step definition to execute.

        Returns:
            StepResult with timing, status, and retry metadata.
        """
        logger.info("Starting step", step=step.name, description=step.description)
        start = time.perf_counter()
        retry_count = 0

        try:
            if step.retryable and step.max_retries > 1:
                for attempt in Retrying(
                    stop=stop_after_attempt(step.max_retries),
                    wait=wait_exponential(multiplier=2, min=2, max=60),
                    retry=retry_if_exception_type(TRANSIENT_EXCEPTIONS),
                    reraise=True,
                ):
                    with attempt:
                        step.callable()
                        retry_count = attempt.retry_state.attempt_number - 1
            else:
                step.callable()

            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.info(
                "Step completed",
                step=step.name,
                duration_ms=round(elapsed_ms, 1),
                retries=retry_count,
            )
            return StepResult(
                name=step.name,
                status=StepStatus.SUCCESS,
                duration_ms=elapsed_ms,
                retry_count=retry_count,
            )

        except Exception as exc:  # noqa: BLE001 -- must catch all to record in log
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.error(
                "Step failed",
                step=step.name,
                error=str(exc),
                duration_ms=round(elapsed_ms, 1),
            )
            return StepResult(
                name=step.name,
                status=StepStatus.FAILED,
                duration_ms=elapsed_ms,
                retry_count=retry_count,
                error=str(exc),
            )

    # -- Main run loop -------------------------------------------------------

    def run(self) -> ExecutionLog:
        """Execute the pipeline.

        1. Build & filter step registry.
        2. Write initial execution log.
        3. Iterate steps; write incremental log after each.
        4. Abort on critical failure; continue on non-critical.

        Returns:
            Final ExecutionLog with run results.

        Raises:
            RuntimeError: On critical step failure (after recording in log).
        """
        all_steps = build_step_registry()
        steps = self._filter_steps(all_steps)

        # Write initial log snapshot
        write_execution_log_atomic(self._log, LOG_PATH)

        has_non_critical_failure = False

        for step in steps:
            result = self._execute_step(step)

            # Record in log
            self._log.steps.append(
                StepLogEntry(
                    name=result.name,
                    status=result.status.value,
                    duration_ms=result.duration_ms,
                    retry_count=result.retry_count,
                    error=result.error,
                )
            )

            # Write incremental log snapshot after each step
            write_execution_log_atomic(self._log, LOG_PATH)

            if result.status == StepStatus.FAILED:
                if step.critical:
                    self._log.status = "failed"
                    self._log.error = result.error
                    self._log.end_time = datetime.now(ET).isoformat()
                    self._log.total_duration_ms = sum(
                        s.duration_ms for s in self._log.steps
                    )
                    write_execution_log_atomic(self._log, LOG_PATH)
                    raise RuntimeError(
                        result.error or f"Critical step failed: {step.name}"
                    )

                has_non_critical_failure = True
                self._log.warnings.append(
                    f"Non-critical step '{step.name}' failed: {result.error}"
                )

        # Final status
        if has_non_critical_failure:
            self._log.status = "degraded"
        else:
            self._log.status = "success"

        self._log.end_time = datetime.now(ET).isoformat()
        self._log.total_duration_ms = sum(s.duration_ms for s in self._log.steps)

        # Final log write
        write_execution_log_atomic(self._log, LOG_PATH)

        logger.info(
            "Pipeline completed",
            status=self._log.status,
            total_duration_ms=round(self._log.total_duration_ms, 1),
            steps_run=len(self._log.steps),
            warnings=len(self._log.warnings),
        )
        return self._log

    # -- Dry-run mode --------------------------------------------------------

    def dry_run(self) -> list[str]:
        """List pipeline steps without executing them.

        Returns:
            List of 'name: description' strings.
        """
        all_steps = build_step_registry()
        steps = self._filter_steps(all_steps)
        return [f"{s.name}: {s.description}" for s in steps]
