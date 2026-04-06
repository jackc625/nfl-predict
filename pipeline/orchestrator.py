"""Friday pipeline orchestrator.

Runs all 18 data-to-prediction steps in sequence with retry logic,
phase filtering, and structured execution logging.

Integrates pre-flight staleness gate, health checks, and post-run health
checks with exactly-one alerting per pipeline outcome.

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

from pipeline.alert import PipelineAlertManager
from pipeline.execution_log import (
    ExecutionLog,
    StepLogEntry,
    write_execution_log_atomic,
)
from pipeline.health import PipelineHealthChecker
from pipeline.staleness import StalenessGate
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

    Integrates:
    - StalenessGate: season + data freshness checks (bypassed by --force)
    - PipelineHealthChecker: pre-flight and post-run health (always runs)
    - PipelineAlertManager: exactly one alert per pipeline outcome
    """

    def __init__(self, force: bool = False, mode: str = "full") -> None:
        """Initialize the pipeline.

        Args:
            force: Bypass pre-flight staleness/season checks.
                   Health checks still run in advisory mode when forced.
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

        # Integration components from Plan 02
        self.staleness_gate = StalenessGate(season, week, force=force)
        self.health_checker = PipelineHealthChecker()
        self.alert_manager = PipelineAlertManager()

    # -- Public properties ---------------------------------------------------

    @property
    def execution_log(self) -> ExecutionLog:
        """Access the current execution log state."""
        return self._log

    @property
    def season(self) -> int:
        """NFL season year for this pipeline run."""
        return self._log.season

    @property
    def week(self) -> int:
        """NFL week number for this pipeline run."""
        return self._log.week

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

    # -- Finalization helpers ------------------------------------------------

    def _finalize_log(self) -> None:
        """Set end_time, total_duration_ms, and write log atomically."""
        self._log.end_time = datetime.now(ET).isoformat()
        self._log.total_duration_ms = sum(s.duration_ms for s in self._log.steps)
        write_execution_log_atomic(self._log, LOG_PATH)

    # -- Main run loop -------------------------------------------------------

    def run(self) -> ExecutionLog:
        """Execute the pipeline with pre-flight gates, steps, and post-run checks.

        Flow:
        A. Pre-flight staleness gate (bypassed by --force)
        B. Pre-flight health check (ALWAYS runs; advisory with --force)
        C. Step execution (existing logic from Plan 01)
        D. Post-run health check (comprehensive, advisory only)
        E. Completion alerting (exactly ONE alert per outcome)

        Returns:
            Final ExecutionLog with run results.

        Raises:
            RuntimeError: On critical step failure or gate failure.
        """
        # ---------------------------------------------------------------
        # Phase A: Pre-flight staleness gate (bypassed by --force)
        # ---------------------------------------------------------------
        staleness_result = self.staleness_gate.run_all_checks()
        if staleness_result.warnings:
            self._log.warnings.extend(staleness_result.warnings)
            self.alert_manager.alert_staleness_warning(
                staleness_result.warnings, self.season, self.week
            )
        if not staleness_result.passed:
            self._log.status = "failed"
            self._log.error = (
                f"Pre-flight staleness checks failed: "
                f"{'; '.join(staleness_result.errors)}"
            )
            self._finalize_log()
            raise RuntimeError(self._log.error)

        # ---------------------------------------------------------------
        # Phase B: Pre-flight health check (ALWAYS runs, even with --force)
        # With --force: health results are advisory (warn but don't abort)
        # Without --force: health failure aborts pipeline
        # ---------------------------------------------------------------
        preflight = self.health_checker.run_preflight()
        if preflight["status"] == "unhealthy":
            if self.force:
                logger.warning(
                    "Pre-flight health check unhealthy (--force: continuing anyway)"
                )
                self._log.warnings.append("Pre-flight health: unhealthy (forced)")
            else:
                self._log.status = "failed"
                self._log.error = "Pre-flight health check failed"
                self._finalize_log()
                self.alert_manager.alert_pipeline_failure(
                    "Pre-flight health check failed",
                    0,
                    0,
                    self.season,
                    self.week,
                )
                raise RuntimeError("Pre-flight health check failed")

        # ---------------------------------------------------------------
        # Phase C: Step execution
        # ---------------------------------------------------------------
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
                    self._finalize_log()
                    # Exactly ONE alert: failure
                    self.alert_manager.alert_pipeline_failure(
                        result.error or f"Critical step failed: {step.name}",
                        len(
                            [
                                s
                                for s in self._log.steps
                                if s.status == StepStatus.SUCCESS.value
                            ]
                        ),
                        len(
                            [
                                s
                                for s in self._log.steps
                                if s.status == StepStatus.FAILED.value
                            ]
                        ),
                        self.season,
                        self.week,
                    )
                    raise RuntimeError(
                        result.error or f"Critical step failed: {step.name}"
                    )

                has_non_critical_failure = True
                self._log.warnings.append(
                    f"Non-critical step '{step.name}' failed: {result.error}"
                )

        # Final step status
        if has_non_critical_failure:
            self._log.status = "degraded"
        else:
            self._log.status = "success"

        # ---------------------------------------------------------------
        # Phase D: Post-run health check (comprehensive, advisory only)
        # Does NOT change pipeline status or exit code (review: Codex)
        # ---------------------------------------------------------------
        postrun = self.health_checker.run_postrun()
        if postrun["status"] == "unhealthy":
            self._log.warnings.append("Post-run health check: unhealthy")

        # ---------------------------------------------------------------
        # Phase E: Completion alerting (exactly ONE alert per outcome)
        # ---------------------------------------------------------------
        failed_steps = [
            s.name for s in self._log.steps if s.status == StepStatus.FAILED.value
        ]
        if self._log.status == "failed":
            # Already alerted above during critical failure, but this branch
            # handles the theoretical case of a non-raise failure path.
            self.alert_manager.alert_pipeline_failure(
                self._log.error or "Unknown error",
                len(
                    [s for s in self._log.steps if s.status == StepStatus.SUCCESS.value]
                ),
                len(failed_steps),
                self.season,
                self.week,
            )
        elif self._log.status == "degraded":
            self.alert_manager.alert_degraded_completion(
                failed_steps,
                len(
                    [s for s in self._log.steps if s.status == StepStatus.SUCCESS.value]
                ),
                self.season,
                self.week,
            )
        else:  # success
            self.alert_manager.alert_pipeline_success(
                predictions_count=sum(
                    1
                    for s in self._log.steps
                    if s.name == "generate_predictions"
                    and s.status == StepStatus.SUCCESS.value
                ),
                execution_time_ms=self._log.total_duration_ms,
                warnings=self._log.warnings,
                season=self.season,
                week=self.week,
            )

        self._finalize_log()

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
