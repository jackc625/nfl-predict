"""Friday pipeline orchestrator.

Runs all 19 data-to-prediction steps in sequence with retry logic,
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

from pipeline import live_skip
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
    RunStatus,
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

    def __init__(
        self,
        force: bool = False,
        mode: str = "full",
        history_mode: bool = False,
        steps: list[StepDefinition] | None = None,
    ) -> None:
        """Initialize the pipeline.

        Args:
            force: Bypass pre-flight staleness/season checks.
                   Health checks still run in advisory mode when forced.
            mode: One of 'full', 'data-only', 'predictions-only'.
            history_mode: True for a HISTORY (training/gold) run, which must stop and save
                nothing on an information-time refusal (D33.2-05, first half). STATED
                EXPLICITLY, never inferred from ``mode``: ``mode`` selects which PHASES run,
                and a predictions-only live night and a full live night are both LIVE. The
                default is the live run this orchestrator exists for; a caller building
                history must say so.
            steps: The steps to run instead of the Friday registry -- the daily lock-time run
                passes its own (Plan 33.2-27). ``None`` runs ``build_step_registry()``.
        """
        self.force = force
        self._steps = steps
        self.mode = mode
        self.history_mode = history_mode

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
        # The skip record's identity for this run: the run id is its ISO start instant, and
        # the natural key's day is its EASTERN calendar date (start_time is an ET instant).
        self._run_date_et = (
            datetime.fromisoformat(self._log.start_time).date().isoformat()
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
            retry_count = self._invoke_with_live_skip(step)

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

    def _invoke(self, step: StepDefinition) -> int:
        """Call the step once (with transient-error retries); return the retry count."""
        if step.retryable and step.max_retries > 1:
            retry_count = 0
            for attempt in Retrying(
                stop=stop_after_attempt(step.max_retries),
                wait=wait_exponential(multiplier=2, min=2, max=60),
                retry=retry_if_exception_type(TRANSIENT_EXCEPTIONS),
                reraise=True,
            ):
                with attempt:
                    step.callable()
                    retry_count = attempt.retry_state.attempt_number - 1
            return retry_count
        step.callable()
        return 0

    def _invoke_with_live_skip(self, step: StepDefinition) -> int:
        """Run the step; on a per-game refusal in a LIVE run, drop the games and re-run.

        THE STEP SEAM D33.2-05'S LIVE HALF NEEDS (Plan 33.2-03). ``_execute_step`` turns
        every exception into a FAILED step, and a critical step's failure ends the run -- so,
        before this, one post-lock value denied the whole night's clean games their
        predictions (RESEARCH P10). A refusal from ``live_skip.LIVE_SKIP_EXCEPTIONS`` is now
        caught HERE: its games are recorded durably and excluded
        (``live_skip.apply_skip_policy``) and the step runs again on the remainder. Anything
        outside that closed set falls through untouched to today's handling.

        BOUNDED. At most ``live_skip.max_skip_rounds()`` skip rounds (the feature-source count
        plus one, derived from the registry); a refusal naming an already-excluded game raises
        ``SkipNotConvergingError`` on the first repeat. Either way the step FAILS, loudly.

        THE COST, STATED RATHER THAN DISCOVERED IN PRODUCTION. ``step_build_features`` rebuilds
        ALL seasons (measured 591.5 s), so a skip round inside it repeats that whole build. The
        owner ruled on 2026-09-23 to keep the full rebuild nightly (a one-season build diverges
        from it on 139 of 186 columns), superseding D33.2-19's scoped build, so the daily run's
        start time must leave room for a round. A refusal from a HISTORICAL row drops that
        historical game (named in the skip record) rather than tonight's slate.

        Returns:
            The transient-retry count of the final, successful call.
        """
        # HISTORY MODE FIRST, and it never consults the policy: a training/gold build must stop
        # and save nothing (D33.2-05, first half), so the refusal propagates to _execute_step's
        # ordinary handler exactly as it did before this seam existed.
        if self.history_mode:
            return self._invoke(step)

        cap: int | None = None
        rounds = 0
        while True:
            try:
                return self._invoke(step)
            except live_skip.LIVE_SKIP_EXCEPTIONS as refusal:
                if cap is None:
                    cap = live_skip.max_skip_rounds()
                rounds += 1
                if rounds > cap:
                    msg = (
                        f"the live skip did not converge in step {step.name!r}: {cap} skip "
                        "round(s) were taken and it refused again. Excluded so far: "
                        f"{sorted(live_skip.excluded_games())}. Refusal: {refusal}"
                    )
                    raise live_skip.SkipNotConvergingError(msg) from refusal
                dropped = live_skip.apply_skip_policy(
                    refusal, run_id=self._log.start_time, run_date_et=self._run_date_et
                )
                self._log.skipped_games = sorted(live_skip.excluded_games())
                logger.warning(
                    "Per-game refusal: games dropped, re-running step on the rest",
                    step=step.name,
                    dropped=sorted(dropped),
                    skip_round=rounds,
                )

    # -- Finalization helpers ------------------------------------------------

    def _stamp_duration(self) -> None:
        """Set ``end_time`` and ``total_duration_ms`` from the recorded steps.

        SPLIT OUT OF :meth:`_finalize_log` (WR-09). Phase E's success alert reads
        ``total_duration_ms`` and ``_finalize_log`` -- the only place that computed it -- ran
        AFTER, so every successful Friday run alerted ``execution_time_ms=0``
        (``ExecutionLog.total_duration_ms`` defaults to 0). Since ``pipeline/alert.py`` is
        log-only by default, that alert line is the one place a pipeline slowing down would show,
        and it was permanently zero. The failure and degraded branches never read the field, so
        they were unaffected.

        Idempotent: calling it again recomputes the same sum from the same steps.
        """
        self._log.end_time = datetime.now(ET).isoformat()
        self._log.total_duration_ms = sum(s.duration_ms for s in self._log.steps)

    def _finalize_log(self) -> None:
        """Set end_time, total_duration_ms, and write log atomically."""
        self._stamp_duration()
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
            self._log.status = RunStatus.FAILED.value
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
                self._log.status = RunStatus.FAILED.value
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
        all_steps = self._steps if self._steps is not None else build_step_registry()
        steps = self._filter_steps(all_steps)

        # A FRESH exclusion register for every run (T-33.2-03-10): games dropped by an earlier
        # in-process run must never suppress tonight's. Reset before the first step, always.
        live_skip.reset_excluded_games()
        self._log.skipped_games = []

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
                    self._log.status = RunStatus.FAILED.value
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

        # Final run status -- a THREE-way decision after the critical-failure path above
        # (which has already set RunStatus.FAILED and raised). Plan 33.2-03, D33.2-05.
        #
        # FINISHED_WITH_SKIPS WINS OVER DEGRADED, and nothing is lost by the precedence. A
        # dropped game is the fact the operator must act on: a game they expected to see has
        # no prediction and no bet. A non-critical step failure is still recorded in
        # ``warnings`` either way (above), so a run that did both says both -- the status
        # names the one that changes what was published.
        # The decision reads the REGISTER, the one home of the excluded games; the log field
        # is its serialised mirror.
        skipped = live_skip.excluded_games()
        self._log.skipped_games = sorted(skipped)
        if skipped:
            self._log.status = RunStatus.FINISHED_WITH_SKIPS.value
        elif has_non_critical_failure:
            self._log.status = "degraded"
        else:
            self._log.status = RunStatus.SUCCESS.value

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
        # The duration is stamped BEFORE the alert reads it (WR-09). _finalize_log below still
        # stamps it (idempotently) on its way to writing the log, so the written log and the
        # alerted number are the same figure computed from the same steps.
        self._stamp_duration()
        failed_steps = [
            s.name for s in self._log.steps if s.status == StepStatus.FAILED.value
        ]
        if self._log.status == RunStatus.FAILED.value:
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
        elif self._log.status == RunStatus.FINISHED_WITH_SKIPS.value:
            # ITS OWN BRANCH, and it must precede the terminal else (T-33.2-03-05). Falling
            # into that else would alert a CLEAN SUCCESS for a run that left games out.
            self.alert_manager.alert_finished_with_skips(
                list(self._log.skipped_games),
                len(
                    [s for s in self._log.steps if s.status == StepStatus.SUCCESS.value]
                ),
                self.season,
                self.week,
            )
        elif self._log.status == "degraded":
            # Audited (Plan 33.2-03): unchanged. "degraded" is not a RunStatus member and is
            # only reachable when no game was skipped -- the skip branch above takes precedence.
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
        all_steps = self._steps if self._steps is not None else build_step_registry()
        steps = self._filter_steps(all_steps)
        return [f"{s.name}: {s.description}" for s in steps]
