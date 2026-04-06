"""Pipeline-specific alert manager with four non-overlapping alert methods.

Wraps the existing utils.alert_manager.AlertManager with pipeline-specific
convenience methods. Each method maps to exactly ONE event type to prevent
alert duplication.

Alert routing:
- Failure     -> alert_pipeline_failure (CRITICAL)
- Success     -> alert_pipeline_success (INFO)
- Staleness   -> alert_staleness_warning (WARNING, pre-flight only)
- Degraded    -> alert_degraded_completion (WARNING, replaces success for degraded)

Alerts default to log-only (console channel). Email/Slack channels are only used
if configured in .env. This is already handled by the existing AlertManager's
should_send_alert() and channel routing logic.

Addresses review concern: Alert duplication (MEDIUM, Codex).
"""

from typing import Any

from utils.alert_manager import AlertLevel, AlertType, get_alert_manager
from utils.logging_config import get_logger

logger = get_logger(__name__)


class PipelineAlertManager:
    """Pipeline-specific alert wrapper with four non-overlapping methods.

    The orchestrator calls exactly one alert method per pipeline outcome:
    - Failure -> alert_pipeline_failure (only)
    - Success -> alert_pipeline_success (only)
    - Staleness warnings -> alert_staleness_warning (only, before run starts)
    - Degraded completion -> alert_degraded_completion (only)
    """

    def __init__(self) -> None:
        """Initialize with the global AlertManager instance."""
        self._am = get_alert_manager()

    def alert_pipeline_failure(
        self,
        error: str,
        steps_completed: int,
        steps_failed: int,
        season: int,
        week: int,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Alert for pipeline failure. Creates a CRITICAL alert.

        Args:
            error: Error description.
            steps_completed: Number of steps that completed before failure.
            steps_failed: Number of steps that failed.
            season: NFL season year.
            week: NFL week number.
            details: Optional additional details.
        """
        alert = self._am.create_alert(
            level=AlertLevel.CRITICAL,
            alert_type=AlertType.SYSTEM_HEALTH,
            title="Pipeline Failed",
            message=(
                f"S{season}W{week}: Pipeline failed after {steps_completed} steps. "
                f"Error: {error}"
            ),
            details=details
            or {
                "steps_completed": steps_completed,
                "steps_failed": steps_failed,
            },
            source="friday_pipeline",
        )
        self._am.send_alert(alert)

    def alert_pipeline_success(
        self,
        predictions_count: int,
        execution_time_ms: float,
        warnings: list[str],
        season: int,
        week: int,
    ) -> None:
        """Alert for successful pipeline completion. Creates an INFO alert.

        Args:
            predictions_count: Number of predictions generated.
            execution_time_ms: Total execution time in milliseconds.
            warnings: List of non-blocking warnings from the run.
            season: NFL season year.
            week: NFL week number.
        """
        alert = self._am.create_alert(
            level=AlertLevel.INFO,
            alert_type=AlertType.SYSTEM_HEALTH,
            title="Pipeline Completed Successfully",
            message=(
                f"S{season}W{week}: Pipeline completed in "
                f"{execution_time_ms / 1000:.1f}s. Warnings: {len(warnings)}"
            ),
            details={
                "predictions_count": predictions_count,
                "warnings": warnings,
            },
            source="friday_pipeline",
        )
        self._am.send_alert(alert)

    def alert_staleness_warning(
        self,
        warnings: list[str],
        season: int,
        week: int,
    ) -> None:
        """Alert for pre-flight staleness warnings. Creates a WARNING alert.

        Called once before the pipeline run starts, if staleness checks
        produce advisory warnings (e.g., old models, missing odds).

        Args:
            warnings: List of staleness warning messages.
            season: NFL season year.
            week: NFL week number.
        """
        alert = self._am.create_alert(
            level=AlertLevel.WARNING,
            alert_type=AlertType.SYSTEM_HEALTH,
            title="Pipeline Staleness Warnings",
            message=(f"S{season}W{week}: Pre-flight produced {len(warnings)} warnings"),
            details={"warnings": warnings},
            source="friday_pipeline",
        )
        self._am.send_alert(alert)

    def alert_degraded_completion(
        self,
        failed_steps: list[str],
        completed_steps: int,
        season: int,
        week: int,
    ) -> None:
        """Alert for degraded pipeline completion. Creates a WARNING alert.

        Called when the pipeline completes but some non-critical steps failed.
        Replaces alert_pipeline_success for degraded runs.

        Args:
            failed_steps: Names of steps that failed.
            completed_steps: Total number of steps that completed.
            season: NFL season year.
            week: NFL week number.
        """
        alert = self._am.create_alert(
            level=AlertLevel.WARNING,
            alert_type=AlertType.SYSTEM_HEALTH,
            title="Pipeline Completed with Failures",
            message=(
                f"S{season}W{week}: {len(failed_steps)} non-critical steps failed: "
                f"{', '.join(failed_steps)}"
            ),
            details={
                "failed_steps": failed_steps,
                "completed_steps": completed_steps,
            },
            source="friday_pipeline",
        )
        self._am.send_alert(alert)
