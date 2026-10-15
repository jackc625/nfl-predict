"""Alert Manager for NFL Prediction System -- log-only by design.

An alert is ONE log line, ``ALERT [<LEVEL>] <title>: <message>``, written at the alert's own
level by :meth:`AlertManager.create_alert`. That line is the whole delivery: there is no send
step, no channel routing, no cooldown and no rate limit, so nothing here can suppress an alert.

The phone push, the Saturday email and the outside watcher are a SEPARATE path, the ``notify/``
package (Phase 35), invoked only by the daily run, its ledger side steps, the closing-line task
and the TEST command. This layer is never rerouted into it: ``pipeline/alert.py`` wraps it for the
orchestrator, which the push-free manual ``scripts/friday_pipeline.py`` tool shares (Phase 35 D-05
deleted the inert email/Slack branches, the cooldown and the unused alert methods).
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any

from utils.logging_config import get_logger

logger = get_logger(__name__)


class AlertLevel(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of alerts."""

    DATA_INGESTION = "data_ingestion"
    FEATURE_ENGINEERING = "feature_engineering"
    MODEL_PREDICTION = "model_prediction"
    API_HEALTH = "api_health"
    PERFORMANCE = "performance"
    SYSTEM_HEALTH = "system_health"


@dataclass
class Alert:
    """Alert data structure."""

    level: AlertLevel
    alert_type: AlertType
    title: str
    message: str
    timestamp: datetime
    details: dict[str, Any] | None = None
    source: str | None = None


class AlertManager:
    """Writes one log line per alert; nothing else."""

    def create_alert(
        self,
        level: AlertLevel,
        alert_type: AlertType,
        title: str,
        message: str,
        details: dict[str, Any] | None = None,
        source: str | None = None,
    ) -> Alert:
        """Build the alert and write its one log line at the alert's level."""
        alert = Alert(
            level=level,
            alert_type=alert_type,
            title=title,
            message=message,
            timestamp=datetime.now(),
            details=details,
            source=source,
        )

        log_level = {
            AlertLevel.INFO: logger.info,
            AlertLevel.WARNING: logger.warning,
            AlertLevel.ERROR: logger.error,
            AlertLevel.CRITICAL: logger.critical,
        }[level]

        log_level(f"ALERT [{level.value.upper()}] {title}: {message}")

        return alert


# Global alert manager instance
_alert_manager = None


def get_alert_manager() -> AlertManager:
    """Get the global alert manager instance."""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager
