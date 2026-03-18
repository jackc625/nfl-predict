"""
Alert Manager for NFL Prediction System

Centralized alert management system that integrates with the Friday automation
scripts and provides real-time notifications for various system events.
"""

import json
import smtplib
import time
from dataclasses import dataclass
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from pathlib import Path
from typing import Any

import httpx

from conf.settings import get_settings
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

    def to_dict(self) -> dict[str, Any]:
        """Convert alert to dictionary for serialization."""
        return {
            "level": self.level.value,
            "alert_type": self.alert_type.value,
            "title": self.title,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "details": self.details or {},
            "source": self.source,
        }


class AlertManager:
    """Centralized alert management system."""

    def __init__(self):
        """Initialize alert manager."""
        self.settings = get_settings()
        self.monitoring_config = (
            self.settings.monitoring.model_dump()
            if hasattr(self.settings, "monitoring")
            else {}
        )
        self.alert_history: list[Alert] = []
        self.last_alert_time = {}  # For rate limiting

    def create_alert(
        self,
        level: AlertLevel,
        alert_type: AlertType,
        title: str,
        message: str,
        details: dict[str, Any] | None = None,
        source: str | None = None,
    ) -> Alert:
        """Create a new alert."""
        alert = Alert(
            level=level,
            alert_type=alert_type,
            title=title,
            message=message,
            timestamp=datetime.now(),
            details=details,
            source=source,
        )

        # Add to history
        self.alert_history.append(alert)

        # Log the alert
        log_level = {
            AlertLevel.INFO: logger.info,
            AlertLevel.WARNING: logger.warning,
            AlertLevel.ERROR: logger.error,
            AlertLevel.CRITICAL: logger.critical,
        }[level]

        log_level(f"ALERT [{level.value.upper()}] {title}: {message}")

        return alert

    def should_send_alert(self, alert: Alert) -> bool:
        """Check if alert should be sent based on rate limiting."""
        now = time.time()
        alert_key = f"{alert.alert_type.value}_{alert.level.value}"

        # Check cooldown period
        cooldown_seconds = (
            self.monitoring_config.get("notifications", {}).get("cooldown_minutes", 30)
            * 60
        )
        last_time = self.last_alert_time.get(alert_key, 0)

        if now - last_time < cooldown_seconds:
            logger.debug(f"Alert {alert_key} suppressed due to cooldown")
            return False

        # Check hourly rate limit
        max_alerts_per_hour = self.monitoring_config.get("notifications", {}).get(
            "max_alerts_per_hour", 20
        )
        hour_start = now - 3600  # Last hour
        recent_alerts = [
            a
            for a in self.alert_history
            if a.timestamp.timestamp() > hour_start and a.alert_type == alert.alert_type
        ]

        if len(recent_alerts) >= max_alerts_per_hour:
            logger.warning(f"Alert rate limit exceeded for {alert.alert_type.value}")
            return False

        self.last_alert_time[alert_key] = now
        return True

    def send_alert(self, alert: Alert) -> bool:
        """Send alert through configured notification channels."""
        if not self.should_send_alert(alert):
            return False

        notifications_config = self.monitoring_config.get("notifications", {})
        notification_levels = notifications_config.get("notification_levels", {})
        channels = notification_levels.get(alert.level.value, ["console"])

        success = True

        for channel in channels:
            try:
                if channel == "console":
                    self._send_console_alert(alert)
                elif channel == "email" and notifications_config.get(
                    "enable_email", False
                ):
                    self._send_email_alert(alert)
                elif channel == "slack" and notifications_config.get(
                    "enable_slack", False
                ):
                    self._send_slack_alert(alert)
            except (OSError, ValueError, RuntimeError) as e:
                logger.error(f"Failed to send alert via {channel}: {e}")
                success = False

        return success

    def _send_console_alert(self, alert: Alert) -> None:
        """Send alert to console/logs."""

        if alert.details:
            pass

    def _send_email_alert(self, alert: Alert) -> None:
        """Send alert via email."""
        notifications_config = self.monitoring_config.get("notifications", {})
        recipients = notifications_config.get("email_recipients", [])

        if not recipients:
            logger.warning("Email notifications enabled but no recipients configured")
            return

        # Create email message
        msg = MIMEMultipart()
        msg["From"] = self.settings.email_from
        msg["To"] = ", ".join(recipients)
        msg["Subject"] = (
            f"NFL Predict Alert [{alert.level.value.upper()}]: {alert.title}"
        )

        # Create HTML body
        html_body = f"""
        <html>
        <body>
        <h2>NFL Prediction System Alert</h2>
        <p><strong>Level:</strong> {alert.level.value.upper()}</p>
        <p><strong>Type:</strong> {alert.alert_type.value}</p>
        <p><strong>Time:</strong> {alert.timestamp}</p>
        <p><strong>Source:</strong> {alert.source or "Unknown"}</p>
        <p><strong>Message:</strong></p>
        <p>{alert.message}</p>
        """

        if alert.details:
            html_body += f"""
            <p><strong>Details:</strong></p>
            <pre>{json.dumps(alert.details, indent=2)}</pre>
            """

        html_body += """
        </body>
        </html>
        """

        msg.attach(MIMEText(html_body, "html"))

        # Send email
        with smtplib.SMTP(self.settings.smtp_host, self.settings.smtp_port) as server:
            if self.settings.smtp_use_tls:
                server.starttls()
            if self.settings.smtp_username:
                server.login(self.settings.smtp_username, self.settings.smtp_password)
            server.send_message(msg)

        logger.info(f"Email alert sent to {len(recipients)} recipients")

    def _send_slack_alert(self, alert: Alert) -> None:
        """Send alert via Slack webhook."""
        notifications_config = self.monitoring_config.get("notifications", {})
        webhook_url = notifications_config.get("slack_webhook_url")

        if not webhook_url:
            logger.warning("Slack notifications enabled but no webhook URL configured")
            return

        # Create Slack message
        color_map = {
            AlertLevel.INFO: "#36a64f",  # green
            AlertLevel.WARNING: "#ffa500",  # orange
            AlertLevel.ERROR: "#ff4444",  # red
            AlertLevel.CRITICAL: "#8B0000",  # dark red
        }

        slack_message = {
            "text": f"NFL Predict Alert: {alert.title}",
            "attachments": [
                {
                    "color": color_map.get(alert.level, "#808080"),
                    "fields": [
                        {
                            "title": "Level",
                            "value": alert.level.value.upper(),
                            "short": True,
                        },
                        {
                            "title": "Type",
                            "value": alert.alert_type.value,
                            "short": True,
                        },
                        {
                            "title": "Time",
                            "value": alert.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                            "short": True,
                        },
                        {
                            "title": "Source",
                            "value": alert.source or "Unknown",
                            "short": True,
                        },
                        {"title": "Message", "value": alert.message, "short": False},
                    ],
                }
            ],
        }

        if alert.details:
            slack_message["attachments"][0]["fields"].append(
                {
                    "title": "Details",
                    "value": f"```{json.dumps(alert.details, indent=2)}```",
                    "short": False,
                }
            )

        # Send to Slack
        response = httpx.post(webhook_url, json=slack_message)
        response.raise_for_status()

        logger.info("Slack alert sent successfully")

    def alert_data_ingestion_success(self, step: str, details: dict[str, Any]) -> None:
        """Alert for successful data ingestion step."""
        alert = self.create_alert(
            level=AlertLevel.INFO,
            alert_type=AlertType.DATA_INGESTION,
            title=f"Data Ingestion Success: {step}",
            message=f"Successfully completed {step}",
            details=details,
            source="friday_data_update",
        )
        self.send_alert(alert)

    def alert_data_ingestion_failure(
        self, step: str, error: str, details: dict[str, Any]
    ) -> None:
        """Alert for failed data ingestion step."""
        alert = self.create_alert(
            level=AlertLevel.ERROR,
            alert_type=AlertType.DATA_INGESTION,
            title=f"Data Ingestion Failed: {step}",
            message=f"Failed to complete {step}: {error}",
            details=details,
            source="friday_data_update",
        )
        self.send_alert(alert)

    def alert_prediction_success(
        self, predictions_count: int, recommendations_count: int
    ) -> None:
        """Alert for successful prediction generation."""
        alert = self.create_alert(
            level=AlertLevel.INFO,
            alert_type=AlertType.MODEL_PREDICTION,
            title="Predictions Generated Successfully",
            message=f"Generated {predictions_count} predictions and {recommendations_count} recommendations",
            details={
                "predictions_count": predictions_count,
                "recommendations_count": recommendations_count,
            },
            source="friday_predictions_run",
        )
        self.send_alert(alert)

    def alert_prediction_failure(
        self, step: str, error: str, details: dict[str, Any]
    ) -> None:
        """Alert for failed prediction step."""
        alert = self.create_alert(
            level=AlertLevel.CRITICAL,
            alert_type=AlertType.MODEL_PREDICTION,
            title=f"Prediction Failed: {step}",
            message=f"Critical failure in {step}: {error}",
            details=details,
            source="friday_predictions_run",
        )
        self.send_alert(alert)

    def alert_api_health_issue(
        self, endpoint: str, error: str, details: dict[str, Any]
    ) -> None:
        """Alert for API health issues."""
        alert = self.create_alert(
            level=AlertLevel.WARNING,
            alert_type=AlertType.API_HEALTH,
            title=f"API Health Issue: {endpoint}",
            message=f"Health check failed for {endpoint}: {error}",
            details=details,
            source="health_check",
        )
        self.send_alert(alert)

    def alert_performance_issue(
        self, metric: str, value: float, threshold: float, details: dict[str, Any]
    ) -> None:
        """Alert for performance issues."""
        alert = self.create_alert(
            level=AlertLevel.WARNING,
            alert_type=AlertType.PERFORMANCE,
            title=f"Performance Issue: {metric}",
            message=f"{metric} is {value}, exceeds threshold of {threshold}",
            details=details,
            source="operational_monitoring",
        )
        self.send_alert(alert)

    def save_alert_history(self, output_path: str = "logs/alert_history.json") -> None:
        """Save alert history to file."""
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        history_data = {
            "timestamp": datetime.now().isoformat(),
            "total_alerts": len(self.alert_history),
            "alerts": [alert.to_dict() for alert in self.alert_history],
        }

        with open(output_path, "w") as f:
            json.dump(history_data, f, indent=2)

        logger.info(f"Alert history saved to {output_path}")


# Global alert manager instance
_alert_manager = None


def get_alert_manager() -> AlertManager:
    """Get the global alert manager instance."""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager
