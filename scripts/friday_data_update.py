#!/usr/bin/env python3
"""
Friday 5:00 PM ET Data Update Automation Script

This script performs the weekly data update process every Friday at 5:00 PM ET,
preparing for the 6:00 PM odds snapshot and prediction generation.

Key Operations:
- Ingest current week games data
- Update Elo ratings
- Build team form metrics
- Refresh contextual features
- Validate data quality
- Prepare for odds snapshot
"""

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from conf.settings import get_settings
from utils.alert_manager import get_alert_manager
from utils.date_utils import get_current_nfl_week, get_snapshot_time
from utils.exceptions import DataIngestionError
from utils.logging_config import get_logger

logger = get_logger(__name__)


class FridayDataUpdateAutomation:
    """Automated Friday data update pipeline."""

    def __init__(self):
        """Initialize automation system."""
        self.settings = get_settings()
        self.season, self.week = get_current_nfl_week()
        self.snapshot_time = get_snapshot_time()
        self.alert_manager = get_alert_manager()

        # Execution tracking
        self.execution_log = {
            "start_time": datetime.now(),
            "season": self.season,
            "week": self.week,
            "snapshot_time": self.snapshot_time.isoformat(),
            "steps_completed": [],
            "steps_failed": [],
            "warnings": [],
            "duration_seconds": 0,
        }

        logger.info(
            f"Friday Data Update initialized for Season {self.season}, Week {self.week}"
        )
        logger.info(f"Target snapshot time: {self.snapshot_time}")

    def run_command(
        self, command: str, description: str, critical: bool = True
    ) -> bool:
        """
        Execute a shell command with logging and error handling.

        Args:
            command: Command to execute
            description: Human-readable description
            critical: Whether failure should stop the pipeline

        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Starting: {description}")
        logger.debug(f"Command: {command}")
        start = time.time()

        try:
            result = subprocess.run(
                command,
                shell=True,
                check=True,
                capture_output=True,
                text=True,
                timeout=1800,  # 30 minute timeout
            )

            if result.stdout:
                logger.debug(f"Output: {result.stdout}")

            logger.info(f"Completed: {description}")
            self.execution_log["steps_completed"].append(description)

            # Send success alert
            self.alert_manager.alert_data_ingestion_success(
                step=description,
                details={
                    "command": command,
                    "duration_ms": (time.time() - start) * 1000,
                    "season": self.season,
                    "week": self.week,
                },
            )
            return True

        except subprocess.CalledProcessError as e:
            error_msg = f"Failed: {description} - Exit code: {e.returncode}"
            if e.stderr:
                error_msg += f" - Error: {e.stderr}"

            logger.error(error_msg)
            self.execution_log["steps_failed"].append(
                {"step": description, "error": error_msg, "exit_code": e.returncode}
            )

            # Send failure alert
            self.alert_manager.alert_data_ingestion_failure(
                step=description,
                error=error_msg,
                details={
                    "command": command,
                    "exit_code": e.returncode,
                    "stderr": e.stderr,
                    "season": self.season,
                    "week": self.week,
                    "critical": critical,
                },
            )

            if critical:
                raise DataIngestionError(f"Critical step failed: {description}")
            self.execution_log["warnings"].append(error_msg)
            return False

        except subprocess.TimeoutExpired:
            error_msg = f"Timeout: {description} exceeded 30 minutes"
            logger.error(error_msg)
            self.execution_log["steps_failed"].append(
                {"step": description, "error": error_msg, "exit_code": "TIMEOUT"}
            )

            if critical:
                raise DataIngestionError(f"Critical step timeout: {description}")
            self.execution_log["warnings"].append(error_msg)
            return False

    def step_1_ingest_games_data(self) -> bool:
        """Step 1: Ingest current week games data."""
        return self.run_command(
            "python scripts/ingest_games.py --current",
            "Ingest current week games data",
            critical=True,
        )

    def step_2_ingest_weather_data(self) -> bool:
        """Step 2: Ingest weather forecasts for current week."""
        return self.run_command(
            "python scripts/ingest_weather.py --current-week",
            "Ingest weather forecasts",
            critical=False,  # Weather is nice-to-have but not critical
        )

    def step_3_data_quality_validation(self) -> bool:
        """Step 3: Validate data quality and completeness."""
        return self.run_command(
            "python scripts/data_qa.py --current-week --strict",
            "Data quality validation",
            critical=True,
        )

    def step_4_update_elo_ratings(self) -> bool:
        """Step 4: Update Elo ratings with latest results."""
        return self.run_command(
            "python scripts/build_elo.py --incremental --current-week",
            "Update Elo ratings",
            critical=True,
        )

    def step_5_build_team_form_metrics(self) -> bool:
        """Step 5: Build rolling team form metrics."""
        return self.run_command(
            "python scripts/build_team_form.py --incremental --current-week",
            "Build team form metrics",
            critical=True,
        )

    def step_6_build_contextual_features(self) -> bool:
        """Step 6: Build contextual features (travel, rest, etc.)."""
        return self.run_command(
            "python scripts/build_contextual.py --current-week",
            "Build contextual features",
            critical=True,
        )

    def step_7_build_weather_features(self) -> bool:
        """Step 7: Build weather-based features."""
        return self.run_command(
            "python scripts/build_weather.py --current-week",
            "Build weather features",
            critical=False,  # Non-critical if weather data missing
        )

    def step_8_prepare_for_odds_snapshot(self) -> bool:
        """Step 8: Prepare system for 6 PM odds snapshot."""
        logger.info("Preparing system for 6:00 PM ET odds snapshot")

        # Verify all required data is in place
        checks = [
            ("Games data", "data/silver/games.parquet"),
            ("Elo ratings", "data/silver/elo_ratings.parquet"),
            ("Team form", "data/silver/team_form.parquet"),
            ("Features directory", "data/gold/"),
        ]

        missing_data = []
        for name, path in checks:
            if not Path(path).exists():
                missing_data.append(name)
                logger.warning(f"Missing: {name} at {path}")

        if missing_data:
            self.execution_log["warnings"].append(
                f"Missing data: {', '.join(missing_data)}"
            )

        logger.info("System prepared for odds snapshot")
        self.execution_log["steps_completed"].append("Prepare for odds snapshot")
        return True

    def run_data_update_pipeline(self) -> dict[str, Any]:
        """
        Execute the complete Friday 5:00 PM data update pipeline.

        Returns:
            Execution summary with results and timing
        """
        logger.info("Starting Friday 5:00 PM ET Data Update Pipeline")
        logger.info(f"Season: {self.season}, Week: {self.week}")

        try:
            # Execute pipeline steps
            self.step_1_ingest_games_data()
            self.step_2_ingest_weather_data()
            self.step_3_data_quality_validation()
            self.step_4_update_elo_ratings()
            self.step_5_build_team_form_metrics()
            self.step_6_build_contextual_features()
            self.step_7_build_weather_features()
            self.step_8_prepare_for_odds_snapshot()

            # Calculate execution time
            end_time = datetime.now()
            duration = (end_time - self.execution_log["start_time"]).total_seconds()
            self.execution_log["duration_seconds"] = duration
            self.execution_log["end_time"] = end_time
            self.execution_log["status"] = "SUCCESS"

            logger.info(
                f"Friday Data Update completed successfully in {duration:.1f} seconds"
            )
            logger.info(
                f"Steps completed: {len(self.execution_log['steps_completed'])}"
            )
            logger.info(f"Warnings: {len(self.execution_log['warnings'])}")

            return self.execution_log

        except Exception as e:
            # Calculate execution time for failed run
            end_time = datetime.now()
            duration = (end_time - self.execution_log["start_time"]).total_seconds()
            self.execution_log["duration_seconds"] = duration
            self.execution_log["end_time"] = end_time
            self.execution_log["status"] = "FAILED"
            self.execution_log["error"] = str(e)

            logger.error(f"Friday Data Update failed after {duration:.1f} seconds: {e}")
            logger.error(
                f"Steps completed: {len(self.execution_log['steps_completed'])}"
            )
            logger.error(f"Steps failed: {len(self.execution_log['steps_failed'])}")

            return self.execution_log

    def save_execution_log(
        self, output_path: str = "logs/friday_data_update.json"
    ) -> None:
        """Save execution log to file."""
        import json

        # Ensure logs directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Convert datetime objects to strings for JSON serialization
        log_copy = self.execution_log.copy()
        for key, value in log_copy.items():
            if isinstance(value, datetime):
                log_copy[key] = value.isoformat()

        with open(output_path, "w") as f:
            json.dump(log_copy, f, indent=2)

        logger.info(f"Execution log saved to {output_path}")


def main():
    """Main entry point for Friday data update automation."""
    parser = argparse.ArgumentParser(
        description="Friday 5:00 PM ET Data Update Automation"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be executed without running commands",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level",
    )
    parser.add_argument(
        "--output-log",
        default="logs/friday_data_update.json",
        help="Path to save execution log",
    )

    args = parser.parse_args()

    # Set logging level
    import logging

    logging.getLogger().setLevel(getattr(logging, args.log_level))

    try:
        # Initialize automation
        automation = FridayDataUpdateAutomation()

        if args.dry_run:
            logger.info("DRY RUN MODE - No commands will be executed")
            logger.info("Pipeline would execute the following steps:")
            steps = [
                "1. Ingest current week games data",
                "2. Ingest weather forecasts",
                "3. Data quality validation",
                "4. Update Elo ratings",
                "5. Build team form metrics",
                "6. Build contextual features",
                "7. Build weather features",
                "8. Prepare for odds snapshot",
            ]
            for step in steps:
                logger.info(f"  {step}")
            return 0

        # Run the data update pipeline
        execution_log = automation.run_data_update_pipeline()

        # Save execution log
        automation.save_execution_log(args.output_log)

        # Return appropriate exit code
        if execution_log["status"] == "SUCCESS":
            logger.info("Friday Data Update completed successfully!")
            return 0
        logger.error("Friday Data Update failed!")
        return 1

    except KeyboardInterrupt:
        logger.warning("Friday Data Update interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error in Friday Data Update: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
