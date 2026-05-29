#!/usr/bin/env python3
"""NFL Prediction System - Scheduling Setup Script.

Sets up automated scheduling for the unified Friday pipeline on Windows via
Task Scheduler. This is the single canonical installer: it registers the
committed deployment/windows_scheduler.xml via `schtasks /create /xml ... /f`
so the installer and the XML cannot drift (one automation story).

Consolidates the old two-task approach (DataUpdate + Predictions) into a
single NFL_Predict_Pipeline task at 6:00 PM ET on Fridays.

Usage:
    python deployment/setup_scheduling.py --platform windows --install
    python deployment/setup_scheduling.py --platform windows --dry-run
"""

import argparse
import platform
import subprocess
import sys
from pathlib import Path

from utils.logging_config import get_logger

logger = get_logger(__name__)

# Old task names to clean up during migration
OLD_TASK_NAMES = ["NFL_Predict_DataUpdate", "NFL_Predict_Predictions"]


class SchedulingSetup:
    """Set up Friday-pipeline scheduling via the Windows Task Scheduler."""

    def __init__(self, nfl_predict_home: str) -> None:
        """Initialize scheduling setup."""
        self.nfl_predict_home = Path(nfl_predict_home).resolve()
        self.platform = platform.system().lower()
        self.python_path = sys.executable

        # Detect virtual environment python if available
        venv_python = (
            self.nfl_predict_home
            / ".venv"
            / ("Scripts" if self.platform == "windows" else "bin")
            / "python"
        )
        if venv_python.exists():
            self.python_path = str(venv_python)

        logger.info(f"Platform detected: {self.platform}")
        logger.info(f"Project home: {self.nfl_predict_home}")
        logger.info(f"Python path: {self.python_path}")

    def validate_prerequisites(self) -> bool:
        """Validate that all prerequisites are met."""
        logger.info("Validating prerequisites...")

        # Check for the unified pipeline script
        if not (self.nfl_predict_home / "scripts" / "friday_pipeline.py").exists():
            logger.error(
                f"Friday pipeline script not found at "
                f"{self.nfl_predict_home / 'scripts' / 'friday_pipeline.py'}"
            )
            return False

        # Check Python path
        if not Path(self.python_path).exists():
            logger.error(f"Python executable not found: {self.python_path}")
            return False

        # Create logs directory if it doesn't exist
        logs_dir = self.nfl_predict_home / "logs"
        logs_dir.mkdir(exist_ok=True)

        logger.info("Prerequisites validation passed")
        return True

    def _cleanup_old_tasks(self) -> None:
        """Idempotently remove old scheduled tasks from before the unified pipeline.

        Removes NFL_Predict_DataUpdate and NFL_Predict_Predictions tasks.
        Ignores errors if tasks don't exist (idempotent).
        """
        for old_name in OLD_TASK_NAMES:
            try:
                result = subprocess.run(
                    ["schtasks", "/delete", "/tn", old_name, "/f"],
                    capture_output=True,
                    text=True,
                )
                if result.returncode == 0:
                    logger.info(f"Removed old task: {old_name}")
                else:
                    logger.debug(f"Old task {old_name} not found (already removed)")
            except Exception:
                logger.debug(f"Old task {old_name} not found (already removed)")

    def setup_windows_scheduler(self, dry_run: bool = False) -> bool:
        """Set up Windows Task Scheduler with a single unified pipeline task.

        Registers the committed deployment/windows_scheduler.xml via
        `schtasks /create /xml ... /f` so the installer and the XML cannot
        drift (the XML carries the real trigger time, principal, and command).
        Idempotently cleans up old tasks (NFL_Predict_DataUpdate,
        NFL_Predict_Predictions) before creating the new one.
        """
        logger.info("Setting up Windows Task Scheduler...")

        # Check if schtasks is available
        try:
            subprocess.run(["schtasks", "/?"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error(
                "schtasks command not available. "
                "Task Scheduler setup requires administrator privileges."
            )
            return False

        # Clean up old tasks before creating the new one (idempotent)
        if not dry_run:
            self._cleanup_old_tasks()
        else:
            for old_name in OLD_TASK_NAMES:
                logger.info(f"DRY RUN: Would delete old task: {old_name}")

        # Single unified task definition. The trigger time, principal, and command
        # are carried by the committed XML (installed via /xml below); the fields
        # here are informational labels only.
        tasks = [
            {
                "name": "NFL_Predict_Pipeline",
                "description": "Friday 6:00 PM ET - Unified Pipeline (data + predictions)",
                "script": "friday_pipeline.py",
                "time": "18:00",
                "day": "FRI",
            },
        ]

        # Path to the canonical task definition (single source of truth, D-07)
        xml_path = self.nfl_predict_home / "deployment" / "windows_scheduler.xml"

        for task in tasks:
            logger.info(f"Setting up task: {task['name']}")

            # Register the canonical XML so the installer and the XML cannot drift.
            # /f overwrites an existing task of the same name (idempotent re-install).
            cmd = [
                "schtasks",
                "/create",
                "/tn",
                task["name"],
                "/xml",
                str(xml_path),
                "/f",  # Force creation (overwrite if exists = idempotent)
            ]

            if dry_run:
                logger.info(f"DRY RUN: Would execute: {' '.join(cmd)}")
                continue

            try:
                result = subprocess.run(
                    cmd, cwd=str(self.nfl_predict_home), capture_output=True, text=True
                )
                if result.returncode == 0:
                    logger.info(f"Task '{task['name']}' created successfully")
                else:
                    logger.error(
                        f"Failed to create task '{task['name']}': {result.stderr}"
                    )
                    return False
            except Exception as e:
                logger.error(f"Error creating task '{task['name']}': {e}")
                return False

        if not dry_run:
            logger.info("Windows Task Scheduler setup completed")
            logger.info(
                "Tasks can be managed via Task Scheduler GUI or schtasks command"
            )

        return True

    def test_scripts(self, dry_run: bool = True) -> bool:
        """Test the unified pipeline script with dry-run."""
        logger.info("Testing automation scripts...")

        scripts = [
            ("Unified Pipeline", "scripts/friday_pipeline.py"),
        ]

        for name, script in scripts:
            logger.info(f"Testing {name} script...")

            # Use --force for offseason testing (Pitfall 6)
            cmd = [self.python_path, script, "--dry-run", "--force"]

            try:
                result = subprocess.run(
                    cmd,
                    cwd=str(self.nfl_predict_home),
                    capture_output=True,
                    text=True,
                    timeout=60,
                )

                if result.returncode == 0:
                    logger.info(f"{name} script test passed")
                else:
                    logger.error(f"{name} script test failed: {result.stderr}")
                    return False

            except subprocess.TimeoutExpired:
                logger.error(f"{name} script test timed out")
                return False
            except Exception as e:
                logger.error(f"Error testing {name} script: {e}")
                return False

        logger.info("All script tests passed")
        return True

    def show_status(self) -> None:
        """Show current scheduling status via the Windows Task Scheduler."""
        logger.info("Current scheduling status:")

        try:
            result = subprocess.run(
                ["schtasks", "/query", "/fo", "csv"], capture_output=True, text=True
            )
            if result.returncode == 0:
                nfl_tasks = [
                    line for line in result.stdout.split("\n") if "NFL_Predict" in line
                ]
                if nfl_tasks:
                    logger.info("Active scheduled tasks:")
                    for task in nfl_tasks:
                        logger.info(f"  {task}")
                else:
                    logger.info("No NFL prediction scheduled tasks found")
        except Exception as e:
            logger.error(f"Error checking scheduled tasks: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="NFL Prediction System - Scheduling Setup"
    )
    parser.add_argument(
        "--platform",
        choices=["auto", "windows"],
        default="auto",
        help="Target platform for scheduling setup (Windows Task Scheduler only)",
    )
    parser.add_argument(
        "--install", action="store_true", help="Install the scheduling configuration"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument("--test", action="store_true", help="Test automation scripts")
    parser.add_argument(
        "--status", action="store_true", help="Show current scheduling status"
    )
    parser.add_argument(
        "--project-home", default=".", help="Path to NFL prediction project root"
    )

    args = parser.parse_args()

    # Auto-detect platform if needed (Windows Task Scheduler is the only
    # supported automation mechanism)
    if args.platform == "auto":
        system = platform.system().lower()
        if system == "windows":
            args.platform = "windows"
        else:
            logger.error(
                f"Unsupported platform: {system}. "
                "Scheduling is supported via Windows Task Scheduler only."
            )
            return 1

    # Initialize setup
    setup = SchedulingSetup(args.project_home)

    # Validate prerequisites
    if not setup.validate_prerequisites():
        logger.error("Prerequisites validation failed")
        return 1

    # Execute requested action
    success = True

    if args.status:
        setup.show_status()

    if args.test:
        success = setup.test_scripts(dry_run=True)

    if args.install or args.dry_run:
        success = setup.setup_windows_scheduler(dry_run=args.dry_run)

    if success:
        if args.install:
            logger.info("Scheduling setup completed successfully")
        elif args.dry_run:
            logger.info("Dry run completed successfully")
        return 0
    logger.error("Scheduling setup failed")
    return 1


if __name__ == "__main__":
    sys.exit(main())
