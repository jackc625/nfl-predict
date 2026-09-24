#!/usr/bin/env python3
"""NFL Prediction System - Scheduling Setup Script.

Sets up the DAILY lock-time run on Windows via Task Scheduler (Plan 33.2-28). This is the single
canonical installer: it registers the committed deployment/windows_scheduler.xml via
`schtasks /create /xml ... /f` so the installer and the XML cannot drift (one automation story).

The task keeps its name, NFL_Predict_Pipeline, so installing overwrites the weekly Friday
definition in place: the weekly trigger cannot survive beside the daily one. It fires daily at
5:00 PM local, which is 5:00 PM ET only on an Eastern machine -- so an install refuses any other
Windows time zone.

Usage:
    python deployment/setup_scheduling.py --platform windows --install
    python deployment/setup_scheduling.py --platform windows --dry-run
    python deployment/setup_scheduling.py --platform windows --test [--rehearsal-date YYYY-MM-DD]
    python deployment/setup_scheduling.py --platform windows --verify-installed
"""

import argparse
import csv
import io
import os
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET
from datetime import date, datetime
from pathlib import Path

from utils.logging_config import get_logger

logger = get_logger(__name__)

# Old task names to clean up during migration
OLD_TASK_NAMES = ["NFL_Predict_DataUpdate", "NFL_Predict_Predictions"]

#: The one scheduled task. Kept from the weekly era so /f overwrites that definition in place.
TASK_NAME = "NFL_Predict_Pipeline"

#: The entry point the committed XML schedules, relative to the project home.
DAILY_ENTRY_POINT = "scripts/daily_lock_pipeline.py"

#: The Windows time-zone id `tzutil /g` reports for Eastern time (it covers daylight time too).
EASTERN_WINDOWS_ZONE = "Eastern Standard Time"

#: A rehearsal is a real dry run of the collection stage; it needs more than a minute.
REHEARSAL_TIMEOUT_SECONDS = 900


class NonEasternTimeZoneError(RuntimeError):
    """The machine is not on Eastern time, so the local-time trigger would fire at the wrong hour."""


def read_windows_time_zone() -> str:
    """The machine's Windows time-zone id, as `tzutil /g` reports it."""
    result = subprocess.run(
        ["tzutil", "/g"], capture_output=True, text=True, check=True
    )
    return result.stdout.strip()


def require_eastern_time_zone(zone: str) -> None:
    """Refuse any zone but Eastern: the trigger's StartBoundary carries no zone suffix.

    Raises:
        NonEasternTimeZoneError: *zone* is not :data:`EASTERN_WINDOWS_ZONE`.
    """
    if zone != EASTERN_WINDOWS_ZONE:
        msg = (
            f"this machine's time zone is {zone!r}, not {EASTERN_WINDOWS_ZONE!r}. The daily "
            "trigger fires at 17:00 LOCAL time, which is one hour before the 18:00 ET lock only "
            "on an Eastern machine; refusing to install."
        )
        raise NonEasternTimeZoneError(msg)


# ---------------------------------------------------------------------------
# The installed-task read-back (Plan 33.2-28 Task 3)
# ---------------------------------------------------------------------------

_TASK_NS = "{http://schemas.microsoft.com/windows/2004/02/mit/task}"

#: The fields the read-back compares, in the order it prints them.
READBACK_FIELDS: tuple[str, ...] = (
    "Trigger",
    "StartTime",
    "Command",
    "Arguments",
    "StartWhenAvailable",
)


def _local(tag: str) -> str:
    return tag.rsplit("}", maxsplit=1)[-1]


def _trigger_signature(trigger: ET.Element) -> str:
    """A trigger's kind and schedule, e.g. ``CalendarTrigger ScheduleByDay(DaysInterval=1)``."""
    parts = [_local(trigger.tag)]
    for child in trigger:
        if _local(child.tag).startswith("ScheduleBy"):
            inner = sorted(
                _local(e.tag) + (f"={e.text.strip()}" if (e.text or "").strip() else "")
                for e in child.iter()
                if e is not child
            )
            parts.append(f"{_local(child.tag)}({','.join(inner)})")
    return " ".join(parts)


def task_fields(xml_text: str) -> dict[str, str]:
    """The compared fields of a task definition, NORMALISED so only a real difference differs.

    Windows' export omits settings left at their schema default, so an absent
    ``StartWhenAvailable`` reads as ``false`` (the schema default). The start is compared as a
    time of day (plus any zone suffix, which would be a real change); the command
    case-insensitively, as Windows resolves it; the arguments with whitespace collapsed.
    """
    root = ET.fromstring(xml_text.strip())
    triggers = root.find(f"{_TASK_NS}Triggers")
    boundaries = [(e.text or "").strip() for e in root.iter(f"{_TASK_NS}StartBoundary")]
    starts = []
    for boundary in boundaries:
        instant = datetime.fromisoformat(boundary)
        zone = "" if instant.tzinfo is None else f" {instant.utcoffset()}"
        starts.append(instant.time().isoformat() + zone)

    def _texts(name: str) -> list[str]:
        return [(e.text or "").strip() for e in root.iter(f"{_TASK_NS}{name}")]

    start_when_available = _texts("StartWhenAvailable")
    return {
        "Trigger": "; ".join(
            _trigger_signature(t) for t in (triggers if triggers is not None else [])
        ),
        "StartTime": "; ".join(starts),
        "Command": "; ".join(c.casefold() for c in _texts("Command")),
        "Arguments": "; ".join(" ".join(a.split()) for a in _texts("Arguments")),
        "StartWhenAvailable": (start_when_available or ["false"])[0].lower(),
    }


def compare_task_fields(
    committed_xml: str, installed_xml: str
) -> list[tuple[str, str, str, bool]]:
    """``(field, committed, installed, match)`` for each of :data:`READBACK_FIELDS`."""
    committed, installed = task_fields(committed_xml), task_fields(installed_xml)
    return [
        (name, committed[name], installed[name], committed[name] == installed[name])
        for name in READBACK_FIELDS
    ]


def _decode_task_export(raw: bytes) -> str:
    """``schtasks /query /xml`` output: UTF-16 with a BOM, or the console's ANSI text."""
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return raw.decode("mbcs" if sys.platform == "win32" else "latin-1")


def _emit(line: str) -> None:
    """One read-back line on stdout, where the owner and the plan's verify read it."""
    sys.stdout.write(line + "\n")


def nfl_task_names(csv_listing: str) -> list[str]:
    """Every scheduled task whose name mentions NFL, from ``schtasks /query /fo csv /nh``."""
    names = []
    for row in csv.reader(io.StringIO(csv_listing)):
        if row and "nfl" in row[0].casefold():
            names.append(row[0].lstrip("\\"))
    return sorted(set(names))


class SchedulingSetup:
    """Set up the daily lock-time run via the Windows Task Scheduler."""

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

        # Check for the script the XML schedules
        entry_point = self.nfl_predict_home / DAILY_ENTRY_POINT
        if not entry_point.exists():
            logger.error(f"Daily lock-time script not found at {entry_point}")
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

        # Single unified task definition. ONLY the task name is consumed here; the
        # trigger time, day, principal, and command all live in the committed XML
        # (installed via /xml below), so there is no second copy of the trigger to
        # drift out of sync with the XML (IN-02). Do NOT re-add time/day/description
        # fields here -- they would be silently ignored and could mislead a maintainer
        # into thinking editing them changes the schedule.
        tasks = [
            {"name": TASK_NAME},
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

    def test_scripts(self, rehearsal_date: date | None = None) -> bool:
        """Rehearse the scheduled entry point with its own --dry-run: collection, no writes.

        Off-season there is nothing to collect for today, so pass *rehearsal_date* -- an
        in-season ET day -- which becomes the entry point's --date. The Odds API key is blanked
        for the rehearsal, which puts the odds client in mock mode: a rehearsal never spends the
        paid request quota.
        """
        logger.info("Rehearsing the daily lock-time run (dry run, no writes)...")

        cmd = [self.python_path, DAILY_ENTRY_POINT, "--dry-run"]
        if rehearsal_date is not None:
            cmd += ["--date", rehearsal_date.isoformat()]

        try:
            result = subprocess.run(
                cmd,
                cwd=str(self.nfl_predict_home),
                capture_output=True,
                text=True,
                timeout=REHEARSAL_TIMEOUT_SECONDS,
                env={**os.environ, "ODDS_API_KEY": ""},
            )
        except subprocess.TimeoutExpired:
            logger.error("Daily lock-time rehearsal timed out")
            return False
        except Exception as e:
            logger.error(f"Error rehearsing the daily lock-time run: {e}")
            return False

        if result.returncode != 0:
            logger.error(f"Daily lock-time rehearsal failed: {result.stderr}")
            return False
        logger.info("Daily lock-time rehearsal passed")
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

    def verify_installed(self) -> bool:
        """Compare the INSTALLED task with the committed XML, field by field. Read-only.

        Prints one ``FIELD= committed | installed | MATCH`` line per compared field, the NFL
        tasks installed on this machine, and ``READBACK_MATCH= True|False``. True needs every
        field to match AND exactly one NFL task installed -- this one -- so no weekly task can be
        left firing beside the daily one.
        """
        committed = (
            (self.nfl_predict_home / "deployment" / "windows_scheduler.xml")
            .read_bytes()
            .decode("utf-16")
        )
        export = subprocess.run(
            ["schtasks", "/query", "/tn", TASK_NAME, "/xml"], capture_output=True
        )
        if export.returncode != 0:
            _emit(f"TASK_NAME= {TASK_NAME} is not installed")
            _emit("READBACK_MATCH= False")
            return False
        listing = subprocess.run(
            ["schtasks", "/query", "/fo", "csv", "/nh"], capture_output=True, text=True
        )
        installed_tasks = nfl_task_names(listing.stdout)

        _emit(f"TASK_NAME= {TASK_NAME}")
        rows = compare_task_fields(committed, _decode_task_export(export.stdout))
        for name, want, got, match in rows:
            _emit(f"{name}= {want} | {got} | {'MATCH' if match else 'MISMATCH'}")
        _emit(f"NFL_TASKS= {installed_tasks}")
        matched = all(match for *_row, match in rows) and installed_tasks == [TASK_NAME]
        _emit(f"READBACK_MATCH= {matched}")
        return matched


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
    parser.add_argument(
        "--test",
        action="store_true",
        help="Rehearse the daily entry point with --dry-run (no writes)",
    )
    parser.add_argument(
        "--rehearsal-date",
        type=date.fromisoformat,
        help="The ET day the --test rehearsal runs as (YYYY-MM-DD); use one in season",
    )
    parser.add_argument(
        "--status", action="store_true", help="Show current scheduling status"
    )
    parser.add_argument(
        "--verify-installed",
        action="store_true",
        help="Compare the installed task with the committed XML (read-only)",
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

    if args.verify_installed:
        return 0 if setup.verify_installed() else 1

    if args.test:
        success = setup.test_scripts(rehearsal_date=args.rehearsal_date)

    if args.install:
        try:
            require_eastern_time_zone(read_windows_time_zone())
        except NonEasternTimeZoneError as exc:
            logger.error(str(exc))
            return 1

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
