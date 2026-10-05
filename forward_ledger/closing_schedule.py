"""The closing-line wake task: triggers generated from the real schedule (Phase 34, LDGR-07, D-07).

ONE PERSISTENT TASK, TRIGGERS REGENERATED DAILY
-----------------------------------------------
``NFL_Predict_Closing`` is defined by the committed template ``deployment/
windows_closing_scheduler.xml``, which carries the daily task's principal and settings and NO
trigger. :func:`build_closing_task_xml` inserts one ``TimeTrigger`` per reading window
(:func:`forward_ledger.closing_windows.windows_for_schedule`) over the next :data:`HORIZON_DAYS`
days, and :func:`register_closing_triggers` -- called by the daily run -- re-registers the task with
``schtasks /create /xml <file> /f`` and reads it back. 34-RESEARCH section C chose this over a
fixed weekly trigger set (which cannot cover Thanksgiving, Christmas, flexed or playoff slots
without becoming a polling timer) and over many one-shot tasks (which breaks the installer's
read-back and needs cleanup). Each registration covers 8 days, so one failed daily run loses
nothing.

THE TRIGGER BOUNDARIES
----------------------
``StartBoundary`` is naive Eastern wall-clock time: Task Scheduler fires it in machine-local time,
which is Eastern only on an Eastern machine -- so registration refuses any other zone, exactly as
the installer does. ``EndBoundary`` is :data:`TRIGGER_END_AFTER` later, so a trigger that was not
fired in time can never fire late (and ``StartWhenAvailable`` is false in the template, so a missed
one is never run on wake). The read-back compares the FULL ``(StartBoundary, EndBoundary,
Enabled)`` tuples (:func:`expected_triggers`): a drifted or missing EndBoundary, or a disabled
trigger, is a mismatch even when every start time matches.

NEVER RAISES, NEVER TOUCHES THE SCHEDULER IN A TEST
---------------------------------------------------
Every ``schtasks`` call goes through a :class:`SchtasksRunner`; the default
(:func:`run_schtasks`) is the only place a process is spawned. Registration records why it skipped
(not installed, currently running, machine not Eastern, dry run) or failed, and never raises: the
previous registration still covers the coming days, and the owner-run ``python
deployment/setup_scheduling.py --install-closing`` remains the fallback.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import csv
import io
import subprocess
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from data.write_sink import current_sink
from forward_ledger.closing_windows import ReadingWindow, windows_for_schedule
from utils.date_utils import ET as EASTERN
from utils.logging_config import get_logger

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "CLOSING_TASK_NAME",
    "CLOSING_TEMPLATE_PATH",
    "GENERATED_TASK_XML_PATH",
    "HORIZON_DAYS",
    "TRIGGER_END_AFTER",
    "ClosingReadback",
    "SchtasksResult",
    "SchtasksRunner",
    "TriggerRegistration",
    "build_closing_task_xml",
    "expected_trigger_starts",
    "expected_triggers",
    "install_closing_task",
    "installed_triggers",
    "read_back_closing_task",
    "register_closing_triggers",
    "run_schtasks",
]

logger = get_logger(__name__)

#: The closing task's name; the installer read-back allows exactly this one beside the daily task.
CLOSING_TASK_NAME = "NFL_Predict_Closing"

#: The committed template, relative to the project home (the scheduled run's working directory).
CLOSING_TEMPLATE_PATH = Path("deployment/windows_closing_scheduler.xml")

#: Where the generated definition is written for ``schtasks /create /xml``; relative to the home.
GENERATED_TASK_XML_PATH = Path("logs/closing_task_generated.xml")

#: How long after its start a trigger stays able to fire; after that it is stale and never fires.
TRIGGER_END_AFTER = timedelta(minutes=30)

#: How many days ahead each registration covers. The daily run re-registers, so the horizon rolls.
HORIZON_DAYS = 8

_TASK_NS = "http://schemas.microsoft.com/windows/2004/02/mit/task"
_NS = f"{{{_TASK_NS}}}"
_XML_DECLARATION = '<?xml version="1.0" encoding="UTF-16"?>\n'

#: One trigger as the read-back compares it: ``(StartBoundary, EndBoundary, Enabled)``.
TriggerTuple = tuple[str, str, str]


# ---------------------------------------------------------------------------
# The trigger boundaries -- the ONE value every build and every read-back uses
# ---------------------------------------------------------------------------


def _wall_clock(instant: datetime) -> str:
    """*instant* as a naive Eastern wall-clock boundary, e.g. ``2026-10-11T12:50:00``."""
    return (
        instant.astimezone(EASTERN).replace(tzinfo=None).isoformat(timespec="seconds")
    )


def _trigger_tuples(windows: Iterable[ReadingWindow]) -> list[TriggerTuple]:
    """The trigger each window becomes, in window order."""
    return [
        (
            _wall_clock(window.start_utc),
            _wall_clock(window.start_utc + TRIGGER_END_AFTER),
            "true",
        )
        for window in windows
    ]


def expected_trigger_starts(games: pd.DataFrame, now: datetime) -> list[str]:
    """The StartBoundary of every trigger a registration at *now* carries, earliest first."""
    windows = windows_for_schedule(games, now, days=HORIZON_DAYS)
    return [start for start, _end, _enabled in _trigger_tuples(windows)]


def expected_triggers(games: pd.DataFrame, now: datetime) -> frozenset[TriggerTuple]:
    """The ``(StartBoundary, EndBoundary, Enabled)`` set a registration at *now* carries."""
    return frozenset(
        _trigger_tuples(windows_for_schedule(games, now, days=HORIZON_DAYS))
    )


# ---------------------------------------------------------------------------
# Building the task definition
# ---------------------------------------------------------------------------


def _parse(xml_text: str) -> ET.Element:
    """Parse a task definition. Comments are not kept: the committed template documents the
    task, and the generated definition is a derived artifact."""
    return ET.fromstring(xml_text.strip())


def _triggers_element(root: ET.Element) -> ET.Element:
    triggers = root.find(f"{_NS}Triggers")
    if triggers is None:
        msg = "the closing task template has no <Triggers> element"
        raise ValueError(msg)
    return triggers


def build_closing_task_xml(
    template_bytes: bytes, windows: Sequence[ReadingWindow]
) -> bytes:
    """The template with one ``TimeTrigger`` per window, as UTF-16 LE with a BOM and CRLF.

    Everything but the triggers is the template's, unchanged. Each trigger carries its
    ``StartBoundary``, ``EndBoundary`` (+:data:`TRIGGER_END_AFTER`) and ``Enabled`` true, in the
    order the task schema requires.
    """
    root = _parse(template_bytes.decode("utf-16"))
    triggers = _triggers_element(root)
    for trigger in list(triggers):
        triggers.remove(trigger)
    for start, end, enabled in _trigger_tuples(windows):
        trigger = ET.SubElement(triggers, f"{_NS}TimeTrigger")
        ET.SubElement(trigger, f"{_NS}StartBoundary").text = start
        ET.SubElement(trigger, f"{_NS}EndBoundary").text = end
        ET.SubElement(trigger, f"{_NS}Enabled").text = enabled
    ET.indent(root, space="  ")
    # Serialize the task namespace as the DEFAULT one, as Task Scheduler writes it, not as an
    # ``ns0:`` prefix. tostring's ``default_namespace`` option cannot be used: it refuses the
    # schema's unqualified attributes (``version``, ``id``). Registering the prefix is
    # idempotent and only affects how this namespace is spelled.
    ET.register_namespace("", _TASK_NS)
    body = ET.tostring(root, encoding="unicode")
    text = (_XML_DECLARATION + body + "\n").replace("\r\n", "\n").replace("\n", "\r\n")
    return b"\xff\xfe" + text.encode("utf-16-le")


# ---------------------------------------------------------------------------
# The read-back
# ---------------------------------------------------------------------------


def _enabled(element: ET.Element) -> str:
    """An element's ``<Enabled>`` child; absent means the schema default, ``true``."""
    flag = element.find(f"{_NS}Enabled")
    return "true" if flag is None else (flag.text or "").strip().lower()


def _child_text(element: ET.Element, name: str) -> str:
    child = element.find(f"{_NS}{name}")
    return "" if child is None else (child.text or "").strip()


def installed_triggers(xml_text: str) -> list[TriggerTuple]:
    """Every trigger of a task definition as a ``(StartBoundary, EndBoundary, Enabled)`` tuple.

    A trigger that is not a ``TimeTrigger`` (a repeating calendar trigger would be the polling
    timer D-07 forbids) carries its kind in the start slot, so it can never equal an expected one.
    """
    tuples = []
    for trigger in _triggers_element(_parse(xml_text)):
        start = _child_text(trigger, "StartBoundary")
        kind = trigger.tag.rsplit("}", maxsplit=1)[-1]
        if kind != "TimeTrigger":
            start = f"{kind} {start}"
        tuples.append((start, _child_text(trigger, "EndBoundary"), _enabled(trigger)))
    return tuples


def _without_triggers(xml_text: str) -> str:
    """*xml_text* with every trigger removed: what is left is the task's settings."""
    root = _parse(xml_text)
    triggers = _triggers_element(root)
    for trigger in list(triggers):
        triggers.remove(trigger)
    return ET.tostring(root, encoding="unicode")


@dataclass(frozen=True)
class ClosingReadback:
    """The installed closing task compared with the expected one.

    Attributes:
        rows: ``(field, expected, installed, match)`` per settings field, the daily task's
            read-back fields without the two trigger ones (the triggers are compared below).
        expected_triggers: The expected trigger tuples, sorted.
        installed_triggers: The installed trigger tuples, sorted (duplicates kept).
    """

    rows: tuple[tuple[str, str, str, bool], ...]
    expected_triggers: tuple[TriggerTuple, ...]
    installed_triggers: tuple[TriggerTuple, ...]

    @property
    def triggers_match(self) -> bool:
        return self.expected_triggers == self.installed_triggers

    @property
    def match(self) -> bool:
        return self.triggers_match and all(match for *_row, match in self.rows)

    @property
    def triggers_line(self) -> str:
        verdict = "MATCH" if self.triggers_match else "MISMATCH"
        return (
            f"CLOSING_TRIGGERS= {len(self.expected_triggers)} | "
            f"{len(self.installed_triggers)} | {verdict}"
        )


def read_back_closing_task(
    expected_xml: str, installed_xml: str, expected: frozenset[TriggerTuple]
) -> ClosingReadback:
    """Compare an installed closing task with *expected_xml* (settings) and *expected* (triggers).

    The settings are compared with the installer's own normalising field comparison, so the
    closing task is held to the same read-back the daily task has.
    """
    # Lazy import: deployment.setup_scheduling imports this module at load time.
    from deployment.setup_scheduling import READBACK_FIELDS, compare_task_fields

    settings_fields = [f for f in READBACK_FIELDS if f not in ("Trigger", "StartTime")]
    rows = [
        row
        for row in compare_task_fields(
            _without_triggers(expected_xml), _without_triggers(installed_xml)
        )
        if row[0] in settings_fields
    ]
    return ClosingReadback(
        rows=tuple(rows),
        expected_triggers=tuple(sorted(expected)),
        installed_triggers=tuple(sorted(installed_triggers(installed_xml))),
    )


# ---------------------------------------------------------------------------
# The schtasks seam
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SchtasksResult:
    """One ``schtasks`` invocation's exit code and raw output."""

    returncode: int
    stdout: bytes
    stderr: bytes = b""


class SchtasksRunner(Protocol):
    """Runs ``schtasks`` with *args* (everything after the program name)."""

    def __call__(self, args: Sequence[str]) -> SchtasksResult: ...


def run_schtasks(args: Sequence[str]) -> SchtasksResult:
    """The default runner, and the only place this module spawns a process."""
    completed = subprocess.run(["schtasks", *args], capture_output=True, check=False)
    return SchtasksResult(completed.returncode, completed.stdout, completed.stderr)


def _console_text(raw: bytes) -> str:
    """``schtasks`` output: UTF-16 with a BOM (an XML export), else console text."""
    from deployment.setup_scheduling import _decode_task_export

    return _decode_task_export(raw)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TriggerRegistration:
    """What a registration did, for the run log.

    Attributes:
        outcome: ``registered``, ``skipped`` or ``failed``.
        reason: Why it skipped or failed, or why a registration did not read back; None when
            it registered and read back as expected.
        expected_starts: The StartBoundary of every trigger it registered (or would have).
        readback_match: True only when the installed task read back exactly as expected.
        readback: The full comparison, when one was made.
    """

    outcome: str
    reason: str | None
    expected_starts: tuple[str, ...]
    readback_match: bool
    readback: ClosingReadback | None = None

    @property
    def readback_line(self) -> str:
        return f"CLOSING_READBACK_MATCH= {self.readback_match}"


def install_closing_task(
    games: pd.DataFrame,
    now: datetime,
    *,
    runner: SchtasksRunner | None = None,
    template_path: Path = CLOSING_TEMPLATE_PATH,
    generated_path: Path = GENERATED_TASK_XML_PATH,
) -> TriggerRegistration:
    """Register the closing task with the triggers due at *now*, then read it back.

    Creates the task when it is absent and replaces it (``/f``) when present. The generated
    definition is written through the write sink, so a dry run records the write and registers
    nothing. Never raises: every error becomes a ``failed`` outcome.
    """
    run = runner or run_schtasks
    starts: tuple[str, ...] = ()
    try:
        windows = windows_for_schedule(games, now, days=HORIZON_DAYS)
        expected = frozenset(_trigger_tuples(windows))
        starts = tuple(start for start, _end, _enabled in _trigger_tuples(windows))
        generated = build_closing_task_xml(template_path.read_bytes(), windows)

        if not current_sink().authorize(str(generated_path), "closing_task_xml"):
            return TriggerRegistration("skipped", "dry_run", starts, False)
        generated_path.parent.mkdir(parents=True, exist_ok=True)
        generated_path.write_bytes(generated)

        created = run(
            ["/create", "/tn", CLOSING_TASK_NAME, "/xml", str(generated_path), "/f"]
        )
        if created.returncode != 0:
            detail = _console_text(created.stderr).strip()
            return TriggerRegistration(
                "failed", f"create_failed: {detail}", starts, False
            )

        export = run(["/query", "/tn", CLOSING_TASK_NAME, "/xml"])
        if export.returncode != 0:
            detail = _console_text(export.stderr).strip()
            return TriggerRegistration(
                "failed", f"readback_query_failed: {detail}", starts, False
            )
        readback = read_back_closing_task(
            generated.decode("utf-16"), _console_text(export.stdout), expected
        )
    except Exception as exc:  # noqa: BLE001 -- recorded, never raised (the next run retries)
        logger.warning(f"closing task registration failed: {exc!r}")
        return TriggerRegistration("failed", f"error: {exc!r}", starts, False)

    registration = TriggerRegistration(
        "registered",
        None if readback.match else "readback_mismatch",
        starts,
        readback.match,
        readback,
    )
    logger.info(f"{readback.triggers_line}; {registration.readback_line}")
    return registration


def _default_zone() -> str:
    from deployment.setup_scheduling import read_windows_time_zone

    return read_windows_time_zone()


def _task_status(listing: str) -> str:
    """The Status column of the closing task's row in ``schtasks /query /fo csv /nh``."""
    for row in csv.reader(io.StringIO(listing)):
        if row and row[0].lstrip("\\") == CLOSING_TASK_NAME:
            return row[-1].strip()
    return ""


def register_closing_triggers(
    games: pd.DataFrame,
    now: datetime,
    *,
    runner: SchtasksRunner | None = None,
    zone_reader: Callable[[], str] | None = None,
    template_path: Path = CLOSING_TEMPLATE_PATH,
    generated_path: Path = GENERATED_TASK_XML_PATH,
) -> TriggerRegistration:
    """Re-register the INSTALLED closing task's triggers from the schedule (the daily run's call).

    Skips -- never fails -- when the machine is not on Eastern time (``machine_not_eastern``),
    when the closing task is not installed (``closing_task_not_installed``: installing it is the
    owner's ``--install-closing``), or when it is running right now (``closing_task_running``:
    the previous registration still covers the coming days). Otherwise
    :func:`install_closing_task`. Never raises.
    """
    # Lazy import: deployment.setup_scheduling imports this module at load time.
    from deployment.setup_scheduling import EASTERN_WINDOWS_ZONE

    run = runner or run_schtasks
    try:
        zone = (zone_reader or _default_zone)()
        if zone != EASTERN_WINDOWS_ZONE:
            return TriggerRegistration("skipped", "machine_not_eastern", (), False)

        listing = run(["/query", "/tn", CLOSING_TASK_NAME, "/fo", "csv", "/nh"])
        if listing.returncode != 0:
            return TriggerRegistration(
                "skipped", "closing_task_not_installed", (), False
            )
        if _task_status(_console_text(listing.stdout)).casefold() == "running":
            return TriggerRegistration("skipped", "closing_task_running", (), False)
    except Exception as exc:  # noqa: BLE001 -- recorded, never raised (the next run retries)
        logger.warning(f"closing task registration failed: {exc!r}")
        return TriggerRegistration("failed", f"error: {exc!r}", (), False)

    return install_closing_task(
        games,
        now,
        runner=run,
        template_path=template_path,
        generated_path=generated_path,
    )
