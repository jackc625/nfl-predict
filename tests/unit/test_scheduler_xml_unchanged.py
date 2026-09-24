"""The Windows scheduler definition is PINNED byte-for-byte (Phase 31, plan 31-17; T-31-87).

WHY A PIN AND NOT AN EDIT
--------------------------
Plan 31-17 replaced the INTERNALS of the Friday pipeline's weekly recommendation step. It did not
add a step, remove one, change the entry point, or change the schedule -- the step is ALREADY
scheduled, and ``deployment/windows_scheduler.xml`` names the pipeline entry point, not the step.
So the correct diff to that file is the empty one, and this test is what makes "byte-unchanged" a
fact the suite re-checks rather than a claim in a summary.

The Phase-29 precedent is the same rule seen from the other side: when that phase genuinely needed
a new scheduled job it REGISTERED A SEPARATE ``windows_odds_timeline_scheduler.xml`` rather than
editing the pinned one. Either the definition is untouched, or a new definition is added beside it.

WHY A CONTENT HASH RATHER THAN A GIT CHECK
-------------------------------------------
``git diff -- deployment/windows_scheduler.xml`` answers a question about the INDEX. A content hash
answers the question actually being asked -- are these the same bytes -- and it keeps answering it
after the change would have been committed, which is exactly when a silent edit stops being
visible in a diff.

The hash is over the file's RAW BYTES. A line-ending normalisation would be a real change to a
file Windows Task Scheduler imports verbatim, so it must fail here rather than be smoothed away.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import hashlib
import pathlib
import xml.etree.ElementTree as ET
from datetime import datetime, time

SCHEDULER_PATH = pathlib.Path("deployment") / "windows_scheduler.xml"

# Resolved from the committed file at test-authoring time (plan 31-17, 2026-09-06), which is the
# same content Phase 21 documented and Phase 29 left BYTE-UNCHANGED when it registered its own
# separate timeline-capture definition.
EXPECTED_SHA256 = "ac93a32a07574eb2f3998439f600c5c4fa773998e39a347e381bdf0ca2a767ff"
EXPECTED_BYTES = 6186

#: THE LOCK every scheduled run must beat: 18:00 ET on the day before each game (D33.2-01).
LOCK_TIME_OF_DAY = time(18, 0)

#: How far ahead of the lock the daily trigger must fire. Plan 33.2-27 Task 2c's ruling is
#: ``finish-before-lock`` -- collection, the full-history build, prediction and emission ALL
#: finish before the lock -- so this covers the WHOLE run, not collection alone. MEASURED
#: 2026-09-24 (Plan 33.2-27, step 27b proof run): 850.5 s of steps plus about 20 s of schedule
#: refresh = 14.5 min; plus one live-skip round, which repeats the full build (743.8 s inside the
#: proof run) = 12.4 min. 14.5 + 12.4 = 26.9, rounded up.
MIN_COLLECTION_MARGIN_MINUTES = 27

_TASK_NS = "{http://schemas.microsoft.com/windows/2004/02/mit/task}"

_DAY_OF_WEEK_ELEMENTS = frozenset(
    {
        "DaysOfWeek",
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    }
)

_WHY = (
    "The Friday step whose internals plan 31-17 replaced is ALREADY scheduled, and this file "
    "names the pipeline ENTRY POINT rather than any individual step -- so replacing a step body "
    "must produce NO diff here. If a genuinely new scheduled job is needed, register a SEPARATE "
    "definition beside this one (the Phase-29 windows_odds_timeline_scheduler.xml precedent) "
    "rather than editing the pinned file. If this pin is being updated deliberately, say so in "
    "the commit and update EXPECTED_SHA256 and EXPECTED_BYTES together."
)


def test_the_scheduler_definition_exists() -> None:
    """Fail fast rather than hashing nothing: an absent file would make the pin vacuous."""
    assert SCHEDULER_PATH.is_file(), (
        f"the pinned scheduler definition is missing at {SCHEDULER_PATH.as_posix()}; the hash "
        f"pin cannot be evaluated. {_WHY}"
    )


def test_the_scheduler_definition_is_byte_unchanged() -> None:
    """The committed definition hashes to its pinned value (T-31-87)."""
    raw = SCHEDULER_PATH.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()

    assert len(raw) == EXPECTED_BYTES, (
        f"{SCHEDULER_PATH.as_posix()} is {len(raw)} bytes, pinned at {EXPECTED_BYTES}. {_WHY}"
    )
    assert actual == EXPECTED_SHA256, (
        f"{SCHEDULER_PATH.as_posix()} content changed.\n  expected sha256 {EXPECTED_SHA256}\n"
        f"  actual   sha256 {actual}\n{_WHY}"
    )


def test_the_pin_covers_the_only_scheduler_definition_in_the_tree() -> None:
    """Every ``*.xml`` under ``deployment/`` is either the pinned file or a declared sibling.

    A second definition is LEGITIMATE -- that is the Phase-29 precedent -- but it must be visible
    here, so a new scheduled job cannot be added without this list being updated. Without this
    test the pin would guard one file while an unpinned second one appeared beside it.
    """
    declared = {SCHEDULER_PATH.as_posix()}
    found = {
        p.as_posix()
        for p in pathlib.Path("deployment").rglob("*.xml")
        if "__pycache__" not in p.parts
    }
    assert found == declared, (
        "the set of scheduler definitions under deployment/ moved.\n"
        f"  declared: {sorted(declared)}\n  found:    {sorted(found)}\n"
        "A new definition beside the pinned one is the correct way to add a scheduled job; "
        "declare it here so it is covered too."
    )


# ---------------------------------------------------------------------------
# What the pinned file SAYS (Plan 33.2-28). A pin proves the bytes did not change, not that they
# are right: a trigger left at or moved past 18:00 would pass the pin and the pin would then
# certify it. Read off the PARSED tree -- the header comment names the weekly Friday run this
# replaced, and an XML comment is not an element.
# ---------------------------------------------------------------------------


def _parsed() -> ET.Element:
    return ET.fromstring(SCHEDULER_PATH.read_bytes().decode("utf-16"))


def _local_names(root: ET.Element) -> set[str]:
    return {element.tag.split("}")[-1] for element in root.iter()}


def _text(root: ET.Element, name: str) -> list[str]:
    return [(e.text or "").strip() for e in root.iter(f"{_TASK_NS}{name}")]


def test_the_trigger_is_daily_with_no_day_of_week_element() -> None:
    names = _local_names(_parsed())
    assert "ScheduleByDay" in names
    assert "ScheduleByWeek" not in names
    assert not names & _DAY_OF_WEEK_ELEMENTS, sorted(names & _DAY_OF_WEEK_ELEMENTS)


def test_the_arguments_name_the_daily_entry_point() -> None:
    (arguments,) = _text(_parsed(), "Arguments")
    assert "daily_lock_pipeline" in arguments
    assert "friday_pipeline" not in arguments


def test_the_trigger_fires_ahead_of_the_lock_by_the_measured_run() -> None:
    boundaries = _text(_parsed(), "StartBoundary")
    assert boundaries, "the trigger has no StartBoundary"
    lock_minutes = LOCK_TIME_OF_DAY.hour * 60 + LOCK_TIME_OF_DAY.minute
    for boundary in boundaries:
        fires = datetime.fromisoformat(boundary)
        assert fires.tzinfo is None, (
            "the trigger fires in machine-local time; no zone suffix"
        )
        fires_minutes = fires.hour * 60 + fires.minute
        assert fires_minutes < lock_minutes, f"{boundary} is not before the 18:00 lock"
        assert lock_minutes - fires_minutes >= MIN_COLLECTION_MARGIN_MINUTES, (
            f"{boundary} leaves {lock_minutes - fires_minutes} min before the lock; the "
            f"measured run needs {MIN_COLLECTION_MARGIN_MINUTES}"
        )


def test_a_missed_trigger_is_never_run_late() -> None:
    """StartWhenAvailable true would run a missed trigger on wake, possibly after the lock."""
    assert _text(_parsed(), "StartWhenAvailable") == ["false"]
