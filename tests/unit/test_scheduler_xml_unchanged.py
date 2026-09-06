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

SCHEDULER_PATH = pathlib.Path("deployment") / "windows_scheduler.xml"

# Resolved from the committed file at test-authoring time (plan 31-17, 2026-09-06), which is the
# same content Phase 21 documented and Phase 29 left BYTE-UNCHANGED when it registered its own
# separate timeline-capture definition.
EXPECTED_SHA256 = "ac93a32a07574eb2f3998439f600c5c4fa773998e39a347e381bdf0ca2a767ff"
EXPECTED_BYTES = 6186

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
