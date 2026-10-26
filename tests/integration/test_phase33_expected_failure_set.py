"""The five deliberate tripwires are still RED, and nothing else in their run is.

WHY THIS GATE EXISTS (T-33-07)
------------------------------
Five tests on this checkout fail on purpose. Each encodes an owner-accepted fact from
Phase 30 or Phase 31 -- a frozen gate baseline that diverges from a re-score in 47 of 68
fields and was deliberately NOT re-frozen; a non-clock column that moved inside a
protected season during the rung-3 rebuild. A phase that turned one of these green would
have ERASED A DISCLOSURE, not repaired a defect.

Phase 33 is eighteen plans long and several of them rebuild a production store. That is
precisely the circumstance under which a tripwire stops failing by accident, and nobody
notices, because a suite reporting four reds instead of five looks like progress.

THE SET COMES FROM ONE COMMITTED CONSTANT
-----------------------------------------
Every node id this module asserts against is read from
``tests.phase33_state.DELIBERATE_TRIPWIRE_NODE_IDS``. None is re-listed here. A gate that
asserts a set it also defines proves nothing: the two copies drift, the copy in the gate
wins, and the record is gone. A test in this module enforces that discipline mechanically
by scanning its own source for node-id-shaped string literals.

OUTCOMES ARE READ FROM JUNIT XML, AND EVERY STATE IS KEPT (Phase 38, D-15)
-------------------------------------------------------------------------
Until Phase 38 this gate parsed the ``FAILED``/``ERROR`` lines of a ``-q`` report and
counted an ERROR as red. A tripwire that ERRORED therefore passed as "still red", and a
tripwire that failed its assertion AND errored at teardown showed only the failure. The
gate now runs each child pytest with ``--junitxml`` and reads OUTCOMES:

* ``junit_key`` turns a node id into the report's ``(classname, name)`` with
  ``_pytest.junitxml.mangle_test_address``, the function that wrote the report;
* ``junit_outcomes`` maps every testcase to the SET of states it shows: ``failed``
  (a ``<failure>`` child whose message is an assertion), ``crashed`` (a ``<failure>``
  child whose message is any other exception, such as ``KeyError``: the test never
  reached its check), ``error`` (an ``<error>`` child: setup, teardown or
  collection), ``xfailed`` (``<skipped type="pytest.xfail">``), ``skipped`` (any other
  ``<skipped>``) or ``passed`` (no child). A call failure plus a teardown error is TWO
  testcase elements for one node; both states are kept, never collapsed to one;
* ``expected_outcome_problems`` is the ONE comparison. An empty list is the passing
  case; each problem names its kind:

  - ``DISCLOSURE ERASED`` -- a tripwire passed, or did not run at all;
  - ``TRIPWIRE NOT A FAILURE`` -- a tripwire crashed, errored, skipped or xfailed
    instead of failing. Only an assertion failure is the owner-accepted red;
  - ``TRIPWIRE ALSO ERRORED`` -- a tripwire failed AND errored (for example at teardown,
    where a COLD-05 per-test production-store guard surfaces). It is its own problem, so
    an error is never hidden behind a tripwire's expected failure;
  - ``REGRESSION`` -- a node that is not a tripwire failed, crashed or errored (one
    message per node, even when it did more than one);
  - ``QUARANTINE`` -- any node anywhere xfailed. The expected set holds zero quarantines.

A strict subset is a violation, not an improvement. That is the whole claim.

THE SESSION-END SWEEP IS READ BESIDE THE REPORT, NOT FROM IT
-----------------------------------------------------------
``tests/conftest.py``'s ``pytest_sessionfinish`` re-hashes every production store at the
end of a session. It adds no testcase and only flips an exit status that was 0, printing
a terminal section headed ``SESSION-END FULL CONTENT SWEEP``. A judge that read JUnit
alone could never see it, so every judged run's OUTPUT is checked for that section with
``carries_closing_sweep`` as well.

PHASE-33 HAZARDS, PRE-DECLARED HERE BEFORE THE PLANS THAT CREATE THEM RUN
------------------------------------------------------------------------
* Tripwire 2, ``test_gold_rebuild_attribution.py::
  TestThePhase31Rung3IsTheFullRebuildOfTheVerdictPopulation::
  test_no_NON_CLOCK_column_moved_in_a_protected_season``, is a GOLD-REBUILD ATTRIBUTION
  test, and Plan 33-14 rebuilds gold. It is expected to STAY RED. A rebuild that happens
  to satisfy it would be clearing an owner-accepted disclosure, not fixing a defect, and
  the correct response would be to investigate why the attribution changed -- never to
  record the green as progress.
* Tripwires 3 and 4 live in ``test_n01_resync_control.py``, the module Plan 33-01 marked
  with a path-scoped ``writes_production_store`` on its ONE legitimate writer. The marker
  is on ``TestTheResyncIsIdempotent`` alone; both tripwires are deliberately unmarked and
  the marker must not change their outcome.

WHAT THE DEFAULT RUN MEASURES, AND WHAT IT DOES NOT
---------------------------------------------------
The verdict subprocess runs EXACTLY the tripwire node ids -- measured at 10.5 s, against
528 s for the whole integration tier. That bounded run fully proves the direction that
matters here (a tripwire that stopped failing, or stopped failing CLEANLY) and proves the
other direction only within its own run.

The whole-tier form, which would also catch a NEW red elsewhere in the tier, is opt-in
via ``NFL_PHASE33_FULL_TIER_VERDICT`` for two reasons that are not cost: this module runs
INSIDE the tier it would measure, so an unguarded tier run re-collects this module and
spawns a child of a child without end; and the tier-wide reconciliation is the phase
close's recorded three-tier measurement, which owns it properly. The recursion guard is
``NFL_PHASE33_EXPECTED_FAILURE_CHILD``, set on every child environment, and a child that
sees it steps aside by name.

THE RECORDED FULL-SUITE RUN OF THE PHASE 38 CLOSE (D-06, D-15, D-28)
--------------------------------------------------------------------
A single-process whole suite is killed for memory on this machine, so one full-suite run
is one owner-approved window of THREE sequential pytest processes -- the instrument of
``tests/phase33_state.py``: ``tests/unit``; ``tests/integration``; and ``tests/api``
together with the root ``tests/test_*.py`` modules. The run step (Plans 38-11 and 38-12)
writes one directory holding, for each tier ``unit``, ``integration`` and ``api``:

* ``<tier>.collect.txt`` -- the tier first listed with ``--collect-only -q``, at the
  same HEAD in the same window;
* ``<tier>.xml`` and ``<tier>.log`` -- the tier then run with
  ``-q -rfE -p no:cacheprovider --junitxml=<dir>/<tier>.xml``, its output in the log.

``full_run_problems`` judges that directory and runs nothing itself. It requires all
nine files and ``head.txt``; requires each listing to close with a clean ``N tests
collected`` line, to name N node ids, all under its own tier's root, and every one of
them to have an outcome in that tier's report; given the slot that records the run, it
also requires ``head.txt`` and each listing's size to equal the slot's
(``INCOMPLETE``: one testcase per tier does not prove a run complete, and a tier killed
by its timeout writes no report); requires ``exits.txt`` (``<tier> <exit code>`` per
line, as the run step appends it) to show unit and api at 0 and integration at 0 or 1;
refuses any tier log carrying the session-end sweep, or not ending on pytest's closing
summary line -- a sweep that raised leaves a clean report and a log ending on its
traceback (``BOUNDARY``); and judges the union of the three reports with the same
``expected_outcome_problems`` the bounded gate uses. The opt-in node
``test_the_recorded_full_run_report_when_given`` reads the directory from
``NFL_PHASE38_FULL_RUN_DIR``; no child this module spawns ever inherits it.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping
from pathlib import Path

import pytest
from _pytest.junitxml import mangle_test_address

from tests import phase33_state, phase34_state

REPO_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_DATA_TREE = REPO_ROOT / "data"

# The tier each tripwire belongs to, DERIVED from its own node id rather than restated.
# Every tripwire lives under tests/integration today; deriving the tier means a tripwire
# that moves tiers is reflected automatically instead of silently mis-grouped.
TIER_ROOTS = ("tests/unit", "tests/integration", "tests/api")

FULL_TIER_ENV = "NFL_PHASE33_FULL_TIER_VERDICT"
CHILD_ENV = "NFL_PHASE33_EXPECTED_FAILURE_CHILD"

# The recorded full-suite run of the Phase 38 close (D-06, D-15, D-28): the directory it
# wrote, and the three tiers it ran as separate processes, in the instrument's order.
FULL_RUN_DIR_ENV = "NFL_PHASE38_FULL_RUN_DIR"
FULL_RUN_TIERS: tuple[str, ...] = ("unit", "integration", "api")
_FULL_RUN_TIER_SUFFIXES = (".xml", ".log", ".collect.txt")
FULL_RUN_HEAD_FILE = "head.txt"

# The node ids each recorded tier may list: the instrument's three processes run
# tests/unit, tests/integration, and tests/api with the root tests/test_*.py modules. A
# listing outside its own root is a run over some other part of the suite.
_FULL_RUN_TIER_ID_RE: dict[str, re.Pattern[str]] = {
    "unit": re.compile(r"^tests/unit/"),
    "integration": re.compile(r"^tests/integration/"),
    "api": re.compile(r"^tests/(api/|test_[^/]*\.py::)"),
}

# `3603 tests collected in 41.2s` / `1 test collected in 0.3s`; a line that also names an
# error is a listing of a broken collection, not a complete one.
_COLLECTED_LINE_RE = re.compile(r"^(\d+) tests? collected")

# Mirrors tests/conftest.py's CLOSING_SWEEP_HEADER exactly, as
# tests/integration/test_data_boundary_guard_arming.py does: conftest is a pytest plugin
# and is not imported. The closing sweep adds no JUnit testcase and only flips an exit
# status that was 0, so a JUnit-only judge cannot see it; the run's output is read too.
CLOSING_SWEEP_HEADER = "SESSION-END FULL CONTENT SWEEP"

# pytest's closing stats line, e.g. `5 failed, 1767 passed, 59 skipped, 622 warnings in
# 1816.40s (0:30:16)`. The terminal reporter writes it LAST, after every other
# `pytest_sessionfinish` hook has returned -- the conftest's closing sweep included. A
# sweep that raised (a locked or mid-replace store refuses by name) leaves a log that
# ends on its traceback instead, so a log that does not end on this line is a session
# whose sweep never completed.
_SUMMARY_LINE_RE = re.compile(
    r"^=*\s*\d+ (failed|passed|skipped|errors?|xfailed|xpassed)\b.* in \d"
)

# What the run step appends to `exits.txt` after each tier: `<tier> <exit code>`. Only
# the tier holding the tripwires may exit 1; an internal error or interrupt is no run.
FULL_RUN_EXITS_FILE = "exits.txt"
_ALLOWED_TIER_EXITS: dict[str, frozenset[int]] = {
    "unit": frozenset({0}),
    "integration": frozenset({0, 1}),
    "api": frozenset({0}),
}

NO_PRODUCTION_DATA_SKIP = (
    "the deliberate tripwires cannot be RUN on this checkout: the production data tree "
    "at data/ is absent, and every one of them reads gold parquet, the silver lake or "
    "the DuckDB store. The tripwires' node ids are still checked for collectibility "
    "(that half needs no data); only the verdict half steps aside. This is a control "
    "that did NOT run here, not a control that passed."
)

CHILD_SESSION_SKIP = (
    "this module is running inside a child pytest session spawned by its own whole-tier "
    "verdict (NFL_PHASE33_EXPECTED_FAILURE_CHILD is set). Spawning a grandchild would "
    "recurse without end, so the subprocess halves step aside by name in the child."
)

FULL_TIER_PENDING_SKIP = (
    "the whole-tier verdict is opt-in (set NFL_PHASE33_FULL_TIER_VERDICT=1). It re-runs "
    "the entire integration tier, which this module is itself part of, and the tier-wide "
    "reconciliation is Plan 33-18's closing three-tier measurement. The bounded verdict "
    "over the tripwire node ids runs by default and is what gates this phase."
)

FULL_RUN_PENDING_SKIP = (
    "the recorded full run of the Phase 38 close is judged by setting "
    "NFL_PHASE38_FULL_RUN_DIR to the run's directory. The run itself is an "
    "owner-approved step (D-06): three sequential tier processes that write each tier's "
    "JUnit report, log and collect listing there. This node only reads those files."
)

# How pytest's JUnit writer starts the message of an assertion failure (see
# ``_testcase_state``); any other call-phase failure message names a crash.
_ASSERTION_PREFIXES = ("AssertionError", "assert ")

# The states that make a node red: a non-tripwire showing any of them is a REGRESSION.
_RED_STATES = frozenset({"failed", "crashed", "error"})

# A JUnit outcome map: (classname, name) -> every state that testcase key showed.
Outcomes = Mapping[tuple[str, str], frozenset[str]]


def tier_of(node_id: str) -> str:
    """The tier root a node id belongs to.

    Args:
        node_id: A pytest node id, ``path::Class::test`` style.

    Returns:
        One of ``TIER_ROOTS``.

    Raises:
        AssertionError: when the node id sits under no known tier root.
    """
    normalised = node_id.replace("\\", "/")
    for root in TIER_ROOTS:
        if normalised.startswith(root + "/"):
            return root
    raise AssertionError(
        f"tripwire node id {node_id!r} sits under none of the tier roots {TIER_ROOTS!r}. "
        "The three-tier split is how this suite is run at all (a single-process whole "
        "suite is killed for memory on this machine), so a node outside it cannot be "
        "reconciled against any tier line."
    )


def normalised_node_id(node_id: str) -> str:
    """A node id with Windows path separators turned into ``/``, before any ``[`` only.

    A module, class or function name cannot hold a backslash, so one before the first
    ``[`` is a path separator. One after it belongs to the parametrize id, where pytest
    writes its escapes (``\\n``, ``\\xb2``, ``\\\\``) the same way in a collect listing and
    in a JUnit name; rewriting it would make the two disagree. This is the split
    ``mangle_test_address`` itself makes.

    Args:
        node_id: A pytest node id, ``path::Class::test[params]`` style.

    Returns:
        The node id with only its path part normalised.
    """
    path, bracket, params = node_id.partition("[")
    return path.replace("\\", "/") + bracket + params


def junit_key(node_id: str) -> tuple[str, str]:
    """The ``(classname, name)`` a JUnit report files a node id under.

    Built with ``_pytest.junitxml.mangle_test_address``, the function that wrote the
    report, so the mapping cannot drift from pytest's own (a parametrize id stays on the
    name). Forward mapping only: a classname cannot tell a module dot from a class dot.

    Args:
        node_id: A pytest node id; backslash path separators are normalised first, never
            inside the parametrize id (``normalised_node_id``).

    Returns:
        The testcase's ``(classname, name)``.
    """
    names = mangle_test_address(normalised_node_id(node_id))
    return ".".join(names[:-1]), names[-1]


def _testcase_state(testcase: ET.Element) -> str:
    """The one state a single ``<testcase>`` element records.

    pytest writes a ``<failure>`` for EVERY exception raised in the call phase, so its
    message decides which it was. An assertion is written ``assert ...`` (a rewritten
    bare assert) or ``AssertionError: ...`` (one with a message, or raised directly);
    anything else -- ``KeyError: ...``, ``FileNotFoundError: ...`` -- is a crash before
    the test reached its check, and is ``crashed``, never ``failed``.
    """
    children = {child.tag: child for child in testcase}
    if "failure" in children:
        message = children["failure"].get("message", "")
        return "failed" if message.startswith(_ASSERTION_PREFIXES) else "crashed"
    tags = {tag: child.get("type") for tag, child in children.items()}
    if "error" in tags:
        return "error"
    if "skipped" in tags:
        return "xfailed" if tags["skipped"] == "pytest.xfail" else "skipped"
    return "passed"


def junit_outcomes(report: Path) -> dict[tuple[str, str], frozenset[str]]:
    """Every testcase key in a JUnit XML report, with EVERY state it showed.

    A call failure plus a teardown error is written as two testcase elements for one
    node; the key then holds ``{"failed", "error"}``. No state is ever dropped, because
    dropping one is exactly how a teardown error hides behind an expected failure.

    Args:
        report: Path to a pytest ``--junitxml`` report.

    Returns:
        ``(classname, name)`` -> frozenset of ``failed`` / ``error`` / ``xfailed`` /
        ``skipped`` / ``passed``.
    """
    states: dict[tuple[str, str], set[str]] = {}
    for testcase in ET.parse(report).getroot().iter("testcase"):
        key = (testcase.get("classname", ""), testcase.get("name", ""))
        states.setdefault(key, set()).add(_testcase_state(testcase))
    return {key: frozenset(found) for key, found in states.items()}


def _shown(states: Iterable[str]) -> str:
    return "[" + ", ".join(sorted(states)) + "]"


def expected_outcome_problems(
    outcomes: Outcomes, expected_node_ids: Iterable[str]
) -> list[str]:
    """The ONE comparison: every way an outcome map differs from "exactly these fail".

    Args:
        outcomes: A JUnit outcome map (see ``junit_outcomes``).
        expected_node_ids: The node ids that must fail, and fail cleanly.

    Returns:
        One message per problem, each starting with its kind (``DISCLOSURE ERASED``,
        ``TRIPWIRE NOT A FAILURE``, ``TRIPWIRE ALSO ERRORED``, ``REGRESSION``,
        ``QUARANTINE``). An empty list is the passing case. A tripwire that only
        ``crashed`` is NOT A FAILURE: it never reached the check its disclosure is.
    """
    problems: list[str] = []
    expected_keys: set[tuple[str, str]] = set()
    for node_id in expected_node_ids:
        key = junit_key(node_id)
        expected_keys.add(key)
        states = outcomes.get(key)
        if states is None:
            problems.append(f"DISCLOSURE ERASED: {node_id} did not run")
        elif states == {"passed"}:
            problems.append(f"DISCLOSURE ERASED: {node_id} passed")
        elif "failed" not in states:
            problems.append(f"TRIPWIRE NOT A FAILURE: {node_id} {_shown(states)}")
        elif "error" in states:
            problems.append(f"TRIPWIRE ALSO ERRORED: {node_id} {_shown(states)}")

    for (classname, name), states in sorted(outcomes.items()):
        if (classname, name) not in expected_keys and states & _RED_STATES:
            problems.append(f"REGRESSION: {classname}::{name} {_shown(states)}")
        if "xfailed" in states:
            problems.append(f"QUARANTINE: {classname}::{name} xfailed")
    return problems


def carries_closing_sweep(text: str) -> bool:
    """Whether a run's output carries the COLD-05 session-end sweep section.

    The terminal summary writes the header unindented at the start of a line. The same
    words quoted inside an indented ``E   `` traceback line -- a failing guard test
    echoing its own child's output -- are not this session's sweep.
    """
    return any(line.startswith(CLOSING_SWEEP_HEADER) for line in text.splitlines())


def session_completed(text: str) -> bool:
    """Whether a run's output ends on pytest's closing stats line, with no INTERNALERROR.

    That line is written only after the conftest's session-end sweep returned, so it is
    the positive proof that the sweep ran to the end and compared the stores.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    return (
        bool(lines)
        and _SUMMARY_LINE_RE.match(lines[-1]) is not None
        and not any(line.startswith("INTERNALERROR") for line in lines)
    )


def _head_problems(head: Path, recorded: Mapping[str, object] | None) -> list[str]:
    """Whether ``head.txt`` exists and names the head the recording slot holds.

    The comparison is with the slot, never with live HEAD: HEAD moves with every
    commit made after the run, and the run is a record of one commit.
    """
    if not head.is_file():
        return [f"INCOMPLETE: {head.name} missing"]
    if recorded is None:
        return []
    found = head.read_text(encoding="utf-8", errors="replace").strip()
    if found != recorded.get("head"):
        return [
            f"INCOMPLETE: {head.name} names {found!r}, not the recorded head "
            f"{recorded.get('head')!r}"
        ]
    return []


def _exit_problems(exits: Path) -> list[str]:
    """Whether ``exits.txt`` shows every tier ending with an exit code a run may have."""
    if not exits.is_file():
        return [f"INCOMPLETE: {exits.name} missing"]
    recorded: dict[str, str] = {}
    for line in exits.read_text(encoding="utf-8", errors="replace").splitlines():
        tier, _, code = line.strip().partition(" ")
        if tier:
            recorded[tier] = code.strip()
    problems: list[str] = []
    for tier, allowed in _ALLOWED_TIER_EXITS.items():
        code = recorded.get(tier)
        if code is None or not code.isdigit() or int(code) not in allowed:
            problems.append(
                f"INCOMPLETE: {tier} exit code {code!r} in {exits.name}, not one of "
                f"{sorted(allowed)}"
            )
    return problems


def _listing_problems(
    tier: str,
    listing: str,
    outcomes: Outcomes | None,
    recorded_collected: int | None = None,
) -> list[str]:
    """Whether a tier's ``--collect-only -q`` listing proves its report is complete.

    Args:
        tier: The tier name, for the messages; a ``FULL_RUN_TIERS`` name also holds
            every listed id to that tier's root.
        listing: The text of ``<tier>.collect.txt``.
        outcomes: The tier report's outcome map; None when the report is missing (that
            is already its own problem, so collected ids are not checked against it).
        recorded_collected: The collected count a recorded slot holds for this tier;
            None checks no size.

    Returns:
        ``INCOMPLETE`` problems; empty when every collected node id has an outcome.
    """
    lines = listing.splitlines()
    counted = [
        match
        for line in lines
        if (match := _COLLECTED_LINE_RE.match(line)) and "error" not in line
    ]
    if not counted:
        return [
            f"INCOMPLETE: {tier} collect listing has no clean 'N tests collected' line"
        ]

    listed = [line for line in lines if line.startswith("tests/") and "::" in line]
    collected = int(counted[-1].group(1))
    if len(listed) != collected:
        return [
            f"INCOMPLETE: {tier} collect listing names {len(listed)} node ids but "
            f"reports {collected} collected (a listing that names no node id cannot "
            "prove the report complete)"
        ]
    root = _FULL_RUN_TIER_ID_RE.get(tier)
    outside = [node_id for node_id in listed if root and not root.match(node_id)]
    if outside:
        return [
            f"INCOMPLETE: {tier} collect listing names {len(outside)} node ids outside "
            f"its tier root: {', '.join(outside[:5])}"
        ]
    if recorded_collected is not None and collected != recorded_collected:
        return [
            f"INCOMPLETE: {tier} collect listing reports {collected} collected, not the "
            f"{recorded_collected} the recorded slot holds"
        ]
    if outcomes is None:
        return []

    unreported = [node_id for node_id in listed if junit_key(node_id) not in outcomes]
    if not unreported:
        return []
    return [
        f"INCOMPLETE: {tier} report has no outcome for {len(unreported)} collected node "
        f"ids: {', '.join(unreported[:5])}"
    ]


def full_run_problems(
    run_dir: Path,
    expected_node_ids: Iterable[str],
    recorded: Mapping[str, object] | None = None,
) -> list[str]:
    """Judge the recorded full-suite run written into *run_dir*; it runs nothing itself.

    For each tier in ``FULL_RUN_TIERS`` it reads ``<tier>.xml`` (the JUnit report),
    ``<tier>.log`` (the run's output) and ``<tier>.collect.txt`` (the tier's
    ``--collect-only -q`` listing taken at the same HEAD in the same window), then judges
    the union of the three outcome maps with ``expected_outcome_problems``. Every problem
    is returned; the judge never stops at the first.

    Args:
        run_dir: The run directory.
        expected_node_ids: The node ids that must fail, and fail cleanly.
        recorded: The slot that records this run (``head`` and per-tier
            ``collected``); ``head.txt`` and each listing's size must equal it. None
            checks only that ``head.txt`` exists.

    Returns:
        ``INCOMPLETE`` (a missing file, an empty report, a collected test with no
        outcome, or a tier exit code in ``exits.txt`` no complete run has), ``BOUNDARY``
        (a tier log carrying the session-end production store sweep, or one that does
        not end on pytest's closing summary line, so the sweep never completed) and
        every kind ``expected_outcome_problems`` names. Empty is the pass.
    """
    problems = _head_problems(run_dir / FULL_RUN_HEAD_FILE, recorded)
    recorded_collected = recorded.get("collected", {}) if recorded else {}
    assert isinstance(recorded_collected, Mapping), recorded
    union: dict[tuple[str, str], frozenset[str]] = {}
    for tier in FULL_RUN_TIERS:
        report, log, listing = (
            run_dir / f"{tier}{suffix}" for suffix in _FULL_RUN_TIER_SUFFIXES
        )
        problems.extend(
            f"INCOMPLETE: {tier} {path.name} missing"
            for path in (report, log, listing)
            if not path.is_file()
        )

        outcomes = junit_outcomes(report) if report.is_file() else None
        if outcomes is not None and not outcomes:
            problems.append(f"INCOMPLETE: {tier} report holds no testcase")
        if listing.is_file():
            listing_text = listing.read_text(encoding="utf-8", errors="replace")
            problems.extend(
                _listing_problems(
                    tier, listing_text, outcomes, recorded_collected.get(tier)
                )
            )
        log_text = (
            log.read_text(encoding="utf-8", errors="replace") if log.is_file() else None
        )
        if log_text is not None and carries_closing_sweep(log_text):
            problems.append(
                f"BOUNDARY: {tier} log carries the session-end production store sweep "
                "-- a production store moved during the run"
            )
        if log_text is not None and not session_completed(log_text):
            problems.append(
                f"BOUNDARY: {tier} log does not end on pytest's closing summary line -- "
                "the session-end sweep did not complete, so the stores were not compared"
            )

        for key, states in (outcomes or {}).items():
            union[key] = union.get(key, frozenset()) | states

    problems.extend(_exit_problems(run_dir / FULL_RUN_EXITS_FILE))
    problems.extend(expected_outcome_problems(union, expected_node_ids))
    return problems


def _child_env() -> dict[str, str]:
    """The environment for a child pytest session, carrying the recursion guard."""
    env = dict(os.environ)
    env[CHILD_ENV] = "1"
    env.pop(FULL_TIER_ENV, None)
    env.pop(FULL_RUN_DIR_ENV, None)
    return env


def _run_pytest(arguments: list[str], timeout: int, report: Path | None = None) -> str:
    """Run pytest in a subprocess and return its combined output.

    ``-o addopts=`` neutralises whatever ``addopts`` this repository sets, so the
    child's output is always the short ``-q`` form the collectibility check reads. The
    same override is used by ``tests/unit/test_phase32_state_guard.py``.

    Args:
        arguments: Arguments after ``-m pytest``.
        timeout: Seconds before the child is killed.
        report: Where to write a JUnit XML report. None (the ``--collect-only`` caller)
            writes no report.

    Returns:
        stdout concatenated with stderr.
    """
    junit = [f"--junitxml={report}"] if report is not None else []
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-o", "addopts=", "-q", *junit, *arguments],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=_child_env(),
        check=False,
    )
    return result.stdout + result.stderr


def _judged_outcomes(
    report: Path, output: str
) -> dict[tuple[str, str], frozenset[str]]:
    """The outcome map of a child run, failing loudly when the child wrote no report."""
    assert report.is_file(), (
        f"the child pytest wrote no JUnit report at {report}; it did not run to the "
        f"end, so nothing can be judged.\n\nFull output:\n{output}"
    )
    return junit_outcomes(report)


def _require_runnable_tripwires() -> None:
    """Step aside by NAME when the tripwires cannot be run on this checkout.

    Raises:
        Skipped: when this is a child session, or the production data tree is absent.
    """
    if os.environ.get(CHILD_ENV):
        pytest.skip(CHILD_SESSION_SKIP)
    if not PRODUCTION_DATA_TREE.is_dir():
        pytest.skip(NO_PRODUCTION_DATA_SKIP)


# ---------------------------------------------------------------------------
# Anti-vacuity.
# ---------------------------------------------------------------------------


def test_the_gate_reads_a_non_empty_tripwire_set() -> None:
    """Every assertion below is over the constant; an empty constant would pass them all.

    Five is not an arbitrary number: it is the partition Phase 32 closed on and Phase 33
    measured again. A constant that has shrunk is itself the event this gate exists to
    report, so the count is pinned.
    """
    assert phase33_state.DELIBERATE_TRIPWIRE_NODE_IDS, (
        "DELIBERATE_TRIPWIRE_NODE_IDS is empty -- every check in this module would then "
        "run over nothing and pass while proving nothing."
    )
    assert len(phase33_state.DELIBERATE_TRIPWIRE_NODE_IDS) == 5, (
        f"DELIBERATE_TRIPWIRE_NODE_IDS holds "
        f"{len(phase33_state.DELIBERATE_TRIPWIRE_NODE_IDS)} ids, not the five "
        "owner-accepted disclosures Phase 32 closed on and Phase 33 re-measured. A "
        "tripwire dropped from the constant is a disclosure erased from the record."
    )
    assert len(set(phase33_state.DELIBERATE_TRIPWIRE_NODE_IDS)) == 5


def test_every_tripwire_resolves_to_a_known_tier() -> None:
    """The tier split is how this suite is run, so every tripwire must land in one."""
    tiers = {tier_of(node_id) for node_id in phase33_state.DELIBERATE_TRIPWIRE_NODE_IDS}
    assert tiers, phase33_state.DELIBERATE_TRIPWIRE_NODE_IDS
    assert tiers <= set(TIER_ROOTS)


# ---------------------------------------------------------------------------
# The comparison and the reader, proven on planted outcomes before they judge anything.
# ---------------------------------------------------------------------------

# Planted ids sit under a FAKE unit path, so the self-scan for node-id literals stays clean.
_PLANTED_MODULE = "tests/unit/test_planted_gate.py"
_PLANTED_TRIPWIRES: tuple[str, ...] = tuple(
    f"{_PLANTED_MODULE}::TestPlanted::test_tripwire_{number}" for number in range(1, 6)
)
_PLANTED_BYSTANDER = f"{_PLANTED_MODULE}::test_bystander"
# Concatenated, so the self-scan for node-id literals stays clean.
_PLANTED_INTEGRATION_MODULE = "tests/integration" + "/test_planted_gate.py"
_PLANTED_HEAD = "0" * 40


def _kinds(problems: list[str]) -> list[str]:
    """The kind each problem message starts with, sorted."""
    return sorted(problem.split(":", 1)[0] for problem in problems)


def test_the_comparison_names_each_way_a_set_can_be_wrong() -> None:
    """SPEC AC 14: each way an outcome map can be wrong gets its own message.

    Planted state sets, not a run: the comparison is proven before it judges anything.
    """
    clean = {
        junit_key(node_id): frozenset({"failed"}) for node_id in _PLANTED_TRIPWIRES
    }
    first = junit_key(_PLANTED_TRIPWIRES[0])
    bystander = junit_key(_PLANTED_BYSTANDER)

    def judged(changes: dict[tuple[str, str], frozenset[str]]) -> list[str]:
        return expected_outcome_problems({**clean, **changes}, _PLANTED_TRIPWIRES)

    assert expected_outcome_problems(clean, _PLANTED_TRIPWIRES) == []
    # A sixth node that passed or skipped is not a problem.
    assert judged({bystander: frozenset({"passed"})}) == []
    assert judged({bystander: frozenset({"skipped"})}) == []

    # Four failing: one tripwire went green, or never ran.
    assert _kinds(judged({first: frozenset({"passed"})})) == ["DISCLOSURE ERASED"]
    without_first = {key: states for key, states in clean.items() if key != first}
    absent = expected_outcome_problems(without_first, _PLANTED_TRIPWIRES)
    assert _kinds(absent) == ["DISCLOSURE ERASED"], absent
    assert "did not run" in absent[0]

    # A sixth red, whether it failed, crashed, errored, or both: exactly one REGRESSION.
    for states in ({"failed"}, {"crashed"}, {"error"}, {"failed", "error"}):
        problems = judged({bystander: frozenset(states)})
        assert _kinds(problems) == ["REGRESSION"], problems
        assert "test_bystander" in problems[0]

    # Only an assertion failure is the accepted red; a crash never reached the check.
    for states in ({"crashed"}, {"error"}, {"skipped"}):
        problems = judged({first: frozenset(states)})
        assert _kinds(problems) == ["TRIPWIRE NOT A FAILURE"], problems
    xfailed_tripwire = judged({first: frozenset({"xfailed"})})
    assert _kinds(xfailed_tripwire) == ["QUARANTINE", "TRIPWIRE NOT A FAILURE"]

    # A tripwire that failed and also errored never reads as accepted.
    also_errored = judged({first: frozenset({"failed", "error"})})
    assert _kinds(also_errored) == ["TRIPWIRE ALSO ERRORED"], also_errored

    # Zero quarantines: an xfail anywhere fails the gate.
    assert _kinds(judged({bystander: frozenset({"xfailed"})})) == ["QUARANTINE"]


_PLANTED_REPORT_MODULE = """
import pytest


@pytest.fixture
def broken_setup():
    raise RuntimeError("planted setup error")


@pytest.fixture
def broken_teardown():
    yield
    raise RuntimeError("planted teardown error")


def test_fails():
    assert False


def test_fails_with_a_message():
    assert False, "planted check"


def test_crashes():
    {}["k"]


def test_errors_at_setup(broken_setup):
    pass


def test_skips():
    pytest.skip("planted skip")


@pytest.mark.xfail(reason="planted quarantine", strict=True)
def test_xfails():
    assert False


def test_passes():
    pass


def test_fails_and_errors_at_teardown(broken_teardown):
    assert False
"""

_PLANTED_REPORT_STATES: dict[str, frozenset[str]] = {
    "test_fails": frozenset({"failed"}),
    "test_fails_with_a_message": frozenset({"failed"}),
    "test_crashes": frozenset({"crashed"}),
    "test_errors_at_setup": frozenset({"error"}),
    "test_skips": frozenset({"skipped"}),
    "test_xfails": frozenset({"xfailed"}),
    "test_passes": frozenset({"passed"}),
    "test_fails_and_errors_at_teardown": frozenset({"failed", "error"}),
}


def test_the_reader_classifies_a_real_pytest_report(tmp_path: Path) -> None:
    """A report pytest itself wrote, read back state by state.

    The fails-and-errors node is two testcase elements in the report; the reader keeps
    both states, and the comparison judges it a problem whether it is a tripwire or not.
    """
    if os.environ.get(CHILD_ENV):
        pytest.skip(CHILD_SESSION_SKIP)

    module = tmp_path / "test_planted_outcomes.py"
    module.write_text(_PLANTED_REPORT_MODULE, encoding="utf-8")
    report = tmp_path / "planted.xml"
    output = _run_pytest(
        [str(module), f"--rootdir={tmp_path}", "-p", "no:cacheprovider"],
        timeout=300,
        report=report,
    )

    outcomes = _judged_outcomes(report, output)
    expected = {
        junit_key(f"{module.name}::{name}"): states
        for name, states in _PLANTED_REPORT_STATES.items()
    }
    assert outcomes == expected, f"{outcomes}\n\nFull output:\n{output}"

    both = f"{module.name}::test_fails_and_errors_at_teardown"

    def problems_naming(problems: list[str]) -> list[str]:
        return [problem for problem in problems if "test_fails_and_errors" in problem]

    as_tripwire = problems_naming(expected_outcome_problems(outcomes, [both]))
    assert _kinds(as_tripwire) == ["TRIPWIRE ALSO ERRORED"], as_tripwire

    as_bystander = expected_outcome_problems(outcomes, [])
    assert _kinds(problems_naming(as_bystander)) == ["REGRESSION"], as_bystander
    assert _kinds(as_bystander) == ["QUARANTINE"] + ["REGRESSION"] * 5, as_bystander

    # The crash, judged as a tripwire, is never the accepted red.
    crash = f"{module.name}::test_crashes"
    crash_key = junit_key(crash)
    as_crashed_tripwire = expected_outcome_problems(
        {crash_key: outcomes[crash_key]}, [crash]
    )
    assert _kinds(as_crashed_tripwire) == ["TRIPWIRE NOT A FAILURE"], (
        as_crashed_tripwire
    )


# ---------------------------------------------------------------------------
# Half one: resolvability. A renamed tripwire is a hard failure.
# ---------------------------------------------------------------------------


def test_every_tripwire_node_id_is_still_collectible() -> None:
    """Collection catches deletion, renaming and class/function moves.

    It does NOT catch a tripwire whose body was edited to pass while its name stays put.
    That failure mode needs the node to actually RUN, which the verdict half below does.
    """
    if os.environ.get(CHILD_ENV):
        pytest.skip(CHILD_SESSION_SKIP)

    node_ids = list(phase33_state.DELIBERATE_TRIPWIRE_NODE_IDS)
    output = _run_pytest(["--collect-only", *node_ids], timeout=300)

    assert "ERROR" not in output, (
        "pytest --collect-only reported a collection ERROR against one or more "
        "deliberate tripwire node ids. A renamed tripwire must be RE-RECORDED in "
        "tests/phase33_state.py under the append protocol -- appended as a new slot "
        f"naming the plan that renamed it -- and NEVER dropped.\n\n{output}"
    )
    collected = "\n".join(normalised_node_id(line) for line in output.splitlines())
    for node_id in node_ids:
        assert node_id in collected, (
            f"deliberate tripwire node id no longer collects:\n  {node_id}\n\n"
            "It was deleted, renamed, or moved to another module or class. A renamed "
            "tripwire must be re-recorded in tests/phase33_state.py under the append "
            "protocol, never dropped: dropping it removes an owner-accepted disclosure "
            f"from the record.\n\nFull collect-only output:\n{output}"
        )


# ---------------------------------------------------------------------------
# Half two: the verdict. Exactly the tripwires fail, cleanly, and nothing else is red.
# ---------------------------------------------------------------------------


def test_the_measured_failing_set_equals_the_tripwire_constant(tmp_path: Path) -> None:
    """Every outcome problem, over a run of exactly the tripwires, each with its meaning.

    A tripwire that passed or did not run is a DISCLOSURE ERASED; one that errored,
    skipped or xfailed is NOT A FAILURE; one that failed and also errored is its own
    problem; any other red reached through the tripwires' own modules is a REGRESSION.
    The child's output must also carry no session-end production store sweep.
    """
    _require_runnable_tripwires()

    expected = sorted(phase33_state.DELIBERATE_TRIPWIRE_NODE_IDS)
    report = tmp_path / "tripwires.xml"
    output = _run_pytest(expected, timeout=900, report=report)
    problems = expected_outcome_problems(_judged_outcomes(report, output), expected)

    assert problems == [], (
        "the tripwire run is not exactly the five deliberate tripwires failing:\n"
        + "\n".join(f"  - {problem}" for problem in problems)
        + "\n\nA DISCLOSURE ERASED tripwire encodes an owner-accepted fact from Phase 30 "
        "or Phase 31; if a rebuild caused it, investigate why the attribution changed "
        f"and record the finding -- never accept the green as progress.\n\nFull "
        f"report:\n{output}"
    )
    assert not carries_closing_sweep(output), (
        "the tripwire run's output carries the session-end production store sweep: a "
        f"production store moved during the run.\n\nFull report:\n{output}"
    )
    assert session_completed(output), (
        "the tripwire run's output does not end on pytest's closing summary line: the "
        "session-end sweep did not complete, so the production stores were never "
        f"compared.\n\nFull report:\n{output}"
    )


def test_the_whole_tier_verdict_when_opted_in(tmp_path: Path) -> None:
    """The stronger form: no red ANYWHERE in a tripwire's tier beyond the constant.

    Opt-in, because this module runs inside the tier it measures. See the module
    docstring: the recursion guard is what makes the child safe, and the phase close's
    recorded three-tier measurement is what owns the tier-wide reconciliation properly.
    """
    _require_runnable_tripwires()
    if not os.environ.get(FULL_TIER_ENV):
        pytest.skip(FULL_TIER_PENDING_SKIP)

    expected_by_tier: dict[str, list[str]] = {}
    for node_id in phase33_state.DELIBERATE_TRIPWIRE_NODE_IDS:
        expected_by_tier.setdefault(tier_of(node_id), []).append(node_id)

    for tier, expected in sorted(expected_by_tier.items()):
        report = tmp_path / f"{tier.rsplit('/', 1)[-1]}.xml"
        output = _run_pytest([tier], timeout=3600, report=report)
        problems = expected_outcome_problems(_judged_outcomes(report, output), expected)
        assert problems == [], (
            f"{tier} is not exactly its deliberate tripwires failing:\n"
            + "\n".join(f"  - {problem}" for problem in problems)
            + f"\n\nFull report:\n{output}"
        )
        assert not carries_closing_sweep(output), (
            f"{tier}'s output carries the session-end production store sweep: a "
            f"production store moved during the run.\n\nFull report:\n{output}"
        )
        assert session_completed(output), (
            f"{tier}'s output does not end on pytest's closing summary line: the "
            f"session-end sweep did not complete.\n\nFull report:\n{output}"
        )


# ---------------------------------------------------------------------------
# The recorded full-suite run of the phase close: three tier reports, judged.
# ---------------------------------------------------------------------------


def recorded_crash_problems(run_dir: Path) -> list[str]:
    """The one problem a correction slot accepts for the run it names, else none.

    ``tests.phase34_state.P38_TRIPWIRE4_CRASH_CORRECTION`` records that run 2's
    tripwire 4 crashed instead of failing on its check. It applies only to the run
    directory whose ``head.txt`` is the head the slot records, never to live HEAD.

    Args:
        run_dir: The recorded run directory.

    Returns:
        ``["TRIPWIRE NOT A FAILURE: <tripwire> [crashed]"]`` for that run; empty otherwise.
    """
    correction = phase34_state.P38_TRIPWIRE4_CRASH_CORRECTION
    head = run_dir / "head.txt"
    recorded_head = head.read_text(encoding="utf-8").strip() if head.is_file() else None
    if recorded_head != correction["run_head"]:
        return []
    index = correction["tripwire_index"]
    assert isinstance(index, int), correction
    node_id = phase33_state.DELIBERATE_TRIPWIRE_NODE_IDS[index]
    return [f"TRIPWIRE NOT A FAILURE: {node_id} [{correction['run_state']}]"]


def test_the_recorded_full_run_report_when_given() -> None:
    """SPEC AC 15: the recorded run is exactly the five tripwires failing, and complete.

    Opt-in: the run itself is an owner-approved step of the phase close (D-06), and this
    node only reads the directory that run wrote. An env var that names no directory is
    a failure, never a skip, so a mistyped path cannot read as "not judged".

    The run is judged against the slot that records it,
    ``tests.phase34_state.P38_FULL_SUITE_BASELINE``: its ``head.txt`` and each tier's
    collected count must equal the slot's. A later recorded run is judged once its own
    slot is appended and named here.
    """
    named = os.environ.get(FULL_RUN_DIR_ENV)
    if not named:
        pytest.skip(FULL_RUN_PENDING_SKIP)
    run_dir = Path(named)
    assert run_dir.is_dir(), f"{FULL_RUN_DIR_ENV}={named!r} names no directory"

    problems = full_run_problems(
        run_dir,
        phase33_state.DELIBERATE_TRIPWIRE_NODE_IDS,
        recorded=phase34_state.P38_FULL_SUITE_BASELINE,
    )
    accepted = recorded_crash_problems(run_dir)
    missing = [problem for problem in accepted if problem not in problems]
    assert not missing, (
        f"the correction slot records a crash for the run in {run_dir} that the judge no "
        f"longer finds: {missing}. The slot describes another run, or this one changed."
    )
    unaccepted = [problem for problem in problems if problem not in accepted]
    assert unaccepted == [], (
        f"the recorded full run in {run_dir} is not exactly the five deliberate "
        "tripwires failing, complete and inside the production store boundary:\n"
        + "\n".join(f"  - {problem}" for problem in unaccepted)
    )


def _write_planted_tier(
    run_dir: Path,
    tier: str,
    outcomes: dict[str, str],
    *,
    listed: list[str] | None = None,
    log_text: str | None = None,
) -> None:
    """Write one tier's report, log and collect listing the way the run step does.

    Args:
        run_dir: The planted run directory.
        tier: One of ``FULL_RUN_TIERS``.
        outcomes: node id -> ``failed``, ``crashed`` or ``passed``; each becomes one
            testcase, a failure written with the message pytest writes for that kind.
        listed: The collect listing's node ids; defaults to the report's own.
        log_text: The tier log; defaults to a plain pytest summary line.
    """
    suite = ET.Element("testsuite", name="pytest")
    for node_id, outcome in outcomes.items():
        classname, name = junit_key(node_id)
        testcase = ET.SubElement(suite, "testcase", classname=classname, name=name)
        if outcome == "failed":
            ET.SubElement(testcase, "failure", message="AssertionError: planted check")
        elif outcome == "crashed":
            ET.SubElement(testcase, "failure", message="KeyError: 'planted'")
    root = ET.Element("testsuites")
    root.append(suite)
    ET.ElementTree(root).write(run_dir / f"{tier}.xml", encoding="utf-8")

    listed = list(outcomes) if listed is None else listed
    plural = "" if len(listed) == 1 else "s"
    (run_dir / f"{tier}.collect.txt").write_text(
        "\n".join(listed) + f"\n\n{len(listed)} test{plural} collected in 0.01s\n",
        encoding="utf-8",
    )
    failed = sum(outcome != "passed" for outcome in outcomes.values())
    passed = len(outcomes) - failed
    summary = f"{failed} failed, {passed} passed" if failed else f"{passed} passed"
    (run_dir / f"{tier}.log").write_text(
        log_text if log_text is not None else f"{summary} in 0.01s\n", encoding="utf-8"
    )


def _planted_run(run_dir: Path) -> Path:
    """A CLEAN planted run: the five tripwires fail in integration, one pass per tier."""
    run_dir.mkdir()
    tripwires = phase33_state.DELIBERATE_TRIPWIRE_NODE_IDS
    _write_planted_tier(
        run_dir,
        "integration",
        {
            **dict.fromkeys(tripwires, "failed"),
            f"{_PLANTED_INTEGRATION_MODULE}::test_integration_bystander": "passed",
        },
    )
    _write_planted_tier(run_dir, "unit", {_PLANTED_BYSTANDER: "passed"})
    _write_planted_tier(
        run_dir, "api", {"tests/api/test_planted_gate.py::test_api_bystander": "passed"}
    )
    (run_dir / FULL_RUN_EXITS_FILE).write_text(
        "unit 0\nintegration 1\napi 0\n", encoding="utf-8"
    )
    (run_dir / FULL_RUN_HEAD_FILE).write_text(_PLANTED_HEAD + "\n", encoding="utf-8")
    return run_dir


def test_the_full_run_judge_refuses_a_bad_report(tmp_path: Path) -> None:
    """The judge passes a clean planted run and names each way a recorded run is bad.

    Every mutation is made on a fresh planted directory, so each proves one kind alone.
    """
    tripwires = phase33_state.DELIBERATE_TRIPWIRE_NODE_IDS
    clean = _planted_run(tmp_path / "clean")
    assert full_run_problems(clean, tripwires) == []

    # A tier killed before it wrote anything: all three of its files are absent.
    no_api = _planted_run(tmp_path / "no_api")
    for suffix in (".xml", ".log", ".collect.txt"):
        (no_api / f"api{suffix}").unlink()
    missing = full_run_problems(no_api, tripwires)
    assert _kinds(missing) == ["INCOMPLETE"] * 3, missing
    assert all(" api " in problem for problem in missing), missing

    # A truncated report: the listing names a node id the report has no outcome for.
    truncated = _planted_run(tmp_path / "truncated")
    truncated_id = f"{_PLANTED_MODULE}::test_truncated_away"
    _write_planted_tier(
        truncated,
        "unit",
        {_PLANTED_BYSTANDER: "passed"},
        listed=[_PLANTED_BYSTANDER, truncated_id],
    )
    cut_short = full_run_problems(truncated, tripwires)
    assert _kinds(cut_short) == ["INCOMPLETE"], cut_short
    assert "test_truncated_away" in cut_short[0]

    # A listing with no clean "N tests collected" line, and one that names no node id
    # (the tree form a verbose --collect-only prints) cannot prove completeness.
    for listing in (
        f"{_PLANTED_BYSTANDER}\n\n1 test collected, 1 error in 0.01s\n",
        "<Module test_planted_gate.py>\n  <Function test_bystander>\n\n"
        "1 test collected in 0.01s\n",
    ):
        unproven = _planted_run(tmp_path / f"unproven_{len(listing)}")
        (unproven / "unit.collect.txt").write_text(listing, encoding="utf-8")
        not_proven = full_run_problems(unproven, tripwires)
        assert _kinds(not_proven) == ["INCOMPLETE"], not_proven

    # A tier log carrying the session-end sweep: a production store moved.
    swept = _planted_run(tmp_path / "swept")
    sweep_log = (
        f"{CLOSING_SWEEP_HEADER} -- a production store moved\n1 passed in 0.01s\n"
    )
    (swept / "unit.log").write_text(sweep_log, encoding="utf-8")
    boundary = full_run_problems(swept, tripwires)
    assert _kinds(boundary) == ["BOUNDARY"], boundary
    # The same words quoted inside a traceback line are not this run's sweep.
    quoted = _planted_run(tmp_path / "quoted")
    quoted_log = f"E   AssertionError: {CLOSING_SWEEP_HEADER} seen\n1 passed in 0.01s\n"
    (quoted / "unit.log").write_text(quoted_log, encoding="utf-8")
    assert full_run_problems(quoted, tripwires) == []

    # A session-end sweep that raised: the report is complete and clean, but the log
    # ends on the traceback, not on pytest's closing summary line.
    crashed_sweep = _planted_run(tmp_path / "crashed_sweep")
    (crashed_sweep / "unit.log").write_text(
        "Traceback (most recent call last):\n"
        '  File "tests/conftest.py", line 1, in pytest_sessionfinish\n'
        "RuntimeError: planted locked store\n",
        encoding="utf-8",
    )
    not_compared = full_run_problems(crashed_sweep, tripwires)
    assert _kinds(not_compared) == ["BOUNDARY"], not_compared
    assert "did not complete" in not_compared[0]
    internal = _planted_run(tmp_path / "internal")
    (internal / "api.log").write_text(
        "INTERNALERROR> planted\n1 passed in 0.01s\n", encoding="utf-8"
    )
    assert _kinds(full_run_problems(internal, tripwires)) == ["BOUNDARY"]

    # exits.txt is required, and only the tripwire tier may exit 1.
    no_exits = _planted_run(tmp_path / "no_exits")
    (no_exits / FULL_RUN_EXITS_FILE).unlink()
    assert _kinds(full_run_problems(no_exits, tripwires)) == ["INCOMPLETE"]
    bad_exit = _planted_run(tmp_path / "bad_exit")
    (bad_exit / FULL_RUN_EXITS_FILE).write_text(
        "unit 3\nintegration 1\napi 0\n", encoding="utf-8"
    )
    bad = full_run_problems(bad_exit, tripwires)
    assert _kinds(bad) == ["INCOMPLETE"] and " unit " in bad[0], bad

    # A run over part of the suite: a tier listing a node outside its own root, a run
    # without head.txt, and one whose head or size is not the recorded run's.
    misplaced = _planted_run(tmp_path / "misplaced")
    _write_planted_tier(
        misplaced,
        "integration",
        {
            **dict.fromkeys(tripwires, "failed"),
            f"{_PLANTED_MODULE}::test_integration_bystander": "passed",
        },
    )
    outside = full_run_problems(misplaced, tripwires)
    assert _kinds(outside) == ["INCOMPLETE"] and "outside" in outside[0], outside
    headless = _planted_run(tmp_path / "headless")
    (headless / FULL_RUN_HEAD_FILE).unlink()
    assert _kinds(full_run_problems(headless, tripwires)) == ["INCOMPLETE"]
    collected = {"unit": 1, "integration": len(tripwires) + 1, "api": 1}
    recorded = {"head": _PLANTED_HEAD, "collected": collected}
    assert full_run_problems(clean, tripwires, recorded=recorded) == []
    other_head = full_run_problems(clean, tripwires, recorded={**recorded, "head": "1"})
    assert _kinds(other_head) == ["INCOMPLETE"], other_head
    smaller = {**recorded, "collected": {**collected, "unit": 2}}
    subset = full_run_problems(clean, tripwires, recorded=smaller)
    assert _kinds(subset) == ["INCOMPLETE"] and " unit " in subset[0], subset

    # A sixth failing testcase.
    sixth = _planted_run(tmp_path / "sixth")
    _write_planted_tier(sixth, "unit", {_PLANTED_BYSTANDER: "failed"})
    regression = full_run_problems(sixth, tripwires)
    assert _kinds(regression) == ["REGRESSION"], regression

    # A tripwire that crashed before its check is not the accepted red.
    crashed = _planted_run(tmp_path / "crashed")
    _write_planted_tier(
        crashed,
        "integration",
        {
            **dict.fromkeys(tripwires, "failed"),
            tripwires[0]: "crashed",
            f"{_PLANTED_INTEGRATION_MODULE}::test_integration_bystander": "passed",
        },
    )
    not_a_failure = full_run_problems(crashed, tripwires)
    assert _kinds(not_a_failure) == ["TRIPWIRE NOT A FAILURE"], not_a_failure


# Parametrize ids pytest writes with an escape: a newline (`\n`), a non-ASCII character
# (`\xb2`) and a literal backslash (`\\`). A raw string, so the planted module's source
# holds the escapes exactly as written here.
_PLANTED_ESCAPED_IDS_MODULE = r"""
import pytest


@pytest.mark.parametrize("text", ["import x\n", "\xb2", "a\\b", "plain"])
def test_escaped(text):
    pass


def test_plain():
    pass
"""


def test_the_judge_keys_a_listing_and_a_report_pytest_wrote(tmp_path: Path) -> None:
    """Every collected id pytest listed has an outcome, escaped parametrize ids included.

    ``test_the_full_run_judge_refuses_a_bad_report`` builds its report WITH ``junit_key``,
    so its report and listing always agree whatever that function does, and it can never
    catch a key bug. Here pytest itself writes both the ``--collect-only -q`` listing and
    the JUnit report, the way the recorded run does. Phase 38's discovery run found the
    bug this pins: the path-separator normalisation ran over the whole node id, turned
    the backslash of every escaped parametrize id into a slash, and reported 97 tests
    that had passed as having no outcome.
    """
    if os.environ.get(CHILD_ENV):
        pytest.skip(CHILD_SESSION_SKIP)

    tier_root = tmp_path / "tests" / "unit"
    tier_root.mkdir(parents=True)
    module = tier_root / "test_planted_escapes.py"
    module.write_text(_PLANTED_ESCAPED_IDS_MODULE, encoding="utf-8")
    arguments = [str(module), f"--rootdir={tmp_path}", "-p", "no:cacheprovider"]
    listing = _run_pytest(["--collect-only", *arguments], timeout=300)
    report = tmp_path / "unit.xml"
    output = _run_pytest(arguments, timeout=300, report=report)
    outcomes = _judged_outcomes(report, output)

    listed = [
        line
        for line in listing.splitlines()
        if line.startswith("tests/") and "::" in line
    ]
    escaped = [node_id for node_id in listed if "\\" in node_id]
    assert len(listed) == 5 and len(escaped) == 3, f"{listed}\n\n{listing}"
    assert _listing_problems("unit", listing, outcomes) == [], listing

    # A collected test whose testcase is missing is still caught, escaped id or not.
    for missing in (*escaped, f"{module.relative_to(tmp_path).as_posix()}::test_plain"):
        cut = {
            key: states for key, states in outcomes.items() if key != junit_key(missing)
        }
        assert len(cut) == len(outcomes) - 1, (missing, sorted(outcomes))
        problems = _listing_problems("unit", listing, cut)
        assert _kinds(problems) == ["INCOMPLETE"], problems
        assert problems[0].endswith(missing), problems

    # Windows path separators are still normalised, and only before the parametrize id.
    assert junit_key("tests\\unit\\test_x.py::TestK::test_a[a\\n]") == (
        "tests.unit.test_x.TestK",
        "test_a[a\\n]",
    )


# ---------------------------------------------------------------------------
# Fail-closed controls on this module's own guards.
# ---------------------------------------------------------------------------


def test_the_absent_data_tree_skip_guard_actually_fires(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A guard only ever observed NOT firing is indistinguishable from an unwired one.

    Drives exactly the path a checkout without the gitignored lake would take, and
    asserts the skip is issued with the PINNED message rather than the verdict quietly
    passing.
    """
    monkeypatch.delenv(CHILD_ENV, raising=False)
    monkeypatch.setattr(
        "tests.integration.test_phase33_expected_failure_set.PRODUCTION_DATA_TREE",
        REPO_ROOT / "data_tree_that_does_not_exist",
    )
    with pytest.raises(pytest.skip.Exception) as excinfo:
        _require_runnable_tripwires()
    assert str(excinfo.value) == NO_PRODUCTION_DATA_SKIP


def test_the_child_session_recursion_guard_actually_fires(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The recursion guard steps aside by name, and takes precedence over the data check.

    Ordering matters: a child session on a machine that HAS the lake must still step
    aside, so the child check has to come first. Asserting the message proves it did.
    """
    monkeypatch.setenv(CHILD_ENV, "1")
    with pytest.raises(pytest.skip.Exception) as excinfo:
        _require_runnable_tripwires()
    assert str(excinfo.value) == CHILD_SESSION_SKIP


def test_the_child_environment_carries_the_recursion_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every child this module spawns is marked, and never inherits either opt-in.

    A child that inherited ``NFL_PHASE33_FULL_TIER_VERDICT`` would run the whole tier
    again from inside the whole-tier run; one that inherited ``NFL_PHASE38_FULL_RUN_DIR``
    would judge a recorded run from inside a child. Both opt-ins are SET here first, so
    their absence proves the child environment drops them rather than never saw them.
    """
    monkeypatch.setenv(FULL_TIER_ENV, "1")
    monkeypatch.setenv(FULL_RUN_DIR_ENV, str(REPO_ROOT))
    env = _child_env()
    assert env[CHILD_ENV] == "1"
    assert FULL_TIER_ENV not in env
    assert FULL_RUN_DIR_ENV not in env


def test_this_gate_re_lists_no_node_id_of_its_own() -> None:
    """The discipline in the docstring, enforced: the set comes from ONE committed source.

    The needle is BUILT BY CONCATENATION rather than written out, because a literal
    spelling of it here would be the very thing this test forbids -- a check that cannot
    satisfy its own rule is a check somebody deletes.

    The module docstring is exempt: it pre-declares the Phase-33 gold-rebuild hazard in
    prose, which is the point of pre-declaring it in source.
    """
    import ast

    needle = "tests/" + "integration/" + "test_"
    source_path = Path(__file__).resolve()
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))

    docstring_node = None
    if tree.body and isinstance(tree.body[0], ast.Expr):
        candidate = tree.body[0].value
        if isinstance(candidate, ast.Constant) and isinstance(candidate.value, str):
            docstring_node = candidate

    offenders = [
        f"line {node.lineno}: {node.value!r}"
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and node is not docstring_node
        and needle in node.value
    ]
    assert not offenders, (
        "this gate contains node-id-shaped string literal(s) outside its docstring:\n"
        + "\n".join(f"  - {line}" for line in offenders)
        + "\n\nEvery node id must be read from "
        "tests.phase33_state.DELIBERATE_TRIPWIRE_NODE_IDS. A gate that asserts a set it "
        "also defines proves nothing: the two copies drift and the gate's copy wins."
    )
