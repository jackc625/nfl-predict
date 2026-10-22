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
  (a ``<failure>`` child), ``error`` (an ``<error>`` child: setup, teardown or
  collection), ``xfailed`` (``<skipped type="pytest.xfail">``), ``skipped`` (any other
  ``<skipped>``) or ``passed`` (no child). A call failure plus a teardown error is TWO
  testcase elements for one node; both states are kept, never collapsed to one;
* ``expected_outcome_problems`` is the ONE comparison. An empty list is the passing
  case; each problem names its kind:

  - ``DISCLOSURE ERASED`` -- a tripwire passed, or did not run at all;
  - ``TRIPWIRE NOT A FAILURE`` -- a tripwire errored, skipped or xfailed instead of
    failing. Only an assertion failure is the owner-accepted red;
  - ``TRIPWIRE ALSO ERRORED`` -- a tripwire failed AND errored (for example at teardown,
    where a COLD-05 per-test production-store guard surfaces). It is its own problem, so
    an error is never hidden behind a tripwire's expected failure;
  - ``REGRESSION`` -- a node that is not a tripwire failed or errored (one message per
    node, even when it did both);
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

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping
from pathlib import Path

import pytest
from _pytest.junitxml import mangle_test_address

from tests import phase33_state

REPO_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_DATA_TREE = REPO_ROOT / "data"

# The tier each tripwire belongs to, DERIVED from its own node id rather than restated.
# Every tripwire lives under tests/integration today; deriving the tier means a tripwire
# that moves tiers is reflected automatically instead of silently mis-grouped.
TIER_ROOTS = ("tests/unit", "tests/integration", "tests/api")

FULL_TIER_ENV = "NFL_PHASE33_FULL_TIER_VERDICT"
CHILD_ENV = "NFL_PHASE33_EXPECTED_FAILURE_CHILD"

# Mirrors tests/conftest.py's CLOSING_SWEEP_HEADER exactly, as
# tests/integration/test_data_boundary_guard_arming.py does: conftest is a pytest plugin
# and is not imported. The closing sweep adds no JUnit testcase and only flips an exit
# status that was 0, so a JUnit-only judge cannot see it; the run's output is read too.
CLOSING_SWEEP_HEADER = "SESSION-END FULL CONTENT SWEEP"

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


def junit_key(node_id: str) -> tuple[str, str]:
    """The ``(classname, name)`` a JUnit report files a node id under.

    Built with ``_pytest.junitxml.mangle_test_address``, the function that wrote the
    report, so the mapping cannot drift from pytest's own (a parametrize id stays on the
    name). Forward mapping only: a classname cannot tell a module dot from a class dot.

    Args:
        node_id: A pytest node id; backslash path separators are normalised first.

    Returns:
        The testcase's ``(classname, name)``.
    """
    names = mangle_test_address(node_id.replace("\\", "/"))
    return ".".join(names[:-1]), names[-1]


def _testcase_state(testcase: ET.Element) -> str:
    """The one state a single ``<testcase>`` element records."""
    tags = {child.tag: child.get("type") for child in testcase}
    if "failure" in tags:
        return "failed"
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
        ``QUARANTINE``). An empty list is the passing case.
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
        if (classname, name) not in expected_keys and states & {"failed", "error"}:
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


def _child_env() -> dict[str, str]:
    """The environment for a child pytest session, carrying the recursion guard."""
    env = dict(os.environ)
    env[CHILD_ENV] = "1"
    env.pop(FULL_TIER_ENV, None)
    return env


def _run_pytest(arguments: list[str], timeout: int, report: Path | None = None) -> str:
    """Run pytest in a subprocess and return its combined output.

    ``-o addopts=`` neutralises this repository's ``addopts = ["-v", ...]``, so the
    child's output is the short ``-q`` form the collectibility check reads. The same
    override is used by ``tests/unit/test_phase32_state_guard.py``.

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

    # A sixth red, whether it failed, errored, or both: exactly one REGRESSION.
    for states in ({"failed"}, {"error"}, {"failed", "error"}):
        problems = judged({bystander: frozenset(states)})
        assert _kinds(problems) == ["REGRESSION"], problems
        assert "test_bystander" in problems[0]

    # Only an assertion failure is the accepted red.
    for states in ({"error"}, {"skipped"}):
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
    assert _kinds(as_bystander) == [
        "QUARANTINE",
        "REGRESSION",
        "REGRESSION",
        "REGRESSION",
    ]


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
    for node_id in node_ids:
        assert node_id in output.replace("\\", "/"), (
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


def test_the_child_environment_carries_the_recursion_guard() -> None:
    """Every child this module spawns is marked, and never inherits the opt-in.

    A child that inherited ``NFL_PHASE33_FULL_TIER_VERDICT`` would run the whole tier
    again from inside the whole-tier run. The guard and the opt-out are asserted together
    because either one alone still recurses.
    """
    env = _child_env()
    assert env[CHILD_ENV] == "1"
    assert FULL_TIER_ENV not in env


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
