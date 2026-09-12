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

BOTH DIRECTIONS, WITH DIFFERENT MEANINGS
----------------------------------------
Set equality is asserted in both directions and each direction gets its own message,
because they are different events:

* an EXTRA failing id is a REGRESSION this phase introduced;
* a MISSING failing id is a DISCLOSURE this phase erased.

A strict subset is a violation, not an improvement. That is the whole claim.

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
matters here (a tripwire that stopped failing) and proves the other direction only within
its own run.

The whole-tier form, which would also catch a NEW red elsewhere in the tier, is opt-in
via ``NFL_PHASE33_FULL_TIER_VERDICT`` for two reasons that are not cost: this module runs
INSIDE the tier it would measure, so an unguarded tier run re-collects this module and
spawns a child of a child without end; and the tier-wide reconciliation is Plan 33-18's
closing three-tier measurement, which owns it properly. The recursion guard is
``NFL_PHASE33_EXPECTED_FAILURE_CHILD``, set on every child environment, and a child that
sees it steps aside by name.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from tests import phase33_state

REPO_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_DATA_TREE = REPO_ROOT / "data"

# The tier each tripwire belongs to, DERIVED from its own node id rather than restated.
# Every tripwire lives under tests/integration today; deriving the tier means a tripwire
# that moves tiers is reflected automatically instead of silently mis-grouped.
TIER_ROOTS = ("tests/unit", "tests/integration", "tests/api")

FULL_TIER_ENV = "NFL_PHASE33_FULL_TIER_VERDICT"
CHILD_ENV = "NFL_PHASE33_EXPECTED_FAILURE_CHILD"

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

# `FAILED tests/integration/x.py::C::t - AssertionError: ...` and the bare form, plus
# ERROR lines: a tripwire that ERRORS is not a tripwire that went green, and an errored
# node that is not a tripwire is a regression exactly as a failed one is.
_OUTCOME_RE = re.compile(r"^(?:FAILED|ERROR)\s+(\S+)", re.MULTILINE)


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


def failing_node_ids(report: str) -> set[str]:
    """Every FAILED or ERROR node id in a pytest ``-q`` report.

    Args:
        report: Combined stdout and stderr of a pytest run.

    Returns:
        The node ids, path separators normalised to forward slashes.
    """
    return {match.replace("\\", "/") for match in _OUTCOME_RE.findall(report)}


def _child_env() -> dict[str, str]:
    """The environment for a child pytest session, carrying the recursion guard."""
    env = dict(os.environ)
    env[CHILD_ENV] = "1"
    env.pop(FULL_TIER_ENV, None)
    return env


def _run_pytest(arguments: list[str], timeout: int) -> str:
    """Run pytest in a subprocess and return its combined output.

    ``-o addopts=`` neutralises this repository's ``addopts = ["-v", ...]``. The
    repo-wide ``-v`` replaces the parseable short summary with a verbose per-test report,
    so the ``FAILED <nodeid>`` lines this module reads only appear once it is off. The
    same override, for the same reason, is used by
    ``tests/unit/test_phase32_state_guard.py``.

    Args:
        arguments: Arguments after ``-m pytest``.
        timeout: Seconds before the child is killed.

    Returns:
        stdout concatenated with stderr.
    """
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-o", "addopts=", "-q", *arguments],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=_child_env(),
        check=False,
    )
    return result.stdout + result.stderr


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
# Half two: the verdict. The measured failing set EQUALS the constant.
# ---------------------------------------------------------------------------


def test_the_measured_failing_set_equals_the_tripwire_constant() -> None:
    """Both directions, each with its own meaning, over a run of exactly the tripwires.

    An EXTRA failing id in this run is a regression reached through the tripwires' own
    modules. A MISSING one is a tripwire that has gone green -- an owner-accepted
    disclosure erased.
    """
    _require_runnable_tripwires()

    expected = set(phase33_state.DELIBERATE_TRIPWIRE_NODE_IDS)
    output = _run_pytest(sorted(expected), timeout=900)
    measured = failing_node_ids(output)

    extra = measured - expected
    assert not extra, (
        "REGRESSION: node id(s) failed that are NOT deliberate tripwires. This phase "
        "introduced a red that nobody accepted:\n"
        + "\n".join(f"  - {node_id}" for node_id in sorted(extra))
        + f"\n\nFull report:\n{output}"
    )

    missing = expected - measured
    assert not missing, (
        "DISCLOSURE ERASED: deliberate tripwire(s) did NOT fail. Each of these encodes "
        "an owner-accepted fact from Phase 30 or Phase 31, and a green one means the "
        "record was cleared rather than a defect fixed:\n"
        + "\n".join(f"  - {node_id}" for node_id in sorted(missing))
        + "\n\nIf a rebuild in this phase caused it, the correct response is to "
        "investigate why the attribution changed and to record the finding -- never to "
        f"accept the green as progress.\n\nFull report:\n{output}"
    )

    assert measured == expected


def test_the_whole_tier_verdict_when_opted_in() -> None:
    """The stronger form: no red ANYWHERE in a tripwire's tier beyond the constant.

    Opt-in, because this module runs inside the tier it measures. See the module
    docstring: the recursion guard is what makes the child safe, and Plan 33-18's closing
    three-tier measurement is what owns the tier-wide reconciliation properly.
    """
    _require_runnable_tripwires()
    if not os.environ.get(FULL_TIER_ENV):
        pytest.skip(FULL_TIER_PENDING_SKIP)

    expected_by_tier: dict[str, set[str]] = {}
    for node_id in phase33_state.DELIBERATE_TRIPWIRE_NODE_IDS:
        expected_by_tier.setdefault(tier_of(node_id), set()).add(node_id)

    for tier, expected in sorted(expected_by_tier.items()):
        output = _run_pytest([tier], timeout=3600)
        measured = failing_node_ids(output)

        extra = measured - expected
        assert not extra, (
            f"REGRESSION in {tier}: node id(s) failed that are NOT deliberate "
            "tripwires:\n" + "\n".join(f"  - {node_id}" for node_id in sorted(extra))
        )
        missing = expected - measured
        assert not missing, (
            f"DISCLOSURE ERASED in {tier}: deliberate tripwire(s) did NOT fail:\n"
            + "\n".join(f"  - {node_id}" for node_id in sorted(missing))
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
