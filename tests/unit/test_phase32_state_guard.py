"""tests/phase32_state.py has zero importers anywhere else in the repo -- this module is the
first and only one.

WHY THIS MODULE EXISTS
-----------------------
``tests/phase32_state.py``'s own docstring says: "A later plan that asserts against a failure
set whose members were never enumerated is asserting against a number, not against a fact."
That is exactly the gap this module closes. Without it, the manifest is inert prose: nothing
proves the 5 ``DELIBERATE_TRIPWIRE_NODE_IDS`` still exist on a fresh checkout, and nothing
proves the manifest's own internal partition (six pre-phase failures, five of them deliberate,
one of them the pin's 2026 refusal) is still arithmetically coherent.

TWO THINGS THIS MODULE DOES, AND ONE THING IT DELIBERATELY DOES NOT DO
-----------------------------------------------------------------------
1. TRIPWIRE EXISTENCE: for each of the 5 tripwire node ids, prove pytest can still COLLECT it
   (import its module, find its class, find its function) via a ``--collect-only`` subprocess.
   Collection catches deletion, renaming, and class/function moves -- the realistic ways a
   disclosure gets silently erased during later refactors. Collection does NOT catch a tripwire
   whose body was edited to pass while its name stays put (a test silently "fixed" in place);
   that failure mode needs the test to actually RUN, which this module deliberately does not do,
   because these are slow integration tests that read gold parquet and the unit tier must not
   inherit that cost. Running them lives in the two-tier command this repo already uses.
2. PARTITION COHERENCE: the manifest's own six-way / five-way arithmetic, checked as real
   assertions instead of a one-off plan-time one-liner that left no durable artifact.

CONSTANTS-ONLY DISCIPLINE (gap 2)
----------------------------------
An AST scan proving ``tests/phase32_state.py`` still honors its own stated constraint: no
import beyond ``from __future__``, no function/class definitions, no module-level calls, and
pure ASCII. Modeled on ``tests/unit/test_p31_constants_isolation.py``'s scan style, including
its anti-vacuity discipline: a scan that visited a trivial (or empty) module would be a green
test that proves nothing, so this module also asserts it visited a real number of assignments.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

from tests import phase32_state

REPO_ROOT = Path(__file__).resolve().parents[2]
STATE_MODULE_PATH = REPO_ROOT / "tests" / "phase32_state.py"


# ---------------------------------------------------------------------------
# Gap 1a: the 5 deliberate tripwires must still be collectible.
# ---------------------------------------------------------------------------


def test_the_scan_visits_a_non_empty_tripwire_list() -> None:
    """Anti-vacuity: there must be at least one tripwire id to check.

    Every assertion below is of the form "pytest could still collect this id", which is
    trivially satisfiable by checking zero ids. This is what makes the others mean something.
    """
    assert phase32_state.DELIBERATE_TRIPWIRE_NODE_IDS, (
        "DELIBERATE_TRIPWIRE_NODE_IDS is empty -- the collectibility check below would then "
        "run over nothing and pass while proving nothing."
    )
    assert len(phase32_state.DELIBERATE_TRIPWIRE_NODE_IDS) == 5


def test_every_deliberate_tripwire_is_still_collectible() -> None:
    """Each of the 5 owner-accepted disclosures must still resolve to a real, findable test.

    This runs `pytest --collect-only -q` in a SUBPROCESS over all 5 node ids at once and
    asserts each one is echoed back in the collected output with no collection error. It does
    NOT run the tripwires (they are slow, gold-reading integration tests) and it does NOT prove
    a tripwire still fails for the right reason -- only that nobody deleted it, renamed it, or
    moved its class/function so the id no longer resolves. A tripwire quietly turned green in
    place (body edited, name and location untouched) is invisible to collection; catching that
    requires actually running the node, which is out of scope for the unit tier by design.
    """
    node_ids = list(phase32_state.DELIBERATE_TRIPWIRE_NODE_IDS)
    result = subprocess.run(
        # `-o addopts=""` overrides pyproject.toml's `addopts = ["-v", ...]`: the repo-wide
        # `-v` forces pytest's verbose TREE collection report (one line per package/module/
        # class), which does not print a matchable "path::Class::test" node id per line. The
        # plain `-q` report this test parses only appears once `-v` is neutralised.
        [
            sys.executable,
            "-m",
            "pytest",
            "-o",
            "addopts=",
            "--collect-only",
            "-q",
            *node_ids,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=120,
    )
    combined_output = result.stdout + result.stderr

    assert "ERROR" not in combined_output, (
        "pytest --collect-only reported a collection ERROR against one or more deliberate "
        f"tripwire node ids -- one was likely deleted, renamed, or moved.\n\n{combined_output}"
    )
    assert "no tests ran" not in combined_output.lower(), (
        f"pytest --collect-only found none of the deliberate tripwire node ids.\n\n{combined_output}"
    )

    for node_id in node_ids:
        # The collect-only report prints each collected item's own node id (module path plus
        # ``::Class::test_name``), so a straight substring check is the right level of proof
        # here: it fails if the class or function name was renamed or moved to another module.
        assert node_id in combined_output, (
            f"deliberate tripwire node id no longer collects:\n  {node_id}\n\n"
            f"Full collect-only output:\n{combined_output}"
        )


# ---------------------------------------------------------------------------
# Gap 1b: the manifest's own partition arithmetic is coherent.
# ---------------------------------------------------------------------------


def test_the_pre_phase_failure_set_partitions_into_five_tripwires_and_one_sixth_red() -> (
    None
):
    """32-01 Task 3's one-liner, reproduced as a durable, re-runnable assertion.

    Six pre-phase failures, five of them deliberate tripwires that must stay red, one of them
    (SIXTH_RED_NODE_ID) the pin's own 2026 refusal -- the thing Phase 32 is actually about.
    """
    assert len(phase32_state.PRE_PHASE_FAILING_NODE_IDS) == 6
    assert len(phase32_state.DELIBERATE_TRIPWIRE_NODE_IDS) == 5

    pre_phase = set(phase32_state.PRE_PHASE_FAILING_NODE_IDS)
    tripwires = set(phase32_state.DELIBERATE_TRIPWIRE_NODE_IDS)

    assert phase32_state.SIXTH_RED_NODE_ID in pre_phase
    assert tripwires <= pre_phase
    assert phase32_state.SIXTH_RED_NODE_ID not in tripwires

    # The partition must be exhaustive: five tripwires plus the sixth red account for all six.
    assert pre_phase == tripwires | {phase32_state.SIXTH_RED_NODE_ID}


def test_the_post_phase_failure_set_is_exactly_the_tripwires_no_more_no_less() -> None:
    """No disclosure was erased and no new red was introduced.

    This is the set-equality check the manifest's own prose promises: every remaining red after
    Phase 32 is a deliberate tripwire, and every deliberate tripwire is still red. A tripwire
    that quietly disappeared from POST_PHASE_FAILING_NODE_IDS, or a stray new failure that
    appeared alongside them, would break this equality.
    """
    assert set(phase32_state.POST_PHASE_FAILING_NODE_IDS) == set(
        phase32_state.DELIBERATE_TRIPWIRE_NODE_IDS
    )


# ---------------------------------------------------------------------------
# Gap 2: constants-only discipline, checked by AST scan.
# ---------------------------------------------------------------------------


def _module_source() -> str:
    return STATE_MODULE_PATH.read_text(encoding="ascii")


def test_the_scan_visits_a_non_trivial_module() -> None:
    """Anti-vacuity: the constants-only scan must be checking a real, substantial module.

    A scan that ran against an empty or near-empty file would pass every "no violation found"
    assertion below while proving nothing about actual discipline.
    """
    tree = ast.parse(_module_source(), filename=str(STATE_MODULE_PATH))
    module_level_assignments = [
        node for node in tree.body if isinstance(node, (ast.Assign, ast.AnnAssign))
    ]
    assert len(module_level_assignments) >= 10, (
        f"tests/phase32_state.py has only {len(module_level_assignments)} module-level "
        "assignments -- too few for the discipline scan below to mean anything."
    )


def test_the_state_module_is_pure_ascii() -> None:
    """The manifest's own stated constraint: ASCII only, so it decodes safely everywhere."""
    raw_bytes = STATE_MODULE_PATH.read_bytes()
    try:
        raw_bytes.decode("ascii")
    except UnicodeDecodeError as exc:
        raise AssertionError(
            f"tests/phase32_state.py contains a non-ASCII byte at offset {exc.start}, "
            "violating its own stated ASCII-only constraint."
        ) from exc


def test_the_state_module_imports_nothing_but_future_annotations() -> None:
    """No project imports, per the manifest's own stated constraint.

    A project import would let a constants-only module (importable by any test at any tier,
    per its own docstring) pull in I/O or heavyweight dependencies transitively, defeating the
    point of a cheap, side-effect-free manifest.
    """
    tree = ast.parse(_module_source(), filename=str(STATE_MODULE_PATH))
    violations: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.module != "__future__":
                violations.append(
                    f"line {node.lineno}: `from {node.module} import ...` -- only "
                    "`from __future__ import annotations` is permitted."
                )
        elif isinstance(node, ast.Import):
            violations.append(
                f"line {node.lineno}: `import {', '.join(a.name for a in node.names)}` -- "
                "no plain imports are permitted at all."
            )

    assert not violations, (
        "constants-only violation(s) in tests/phase32_state.py:\n"
        + "\n".join(f"  - {line}" for line in violations)
    )


def test_the_state_module_defines_no_functions_or_classes() -> None:
    """No logic, per the manifest's own stated constraint.

    A def or class would give this "constants only" module behavior that could itself have
    bugs, side effects, or hidden state -- exactly what the manifest's docstring says it must
    never be, because every tier's tests are expected to import it at zero cost.
    """
    tree = ast.parse(_module_source(), filename=str(STATE_MODULE_PATH))
    violations = [
        f"line {node.lineno}: {'class' if isinstance(node, ast.ClassDef) else 'function'} "
        f"'{node.name}' defined"
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    assert not violations, "logic found in tests/phase32_state.py:\n" + "\n".join(
        f"  - {line}" for line in violations
    )


def test_the_state_module_has_no_module_level_calls() -> None:
    """No I/O, per the manifest's own stated constraint.

    A module-level call (a file read, a print, a function invocation used to compute a
    "constant") would give this module a side effect on import, breaking the promise that any
    test at any tier can import it without cost.
    """
    tree = ast.parse(_module_source(), filename=str(STATE_MODULE_PATH))
    violations: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            violations.append(f"line {node.lineno}: call expression found")

    assert not violations, (
        "I/O-shaped call(s) found in tests/phase32_state.py:\n"
        + "\n".join(f"  - {line}" for line in violations)
    )
