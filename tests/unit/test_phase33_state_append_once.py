"""The APPEND-ONCE protocol, enforced mechanically instead of by convention.

WHAT THE PROTOCOL SAYS, AND WHY A COMMENT IS NOT ENOUGH
------------------------------------------------------
``tests/phase32_state.py:30-34`` states the rule these manifests live by:

    Each slot is APPENDED ONCE by its owning plan and is NOT edited afterwards.

The reason is in ``tests/phase32_state.py:151-153``: "a state manifest whose earlier
slots move is a manifest that cannot be used to reconstruct what was believed when."
The whole value of the artifact is that a reader can go back and see what a plan
measured AT THE TIME, not what a later plan wished it had measured.

Phase 33 is eighteen plans long and every one of them appends. Eighteen chances for a
plan to redefine a name a previous plan already set -- and Python will not complain: the
last assignment silently wins, the earlier record is gone, and the module still imports
cleanly. There is nothing to notice.

So the protocol is checked here rather than stated. An AST scan collects every
module-level assignment TARGET NAME in ``tests/phase33_state.py`` and asserts each
appears exactly once. A redefinition is a test failure that names the offending names.

WHY THIS IS A SEPARATE MODULE FROM THE SHAPE SCAN
-------------------------------------------------
``tests/unit/test_phase33_state_shape.py`` checks what the manifest MAY CONTAIN
(constants, no imports, no logic, ASCII). This checks a different claim -- that a name
is written once -- and it carries its own planted-violation control, because a scan that
has only ever been observed finding nothing is indistinguishable from a scan that is not
wired up.

FOUR CONTROLS, following ``tests/unit/test_p31_constants_isolation.py``'s form:
non-vacuity (the scan collected a non-empty name list), the assertion itself, a PLANTED
violation (a temp module with one name assigned twice IS flagged), and a
no-false-positive control (a module with eighteen distinct names is clean).

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
STATE_MODULE_PATH = REPO_ROOT / "tests" / "phase33_state.py"


def module_level_assignment_names(path: Path) -> list[str]:
    """Every module-level assignment target name in *path*, in source order.

    Only the MODULE body is walked, never nested scopes: a name bound inside a function
    is not a manifest slot and shadowing one is not a protocol violation. (The manifest
    is separately proven to contain no functions at all, so today the distinction is
    theoretical -- but a scan whose scope is accidental rather than chosen is a scan
    whose meaning changes when the file does.)

    Args:
        path: A Python source file.

    Returns:
        The target names, with duplicates preserved so the caller can count them.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                names.append(node.target.id)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.append(target.id)
                elif isinstance(target, (ast.Tuple, ast.List)):
                    names.extend(
                        element.id
                        for element in target.elts
                        if isinstance(element, ast.Name)
                    )
    return names


def redefined_names(path: Path) -> list[str]:
    """The module-level names *path* assigns more than once, sorted.

    Args:
        path: A Python source file.

    Returns:
        Every name assigned two or more times at module level.
    """
    counts = Counter(module_level_assignment_names(path))
    return sorted(name for name, count in counts.items() if count > 1)


# ---------------------------------------------------------------------------
# Control 1: non-vacuity.
# ---------------------------------------------------------------------------


def test_the_scan_collected_a_non_empty_name_list() -> None:
    """A scan over an empty name list reports nothing and proves nothing.

    Every "no redefinition found" assertion below is trivially satisfiable by collecting
    nothing at all -- an `ast.walk` typo, a body shape the collector does not recognise,
    a path that does not resolve. This is the test that makes the others mean something.
    """
    names = module_level_assignment_names(STATE_MODULE_PATH)
    assert names, (
        "the append-once scan collected ZERO module-level assignment names from "
        "tests/phase33_state.py. The no-redefinition assertion would then pass while "
        "checking nothing."
    )
    assert len(names) >= 10, (
        f"the append-once scan collected only {len(names)} names "
        f"({names}) -- fewer than the manifest is known to define, so the collector is "
        "missing assignment shapes it should see."
    )


# ---------------------------------------------------------------------------
# Control 2: the assertion itself.
# ---------------------------------------------------------------------------


def test_no_name_in_the_phase33_manifest_is_assigned_twice() -> None:
    """APPEND ONCE: every slot is written by exactly one plan and never edited.

    A redefinition is not a merge conflict and not a syntax error -- Python takes the
    last assignment and discards the first without a word. The earlier plan's measured
    value is simply gone, and the manifest still imports cleanly, so nothing anywhere
    else in the suite would notice.
    """
    duplicates = redefined_names(STATE_MODULE_PATH)
    assert not duplicates, (
        "APPEND-ONCE VIOLATION in tests/phase33_state.py. These names are assigned more "
        "than once at module level, so an earlier plan's measured value has been "
        "silently overwritten:\n"
        + "\n".join(f"  - {name}" for name in duplicates)
        + "\n\nA later plan that needs a different value APPENDS A NEW SLOT under its "
        "own separator naming the plan, the task and the date. It does not edit an "
        "existing one: a manifest whose earlier slots move cannot be used to "
        "reconstruct what was believed when (tests/phase32_state.py:151-153)."
    )


# ---------------------------------------------------------------------------
# Control 3: the PLANTED violation -- the scan is proven to fire.
# ---------------------------------------------------------------------------


def test_the_scan_flags_a_planted_redefinition(tmp_path: Path) -> None:
    """Fail-closed control: a module with one name assigned twice IS reported.

    The temp module is written under ``tmp_path`` and never under ``data/`` or
    ``artifacts/``, so the autouse production-store write guard has nothing to say about
    it. It drives the REAL collector, not a copy of its logic.
    """
    planted = tmp_path / "planted_state.py"
    planted.write_text(
        "\n".join(
            [
                '"""A planted append-once violation."""',
                "",
                "from __future__ import annotations",
                "",
                "TESTS_ADDED_33_01: int = 44",
                "TESTS_ADDED_33_02: int = 7",
                "TESTS_ADDED_33_01: int = 45",
                "",
            ]
        ),
        encoding="utf-8",
    )

    assert redefined_names(planted) == ["TESTS_ADDED_33_01"], (
        "the append-once scan did NOT flag a module that assigns TESTS_ADDED_33_01 "
        "twice. The real assertion above is therefore a check that has only ever been "
        "observed passing, which is indistinguishable from one that cannot fail."
    )


def test_the_scan_flags_a_planted_plain_assignment_redefinition(tmp_path: Path) -> None:
    """The same control for the UNANNOTATED shape, and for multiple-target assignment.

    The manifest annotates every slot today, so a scan that only understood `AnnAssign`
    would pass forever and stop working the first time somebody writes `X = 1`. Both
    shapes are planted here so neither collector branch is untested.
    """
    planted = tmp_path / "planted_plain.py"
    planted.write_text(
        "\n".join(
            [
                "WAL_SIBLING_OBSERVATIONS = 1",
                "STAT_SIGNATURE_OBSERVATIONS = 8",
                "WAL_SIBLING_OBSERVATIONS = 0",
                "A = B = 3",
                "B = 4",
                "",
            ]
        ),
        encoding="utf-8",
    )

    assert redefined_names(planted) == ["B", "WAL_SIBLING_OBSERVATIONS"]


# ---------------------------------------------------------------------------
# Control 4: no false positive.
# ---------------------------------------------------------------------------


def test_a_module_with_eighteen_distinct_slots_is_clean(tmp_path: Path) -> None:
    """Fail-open control: the eighteen per-plan slots, each written once, is CLEAN.

    This is the shape ``tests/phase33_state.py`` will actually have at phase close. A
    scan that reddened on it would be one a later plan had to weaken, and a weakened
    guard asserts nothing.
    """
    clean = tmp_path / "clean_state.py"
    clean.write_text(
        "from __future__ import annotations\n\n"
        + "".join(
            f"TESTS_ADDED_33_{index:02d}: int = {index}\n" for index in range(1, 19)
        )
        + "TESTS_ADDED_BY_PHASE_33: int = 171\n",
        encoding="utf-8",
    )

    assert module_level_assignment_names(clean) == [
        *(f"TESTS_ADDED_33_{index:02d}" for index in range(1, 19)),
        "TESTS_ADDED_BY_PHASE_33",
    ]
    assert redefined_names(clean) == []
