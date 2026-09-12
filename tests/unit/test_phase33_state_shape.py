"""``tests/phase33_state.py`` still honours its own stated constraints.

WHY THIS MODULE EXISTS
----------------------
``tests/phase33_state.py``'s docstring makes four mechanical promises about itself --
"Constants only. NO project imports, NO I/O and NO logic ... ASCII only" -- and an
eighteen-plan phase is going to append to it eighteen times. A promise that nothing
checks is a comment, and a comment is exactly what gets broken by the fourteenth
append: somebody needs one derived value, writes a two-line helper, and the manifest
every tier imports at zero cost quietly starts doing work on import.

This is a direct port of ``tests/unit/test_phase32_state_guard.py``'s five shape tests
(``test_the_scan_visits_a_non_trivial_module:169`` through
``test_the_state_module_has_no_module_level_calls:245``), retargeted at the Phase-33
manifest. The AST idiom and the byte-offset-naming ASCII failure message are kept
deliberately identical: two spellings of one check is the second-list failure this
project has already paid for, and where a check must exist twice it should at least
read the same.

WHAT THIS MODULE DOES NOT DO
----------------------------
It does not check that a name is assigned only once -- that is the APPEND-ONCE protocol
and it lives in ``tests/unit/test_phase33_state_append_once.py``, because it is a
different claim with its own planted-violation control.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
STATE_MODULE_PATH = REPO_ROOT / "tests" / "phase33_state.py"


def _module_source() -> str:
    return STATE_MODULE_PATH.read_text(encoding="ascii")


def test_the_scan_visits_a_non_trivial_module() -> None:
    """Anti-vacuity: the constants-only scan must be checking a real, substantial module.

    Every other assertion here is of the form "no violation was found", which is
    trivially satisfiable by scanning an empty file. This is what makes the others mean
    something.
    """
    tree = ast.parse(_module_source(), filename=str(STATE_MODULE_PATH))
    module_level_assignments = [
        node for node in tree.body if isinstance(node, (ast.Assign, ast.AnnAssign))
    ]
    assert len(module_level_assignments) >= 10, (
        f"tests/phase33_state.py has only {len(module_level_assignments)} module-level "
        "assignments -- too few for the discipline scan below to mean anything."
    )


def test_the_state_module_is_pure_ascii() -> None:
    """The manifest's own stated constraint: ASCII only, so it decodes safely everywhere."""
    raw_bytes = STATE_MODULE_PATH.read_bytes()
    try:
        raw_bytes.decode("ascii")
    except UnicodeDecodeError as exc:
        raise AssertionError(
            f"tests/phase33_state.py contains a non-ASCII byte at offset {exc.start}, "
            "violating its own stated ASCII-only constraint."
        ) from exc


def test_the_state_module_imports_nothing_but_future_annotations() -> None:
    """No project imports, per the manifest's own stated constraint.

    A project import would let a constants-only module -- importable by any test at any
    tier, per its own docstring -- pull in I/O or heavyweight dependencies transitively,
    defeating the point of a cheap, side-effect-free manifest.
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
        "constants-only violation(s) in tests/phase33_state.py:\n"
        + "\n".join(f"  - {line}" for line in violations)
    )


def test_the_state_module_defines_no_functions_or_classes() -> None:
    """No logic, per the manifest's own stated constraint.

    A def or class would give this "constants only" module behaviour that could itself
    have bugs, side effects, or hidden state -- exactly what the manifest's docstring
    says it must never be.
    """
    tree = ast.parse(_module_source(), filename=str(STATE_MODULE_PATH))
    violations = [
        f"line {node.lineno}: {'class' if isinstance(node, ast.ClassDef) else 'function'} "
        f"'{node.name}' defined"
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]
    assert not violations, "logic found in tests/phase33_state.py:\n" + "\n".join(
        f"  - {line}" for line in violations
    )


def test_the_state_module_has_no_module_level_calls() -> None:
    """No I/O, per the manifest's own stated constraint.

    A module-level call -- a file read, a print, a function invocation used to compute a
    "constant" -- would give this module a side effect on import, breaking the promise
    that any test at any tier can import it without cost.
    """
    tree = ast.parse(_module_source(), filename=str(STATE_MODULE_PATH))
    violations: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            violations.append(f"line {node.lineno}: call expression found")

    assert not violations, (
        "I/O-shaped call(s) found in tests/phase33_state.py:\n"
        + "\n".join(f"  - {line}" for line in violations)
    )


def test_the_module_body_holds_only_assignments_a_docstring_and_the_future_import() -> (
    None
):
    """The positive form of the three scans above, stated once as a whitelist.

    The scans above enumerate what is forbidden. This states what is PERMITTED, so a
    node type nobody thought to forbid -- a module-level ``if``, a ``for``, a ``with`` --
    is caught as well. A manifest with a conditional in it does not have one value per
    slot; it has one value per environment, which is not a record of anything.
    """
    tree = ast.parse(_module_source(), filename=str(STATE_MODULE_PATH))
    offenders: list[str] = []
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant):
            continue
        offenders.append(f"line {node.lineno}: {type(node).__name__}")

    assert not offenders, (
        "tests/phase33_state.py's module body contains node types outside the permitted "
        "set (Assign / AnnAssign / a docstring Expr / the __future__ import):\n"
        + "\n".join(f"  - {line}" for line in offenders)
    )
