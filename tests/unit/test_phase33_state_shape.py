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
import builtins
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
STATE_MODULE_PATH = REPO_ROOT / "tests" / "phase33_state.py"

# The ONLY calls the manifest may make: pure builtins that cannot read or write anything, run
# outside code, or leave a side effect. The set is DERIVED from what ``tests/phase33_state.py``
# actually uses (measured 2026-09-21: ``tuple(...)`` and ``range(...)`` over literal ints, and
# ``dict.fromkeys(...)`` over literal tuples and a generator of literal tuples) and is no broader.
# It names callables, never line numbers or slot names: a new slot using one of these is
# admitted, and a new slot calling anything else is refused, without this list being edited.
#
# Owner ruling 2026-09-21 (Plan 33.2-06, orchestrator-assigned): the eleven entries that use
# these builtins stay exactly as appended -- the append-once protocol holds and no slot is
# rewritten as a hardcoded list. This check was sharpened instead, to test the risk its docstring
# names (I/O and import-time side effects) rather than flag every call expression.
_PURE_BUILTIN_CALLABLES: frozenset[str] = frozenset({"range", "tuple"})
_PURE_BUILTIN_METHODS: frozenset[tuple[str, str]] = frozenset({("dict", "fromkeys")})


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


def _names_bound_in(tree: ast.AST) -> set[str]:
    """Every name the module binds anywhere -- the names that could SHADOW a builtin.

    Assignment and annotation targets, loop and comprehension targets, walrus targets, import
    aliases, and function / class / argument names. A builtin is only trusted as the builtin
    when the module never rebinds its name.
    """
    bound: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            bound.add(node.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            bound.update(
                (alias.asname or alias.name).split(".")[0] for alias in node.names
            )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.arg):
            bound.add(node.arg)
    return bound


def disallowed_calls(source: str) -> list[str]:
    """Every call in *source* that is NOT a pure, unshadowed builtin on the allowlist.

    A call is admitted ONLY when its callee is a bare ``Name`` in
    :data:`_PURE_BUILTIN_CALLABLES`, or a ``Name.attr`` pair in :data:`_PURE_BUILTIN_METHODS`
    whose base is the bare builtin name -- and, in both cases, the module never rebinds that
    name. Everything else is refused: ``open`` / ``print`` / any non-builtin name, any module
    attribute call (``os.getenv(...)``, ``pathlib.Path(...).read_text()``), an allowlisted name
    reached as an attribute of something else (``x.range(...)``), and any call on a call's
    result. Nested calls are each judged on their own, so an allowed call cannot launder a
    forbidden one through its arguments.
    """
    tree = ast.parse(source)
    shadowed = _names_bound_in(tree)
    violations: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            if func.id in _PURE_BUILTIN_CALLABLES and func.id not in shadowed:
                continue
            reason = (
                f"`{func.id}` is shadowed in the module"
                if func.id in _PURE_BUILTIN_CALLABLES
                else f"`{func.id}(...)` is not an allowlisted pure builtin"
            )
        elif (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and (func.value.id, func.attr) in _PURE_BUILTIN_METHODS
        ):
            if func.value.id not in shadowed:
                continue
            reason = f"`{func.value.id}` is shadowed in the module"
        else:
            reason = f"`{ast.unparse(func)}(...)` is not an allowlisted pure builtin"
        violations.append(f"line {node.lineno}: {reason}")
    return violations


def test_the_allowlist_names_only_real_builtins() -> None:
    """The allowlist cannot name something that is not the interpreter's own builtin."""
    for name in _PURE_BUILTIN_CALLABLES:
        assert callable(getattr(builtins, name)), name
    for owner, method in _PURE_BUILTIN_METHODS:
        assert callable(getattr(getattr(builtins, owner), method)), (owner, method)


def test_the_call_scan_reaches_the_calls_the_module_really_makes() -> None:
    """Anti-vacuity: the manifest does make allowlisted calls, so the scan is exercised.

    It also proves the allowlisted names are the builtins HERE: the manifest rebinds none of
    them, so ``tuple`` / ``range`` / ``dict`` in it can only resolve to the interpreter's own.
    """
    tree = ast.parse(_module_source(), filename=str(STATE_MODULE_PATH))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    assert calls, "the manifest makes no calls -- the allowlist would be untested"
    trusted = _PURE_BUILTIN_CALLABLES | {owner for owner, _ in _PURE_BUILTIN_METHODS}
    assert not _names_bound_in(tree) & trusted


def test_the_state_module_has_no_module_level_calls() -> None:
    """No I/O and no import-time side effect, per the manifest's own stated constraint.

    A module-level call -- a file read, a print, a function invocation used to compute a
    "constant" -- would give this module a side effect on import, breaking the promise
    that any test at any tier can import it without cost. The pure builtins in
    :data:`_PURE_BUILTIN_CALLABLES` / :data:`_PURE_BUILTIN_METHODS` can do none of those
    things and are admitted; every other call is refused (see :func:`disallowed_calls`).
    """
    violations = disallowed_calls(_module_source())

    assert not violations, (
        "I/O-shaped call(s) found in tests/phase33_state.py:\n"
        + "\n".join(f"  - {line}" for line in violations)
    )


# Each planted source is fed through the SAME scan the real manifest goes through.
_PLANTED_VIOLATIONS: tuple[tuple[str, str, str], ...] = (
    ("open", "X = open('data/silver/games.parquet').read()", "`open(...)`"),
    ("print", "X = print('side effect')", "`print(...)`"),
    ("module-attribute-getenv", "X = os.getenv('HOME')", "`os.getenv(...)`"),
    ("path-read-text", "X = pathlib.Path('f').read_text()", "read_text"),
    ("unknown-name", "X = compute_the_constant()", "`compute_the_constant(...)`"),
    ("allowlisted-name-as-attribute", "X = x.range(3)", "`x.range(...)`"),
    (
        "allowlisted-method-off-a-module",
        "X = helpers.dict.fromkeys((1,), 0)",
        "`helpers.dict.fromkeys(...)`",
    ),
    ("shadowed-tuple", "tuple = list\nX = tuple((1, 2))", "`tuple` is shadowed"),
    (
        "shadowed-dict",
        "dict = object\nX = dict.fromkeys((1,), 0)",
        "`dict` is shadowed",
    ),
    (
        "forbidden-call-inside-an-allowed-one",
        "X = tuple(open(p) for p in ())",
        "`open(...)`",
    ),
)


@pytest.mark.parametrize(
    ("planted", "expected_fragment"),
    [(source, fragment) for _name, source, fragment in _PLANTED_VIOLATIONS],
    ids=[name for name, _source, _fragment in _PLANTED_VIOLATIONS],
)
def test_a_planted_side_effecting_call_is_refused(
    planted: str, expected_fragment: str
) -> None:
    """PLANTED VIOLATIONS: every shape the check exists to refuse is still refused."""
    violations = disallowed_calls(planted)
    assert violations, f"the scan admitted a planted violation: {planted!r}"
    assert any(expected_fragment in line for line in violations), violations


def test_the_allowlisted_shapes_the_manifest_uses_are_admitted() -> None:
    """NO FALSE POSITIVE: the shapes the eleven appended entries use pass the scan."""
    admitted = "\n".join(
        [
            "A = tuple((season, None) for season in range(2002, 2026))",
            "B = dict.fromkeys(('home_elo', 'away_elo'), 'REASON')",
            "C = dict.fromkeys(((season, None) for season in range(2018, 2026)), 'R')",
            "D = (*((season, None) for season in range(2002, 2018)),)",
        ]
    )
    assert disallowed_calls(admitted) == []


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
