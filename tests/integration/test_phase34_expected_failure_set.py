"""The Phase-34 expected failure set, as MEASURED and recorded, still holds (34-REVIEW WR-24).

``tests.phase34_state.P38_EXPECTED_FAILURE_BASELINE`` enumerates, by node id, the failures the
suite EXPECTS. It is the newest slot; ``P34_EXPECTED_FAILURE_BASELINE`` is the earlier one and is
checked here only as an unchanged record. 34-VALIDATION.md says the set is "asserted unchanged";
until this module, nothing read it. This module reads the slots and never edits them: each is a
committed record under its module's append protocol, and a re-measurement is a new slot written
by a plan, not by a test.

WHAT IS ASSERTED, HALF BY HALF
------------------------------
* The tripwires. The baseline does not re-list them; it names their one home,
  ``tests.phase33_state.DELIBERATE_TRIPWIRE_NODE_IDS``, and records that all of them failed when
  measured. That pointer must still resolve to a set of the recorded size. Whether they are STILL
  red is already enforced, by running them, in
  ``tests/integration/test_phase33_expected_failure_set.py``; that this gate exists is asserted
  here, so the tripwires are not run twice.
* The xfail quarantines. Every recorded node id still collects, and its function still carries
  ``pytest.mark.xfail(strict=True)``. In the other direction, no xfail exists anywhere under
  ``tests/`` outside the recorded set. A quarantine therefore cannot be erased by deleting the
  test or un-marking it, and a new one cannot appear unrecorded.

THE FIGURE IS ZERO (Phase 38). The P34 slot measured nine strict xfails on 2026-10-05 (older
records quote "7 xfailed"; nine is what was measured). Phase 38 fixed one of them (DEF-31-07, the
synthetic-stamp detector) and deleted the other eight by owner ruling, as reproductions of retired
models that today's gold cannot score. The P38 slot records zero, so the scan below now asserts
that no xfail site exists anywhere under ``tests/``. The v3.0 close count is retired; see the
slot's ``retired_count_baseline`` for its date and reason.

The data-dependent nodes are checked for collectability only: their recorded outcomes ("passed")
depend on the wall clock and the live stores, as the baseline itself says.

TEST CLASS: integration. It collects the recorded nodes in a subprocess (no test is run) and reads
source by AST. It writes nothing.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
import importlib
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tests.phase34_state import (
    P34_EXPECTED_FAILURE_BASELINE,
    P38_EXPECTED_FAILURE_BASELINE,
)

pytestmark = pytest.mark.integration

REPO_ROOT = Path(__file__).resolve().parents[2]
TESTS_ROOT = REPO_ROOT / "tests"
PHASE33_GATE = TESTS_ROOT / "integration" / "test_phase33_expected_failure_set.py"
PHASE33_GATE_TEST = "test_the_measured_failing_set_equals_the_tripwire_constant"

_BASELINE: dict[str, Any] = dict(P38_EXPECTED_FAILURE_BASELINE)
_EARLIER_BASELINE: dict[str, Any] = dict(P34_EXPECTED_FAILURE_BASELINE)
_QUARANTINES: tuple[str, ...] = tuple(_BASELINE["xfail_quarantines"])
_DATA_DEPENDENT: tuple[str, ...] = tuple(
    node_id for node_id, _outcome in _BASELINE["data_dependent_reds"]
)


def _site(node_id: str) -> tuple[str, str]:
    """``(file, Class::function)`` of a node id, with any parametrize id stripped."""
    path, _sep, rest = node_id.partition("::")
    return path, rest.split("[", 1)[0]


class _PytestNames:
    """The names a module binds to ``pytest``, ``pytest.mark`` and ``pytest.xfail``.

    ``import pytest as _pytest`` already exists in this suite, so a scan that knew only the
    name ``pytest`` would miss ``_pytest.mark.xfail``; ``from pytest import mark`` and
    ``from pytest import xfail`` bind the other two.
    """

    def __init__(self, tree: ast.AST) -> None:
        self.pytest: set[str] = {"pytest"}
        self.mark: set[str] = set()
        self.xfail: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "pytest":
                        self.pytest.add(alias.asname or "pytest")
            elif isinstance(node, ast.ImportFrom) and node.module == "pytest":
                for alias in node.names:
                    if alias.name == "mark":
                        self.mark.add(alias.asname or "mark")
                    elif alias.name == "xfail":
                        self.xfail.add(alias.asname or "xfail")


def _is_xfail_mark(node: ast.AST, names: _PytestNames) -> bool:
    """An xfail mark or call under any name the module binds (see ``_PytestNames``).

    ``pytest.mark.xfail`` (bare or called), a runtime ``pytest.xfail``, ``mark.xfail``, a
    bare imported ``xfail``, and ``add_marker("xfail")`` / ``applymarker("xfail")``.
    """
    if isinstance(node, ast.Name):
        return node.id in names.xfail
    if isinstance(node, ast.Call):
        func = node.func
        return (
            isinstance(func, ast.Attribute)
            and func.attr in ("add_marker", "applymarker")
            and bool(node.args)
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "xfail"
        )
    if not (isinstance(node, ast.Attribute) and node.attr == "xfail"):
        return False
    owner = node.value
    if isinstance(owner, ast.Name):
        return owner.id in names.pytest or owner.id in names.mark
    return (
        isinstance(owner, ast.Attribute)
        and owner.attr == "mark"
        and isinstance(owner.value, ast.Name)
        and owner.value.id in names.pytest
    )


def _definitions(tree: ast.AST) -> list[tuple[int, int, str, ast.AST]]:
    """``(first line, last line, Class::function, node)`` of every def and class, decorators in."""
    found: list[tuple[int, int, str, ast.AST]] = []

    def visit(node: ast.AST, prefix: tuple[str, ...]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                name = (*prefix, child.name)
                starts = [child.lineno, *(dec.lineno for dec in child.decorator_list)]
                end = child.end_lineno or child.lineno
                found.append((min(starts), end, "::".join(name), child))
                visit(child, name)

    visit(tree, ())
    return found


def _xfail_sites(path: Path) -> set[str]:
    """The innermost def or class enclosing every xfail mark in *path* (``<module>`` if none)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    definitions = _definitions(tree)
    names = _PytestNames(tree)
    sites: set[str] = set()
    for node in ast.walk(tree):
        if not _is_xfail_mark(node, names):
            continue
        line = getattr(node, "lineno", 0)
        enclosing = [
            (end - start, name)
            for start, end, name, _def in definitions
            if start <= line <= end
        ]
        sites.add(min(enclosing)[1] if enclosing else "<module>")
    return sites


def _strict_xfail(definition: ast.AST, names: _PytestNames) -> bool:
    """Whether *definition* is decorated ``pytest.mark.xfail(..., strict=True, ...)``."""
    for decorator in getattr(definition, "decorator_list", ()):
        if isinstance(decorator, ast.Call) and _is_xfail_mark(decorator.func, names):
            for keyword in decorator.keywords:
                if (
                    keyword.arg == "strict"
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value is True
                ):
                    return True
    return False


def test_the_baseline_is_the_measured_record() -> None:
    """Anti-vacuity: every check below runs over these values, so they are pinned as recorded.

    The earlier P34 slot is pinned too: the append protocol leaves it as written, nine
    quarantines on its own head.
    """
    assert _BASELINE["head"] == "60e3e3fd83c59be165f9ae25a20b2a1154af2294"
    assert len(_QUARANTINES) == 0
    assert len(_DATA_DEPENDENT) == 4

    earlier_quarantines = tuple(_EARLIER_BASELINE["xfail_quarantines"])
    assert _EARLIER_BASELINE["head"] == "c75119420f30a38592c59343a32abd02b398c3b5"
    assert len(earlier_quarantines) == len(set(earlier_quarantines)) == 9


def test_the_tripwire_half_resolves_to_the_one_constant_and_its_gate() -> None:
    tripwires = _BASELINE["tripwires"]
    module_name, _sep, attribute = str(tripwires["source"]).rpartition(".")
    constant = getattr(importlib.import_module(module_name), attribute)
    assert len(constant) == len(set(constant)) == tripwires["measured_failed"] == 5, (
        f"{tripwires['source']} holds {len(constant)} tripwires; the baseline recorded "
        f"{tripwires['measured_failed']} failing. A tripwire dropped from the constant is a "
        "disclosure erased from the record."
    )
    gate = ast.parse(PHASE33_GATE.read_text(encoding="utf-8"))
    names = {node.name for node in gate.body if isinstance(node, ast.FunctionDef)}
    assert PHASE33_GATE_TEST in names, (
        f"{PHASE33_GATE.name} no longer defines {PHASE33_GATE_TEST}; nothing would then run the "
        "tripwires to prove they are still red"
    )


def test_every_recorded_quarantine_is_still_a_strict_xfail() -> None:
    """Each recorded quarantine is still a strict xfail.

    Since Phase 38 the recorded set is EMPTY, so this iterates nothing; it stays so that a
    quarantine recorded in a later slot is checked the moment it is recorded. The non-vacuous
    half is ``test_no_xfail_exists_outside_the_recorded_set``, which now asserts zero xfail
    sites under ``tests/``.
    """
    for node_id in _QUARANTINES:
        path, qualname = _site(node_id)
        tree = ast.parse((REPO_ROOT / path).read_text(encoding="utf-8"))
        matches = [
            node for _start, _end, name, node in _definitions(tree) if name == qualname
        ]
        assert matches, f"QUARANTINE ERASED: {node_id} no longer exists"
        assert _strict_xfail(matches[0], _PytestNames(tree)), (
            f"QUARANTINE ERASED: {node_id} is no longer marked xfail(strict=True)"
        )


def test_no_xfail_exists_outside_the_recorded_set(tmp_path: Path) -> None:
    # Control first: the scan sees each form a quarantine can take, attributed to its owner.
    sample = tmp_path / "test_sample.py"
    sample.write_text(
        "import pytest\n"
        "pytestmark = pytest.mark.xfail\n"
        "class TestA:\n"
        "    @pytest.mark.xfail(strict=True, reason='r')\n"
        "    def test_marked(self):\n"
        "        pass\n"
        "def test_runtime():\n"
        "    pytest.xfail('r')\n"
        "def test_clean():\n"
        "    assert 'pytest.mark.xfail'\n",
        encoding="utf-8",
    )
    assert _xfail_sites(sample) == {"<module>", "TestA::test_marked", "test_runtime"}

    # The same forms under the names an import can bind, and a string-named marker.
    aliased = tmp_path / "test_aliased.py"
    aliased.write_text(
        "import pytest as _pytest\n"
        "from pytest import mark as m, xfail as xf\n"
        "@_pytest.mark.xfail\n"
        "def test_aliased_module():\n"
        "    pass\n"
        "@m.xfail(strict=True)\n"
        "def test_aliased_mark():\n"
        "    pass\n"
        "def test_aliased_runtime():\n"
        "    xf('r')\n"
        "def test_string_marker(request):\n"
        "    request.node.add_marker('xfail')\n"
        "def test_clean(request):\n"
        "    request.node.add_marker('slow')\n",
        encoding="utf-8",
    )
    assert _xfail_sites(aliased) == {
        "test_aliased_module",
        "test_aliased_mark",
        "test_aliased_runtime",
        "test_string_marker",
    }

    recorded = {_site(node_id) for node_id in _QUARANTINES}
    found = {
        (path.relative_to(REPO_ROOT).as_posix(), site)
        for path in sorted(TESTS_ROOT.rglob("*.py"))
        if "__pycache__" not in path.parts
        for site in _xfail_sites(path)
    }
    assert found - recorded == set(), (
        "UNRECORDED QUARANTINE: an xfail exists that P34_EXPECTED_FAILURE_BASELINE does not "
        f"enumerate: {sorted(found - recorded)}. Record it in a new tests/phase34_state.py slot."
    )
    assert recorded - found == set(), (
        f"QUARANTINE ERASED: no xfail mark remains at {sorted(recorded - found)}"
    )


def test_every_recorded_node_id_still_collects() -> None:
    """Collection catches renames and lost parametrize ids; no recorded node is RUN here."""
    node_ids = [*_QUARANTINES, *_DATA_DEPENDENT]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-o",
            "addopts=",
            "-q",
            "--collect-only",
            *node_ids,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    output = (result.stdout + result.stderr).replace("\\", "/")
    assert result.returncode == 0, output
    for node_id in node_ids:
        assert node_id in output, (
            f"recorded node id no longer collects: {node_id}. It was renamed, moved or lost "
            f"its parametrize id; re-record it in a new slot, never drop it.\n\n{output}"
        )
