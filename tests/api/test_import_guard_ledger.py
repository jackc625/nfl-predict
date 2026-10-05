"""UIAP-01 SIBLING guard: no module under ``api/`` imports ``forward_ledger`` or ``scripts``.

Why this exists (Phase 34, LDGR-06, D-18). Verifying the forward ledger re-hashes the whole
chain and re-grades every settled row; replaying it re-scores stored decisions. Both are linear
passes over the season, and both are CLIs under ``scripts/`` that the owner runs. If a route
imported either, a page request could start one -- computation in the request path, which
UIAP-01 forbids. The ledger package itself (``forward_ledger``) is likewise reached only by the
daily run and the CLIs; the cache build hands the web layer precomputed tables.

What is scanned. Every ``.py`` under ``api/`` is parsed and every ``import`` / ``from ... import``
node is checked at ANY depth (``ast.walk``), so a lazy import inside a route handler is caught
like a module-level one. Dynamic imports are caught too: a string literal passed to
``importlib.import_module(...)``, a bare ``import_module(...)`` or ``__import__(...)``. Relative
imports resolve inside the ``api`` package and cannot reach either root, so they are skipped.

There is NO allow-list. No exception exists, and none is planned: unlike the inherited
``backtest`` site in ``tests/api/test_import_guard_bets.py``, nothing under ``api/`` has ever
imported these roots.

The negative control parses an IN-MEMORY source string (no file is written) and asserts the
same scanner flags each forbidden form, so the guard cannot pass because it is blind.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
import pathlib

import pytest

_FORBIDDEN_ROOTS = frozenset({"forward_ledger", "scripts"})

# Call targets whose first string argument is a module path.
_DYNAMIC_IMPORT_FUNCTIONS = frozenset({"import_module", "__import__"})


@pytest.fixture()
def api_tree_files() -> list[pathlib.Path]:
    """Every ``*.py`` file under ``api/`` excluding ``__pycache__``; fails fast if empty.

    A guard that walks nothing passes vacuously, so an absent or empty tree is a failure.
    """
    api_dir = pathlib.Path("api")
    assert api_dir.is_dir(), (
        f"Expected api/ directory at {api_dir.resolve()} but it does not exist"
    )
    files = [p for p in api_dir.rglob("*.py") if "__pycache__" not in p.parts]
    assert files, f"api/ tree contains no Python files at {api_dir.resolve()}"
    return files


def _is_forbidden(module_path: str) -> bool:
    return module_path.split(".", maxsplit=1)[0] in _FORBIDDEN_ROOTS


def _dynamic_import_target(node: ast.Call) -> str | None:
    """The string literal a dynamic-import call names, or None if it is not one."""
    func = node.func
    if isinstance(func, ast.Attribute):
        name = func.attr
    elif isinstance(func, ast.Name):
        name = func.id
    else:
        return None
    if name not in _DYNAMIC_IMPORT_FUNCTIONS or not node.args:
        return None
    first = node.args[0]
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        return first.value
    return None


def forbidden_import_sites(source: str, label: str) -> list[str]:
    """Every import of a forbidden root in *source*, as ``label:line description``."""
    sites: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module and _is_forbidden(node.module):
                sites.append(f"{label}:{node.lineno} from {node.module} import ...")
        elif isinstance(node, ast.Import):
            sites.extend(
                f"{label}:{node.lineno} import {alias.name}"
                for alias in node.names
                if _is_forbidden(alias.name)
            )
        elif isinstance(node, ast.Call):
            target = _dynamic_import_target(node)
            if target is not None and _is_forbidden(target):
                sites.append(f"{label}:{node.lineno} dynamic import {target!r}")
    return sites


def test_no_api_module_imports_forward_ledger_or_scripts(
    api_tree_files: list[pathlib.Path],
) -> None:
    """UIAP-01 / D-18: verification and replay are CLIs, never reachable from a route."""
    violations = [
        site
        for path in api_tree_files
        for site in forbidden_import_sites(path.read_text(encoding="utf-8"), str(path))
    ]
    assert not violations, (
        "UIAP-01 / D-18: api/ imports ledger or script code. The verify and replay tools "
        "are CLIs under scripts/ and the request path serves precomputed tables only "
        "(LDGR-06):\n" + "\n".join(violations)
    )


def test_the_scan_covers_the_api_tree_including_the_cache(
    api_tree_files: list[pathlib.Path],
) -> None:
    """The walk is non-empty and reaches ``api/cache.py``, the bet-list persistence layer."""
    keys = {str(path).replace("\\", "/") for path in api_tree_files}
    assert "api/cache.py" in keys, (
        f"api/cache.py was not scanned; walked {sorted(keys)}"
    )


@pytest.mark.parametrize(
    ("source", "expected_fragment"),
    [
        ("from forward_ledger import store\n", "from forward_ledger import"),
        ("import scripts.verify_ledger\n", "import scripts.verify_ledger"),
        (
            "def handler():\n    from forward_ledger.store import read_entries\n",
            "from forward_ledger.store import",
        ),
        (
            "import importlib\nimportlib.import_module('scripts.replay_ledger')\n",
            "dynamic import 'scripts.replay_ledger'",
        ),
        ("__import__('forward_ledger')\n", "dynamic import 'forward_ledger'"),
    ],
)
def test_negative_control_the_scanner_flags_each_forbidden_form(
    source: str, expected_fragment: str
) -> None:
    """The same scanner, on an in-memory source, flags every forbidden import form."""
    sites = forbidden_import_sites(source, "synthetic.py")
    assert len(sites) == 1, sites
    assert expected_fragment in sites[0], sites


def test_control_lookalike_and_relative_imports_are_not_flagged() -> None:
    """No false positive: a root that only STARTS with a forbidden name, or a relative import."""
    source = (
        "import forward_ledger_helpers\n"
        "from scriptsx import thing\n"
        "from .scripts import local\n"
        "from api.cache import connect\n"
        "importlib.import_module('api.services')\n"
    )
    assert forbidden_import_sites(source, "synthetic.py") == []
