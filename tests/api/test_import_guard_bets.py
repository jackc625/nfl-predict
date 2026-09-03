"""UIAP-01 SIBLING guard: no NEW ``backtest`` import may appear under ``api/`` (Phase 31, 31-01).

Why a sibling module rather than an edit. SPEC R5 pins ``tests/api/test_import_guard.py`` as
passing UNCHANGED, and that file's ``forbidden_roots`` is ``{"models", "features", "ratings"}`` --
``backtest`` is NOT on it. So R5's acceptance ("the guard passes unchanged, therefore zero
computation in the request path") is TWO claims and the pinned guard proves only the first. A
``/bets`` handler doing ``from backtest.bet_selector import BetSelector`` would pass the pinned
guard untouched while violating UIAP-01's intent outright. Editing the pinned file to close that
hole would destroy the evidence it exists to provide, so the hole is closed HERE.

Inherited debt is NAMED, not silenced. ``api/charts/core.py`` already imports ``backtest`` twice --
a module-level import of two colour constants from the report module, and a lazy in-function import
of an ECE computation helper (the second is a metric COMPUTATION reachable from a chart-render
path). Both predate Phase 31. They are carried in an EXPLICIT allow-list constant with an EXACT
count, so removing one of them or adding a third both fail.

PHASE-31 CONTRACT, stated here so the next author does not have to re-derive it (REVIEW-IMPORT):
**Phase 31 adds ZERO allow-list entries.** The realized-versus-expected tracker is computation, so
``aggregate_by_provenance`` lives in ``backtest/bet_tracker.py`` and is CALLED from
``pipeline/steps.py``, which is already permitted to import ``backtest``. ``api/cache.py`` stays a
pure persistence layer whose tracker entry point, ``materialize_bet_tracker_blocks(conn,
tracker_df)``, takes a PRECOMPUTED frame. An earlier draft of this phase had ``api/cache.py``
import ``backtest.bet_tracker`` directly, which would have made this guard and the Plan 31-13
tracker work mutually unsatisfiable across Wave 1 and Wave 6; the resolution is the pure-persistence
seam, NOT an allow-list widening.

Selectors (``-k``): no_new_backtest_import, allow_list_is_exact.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
import pathlib
from collections.abc import Iterable

import pytest

# ---------------------------------------------------------------------------
# The allow-list: PRE-EXISTING inherited debt, named rather than silenced
# ---------------------------------------------------------------------------
# Maps a POSIX-normalized path under api/ to the EXACT number of permitted top-level-``backtest``
# import nodes in that file. Both entries predate Phase 31:
#
#   api/charts/core.py  module-level  from backtest.report import SEASON_COLORS, TARGET_COLORS
#   api/charts/core.py  lazy, in-fn   from backtest.metrics import _compute_ece
#
# Phase 31 adds NOTHING to this mapping -- see the module docstring for why the tracker seam is a
# pure-persistence handoff instead of an allow-list entry.
_BACKTEST_IMPORT_ALLOW_LIST: dict[str, int] = {
    "api/charts/core.py": 2,
}

_FORBIDDEN_ROOT = "backtest"


# ---------------------------------------------------------------------------
# Shared fixture (copied in SHAPE from tests/api/test_import_guard.py)
# ---------------------------------------------------------------------------


@pytest.fixture()
def api_tree_files() -> list[pathlib.Path]:
    """Return every ``*.py`` file under ``api/`` excluding ``__pycache__``.

    Fails fast if ``api/`` is missing or empty. A guard that walks nothing passes VACUOUSLY, which
    is the failure mode that makes a control look green while proving nothing.
    """
    api_dir = pathlib.Path("api")
    assert api_dir.is_dir(), (
        f"Expected api/ directory at {api_dir.resolve()} but it does not exist"
    )
    files = [p for p in api_dir.rglob("*.py") if "__pycache__" not in p.parts]
    assert files, f"api/ tree contains no Python files at {api_dir.resolve()}"
    return files


def _iter_import_nodes(tree: ast.AST) -> Iterable[ast.Import | ast.ImportFrom]:
    """Yield every Import / ImportFrom node, INCLUDING function- and class-scoped ones.

    ``ast.walk`` recurses into every child node, so a lazy import hidden inside a route handler is
    covered as well as a module-level one. The pre-existing ``api/charts/core.py`` ECE import is
    exactly such a lazy import, which is why the walk (not a top-level-body scan) is required.
    """
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node


def _backtest_import_sites(path: pathlib.Path) -> list[str]:
    """Return a human-readable site string for every ``backtest`` import node in *path*."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    sites: list[str] = []
    for node in _iter_import_nodes(tree):
        if isinstance(node, ast.ImportFrom):
            # ``from . import foo`` has module=None; relative imports cannot reach backtest.
            if node.module and node.module.split(".")[0] == _FORBIDDEN_ROOT:
                sites.append(f"{path}:{node.lineno} from {node.module} import ...")
        else:
            for alias in node.names:
                if alias.name.split(".")[0] == _FORBIDDEN_ROOT:
                    sites.append(f"{path}:{node.lineno} import {alias.name}")
    return sites


def _key(path: pathlib.Path) -> str:
    """POSIX-normalized repo-relative key so the allow-list is platform-independent."""
    return str(path).replace("\\", "/")


# ---------------------------------------------------------------------------
# Guard 1: no NEW backtest import anywhere under api/
# ---------------------------------------------------------------------------


def test_no_new_backtest_import_under_api(api_tree_files: list[pathlib.Path]) -> None:
    """No ``backtest`` import exists under ``api/`` outside the two allow-listed sites.

    This is the guard that makes "zero computation in the request path" checkable rather than
    asserted. Adding ``from backtest.bet_selector import BetSelector`` to ``api/routes/pages.py``,
    or ``from backtest.bet_tracker import aggregate_by_provenance`` to ``api/cache.py``, fails
    here -- which is the point: both were live design options and both are refused.
    """
    violations: list[str] = []
    for path in api_tree_files:
        sites = _backtest_import_sites(path)
        if not sites:
            continue
        permitted = _BACKTEST_IMPORT_ALLOW_LIST.get(_key(path), 0)
        if len(sites) > permitted:
            violations.extend(sites[permitted:] if permitted else sites)

    assert not violations, (
        "UIAP-01: new backtest import(s) under api/ (Phase 31 adds ZERO allow-list entries; "
        "computation belongs in backtest/ and is called from pipeline/steps.py):\n"
        + "\n".join(violations)
    )


# ---------------------------------------------------------------------------
# Guard 2: the allow-list count is EXACT, not a ceiling
# ---------------------------------------------------------------------------


def test_allow_list_count_is_exact(api_tree_files: list[pathlib.Path]) -> None:
    """Each allow-listed file carries EXACTLY its declared number of ``backtest`` import nodes.

    Exactness matters in both directions. A third site in ``api/charts/core.py`` is new surface
    and must fail. But REMOVING one of the two inherited sites must also fail, because the debt is
    then paid and the allow-list entry has become a licence nobody is using -- a stale allowance
    is how a guard quietly stops guarding.
    """
    by_key = {_key(p): p for p in api_tree_files}
    for allowed_path, expected in _BACKTEST_IMPORT_ALLOW_LIST.items():
        assert allowed_path in by_key, (
            f"allow-list names {allowed_path}, which the api/ walk did not find; "
            "the entry is stale or the walk is blind"
        )
        actual = len(_backtest_import_sites(by_key[allowed_path]))
        assert actual == expected, (
            f"{allowed_path} carries {actual} backtest import node(s), allow-list declares "
            f"{expected}. Removing an inherited site means the allow-list entry must go too; "
            "adding one means new surface was opened."
        )


def test_allow_list_names_only_the_inherited_file() -> None:
    """The allow-list names ``api/charts/core.py`` and NOTHING else (the Phase-31 contract)."""
    assert set(_BACKTEST_IMPORT_ALLOW_LIST) == {"api/charts/core.py"}
    assert _BACKTEST_IMPORT_ALLOW_LIST["api/charts/core.py"] == 2
