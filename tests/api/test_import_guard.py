"""UIAP-01 compliance tests: API must not import model / feature / rating code.

Three layered guards:

1. ``test_api_does_not_import_model_classes`` — the legacy guard that catches
   named model/blender/engine classes anywhere under ``api/``. Complementary
   coverage for class-name aliases that slip past the module-prefix check.
2. ``test_api_does_not_import_forbidden_modules_anywhere`` — walks every AST
   node under ``api/`` and rejects ``import`` / ``from`` statements whose top
   module matches ``models``, ``features`` or ``ratings``. This covers both
   module-level *and* function-scoped (lazy) imports, closing the escape hatch
   where a future developer tries to hide a forbidden import inside a route
   handler.
3. ``test_api_tree_covers_charts_package`` — regression guard that ensures the
   walk covers either the flat ``api/charts.py`` module or the upcoming
   ``api/charts/`` package (Plan 16-02). If the guard silently stops walking
   the package, we want the test to fail.
"""

from __future__ import annotations

import ast
import pathlib
from collections.abc import Iterable

import pytest

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def api_tree_files() -> list[pathlib.Path]:
    """Return every ``*.py`` file under ``api/`` excluding ``__pycache__``.

    Fails fast with a clear error if the ``api/`` directory does not exist —
    downstream tests rely on having *something* to walk.
    """
    api_dir = pathlib.Path("api")
    assert api_dir.is_dir(), (
        f"Expected api/ directory at {api_dir.resolve()} but it does not exist"
    )
    files = [p for p in api_dir.rglob("*.py") if "__pycache__" not in p.parts]
    assert files, f"api/ tree contains no Python files at {api_dir.resolve()}"
    return files


# ---------------------------------------------------------------------------
# Guard 1: named-class imports (legacy coverage)
# ---------------------------------------------------------------------------


def test_api_does_not_import_model_classes() -> None:
    """UIAP-01: API serves precomputed artifacts only -- no model inference."""
    forbidden = {
        "WPTrainer",
        "ATSTrainer",
        "OUTrainer",
        "MarketBlender",
        "BacktestEngine",
        "BettingSimulator",
    }
    api_dir = pathlib.Path("api")
    for py_file in api_dir.rglob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [alias.name for alias in node.names]
                overlap = forbidden.intersection(names)
                assert not overlap, f"{py_file} imports forbidden class: {overlap}"


# ---------------------------------------------------------------------------
# Guard 2: module-prefix imports (module-level AND function-scoped)
# ---------------------------------------------------------------------------


def _iter_import_nodes(tree: ast.AST) -> Iterable[ast.Import | ast.ImportFrom]:
    """Yield every Import / ImportFrom node in *tree*, including those nested
    inside function, async-function, and class bodies.

    ``ast.walk`` already recurses into every child node, so a single pass is
    enough. We keep the helper so the intent is documented and future guards
    can reuse it.
    """
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node


def test_api_does_not_import_forbidden_modules_anywhere(
    api_tree_files: list[pathlib.Path],
) -> None:
    """UIAP-01: No module-level OR function-level import of models / features /
    ratings anywhere under ``api/``.

    Walks every AST node (not just the top-level module body) so lazy imports
    inside route handlers are caught as well.
    """
    forbidden_roots = {"models", "features", "ratings"}
    violations: list[str] = []
    for path in api_tree_files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in _iter_import_nodes(tree):
            if isinstance(node, ast.ImportFrom):
                # ``from . import foo`` has module=None; ignore relative imports.
                if node.module and node.module.split(".")[0] in forbidden_roots:
                    violations.append(
                        f"{path}:{node.lineno} from {node.module} import ...",
                    )
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in forbidden_roots:
                        violations.append(
                            f"{path}:{node.lineno} import {alias.name}",
                        )
    assert not violations, "UIAP-01 violations:\n" + "\n".join(violations)


# ---------------------------------------------------------------------------
# Guard 3: package-layout coverage regression
# ---------------------------------------------------------------------------


def test_api_tree_covers_charts_package(
    api_tree_files: list[pathlib.Path],
) -> None:
    """Regression guard: once ``api/charts/`` lands in Plan 16-02, ensure the
    fixture walks every submodule. Today the flat ``api/charts.py`` file is the
    accepted form; after Plan 02 the accepted form is the package. Either form
    passes — but *neither* form present means the guard is blind.
    """
    paths = {str(p).replace("\\", "/") for p in api_tree_files}
    has_flat = any(p.endswith("api/charts.py") for p in paths)
    has_pkg = any("api/charts/" in p and p.endswith(".py") for p in paths)
    assert has_flat or has_pkg, (
        "Guard must cover either api/charts.py or api/charts/* package; "
        f"walked paths: {sorted(paths)}"
    )
