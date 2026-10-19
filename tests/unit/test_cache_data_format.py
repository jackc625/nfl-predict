"""The cache's contract, writer side (Phase 36, Plan 36-01).

What this file pins, and why:

* ONE SPELLING PER WRITER CONSTANT (D-11, research PITFALLS E2). The request path reads the cache
  writer's constants and parsers, but must never import the writer (``api/cache.py`` loads numpy,
  pandas and scipy). The names moved to the standard-library leaf ``api/cache_schema.py`` and the
  writer imports them back, so ``api.cache.X is api.cache_schema.X`` for every moved name. A second
  definition would be two spellings that can drift.
* THE SERVICE NO LONGER REACHES THE WRITER. ``api.services`` and ``api.routes.pages`` import the
  leaf, never ``api.cache``, and a fresh import of ``api.services`` leaves ``api.cache`` unloaded.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import api.cache
from api import cache_schema

REPO_ROOT = Path(__file__).resolve().parents[2]

# Every name Plan 36-01 moved out of api/cache.py into api/cache_schema.py: the 23 the request path
# imports, their dependents, and the block-mates of the two closed vocabularies they belong to.
MOVED_NAMES: tuple[str, ...] = (
    "ARM_LIVE",
    "ARM_SHADOW",
    "ARMS",
    "BET_STATUS_LIVE",
    "BET_LIST_IMMUTABLE_COLUMNS",
    "BET_LIST_GRADING_COLUMNS",
    "BET_LIST_FILL_COLUMNS",
    "BET_LIST_CLOSING_COLUMNS",
    "BET_LIST_COLUMNS",
    "BET_GRADED_OUTCOMES_COLUMNS",
    "BET_LIST_CORRECTIONS_COLUMNS",
    "BET_TRACKER_BLOCK_COLUMNS",
    "BET_WEEK_RESULTS_COLUMNS",
    "BET_WEEK_CLASS_ORDER",
    "PROVENANCE_BACKTEST_REPLAY",
    "PROVENANCE_FORWARD",
    "VALIDATION_TYPE_CONTAMINATED",
    "VALIDATION_TYPE_CLEAN_HOLDOUT",
    "VALIDATION_TYPE_FORWARD_REALIZED",
    "VALIDATION_TYPE_PRE_VERDICT",
    "_REPLAY_CONTAMINATED_SEASONS",
    "_REPLAY_CLEAN_HOLDOUT_SEASON",
    "REPLAY_VALIDATION_TYPE_SEASONS",
    "BETTING_SIM_PROVENANCE_KEY",
    "BETTING_SIM_VALIDATION_TYPE_KEY",
    "FORWARD_VERDICT_DECLARED_KEY",
    "FORWARD_VERDICT_SEASON_KEY",
    "FORWARD_VERDICT_START_WEEK_KEY",
    "PREDICTIONS_TABLE_COLUMNS",
    "BET_LIST_POPULATED_AT_KEY",
    "BET_LIST_POPULATED_AT_SEPARATOR",
    "bet_list_populated_at_key",
    "CURRENT_SLATE_KEY",
    "_CURRENT_SLATE_SEPARATOR",
    "format_current_slate",
    "parse_current_slate",
    "BACKTEST_SEASON_RANGE_KEY",
    "BETTING_SEASON_RANGE_KEY",
    "NO_SEASONS",
    "format_season_range",
    "parse_season_range",
)


def _fresh_interpreter(probe: str) -> list[str]:
    """Run *probe* in a fresh interpreter at the repo root and return the JSON list it prints.

    Not ``-I``: isolated mode drops the working directory from ``sys.path``, so ``api`` would not
    import at all.
    """
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_every_moved_name_has_one_spelling() -> None:
    """Each moved name is ONE object, reachable from both modules (T-36-01)."""
    for name in MOVED_NAMES:
        assert getattr(api.cache, name) is getattr(cache_schema, name), (
            f"api.cache.{name} is not api.cache_schema.{name}: the name has two definitions"
        )


def test_the_leaf_and_the_service_load_no_writer() -> None:
    """The leaf imports only the standard library; the service never loads the writer.

    The leaf is checked on its own import statements, because a fresh ``import api.cache_schema``
    still runs ``api/__init__.py``, which builds the whole app until Plan 36-03 empties it. The
    service is checked by what a fresh interpreter actually loads.
    """
    leaf = REPO_ROOT / "api" / "cache_schema.py"
    roots: set[str] = set()
    for node in ast.walk(ast.parse(leaf.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "the leaf must not import a sibling module"
            roots.add((node.module or "").split(".")[0])
    outside = sorted(roots - set(sys.stdlib_module_names))
    assert roots, "the leaf's imports were not read"
    assert outside == [], (
        f"api/cache_schema.py imports outside the standard library: {outside}"
    )

    for path in ("api/services.py", "api/routes/pages.py"):
        tree = ast.parse((REPO_ROOT / path).read_text(encoding="utf-8"))
        writer_imports = [
            node.lineno
            for node in ast.walk(tree)
            if (isinstance(node, ast.ImportFrom) and node.module == "api.cache")
            or (
                isinstance(node, ast.Import)
                and any(alias.name == "api.cache" for alias in node.names)
            )
        ]
        assert writer_imports == [], (
            f"{path} still imports api.cache at {writer_imports}"
        )

    loaded = _fresh_interpreter(
        "import json, sys, api.services; print(json.dumps(sorted(sys.modules)))"
    )
    assert "api.services" in loaded
    assert "api.cache" not in loaded, (
        "a fresh import of api.services loaded the cache writer"
    )
