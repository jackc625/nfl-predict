"""The season floor lives in ONE module, and that module imports nothing from this project.

Plan 33.2-18 Task 2 (D33.2-14, SPEC R10). Two subjects, each with its own controls.

THE LEAF-IMPORT PROOF
---------------------
``conf/season_partition.py`` is read by ``conf.settings``, ``models/``, ``backtest/``,
``scripts/`` and ``features/``. It can only be that shared source without an import cycle if it
imports nothing from the project (the module docstring's "WHY THIS MODULE LIVES IN conf/"). The
proof runs in a FRESH interpreter (``sys.executable -c``) that imports ONLY the rule module and
reports every top-level package it resolved. It has to be a fresh process: the identity-binding
assertion elsewhere imports ``features.team_form``, and a check made in a process that has already
loaded ``features`` answers a different question and passes regardless.

``LEAF_IMPORT_FORBIDDEN_PACKAGES`` is the guard's DENYLIST and must never be emptied.
``LEAF_IMPORT_MODULE_COUNT`` is how many top-level packages the fresh process actually resolved --
a non-vacuity witness, so a subprocess that printed nothing cannot pass as "nothing forbidden".

THE FLOOR-LITERAL SCAN
----------------------
No ``*FIRST_SEASON*`` / ``*first_season*`` name is assigned a four-digit year literal anywhere
in the repository except in the rule module. The scan matches the SHAPE over PARSED nodes -- an
``ast.Assign`` OR an ``ast.AnnAssign`` -- never a specific number: a scan for ``2018`` went stale
the moment the value moved. The ``AnnAssign`` arm is required, because
``SELECTION_WINDOW_FIRST_SEASON: int = 2002`` is how the rule module itself declares the constant.
The scanned files are every git-tracked ``.py`` file, so the ``conf`` root is covered and a second
literal placed beside the rule module is visible.

FOUR STRUCTURAL CONTROLS (the shape of ``tests/unit/test_freeze_parse_single_source.py``)
1. NON-VACUITY: the scanned root set equals a declared tuple of known length, and the scan finds
   exactly the legitimate assignments inside ``conf/season_partition.py`` -- so a scanner that
   matches nothing anywhere cannot report a clean tree.
2. THE ASSERTION: nothing is found outside the rule module.
3. TWO PLANTED VIOLATIONS, a bare ``FIRST_SEASON = 2013`` and an annotated
   ``FIRST_SEASON: int = 2013``, both flagged.
4. NO FALSE POSITIVE: a module that READS ``conf.season_partition.SELECTION_WINDOW_FIRST_SEASON``
   is not flagged.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

from conf import season_partition

REPO_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Subject 1: the leaf-import proof
# ---------------------------------------------------------------------------

#: The project packages the rule module must never drag in. The DENYLIST -- never emptied.
LEAF_IMPORT_FORBIDDEN_PACKAGES: frozenset[str] = frozenset(
    {"api", "backtest", "features", "models", "scripts"}
)

_PROBE = (
    "import sys, {module}\n"
    "print(sorted({{name.split('.')[0] for name in sys.modules}}))"
)


def resolve_top_level_packages(module: str) -> frozenset[str]:
    """Import *module* in a FRESH interpreter and return every top-level package it loaded.

    Run from the repository root so the project's packages resolve exactly as they do for the
    suite. ``ast.literal_eval`` reads the printed list; nothing is executed from it.
    """
    completed = subprocess.run(
        [sys.executable, "-c", _PROBE.format(module=module)],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
    )
    return frozenset(ast.literal_eval(completed.stdout.strip()))


_LEAF_IMPORT_PACKAGES: frozenset[str] = resolve_top_level_packages(
    "conf.season_partition"
)

#: How many top-level packages the fresh interpreter resolved for the rule module alone.
LEAF_IMPORT_MODULE_COUNT: int = len(_LEAF_IMPORT_PACKAGES)


class TestTheRuleModuleIsALeaf:
    """Importing the rule module drags in no project package."""

    def test_the_denylist_names_every_heavy_package(self) -> None:
        """The guard is armed: its denylist is non-empty and names all five."""
        assert {"api", "backtest", "features", "models", "scripts"} <= (
            LEAF_IMPORT_FORBIDDEN_PACKAGES
        )

    def test_the_probe_resolved_something(self) -> None:
        """NON-VACUITY: an empty resolution would make the assertion below pass."""
        assert LEAF_IMPORT_MODULE_COUNT > 0
        assert "conf" in _LEAF_IMPORT_PACKAGES

    def test_no_forbidden_package_is_loaded(self) -> None:
        loaded = sorted(_LEAF_IMPORT_PACKAGES & LEAF_IMPORT_FORBIDDEN_PACKAGES)
        assert loaded == [], (
            f"importing conf.season_partition loaded {loaded}. The rule module must import "
            "only the standard library, or conf.settings and every consumer inherit a cycle."
        )

    def test_the_probe_would_catch_a_heavy_import(self) -> None:
        """PLANTED VIOLATION: the same probe, on a module that DOES import features."""
        loaded = resolve_top_level_packages("features.team_form")
        assert "features" in loaded & LEAF_IMPORT_FORBIDDEN_PACKAGES


# ---------------------------------------------------------------------------
# Subject 2: the floor-literal scan
# ---------------------------------------------------------------------------

#: The top-level roots holding git-tracked Python, measured 2026-09-22 (``""`` is the repository
#: root itself). Declared so a new root cannot be scanned -- or skipped -- silently.
SCANNED_ROOTS: tuple[str, ...] = (
    "",
    "api",
    "audit",
    "backtest",
    "conf",
    "config",
    "data",
    "deployment",
    "features",
    "models",
    "pipeline",
    "ratings",
    "scripts",
    "tests",
    "utils",
)

#: The floor assignments that legitimately live in the rule module.
LEGITIMATE_FLOOR_NAMES: tuple[str, ...] = (
    "CORPUS_FIRST_SEASON",
    "SELECTION_WINDOW_FIRST_SEASON",
)


def _assigned_names(node: ast.Assign | ast.AnnAssign) -> list[str]:
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    names: list[str] = []
    for target in targets:
        for sub in ast.walk(target):
            if isinstance(sub, ast.Name):
                names.append(sub.id)
            elif isinstance(sub, ast.Attribute):
                names.append(sub.attr)
    return names


def _is_year_literal(value: ast.expr | None) -> bool:
    return (
        isinstance(value, ast.Constant)
        and type(value.value) is int
        and 1000 <= value.value <= 9999
    )


def floor_literals_in_source(source: str) -> list[tuple[int, str, int]]:
    """``(line, name, year)`` for every floor-shaped assignment in *source*.

    A floor-shaped assignment binds a name containing ``FIRST_SEASON`` or ``first_season`` to a
    four-digit integer literal, through ``=`` or an annotated ``: int =``. A name bound to
    another name (the identity binding in ``features/team_form.py``) is not a literal.
    """
    found: list[tuple[int, str, int]] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Assign | ast.AnnAssign):
            continue
        if not _is_year_literal(node.value):
            continue
        for name in _assigned_names(node):
            if "FIRST_SEASON" in name or "first_season" in name:
                found.append((node.lineno, name, node.value.value))  # type: ignore[union-attr]
    return found


def tracked_python_files() -> list[str]:
    """Every git-tracked ``.py`` path, POSIX-style, relative to the repository root."""
    listing = subprocess.run(
        ["git", "ls-files", "*.py"],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO_ROOT,
    )
    return sorted(path for path in listing.stdout.splitlines() if path)


def _root_of(path: str) -> str:
    return path.split("/", 1)[0] if "/" in path else ""


def scan_repository() -> dict[str, list[tuple[int, str, int]]]:
    """Floor-shaped assignments per tracked file, keeping only files that have any."""
    hits: dict[str, list[tuple[int, str, int]]] = {}
    for path in tracked_python_files():
        found = floor_literals_in_source((REPO_ROOT / path).read_text(encoding="utf-8"))
        if found:
            hits[path] = found
    return hits


class TestNoFloorLiteralOutsideTheRule:
    """Four controls: non-vacuity, the assertion, two planted violations, no false positive."""

    def test_the_scanned_roots_are_the_declared_set(self) -> None:
        """NON-VACUITY, part 1: the files scanned span exactly the declared roots."""
        files = tracked_python_files()
        assert len(files) > 100, len(files)
        assert len(SCANNED_ROOTS) == 15
        assert {_root_of(path) for path in files} == set(SCANNED_ROOTS)
        assert season_partition.PARTITION_RULE_PATH in files

    def test_the_scan_finds_the_rule_modules_own_floors(self) -> None:
        """NON-VACUITY, part 2: a scanner matching nothing cannot report a clean tree."""
        hits = scan_repository()
        in_rule = hits.get(season_partition.PARTITION_RULE_PATH, [])
        assert sorted(name for _, name, _ in in_rule) == sorted(LEGITIMATE_FLOOR_NAMES)
        assert len(in_rule) == 2

    def test_no_floor_literal_exists_outside_the_rule_module(self) -> None:
        """THE ASSERTION."""
        outside = {
            path: found
            for path, found in scan_repository().items()
            if path != season_partition.PARTITION_RULE_PATH
        }
        assert outside == {}, (
            "a *FIRST_SEASON* year literal outside conf/season_partition.py is a second "
            f"declaration of a floor the rule module owns: {outside}. Bind it to "
            "conf.season_partition instead."
        )

    def test_a_planted_bare_assignment_is_flagged(self) -> None:
        """PLANTED VIOLATION 1: ``FIRST_SEASON = 2013``."""
        assert floor_literals_in_source("FIRST_SEASON = 2013\n") == [
            (1, "FIRST_SEASON", 2013)
        ]

    def test_a_planted_annotated_assignment_is_flagged(self) -> None:
        """PLANTED VIOLATION 2: ``FIRST_SEASON: int = 2013`` -- the rule module's own spelling."""
        assert floor_literals_in_source("FIRST_SEASON: int = 2013\n") == [
            (1, "FIRST_SEASON", 2013)
        ]

    def test_a_module_reading_the_rule_is_not_flagged(self) -> None:
        """NO FALSE POSITIVE: reading the floor from the rule is the correct shape."""
        reader = (
            "from conf import season_partition\n"
            "FIRST_SEASON = season_partition.SELECTION_WINDOW_FIRST_SEASON\n"
            "first_season: int = season_partition.CORPUS_FIRST_SEASON\n"
        )
        assert floor_literals_in_source(reader) == []
