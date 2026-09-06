"""EXACTLY ONE weekly recommendation path exists (Phase 31, plan 31-17; SPEC R4, D31-31, T-31-84/85).

WHY THIS GUARD IS COMMITTED RATHER THAN A ONE-TIME SCAN
-------------------------------------------------------
A scan recorded in a summary proves the state on ONE DAY. This repository has its own case of
something passing registration while the real behaviour silently differed (Phase 28's
``combine_features``), so the one-path claim is re-checked on EVERY suite invocation: a
reintroduced second path turns the suite red on the next run rather than being discovered later.

Written as a sibling to ``tests/api/test_import_guard.py`` and in its style -- the fail-fast tree
fixture, the ``ast.walk`` node scan, the explicit allow-list with an EXACT count. That file is
PINNED and is deliberately NOT edited here.

WHY AN IMPORT-GRAPH TEST ALONE IS INSUFFICIENT
-----------------------------------------------
``backtest/simulation.py`` receives a selector by INJECTION -- a duck-typed constructor argument
(``ou_bet_selector`` / ``ats_bet_selector``) with NO import edge to key on. A pure import graph
therefore cannot prove that the Friday step routes through the REAL selector, and it equally cannot
catch a legacy selection rule REIMPLEMENTED INLINE inside a step. That is the concrete mechanism
behind "bypassed rather than gone". So this module asserts three separate things: the step's own
import chain, the absence of a second selection entry point reachable from the pipeline, and the
absence of a write of the retired weekly artifact.

WHY EVERY CHECK KEYS ON MODULE PATH AND NEVER ON A CLASS NAME
--------------------------------------------------------------
The dead v1.0 cluster stays in the tree (SPEC out of scope) and its class has the SAME NAME as the
live one:

    utils/bet_selector.py     class BetSelector  -- DEAD. Confidence/edge-threshold filtering.
    backtest/bet_selector.py  class BetSelector  -- LIVE. The LOCKED-2 single bet-decision source.

``utils/__init__.py`` RE-EXPORTS the dead class into the utilities namespace, and
``backtest/bet_selector.py`` imports ``get_logger`` from that same package -- so importing the LIVE
class always loads the DEAD one as a side effect, in the same process. A name-keyed scan gives a
false result in BOTH directions: it finds the dead class and reports a second path that does not
exist, or it matches the dead class and reports the live path as present when it is not. Every
assertion below therefore compares MODULE PATHS.

Selectors (``-k``): step_imports, facade_chain, only_facade, importers, retired_artifact,
not_vacuous, dead_cluster.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
import pathlib
import re

import pytest

# ---------------------------------------------------------------------------
# The module paths this guard reasons about. NEVER class names -- see the docstring.
# ---------------------------------------------------------------------------

# The analytics package. Selection lives here and nowhere else.
_ANALYTICS_PACKAGE = "backtest"

# The LOCKED-2 single bet-decision source.
_DECISION_ENGINE_MODULE = "backtest.bet_selector"

# The weekly selection FACADE the pipeline is permitted to reach. It is the only module the
# pipeline may import out of the selection set below.
_WEEKLY_FACADE_MODULE = "backtest.weekly_bet_list"

# Every module that can DECIDE a bet. A pipeline module importing any of these other than the
# facade would be a second path into the decision engine.
_SELECTION_MODULES: frozenset[str] = frozenset(
    {
        _DECISION_ENGINE_MODULE,
        "backtest.selector_strategies",
        _WEEKLY_FACADE_MODULE,
    }
)

# The DEAD v1.0 cluster, named so the guard can assert it never appears in a live import chain and
# so a failure message can never point at it. Its class name collides with the live one; its
# MODULE PATH does not.
_DEAD_CLUSTER_MODULES: frozenset[str] = frozenset(
    {"utils.bet_selector", "utils.bet_recommender", "utils.api_recommendation_bridge"}
)

# The step whose internals plan 31-17 replaced.
_RECOMMENDATION_STEP_MODULE = pathlib.Path("pipeline") / "steps.py"
_RECOMMENDATION_STEP_FUNCTION = "step_generate_recommendations"

# The retired weekly artifact's filename prefix (D31-32). Checked over STRING-LITERAL nodes only,
# and never over a bare docstring expression, so prose recording the retirement cannot trip the
# guard while an actual reintroduced write does.
_RETIRED_ARTIFACT_PREFIX = "recommendations_"

# The retired artifact's FILENAME SHAPE: ``recommendations_<season>_week<week>.json``. The prefix
# alone is too broad to be the pattern -- ``utils/alert_manager.py`` carries a legitimate
# ``recommendations_count`` dict key, and failing on that would report a run summary as a
# reintroduced weekly write. The pattern therefore requires the prefix AND the ``week`` segment or
# the ``.json`` suffix, which is the filename and nothing else. F-strings are reconstructed whole
# before matching (see ``_joined_str_text``), so the real write's three-Constant form still hits.
_RETIRED_ARTIFACT_PATTERN = re.compile(
    r"recommendations_.*(?:week|\.json)", re.IGNORECASE
)

# The production packages walked. ``tests`` is excluded on purpose: a test may legitimately name
# the retired artifact in order to assert its ABSENCE, which is what two of them now do.
_PRODUCTION_PACKAGES: tuple[str, ...] = (
    "api",
    "backtest",
    "data",
    "features",
    "models",
    "pipeline",
    "ratings",
    "scripts",
    "utils",
    "web",
)

# The EXACT set of modules permitted to import the decision engine, with its count pinned. Every
# member is inside the analytics package. Exactness matters in both directions: a new importer is
# a new path, and a departed one means this list has become a licence nobody is using.
_DECISION_ENGINE_IMPORTERS: frozenset[str] = frozenset(
    {
        "backtest/ou_monetization.py",
        "backtest/profitability_2025.py",
        "backtest/weekly_bet_list.py",
    }
)


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _key(path: pathlib.Path) -> str:
    """POSIX-normalized repo-relative key so assertions are platform-independent."""
    return str(path).replace("\\", "/")


@pytest.fixture()
def production_files() -> list[pathlib.Path]:
    """Every ``*.py`` under the production packages, excluding ``__pycache__``.

    Fails fast when the walk finds nothing. A guard that scans an empty list passes VACUOUSLY,
    which is the failure mode that makes a control look green while proving nothing.
    """
    files: list[pathlib.Path] = []
    for package in _PRODUCTION_PACKAGES:
        root = pathlib.Path(package)
        if not root.is_dir():
            continue
        files += [p for p in root.rglob("*.py") if "__pycache__" not in p.parts]
    assert files, (
        "the production walk found no Python files; the guard would pass vacuously. "
        f"Packages searched: {list(_PRODUCTION_PACKAGES)}"
    )
    return sorted(files)


def _imported_modules(node: ast.AST) -> set[str]:
    """Every module PATH imported anywhere under *node*, including function-scoped imports.

    ``ast.walk`` recurses into every child, so a lazy import inside a step body is covered. Every
    step adapter in ``pipeline/steps.py`` uses exactly such a deferred import, so a top-level-body
    scan would see nothing at all here.
    """
    modules: set[str] = set()
    for child in ast.walk(node):
        if isinstance(child, ast.ImportFrom):
            if child.module:  # ``from . import x`` has module=None
                modules.add(child.module)
        elif isinstance(child, ast.Import):
            modules.update(alias.name for alias in child.names)
    return modules


def _function_def(path: pathlib.Path, name: str) -> ast.FunctionDef:
    """The named top-level function definition in *path*."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    msg = f"{_key(path)} defines no function named {name!r}"
    raise AssertionError(msg)


def _docstring_expressions(tree: ast.AST) -> set[int]:
    """The ``id()`` of every string Constant that is a bare expression statement.

    A module, class or function docstring -- and any free-standing "comment string" -- is an
    ``ast.Expr`` wrapping a string Constant. It STATES a filename; it cannot WRITE one. Excluding
    exactly these is what lets this file, ``pipeline/steps.py`` and ``AUTOMATION.md``'s companions
    all record the retirement in prose without tripping the literal check.
    """
    return {
        id(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    }


def _joined_str_text(node: ast.JoinedStr) -> str:
    """Reconstruct an f-string as text, with each formatted field rendered as ``{}``.

    REQUIRED, not a nicety. ``f"recommendations_{season}_week{week}.json"`` parses into THREE
    separate string Constants -- ``'recommendations_'``, ``'_week'`` and ``'.json'`` -- so a
    per-Constant scan would never see the whole filename and the narrow pattern below could not
    match the exact shape the retired write had.
    """
    parts: list[str] = []
    for value in node.values:
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            parts.append(value.value)
        else:
            parts.append("{}")
    return "".join(parts)


def _used_string_literals(path: pathlib.Path) -> list[tuple[int, str]]:
    """Every ``(lineno, text)`` string EXPRESSION in *path* that is USED rather than merely stated.

    Comments never reach the AST at all, and bare docstring expressions are filtered out above, so
    what remains is a string an expression actually consumes -- an assignment, an f-string, a call
    argument, a return value. That is the shape an actual reintroduced write has.

    An f-string is reported ONCE, whole; its constituent Constants are not reported separately.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = _docstring_expressions(tree)

    joined: list[tuple[int, str]] = []
    inside_joined: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.JoinedStr):
            joined.append((node.lineno, _joined_str_text(node)))
            inside_joined.update(id(child) for child in ast.walk(node))

    plain = [
        (node.lineno, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
        and id(node) not in inside_joined
    ]
    return sorted(joined + plain)


# ---------------------------------------------------------------------------
# Assertion 1: the step's own import chain, keyed on module path
# ---------------------------------------------------------------------------


def test_the_recommendation_step_imports_the_analytics_selection_facade() -> None:
    """The step body imports ``backtest.weekly_bet_list`` and nothing from the dead cluster."""
    step = _function_def(_RECOMMENDATION_STEP_MODULE, _RECOMMENDATION_STEP_FUNCTION)
    modules = _imported_modules(step)

    assert _WEEKLY_FACADE_MODULE in modules, (
        f"{_key(_RECOMMENDATION_STEP_MODULE)}:{_RECOMMENDATION_STEP_FUNCTION} does not import "
        f"{_WEEKLY_FACADE_MODULE!r}; the weekly recommendation must route through the analytics "
        f"selection facade. Imported modules: {sorted(modules)}"
    )
    intruders = modules & _DEAD_CLUSTER_MODULES
    assert not intruders, (
        f"{_key(_RECOMMENDATION_STEP_MODULE)}:{_RECOMMENDATION_STEP_FUNCTION} imports the DEAD "
        f"v1.0 cluster {sorted(intruders)}. Note the collision: that package's class is ALSO "
        "named BetSelector, so this is caught by module path and could not be caught by name."
    )


def test_the_facade_reaches_the_locked_decision_engine() -> None:
    """The second hop: the facade imports the LOCKED-2 engine, so the chain is complete.

    Asserted separately from the step's own import because the chain is what matters: a facade
    that stopped importing the engine would still satisfy the step-side assertion while deciding
    bets by some other rule.
    """
    facade_path = pathlib.Path(_WEEKLY_FACADE_MODULE.replace(".", "/") + ".py")
    assert facade_path.is_file(), (
        f"the weekly selection facade {_key(facade_path)} does not exist; the one-path chain "
        "cannot be verified"
    )
    modules = _imported_modules(ast.parse(facade_path.read_text(encoding="utf-8")))
    assert _DECISION_ENGINE_MODULE in modules, (
        f"{_key(facade_path)} does not import {_DECISION_ENGINE_MODULE!r}; the weekly path no "
        f"longer reaches the single bet-decision source. Imported: {sorted(modules)}"
    )


# ---------------------------------------------------------------------------
# Assertion 2: no second selection entry point reachable from the pipeline
# ---------------------------------------------------------------------------


def test_the_only_selection_entry_point_reachable_from_the_pipeline_is_the_facade() -> (
    None
):
    """Every ``pipeline/`` module reaches AT MOST the facade out of the selection module set."""
    pipeline_files = [
        p
        for p in pathlib.Path("pipeline").rglob("*.py")
        if "__pycache__" not in p.parts
    ]
    assert pipeline_files, (
        "the pipeline walk found no modules; the guard would pass vacuously"
    )

    offenders: list[str] = []
    for path in pipeline_files:
        modules = _imported_modules(ast.parse(path.read_text(encoding="utf-8")))
        extra = (modules & _SELECTION_MODULES) - {_WEEKLY_FACADE_MODULE}
        offenders += [f"{_key(path)} imports {module}" for module in sorted(extra)]

    assert not offenders, (
        "a SECOND weekly selection path is reachable from the pipeline. The only selection "
        f"module the pipeline may import is {_WEEKLY_FACADE_MODULE!r}; found:\n"
        + "\n".join(offenders)
    )


def test_the_decision_engine_is_imported_only_from_inside_the_analytics_package(
    production_files: list[pathlib.Path],
) -> None:
    """The importers of the decision engine are EXACTLY the pinned analytics set.

    Exact in both directions. A new importer outside the analytics package is a new bet-decision
    path; a new one inside it is new surface that must be declared; and a departed one means this
    list has gone stale, which is how a guard quietly stops guarding.
    """
    importers: set[str] = set()
    for path in production_files:
        modules = _imported_modules(ast.parse(path.read_text(encoding="utf-8")))
        if _DECISION_ENGINE_MODULE in modules:
            importers.add(_key(path))

    outside = {key for key in importers if not key.startswith(_ANALYTICS_PACKAGE + "/")}
    assert not outside, (
        f"{sorted(outside)} import {_DECISION_ENGINE_MODULE!r} from outside the analytics "
        "package; a bet decision made there would be a second path."
    )
    assert importers == set(_DECISION_ENGINE_IMPORTERS), (
        f"the decision engine's importers moved.\n  expected: {sorted(_DECISION_ENGINE_IMPORTERS)}"
        f"\n  found:    {sorted(importers)}\n"
        "A new entry is new surface and must be justified; a missing one means this pin is stale."
    )


# ---------------------------------------------------------------------------
# Assertion 3: the retired weekly artifact is never written again
# ---------------------------------------------------------------------------


def test_no_production_module_writes_the_retired_weekly_artifact(
    production_files: list[pathlib.Path],
) -> None:
    """The retired filename appears in no USED string literal anywhere in the production tree.

    Narrow on purpose (D31-31): the scan covers string-literal nodes and excludes bare docstring
    expressions, so ``pipeline/steps.py``'s own docstring recording the retirement does NOT trip
    it while an f-string building the path WOULD.
    """
    offenders: list[str] = []
    for path in production_files:
        for lineno, value in _used_string_literals(path):
            if _RETIRED_ARTIFACT_PATTERN.search(value):
                offenders.append(f"{_key(path)}:{lineno} {value!r}")

    assert not offenders, (
        f"the retired weekly artifact filename shape {_RETIRED_ARTIFACT_PATTERN.pattern!r} is "
        "used in a string literal again (D31-32 retired it; it had no code consumer):\n"
        + "\n".join(offenders)
    )


def test_a_docstring_naming_the_retired_artifact_does_not_trip_the_literal_check() -> (
    None
):
    """The step's own docstring names the retired artifact, and the guard is silent about it.

    This is the guard's own narrowness, proven rather than asserted: without the docstring
    exclusion, the commit that RECORDS the retirement would fail the check that the retirement
    happened.
    """
    step_source = _RECOMMENDATION_STEP_MODULE.read_text(encoding="utf-8")
    assert _RETIRED_ARTIFACT_PREFIX in step_source, (
        "the step no longer mentions the retired artifact at all, so this test proves nothing "
        "about the exclusion; point it at whichever module now records the retirement"
    )
    used = [
        value
        for _lineno, value in _used_string_literals(_RECOMMENDATION_STEP_MODULE)
        if _RETIRED_ARTIFACT_PATTERN.search(value)
    ]
    assert not used, (
        f"the retired artifact prefix reached a USED string literal in the step: {used}"
    )


# ---------------------------------------------------------------------------
# Assertion 4: the scan is not vacuous, and never blames the dead cluster
# ---------------------------------------------------------------------------


def test_the_scan_reports_a_nonzero_module_count(
    production_files: list[pathlib.Path],
) -> None:
    """A guard that visited nothing passes for the wrong reason."""
    scanned = len(production_files)
    assert scanned > 0, "the one-path scan visited ZERO modules"
    # A floor, not an equality: the tree grows. Well below today's count and well above zero, so
    # it catches a walk that silently stopped recursing without breaking on every new module.
    assert scanned >= 50, (
        f"the one-path scan visited only {scanned} module(s); the production tree is far larger, "
        "so the walk has stopped recursing"
    )


def test_the_dead_cluster_is_still_present_and_still_collides_by_name() -> None:
    """The premise this guard is built on, asserted rather than assumed.

    If the dead cluster were ever removed, the module-path-not-class-name rule would still be
    correct but would no longer be LOAD-BEARING -- and a future reader deleting the care taken
    here would be right to. This test is what tells them the collision is gone.
    """
    dead = pathlib.Path("utils/bet_selector.py")
    live = pathlib.Path("backtest/bet_selector.py")
    if not dead.is_file():
        pytest.skip(
            "the dead v1.0 selector cluster has been removed; the collision is gone"
        )

    def _class_names(path: pathlib.Path) -> set[str]:
        return {
            node.name
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
            if isinstance(node, ast.ClassDef)
        }

    collision = _class_names(dead) & _class_names(live)
    assert collision == {"BetSelector"}, (
        "the dead and live selector modules no longer share a class name; if the collision is "
        "genuinely gone, this guard's module-path rule can be simplified -- but verify first. "
        f"Shared names: {sorted(collision)}"
    )
    # And the dead class is re-exported into the utilities namespace, so it is loaded in the same
    # process as the live one on every run. This is the reason a name-keyed scan is unusable.
    assert "bet_selector" in pathlib.Path("utils/__init__.py").read_text(
        encoding="utf-8"
    )


def test_no_failure_message_in_this_module_names_a_dead_cluster_file() -> None:
    """A guard that blames the dead cluster sends the reader to the wrong file (T-31-85).

    The dead modules are NAMED in this file's constants and prose -- they must be, that is the
    collision being navigated -- but no assertion INTERPOLATES a dead-cluster path as the
    offender. The offenders every message reports come from the scanned tree.
    """
    source = pathlib.Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assert) or node.msg is None:
            continue
        for literal in ast.walk(node.msg):
            if isinstance(literal, ast.Constant) and isinstance(literal.value, str):
                assert "utils/bet_selector.py" not in literal.value, (
                    f"the assertion at line {node.lineno} names a dead-cluster FILE as the "
                    "offender; report the scanned module instead"
                )
