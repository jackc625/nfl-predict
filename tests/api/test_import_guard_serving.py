"""UIAP-01 SIBLING guard: the request path loads none of the scientific stack (Phase 36, SPEC R1).

What it guards. The hosted dashboard installs only ``requirements.txt``, so a page request that
reached pandas, scipy, scikit-learn, plotly or ``backtest`` would crash on Vercel, and would mean
computation had crept into the request path. This file strengthens UIAP-01 without bending it: the
pinned ``tests/api/test_import_guard.py`` and its two siblings stay as they are.

How it differs from the three source-scan guards. Those scan every file under ``api/`` for a few
named roots. This one checks what ACTUALLY loads: a fresh interpreter imports ``api.main`` and
reports every module it loaded. Those modules are the request-serving set. Then it walks the set
statically -- module-level AND function-body imports, skipping only ``if TYPE_CHECKING:`` blocks --
and follows function-body imports of project modules transitively, because an import inside a
function never runs at load time and a load-time check alone would miss it. Roots are matched
exactly, so ``pandas_helpers`` is not ``pandas``.

The one allowance. ``utils/date_utils.py`` is on the request path for ``ET`` and
``ensure_utc_aware``. Two of its OTHER functions import lazily: ``get_current_nfl_week`` reaches
``utils.current_slate`` (and through it pandas), and ``kickoff_wall_clock_et`` imports pandas. No
other serving module calls either function, which a test below asserts, so those two edges are
allowed with exact counts: a third edge, or one of them disappearing, fails.

The must-NOT scan (SPEC R1/R4/R8/R10). No serving module imports an HTTP client or opens a file
for writing.

The hosted list. Every ``requirements.txt`` pin equals ``uv.lock``, and no scientific package is
listed.

What this proves and what it does not. It proves the SOURCE closure: what this code imports. What
Vercel actually installed is proven by the build log Plan 36-09 records.

The run-time check (review WR-06). A library can import a package at RUN time that no project
``import`` statement names: DuckDB imports ``pytz`` to hand back a TIMESTAMPTZ, which is how the
first real push served ``/``, ``/bets`` and ``/fragments/games`` as 500 on the host (Plan 36-10).
So one test serves every route of the live check's own list from a fresh interpreter in which the
packages the host lacks cannot be imported, and requires each to answer as the live check expects.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from collections.abc import Collection, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# SPEC R1's five roots: none may be loaded or imported by the request path.
FORBIDDEN_ROOTS = frozenset({"pandas", "scipy", "sklearn", "plotly", "backtest"})

# Not named by SPEC R1, but absent from the hosted list, so the request path must not load them.
NOT_HOSTED_ROOTS = frozenset({"numpy", "pyarrow"})

# POSIX path -> {imported module: exact number of import nodes naming it}. The two
# function-body edges of utils/date_utils.py that no serving module calls (module docstring).
ALLOW_LIST: dict[str, dict[str, int]] = {
    "utils/date_utils.py": {"utils.current_slate": 1, "pandas": 1},
}

HTTP_CLIENT_ROOTS = frozenset({"httpx", "requests", "aiohttp", "urllib3"})
HTTP_CLIENT_MODULES = frozenset({"urllib.request", "http.client"})

# The two date_utils functions whose lazy imports the allow-list permits.
_ALLOWED_EDGE_FUNCTIONS = frozenset({"get_current_nfl_week", "kickoff_wall_clock_et"})

# Packages the hosted list must never name (PEP 503 names).
_SCIENTIFIC_DISTRIBUTIONS = frozenset(
    {"pandas", "scipy", "scikit-learn", "plotly", "numpy", "pyarrow", "joblib"}
)

_WRITE_MODE_CHARS = frozenset("wax+")
_MODE_LITERAL = re.compile(r"[rwxabtU+]+")
_PIN_LINE = re.compile(r"([A-Za-z0-9][A-Za-z0-9._-]*)==([A-Za-z0-9.+!_-]+)")

# Run in a fresh interpreter at the repo root: import the app, then report every top-level module
# name and every loaded module whose file is a project file (under the repo, outside .venv).
PROBE = """
import json, os, sys
from pathlib import Path
import api.main
root = Path(os.getcwd()).resolve()
modules = {}
for name, module in list(sys.modules.items()):
    file = getattr(module, "__file__", None)
    if not file:
        continue
    try:
        rel = Path(file).resolve().relative_to(root)
    except ValueError:
        continue
    if rel.parts and rel.parts[0] != ".venv":
        modules[name] = rel.as_posix()
print(json.dumps({"top_level": sorted({n.split(".")[0] for n in sys.modules}),
                  "modules": modules}))
"""


# The packages the hosted copy does not install, made unimportable for the run-time check.
HOSTED_MISSING = ("pandas", "numpy", "pyarrow", "pytz", "scipy", "plotly", "sklearn")

# Serve every path in argv[2] from the cache in argv[1], with every package in argv[3] blocked
# before anything is imported, as the hosted copy would (VERCEL=1 is set by the caller). Prints
# each answer's status, the two headers the live check reads, and its body, as one JSON line.
HOSTED_PROBE = """
import json, sys
for name in json.loads(sys.argv[3]):
    sys.modules[name] = None
from pathlib import Path
import api.dependencies as deps
import api.routes.health as health
deps.DB_PATH = health.DB_PATH = Path(sys.argv[1])
from fastapi.testclient import TestClient
from api.main import app
answers = []
with TestClient(app, raise_server_exceptions=False) as client:
    for path in json.loads(sys.argv[2]):
        response = client.get(path, follow_redirects=False)
        headers = {k: v for k, v in response.headers.items() if k in ("content-type", "location")}
        answers.append({"path": path, "status": response.status_code, "headers": headers,
                        "body": response.text})
print(json.dumps(answers))
"""


@dataclass(frozen=True)
class ServingSet:
    """What a fresh ``import api.main`` loaded."""

    top_level: frozenset[str]
    modules: dict[str, str]  # module name -> POSIX path relative to the repo root


def _key(path: Path | str) -> str:
    """POSIX-normalized relative key so the allow-list is platform-independent."""
    return str(path).replace("\\", "/")


def _root(module_name: str) -> str:
    return module_name.split(".", maxsplit=1)[0]


def measure_serving_set() -> ServingSet:
    """Import ``api.main`` in a fresh interpreter and report what it loaded.

    Not ``-I``: isolated mode drops the working directory from ``sys.path``, so ``api`` would not
    import at all.
    """
    completed = subprocess.run(
        [sys.executable, "-c", PROBE],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    return ServingSet(frozenset(payload["top_level"]), dict(payload["modules"]))


def serving_set_problems(modules: Collection[str]) -> list[str]:
    """Why a module set cannot stand for the request path (a guard over nothing proves nothing)."""
    problems: list[str] = []
    if not modules:
        problems.append("the serving set is empty")
    if "api.main" not in modules:
        problems.append("the serving set lacks api.main")
    return problems


def _is_type_checking(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _iter_nodes(
    node: ast.AST, *, in_function: bool = False
) -> Iterator[tuple[ast.AST, bool]]:
    """Yield every node below *node* with whether it sits in a function, skipping TYPE_CHECKING."""
    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.If) and _is_type_checking(child.test):
            for sub in (
                child.orelse
            ):  # the TYPE_CHECKING body never runs; its else branch does
                yield sub, in_function
                yield from _iter_nodes(sub, in_function=in_function)
            continue
        inside = in_function or isinstance(
            child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
        )
        yield child, in_function
        yield from _iter_nodes(child, in_function=inside)


def iter_imports(tree: ast.AST) -> Iterator[tuple[ast.Import | ast.ImportFrom, bool]]:
    """Yield every Import / ImportFrom with whether it sits inside a function body.

    Module-level and function-body imports are both yielded; imports under ``if TYPE_CHECKING:``
    are skipped, because they never run (the lazy ``utils/__init__.py`` relies on that).
    """
    for node, in_function in _iter_nodes(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node, in_function


def _imported_names(
    node: ast.Import | ast.ImportFrom, module_name: str, module_path: str
) -> list[str]:
    """The absolute module names an import node names (``from a import b`` also yields ``a.b``)."""
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if node.level:
        parts = module_name.split(".")
        if not module_path.endswith("__init__.py"):
            parts = parts[:-1]  # a plain module's package is its parent
        parts = parts[: len(parts) - (node.level - 1)]
        base = ".".join([*parts, node.module] if node.module else parts)
    else:
        base = node.module or ""
    return [
        base,
        *(f"{base}.{alias.name}" for alias in node.names if alias.name != "*"),
    ]


def _project_file(module_name: str, search_root: Path) -> str | None:
    """The POSIX path of a project module under *search_root*, or None if it is not one."""
    parts = module_name.split(".")
    if not parts[0] or parts[0] == ".venv":
        return None
    for candidate in (Path(*parts).with_suffix(".py"), Path(*parts, "__init__.py")):
        if (search_root / candidate).is_file():
            return candidate.as_posix()
    return None


def forbidden_imports(
    modules: Mapping[str, str],
    *,
    search_root: Path,
    allow_list: Mapping[str, Mapping[str, int]] = ALLOW_LIST,
) -> list[str]:
    """Every forbidden import reachable from *modules*, walking function bodies transitively.

    *modules* maps a module name to its POSIX path under *search_root*. Each module is parsed and
    every import (module level and function body, outside ``if TYPE_CHECKING:``) is checked by
    EXACT root against the forbidden and not-hosted roots. A project module named by an import and
    not yet visited is walked too, so a function-body import of a project module is followed. An
    allow-listed file must carry exactly its declared count of each allowed edge; an allowed edge
    is neither flagged nor followed.
    """
    findings: list[str] = []
    blocked = FORBIDDEN_ROOTS | NOT_HOSTED_ROOTS
    queue = list(modules.items())
    visited = set(modules)
    while queue:
        module_name, module_path = queue.pop()
        key = _key(module_path)
        allowed = allow_list.get(key, {})
        allowed_counts = dict.fromkeys(allowed, 0)
        tree = ast.parse((search_root / module_path).read_text(encoding="utf-8"))
        for node, in_function in iter_imports(tree):
            names = _imported_names(node, module_name, module_path)
            if names[0] in allowed:
                allowed_counts[names[0]] += 1
                continue
            where = "function body" if in_function else "module level"
            for name in names:
                if _root(name) in blocked:
                    findings.append(f"{key}:{node.lineno} ({where}) imports {name}")
                    break
            for name in names:
                path = _project_file(name, search_root)
                if path is not None and name not in visited:
                    visited.add(name)
                    queue.append((name, path))
        for name, expected in allowed.items():
            if allowed_counts[name] != expected:
                findings.append(
                    f"{key}: allow-list declares {expected} import(s) of {name}, found "
                    f"{allowed_counts[name]} (a removed edge must leave the list; a new one is new surface)"
                )
    return findings


def _mode_writes(call: ast.Call) -> bool:
    """Whether an ``open`` call carries a literal mode that writes (w, a, x or +)."""
    candidates = [*call.args[:2], *(k.value for k in call.keywords if k.arg == "mode")]
    for arg in candidates:
        if (
            isinstance(arg, ast.Constant)
            and isinstance(arg.value, str)
            and _MODE_LITERAL.fullmatch(arg.value)
            and _WRITE_MODE_CHARS & set(arg.value)
        ):
            return True
    return False


def must_not_findings(
    modules: Mapping[str, str], *, search_root: Path = REPO_ROOT
) -> list[str]:
    """HTTP-client imports and file writes in the serving modules (SPEC must-NOT)."""
    findings: list[str] = []
    for module_name, module_path in modules.items():
        key = _key(module_path)
        tree = ast.parse((search_root / module_path).read_text(encoding="utf-8"))
        for node, _in_function in iter_imports(tree):
            for name in _imported_names(node, module_name, module_path):
                if _root(name) in HTTP_CLIENT_ROOTS or name in HTTP_CLIENT_MODULES:
                    findings.append(f"{key}:{node.lineno} imports HTTP client {name}")
                    break
        for node, _in_function in _iter_nodes(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = (
                func.id
                if isinstance(func, ast.Name)
                else func.attr
                if isinstance(func, ast.Attribute)
                else ""
            )
            if name == "open" and _mode_writes(node):
                findings.append(f"{key}:{node.lineno} opens a file for writing")
            elif name in ("write_text", "write_bytes") and isinstance(
                func, ast.Attribute
            ):
                findings.append(f"{key}:{node.lineno} calls .{name}(")
    return findings


def _normalize(name: str) -> str:
    """PEP 503 normalized distribution name."""
    return re.sub(r"[-_.]+", "-", name).lower()


def read_requirements(path: Path) -> dict[str, str]:
    """Parse a hosted requirements file of ``name==version`` lines (comments and blanks aside)."""
    pins: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = _PIN_LINE.fullmatch(stripped)
        if match is None:
            raise ValueError(f"not an exact name==version pin: {stripped!r}")
        pins[_normalize(match.group(1))] = match.group(2)
    return pins


def lock_versions() -> dict[str, str]:
    """Every package version in ``uv.lock``, keyed by PEP 503 normalized name."""
    with (REPO_ROOT / "uv.lock").open("rb") as handle:
        lock = tomllib.load(handle)
    return {_normalize(p["name"]): p["version"] for p in lock["package"]}


# ---------------------------------------------------------------------------
# The measured serving set (one fresh interpreter for the module)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def serving() -> ServingSet:
    return measure_serving_set()


def test_a_fresh_import_of_api_main_loads_no_forbidden_root(
    serving: ServingSet,
) -> None:
    """SPEC R1 acceptance 1: none of the five roots (nor numpy, pyarrow) is in sys.modules."""
    loaded = serving.top_level & (FORBIDDEN_ROOTS | NOT_HOSTED_ROOTS)
    assert not loaded, f"a fresh import of api.main loaded {sorted(loaded)}"


def test_the_serving_set_must_be_real(serving: ServingSet) -> None:
    """The measured set is the app's, and an empty or app-less set is refused (R1 edge: empty)."""
    assert serving_set_problems(serving.modules) == []
    assert len(serving.modules) > 10, sorted(serving.modules)
    assert serving_set_problems(set())
    assert serving_set_problems({"api.dependencies": ...})


def test_the_static_walk_finds_no_forbidden_import(serving: ServingSet) -> None:
    """Module-level and function-body imports of the serving set reach no forbidden root."""
    assert "utils/date_utils.py" in {_key(p) for p in serving.modules.values()}, (
        "the allow-listed file left the serving set; its allow-list entry is stale"
    )
    assert forbidden_imports(serving.modules, search_root=REPO_ROOT) == []

    callers: list[str] = []
    for module_path in serving.modules.values():
        if _key(module_path) == "utils/date_utils.py":
            continue
        tree = ast.parse((REPO_ROOT / module_path).read_text(encoding="utf-8"))
        for node, _in_function in _iter_nodes(tree):
            names: list[str] = []
            if isinstance(node, ast.Name):
                names = [node.id]
            elif isinstance(node, ast.Attribute):
                names = [node.attr]
            elif isinstance(node, ast.ImportFrom):
                names = [alias.name for alias in node.names]
            if _ALLOWED_EDGE_FUNCTIONS & set(names):
                callers.append(f"{_key(module_path)}:{getattr(node, 'lineno', '?')}")
    assert not callers, (
        "a serving module references a date_utils function whose lazy import is only allowed "
        f"because nothing on the request path calls it: {callers}"
    )


_PLANTED: list[tuple[str, str, str | None]] = [
    ("module_level", "import pandas\n", "forbidden"),
    ("function_body", "def f():\n    import scipy\n", "forbidden"),
    ("lookalike", "import pandas_helpers\n", None),
    (
        "type_checking_only",
        "from typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import pandas\n",
        None,
    ),
    ("http_client", "import httpx\n", "must_not"),
    ("write_mode_open", "def f(p):\n    open(p, 'w')\n", "must_not"),
]


@pytest.mark.parametrize(
    ("source", "expected"),
    [pytest.param(source, expected, id=label) for label, source, expected in _PLANTED],
)
def test_planted_violations_are_caught(
    tmp_path: Path, source: str, expected: str | None
) -> None:
    """SPEC R1 acceptances 2 and 3: planted imports are flagged; lookalikes and typing-only are not."""
    package = tmp_path / "planted"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "mod.py").write_text(source, encoding="utf-8")
    modules = {"planted.mod": "planted/mod.py"}

    forbidden = forbidden_imports(modules, search_root=tmp_path)
    must_not = must_not_findings(modules, search_root=tmp_path)
    assert bool(forbidden) == (expected == "forbidden"), forbidden
    assert bool(must_not) == (expected == "must_not"), must_not


def test_no_serving_module_imports_an_http_client_or_writes_a_file(
    serving: ServingSet,
) -> None:
    """SPEC must-NOT: no HTTP client and no write-mode open in the request-serving modules."""
    assert must_not_findings(serving.modules) == []


def _hosted_cache(test_db: Path, served: Path) -> None:
    """The fixture cache plus one scheduled bet week, so ``/bets`` reads its TIMESTAMPTZ locks."""
    import duckdb
    import pandas as pd

    from api.cache import materialize_available_bet_weeks, materialize_bet_week_freeze

    shutil.copyfile(test_db, served)
    lock = pd.Timestamp("2024-09-04 22:00", tz="UTC")
    schedule = pd.DataFrame(
        [
            {"game_id": game_id, "season": 2024, "week": 1, "game_freeze_ts": lock}
            for game_id in ("2024_W01_BUF@KC", "2024_W01_PHI@GB")
        ]
    )
    conn = duckdb.connect(str(served))
    try:
        materialize_available_bet_weeks(conn, schedule)
        materialize_bet_week_freeze(conn, schedule)
    finally:
        conn.close()


def test_every_route_answers_without_the_packages_the_host_lacks(
    test_db: Path, tmp_path: Path
) -> None:
    """Every route of the live check passes with pandas, numpy, pyarrow, pytz, ... unimportable."""
    import httpx

    from scripts.fetch_hosted_routes import check_route, routes

    served = tmp_path / "web_cache.duckdb"
    _hosted_cache(test_db, served)
    expected_health = (
        "test-publish-id",
        hashlib.sha256(served.read_bytes()).hexdigest(),
    )
    route_list = routes("2024_W01_BUF@KC")

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            HOSTED_PROBE,
            str(served),
            json.dumps([route.path for route in route_list]),
            json.dumps(HOSTED_MISSING),
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "VERCEL": "1"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    answers = json.loads(completed.stdout.strip().splitlines()[-1])

    failures = []
    for route, answer in zip(route_list, answers, strict=True):
        response = httpx.Response(
            answer["status"], headers=answer["headers"], text=answer["body"]
        )
        if not check_route(route, response, expected_health):
            failures.append(f"{route.path} -> {answer['status']}")
    assert failures == [], (
        "these routes do not answer on a copy without the packages the host lacks (a library "
        f"imported one at run time): {failures}"
    )


def test_requirements_pins_equal_uv_lock_and_name_no_scientific_package() -> None:
    """SPEC R1 acceptance 4 / D-12: exact uv.lock pins, and no scientific package."""
    pins = read_requirements(REPO_ROOT / "requirements.txt")
    locked = lock_versions()
    assert pins, "requirements.txt names no package"
    assert {
        name: version for name, version in pins.items() if locked.get(name) != version
    } == {}
    assert not set(pins) & _SCIENTIFIC_DISTRIBUTIONS, sorted(
        set(pins) & _SCIENTIFIC_DISTRIBUTIONS
    )
