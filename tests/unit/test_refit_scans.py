"""The re-fit's must-NOTs as AST scans (Plan 37-15; RFIT-02, RFIT-04, RFIT-07; must-NOT P1, P2).

Three properties make the 2026 re-fit a clean forward measurement. Each is only intended until a
test fails when it breaks, so each is a scan here.

P1 -- THE TRIGGER AND THE SWAP PATH READ NO PERFORMANCE FIGURE (RFIT-02, R2, D-02)
---------------------------------------------------------------------------------
* ``refit/cadence.py`` (the trigger) imports only the standard library,
  ``backtest.refit_preregistration_2026`` and ``refit.records``, and ``decide`` takes exactly
  ``now_utc, aa_records, date_events``: the frozen lists, the clock and the records.
* ``refit/integrity.py`` and ``refit/swap.py`` (what decides whether and what to swap) and
  ``refit/hook.py`` (the decision path that calls them) import nothing from ``refit.comparison``,
  ``models.clv``, ``forward_ledger.verdict`` or ``api``; ``backtest.diagnose`` only for
  ``score_deployed_artifacts`` (the scores check). The trigger and those three files name no
  identifier -- variable, attribute, function, class, parameter, keyword or imported name --
  containing ``clv``, ``drift``, ``roi``, ``brier``, ``p_value`` or ``comparison``
  (case-insensitive).

P2 -- THE SHADOW PATH NEVER TOUCHES latest.json AND NOTHING WRITES THE PIN (RFIT-04)
----------------------------------------------------------------------------------
* ``forward_ledger/shadow.py`` and ``forward_ledger/rescore.py`` carry no code string naming
  ``latest.json`` (docstrings explaining the rule are allowed), call no manifest resolver or
  writer (``resolve_production_artifacts``, ``_resolve_once``, ``replace_manifest``,
  ``update_manifest``, ``_atomic_write_json``, ``save_model_artifact``), and pass an explicit
  ``version=`` on every scorer and blend load (``score_deployed_artifacts``, ``scorer``,
  ``_bound_spread_slope``, ``MarketBlender.from_artifacts``).
* Across every production package, no code string names the pin's file ``shadow_pin_2026.py``
  (a path an open or write call could use), no ``setattr`` targets the pin module, and no
  assignment or ``del`` targets an attribute of it.

THE P2 SCAN'S LIMIT, STATED PLAINLY. The scan proves no write to the pin's FILE path and no
write to an ATTRIBUTE of the pin MODULE. It does not follow aliases: an item mutation of the id
mapping through a name it was imported under (``ids = SHADOW_MODEL_IDS; ids["wp"] = ...``) is not
tracked here. That case is made impossible structurally instead: ``SHADOW_MODEL_IDS`` is a
``types.MappingProxyType`` (Plan 37-01), so item assignment, ``del``, ``.update()`` and
``.pop()`` fail at run time, which :func:`test_the_pin_mapping_is_read_only` proves.

RFIT-07 -- NO SEARCH IS REACHABLE FROM THE RE-FIT (R15)
-------------------------------------------------------
No module under ``refit/`` imports ``optuna``, ``models.tuning`` or
``config.tuning_preregistration``, calls ``tune_hyperparameters``, ``use_phase30_tuning``,
``use_phase332_tuning``, ``use_backtest_tuning`` or ``OptunaTuner``, or passes ``tune=``
anything but the literal ``False``. (The blend-weight tune, ``backtest.tune.run_blend_tuning``,
is the frozen row-19 procedure the record re-runs per generation, not a hyperparameter search.)

WHY AST, NOT TEXT
-----------------
A docstring or comment that explains a rule ("no CLV ever moves a date") must not trip a scan, and
a name hidden in an f-string or a long line must not slip past one. Every scan reads the parsed
tree: imports, identifiers, calls and keywords, never the raw text.

EVERY SCAN HAS A CONTROL
------------------------
Each scanned set is asserted non-empty and present on disk, so a scan cannot pass by scanning
nothing, and each detector runs over planted in-memory violations that it must flag (the
``tests/unit/test_ledger_never_writes_latest.py`` shape), so a detector that silently stopped
detecting fails here rather than reporting clean code.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# The scan helper: one parsed view of a source (imports, identifiers, calls)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImportedName:
    """One import statement's module and the names it takes from it.

    Attributes:
        lineno: The statement's line.
        module: The absolute module (``a.b`` for ``import a.b`` / ``from a.b import c``).
        names: The names taken (``()`` for a plain ``import``).
    """

    lineno: int
    module: str
    names: tuple[str, ...]

    def candidates(self) -> tuple[str, ...]:
        """Every dotted module this import can reach: the module and each ``module.name``."""
        return (self.module, *(f"{self.module}.{name}" for name in self.names))


def _package_of(filename: str) -> str:
    """The dotted package of a repository-relative path (``refit/hook.py`` -> ``refit``)."""
    return ".".join(Path(filename).with_suffix("").parts[:-1])


def imports_of(tree: ast.AST, filename: str) -> list[ImportedName]:
    """Every import in *tree*, relative imports resolved against *filename*'s package."""
    found: list[ImportedName] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend(
                ImportedName(node.lineno, alias.name, ()) for alias in node.names
            )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if node.level:
                base = _package_of(filename).split(".")
                base = base[: len(base) - (node.level - 1)] if node.level > 1 else base
                module = ".".join(part for part in (*base, module) if part)
            names = tuple(alias.name for alias in node.names)
            found.append(ImportedName(node.lineno, module, names))
    return found


def _spelled_name(node: ast.AST) -> str | None:
    """The identifier *node* itself spells, or None when it spells none."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        return node.name
    if isinstance(node, ast.arg | ast.keyword):
        return node.arg  # None for a ``**kwargs`` keyword
    if isinstance(node, ast.alias):
        return node.asname or node.name
    return None


def identifiers_of(tree: ast.AST) -> list[tuple[int, str]]:
    """Every name the code itself spells: variables, attributes, defs, parameters, keywords.

    String constants (docstrings, comments are not in the tree at all) are never identifiers.
    """
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        name = _spelled_name(node)
        if name is not None:
            found.append((getattr(node, "lineno", 0), name))
    return found


def called_name(func: ast.expr) -> str | None:
    """The bare name a call reaches: ``f`` for ``f(...)`` and ``obj.f(...)``."""
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _reaches(candidate: str, prefix: str) -> bool:
    return candidate == prefix or candidate.startswith(f"{prefix}.")


def _relative(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=_relative(path))


# ---------------------------------------------------------------------------
# P1: the trigger reads only its inputs; the swap path reads no performance figure
# ---------------------------------------------------------------------------

TRIGGER_FILE = REPO_ROOT / "refit" / "cadence.py"
SWAP_PATH_FILES: tuple[Path, ...] = (
    REPO_ROOT / "refit" / "integrity.py",
    REPO_ROOT / "refit" / "swap.py",
    REPO_ROOT / "refit" / "hook.py",
)
P1_FILES: tuple[Path, ...] = (TRIGGER_FILE, *SWAP_PATH_FILES)

TRIGGER_ALLOWED_MODULES: frozenset[str] = frozenset(
    {"backtest.refit_preregistration_2026", "refit.records"}
)
TRIGGER_PARAMETERS: tuple[str, ...] = ("now_utc", "aa_records", "date_events")

METRIC_MODULES: tuple[str, ...] = (
    "refit.comparison",
    "models.clv",
    "forward_ledger.verdict",
    "api",
)
DIAGNOSE_MODULE = "backtest.diagnose"
DIAGNOSE_ALLOWED_NAMES: frozenset[str] = frozenset({"score_deployed_artifacts"})
METRIC_SUBSTRINGS: tuple[str, ...] = (
    "clv",
    "drift",
    "roi",
    "brier",
    "p_value",
    "comparison",
)


def _is_stdlib(module: str) -> bool:
    return module.split(".", maxsplit=1)[0] in sys.stdlib_module_names


def trigger_import_findings(
    source: str, filename: str = "refit/cadence.py"
) -> list[str]:
    """Every import of *source* that is not stdlib, the frozen record or ``refit.records``."""
    findings: list[str] = []
    for imported in imports_of(ast.parse(source, filename=filename), filename):
        if _is_stdlib(imported.module) or imported.module in TRIGGER_ALLOWED_MODULES:
            continue
        if imported.names and all(
            f"{imported.module}.{name}" in TRIGGER_ALLOWED_MODULES
            for name in imported.names
        ):
            continue
        findings.append(
            f"{filename}:{imported.lineno} imports {imported.module} {list(imported.names)}"
        )
    return findings


def metric_findings(source: str, filename: str = "refit/swap.py") -> list[str]:
    """Every import of a metric module, and every identifier naming a performance figure."""
    tree = ast.parse(source, filename=filename)
    findings: list[str] = []
    for imported in imports_of(tree, filename):
        where = f"{filename}:{imported.lineno}"
        for candidate in imported.candidates():
            hit = next((m for m in METRIC_MODULES if _reaches(candidate, m)), None)
            if hit is not None:
                findings.append(f"{where} imports {candidate} (a metric module: {hit})")
                break
        if imported.module == DIAGNOSE_MODULE:
            extra = sorted(set(imported.names) - DIAGNOSE_ALLOWED_NAMES)
            if extra or not imported.names:
                findings.append(
                    f"{where} takes {extra or 'the whole module'} from {DIAGNOSE_MODULE}; "
                    f"only {sorted(DIAGNOSE_ALLOWED_NAMES)} may be imported"
                )
        elif DIAGNOSE_MODULE in imported.candidates()[1:]:
            findings.append(f"{where} imports the whole {DIAGNOSE_MODULE} module")
    for lineno, name in identifiers_of(tree):
        lowered = name.lower()
        hit = next((s for s in METRIC_SUBSTRINGS if s in lowered), None)
        if hit is not None:
            findings.append(f"{filename}:{lineno} names {name!r} (contains {hit!r})")
    return findings


def decide_parameters(source: str) -> tuple[str, ...] | None:
    """The parameter names of the module-level ``decide`` in *source*, or None without one."""
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef) and node.name == "decide":
            arguments = node.args
            every = [*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs]
            if arguments.vararg is not None:
                every.append(arguments.vararg)
            if arguments.kwarg is not None:
                every.append(arguments.kwarg)
            return tuple(argument.arg for argument in every)
    return None


def test_the_scanned_set_is_not_empty() -> None:
    assert len(P1_FILES) == 4
    for path in P1_FILES:
        assert path.is_file(), (
            f"the P1 scan's file {_relative(path)} moved or was removed"
        )
        assert _parse(path).body, f"{_relative(path)} parsed to nothing"


def test_the_trigger_imports_only_its_inputs() -> None:
    source = TRIGGER_FILE.read_text(encoding="utf-8")

    assert trigger_import_findings(source, _relative(TRIGGER_FILE)) == []
    assert decide_parameters(source) == TRIGGER_PARAMETERS, (
        "refit.cadence.decide must take exactly the clock and the two record lists"
    )


def test_the_swap_path_reads_no_performance_figure() -> None:
    findings = [
        finding
        for path in P1_FILES
        for finding in metric_findings(
            path.read_text(encoding="utf-8"), _relative(path)
        )
    ]

    assert findings == [], (
        "the re-fit trigger or swap path can read a performance, drift, CLV or comparison "
        "figure (must-NOT P1):\n" + "\n".join(findings)
    )


@pytest.mark.parametrize(
    "source",
    [
        "from refit.comparison import paired_frame\n",
        "from models.clv import compute_clv_for_predictions\n",
        "x = report.mean_clv\n",
        "def due(now, drift):\n    return now\n",
        "from refit import comparison\n",
        "import api.services\n",
        "from forward_ledger.verdict import declare\n",
        "from backtest.diagnose import score_deployed_artifacts, clv_significance\n",
        "from backtest import diagnose\n",
        "ok = check(p_value=0.04)\n",
        "class BrierGate:\n    pass\n",
    ],
)
def test_the_metric_scan_fires(source: str) -> None:
    assert metric_findings(source), (
        f"the P1 scan missed a planted metric read:\n{source}"
    )


@pytest.mark.parametrize(
    "source",
    [
        '"""No CLV, drift, ROI or Brier figure ever moves a re-fit date."""\n',
        'def due(now):\n    """A comparison p_value never decides this."""\n    return now\n',
        "from backtest.diagnose import score_deployed_artifacts\n",
        "message = 'the clv is report-only'\n",
    ],
)
def test_the_metric_scan_passes_prose_and_the_scores_import(source: str) -> None:
    assert metric_findings(source) == [], source


@pytest.mark.parametrize(
    ("source", "flagged"),
    [
        ("import pandas as pd\n", True),
        ("from refit.integrity import IntegrityRefusal\n", True),
        ("from refit import records, swap\n", True),
        ("from datetime import UTC\nfrom refit import records\n", False),
        ("from backtest.refit_preregistration_2026 import REFIT_DATE_LISTS\n", False),
    ],
)
def test_the_trigger_import_scan_fires(source: str, flagged: bool) -> None:
    assert bool(trigger_import_findings(source)) is flagged, source


def test_the_trigger_parameter_check_fires() -> None:
    planted = (
        "def decide(now_utc, aa_records, date_events, mean_clv):\n    return None\n"
    )

    assert decide_parameters(planted) != TRIGGER_PARAMETERS


# ---------------------------------------------------------------------------
# Shared by P2: the code strings of a source (docstrings excluded)
# ---------------------------------------------------------------------------


def _docstring_ids(tree: ast.AST) -> set[int]:
    """The ``id`` of every module, class and function docstring constant in *tree*."""
    found: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(
            node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef
        ):
            first = node.body[0] if node.body else None
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                found.add(id(first.value))
    return found


def code_strings(tree: ast.AST) -> list[tuple[int, str]]:
    """Every string constant of *tree* that is not a docstring (f-string parts included)."""
    docstrings = _docstring_ids(tree)
    return [
        (node.lineno, node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and id(node) not in docstrings
    ]


# ---------------------------------------------------------------------------
# P2: the shadow path never touches latest.json; nothing writes the pin
# ---------------------------------------------------------------------------

SHADOW_PATH_FILES: tuple[Path, ...] = (
    REPO_ROOT / "forward_ledger" / "shadow.py",
    REPO_ROOT / "forward_ledger" / "rescore.py",
)
MANIFEST_FILE_NAME = "latest.json"
MANIFEST_RESOLVERS_AND_WRITERS: frozenset[str] = frozenset(
    {
        "resolve_production_artifacts",
        "_resolve_once",
        "replace_manifest",
        "update_manifest",
        "_atomic_write_json",
        "save_model_artifact",
    }
)
# Every load the shadow path makes must name its artifact: ``MarketBlender.from_artifacts`` is
# reached as the attribute ``from_artifacts``.
VERSIONED_LOADS: frozenset[str] = frozenset(
    {"score_deployed_artifacts", "scorer", "_bound_spread_slope", "from_artifacts"}
)

PIN_MODULE = "backtest.shadow_pin_2026"
PIN_MODULE_NAME = "shadow_pin_2026"
PIN_FILE_TEXT = "shadow_pin_2026.py"
PIN_SCAN_ROOTS: tuple[str, ...] = (
    "api",
    "backtest",
    "data",
    "features",
    "forward_ledger",
    "models",
    "pipeline",
    "refit",
    "scripts",
    "utils",
)


def shadow_path_findings(
    source: str, filename: str = "forward_ledger/shadow.py"
) -> list[str]:
    """Every way *source* could read or write the production manifest instead of the pin."""
    tree = ast.parse(source, filename=filename)
    findings = [
        f"{filename}:{lineno} carries the code string {text!r}"
        for lineno, text in code_strings(tree)
        if MANIFEST_FILE_NAME in text
    ]
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = called_name(node.func)
        where = f"{filename}:{node.lineno}"
        if name in MANIFEST_RESOLVERS_AND_WRITERS:
            findings.append(f"{where} calls the manifest resolver or writer {name}()")
        elif name in VERSIONED_LOADS and not any(
            keyword.arg == "version" for keyword in node.keywords
        ):
            findings.append(f"{where} calls {name}() without an explicit version=")
    return findings


def _pin_aliases(tree: ast.AST) -> set[str]:
    """The bare names *tree* binds to the pin module (its own name, plus any ``as`` alias)."""
    aliases = {PIN_MODULE_NAME}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            aliases.update(
                alias.asname
                for alias in node.names
                if alias.name == PIN_MODULE and alias.asname
            )
        elif isinstance(node, ast.ImportFrom) and node.module == "backtest":
            aliases.update(
                alias.asname or alias.name
                for alias in node.names
                if alias.name == PIN_MODULE_NAME
            )
    return aliases


def _names_the_pin(node: ast.expr, aliases: set[str]) -> bool:
    """True when *node* is the pin module: a bound alias, or ``<pkg>.shadow_pin_2026``."""
    if isinstance(node, ast.Name):
        return node.id in aliases
    return isinstance(node, ast.Attribute) and node.attr == PIN_MODULE_NAME


def _assignment_targets(node: ast.AST) -> list[ast.expr]:
    if isinstance(node, ast.Assign | ast.Delete):
        return list(node.targets)
    if isinstance(node, ast.AugAssign | ast.AnnAssign):
        return [node.target]
    return []


def pin_write_findings(source: str, filename: str = "scripts/example.py") -> list[str]:
    """Every way *source* could write the pin: its file path, setattr, or attribute assignment."""
    tree = ast.parse(source, filename=filename)
    aliases = _pin_aliases(tree)
    findings = [
        f"{filename}:{lineno} carries the pin's file path in {text!r}"
        for lineno, text in code_strings(tree)
        if PIN_FILE_TEXT in text
    ]
    for node in ast.walk(tree):
        where = f"{filename}:{getattr(node, 'lineno', 0)}"
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "setattr"
            and node.args
            and _names_the_pin(node.args[0], aliases)
        ):
            findings.append(f"{where} calls setattr() on the pin module")
        for target in _assignment_targets(node):
            if any(
                isinstance(part, ast.Attribute) and _names_the_pin(part.value, aliases)
                for part in ast.walk(target)
            ):
                findings.append(f"{where} assigns to an attribute of the pin module")
    return findings


def pin_scan_paths() -> list[Path]:
    """Every production ``.py`` under :data:`PIN_SCAN_ROOTS`, in a stable order."""
    return sorted(
        path
        for root in PIN_SCAN_ROOTS
        for path in (REPO_ROOT / root).rglob("*.py")
        if "__pycache__" not in path.parts
    )


def test_the_p2_scanned_sets_are_not_empty() -> None:
    for path in SHADOW_PATH_FILES:
        assert path.is_file(), f"the shadow-path file {_relative(path)} moved"
    paths = pin_scan_paths()
    for root in PIN_SCAN_ROOTS:
        assert any(path.is_relative_to(REPO_ROOT / root) for path in paths), (
            f"the pin-write scan found no module under {root}/"
        )
    assert REPO_ROOT / "backtest" / PIN_FILE_TEXT in paths
    assert REPO_ROOT / "forward_ledger" / "shadow.py" in paths


def test_the_shadow_path_never_reads_latest_json() -> None:
    findings = [
        finding
        for path in SHADOW_PATH_FILES
        for finding in shadow_path_findings(
            path.read_text(encoding="utf-8"), _relative(path)
        )
    ]

    assert findings == [], (
        "the frozen shadow path can resolve or write artifacts/latest.json, or loads an "
        "artifact without naming it (must-NOT P2):\n" + "\n".join(findings)
    )


def test_nothing_writes_the_pin() -> None:
    findings = [
        finding
        for path in pin_scan_paths()
        for finding in pin_write_findings(
            path.read_text(encoding="utf-8"), _relative(path)
        )
    ]

    assert findings == [], (
        "production code can write the frozen shadow pin (must-NOT P2):\n"
        + "\n".join(findings)
    )


def test_the_pin_mapping_is_read_only() -> None:
    from backtest import shadow_pin_2026
    from forward_ledger.shadow import SHADOW_PIN

    ids = shadow_pin_2026.SHADOW_MODEL_IDS
    before = dict(ids)
    # Typed as Any so the test can attempt the mutations the type checker rightly forbids.
    mutable = cast("Any", ids)

    with pytest.raises(TypeError):
        mutable["wp"] = "x"
    with pytest.raises(TypeError):
        del mutable["wp"]
    with pytest.raises(AttributeError):
        mutable.update(wp="x")
    with pytest.raises(AttributeError):
        mutable.pop("wp")
    assert SHADOW_PIN.model_ids is ids, "the shadow pin must hold the proxy, not a copy"
    with pytest.raises(TypeError):
        cast("Any", SHADOW_PIN.model_ids)["wp"] = "x"

    assert dict(ids) == before
    assert sorted(before) == ["ats", "ou", "wp"]


# ---------------------------------------------------------------------------
# RFIT-07: no search is reachable from the re-fit
# ---------------------------------------------------------------------------

REFIT_PACKAGE = REPO_ROOT / "refit"
SEARCH_MODULES: tuple[str, ...] = (
    "optuna",
    "models.tuning",
    "config.tuning_preregistration",
)
SEARCH_CALLS: frozenset[str] = frozenset(
    {
        "tune_hyperparameters",
        "use_phase30_tuning",
        "use_phase332_tuning",
        "use_backtest_tuning",
        "OptunaTuner",
    }
)


def search_findings(source: str, filename: str = "refit/example.py") -> list[str]:
    """Every import, call or ``tune=`` keyword in *source* that could reach a search."""
    tree = ast.parse(source, filename=filename)
    findings: list[str] = []
    for imported in imports_of(tree, filename):
        for candidate in imported.candidates():
            hit = next((m for m in SEARCH_MODULES if _reaches(candidate, m)), None)
            if hit is not None:
                findings.append(f"{filename}:{imported.lineno} imports {candidate}")
                break
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        where = f"{filename}:{node.lineno}"
        name = called_name(node.func)
        if name in SEARCH_CALLS:
            findings.append(f"{where} calls the search entry point {name}()")
        for keyword in node.keywords:
            if keyword.arg == "tune" and not (
                isinstance(keyword.value, ast.Constant) and keyword.value.value is False
            ):
                findings.append(
                    f"{where} passes tune={ast.unparse(keyword.value)}, not False"
                )
    return findings


def test_no_search_is_reachable_from_refit() -> None:
    paths = sorted(REFIT_PACKAGE.rglob("*.py"))
    assert REFIT_PACKAGE / "training.py" in paths, "the refit package moved"

    findings = [
        finding
        for path in paths
        for finding in search_findings(
            path.read_text(encoding="utf-8"), _relative(path)
        )
    ]

    assert findings == [], (
        "a hyperparameter search is reachable from the re-fit (RFIT-07):\n"
        + "\n".join(findings)
    )


@pytest.mark.parametrize(
    ("detector", "source"),
    [
        (shadow_path_findings, 'json.loads((root / "latest.json").read_text())\n'),
        (
            shadow_path_findings,
            'score_deployed_artifacts("wp", gold_df=g, artifacts_dir=a)\n',
        ),
        (shadow_path_findings, "scorer(target, gold_df=g, artifacts_dir=a)\n"),
        (shadow_path_findings, "MarketBlender.from_artifacts(artifacts_dir)\n"),
        (shadow_path_findings, "ids = resolve_production_artifacts(artifacts_dir)\n"),
        (pin_write_findings, 'setattr(shadow_pin_2026, "SHADOW_BLEND", "x")\n'),
        (pin_write_findings, 'Path("backtest/shadow_pin_2026.py").write_text("")\n'),
        (
            pin_write_findings,
            "import backtest.shadow_pin_2026 as pin\npin.SHADOW_BLEND = 'x'\n",
        ),
        (pin_write_findings, "backtest.shadow_pin_2026.SHADOW_RECIPE = 'x'\n"),
        (search_findings, "import optuna\n"),
        (search_findings, "from models.tuning import OptunaTuner\n"),
        (search_findings, "trainer.tune_hyperparameters(X, y)\n"),
        (search_findings, "train_and_evaluate(f, tune=True)\n"),
        (search_findings, "train_and_evaluate(f, tune=settings.tune)\n"),
    ],
)
def test_the_p2_and_search_scans_fire(detector: Any, source: str) -> None:
    assert detector(source), (
        f"{detector.__name__} missed a planted violation:\n{source}"
    )


@pytest.mark.parametrize(
    ("detector", "source"),
    [
        (shadow_path_findings, '"""It never reads ``latest.json``."""\n'),
        (
            shadow_path_findings,
            "scorer(target, gold_df=g, artifacts_dir=a, version=v)\n",
        ),
        (pin_write_findings, "from backtest.shadow_pin_2026 import SHADOW_BLEND\n"),
        (pin_write_findings, '"""Nothing writes ``shadow_pin_2026.py``."""\n'),
        (search_findings, "train_and_evaluate(f, tune=False)\n"),
        (search_findings, '"""No Optuna study, no tune_hyperparameters."""\n'),
    ],
)
def test_the_p2_and_search_scans_pass_prose_and_explicit_reads(
    detector: Any, source: str
) -> None:
    assert detector(source) == [], source
