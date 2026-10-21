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
