"""No ledger code path writes ``artifacts/latest.json`` (Plan 34-05, SPEC must-not, LDGR-03).

``latest.json`` is the ONE production swap surface. Only ``models.artifacts.update_manifest`` and
``replace_manifest`` may change it, and they are reached through the promotion tools -- never from
the forward ledger. The ledger path READS the manifest once per decision
(``models.artifacts.resolve_production_artifacts``) and must never write it, or a ledger run
could swap the models that score the next decision.

HOW THE SCAN WORKS
------------------
An ``ast`` walk over every ``.py`` under ``forward_ledger/``, the ledger CLIs under ``scripts/``,
and the daily-run modules the ledger path runs through (``scripts/daily_lock_pipeline.py``,
``pipeline/daily_steps.py``, ``backtest/weekly_bet_list.py``). It flags:

* a call to any of the manifest writers -- ``update_manifest``, ``replace_manifest``,
  ``_atomic_write_json``, ``save_model_artifact`` -- by bare name or as an attribute;
* a write-mode ``open`` (mode carrying ``w``, ``a``, ``x`` or ``+``), a ``write_text``, a
  ``write_bytes``, an ``os.replace`` / ``rename`` / ``Path.replace``, a ``shutil.copy*`` /
  ``move``, or a ``json.dump`` that names the manifest file -- literally, or through a name the
  module binds to it (``path = root / "latest.json"``, a temp derived from it, a handle opened on
  it). A path that reaches a writer only as a bare function parameter is not traced.

The scanned set is GLOBBED, so a module or ledger CLI added by a later plan is covered without
editing this file, and it is asserted non-empty so the scan cannot pass by scanning nothing. A
negative control runs the same detector over synthetic in-memory sources, so a detector that
had silently stopped detecting would fail here rather than report a clean ledger.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

LEDGER_PACKAGE = REPO_ROOT / "forward_ledger"
SCRIPTS_DIR = REPO_ROOT / "scripts"

# The ledger CLIs whose names do not carry "ledger". Listed by the plans that add them; one
# that does not exist yet is simply not scanned until it does.
NAMED_LEDGER_CLIS: tuple[str, ...] = (
    "record_fill.py",
    "migrate_forward_rows.py",
    "capture_closing_lines.py",
    "sync_ledger.py",
)

MANIFEST_FILE_NAME = "latest.json"

MANIFEST_WRITERS: frozenset[str] = frozenset(
    {"update_manifest", "replace_manifest", "_atomic_write_json", "save_model_artifact"}
)

_WRITE_MODE_CHARACTERS = frozenset("wax+")
_MODE_CHARACTERS = frozenset("rwxabt+")

# Calls that put bytes at a path by moving or copying a file there.
_PATH_WRITERS: frozenset[str] = frozenset(
    {"replace", "rename", "copy", "copy2", "copyfile", "copytree", "move"}
)

# The daily run's modules: the ledger path runs THROUGH them (the entry point, its steps, and the
# decision bundle the runner builds), so a manifest write there could swap models mid-run too.
DAILY_RUN_MODULES: tuple[str, ...] = (
    "scripts/daily_lock_pipeline.py",
    "pipeline/daily_steps.py",
    "backtest/weekly_bet_list.py",
)


def scanned_paths() -> list[Path]:
    """Every ledger module and ledger CLI on disk, deduplicated, in a stable order."""
    paths = set(LEDGER_PACKAGE.rglob("*.py"))
    paths |= set(SCRIPTS_DIR.glob("*ledger*.py"))
    paths |= {
        SCRIPTS_DIR / name
        for name in NAMED_LEDGER_CLIS
        if (SCRIPTS_DIR / name).is_file()
    }
    paths |= {REPO_ROOT / relative for relative in DAILY_RUN_MODULES}
    return sorted(paths)


def _called_name(func: ast.expr) -> str | None:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _open_mode(call: ast.Call) -> str:
    """The mode an ``open`` call was given, or ``"r"`` when it names none."""
    for keyword in call.keywords:
        if (
            keyword.arg == "mode"
            and isinstance(keyword.value, ast.Constant)
            and isinstance(keyword.value.value, str)
        ):
            return keyword.value.value
    # Builtin ``open(path, mode)`` and ``Path.open(mode)`` put the mode at different
    # positions, so the mode is whichever positional string literal reads as one.
    for argument in call.args:
        if (
            isinstance(argument, ast.Constant)
            and isinstance(argument.value, str)
            and 0 < len(argument.value) <= 3
            and set(argument.value) <= _MODE_CHARACTERS
        ):
            return argument.value
    return "r"


def _mentions(node: ast.AST, bound: set[str]) -> bool:
    """True when *node* names the manifest: the literal itself, or a name bound to it."""
    return MANIFEST_FILE_NAME in ast.unparse(node) or any(
        isinstance(child, ast.Name) and child.id in bound for child in ast.walk(node)
    )


def _manifest_bound_names(tree: ast.AST) -> set[str]:
    """Every name the module binds to the manifest path, to a fixed point (34-REVIEW WR-28).

    ``path = artifacts_dir / "latest.json"`` binds ``path``; ``tmp = path.with_suffix(".tmp")``
    then binds ``tmp``; ``with open(path, "w") as fh`` binds ``fh``. Module-wide rather than per
    function: a name bound to the manifest anywhere in a ledger module is suspect everywhere in it.
    """
    bindings: list[tuple[list[ast.expr], ast.expr]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            bindings.append((list(node.targets), node.value))
        elif isinstance(node, ast.AnnAssign | ast.NamedExpr) and node.value is not None:
            bindings.append(([node.target], node.value))
        elif isinstance(node, ast.withitem) and node.optional_vars is not None:
            bindings.append(([node.optional_vars], node.context_expr))
    bound: set[str] = set()
    while True:
        grown = {
            name.id
            for targets, value in bindings
            if _mentions(value, bound)
            for target in targets
            for name in ast.walk(target)
            if isinstance(name, ast.Name)
        }
        if grown <= bound:
            return bound
        bound |= grown


def manifest_writes(source: str, filename: str = "<memory>") -> list[str]:
    """Every call in *source* that writes, or could write, the production manifest.

    Writer shapes: a manifest-writer call; ``write_text`` / ``write_bytes`` / a write-mode
    ``open``; ``os.replace`` / ``os.rename`` / ``Path.replace`` / ``Path.rename`` /
    ``shutil.copy*`` / ``shutil.move``; and ``json.dump`` into a handle -- each when the call names
    the manifest literally or through a name bound to it.
    """
    tree = ast.parse(source, filename=filename)
    bound = _manifest_bound_names(tree)
    findings: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = _called_name(node.func)
        if name is None:
            continue
        where = f"{filename}:{node.lineno}"
        if name in MANIFEST_WRITERS:
            findings.append(f"{where} calls the manifest writer {name}()")
            continue
        if not _mentions(node, bound):
            continue
        if name in ("write_text", "write_bytes"):
            findings.append(f"{where} {name}() names {MANIFEST_FILE_NAME}")
        elif name == "open" and set(_open_mode(node)) & _WRITE_MODE_CHARACTERS:
            findings.append(f"{where} opens {MANIFEST_FILE_NAME} for writing")
        elif name in _PATH_WRITERS:
            findings.append(
                f"{where} {name}() moves or copies onto {MANIFEST_FILE_NAME}"
            )
        elif name in ("dump", "to_json"):
            findings.append(f"{where} {name}() writes into {MANIFEST_FILE_NAME}")
    return findings


def test_the_scanned_set_is_not_empty_and_covers_the_store() -> None:
    paths = scanned_paths()

    assert paths, "the latest.json scan found no ledger module to scan"
    assert LEDGER_PACKAGE / "store.py" in paths, [p.as_posix() for p in paths]
    for relative in DAILY_RUN_MODULES:
        assert (REPO_ROOT / relative).is_file(), f"daily-run module {relative} moved"


def test_no_ledger_path_writes_latest_json() -> None:
    findings = [
        finding
        for path in scanned_paths()
        for finding in manifest_writes(
            path.read_text(encoding="utf-8"),
            path.relative_to(REPO_ROOT).as_posix(),
        )
    ]

    assert findings == [], (
        "a ledger code path writes, or calls a writer of, artifacts/latest.json -- the ONE "
        "production swap surface the ledger may only read:\n" + "\n".join(findings)
    )


@pytest.mark.parametrize(
    "source",
    [
        "from models.artifacts import update_manifest\nupdate_manifest('wp', 'wp_x', root)\n",
        "from models import artifacts\nartifacts.replace_manifest(mapping, root)\n",
        "from models.artifacts import _atomic_write_json\n"
        "_atomic_write_json(root / 'latest.json', {})\n",
        "save_model_artifact(model, 'wp', {}, [], update_latest=True)\n",
        "(root / 'latest.json').write_text('{}')\n",
        "(root / 'latest.json').write_bytes(b'{}')\n",
        "open(root / 'latest.json', 'w')\n",
        "(root / 'latest.json').open(mode='a')\n",
        # WR-28: the manifest reached through a bound name, and the move / copy / dump shapes.
        "path = artifacts_dir / 'latest.json'\npath.write_text('{}')\n",
        "def f(root):\n    manifest = root / 'latest.json'\n    tmp = manifest.with_suffix('.tmp')\n"
        "    tmp.write_text('{}')\n    os.replace(tmp, manifest)\n",
        "MANIFEST = 'latest.json'\ntarget = root / MANIFEST\nshutil.copy(src, target)\n",
        "manifest = root / 'latest.json'\nshutil.move(src, manifest)\n",
        "manifest_path = root / 'latest.json'\nstaged.replace(manifest_path)\n",
        "path = root / 'latest.json'\nwith open(path, 'w') as fh:\n    json.dump({}, fh)\n",
        "path = root / 'latest.json'\nwith path.open('w') as fh:\n    json.dump({}, fh)\n",
    ],
)
def test_negative_control_flags_a_synthetic_manifest_write(source: str) -> None:
    assert manifest_writes(source), f"the detector missed a manifest write:\n{source}"


@pytest.mark.parametrize(
    "source",
    [
        "json.loads((root / 'latest.json').read_text())\n",
        "open(root / 'latest.json')\n",
        "open(root / 'latest.json', 'r')\n",
        "(root / 'rows.jsonl').write_text('')\n",
        "path = root / 'latest.json'\nmapping = json.loads(path.read_text())\n",
        "path = root / 'latest.json'\nif path.exists():\n    other.write_text('{}')\n",
        "rows = root / 'rows.jsonl'\nos.replace(tmp, rows)\n",
    ],
)
def test_negative_control_passes_a_manifest_read(source: str) -> None:
    assert manifest_writes(source) == [], source
