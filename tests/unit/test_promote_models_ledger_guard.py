"""No tool deletes an artifact a ledger row references (Plan 34-12 Task 3; D-11, LDGR-09).

``forward_ledger.retention.referenced_artifact_ids`` names every artifact the ledger's replay
depends on: each row's model and blend id, plus every directory the ledger keeps a copy of under
``ledger/artifacts/`` (which includes the blend's bound converter). The only production tool that
deletes model directories -- ``scripts/promote_models.py``'s ``_clear_staging_dir`` -- refuses a
referenced one by name and still clears the rest.

The inventory test counts EVERY deletion-shaped call in the production tree against a reviewed
constant, per function and per verb, each site with its reason: directory removal (``rmtree`` /
``rmdir`` / ``removedirs``), an ``unlink`` / ``remove`` / ``move`` / ``rename`` / ``replace`` that
names an artifact, ledger or staging path, and a shell ``rd`` / ``rmdir`` / ``del`` / ``rm`` run
through a subprocess. A new deletion cannot appear unreviewed -- not even a second one inside a
reviewed function. (The mover filter reads the call's own text and the names its scope binds to a
protected path -- ``p = artifacts_dir / x``, ``for f in artifacts_dir.rglob(...)``; a path that
reaches a mover any other way, such as a bare parameter, is not seen. Directory removal is counted
whatever it names.)

Ledgers and staging trees live under ``tmp_path``; the production ``ledger/`` and ``artifacts/``
are never touched (COLD-05).

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
import textwrap
from collections import Counter
from pathlib import Path

import pytest

from forward_ledger.retention import referenced_artifact_ids
from forward_ledger.store import LedgerFormatError, append_rows, ledger_path
from scripts import promote_models
from scripts.promote_models import _clear_staging_dir
from tests.unit.test_forward_ledger_store import make_row

REPO_ROOT = Path(__file__).resolve().parents[2]

_SCANNED_ROOTS: tuple[str, ...] = (
    "scripts",
    "models",
    "backtest",
    "pipeline",
    "data",
    "utils",
    "forward_ledger",
    "features",
    "ratings",
    "api",
)

# Every deletion-shaped call in the production tree, keyed (file, enclosing function) -> (the
# reviewed count of each deletion verb there, the reason none of them can delete an artifact a
# ledger row references). Counted, so a SECOND unguarded call inside a reviewed function fails the
# inventory too (34-REVIEW WR-27).
REVIEWED_DELETION_SITES: dict[tuple[str, str], tuple[dict[str, int], str]] = {
    ("scripts/promote_models.py", "_clear_staging_dir"): (
        {"rmtree": 1, "unlink": 1},
        "stale staging candidates; GUARDED: refuses any id referenced_artifact_ids() names. "
        "The unlink is the staging root's own stale latest.json, never a model directory",
    ),
    ("scripts/elo_generation.py", "prune_elo_generations"): (
        {"rmtree": 1},
        "old Elo generations under data/silver, never model artifact directories",
    ),
    ("scripts/repair_odds_snapshot.py", "run"): (
        {"rmtree": 1},
        "stray silver odds partition directories, never model artifact directories",
    ),
    ("scripts/setup_ledger_remote.py", "cmd_pin_host_keys"): (
        {"replace": 1},
        "atomic replace of the deploy keys' known_hosts file, never an artifact or ledger path",
    ),
    ("data/storage.py", "ParquetManager.delete"): (
        {"rmtree": 1},
        "a data-lake parquet path under the storage base, never artifacts/ or ledger/artifacts/",
    ),
    ("forward_ledger/artifacts_copy.py", "ensure_artifact_copies"): (
        {"rmtree": 1, "rename": 1},
        "the ledger's own private .staging-* copy directory: renamed INTO place on success, "
        "removed only on a failed copy",
    ),
    ("forward_ledger/artifacts_copy.py", "ensure_recipe_record_copy"): (
        {"replace": 1},
        "the chain-fit copy's own temp file replaced INTO ledger/recipes/, never moved out",
    ),
    ("forward_ledger/snapshots.py", "write_decision_snapshot"): (
        {"rmtree": 2, "rename": 1},
        "the snapshot's own private .staging-* directory: renamed into place, or removed; never "
        "a model artifact",
    ),
}

# The verbs that remove a directory outright: every call is counted.
_DIRECTORY_REMOVERS: frozenset[str] = frozenset({"rmtree", "rmdir", "removedirs"})
# The verbs that delete or move a path: counted when the call names an artifact, ledger or staging
# path (a ``Path.unlink`` loop, ``shutil.move`` / ``rename`` / ``replace`` out of ``artifacts/``).
_PATH_MOVERS: frozenset[str] = frozenset(
    {"unlink", "remove", "move", "rename", "replace"}
)
_PROTECTED_PATH_WORDS: tuple[str, ...] = ("artifact", "ledger", "staging")
# A shell deletion (``rd /s``, ``rmdir``, ``del``, ``rm -rf``) through a subprocess call.
_SUBPROCESS_CALLS: frozenset[str] = frozenset(
    {"run", "Popen", "call", "check_call", "check_output", "system"}
)
_SHELL_DELETE_WORDS: frozenset[str] = frozenset({"rd", "rmdir", "del", "rm"})


def _ledger_with_rows(ledger: Path) -> None:
    append_rows(
        ledger,
        [
            make_row(model_artifact_id="wp_X", blend_id="blend_Y", target="wp"),
            make_row(game_id="2026_06_DAL_NYG"),
        ],
    )


# ---------------------------------------------------------------------------
# referenced_artifact_ids
# ---------------------------------------------------------------------------


def test_referenced_ids_from_rows_and_copies(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger"
    _ledger_with_rows(ledger)
    (ledger / "artifacts" / "market_probability_Z").mkdir(parents=True)
    second = make_row(game_id="2026_06_DAL_NYG")

    assert referenced_artifact_ids(ledger) == {
        "wp_X",
        "blend_Y",
        "market_probability_Z",
        second["model_artifact_id"],
        second["blend_id"],
    }
    assert referenced_artifact_ids(tmp_path / "absent") == frozenset()

    ledger_path(ledger).write_bytes(b"not json\n")
    with pytest.raises(LedgerFormatError):
        referenced_artifact_ids(ledger)


# ---------------------------------------------------------------------------
# The promote_models guard
# ---------------------------------------------------------------------------


def test_clear_staging_refuses_referenced_dir(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger"
    _ledger_with_rows(ledger)
    staging = tmp_path / "staging"
    production = tmp_path / "artifacts"
    (staging / "wp_X").mkdir(parents=True)

    with pytest.raises(RuntimeError, match="wp_X"):
        _clear_staging_dir(staging, production, ledger_dir=ledger)

    assert (staging / "wp_X").is_dir()

    # Pointed at the production root, a referenced id is refused even with a ledger copy kept.
    (ledger / "artifacts" / "wp_X").mkdir(parents=True)
    (production / "wp_X").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="wp_X"):
        _clear_staging_dir(production, production, ledger_dir=ledger)
    assert (production / "wp_X").is_dir()

    # After a promote copied it into production, the staging duplicate may go (WR-08).
    _clear_staging_dir(staging, production, ledger_dir=ledger)
    assert not (staging / "wp_X").exists()
    assert (production / "wp_X").is_dir()


def test_clear_staging_still_clears_unreferenced(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger"
    _ledger_with_rows(ledger)
    staging = tmp_path / "staging"
    (staging / "wp_OLD").mkdir(parents=True)
    (staging / "latest.json").write_text("{}", encoding="utf-8")

    _clear_staging_dir(staging, tmp_path / "artifacts", ledger_dir=ledger)

    assert not (staging / "wp_OLD").exists()
    assert not (staging / "latest.json").exists()


def test_default_ledger_is_the_repositorys_not_the_working_directorys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """WR-26: the guard must find the ledger wherever the operator stands.

    Resolved against the working directory, a run started from any other directory found no
    ledger, read that as "nothing referenced" and deleted a referenced artifact. The default is
    now the repository's ``ledger/``. Here it is pointed at a sandbox ledger that references
    ``wp_X``, the process stands in a FOREIGN directory holding no ledger, and the call -- with no
    ``ledger_dir`` -- must still refuse.
    """
    assert promote_models._REPO_LEDGER_DIR == REPO_ROOT / "ledger"
    assert promote_models._REPO_LEDGER_DIR.is_absolute()

    ledger = tmp_path / "repo" / "ledger"
    _ledger_with_rows(ledger)
    monkeypatch.setattr(promote_models, "_REPO_LEDGER_DIR", ledger)
    foreign = tmp_path / "elsewhere"
    foreign.mkdir()
    monkeypatch.chdir(foreign)
    staging = tmp_path / "staging"
    (staging / "wp_X").mkdir(parents=True)

    with pytest.raises(RuntimeError, match="wp_X"):
        _clear_staging_dir(staging, tmp_path / "artifacts")
    assert (staging / "wp_X").is_dir()

    # Negative control: the fail-open shape this replaces -- a ledger looked up where the process
    # stands finds nothing in the foreign directory and protects nothing.
    assert referenced_artifact_ids(Path("ledger")) == frozenset()


# ---------------------------------------------------------------------------
# The inventory of every deletion site
# ---------------------------------------------------------------------------


def _call_name(node: ast.Call) -> str | None:
    func = node.func
    return func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)


def _names_protected_path(node: ast.AST) -> bool:
    text = ast.unparse(node).lower()
    return any(word in text for word in _PROTECTED_PATH_WORDS)


def _protected_names(scope: ast.AST) -> frozenset[str]:
    """Names *scope* binds to a protected path: ``p = artifacts_dir / x``, ``for f in
    artifacts_dir.rglob("*")`` and the comprehension form -- so ``f.unlink()`` in such a loop is
    seen even though its own text names no artifact."""
    names: set[str] = set()
    for node in ast.walk(scope):
        if isinstance(node, ast.Assign | ast.AnnAssign) and node.value is not None:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            bound, source = targets, node.value
        elif isinstance(node, ast.For | ast.AsyncFor | ast.comprehension):
            bound, source = [node.target], node.iter
        else:
            continue
        if _names_protected_path(source):
            names |= {
                name.id
                for target in bound
                for name in ast.walk(target)
                if isinstance(name, ast.Name)
            }
    return frozenset(names)


def _deletion_verb(node: ast.Call, protected: frozenset[str]) -> str | None:
    """The deletion verb *node* is, or None when it cannot delete or move a protected path."""
    name = _call_name(node)
    if name in _DIRECTORY_REMOVERS:
        return name
    if name in _PATH_MOVERS:
        # ``frame.rename(columns=...)`` relabels a table; it moves no file.
        if any(
            keyword.arg in {"columns", "index", "mapper"} for keyword in node.keywords
        ):
            return None
        uses_protected_name = any(
            isinstance(child, ast.Name) and child.id in protected
            for child in ast.walk(node)
        )
        return name if uses_protected_name or _names_protected_path(node) else None
    if name in _SUBPROCESS_CALLS:
        words = {
            constant.value.split()[0].lower()
            for constant in ast.walk(node)
            if isinstance(constant, ast.Constant)
            and isinstance(constant.value, str)
            and constant.value.strip()
        }
        return "shell_delete" if words & _SHELL_DELETE_WORDS else None
    return None


def _deletion_sites(relative: str, source: str) -> Counter[tuple[str, str, str]]:
    """``(file, enclosing function, verb)`` -> calls, for every deletion-shaped call in *source*."""
    sites: Counter[tuple[str, str, str]] = Counter()

    def visit(node: ast.AST, scope: tuple[str, ...], protected: frozenset[str]) -> None:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            scope = (*scope, node.name)
            protected = protected | _protected_names(node)
        if isinstance(node, ast.Call):
            verb = _deletion_verb(node, protected)
            if verb is not None:
                sites[(relative, ".".join(scope) or "<module>", verb)] += 1
        for child in ast.iter_child_nodes(node):
            visit(child, scope, protected)

    tree = ast.parse(source)
    module_level = frozenset(
        name
        for statement in tree.body
        if not isinstance(
            statement, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
        )
        for name in _protected_names(statement)
    )
    visit(tree, (), module_level)
    return sites


def _reviewed_counts() -> Counter[tuple[str, str, str]]:
    return Counter(
        {
            (relative, function, verb): count
            for (relative, function), (verbs, _) in REVIEWED_DELETION_SITES.items()
            for verb, count in verbs.items()
        }
    )


def test_rmtree_inventory() -> None:
    found: Counter[tuple[str, str, str]] = Counter()
    for root in _SCANNED_ROOTS:
        for path in sorted((REPO_ROOT / root).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            relative = path.relative_to(REPO_ROOT).as_posix()
            found += _deletion_sites(relative, path.read_text(encoding="utf-8"))

    reviewed = _reviewed_counts()
    assert found == reviewed, (
        f"unreviewed deletion call(s) {sorted((found - reviewed).items())}; "
        f"reviewed call(s) no longer found {sorted((reviewed - found).items())}"
    )
    assert all(reason.strip() for _, reason in REVIEWED_DELETION_SITES.values())


def test_rmtree_inventory_controls() -> None:
    """The scan sees each deletion shape, and counts a second call in a reviewed function."""
    reviewed_fn = textwrap.dedent(
        """
        def _clear_staging_dir(staging_dir):
            shutil.rmtree(stale)
            shutil.rmtree(staging_dir / "wp_extra")
        """
    )
    site = ("scripts/promote_models.py", "_clear_staging_dir", "rmtree")
    found = _deletion_sites(site[0], reviewed_fn)
    assert found[site] == 2
    assert found[site] != _reviewed_counts()[site], "a second rmtree would pass"

    planted = {
        "os.removedirs(path)": "removedirs",
        "[f.unlink() for f in artifacts_dir.rglob('*')]": "unlink",
        "os.remove(ledger_dir / 'forward_2026.jsonl')": "remove",
        "shutil.move(artifacts_dir / 'wp_X', elsewhere)": "move",
        "(ARTIFACTS / 'wp_X').rename(trash)": "rename",
        "os.replace(staging_root, target)": "replace",
        "subprocess.run(['cmd', '/c', 'rd', '/s', '/q', str(p)])": "shell_delete",
        "os.system('rmdir /s /q artifacts')": "shell_delete",
    }
    for body, verb in planted.items():
        source = f"def f():\n    {body}\n"
        assert list(_deletion_sites("backtest/x.py", source)) == [
            ("backtest/x.py", "f", verb)
        ], body

    # Not deletions: a table relabel, a string edit, an unrelated unlink, a plain subprocess.
    benign = textwrap.dedent(
        """
        def f(frame, text, tmp):
            frame.rename(columns={"ledger_x": "y"})
            text.replace("a", "b")
            tmp.unlink()
            subprocess.run(["git", "status"])
        """
    )
    assert _deletion_sites("backtest/x.py", benign) == Counter()
