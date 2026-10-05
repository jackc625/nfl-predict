"""No tool deletes an artifact a ledger row references (Plan 34-12 Task 3; D-11, LDGR-09).

``forward_ledger.retention.referenced_artifact_ids`` names every artifact the ledger's replay
depends on: each row's model and blend id, plus every directory the ledger keeps a copy of under
``ledger/artifacts/`` (which includes the blend's bound converter). The only production tool that
deletes model directories -- ``scripts/promote_models.py``'s ``_clear_staging_dir`` -- refuses a
referenced one by name and still clears the rest.

The inventory test lists EVERY directory-deletion call site in the production tree against a
reviewed constant, each with its reason, so a new ``rmtree`` cannot appear unreviewed.

Ledgers and staging trees live under ``tmp_path``; the production ``ledger/`` and ``artifacts/``
are never touched (COLD-05).

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from forward_ledger.retention import referenced_artifact_ids
from forward_ledger.store import LedgerFormatError, append_rows, ledger_path
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

# Every directory-deletion call site in the production tree, keyed (file, enclosing function),
# each with the reason it can never delete an artifact a ledger row references.
REVIEWED_DELETION_SITES: dict[tuple[str, str], str] = {
    ("scripts/promote_models.py", "_clear_staging_dir"): (
        "stale staging candidates; GUARDED: refuses any id referenced_artifact_ids() names"
    ),
    ("scripts/elo_generation.py", "prune_elo_generations"): (
        "old Elo generations under data/silver, never model artifact directories"
    ),
    ("scripts/repair_odds_snapshot.py", "run"): (
        "stray silver odds partition directories, never model artifact directories"
    ),
    ("data/storage.py", "ParquetManager.delete"): (
        "a data-lake parquet path under the storage base, never artifacts/ or ledger/artifacts/"
    ),
    ("forward_ledger/artifacts_copy.py", "ensure_artifact_copies"): (
        "the ledger's own private .staging-* copy directory, removed only on a failed copy"
    ),
    ("forward_ledger/snapshots.py", "write_decision_snapshot"): (
        "the snapshot's own private .staging-* directory, never a model artifact"
    ),
}


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


def test_clear_staging_refuses_referenced_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)  # the default ledger directory, "ledger", resolves here
    _ledger_with_rows(tmp_path / "ledger")
    staging = tmp_path / "staging"
    (staging / "wp_X").mkdir(parents=True)

    with pytest.raises(RuntimeError, match="wp_X"):
        _clear_staging_dir(staging)

    assert (staging / "wp_X").is_dir()


def test_clear_staging_still_clears_unreferenced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _ledger_with_rows(tmp_path / "ledger")
    staging = tmp_path / "staging"
    (staging / "wp_OLD").mkdir(parents=True)
    (staging / "latest.json").write_text("{}", encoding="utf-8")

    _clear_staging_dir(staging)

    assert not (staging / "wp_OLD").exists()
    assert not (staging / "latest.json").exists()


# ---------------------------------------------------------------------------
# The inventory of every deletion site
# ---------------------------------------------------------------------------


def _deletion_sites(relative: str, source: str) -> set[tuple[str, str]]:
    """``(file, enclosing function)`` for every rmtree / rmdir call in *source*."""
    sites: set[tuple[str, str]] = set()

    def visit(node: ast.AST, scope: tuple[str, ...]) -> None:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            scope = (*scope, node.name)
        if isinstance(node, ast.Call):
            func = node.func
            name = (
                func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
            )
            if name in ("rmtree", "rmdir"):
                sites.add((relative, ".".join(scope) or "<module>"))
        for child in ast.iter_child_nodes(node):
            visit(child, scope)

    visit(ast.parse(source), ())
    return sites


def test_rmtree_inventory() -> None:
    found: set[tuple[str, str]] = set()
    for root in _SCANNED_ROOTS:
        for path in sorted((REPO_ROOT / root).rglob("*.py")):
            if "__pycache__" in path.parts:
                continue
            relative = path.relative_to(REPO_ROOT).as_posix()
            found |= _deletion_sites(relative, path.read_text(encoding="utf-8"))

    assert found == set(REVIEWED_DELETION_SITES), (
        f"unreviewed deletion site(s) {sorted(found - set(REVIEWED_DELETION_SITES))}; "
        f"reviewed site(s) no longer found {sorted(set(REVIEWED_DELETION_SITES) - found)}"
    )
    assert all(reason.strip() for reason in REVIEWED_DELETION_SITES.values())
