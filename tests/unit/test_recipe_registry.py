"""The recipe registry resolves, and its first entry equals what actually runs (Phase 34, LDGR-03).

``backtest/recipe_registry.py`` is a pre-registration: every 2026 ledger row's ``recipe_id`` must
resolve to an entry in it. The first entry is pinned here to the LIVE bet-rule constants and to the
artifacts it names. The artifact checks read the gitignored ``artifacts/`` directory and the
chain-fit record under ``outputs/``; on a checkout without them they skip with a named reason
rather than pass silently.

The recorded artifact ids are deliberately NOT compared with ``artifacts/latest.json``: a later
re-fit may legitimately move the production pointer, while the entry must keep naming the models
this recipe produced. What is checked is that they are still retained and still bound together.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
import hashlib
import sys
from pathlib import Path

import pytest

from backtest import neutral_hfa_cold_start_constants as live_bet_rule
from backtest import recipe_registry
from backtest.neutral_hfa_ev_chain_constants import CORRECTED_CHAIN_FIT_RECORD_PATH
from backtest.recipe_registry import RECIPE_REGISTRY, RecipeEntry

_ARTIFACTS_DIR = Path("artifacts")
_ARTIFACTS_ABSENT = (
    "artifacts/ is absent on this checkout (gitignored); the recorded artifact ids cannot be "
    "checked against the directories they name"
)
_ENTRY_ID = "recipe-2026-row19-v1"


def _entry() -> RecipeEntry:
    return RECIPE_REGISTRY[_ENTRY_ID]


def test_first_entry_resolves() -> None:
    assert list(RECIPE_REGISTRY) == [_ENTRY_ID]
    assert recipe_registry.IN_FORCE_RECIPE_ID == _ENTRY_ID
    entry = _entry()
    assert entry.recipe_id == _ENTRY_ID
    assert entry.fill_convention_id == "fill-v1"
    assert set(entry.model_artifact_ids) == {"wp", "ats", "ou"}
    assert set(entry.training) == {"wp", "ats", "ou"}


def test_bet_rule_values_equal_live_modules() -> None:
    entry = _entry()
    assert dict(entry.edge_tier_thresholds) == dict(
        live_bet_rule.EDGE_TIER_THRESHOLDS_BY_TARGET
    )
    assert entry.chain_fit_record_path == CORRECTED_CHAIN_FIT_RECORD_PATH


def test_chain_fit_record_sha_matches_disk() -> None:
    entry = _entry()
    record = Path(entry.chain_fit_record_path)
    if not record.is_file():
        pytest.skip(
            f"{record.as_posix()} is absent on this checkout (gitignored generator output); "
            "its recorded sha256 cannot be compared"
        )
    assert hashlib.sha256(record.read_bytes()).hexdigest() == (
        entry.chain_fit_record_sha256
    )


def test_recorded_artifacts_are_retained() -> None:
    if not _ARTIFACTS_DIR.is_dir():
        pytest.skip(_ARTIFACTS_ABSENT)
    entry = _entry()
    recorded = [*entry.model_artifact_ids.values(), entry.blend_id, entry.converter_id]
    missing = [aid for aid in recorded if not (_ARTIFACTS_DIR / aid).is_dir()]
    assert not missing, f"recorded artifacts no longer retained: {missing}"


def test_recorded_converter_is_the_blends_own_binding() -> None:
    """Read from the real blend directory shape: blend_weights.json, never a metadata.json."""
    if not _ARTIFACTS_DIR.is_dir():
        pytest.skip(_ARTIFACTS_ABSENT)
    from models.artifacts import ARTIFACT_VALIDATORS

    entry = _entry()
    facts = ARTIFACT_VALIDATORS["blend"](entry.blend_id, _ARTIFACTS_DIR)

    assert facts.market_probability_artifact_id == entry.converter_id
    assert facts.source_artifact_ids == dict(entry.model_artifact_ids)
    assert facts.source_artifact_ids == dict(
        entry.blend_provenance["source_artifact_ids"]
    )
    assert (
        facts.gold_generation_digest
        == (entry.blend_provenance["gold_generation_digest"])
    )


def test_registry_has_no_project_imports() -> None:
    """A record that imports project code could change meaning when that code moves."""
    source = Path(recipe_registry.__file__).read_text(encoding="ascii")
    imported: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "relative import in the recipe registry"
            imported.add((node.module or "").split(".")[0])

    allowed = {"__future__", *sys.stdlib_module_names}
    assert imported <= allowed, f"non-stdlib imports: {sorted(imported - allowed)}"
