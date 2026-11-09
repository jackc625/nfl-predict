"""The early-week arm's record agrees with the code it describes, and the ledger accepts its ids.

``backtest/early_arm_2026.py`` copies values other code also holds: the v2 recipe name, and the
start week the page keeps its own copy of because ``api/`` may not import ``backtest``. These tests
prove the copies agree, that the record stays constants-only, and that the ledger's one loader now
resolves the record's recipe and fill convention while still refusing anything else by name.

No file is written and no data is read.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

import api.cache_schema
import backtest.early_arm_2026 as record
import forward_ledger.schema
from backtest.midseason_switch_2026 import INSTALLED_RECIPE
from forward_ledger.declarations import (
    RefitRecipe,
    UnknownRecipeError,
    resolve_fill_convention,
    resolve_recipe,
)
from models.v2_recipe import V2_RECIPE_NAME

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_the_recipe_is_the_installed_one() -> None:
    assert record.EARLY_RECIPE == INSTALLED_RECIPE == V2_RECIPE_NAME


def test_moneyline_is_out() -> None:
    assert record.EARLY_TARGETS == ("ats", "ou")


def test_the_page_copy_of_the_start_week_agrees() -> None:
    assert api.cache_schema.EARLY_ARM_FROM == record.EARLY_FROM


def test_the_ledger_schema_re_exports_the_arm_and_the_start_week() -> None:
    assert forward_ledger.schema.ARM_EARLY == "early"
    assert forward_ledger.schema.ARM_EARLY in forward_ledger.schema.ARMS
    assert forward_ledger.schema.EARLY_ARM_FROM == record.EARLY_FROM


def test_the_record_imports_only_the_standard_library() -> None:
    source = (REPO_ROOT / record.RECORD_PATH).read_text(encoding="utf-8")
    roots: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "the record must not import a sibling module"
            roots.add((node.module or "").split(".")[0])
    assert roots, "the record's imports were not read"
    outside = sorted(roots - set(sys.stdlib_module_names))
    assert outside == [], f"the record imports outside the standard library: {outside}"


def test_fill_v2_resolves_and_fill_v1_still_does() -> None:
    assert resolve_fill_convention("fill-v2") == "fill-v2"
    assert resolve_fill_convention("fill-v1") == "fill-v1"


def test_the_v2_recipe_resolves() -> None:
    entry = resolve_recipe("recipe-2026-v2")
    assert isinstance(entry, RefitRecipe)
    assert entry.recipe_id == "recipe-2026-v2"
    assert entry.fill_convention_id == "fill-v2"


def test_an_unknown_recipe_is_still_refused() -> None:
    with pytest.raises(UnknownRecipeError) as refused:
        resolve_recipe("recipe-nope")
    assert "recipe-nope" in str(refused.value)
    assert record.EARLY_RECIPE in str(refused.value), (
        "the refusal no longer names what the ledger does accept"
    )
