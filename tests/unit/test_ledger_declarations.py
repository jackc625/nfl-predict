"""The ledger's declarations loader and its cutover switch (Phase 34, Plan 34-04 Task 1).

One loader resolves every pre-registration record the ledger stamps -- the 2026 verdict-scope
declaration, the recipe registry and the fill conventions -- and refuses by name rather than
defaulting: a missing declaration is never "no scope", an unknown recipe or fill id is never
stamped. The cutover switch is the one read point that decides whether forward rows go to the
ledger, and it is OFF until Plan 34-19's go-live commit flips it.

No test here writes a file. The malformed and well-formed declarations are injected as in-memory
modules through ``sys.modules``, so nothing is added to the repository's import path.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

from forward_ledger import cutover
from forward_ledger.declarations import (
    BOOTSTRAP_REGIME_WEEKS,
    VERDICT_SCOPE_MODULE,
    UnknownFillConventionError,
    UnknownRecipeError,
    VerdictScope,
    VerdictScopeMalformedError,
    VerdictScopeUndeclaredError,
    load_verdict_scope,
    resolve_fill_convention,
    resolve_recipe,
    verdict_scope_label,
)
from forward_ledger.schema import VERDICT_SCOPE_PRE_VERDICT, VERDICT_SCOPE_VERDICT

_FAKE_MODULE = "tests_fake_verdict_scope_declaration"


def _declaration_constants(**overrides: Any) -> dict[str, Any]:
    constants: dict[str, Any] = {
        "VERDICT_SEASON": 2026,
        "VERDICT_START_WEEK": 6,
        "VERDICT_END_WEEK": 22,
        "INCLUDES_PLAYOFF_WEEKS": True,
        "INCLUDES_NEUTRAL_SITE_GAMES": True,
        "COUNTED_ARM": "live",
        "OUTCOME_RULE": "corrected outcome in force (D-06)",
        "FILL_CONVENTION_ID": "fill-v1",
        "BOOTSTRAP_REGIME_WEEKS": (2, 3, 4),
    }
    constants.update(overrides)
    return constants


def _inject_module(
    monkeypatch: pytest.MonkeyPatch, constants: dict[str, Any]
) -> types.ModuleType:
    module = types.ModuleType(_FAKE_MODULE)
    for name, value in constants.items():
        setattr(module, name, value)
    monkeypatch.setitem(sys.modules, _FAKE_MODULE, module)
    return module


def _scope(**overrides: Any) -> VerdictScope:
    fields: dict[str, Any] = {
        "season": 2026,
        "start_week": 6,
        "end_week": 22,
        "includes_playoff_weeks": True,
        "includes_neutral_site_games": True,
        "counted_arm": "live",
        "outcome_rule": "corrected outcome in force (D-06)",
        "fill_convention_id": "fill-v1",
        "bootstrap_regime_weeks": (2, 3, 4),
    }
    fields.update(overrides)
    return VerdictScope(**fields)


def test_absent_declaration_is_named() -> None:
    assert VERDICT_SCOPE_MODULE == "backtest.verdict_scope_2026"
    absent = "backtest.verdict_scope_does_not_exist_2026"
    with pytest.raises(VerdictScopeUndeclaredError, match=absent):
        load_verdict_scope(absent)
    # Named, and deliberately NOT a ValueError: a broad handler must not turn "no declaration"
    # into "no scope".
    assert not issubclass(VerdictScopeUndeclaredError, ValueError | RuntimeError)


def test_label_without_declaration_is_pre_verdict() -> None:
    assert (
        verdict_scope_label(2026, 9, None) == VERDICT_SCOPE_PRE_VERDICT == "pre_verdict"
    )


def test_label_with_declaration() -> None:
    scope = _scope(start_week=6, end_week=22)
    assert verdict_scope_label(2026, 5, scope) == VERDICT_SCOPE_PRE_VERDICT
    assert verdict_scope_label(2026, 6, scope) == VERDICT_SCOPE_VERDICT
    assert verdict_scope_label(2026, 22, scope) == VERDICT_SCOPE_VERDICT
    assert verdict_scope_label(2026, 23, scope) == VERDICT_SCOPE_PRE_VERDICT
    assert verdict_scope_label(2025, 10, scope) == VERDICT_SCOPE_PRE_VERDICT


def test_malformed_declaration_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    # Control: the complete declaration loads, constant for constant.
    _inject_module(monkeypatch, _declaration_constants())
    assert load_verdict_scope(_FAKE_MODULE) == _scope()

    constants = _declaration_constants()
    del constants["VERDICT_START_WEEK"]
    _inject_module(monkeypatch, constants)
    with pytest.raises(VerdictScopeMalformedError, match="VERDICT_START_WEEK"):
        load_verdict_scope(_FAKE_MODULE)
    assert issubclass(VerdictScopeMalformedError, ValueError)


def test_resolve_recipe_and_fill() -> None:
    entry = resolve_recipe("recipe-2026-row19-v1")
    assert entry.recipe_id == "recipe-2026-row19-v1"
    with pytest.raises(UnknownRecipeError, match="recipe-unknown"):
        resolve_recipe("recipe-unknown")
    with pytest.raises(UnknownRecipeError):
        resolve_recipe(None)

    assert resolve_fill_convention("fill-v1") == "fill-v1"
    with pytest.raises(UnknownFillConventionError):
        resolve_fill_convention(None)
    with pytest.raises(UnknownFillConventionError, match="fill-v2"):
        resolve_fill_convention("fill-v2")


def test_bootstrap_weeks() -> None:
    assert BOOTSTRAP_REGIME_WEEKS == (2, 3, 4)


def test_cutover_committed_on(monkeypatch: pytest.MonkeyPatch) -> None:
    # Was: the committed default was OFF; Plan 34-19 flipped it in commit 300e44e. Read the
    # committed source, since tests/conftest.py pins the in-memory switch OFF for every test.
    source = Path(cutover.__file__).read_text(encoding="utf-8")
    assert "FORWARD_ROWS_GO_TO_LEDGER: bool = True" in source
    # The accessor reads the module constant at call time, so the ONE flip is the constant.
    monkeypatch.setattr(cutover, "FORWARD_ROWS_GO_TO_LEDGER", True)
    assert cutover.forward_rows_go_to_ledger() is True
    monkeypatch.setattr(cutover, "FORWARD_ROWS_GO_TO_LEDGER", False)
    assert cutover.forward_rows_go_to_ledger() is False
