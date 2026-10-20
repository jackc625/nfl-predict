"""Generation 0 by the pin: a row's generation derived from its stamped model id (D-19, R11, RFIT-08).

A row's generation is never stored; it is read from the row's own ``model_artifact_id``. The
pinned row-19 ids are generation 0, a NULL stamp (weeks 3-4) is no generation, an installed
generation's ids map to its number, and any other id is refused by name. A shadow row is
generation 0 ONLY with a pinned id: the arm alone never decides the generation.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from backtest.refit_preregistration_2026 import REFIT_RECIPE_ID
from forward_ledger.schema import ARM_LIVE, ARM_SHADOW
from forward_ledger.shadow import SHADOW_PIN
from models.artifacts import ResolvedArtifacts
from refit.generations import (
    GENERATION_ZERO,
    Generation,
    UnknownGenerationError,
    generation_of_row,
    generation_zero_ids,
)
from tests.unit.test_forward_ledger_store import make_row

_PINNED_MODEL_IDS: tuple[str, ...] = tuple(SHADOW_PIN.model_ids.values())

_GENERATION_ONE = ResolvedArtifacts(
    wp="wp_20261101_050000",
    ats="ats_20261101_050000",
    ou="ou_20261101_050000",
    blend="blend_20261101_050000",
    converter="market_probability_20260923_195443",
)
_INSTALLS: tuple[Generation, ...] = (
    Generation(
        number=1,
        ids=_GENERATION_ONE,
        recipe=REFIT_RECIPE_ID,
        bet_rule_path=Path("ledger/refit/bet_rules") / f"{'a' * 64}.json",
        date_index=1,
        list_week=8,
    ),
)


def _generation(
    row: dict[str, object], installs: Sequence[Generation] = ()
) -> int | None:
    model_id = row["model_artifact_id"]
    assert model_id is None or isinstance(model_id, str)
    return generation_of_row(model_id, str(row["arm"]), installs)


def test_generation_zero_ids_are_the_pin() -> None:
    assert GENERATION_ZERO == 0
    assert generation_zero_ids() == SHADOW_PIN.resolved()


@pytest.mark.parametrize("model_id", _PINNED_MODEL_IDS)
def test_shadow_rows_are_generation_zero(model_id: str) -> None:
    row = make_row(arm=ARM_SHADOW, model_artifact_id=model_id)
    assert _generation(row) == GENERATION_ZERO


@pytest.mark.parametrize(
    ("model_id", "installs"),
    [
        ("ats_20250101_000000", ()),
        (None, ()),
        (_GENERATION_ONE.ats, _INSTALLS),
    ],
    ids=["unknown-id", "null-id", "installed-generation-id"],
)
def test_a_shadow_row_with_any_other_id_is_refused(
    model_id: str | None, installs: tuple[Generation, ...]
) -> None:
    row = make_row(arm=ARM_SHADOW, model_artifact_id=model_id)
    with pytest.raises(UnknownGenerationError) as raised:
        _generation(row, installs)
    message = str(raised.value)
    assert repr(model_id) in message
    assert ARM_SHADOW in message


@pytest.mark.parametrize("model_id", _PINNED_MODEL_IDS)
def test_pinned_ids_on_a_live_row_are_generation_zero(model_id: str) -> None:
    row = make_row(arm=ARM_LIVE, model_artifact_id=model_id)
    assert _generation(row) == GENERATION_ZERO
    assert _generation(row, _INSTALLS) == GENERATION_ZERO


def test_null_stamp_maps_to_no_generation() -> None:
    row = make_row(arm=ARM_LIVE, week=3, model_artifact_id=None)
    assert _generation(row) is None
    assert _generation(row, _INSTALLS) is None


def test_unregistered_id_is_refused() -> None:
    row = make_row(arm=ARM_LIVE, model_artifact_id="ats_20250101_000000")
    with pytest.raises(UnknownGenerationError) as raised:
        _generation(row, _INSTALLS)
    assert "'ats_20250101_000000'" in str(raised.value)


@pytest.mark.parametrize("target", ["wp", "ats", "ou"])
def test_installed_ids_map_to_their_generation(target: str) -> None:
    row = make_row(arm=ARM_LIVE, model_artifact_id=_GENERATION_ONE.model_id_for(target))
    assert _generation(row, _INSTALLS) == 1
