"""WR-03: the cache blends at the DEPLOYED dynamic weight, not a static scalar.

WHAT WAS WRONG
--------------
``api.cache._load_predictions`` read only ``blend_weights.json["weights"]`` and applied ONE scalar
per target to every game. The deployed artifact carries a ``dynamic`` section whose
``mode_by_target`` is ``{"wp": "dynamic", "ats": "dynamic", "ou": "dynamic"}``, which
``MarketBlender.from_artifacts`` auto-detects and applies as a per-week sigmoid.
``scripts/generate_current_week_predictions.apply_blending`` goes through that path; the cache did
not. So ``/``, ``/games/{id}`` and both export endpoints published a different ``blended_*`` for
the same game than the current-week CSV, and the one on the website was the one the deployed blend
config says is wrong.

Divergence on the live artifact (model weight):

    target   static (cache)   dynamic wk1   dynamic wk9   dynamic wk18
    wp       0.5917           0.5241        0.5964        0.6686
    ats      0.5456           0.5403        0.5459        0.5522
    ou       0.6040           0.5273        0.6093        0.6870

THE PARITY CLAIM IS ASSERTED AGAINST THE REAL IMPLEMENTATION. ``api/cache.py`` may not import
``models`` (UIAP-01, ``tests/api/test_import_guard.py``), so the sigmoid is reproduced there as
plain arithmetic. A reproduced formula is a formula that can drift, so the tests below compare it
cell-for-cell against ``models.blending.DynamicBlendWeights.get_weight`` -- which a TEST is free to
import. The era rule (17 weeks through 2020, D-05) and the playoff clamp (D-04) are covered
explicitly because both are places a reproduction typically diverges.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from api.cache import _blend_weight_array
from models.blending import DynamicBlendWeights

_LIVE_ARTIFACT = Path("artifacts/blend_dynamic_20260606_020635/blend_weights.json")
_TARGETS = ("wp", "ats", "ou")


def _blend_data() -> dict[str, Any]:
    if not _LIVE_ARTIFACT.exists():
        pytest.skip(
            f"evidence-backed skip: the deployed blend artifact {_LIVE_ARTIFACT} is absent "
            "from this checkout, so there is no deployed schedule to assert parity against"
        )
    return json.loads(_LIVE_ARTIFACT.read_text())


def _static_only(blend_data: dict[str, Any]) -> dict[str, Any]:
    """The same artifact with its dynamic section removed."""
    return {k: v for k, v in blend_data.items() if k != "dynamic"}


@pytest.mark.parametrize("target", _TARGETS)
@pytest.mark.parametrize("season", [2019, 2020, 2021, 2025])
def test_the_cache_schedule_matches_the_real_blender_week_for_week(
    target: str, season: int
) -> None:
    """Cell-for-cell parity with `DynamicBlendWeights.get_weight`, including both week eras."""
    blend_data = _blend_data()
    dynamic = DynamicBlendWeights.from_dict(blend_data["dynamic"])
    weeks = list(range(1, 23))  # through the playoff clamp on both eras

    from_cache = _blend_weight_array(
        blend_data, target, pd.Series([season] * len(weeks)), pd.Series(weeks)
    )
    from_blender = [dynamic.get_weight(target, week, season) for week in weeks]

    for week, cached, real in zip(weeks, from_cache, from_blender, strict=True):
        assert float(cached) == pytest.approx(real), (
            f"{target} season {season} week {week}: the cache blends at {cached} and the real "
            f"blender at {real}. The cache reproduces the sigmoid rather than importing it, so "
            "this is exactly the drift the reproduction can suffer."
        )


@pytest.mark.parametrize("target", _TARGETS)
def test_the_dynamic_weight_is_not_the_static_one(target: str) -> None:
    """The control. Without it, parity would also hold for a cache that ignored the schedule."""
    blend_data = _blend_data()
    static = float(blend_data["weights"][target])

    week_one = float(
        _blend_weight_array(blend_data, target, pd.Series([2025]), pd.Series([1]))[0]
    )
    week_eighteen = float(
        _blend_weight_array(blend_data, target, pd.Series([2025]), pd.Series([18]))[0]
    )

    assert week_one != pytest.approx(static), (
        f"{target} week 1 blends at the static {static}, so the deployed dynamic section is "
        "still being ignored"
    )
    assert week_eighteen > week_one, (
        "the schedule does not rise across the season, so it is not the deployed sigmoid"
    )


@pytest.mark.parametrize("target", _TARGETS)
def test_an_artifact_with_no_dynamic_section_still_blends_at_the_static_weight(
    target: str,
) -> None:
    """A pre-dynamic artifact must keep working -- the fallback is per row, not a raise."""
    blend_data = _blend_data()
    static = float(blend_data["weights"][target])

    weights = _blend_weight_array(
        _static_only(blend_data), target, pd.Series([2025, 2025]), pd.Series([1, 18])
    )

    assert [float(w) for w in weights] == pytest.approx([static, static])


@pytest.mark.parametrize("target", _TARGETS)
def test_a_target_gated_back_to_static_mode_is_honoured(target: str) -> None:
    """`mode_by_target` is the gate the tuning wrote; a "static" target must not be scheduled."""
    blend_data = _blend_data()
    gated = json.loads(json.dumps(blend_data))
    gated["dynamic"]["mode_by_target"][target] = "static"

    weight = float(
        _blend_weight_array(gated, target, pd.Series([2025]), pd.Series([1]))[0]
    )

    assert weight == pytest.approx(float(blend_data["weights"][target]))


def test_an_unreadable_week_falls_back_per_row_not_for_the_whole_frame() -> None:
    """One bad row must not silently restaticize every other game in the cache."""
    blend_data = _blend_data()
    static = float(blend_data["weights"]["wp"])

    weights = _blend_weight_array(
        blend_data,
        "wp",
        pd.Series([2025, 2025, 2025]),
        pd.Series([pd.NA, 0, 9], dtype="Int64"),
    )

    assert float(weights[0]) == pytest.approx(static), (
        "an absent week is not schedulable"
    )
    assert float(weights[1]) == pytest.approx(static), "week 0 is invalid, per D-04"
    assert float(weights[2]) != pytest.approx(static), (
        "the usable row was restaticized along with its neighbours"
    )
