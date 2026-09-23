"""The cache and the model path publish the SAME blended number for the same game (WR-03).

WHY THIS MODULE KEEPS ITS NAME WHILE ITS SUBJECT CHANGES
-------------------------------------------------------
It was written for WR-03: ``api.cache._load_predictions`` applied one static scalar per target
while ``scripts/generate_current_week_predictions.apply_blending`` applied the deployed
week-varying sigmoid, so ``/``, ``/games/{id}`` and both exports published a different
``blended_*`` for the same game than the current-week CSV (measured: WP 0.5917 against 0.5241
at week 1). It pinned parity by comparing the cache's arithmetic copy of the sigmoid,
``_blend_weight_array``, cell-for-cell against ``models.blending.DynamicBlendWeights``.

Plan 33.2-24 RETIRES the week-varying blend (D33.2-10) on EVERY live surface: the cache's copy
of the sigmoid is deleted with the model-side one, both surfaces read ONE scalar per target, and
both price the WP market side off the SPREAD through the fitted converter rather than a devigged
moneyline (D33.2-09). The INTENT is unchanged -- the cache and the model path must publish the
same number for the same game -- so the module keeps its filename as the history of that intent
while its cases move onto the new shape. (Renaming a tracked test module would lose the
``git log --follow`` trail from the WR-03 fix to this one.)

THE PARITY CLAIM IS ASSERTED AGAINST THE REAL IMPLEMENTATION. ``api/cache.py`` may not import
``models`` (UIAP-01, ``tests/api/test_import_guard.py``), so the one-parameter logistic is
reproduced there as plain arithmetic. A reproduced formula can drift, so it is compared
cell-for-cell against ``models.market_probability.market_home_win_probability`` -- which a TEST
is free to import -- with the SIGN asserted on a favourite, an underdog and a pick'em rather than
assumed.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from api import cache

_CONVERTER_ID = "market_probability_20990101_000000"
_SLOPE = 0.1512435881109338
_WEIGHTS = {"wp": 0.42, "ats": 0.37, "ou": 0.58}
_BLEND_ID = "blend_20990101_000000"


def _blend_payload(**extra: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "blender_version": "3.0",
        "weights": dict(_WEIGHTS),
        "market_probability_artifact_id": _CONVERTER_ID,
        "market_probability_slope_beta": _SLOPE,
    }
    payload.update(extra)
    return payload


def _artifacts(root: Path, payload: dict[str, Any]) -> Path:
    """A tmp artifacts tree: latest.json -> one blend dir, plus the converter it binds."""
    converter = root / _CONVERTER_ID
    converter.mkdir(parents=True)
    (converter / "metadata.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "slope_beta": _SLOPE,
                "walk_forward_slopes": {"2021": 0.14},
                "training_seasons": [2020, 2021],
                "n_games": 10,
                "input_digest": "0" * 64,
                "fitted_at": "2099-01-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    blend = root / _BLEND_ID
    blend.mkdir()
    (blend / "blend_weights.json").write_text(json.dumps(payload), encoding="utf-8")
    (root / "latest.json").write_text(
        json.dumps({"wp": "w", "ats": "a", "ou": "o", "blend": _BLEND_ID}),
        encoding="utf-8",
    )
    return root


# ---------------------------------------------------------------------------
# The second blend implementation is gone
# ---------------------------------------------------------------------------


def test_the_cache_sigmoid_copy_is_gone() -> None:
    assert not hasattr(cache, "_blend_weight_array")


def test_the_retired_classes_are_gone_from_the_model_side() -> None:
    from models import blending

    assert not hasattr(blending, "DynamicBlendWeights")
    assert not hasattr(blending, "SigmoidParams")


# ---------------------------------------------------------------------------
# The spread-to-probability arithmetic mirrors the model implementation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("beta", [_SLOPE, 0.05, 0.30])
def test_the_cache_logistic_matches_the_model_cell_for_cell(beta: float) -> None:
    from models.market_probability import market_home_win_probability

    spreads = np.arange(-17.5, 18.0, 0.5)
    from_cache = cache._market_home_win_probability(spreads, beta)
    from_model = market_home_win_probability(spreads, beta)
    np.testing.assert_allclose(from_cache, from_model, rtol=0, atol=1e-15)


def test_the_sign_convention_is_asserted_not_assumed() -> None:
    """Home-margin scale: POSITIVE spread = home favoured -> above 0.5."""
    favourite, underdog, pickem = cache._market_home_win_probability(
        np.array([7.0, -7.0, 0.0]), _SLOPE
    )
    assert favourite > 0.5
    assert underdog < 0.5
    assert pickem == 0.5
    assert favourite == pytest.approx(0.7428, abs=1e-3)


# ---------------------------------------------------------------------------
# One synthetic game: the cache and the current-week path agree
# ---------------------------------------------------------------------------


def _game() -> dict[str, Any]:
    return {
        "game_id": "2026_W03_BUF@KC",
        "wp_prob": 0.61,
        "ats_prediction": 4.5,
        "ou_prediction": 49.0,
        "spread": 2.5,
        "total": 51.5,
        "ml_home": -140,
        "ml_away": 120,
    }


def test_the_cache_and_the_current_week_path_publish_the_same_blend(
    tmp_path: Path,
) -> None:
    from scripts.generate_current_week_predictions import apply_blending

    artifacts = _artifacts(tmp_path, _blend_payload())
    game = _game()

    predictions = pd.DataFrame(
        [
            {
                k: game[k]
                for k in ("game_id", "wp_prob", "ats_prediction", "ou_prediction")
            }
        ]
    )
    market = pd.DataFrame(
        [{k: game[k] for k in ("game_id", "spread", "total", "ml_home", "ml_away")}]
    )
    model_path = apply_blending(predictions.copy(), market, artifacts, no_blend=False)

    cache_frame = pd.DataFrame(
        [
            {
                "game_id": game["game_id"],
                "wp_prob": game["wp_prob"],
                "ats_prediction": game["ats_prediction"],
                "ou_prediction": game["ou_prediction"],
                "market_spread": game["spread"],
                "market_total": game["total"],
                "ml_home": game["ml_home"],
                "ml_away": game["ml_away"],
                "blended_wp": None,
                "blended_ats": None,
                "blended_ou": None,
            }
        ]
    )
    cache._apply_blend(cache_frame, cache._read_blend_artifact(artifacts))

    for column in ("blended_wp", "blended_ats", "blended_ou"):
        assert float(cache_frame.loc[0, column]) == pytest.approx(
            float(model_path.loc[0, column]), abs=1e-12
        ), column


def test_the_parity_is_not_satisfied_by_a_degenerate_blend(tmp_path: Path) -> None:
    """Non-vacuity: a non-trivial weight and a non-zero slope, so BOTH sides move the number."""
    artifacts = _artifacts(tmp_path, _blend_payload())
    blend = cache._read_blend_artifact(artifacts)
    for weight in blend["weights"].values():
        assert 0.0 < float(weight) < 1.0
    assert float(blend["market_probability_slope_beta"]) != 0.0

    frame = pd.DataFrame(
        [
            {
                "game_id": "g",
                "wp_prob": 0.61,
                "ats_prediction": 4.5,
                "ou_prediction": 49.0,
                "market_spread": 2.5,
                "market_total": 51.5,
                "ml_home": -140,
                "ml_away": 120,
                "blended_wp": None,
                "blended_ats": None,
                "blended_ou": None,
            }
        ]
    )
    cache._apply_blend(frame, blend)
    assert float(frame.loc[0, "blended_ats"]) == pytest.approx(0.37 * 4.5 + 0.63 * 2.5)
    blended_wp = float(frame.loc[0, "blended_wp"])
    assert blended_wp not in (pytest.approx(0.61),)
    market = 1.0 / (1.0 + np.exp(-_SLOPE * 2.5))
    assert min(0.61, market) < blended_wp < max(0.61, market)


# ---------------------------------------------------------------------------
# The WP market input switched from the moneyline to the spread
# ---------------------------------------------------------------------------


def test_a_moneyline_without_a_spread_gets_no_blended_wp(tmp_path: Path) -> None:
    blend = cache._read_blend_artifact(_artifacts(tmp_path, _blend_payload()))
    frame = pd.DataFrame(
        [
            {
                "game_id": "ml_only",
                "wp_prob": 0.6,
                "ats_prediction": 3.0,
                "ou_prediction": 45.0,
                "market_spread": np.nan,
                "market_total": 44.0,
                "ml_home": -150,
                "ml_away": 130,
                "blended_wp": None,
                "blended_ats": None,
                "blended_ou": None,
            },
            {
                "game_id": "spread_only",
                "wp_prob": 0.6,
                "ats_prediction": 3.0,
                "ou_prediction": 45.0,
                "market_spread": 3.0,
                "market_total": 44.0,
                "ml_home": np.nan,
                "ml_away": np.nan,
                "blended_wp": None,
                "blended_ats": None,
                "blended_ou": None,
            },
        ]
    )
    cache._apply_blend(frame, blend)
    assert pd.isna(frame.loc[0, "blended_wp"]), (
        "the moneyline is not a fallback yardstick"
    )
    assert not pd.isna(frame.loc[1, "blended_wp"]), "the spread is the WP market input"


def test_the_current_week_path_also_ignores_a_moneyline_without_a_spread(
    tmp_path: Path,
) -> None:
    from scripts.generate_current_week_predictions import apply_blending

    artifacts = _artifacts(tmp_path, _blend_payload())
    predictions = pd.DataFrame(
        {
            "game_id": ["ml_only", "spread_only"],
            "wp_prob": [0.6, 0.6],
            "ats_prediction": [3.0, 3.0],
            "ou_prediction": [45.0, 45.0],
        }
    )
    market = pd.DataFrame(
        {
            "game_id": ["ml_only", "spread_only"],
            "spread": [np.nan, 3.0],
            "total": [44.0, 44.0],
            "ml_home": [-150, np.nan],
            "ml_away": [130, np.nan],
        }
    )
    result = apply_blending(predictions, market, artifacts, no_blend=False)
    assert pd.isna(result.loc[0, "blended_wp"])
    assert not pd.isna(result.loc[1, "blended_wp"])


# ---------------------------------------------------------------------------
# A dynamic payload is refused loudly on both surfaces
# ---------------------------------------------------------------------------


def test_the_cache_refuses_a_dynamic_section_with_an_unswallowed_type(
    tmp_path: Path,
) -> None:
    artifacts = _artifacts(
        tmp_path, _blend_payload(dynamic={"wp": {"midpoint": 0.2, "steepness": 1.0}})
    )
    with pytest.raises(ValueError, match="dynamic") as caught:
        cache._read_blend_artifact(artifacts)
    assert not isinstance(caught.value, (KeyError, FileNotFoundError))
    assert type(caught.value).__name__ == "RetiredDynamicBlendError"


def test_the_current_week_path_raises_rather_than_falling_back(tmp_path: Path) -> None:
    from models.blending import RetiredDynamicBlendError
    from scripts.generate_current_week_predictions import apply_blending

    artifacts = _artifacts(
        tmp_path, _blend_payload(dynamic={"wp": {"midpoint": 0.2, "steepness": 1.0}})
    )
    predictions = pd.DataFrame(
        {
            "game_id": ["g"],
            "wp_prob": [0.6],
            "ats_prediction": [3.0],
            "ou_prediction": [45.0],
        }
    )
    market = pd.DataFrame(
        {
            "game_id": ["g"],
            "spread": [3.0],
            "total": [44.0],
            "ml_home": [-150],
            "ml_away": [130],
        }
    )
    with pytest.raises(RetiredDynamicBlendError):
        apply_blending(predictions, market, artifacts, no_blend=False)


def test_a_blend_with_no_converter_binding_is_refused_by_the_cache(
    tmp_path: Path,
) -> None:
    """No slope, no spread conversion: refused, not silently served as a NULL WP blend."""
    payload = _blend_payload()
    del payload["market_probability_slope_beta"]
    del payload["market_probability_artifact_id"]
    artifacts = _artifacts(tmp_path, payload)
    with pytest.raises(Exception, match="market_probability_slope_beta") as caught:
        cache._read_blend_artifact(artifacts)
    assert not isinstance(caught.value, (KeyError, FileNotFoundError))
