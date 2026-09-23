"""The WP blend refuses by name, and its converter is bound (Plan 33.2-21 Task 2).

TWO PROPERTIES, and they are different properties.

**The refusal.** ``MarketBlender._blend_wp_predictions`` used to log a warning and RETURN
when ``ml_home`` / ``ml_away`` were absent, leaving ``model_prob`` untouched. A silent
no-blend is indistinguishable from a blend with weight zero, so a blend that quietly stops
blending is a model change nobody can see. It now RAISES
``MarketProbabilityUnavailable``, whose base class the repository's degrade-quietly catch
tuples do not catch -- the same reasoning ``data.sealed_probe_log.SealedProbeLogCorrupt``
records. A caller that genuinely wants the unblended probability asks for it by name.

**The binding.** ``artifacts/latest.json`` names four artifacts under SPEC R13 and the
converter is deliberately not a fifth, while ``from_artifacts`` resolves only the blend
directory. Without a binding, nothing guarantees that the converter a blend was TUNED
against is the converter it is SERVED with (reviews round ``f924749``, Codex HIGH). So the
blend carries ``market_probability_artifact_id`` and ``market_probability_slope_beta``,
``from_artifacts`` cross-checks them against the immutable converter directory, and
``_blend_wp_predictions`` converts with the BOUND slope and nothing else. The test that
matters most here is the last one: a NEWER converter directory planted beside the bound
one does not move a bound blender's output by so much as a float.

Everything writes to ``tmp_path``. ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models.blending import (
    BlendConfig,
    BlendWeights,
    MarketBlender,
    MarketProbabilityBindingError,
    MarketProbabilityUnavailable,
)
from models.market_probability import market_home_win_probability

#: The slope the production fit measured on the owned pre-lock corpus.
_BOUND_SLOPE = 0.1512435881109338

#: A DIFFERENT slope, used for the planted second converter directory.
_OTHER_SLOPE = 0.170


def _converter_dir(root: Path, name: str, slope: float) -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "metadata.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "slope_beta": slope,
                "walk_forward_slopes": {"2021": slope},
                "training_seasons": [2020, 2021],
                "n_games": 500,
                "input_digest": "0" * 64,
                "fitted_at": "2026-09-22T00:00:00+00:00",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return directory


def _blend_dir(
    root: Path,
    name: str = "blend_20260922_000000",
    *,
    converter_id: str | None = None,
    converter_slope: float | None = None,
) -> Path:
    directory = root / name
    directory.mkdir(parents=True)
    payload: dict[str, object] = {
        "blender_version": "1.0",
        "weights": {"wp": 0.60, "ats": 0.60, "ou": 0.60},
        "edge_thresholds": {"wp": 0.03, "ats": 1.5, "ou": 1.5},
    }
    if converter_id is not None:
        payload["market_probability_artifact_id"] = converter_id
    if converter_slope is not None:
        payload["market_probability_slope_beta"] = converter_slope
    (directory / "blend_weights.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    (root / "latest.json").write_text(json.dumps({"blend": name}), encoding="utf-8")
    return directory


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["2023_W01_KC@DET", "2023_W01_BUF@NYJ"],
            "season": [2023, 2023],
            "week": [1, 1],
            "model_prob": [0.65, 0.55],
        }
    )


def _prelock_lines() -> pd.DataFrame:
    """Pre-lock spreads on the home-margin convention: POSITIVE = home favoured."""
    return pd.DataFrame(
        {
            "game_id": ["2023_W01_KC@DET", "2023_W01_BUF@NYJ"],
            "spread": [3.0, -2.5],
        }
    )


def _bound_blender() -> MarketBlender:
    return MarketBlender(
        market_probability_artifact_id="market_probability_20260923_025709",
        market_probability_slope_beta=_BOUND_SLOPE,
    )


# ---------------------------------------------------------------------------
# The refusal
# ---------------------------------------------------------------------------


class TestTheRefusal:
    def test_a_present_market_blends_between_the_two_opinions(self) -> None:
        preds = _predictions()
        result = _bound_blender().blend_predictions(preds, _prelock_lines(), "wp")

        market = market_home_win_probability(
            _prelock_lines()["spread"].to_numpy(dtype=float), _BOUND_SLOPE
        )
        for i in range(len(preds)):
            low = min(preds.loc[i, "model_prob"], market[i])
            high = max(preds.loc[i, "model_prob"], market[i])
            assert low < result.loc[i, "model_prob"] < high

    def test_the_present_market_case_actually_moves_the_probability(self) -> None:
        """The non-vacuity control.

        Without it the refusal test below would pass just as happily against a blend that
        does nothing at all, which is the very state this task exists to abolish.
        """
        preds = _predictions()
        result = _bound_blender().blend_predictions(preds, _prelock_lines(), "wp")
        assert not np.allclose(
            result["model_prob"].to_numpy(), preds["model_prob"].to_numpy()
        )

    def test_an_absent_market_raises_and_names_the_game(self) -> None:
        moneyline_only = pd.DataFrame(
            {
                "game_id": ["2023_W01_KC@DET", "2023_W01_BUF@NYJ"],
                "ml_home": [-150, -120],
                "ml_away": [130, 100],
            }
        )
        with pytest.raises(MarketProbabilityUnavailable) as excinfo:
            _bound_blender().blend_predictions(_predictions(), moneyline_only, "wp")
        assert "2023_W01_KC@DET" in str(excinfo.value)

    def test_a_game_with_no_prelock_spread_raises_rather_than_passing_through(
        self,
    ) -> None:
        partial = pd.DataFrame(
            {"game_id": ["2023_W01_KC@DET"], "spread": [3.0]},
        )
        with pytest.raises(MarketProbabilityUnavailable) as excinfo:
            _bound_blender().blend_predictions(_predictions(), partial, "wp")
        # The game the left merge could not match is the one named.
        assert "2023_W01_BUF@NYJ" in str(excinfo.value)

    def test_a_blender_with_no_bound_converter_raises(self) -> None:
        with pytest.raises(MarketProbabilityUnavailable) as excinfo:
            MarketBlender().blend_predictions(_predictions(), _prelock_lines(), "wp")
        assert "market_probability_slope_beta" in str(excinfo.value)

    def test_the_refusal_is_outside_the_degrade_quietly_catch_tuples(self) -> None:
        assert not issubclass(MarketProbabilityUnavailable, ValueError)
        assert not issubclass(MarketProbabilityUnavailable, KeyError)
        assert not issubclass(MarketProbabilityUnavailable, RuntimeError)
        assert not issubclass(MarketProbabilityUnavailable, ImportError)

        caught = False
        try:
            try:
                MarketBlender().blend_predictions(
                    _predictions(), _prelock_lines(), "wp"
                )
            except (ValueError, KeyError):
                caught = True
        except MarketProbabilityUnavailable:
            pass
        assert not caught, (
            "a bare `except (ValueError, KeyError)` swallowed the refusal, which is "
            "exactly the degrade-quietly path this exception's base class avoids"
        )

    def test_the_unblended_probability_has_an_explicit_named_path(self) -> None:
        preds = _predictions()
        unblended = MarketBlender().unblended_wp_predictions(preds)
        np.testing.assert_array_equal(
            unblended["model_prob"].to_numpy(), preds["model_prob"].to_numpy()
        )
        assert unblended is not preds


# ---------------------------------------------------------------------------
# The binding
# ---------------------------------------------------------------------------


class TestTheConverterBinding:
    def test_from_artifacts_carries_both_binding_fields(self, tmp_path: Path) -> None:
        _converter_dir(tmp_path, "market_probability_20260101_000000", _BOUND_SLOPE)
        _blend_dir(
            tmp_path,
            converter_id="market_probability_20260101_000000",
            converter_slope=_BOUND_SLOPE,
        )
        blender = MarketBlender.from_artifacts(artifacts_dir=tmp_path)
        assert (
            blender.market_probability_artifact_id
            == "market_probability_20260101_000000"
        )
        assert blender.market_probability_slope_beta == _BOUND_SLOPE

    def test_a_slope_mismatch_against_the_directory_raises(
        self, tmp_path: Path
    ) -> None:
        _converter_dir(tmp_path, "market_probability_20260101_000000", _BOUND_SLOPE)
        _blend_dir(
            tmp_path,
            converter_id="market_probability_20260101_000000",
            converter_slope=_OTHER_SLOPE,
        )
        with pytest.raises(MarketProbabilityBindingError) as excinfo:
            MarketBlender.from_artifacts(artifacts_dir=tmp_path)
        assert "market_probability_20260101_000000" in str(excinfo.value)

    def test_a_named_converter_with_no_directory_raises(self, tmp_path: Path) -> None:
        _blend_dir(
            tmp_path,
            converter_id="market_probability_absent",
            converter_slope=_BOUND_SLOPE,
        )
        with pytest.raises(MarketProbabilityBindingError) as excinfo:
            MarketBlender.from_artifacts(artifacts_dir=tmp_path)
        assert "market_probability_absent" in str(excinfo.value)

    def test_half_a_binding_raises(self, tmp_path: Path) -> None:
        _converter_dir(tmp_path, "market_probability_20260101_000000", _BOUND_SLOPE)
        _blend_dir(tmp_path, converter_id="market_probability_20260101_000000")
        with pytest.raises(MarketProbabilityBindingError):
            MarketBlender.from_artifacts(artifacts_dir=tmp_path)

    def test_a_blend_with_neither_key_loads_and_then_refuses(
        self, tmp_path: Path
    ) -> None:
        """The live incumbent ``blend_dynamic_20260606_020635`` carries neither key.

        It must LOAD unchanged -- nothing that reads today's blend may break at load time
        -- and then refuse at the point where it would have had to invent a market
        opinion. That is the whole shape of this plan: the refusal is visible, and it is
        Plan 33.2-24 that supplies the bound blend.
        """
        _blend_dir(tmp_path)
        blender = MarketBlender.from_artifacts(artifacts_dir=tmp_path)
        assert blender.market_probability_artifact_id is None
        assert blender.market_probability_slope_beta is None

        with pytest.raises(MarketProbabilityUnavailable):
            blender.blend_predictions(_predictions(), _prelock_lines(), "wp")

    def test_the_binding_error_is_outside_the_degrade_quietly_catch_tuples(
        self,
    ) -> None:
        assert not issubclass(MarketProbabilityBindingError, ValueError)
        assert not issubclass(MarketProbabilityBindingError, KeyError)
        assert not issubclass(MarketProbabilityBindingError, RuntimeError)

    def test_a_newer_planted_converter_does_not_move_a_bound_blender(
        self, tmp_path: Path
    ) -> None:
        """Serving cannot drift to a different converter. This is the point of binding."""
        _converter_dir(tmp_path, "market_probability_20260101_000000", _BOUND_SLOPE)
        _blend_dir(
            tmp_path,
            converter_id="market_probability_20260101_000000",
            converter_slope=_BOUND_SLOPE,
        )
        blender = MarketBlender.from_artifacts(artifacts_dir=tmp_path)
        before = blender.blend_predictions(_predictions(), _prelock_lines(), "wp")

        # A LATER, different converter appears beside the bound one.
        _converter_dir(tmp_path, "market_probability_29991231_235959", _OTHER_SLOPE)

        reloaded = MarketBlender.from_artifacts(artifacts_dir=tmp_path)
        after = reloaded.blend_predictions(_predictions(), _prelock_lines(), "wp")

        np.testing.assert_array_equal(
            before["model_prob"].to_numpy(), after["model_prob"].to_numpy()
        )
        assert (
            reloaded.market_probability_artifact_id
            == "market_probability_20260101_000000"
        )

    def test_the_bound_slope_is_what_converts(self, tmp_path: Path) -> None:
        """Not a module default, not the newest directory -- the bound number."""
        _converter_dir(tmp_path, "market_probability_20260101_000000", _OTHER_SLOPE)
        _blend_dir(
            tmp_path,
            converter_id="market_probability_20260101_000000",
            converter_slope=_OTHER_SLOPE,
        )
        blender = MarketBlender.from_artifacts(artifacts_dir=tmp_path)
        preds = _predictions()
        result = blender.blend_predictions(preds, _prelock_lines(), "wp")

        expected = MarketBlender(
            config=BlendConfig(weights=BlendWeights()),
            market_probability_artifact_id="market_probability_20260101_000000",
            market_probability_slope_beta=_OTHER_SLOPE,
        ).blend_predictions(preds, _prelock_lines(), "wp")

        np.testing.assert_array_equal(
            result["model_prob"].to_numpy(), expected["model_prob"].to_numpy()
        )
