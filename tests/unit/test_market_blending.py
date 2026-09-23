"""Unit tests for MarketBlender class with log-odds and line-space blending.

Tests cover:
- WP blending in log-odds space (logit/expit)
- ATS blending in spread-point space (linear interpolation)
- O/U blending in total-point space (linear interpolation)
- Per-target blend weights via BlendWeights
- DataFrame-level blend_predictions with proper column handling
- Edge cases: boundary probabilities, NaN handling, empty DataFrames

DELETED BY RULING (Plan 33.2-24 Task 2, D33.2-10): ``TestDynamicBlending``,
``TestDynamicArtifacts`` and ``TestModeAwareBlending``. They exercised a blender carrying a
week-varying schedule -- the ``dynamic_weights`` constructor parameter, the per-week weight on
every blend and edge path, the ``dynamic`` artifact round-trip and the per-target mode gate.
That shape is REMOVED, not disabled, and ``tests/unit/test_fixed_blend_weight.py`` now asserts
its absence structurally and that ``from_artifacts`` refuses a ``dynamic`` payload by name.
What they asserted about the SURVIVING behaviour -- a fixed-weight blend needs no week or season
and blends at its configured weight -- is carried by ``TestBlendWP`` / ``TestBlendATS`` /
``TestBlendOU`` and ``test_per_target_weights`` below.

DELETED BY RULING (Plan 33.2-24 Task 2b): ``TestSigmoidParams`` (9), ``TestDynamicBlendWeights``
(13) and ``TestSigmoidProperties`` (2 Hypothesis properties). They validated the two dataclasses
that described the week-varying schedule, and those classes are deleted together with the cache's
arithmetic copy of the same sigmoid. The four test calls that passed ``week=`` / ``season=`` into
``blend_wp`` / ``blend_ats`` / ``blend_ou`` lived in the dynamic classes removed in Task 2; the
methods now take neither, pinned by ``test_the_blend_methods_take_no_week_or_season``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from models.blending import (
    BlendConfig,
    BlendWeights,
    MarketBlender,
    MarketProbabilityUnavailable,
)

# ---------------------------------------------------------------------------
# One fixed weight: the blend methods take no week or season (Plan 33.2-24 Task 2b)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["blend_wp", "blend_ats", "blend_ou"])
def test_the_blend_methods_take_no_week_or_season(method: str) -> None:
    """The weight no longer varies by week, so a week or season parameter would be a hook."""
    import inspect

    parameters = list(inspect.signature(getattr(MarketBlender, method)).parameters)
    assert parameters[:1] == ["self"]  # non-vacuity: the signature was read
    assert "week" not in parameters
    assert "season" not in parameters


# ---------------------------------------------------------------------------
# BlendWeights tests
# ---------------------------------------------------------------------------


class TestBlendWeights:
    """Tests for BlendWeights dataclass validation and defaults."""

    def test_default_weights(self) -> None:
        """BlendWeights defaults to 0.60 for all three targets."""
        weights = BlendWeights()
        assert weights.wp_model_weight == 0.60
        assert weights.ats_model_weight == 0.60
        assert weights.ou_model_weight == 0.60

    def test_valid_custom_weights(self) -> None:
        """BlendWeights accepts custom weights in [0.0, 1.0]."""
        weights = BlendWeights(
            wp_model_weight=0.3,
            ats_model_weight=0.7,
            ou_model_weight=0.5,
        )
        assert weights.wp_model_weight == 0.3
        assert weights.ats_model_weight == 0.7
        assert weights.ou_model_weight == 0.5

    def test_weight_too_high_raises(self) -> None:
        """BlendWeights rejects weight > 1.0."""
        with pytest.raises(ValueError, match=r"in \[0.0, 1.0\]"):
            BlendWeights(wp_model_weight=1.5)

    def test_weight_too_low_raises(self) -> None:
        """BlendWeights rejects weight < 0.0."""
        with pytest.raises(ValueError, match=r"in \[0.0, 1.0\]"):
            BlendWeights(ats_model_weight=-0.1)

    def test_boundary_weights_valid(self) -> None:
        """BlendWeights accepts 0.0 and 1.0 as valid boundaries."""
        weights = BlendWeights(
            wp_model_weight=0.0,
            ats_model_weight=1.0,
            ou_model_weight=0.5,
        )
        assert weights.wp_model_weight == 0.0
        assert weights.ats_model_weight == 1.0


# ---------------------------------------------------------------------------
# WP blending (log-odds space)
# ---------------------------------------------------------------------------


class TestBlendWP:
    """Tests for blend_wp using logit/expit log-odds blending."""

    def test_blend_wp_basic(self) -> None:
        """blend_wp with model=0.7, market=0.6, weight=0.6 produces result in (0.6, 0.7)."""
        config = BlendConfig(weights=BlendWeights(wp_model_weight=0.6))
        blender = MarketBlender(config=config)
        result = blender.blend_wp(
            np.array([0.7]),
            np.array([0.6]),
        )
        assert result[0] > 0.6
        assert result[0] < 0.7

    def test_blend_wp_equal_probabilities(self) -> None:
        """blend_wp with model=0.5, market=0.5, any weight produces exactly 0.5."""
        blender = MarketBlender()
        result = blender.blend_wp(
            np.array([0.5]),
            np.array([0.5]),
        )
        assert_allclose(result, [0.5], atol=1e-10)

    def test_blend_wp_weight_one_returns_model(self) -> None:
        """blend_wp with weight=1.0 returns model_prob."""
        config = BlendConfig(weights=BlendWeights(wp_model_weight=1.0))
        blender = MarketBlender(config=config)
        result = blender.blend_wp(
            np.array([0.75]),
            np.array([0.45]),
        )
        assert_allclose(result, [0.75], atol=1e-6)

    def test_blend_wp_weight_zero_returns_market(self) -> None:
        """blend_wp with weight=0.0 returns market_prob."""
        config = BlendConfig(weights=BlendWeights(wp_model_weight=0.0))
        blender = MarketBlender(config=config)
        result = blender.blend_wp(
            np.array([0.75]),
            np.array([0.45]),
        )
        assert_allclose(result, [0.45], atol=1e-6)

    def test_blend_wp_clips_before_logit(self) -> None:
        """blend_wp clips probabilities to [0.001, 0.999] before logit (no NaN/inf)."""
        blender = MarketBlender()
        result = blender.blend_wp(
            np.array([0.001]),
            np.array([0.999]),
        )
        # Should not produce NaN or inf
        assert np.isfinite(result[0])
        assert 0.0 < result[0] < 1.0

    def test_blend_wp_extreme_inputs_no_nan(self) -> None:
        """blend_wp with model_prob=0.001 and market_prob=0.999 does not produce NaN."""
        blender = MarketBlender()
        result = blender.blend_wp(
            np.array([0.001]),
            np.array([0.999]),
        )
        assert not np.isnan(result[0])
        assert np.isfinite(result[0])

    def test_blend_wp_vectorized(self) -> None:
        """blend_wp works on arrays of multiple values."""
        blender = MarketBlender()
        model = np.array([0.7, 0.3, 0.5])
        market = np.array([0.6, 0.4, 0.5])
        result = blender.blend_wp(model, market)
        assert result.shape == (3,)
        assert np.all(np.isfinite(result))

    def test_blend_wp_output_valid_probabilities(self) -> None:
        """blend_wp always produces valid probabilities in [0, 1]."""
        blender = MarketBlender()
        model = np.array([0.01, 0.5, 0.99])
        market = np.array([0.99, 0.5, 0.01])
        result = blender.blend_wp(model, market)
        assert np.all(result >= 0.0)
        assert np.all(result <= 1.0)


# ---------------------------------------------------------------------------
# ATS blending (spread-point space)
# ---------------------------------------------------------------------------


class TestBlendATS:
    """Tests for blend_ats using linear interpolation in spread space."""

    def test_blend_ats_basic(self) -> None:
        """blend_ats with model=-7, market=-3, weight=0.6 returns -5.4."""
        config = BlendConfig(weights=BlendWeights(ats_model_weight=0.6))
        blender = MarketBlender(config=config)
        result = blender.blend_ats(
            np.array([-7.0]),
            np.array([-3.0]),
        )
        # -7*0.6 + -3*0.4 = -4.2 + -1.2 = -5.4
        assert_allclose(result, [-5.4], atol=1e-10)

    def test_blend_ats_weight_one(self) -> None:
        """blend_ats with weight=1.0 returns model_spread."""
        config = BlendConfig(weights=BlendWeights(ats_model_weight=1.0))
        blender = MarketBlender(config=config)
        result = blender.blend_ats(
            np.array([-7.0]),
            np.array([-3.0]),
        )
        assert_allclose(result, [-7.0])

    def test_blend_ats_weight_zero(self) -> None:
        """blend_ats with weight=0.0 returns market_spread."""
        config = BlendConfig(weights=BlendWeights(ats_model_weight=0.0))
        blender = MarketBlender(config=config)
        result = blender.blend_ats(
            np.array([-7.0]),
            np.array([-3.0]),
        )
        assert_allclose(result, [-3.0])


# ---------------------------------------------------------------------------
# O/U blending (total-point space)
# ---------------------------------------------------------------------------


class TestBlendOU:
    """Tests for blend_ou using linear interpolation in total space."""

    def test_blend_ou_basic(self) -> None:
        """blend_ou with model=48, market=44, weight=0.6 returns 46.4."""
        config = BlendConfig(weights=BlendWeights(ou_model_weight=0.6))
        blender = MarketBlender(config=config)
        result = blender.blend_ou(
            np.array([48.0]),
            np.array([44.0]),
        )
        # 48*0.6 + 44*0.4 = 28.8 + 17.6 = 46.4
        assert_allclose(result, [46.4], atol=1e-10)

    def test_blend_ou_weight_one(self) -> None:
        """blend_ou with weight=1.0 returns model_total."""
        config = BlendConfig(weights=BlendWeights(ou_model_weight=1.0))
        blender = MarketBlender(config=config)
        result = blender.blend_ou(
            np.array([48.0]),
            np.array([44.0]),
        )
        assert_allclose(result, [48.0])


# -- blend_predictions (DataFrame-level) --


class TestBlendPredictions:
    """Tests for blend_predictions DataFrame blending."""

    @pytest.fixture()
    def sample_predictions_wp(self) -> pd.DataFrame:
        """Sample WP predictions DataFrame."""
        return pd.DataFrame(
            {
                "game_id": ["2023_01_KC_DET", "2023_01_BUF_NYJ"],
                "season": [2023, 2023],
                "week": [1, 1],
                "model_prob": [0.65, 0.55],
                "actual": [1, 0],
            }
        )

    @pytest.fixture()
    def sample_market_wp(self) -> pd.DataFrame:
        """Sample market odds DataFrame for WP."""
        return pd.DataFrame(
            {
                "game_id": ["2023_01_KC_DET", "2023_01_BUF_NYJ"],
                "ml_home": [-150, -120],
                "ml_away": [130, 100],
                "spread": [-3.0, -1.5],
                "total": [52.5, 44.0],
            }
        )

    @pytest.fixture()
    def sample_market_wp_moneyline_only(self) -> pd.DataFrame:
        """A market frame carrying ONLY the closing moneyline the WP blend used to devig.

        Plan 33.2-21: the WP half now takes its market opinion from the pre-lock SPREAD
        through the fitted converter, so a moneyline-only frame supplies no market
        opinion at all -- and a blend with no market opinion refuses by name rather than
        quietly returning the unblended model probability.
        """
        return pd.DataFrame(
            {
                "game_id": ["2023_01_KC_DET", "2023_01_BUF_NYJ"],
                "ml_home": [-150, -120],
                "ml_away": [130, 100],
            }
        )

    @pytest.fixture()
    def bound_blender(self) -> MarketBlender:
        """A blender with a converter BOUND to it, as Plan 33.2-24 will write one.

        The slope is the production fit's (``market_probability_20260923_025709``,
        measured over 1,342 graded 2020-2024 games).
        """
        return MarketBlender(
            market_probability_artifact_id="market_probability_20260923_025709",
            market_probability_slope_beta=0.1512435881109338,
        )

    @pytest.fixture()
    def sample_predictions_ats(self) -> pd.DataFrame:
        """Sample ATS predictions DataFrame."""
        return pd.DataFrame(
            {
                "game_id": ["2023_01_KC_DET", "2023_01_BUF_NYJ"],
                "season": [2023, 2023],
                "week": [1, 1],
                "model_spread": [-5.0, -2.5],
                "actual": [-3.0, 7.0],
            }
        )

    @pytest.fixture()
    def sample_predictions_ou(self) -> pd.DataFrame:
        """Sample O/U predictions DataFrame."""
        return pd.DataFrame(
            {
                "game_id": ["2023_01_KC_DET", "2023_01_BUF_NYJ"],
                "season": [2023, 2023],
                "week": [1, 1],
                "model_total": [50.0, 42.0],
                "actual": [48.0, 38.0],
            }
        )

    def test_blend_predictions_wp_modifies_model_prob(
        self,
        sample_predictions_wp: pd.DataFrame,
        sample_market_wp: pd.DataFrame,
        bound_blender: MarketBlender,
    ) -> None:
        """blend_predictions with WP target modifies model_prob column.

        Plan 33.2-21: the market opinion now comes from the pre-lock spread through the
        blender's BOUND converter, not from a devigged closing moneyline.
        """
        result = bound_blender.blend_predictions(
            sample_predictions_wp, sample_market_wp, target="wp"
        )
        # model_prob should be different from original (blended with market)
        assert "model_prob" in result.columns
        # The blended values should differ from the originals (unless equal)
        original_probs = sample_predictions_wp["model_prob"].values
        blended_probs = result["model_prob"].values
        # With default weight 0.6, blended should differ from original
        assert not np.allclose(original_probs, blended_probs)

    def test_blend_predictions_wp_refuses_a_moneyline_only_market(
        self,
        sample_predictions_wp: pd.DataFrame,
        sample_market_wp_moneyline_only: pd.DataFrame,
        bound_blender: MarketBlender,
    ) -> None:
        """A market frame with no pre-lock spread supplies no opinion, so the blend REFUSES.

        This case used to log a warning and return the unblended model probability. A
        silent no-blend is indistinguishable from a blend with weight zero (Plan 33.2-21).
        """
        with pytest.raises(MarketProbabilityUnavailable):
            bound_blender.blend_predictions(
                sample_predictions_wp, sample_market_wp_moneyline_only, target="wp"
            )

    def test_blend_predictions_ats_modifies_model_spread(
        self,
        sample_predictions_ats: pd.DataFrame,
        sample_market_wp: pd.DataFrame,
    ) -> None:
        """blend_predictions with ATS target modifies model_spread column."""
        blender = MarketBlender()
        result = blender.blend_predictions(
            sample_predictions_ats, sample_market_wp, target="ats"
        )
        assert "model_spread" in result.columns
        original = sample_predictions_ats["model_spread"].values
        blended = result["model_spread"].values
        assert not np.allclose(original, blended)

    def test_blend_predictions_ou_modifies_model_total(
        self,
        sample_predictions_ou: pd.DataFrame,
        sample_market_wp: pd.DataFrame,
    ) -> None:
        """blend_predictions with O/U target modifies model_total column."""
        blender = MarketBlender()
        result = blender.blend_predictions(
            sample_predictions_ou, sample_market_wp, target="ou"
        )
        assert "model_total" in result.columns
        original = sample_predictions_ou["model_total"].values
        blended = result["model_total"].values
        assert not np.allclose(original, blended)

    def test_blend_predictions_preserves_other_columns(
        self,
        sample_predictions_wp: pd.DataFrame,
        sample_market_wp: pd.DataFrame,
        bound_blender: MarketBlender,
    ) -> None:
        """blend_predictions preserves DataFrame columns (game_id, season, week, etc.)."""
        result = bound_blender.blend_predictions(
            sample_predictions_wp, sample_market_wp, target="wp"
        )
        # All original columns should still be present
        for col in ["game_id", "season", "week", "actual"]:
            assert col in result.columns
        # Values of non-prediction columns should be unchanged
        pd.testing.assert_series_equal(
            result["game_id"], sample_predictions_wp["game_id"]
        )
        pd.testing.assert_series_equal(
            result["season"], sample_predictions_wp["season"]
        )

    def test_per_target_weights(self) -> None:
        """blend_predictions applies per-target weights independently."""
        config = BlendConfig(
            weights=BlendWeights(
                wp_model_weight=0.8,
                ats_model_weight=0.4,
                ou_model_weight=0.3,
            )
        )
        blender = MarketBlender(config=config)

        # Use the WP weight for WP blend
        wp_result = blender.blend_wp(
            np.array([0.7]),
            np.array([0.5]),
        )
        # The WP result should lean towards model (0.8 weight)
        assert wp_result[0] > 0.6

        # Use the ATS weight for ATS blend
        ats_result = blender.blend_ats(
            np.array([-7.0]),
            np.array([-3.0]),
        )
        # ATS weight = 0.4, so: -7*0.4 + -3*0.6 = -2.8 + -1.8 = -4.6
        assert_allclose(ats_result, [-4.6], atol=1e-10)

        # Use the O/U weight for O/U blend
        ou_result = blender.blend_ou(
            np.array([48.0]),
            np.array([44.0]),
        )
        # OU weight = 0.3, so: 48*0.3 + 44*0.7 = 14.4 + 30.8 = 45.2
        assert_allclose(ou_result, [45.2], atol=1e-10)

    def test_blend_predictions_returns_copy(
        self,
        sample_predictions_wp: pd.DataFrame,
        sample_market_wp: pd.DataFrame,
        bound_blender: MarketBlender,
    ) -> None:
        """blend_predictions returns a copy, not the original DataFrame."""
        result = bound_blender.blend_predictions(
            sample_predictions_wp, sample_market_wp, target="wp"
        )
        assert result is not sample_predictions_wp

    def test_blend_predictions_empty_df(self) -> None:
        """blend_predictions handles empty DataFrames gracefully."""
        blender = MarketBlender()
        empty_preds = pd.DataFrame(columns=["game_id", "model_prob", "season", "week"])
        empty_market = pd.DataFrame(
            columns=["game_id", "ml_home", "ml_away", "spread", "total"]
        )
        result = blender.blend_predictions(empty_preds, empty_market, target="wp")
        assert len(result) == 0
