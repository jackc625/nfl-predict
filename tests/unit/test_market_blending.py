"""Unit tests for MarketBlender class with log-odds and line-space blending.

Tests cover:
- WP blending in log-odds space (logit/expit)
- ATS blending in spread-point space (linear interpolation)
- O/U blending in total-point space (linear interpolation)
- Per-target blend weights via BlendWeights
- DataFrame-level blend_predictions with proper column handling
- Edge cases: boundary probabilities, NaN handling, empty DataFrames
- SigmoidParams and DynamicBlendWeights dataclass validation (until Plan 33.2-24 Task 2b)

DELETED BY RULING (Plan 33.2-24 Task 2, D33.2-10): ``TestDynamicBlending``,
``TestDynamicArtifacts`` and ``TestModeAwareBlending``. They exercised a blender carrying a
week-varying schedule -- the ``dynamic_weights`` constructor parameter, the per-week weight on
every blend and edge path, the ``dynamic`` artifact round-trip and the per-target mode gate.
That shape is REMOVED, not disabled, and ``tests/unit/test_fixed_blend_weight.py`` now asserts
its absence structurally and that ``from_artifacts`` refuses a ``dynamic`` payload by name.
What they asserted about the SURVIVING behaviour -- a fixed-weight blend needs no week or season
and blends at its configured weight -- is carried by ``TestBlendWP`` / ``TestBlendATS`` /
``TestBlendOU`` and ``test_per_target_weights`` below.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.testing import assert_allclose

from models.blending import (
    BlendConfig,
    BlendWeights,
    DynamicBlendWeights,
    MarketBlender,
    MarketProbabilityUnavailable,
    SigmoidParams,
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


# ---------------------------------------------------------------------------
# SigmoidParams tests
# ---------------------------------------------------------------------------


class TestSigmoidParams:
    """Tests for SigmoidParams dataclass validation."""

    def test_valid_construction(self) -> None:
        """SigmoidParams(midpoint=0.5, steepness=1.0) constructs without error."""
        params = SigmoidParams(midpoint=0.5, steepness=1.0)
        assert params.midpoint == 0.5
        assert params.steepness == 1.0

    def test_midpoint_too_low_raises(self) -> None:
        """SigmoidParams rejects midpoint < 0 with ValueError."""
        with pytest.raises(ValueError, match="midpoint"):
            SigmoidParams(midpoint=-0.1, steepness=1.0)

    def test_midpoint_too_high_raises(self) -> None:
        """SigmoidParams rejects midpoint > 1 with ValueError."""
        with pytest.raises(ValueError, match="midpoint"):
            SigmoidParams(midpoint=1.1, steepness=1.0)

    def test_steepness_too_low_raises(self) -> None:
        """SigmoidParams rejects steepness < 0.1 with ValueError."""
        with pytest.raises(ValueError, match="steepness"):
            SigmoidParams(midpoint=0.5, steepness=0.05)

    def test_steepness_too_high_raises(self) -> None:
        """SigmoidParams rejects steepness > 1.5 with ValueError."""
        with pytest.raises(ValueError, match="steepness"):
            SigmoidParams(midpoint=0.5, steepness=2.0)

    def test_boundary_midpoint_zero(self) -> None:
        """SigmoidParams accepts midpoint=0.0."""
        params = SigmoidParams(midpoint=0.0, steepness=0.5)
        assert params.midpoint == 0.0

    def test_boundary_midpoint_one(self) -> None:
        """SigmoidParams accepts midpoint=1.0."""
        params = SigmoidParams(midpoint=1.0, steepness=0.5)
        assert params.midpoint == 1.0

    def test_boundary_steepness_min(self) -> None:
        """SigmoidParams accepts steepness=0.1."""
        params = SigmoidParams(midpoint=0.5, steepness=0.1)
        assert params.steepness == 0.1

    def test_boundary_steepness_max(self) -> None:
        """SigmoidParams accepts steepness=1.5."""
        params = SigmoidParams(midpoint=0.5, steepness=1.5)
        assert params.steepness == 1.5


# ---------------------------------------------------------------------------
# DynamicBlendWeights tests
# ---------------------------------------------------------------------------


class TestDynamicBlendWeights:
    """Tests for DynamicBlendWeights dataclass and get_weight."""

    @pytest.fixture()
    def default_dynamic(self) -> DynamicBlendWeights:
        """Standard DynamicBlendWeights for testing."""
        return DynamicBlendWeights(
            wp=SigmoidParams(midpoint=0.5, steepness=1.0),
            ats=SigmoidParams(midpoint=0.5, steepness=1.0),
            ou=SigmoidParams(midpoint=0.5, steepness=1.0),
        )

    def test_get_weight_week1_in_range(
        self, default_dynamic: DynamicBlendWeights
    ) -> None:
        """get_weight returns float in [0.30, 0.80] for week 1."""
        weight = default_dynamic.get_weight("wp", week=1, season=2022)
        assert isinstance(weight, float)
        assert 0.30 <= weight <= 0.80

    def test_get_weight_week18_in_range(
        self, default_dynamic: DynamicBlendWeights
    ) -> None:
        """get_weight returns float in [0.30, 0.80] for week 18."""
        weight = default_dynamic.get_weight("wp", week=18, season=2022)
        assert isinstance(weight, float)
        assert 0.30 <= weight <= 0.80

    def test_era_pre2021_max_week_17(
        self, default_dynamic: DynamicBlendWeights
    ) -> None:
        """Season 2019 uses max_week=17 (pre-2021 era)."""
        # week 17 should be t=1.0 for pre-2021
        weight_17 = default_dynamic.get_weight("wp", week=17, season=2019)
        # week 18 should clamp to t=1.0 (same as week 17)
        weight_18 = default_dynamic.get_weight("wp", week=18, season=2019)
        assert_allclose(weight_17, weight_18, atol=1e-10)

    def test_era_post2020_max_week_18(
        self, default_dynamic: DynamicBlendWeights
    ) -> None:
        """Season 2022 uses max_week=18 (post-2020 era)."""
        weight = default_dynamic.get_weight("wp", week=18, season=2022)
        assert 0.30 <= weight <= 0.80

    def test_era_boundary_2020_max_week_17(
        self, default_dynamic: DynamicBlendWeights
    ) -> None:
        """Season 2020 -> max_week=17 (last pre-expansion season)."""
        weight_17 = default_dynamic.get_weight("wp", week=17, season=2020)
        weight_18 = default_dynamic.get_weight("wp", week=18, season=2020)
        # Both should be the same since week 18 clamps to max_week=17
        assert_allclose(weight_17, weight_18, atol=1e-10)

    def test_era_boundary_2021_max_week_18(
        self, default_dynamic: DynamicBlendWeights
    ) -> None:
        """Season 2021 -> max_week=18 (first expanded season)."""
        weight_17 = default_dynamic.get_weight("wp", week=17, season=2021)
        weight_18 = default_dynamic.get_weight("wp", week=18, season=2021)
        # week 18 should produce higher weight than week 17
        assert weight_18 >= weight_17

    def test_playoff_clamping(self, default_dynamic: DynamicBlendWeights) -> None:
        """Playoff weeks (> max_week) clamped to t=1.0 -- same as max_week."""
        weight_18 = default_dynamic.get_weight("wp", week=18, season=2022)
        weight_20 = default_dynamic.get_weight("wp", week=20, season=2022)
        assert_allclose(weight_18, weight_20, atol=1e-10)

    def test_week_zero_raises(self, default_dynamic: DynamicBlendWeights) -> None:
        """Week 0 raises ValueError."""
        with pytest.raises(ValueError, match="week must be >= 1"):
            default_dynamic.get_weight("wp", week=0, season=2022)

    def test_all_targets_return_valid_weights(
        self, default_dynamic: DynamicBlendWeights
    ) -> None:
        """All three targets (wp, ats, ou) return valid weights."""
        for target in ("wp", "ats", "ou"):
            weight = default_dynamic.get_weight(target, week=9, season=2022)
            assert isinstance(weight, float)
            assert 0.30 <= weight <= 0.80

    def test_deterministic_known_value(self) -> None:
        """Known deterministic value: midpoint=0.5, steepness=1.0, week=9, season=2022.

        t = 9/18 = 0.5
        weight = 0.30 + 0.50 / (1 + exp(-1.0 * (0.5 - 0.5)))
               = 0.30 + 0.50 / (1 + 1)
               = 0.30 + 0.25
               = 0.55
        """
        dw = DynamicBlendWeights(
            wp=SigmoidParams(midpoint=0.5, steepness=1.0),
            ats=SigmoidParams(midpoint=0.5, steepness=1.0),
            ou=SigmoidParams(midpoint=0.5, steepness=1.0),
        )
        weight = dw.get_weight("wp", week=9, season=2022)
        assert_allclose(weight, 0.55, atol=1e-10)

    def test_to_dict_round_trip(self) -> None:
        """to_dict produces serializable dict that from_dict can reconstruct."""
        original = DynamicBlendWeights(
            wp=SigmoidParams(midpoint=0.45, steepness=0.8),
            ats=SigmoidParams(midpoint=0.50, steepness=1.0),
            ou=SigmoidParams(midpoint=0.40, steepness=0.6),
            mode_by_target={"wp": "dynamic", "ats": "static", "ou": "dynamic"},
        )
        data = original.to_dict()
        restored = DynamicBlendWeights.from_dict(data)
        assert restored.wp.midpoint == original.wp.midpoint
        assert restored.wp.steepness == original.wp.steepness
        assert restored.ats.midpoint == original.ats.midpoint
        assert restored.ou.steepness == original.ou.steepness
        assert restored.low == original.low
        assert restored.high == original.high
        assert restored.mode_by_target == original.mode_by_target

    def test_from_dict_missing_target_raises(self) -> None:
        """from_dict with missing target raises ValueError."""
        data = {
            "wp": {"midpoint": 0.5, "steepness": 1.0},
            "ats": {"midpoint": 0.5, "steepness": 1.0},
            # "ou" missing
        }
        with pytest.raises(ValueError, match="Invalid dynamic blend weights"):
            DynamicBlendWeights.from_dict(data)

    def test_from_dict_invalid_type_raises(self) -> None:
        """from_dict with invalid param type raises ValueError."""
        data = {
            "wp": {"midpoint": "invalid", "steepness": 1.0},
            "ats": {"midpoint": 0.5, "steepness": 1.0},
            "ou": {"midpoint": 0.5, "steepness": 1.0},
        }
        with pytest.raises(ValueError, match="Invalid dynamic blend weights"):
            DynamicBlendWeights.from_dict(data)


# ---------------------------------------------------------------------------
# Hypothesis property-based sigmoid tests
# ---------------------------------------------------------------------------


class TestSigmoidProperties:
    """Hypothesis property-based tests for sigmoid weight bounds and monotonicity."""

    @given(
        week=st.integers(min_value=1, max_value=22),
        season=st.integers(min_value=2010, max_value=2025),
        midpoint=st.floats(min_value=0.15, max_value=0.82, allow_nan=False),
        steepness=st.floats(min_value=0.1, max_value=1.5, allow_nan=False),
    )
    @settings(max_examples=200)
    def test_weight_always_in_bounds(
        self,
        week: int,
        season: int,
        midpoint: float,
        steepness: float,
    ) -> None:
        """Sigmoid weight is always in [0.30, 0.80] for any valid inputs."""
        dw = DynamicBlendWeights(
            wp=SigmoidParams(midpoint=midpoint, steepness=steepness),
            ats=SigmoidParams(midpoint=0.5, steepness=1.0),
            ou=SigmoidParams(midpoint=0.5, steepness=1.0),
        )
        weight = dw.get_weight("wp", week=week, season=season)
        assert 0.30 <= weight <= 0.80, (
            f"weight={weight} out of bounds for week={week}, season={season}, "
            f"midpoint={midpoint}, steepness={steepness}"
        )

    @given(
        w1=st.integers(min_value=1, max_value=21),
        season=st.integers(min_value=2010, max_value=2025),
        midpoint=st.floats(min_value=0.15, max_value=0.82, allow_nan=False),
        steepness=st.floats(min_value=0.1, max_value=1.5, allow_nan=False),
    )
    @settings(max_examples=200)
    def test_monotonically_nondecreasing_in_week(
        self,
        w1: int,
        season: int,
        midpoint: float,
        steepness: float,
    ) -> None:
        """Weight is monotonically non-decreasing in week for fixed params."""
        w2 = w1 + 1
        dw = DynamicBlendWeights(
            wp=SigmoidParams(midpoint=midpoint, steepness=steepness),
            ats=SigmoidParams(midpoint=0.5, steepness=1.0),
            ou=SigmoidParams(midpoint=0.5, steepness=1.0),
        )
        weight_1 = dw.get_weight("wp", week=w1, season=season)
        weight_2 = dw.get_weight("wp", week=w2, season=season)
        assert weight_2 >= weight_1 - 1e-12, (
            f"Non-monotonic: week {w1} -> {weight_1}, week {w2} -> {weight_2}"
        )
