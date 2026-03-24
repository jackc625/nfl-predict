"""Market blending module for combining model predictions with market odds.

Provides:
- BlendWeights: Per-target model weight configuration (WP, ATS, O/U)
- BlendConfig: Full blending configuration with probability clipping
- MarketBlender: Core blending logic with three strategies:
    - WP: Log-odds space blending via logit/expit (D-01)
    - ATS: Linear interpolation in spread-point space
    - O/U: Linear interpolation in total-point space

The log-odds approach for WP ensures that blending respects the
non-linear nature of probabilities -- a 50/50 blend of 0.9 and 0.1
should yield 0.5 (which log-odds gives), not 0.5 (which linear also
gives in this symmetric case, but deviates in asymmetric cases).

For ATS and O/U, linear interpolation is appropriate because spreads
and totals live in a linear point space.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from utils import get_logger
from utils.probability_utils import moneyline_to_probability

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class BlendWeights:
    """Per-target model weights for market blending.

    Each weight controls how much the model's prediction is trusted
    relative to the market. A weight of 1.0 means "trust the model
    completely"; 0.0 means "trust the market completely".

    Attributes:
        wp_model_weight: Model weight for Win Probability blending.
        ats_model_weight: Model weight for Against the Spread blending.
        ou_model_weight: Model weight for Over/Under blending.
    """

    wp_model_weight: float = 0.60
    ats_model_weight: float = 0.60
    ou_model_weight: float = 0.60

    def __post_init__(self) -> None:
        """Validate all weights are in [0.0, 1.0]."""
        for name in ("wp_model_weight", "ats_model_weight", "ou_model_weight"):
            value = getattr(self, name)
            if not (0.0 <= value <= 1.0):
                msg = f"{name}={value} must be in [0.0, 1.0]"
                raise ValueError(msg)


@dataclass
class BlendConfig:
    """Full configuration for market blending.

    Attributes:
        weights: Per-target model weights.
        clip_min: Minimum probability for logit clipping (prevents -inf).
        clip_max: Maximum probability for logit clipping (prevents +inf).
    """

    weights: BlendWeights = field(default_factory=BlendWeights)
    clip_min: float = 0.001
    clip_max: float = 0.999


# ---------------------------------------------------------------------------
# MarketBlender
# ---------------------------------------------------------------------------


class MarketBlender:
    """Blends model predictions with market odds.

    Uses log-odds space for WP (probabilities are non-linear) and
    linear interpolation for ATS/O/U (spreads and totals are linear).

    Usage::

        blender = MarketBlender()
        blended_wp = blender.blend_wp(model_probs, market_probs)
        blended_spread = blender.blend_ats(model_spreads, market_spreads)
        blended_total = blender.blend_ou(model_totals, market_totals)

        # DataFrame-level blending:
        result_df = blender.blend_predictions(preds_df, market_df, target="wp")
    """

    def __init__(self, config: BlendConfig | None = None) -> None:
        self.config = config or BlendConfig()
        self.logger = get_logger(__name__)

    def blend_wp(
        self,
        model_prob: np.ndarray,
        market_prob: np.ndarray,
    ) -> np.ndarray:
        """Blend WP predictions in log-odds space.

        Clips both inputs to [clip_min, clip_max] before applying logit
        to prevent NaN/inf from boundary probabilities. The blended
        logit is converted back to probability via expit.

        Args:
            model_prob: Model's predicted win probabilities.
            market_prob: Market's fair win probabilities.

        Returns:
            Blended win probabilities in [0, 1].
        """
        weight = self.config.weights.wp_model_weight
        clip_min = self.config.clip_min
        clip_max = self.config.clip_max

        # Clip to avoid logit overflow at 0 and 1
        model_clipped = np.clip(model_prob, clip_min, clip_max)
        market_clipped = np.clip(market_prob, clip_min, clip_max)

        # Blend in log-odds space
        blended_logit = weight * logit(model_clipped) + (1 - weight) * logit(
            market_clipped
        )

        return expit(blended_logit)

    def blend_ats(
        self,
        model_spread: np.ndarray,
        market_spread: np.ndarray,
    ) -> np.ndarray:
        """Blend ATS predictions in spread-point space (linear interpolation).

        Args:
            model_spread: Model's predicted spreads (negative = home favored).
            market_spread: Market's closing spreads.

        Returns:
            Blended spreads.
        """
        weight = self.config.weights.ats_model_weight
        return weight * model_spread + (1 - weight) * market_spread

    def blend_ou(
        self,
        model_total: np.ndarray,
        market_total: np.ndarray,
    ) -> np.ndarray:
        """Blend O/U predictions in total-point space (linear interpolation).

        Args:
            model_total: Model's predicted game totals.
            market_total: Market's closing totals.

        Returns:
            Blended totals.
        """
        weight = self.config.weights.ou_model_weight
        return weight * model_total + (1 - weight) * market_total

    def blend_predictions(
        self,
        predictions_df: pd.DataFrame,
        market_df: pd.DataFrame,
        target: str,
    ) -> pd.DataFrame:
        """Blend a predictions DataFrame with market data.

        Merges predictions with market data on game_id, then applies
        the appropriate blending method based on target type.

        For WP: Devigs closing moneylines to get fair market probability,
        then blends model_prob with fair_ml_prob_home in log-odds space.

        For ATS: Blends model_spread with market spread linearly.

        For O/U: Blends model_total with market total linearly.

        Args:
            predictions_df: DataFrame with model predictions. Must contain
                game_id and the target-specific column (model_prob, model_spread,
                or model_total).
            market_df: DataFrame with market odds. Must contain game_id and
                the relevant market columns (ml_home/ml_away for WP,
                spread for ATS, total for O/U).
            target: One of "wp", "ats", "ou".

        Returns:
            Copy of predictions_df with blended values replacing originals.
            All other columns preserved unchanged.
        """
        result = predictions_df.copy()

        if result.empty:
            return result

        # Merge on game_id to get market data alongside predictions
        merged = result.merge(market_df, on="game_id", how="left", suffixes=("", "_market"))

        if target == "wp":
            self._blend_wp_predictions(result, merged)
        elif target == "ats":
            self._blend_ats_predictions(result, merged)
        elif target == "ou":
            self._blend_ou_predictions(result, merged)
        else:
            msg = f"Unknown target: {target}. Must be 'wp', 'ats', or 'ou'."
            raise ValueError(msg)

        return result

    def _blend_wp_predictions(
        self,
        result: pd.DataFrame,
        merged: pd.DataFrame,
    ) -> None:
        """Blend WP predictions in-place using devigged moneylines."""
        if "ml_home" not in merged.columns or "ml_away" not in merged.columns:
            self.logger.warning("Missing ml_home/ml_away columns for WP blending")
            return

        # Compute fair market probability from closing moneylines
        # Using proportional devigging (same as CLV module)
        valid_mask = merged["ml_home"].notna() & merged["ml_away"].notna()

        if not valid_mask.any():
            self.logger.warning("No valid moneyline data for WP blending")
            return

        home_raw = merged.loc[valid_mask, "ml_home"].apply(
            lambda ml: moneyline_to_probability(int(ml))
        )
        away_raw = merged.loc[valid_mask, "ml_away"].apply(
            lambda ml: moneyline_to_probability(int(ml))
        )
        fair_home = home_raw / (home_raw + away_raw)

        # Blend model_prob with fair market probability
        model_prob = result.loc[valid_mask, "model_prob"].values
        market_prob = fair_home.values

        blended = self.blend_wp(
            np.asarray(model_prob, dtype=np.float64),
            np.asarray(market_prob, dtype=np.float64),
        )
        result.loc[valid_mask, "model_prob"] = blended

        n_blended = valid_mask.sum()
        self.logger.info(
            "Blended WP predictions",
            n_blended=int(n_blended),
            n_total=len(result),
        )

    def _blend_ats_predictions(
        self,
        result: pd.DataFrame,
        merged: pd.DataFrame,
    ) -> None:
        """Blend ATS predictions in-place using market spreads."""
        if "spread" not in merged.columns:
            self.logger.warning("Missing spread column for ATS blending")
            return

        valid_mask = merged["spread"].notna()

        if not valid_mask.any():
            self.logger.warning("No valid spread data for ATS blending")
            return

        model_spread = result.loc[valid_mask, "model_spread"].values
        market_spread = merged.loc[valid_mask, "spread"].values

        blended = self.blend_ats(
            np.asarray(model_spread, dtype=np.float64),
            np.asarray(market_spread, dtype=np.float64),
        )
        result.loc[valid_mask, "model_spread"] = blended

        self.logger.info(
            "Blended ATS predictions",
            n_blended=int(valid_mask.sum()),
            n_total=len(result),
        )

    def _blend_ou_predictions(
        self,
        result: pd.DataFrame,
        merged: pd.DataFrame,
    ) -> None:
        """Blend O/U predictions in-place using market totals."""
        if "total" not in merged.columns:
            self.logger.warning("Missing total column for O/U blending")
            return

        valid_mask = merged["total"].notna()

        if not valid_mask.any():
            self.logger.warning("No valid total data for O/U blending")
            return

        model_total = result.loc[valid_mask, "model_total"].values
        market_total = merged.loc[valid_mask, "total"].values

        blended = self.blend_ou(
            np.asarray(model_total, dtype=np.float64),
            np.asarray(market_total, dtype=np.float64),
        )
        result.loc[valid_mask, "model_total"] = blended

        self.logger.info(
            "Blended O/U predictions",
            n_blended=int(valid_mask.sum()),
            n_total=len(result),
        )
