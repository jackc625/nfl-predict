"""Market blending module for combining model predictions with market odds.

Provides:
- BlendWeights: Per-target model weight configuration (WP, ATS, O/U)
- BlendConfig: Full blending configuration with probability clipping
- TuningResult: Output of grid-search weight tuning
- MarketBlender: Core blending logic with three strategies:
    - WP: Log-odds space blending via logit/expit (D-01)
    - ATS: Linear interpolation in spread-point space
    - O/U: Linear interpolation in total-point space
  Plus weight tuning, edge calibration, and artifact management.

The log-odds approach for WP ensures that blending respects the
non-linear nature of probabilities -- a 50/50 blend of 0.9 and 0.1
should yield 0.5 (which log-odds gives), not 0.5 (which linear also
gives in this symmetric case, but deviates in asymmetric cases).

For ATS and O/U, linear interpolation is appropriate because spreads
and totals live in a linear point space.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from models.blending_data import TUNING_SEASONS
from models.clv import compute_line_clv, compute_probability_clv
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
class EdgeThresholds:
    """Per-target edge thresholds for bet flagging.

    Edges below the threshold are not flagged. The 30% cap on
    mean per-week flagging is a DIAGNOSTIC WARNING, not a hard filter --
    games are never removed (per D-09).

    Attributes:
        wp_threshold: WP edge threshold in probability units.
        ats_threshold: ATS edge threshold in spread points.
        ou_threshold: O/U edge threshold in total points.
    """

    wp_threshold: float = 0.03
    ats_threshold: float = 1.5
    ou_threshold: float = 1.5


@dataclass
class BlendConfig:
    """Full configuration for market blending.

    Attributes:
        weights: Per-target model weights.
        clip_min: Minimum probability for logit clipping (prevents -inf).
        clip_max: Maximum probability for logit clipping (prevents +inf).
        edge_thresholds: Per-target edge thresholds for bet flagging.
    """

    weights: BlendWeights = field(default_factory=BlendWeights)
    clip_min: float = 0.001
    clip_max: float = 0.999
    edge_thresholds: EdgeThresholds = field(default_factory=EdgeThresholds)


@dataclass
class TuningResult:
    """Output of blend weight grid-search tuning.

    Attributes:
        weights: Optimal BlendWeights found by grid search.
        per_target_clv: Best CLV per target at the optimal weight.
        per_target_grid: Full (weight, clv) grid per target.
        tuning_seasons: Seasons used for tuning.
        n_games: Number of games per target used in tuning.
    """

    weights: BlendWeights
    per_target_clv: dict[str, float]
    per_target_grid: dict[str, list[tuple[float, float]]]
    tuning_seasons: list[int]
    n_games: dict[str, int]


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

    # -----------------------------------------------------------------------
    # Weight tuning via grid search
    # -----------------------------------------------------------------------

    def tune_weights(
        self,
        tuning_predictions: dict[str, pd.DataFrame],
        tuning_odds: pd.DataFrame,
        weight_range: tuple[float, float] = (0.50, 0.70),
        weight_step: float = 0.01,
    ) -> TuningResult:
        """Tune per-target blend weights via grid search on pre-backtest data.

        For each target, sweeps weight candidates in the given range and
        selects the weight that maximizes mean CLV on the tuning predictions.

        TEMPORAL ISOLATION: Raises ValueError if any prediction season
        exceeds max(TUNING_SEASONS) to prevent holdout data leakage.

        Args:
            tuning_predictions: Dict mapping target ("wp", "ats", "ou") to
                DataFrame of model predictions for tuning period.
            tuning_odds: DataFrame with game_id, spread, total, ml_home, ml_away.
            weight_range: (min_weight, max_weight) for grid search.
            weight_step: Step size for weight candidates.

        Returns:
            TuningResult with optimal weights, per-target CLV, grid, seasons, counts.

        Raises:
            ValueError: If holdout seasons detected in tuning predictions.
        """
        max_tuning_season = max(TUNING_SEASONS)

        # -- Temporal isolation check --
        for target, df in tuning_predictions.items():
            if "season" not in df.columns:
                continue
            max_season = int(df["season"].max())
            if max_season > max_tuning_season:
                msg = (
                    f"Holdout data detected in tuning predictions: "
                    f"season {max_season} in target '{target}'"
                )
                raise ValueError(msg)

        # Build weight candidates
        candidates = np.arange(
            weight_range[0], weight_range[1] + weight_step / 2, weight_step
        )
        # Round to avoid floating point drift
        candidates = np.round(candidates, 4)

        # Collect all tuning seasons
        all_seasons: set[int] = set()
        for df in tuning_predictions.values():
            if "season" in df.columns:
                all_seasons.update(df["season"].unique().tolist())

        per_target_clv: dict[str, float] = {}
        per_target_grid: dict[str, list[tuple[float, float]]] = {}
        n_games: dict[str, int] = {}
        optimal_weights: dict[str, float] = {}

        # Weight attribute mapping: target -> BlendWeights attribute name
        weight_attrs = {
            "wp": "wp_model_weight",
            "ats": "ats_model_weight",
            "ou": "ou_model_weight",
        }

        for target, preds_df in tuning_predictions.items():
            grid: list[tuple[float, float]] = []

            # Merge predictions with odds
            merged = preds_df.merge(tuning_odds, on="game_id", how="inner")
            n_games[target] = len(merged)

            if merged.empty:
                self.logger.warning(
                    "No merged games for target during tuning",
                    target=target,
                )
                per_target_clv[target] = 0.0
                per_target_grid[target] = []
                optimal_weights[target] = weight_range[0]
                continue

            for w in candidates:
                mean_clv = self._compute_mean_clv_for_weight(
                    target, merged, float(w)
                )
                grid.append((float(w), mean_clv))

            # Select the weight with highest mean CLV
            best_weight, best_clv = max(grid, key=lambda x: x[1])
            per_target_clv[target] = best_clv
            per_target_grid[target] = grid
            optimal_weights[target] = best_weight

            self.logger.info(
                "Weight tuning complete for target",
                target=target,
                optimal_weight=best_weight,
                optimal_clv=best_clv,
                n_candidates=len(candidates),
                n_games=len(merged),
            )

        # Build optimized BlendWeights
        tuned_weights = BlendWeights(
            wp_model_weight=optimal_weights.get("wp", self.config.weights.wp_model_weight),
            ats_model_weight=optimal_weights.get("ats", self.config.weights.ats_model_weight),
            ou_model_weight=optimal_weights.get("ou", self.config.weights.ou_model_weight),
        )

        # Update self.config with tuned weights
        self.config.weights = tuned_weights

        # Log comparison with defaults
        default_weights = BlendWeights()
        for target in optimal_weights:
            attr = weight_attrs.get(target, "")
            if attr:
                self.logger.info(
                    "Weight tuning comparison",
                    target=target,
                    default_weight=getattr(default_weights, attr),
                    tuned_weight=getattr(tuned_weights, attr),
                    clv_at_tuned=per_target_clv.get(target, 0.0),
                )

        result = TuningResult(
            weights=tuned_weights,
            per_target_clv=per_target_clv,
            per_target_grid=per_target_grid,
            tuning_seasons=sorted(all_seasons),
            n_games=n_games,
        )
        return result

    def _compute_mean_clv_for_weight(
        self,
        target: str,
        merged: pd.DataFrame,
        weight: float,
    ) -> float:
        """Compute mean CLV for a single weight candidate on merged data.

        Args:
            target: "wp", "ats", or "ou".
            merged: Predictions merged with odds (has both model and market columns).
            weight: Model weight to evaluate.

        Returns:
            Mean CLV across all games at this weight.
        """
        if target == "wp":
            return self._compute_wp_clv_for_weight(merged, weight)
        if target == "ats":
            return self._compute_ats_clv_for_weight(merged, weight)
        if target == "ou":
            return self._compute_ou_clv_for_weight(merged, weight)
        return 0.0

    def _compute_wp_clv_for_weight(
        self, merged: pd.DataFrame, weight: float
    ) -> float:
        """Compute mean probability CLV for WP at a given weight."""
        valid = merged.dropna(subset=["ml_home", "ml_away", "model_prob"])
        if valid.empty:
            return 0.0

        # Build temporary blender with this weight
        tmp_config = BlendConfig(
            weights=BlendWeights(wp_model_weight=weight),
            clip_min=self.config.clip_min,
            clip_max=self.config.clip_max,
        )
        tmp_blender = MarketBlender(config=tmp_config)

        # Compute fair market probabilities for devigging
        home_raw = valid["ml_home"].apply(lambda ml: moneyline_to_probability(int(ml)))
        away_raw = valid["ml_away"].apply(lambda ml: moneyline_to_probability(int(ml)))
        fair_home = (home_raw / (home_raw + away_raw)).values

        # Blend model prob with market fair prob
        model_prob = valid["model_prob"].values
        blended = tmp_blender.blend_wp(
            np.asarray(model_prob, dtype=np.float64),
            np.asarray(fair_home, dtype=np.float64),
        )

        # Compute probability CLV for each game
        clvs = []
        for i, idx in enumerate(valid.index):
            result = compute_probability_clv(
                model_prob=float(blended[i]),
                closing_ml_home=float(valid.at[idx, "ml_home"]),
                closing_ml_away=float(valid.at[idx, "ml_away"]),
                side="home",
            )
            clvs.append(result["probability_clv"])

        return float(np.mean(clvs)) if clvs else 0.0

    def _compute_ats_clv_for_weight(
        self, merged: pd.DataFrame, weight: float
    ) -> float:
        """Compute mean line CLV for ATS at a given weight."""
        valid = merged.dropna(subset=["spread", "model_spread"])
        if valid.empty:
            return 0.0

        model_spread = valid["model_spread"].values
        market_spread = valid["spread"].values
        blended = weight * model_spread + (1 - weight) * market_spread

        clvs = [
            compute_line_clv(
                model_value=float(blended[i]),
                closing_value=float(market_spread[i]),
                direction="spread",
            )
            for i in range(len(blended))
        ]
        return float(np.mean(clvs)) if clvs else 0.0

    def _compute_ou_clv_for_weight(
        self, merged: pd.DataFrame, weight: float
    ) -> float:
        """Compute mean line CLV for O/U at a given weight."""
        valid = merged.dropna(subset=["total", "model_total"])
        if valid.empty:
            return 0.0

        model_total = valid["model_total"].values
        market_total = valid["total"].values
        blended = weight * model_total + (1 - weight) * market_total

        clvs = [
            compute_line_clv(
                model_value=float(blended[i]),
                closing_value=float(market_total[i]),
                direction="total",
            )
            for i in range(len(blended))
        ]
        return float(np.mean(clvs)) if clvs else 0.0

    # -----------------------------------------------------------------------
    # Edge threshold calibration
    # -----------------------------------------------------------------------

    def calibrate_edge_thresholds(
        self,
        tuning_predictions: dict[str, pd.DataFrame],
        tuning_odds: pd.DataFrame,
        max_flag_rate: float = 0.30,
    ) -> EdgeThresholds:
        """Calibrate per-target edge thresholds on tuning data.

        For each target, computes edges on blended tuning predictions,
        then sweeps candidate thresholds to find the first where the
        mean per-week flagging rate is <= max_flag_rate.

        Args:
            tuning_predictions: Dict mapping target to predictions DataFrame.
            tuning_odds: DataFrame with game_id, spread, total, ml_home, ml_away.
            max_flag_rate: Maximum mean per-week flagging rate (default 0.30).

        Returns:
            EdgeThresholds with calibrated per-target thresholds.
        """
        thresholds: dict[str, float] = {}

        # Sweep ranges per target
        sweep_ranges = {
            "wp": np.arange(0.01, 0.15, 0.005),
            "ats": np.arange(0.5, 5.0, 0.25),
            "ou": np.arange(0.5, 5.0, 0.25),
        }

        for target, preds_df in tuning_predictions.items():
            merged = preds_df.merge(tuning_odds, on="game_id", how="inner")
            if merged.empty:
                thresholds[target] = sweep_ranges.get(target, np.array([0.03]))[0]
                continue

            edges = self._compute_edges(target, merged)
            merged["_edge"] = edges

            # Need season and week for per-week grouping
            if "season" not in merged.columns or "week" not in merged.columns:
                thresholds[target] = sweep_ranges.get(target, np.array([0.03]))[0]
                continue

            sweep = sweep_ranges.get(target, np.arange(0.01, 0.15, 0.005))
            best_threshold = float(sweep[-1])  # Default to largest if none works

            for candidate in sweep:
                # Compute per-week flagging rate
                threshold_val = float(candidate)
                weekly_rates = merged.groupby(["season", "week"]).apply(
                    lambda g, t=threshold_val: (g["_edge"] > t).mean(),
                    include_groups=False,
                )
                mean_rate = weekly_rates.mean()

                if mean_rate <= max_flag_rate:
                    best_threshold = float(candidate)
                    break

            thresholds[target] = best_threshold
            self.logger.info(
                "Edge threshold calibrated",
                target=target,
                threshold=best_threshold,
                max_flag_rate=max_flag_rate,
            )

        calibrated = EdgeThresholds(
            wp_threshold=thresholds.get("wp", EdgeThresholds().wp_threshold),
            ats_threshold=thresholds.get("ats", EdgeThresholds().ats_threshold),
            ou_threshold=thresholds.get("ou", EdgeThresholds().ou_threshold),
        )
        self.config.edge_thresholds = calibrated
        return calibrated

    def check_weekly_edge_rate(
        self,
        predictions_df: pd.DataFrame,
        market_df: pd.DataFrame,
        target: str,
    ) -> dict:
        """Check per-week edge flagging rate and emit diagnostic warnings.

        Computes edges for each game, groups by (season, week), and
        logs a WARNING for any week where > 30% of games are flagged.
        Per D-09: This is a diagnostic WARNING, not a filter -- games
        are NOT removed.

        Args:
            predictions_df: Predictions DataFrame with model values.
            market_df: Market odds DataFrame.
            target: One of "wp", "ats", "ou".

        Returns:
            Dict with per_week_rates, mean_rate, warnings.
        """
        merged = predictions_df.merge(market_df, on="game_id", how="inner")
        if merged.empty:
            return {"per_week_rates": [], "mean_rate": 0.0, "warnings": []}

        edges = self._compute_edges(target, merged)
        merged["_edge"] = edges

        # Get the threshold for this target
        threshold_map = {
            "wp": self.config.edge_thresholds.wp_threshold,
            "ats": self.config.edge_thresholds.ats_threshold,
            "ou": self.config.edge_thresholds.ou_threshold,
        }
        threshold = threshold_map.get(target, 0.03)

        # Compute per-week flagging rate
        per_week_rates: list[float] = []
        warnings: list[str] = []

        for (season, week), group in merged.groupby(["season", "week"]):
            rate = float((group["_edge"] > threshold).mean())
            per_week_rates.append(rate)

            if rate > 0.30:
                msg = (
                    f"Edge threshold diagnostic: {rate:.0%} of games flagged "
                    f"in season {season} week {week} for {target}"
                )
                warnings.append(msg)
                self.logger.warning(msg)

        mean_rate = float(np.mean(per_week_rates)) if per_week_rates else 0.0

        return {
            "per_week_rates": per_week_rates,
            "mean_rate": mean_rate,
            "warnings": warnings,
        }

    def _compute_edges(
        self,
        target: str,
        merged: pd.DataFrame,
    ) -> np.ndarray:
        """Compute edge magnitudes for a target on merged predictions+odds.

        Args:
            target: "wp", "ats", or "ou".
            merged: Predictions merged with odds DataFrame.

        Returns:
            Array of absolute edge values.
        """
        if target == "wp":
            # WP edge: |blended_prob - fair_market_prob|
            valid = merged.dropna(subset=["ml_home", "ml_away", "model_prob"])
            if valid.empty:
                return np.array([])

            home_raw = valid["ml_home"].apply(
                lambda ml: moneyline_to_probability(int(ml))
            )
            away_raw = valid["ml_away"].apply(
                lambda ml: moneyline_to_probability(int(ml))
            )
            fair_home = (home_raw / (home_raw + away_raw)).values
            model_prob = valid["model_prob"].values
            blended = self.blend_wp(
                np.asarray(model_prob, dtype=np.float64),
                np.asarray(fair_home, dtype=np.float64),
            )
            edges = np.abs(blended - fair_home)
            # Reindex to match merged
            result = np.zeros(len(merged))
            result[valid.index.to_numpy() - merged.index[0]] = edges
            return result

        if target == "ats":
            valid = merged.dropna(subset=["spread", "model_spread"])
            if valid.empty:
                return np.array([])
            blended = (
                self.config.weights.ats_model_weight * valid["model_spread"].values
                + (1 - self.config.weights.ats_model_weight) * valid["spread"].values
            )
            edges = np.abs(blended - valid["spread"].values)
            result = np.zeros(len(merged))
            result[valid.index.to_numpy() - merged.index[0]] = edges
            return result

        if target == "ou":
            valid = merged.dropna(subset=["total", "model_total"])
            if valid.empty:
                return np.array([])
            blended = (
                self.config.weights.ou_model_weight * valid["model_total"].values
                + (1 - self.config.weights.ou_model_weight) * valid["total"].values
            )
            edges = np.abs(blended - valid["total"].values)
            result = np.zeros(len(merged))
            result[valid.index.to_numpy() - merged.index[0]] = edges
            return result

        return np.zeros(len(merged))

    # -----------------------------------------------------------------------
    # Artifact persistence
    # -----------------------------------------------------------------------

    def save_blend_artifacts(
        self,
        tuning_result: TuningResult,
        artifacts_dir: Path = Path("artifacts"),
    ) -> Path:
        """Save blend weights and metadata as JSON artifacts.

        Creates artifacts/blend_{timestamp}/ with blend_weights.json
        containing weights, CLV, provenance metadata, and tuning config.
        Updates artifacts/latest.json with "blend" key.

        Args:
            tuning_result: TuningResult from tune_weights.
            artifacts_dir: Root directory for artifacts.

        Returns:
            Path to the created artifact directory.
        """
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
        artifact_dir = artifacts_dir / f"blend_{timestamp}"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        # Build JSON payload
        payload = {
            "weights": {
                "wp": tuning_result.weights.wp_model_weight,
                "ats": tuning_result.weights.ats_model_weight,
                "ou": tuning_result.weights.ou_model_weight,
            },
            "edge_thresholds": {
                "wp": self.config.edge_thresholds.wp_threshold,
                "ats": self.config.edge_thresholds.ats_threshold,
                "ou": self.config.edge_thresholds.ou_threshold,
            },
            "per_target_clv": tuning_result.per_target_clv,
            "tuning_seasons": tuning_result.tuning_seasons,
            "n_games": tuning_result.n_games,
            "tuned_at": datetime.now(tz=UTC).isoformat(),
            "weight_range": [0.50, 0.70],
            "weight_step": 0.01,
        }

        weights_path = artifact_dir / "blend_weights.json"
        weights_path.write_text(json.dumps(payload, indent=2))

        # Update latest.json manifest
        latest_path = artifacts_dir / "latest.json"
        if latest_path.exists():
            manifest = json.loads(latest_path.read_text())
        else:
            manifest = {}
        manifest["blend"] = artifact_dir.name
        latest_path.write_text(json.dumps(manifest, indent=2))

        self.logger.info(
            "Saved blend artifacts",
            artifact_dir=str(artifact_dir),
            weights=payload["weights"],
        )

        return artifact_dir

    @classmethod
    def from_artifacts(
        cls,
        artifacts_dir: Path = Path("artifacts"),
        version: str | None = None,
    ) -> MarketBlender:
        """Load a MarketBlender from saved artifacts.

        Reads blend_weights.json from the artifact directory (latest or
        specific version) and returns a new MarketBlender configured
        with the loaded weights.

        Args:
            artifacts_dir: Root directory for artifacts.
            version: Specific artifact directory name. If None, uses latest.

        Returns:
            MarketBlender configured with loaded weights.

        Raises:
            FileNotFoundError: If artifacts not found.
            KeyError: If 'blend' not in latest.json.
        """
        if version is None:
            latest_path = artifacts_dir / "latest.json"
            if not latest_path.exists():
                msg = f"latest.json not found in {artifacts_dir}"
                raise FileNotFoundError(msg)
            manifest = json.loads(latest_path.read_text())
            if "blend" not in manifest:
                msg = "'blend' not found in latest.json manifest"
                raise KeyError(msg)
            version = manifest["blend"]

        artifact_dir = artifacts_dir / version
        weights_path = artifact_dir / "blend_weights.json"

        if not weights_path.exists():
            msg = f"blend_weights.json not found in {artifact_dir}"
            raise FileNotFoundError(msg)

        data = json.loads(weights_path.read_text())
        weights_data = data["weights"]

        blend_weights = BlendWeights(
            wp_model_weight=weights_data["wp"],
            ats_model_weight=weights_data["ats"],
            ou_model_weight=weights_data["ou"],
        )

        # Load edge thresholds if present
        edge_data = data.get("edge_thresholds")
        if edge_data:
            edge_thresholds = EdgeThresholds(
                wp_threshold=edge_data["wp"],
                ats_threshold=edge_data["ats"],
                ou_threshold=edge_data["ou"],
            )
        else:
            edge_thresholds = EdgeThresholds()

        config = BlendConfig(
            weights=blend_weights,
            edge_thresholds=edge_thresholds,
        )

        return cls(config=config)
