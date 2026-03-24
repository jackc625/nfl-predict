"""Unit tests for edge threshold calibration and 30% diagnostic warning.

Tests cover:
- EdgeThresholds stores per-target thresholds
- calibrate_edge_thresholds returns thresholds where <= 30% flagged per week on average
- calibrate_edge_thresholds with tight model returns low thresholds
- calibrate_edge_thresholds with noisy model returns higher thresholds
- When >30% of games in a week pass threshold, a structlog WARNING is emitted
- check_weekly_edge_rate returns dict with per-week flagging rates and warnings
- EdgeThresholds defaults are reasonable
- save_blend_artifacts includes edge_thresholds in JSON alongside weights
- from_artifacts loads EdgeThresholds alongside BlendWeights
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
    EdgeThresholds,
    MarketBlender,
    TuningResult,
)


# ---------------------------------------------------------------------------
# Helpers: Synthetic data for edge calibration
# ---------------------------------------------------------------------------


def _make_wp_predictions_with_edges(
    n_weeks: int = 8,
    games_per_week: int = 5,
    edge_magnitude: str = "small",
    rng_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create WP predictions and odds where edges are controllable.

    edge_magnitude:
        "small": edges are all < 0.02 (tight model)
        "large": edges are uniformly 0.05-0.15 (noisy model)
        "mixed": 60% of games have large edges (triggers >30% warning)
    """
    rng = np.random.default_rng(rng_seed)
    pred_rows = []
    odds_rows = []
    game_counter = 0
    season = 2015

    for week in range(1, n_weeks + 1):
        for _ in range(games_per_week):
            game_counter += 1
            game_id = f"{season}_{week:02d}_{game_counter:04d}"

            # Market fair probability (home team)
            fair_prob = rng.uniform(0.35, 0.65)

            # Model probability with controlled edge
            if edge_magnitude == "small":
                edge = rng.uniform(-0.01, 0.01)
            elif edge_magnitude == "large":
                edge = rng.choice([-1, 1]) * rng.uniform(0.05, 0.15)
            else:  # mixed -- 60% large, 40% small
                if rng.random() < 0.60:
                    edge = rng.choice([-1, 1]) * rng.uniform(0.05, 0.15)
                else:
                    edge = rng.uniform(-0.01, 0.01)

            model_prob = float(np.clip(fair_prob + edge, 0.05, 0.95))
            actual = int(rng.random() < fair_prob)

            pred_rows.append({
                "game_id": game_id,
                "season": season,
                "week": week,
                "model_prob": model_prob,
                "actual": actual,
            })

            # Convert fair_prob to moneylines for odds
            if fair_prob >= 0.5:
                ml_home = -int(100 * fair_prob / (1 - fair_prob))
                ml_away = int(100 * (1 - fair_prob) / fair_prob)
            else:
                ml_home = int(100 * (1 - fair_prob) / fair_prob)
                ml_away = -int(100 * fair_prob / (1 - fair_prob))

            # Avoid zero moneylines
            if ml_home == 0:
                ml_home = 100
            if ml_away == 0:
                ml_away = 100

            spread = rng.normal(-2, 5)
            total = rng.normal(45, 4)

            odds_rows.append({
                "game_id": game_id,
                "ml_home": ml_home,
                "ml_away": ml_away,
                "spread": spread,
                "total": total,
            })

    return pd.DataFrame(pred_rows), pd.DataFrame(odds_rows)


def _make_ats_predictions_with_edges(
    n_weeks: int = 8,
    games_per_week: int = 5,
    edge_magnitude: str = "small",
    rng_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create ATS predictions with controllable spread edges."""
    rng = np.random.default_rng(rng_seed)
    pred_rows = []
    odds_rows = []
    game_counter = 0
    season = 2015

    for week in range(1, n_weeks + 1):
        for _ in range(games_per_week):
            game_counter += 1
            game_id = f"{season}_{week:02d}_{game_counter:04d}"

            market_spread = rng.normal(-2.5, 5.0)

            if edge_magnitude == "small":
                edge = rng.uniform(-0.3, 0.3)
            else:
                edge = rng.choice([-1, 1]) * rng.uniform(2.0, 5.0)

            model_spread = market_spread + edge
            actual_margin = market_spread + rng.normal(0, 14)

            pred_rows.append({
                "game_id": game_id,
                "season": season,
                "week": week,
                "model_spread": model_spread,
                "actual": actual_margin,
            })

            # Moneylines for the odds DataFrame (required for merge)
            ml_home = rng.choice([-150, -130, -120, -110, 100, 110, 130])
            ml_away = -ml_home if ml_home > 0 else abs(ml_home) - 20

            odds_rows.append({
                "game_id": game_id,
                "ml_home": ml_home,
                "ml_away": ml_away,
                "spread": market_spread,
                "total": rng.normal(45, 4),
            })

    return pd.DataFrame(pred_rows), pd.DataFrame(odds_rows)


# ---------------------------------------------------------------------------
# Test class: EdgeThresholds
# ---------------------------------------------------------------------------


class TestEdgeThresholds:
    """Tests for EdgeThresholds dataclass."""

    def test_edge_thresholds_defaults(self) -> None:
        """EdgeThresholds defaults are reasonable values."""
        thresholds = EdgeThresholds()
        assert 0.01 <= thresholds.wp_threshold <= 0.10
        assert 0.5 <= thresholds.ats_threshold <= 3.0
        assert 0.5 <= thresholds.ou_threshold <= 3.0

    def test_edge_thresholds_stored_in_blend_config(self) -> None:
        """BlendConfig includes edge_thresholds field."""
        config = BlendConfig()
        assert hasattr(config, "edge_thresholds")
        assert isinstance(config.edge_thresholds, EdgeThresholds)


# ---------------------------------------------------------------------------
# Test class: Calibration
# ---------------------------------------------------------------------------


class TestCalibrateEdgeThresholds:
    """Tests for calibrate_edge_thresholds threshold sweep."""

    def test_calibrate_returns_edge_thresholds(self) -> None:
        """calibrate_edge_thresholds returns EdgeThresholds."""
        preds, odds = _make_wp_predictions_with_edges(edge_magnitude="small")
        blender = MarketBlender()
        result = blender.calibrate_edge_thresholds(
            tuning_predictions={"wp": preds},
            tuning_odds=odds,
        )
        assert isinstance(result, EdgeThresholds)

    def test_tight_model_returns_low_thresholds(self) -> None:
        """calibrate_edge_thresholds with tight model (all edges < 0.02) returns low thresholds."""
        preds, odds = _make_wp_predictions_with_edges(edge_magnitude="small")
        blender = MarketBlender()
        result = blender.calibrate_edge_thresholds(
            tuning_predictions={"wp": preds},
            tuning_odds=odds,
        )
        # Tight model should have a low threshold
        assert result.wp_threshold <= 0.10

    def test_noisy_model_returns_higher_thresholds(self) -> None:
        """calibrate_edge_thresholds with noisy model returns higher thresholds."""
        preds_tight, odds_tight = _make_wp_predictions_with_edges(
            edge_magnitude="small", rng_seed=1
        )
        preds_noisy, odds_noisy = _make_wp_predictions_with_edges(
            edge_magnitude="large", rng_seed=1
        )

        blender_tight = MarketBlender()
        result_tight = blender_tight.calibrate_edge_thresholds(
            tuning_predictions={"wp": preds_tight},
            tuning_odds=odds_tight,
        )

        blender_noisy = MarketBlender()
        result_noisy = blender_noisy.calibrate_edge_thresholds(
            tuning_predictions={"wp": preds_noisy},
            tuning_odds=odds_noisy,
        )

        # Noisy model should need a higher threshold to keep flagging <= 30%
        assert result_noisy.wp_threshold >= result_tight.wp_threshold

    def test_calibrate_flag_rate_within_target(self) -> None:
        """calibrate_edge_thresholds produces mean per-week flagging rate <= 30%."""
        preds, odds = _make_wp_predictions_with_edges(
            edge_magnitude="large", n_weeks=16, games_per_week=8
        )
        blender = MarketBlender()
        thresholds = blender.calibrate_edge_thresholds(
            tuning_predictions={"wp": preds},
            tuning_odds=odds,
            max_flag_rate=0.30,
        )

        # Verify by re-computing the flag rate at the calibrated threshold
        merged = preds.merge(odds, on="game_id", how="inner")
        from utils.probability_utils import moneyline_to_probability

        home_raw = merged["ml_home"].apply(lambda ml: moneyline_to_probability(int(ml)))
        away_raw = merged["ml_away"].apply(lambda ml: moneyline_to_probability(int(ml)))
        fair_home = home_raw / (home_raw + away_raw)
        edges = np.abs(merged["model_prob"].values - fair_home.values)

        # Compute per-week flag rate
        merged["edge"] = edges
        weekly = merged.groupby(["season", "week"]).apply(
            lambda g: (g["edge"] > thresholds.wp_threshold).mean(),
            include_groups=False,
        )
        mean_rate = weekly.mean()
        assert mean_rate <= 0.35  # Allow small margin for discretization


# ---------------------------------------------------------------------------
# Test class: 30% diagnostic warning
# ---------------------------------------------------------------------------


class TestCheckWeeklyEdgeRate:
    """Tests for check_weekly_edge_rate and the 30% WARNING."""

    def test_check_weekly_edge_rate_returns_dict(self) -> None:
        """check_weekly_edge_rate returns dict with per_week_rates, mean_rate, warnings."""
        preds, odds = _make_wp_predictions_with_edges(edge_magnitude="large")
        blender = MarketBlender()
        result = blender.check_weekly_edge_rate(preds, odds, target="wp")

        assert "per_week_rates" in result
        assert "mean_rate" in result
        assert "warnings" in result
        assert isinstance(result["warnings"], list)

    def test_thirty_percent_warning(self, capsys: pytest.CaptureFixture[str]) -> None:
        """When >30% of games in a week pass threshold, a WARNING is emitted."""
        # Create a scenario where ALL games in one week have huge edges
        season = 2015
        preds_rows = []
        odds_rows = []

        for i in range(5):
            game_id = f"{season}_01_{i:04d}"
            # All 5 games have model_prob far from market (edge ~0.30)
            preds_rows.append({
                "game_id": game_id,
                "season": season,
                "week": 1,
                "model_prob": 0.80,
                "actual": 1,
            })
            odds_rows.append({
                "game_id": game_id,
                "ml_home": -110,
                "ml_away": -110,
                "spread": -1.0,
                "total": 45.0,
            })

        preds = pd.DataFrame(preds_rows)
        odds = pd.DataFrame(odds_rows)

        # Set a low edge threshold so all games are flagged
        config = BlendConfig(
            edge_thresholds=EdgeThresholds(wp_threshold=0.01),
        )
        blender = MarketBlender(config=config)
        result = blender.check_weekly_edge_rate(preds, odds, target="wp")

        # All games flagged in week 1: 100% > 30%, should produce warning
        assert len(result["warnings"]) > 0
        assert any("Edge threshold diagnostic" in w for w in result["warnings"])
        assert result["per_week_rates"][0] > 0.30

    def test_no_warning_when_below_thirty_percent(self) -> None:
        """No warning when per-week flagging rate is <= 30%."""
        preds, odds = _make_wp_predictions_with_edges(
            edge_magnitude="small", n_weeks=4, games_per_week=10
        )

        # High threshold -- very few games flagged
        config = BlendConfig(
            edge_thresholds=EdgeThresholds(wp_threshold=0.50),
        )
        blender = MarketBlender(config=config)
        result = blender.check_weekly_edge_rate(preds, odds, target="wp")

        assert len(result["warnings"]) == 0


# ---------------------------------------------------------------------------
# Test class: Updated artifacts with edge thresholds
# ---------------------------------------------------------------------------


class TestArtifactsWithEdgeThresholds:
    """Tests for save/load artifacts including edge thresholds."""

    @pytest.fixture()
    def sample_tuning_result(self) -> TuningResult:
        return TuningResult(
            weights=BlendWeights(
                wp_model_weight=0.55,
                ats_model_weight=0.62,
                ou_model_weight=0.58,
            ),
            per_target_clv={"wp": 0.012, "ats": 0.35, "ou": 0.28},
            per_target_grid={
                "wp": [(0.55, 0.012)],
                "ats": [(0.62, 0.35)],
                "ou": [(0.58, 0.28)],
            },
            tuning_seasons=list(range(2010, 2018)),
            n_games={"wp": 400, "ats": 400, "ou": 400},
        )

    def test_save_includes_edge_thresholds(
        self,
        sample_tuning_result: TuningResult,
        tmp_path: Path,
    ) -> None:
        """save_blend_artifacts includes edge_thresholds in the JSON."""
        config = BlendConfig(
            edge_thresholds=EdgeThresholds(
                wp_threshold=0.04,
                ats_threshold=1.75,
                ou_threshold=1.50,
            ),
        )
        blender = MarketBlender(config=config)
        artifact_dir = blender.save_blend_artifacts(
            sample_tuning_result, artifacts_dir=tmp_path
        )

        data = json.loads((artifact_dir / "blend_weights.json").read_text())
        assert "edge_thresholds" in data
        assert data["edge_thresholds"]["wp"] == pytest.approx(0.04)
        assert data["edge_thresholds"]["ats"] == pytest.approx(1.75)
        assert data["edge_thresholds"]["ou"] == pytest.approx(1.50)

    def test_from_artifacts_loads_edge_thresholds(
        self,
        sample_tuning_result: TuningResult,
        tmp_path: Path,
    ) -> None:
        """from_artifacts loads EdgeThresholds alongside BlendWeights."""
        config = BlendConfig(
            edge_thresholds=EdgeThresholds(
                wp_threshold=0.04,
                ats_threshold=1.75,
                ou_threshold=1.50,
            ),
        )
        blender = MarketBlender(config=config)
        blender.save_blend_artifacts(sample_tuning_result, artifacts_dir=tmp_path)

        loaded = MarketBlender.from_artifacts(artifacts_dir=tmp_path)
        assert loaded.config.edge_thresholds.wp_threshold == pytest.approx(0.04)
        assert loaded.config.edge_thresholds.ats_threshold == pytest.approx(1.75)
        assert loaded.config.edge_thresholds.ou_threshold == pytest.approx(1.50)
