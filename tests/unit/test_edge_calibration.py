"""Unit tests for edge threshold calibration and 30% diagnostic warning.

Tests cover:
- EdgeThresholds stores per-target thresholds
- When >30% of games in a week pass threshold, a structlog WARNING is emitted
- the WP edge is measured against the pre-lock market, never a closing moneyline
- check_weekly_edge_rate returns dict with per-week flagging rates and warnings
- EdgeThresholds defaults are reasonable
- save_blend_artifacts includes edge_thresholds in JSON alongside weights
- from_artifacts loads EdgeThresholds alongside BlendWeights

DELETED (A33.2-review WR-05): ``TestCalibrateEdgeThresholds``. ``calibrate_edge_thresholds``
had no production caller left and measured WP edges against a devigged CLOSING moneyline, so
the method is deleted with its tests. The 2026 thresholds are Plan 33.2-26's derivation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models.blending import (
    BlendConfig,
    BlendProvenance,
    BlendWeights,
    EdgeThresholds,
    MarketBlender,
    MarketProbabilityUnavailable,
    TuningResult,
)

#: A converter the artifact tests bind to. Since Plan 33.2-24 a blend artifact must carry a
#: converter binding (a blend that cannot convert a spread cannot serve WP), and
#: ``from_artifacts`` cross-checks it against the named directory.
_CONVERTER_ID = "market_probability_20990101_000000"
_CONVERTER_SLOPE = 0.15


def _write_converter(root: Path) -> None:
    directory = root / _CONVERTER_ID
    directory.mkdir(parents=True)
    (directory / "metadata.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "slope_beta": _CONVERTER_SLOPE,
                "walk_forward_slopes": {"2021": 0.14},
                "training_seasons": [2020, 2021],
                "n_games": 10,
                "input_digest": "0" * 64,
                "fitted_at": "2099-01-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )


def _provenance() -> BlendProvenance:
    return BlendProvenance(
        gold_generation_digest="a" * 64,
        source_artifact_ids={"wp": "wp_x", "ats": "ats_x", "ou": "ou_x"},
        tuning_corpus_rows=400,
        excluded_counts={"no_prelock_line": 0, "no_prior_fold_converter": 0},
        thread_limit=1,
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
            elif edge_magnitude == "large" or rng.random() < 0.60:
                edge = rng.choice([-1, 1]) * rng.uniform(0.05, 0.15)
            else:
                edge = rng.uniform(-0.01, 0.01)

            model_prob = float(np.clip(fair_prob + edge, 0.05, 0.95))
            actual = int(rng.random() < fair_prob)

            pred_rows.append(
                {
                    "game_id": game_id,
                    "season": season,
                    "week": week,
                    "model_prob": model_prob,
                    "actual": actual,
                }
            )

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

            odds_rows.append(
                {
                    "game_id": game_id,
                    "ml_home": ml_home,
                    "ml_away": ml_away,
                    "spread": spread,
                    "total": total,
                }
            )

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

            pred_rows.append(
                {
                    "game_id": game_id,
                    "season": season,
                    "week": week,
                    "model_spread": model_spread,
                    "actual": actual_margin,
                }
            )

            # Moneylines for the odds DataFrame (required for merge)
            ml_home = rng.choice([-150, -130, -120, -110, 100, 110, 130])
            ml_away = -ml_home if ml_home > 0 else abs(ml_home) - 20

            odds_rows.append(
                {
                    "game_id": game_id,
                    "ml_home": ml_home,
                    "ml_away": ml_away,
                    "spread": market_spread,
                    "total": rng.normal(45, 4),
                }
            )

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
# Test class: 30% diagnostic warning
# ---------------------------------------------------------------------------


class TestCheckWeeklyEdgeRate:
    """Tests for check_weekly_edge_rate and the 30% WARNING."""

    def test_check_weekly_edge_rate_returns_dict(self) -> None:
        """check_weekly_edge_rate returns dict with per_week_rates, mean_rate, warnings."""
        preds, odds = _make_wp_predictions_with_edges(edge_magnitude="large")
        blender = MarketBlender(market_probability_slope_beta=_CONVERTER_SLOPE)
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
            preds_rows.append(
                {
                    "game_id": game_id,
                    "season": season,
                    "week": 1,
                    "model_prob": 0.80,
                    "actual": 1,
                }
            )
            odds_rows.append(
                {
                    "game_id": game_id,
                    "ml_home": -110,
                    "ml_away": -110,
                    "spread": -1.0,
                    "total": 45.0,
                }
            )

        preds = pd.DataFrame(preds_rows)
        odds = pd.DataFrame(odds_rows)

        # Set a low edge threshold so all games are flagged
        config = BlendConfig(
            edge_thresholds=EdgeThresholds(wp_threshold=0.01),
        )
        blender = MarketBlender(
            config=config, market_probability_slope_beta=_CONVERTER_SLOPE
        )
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
        blender = MarketBlender(
            config=config, market_probability_slope_beta=_CONVERTER_SLOPE
        )
        result = blender.check_weekly_edge_rate(preds, odds, target="wp")

        assert len(result["warnings"]) == 0

    def test_the_wp_edge_never_reads_a_closing_moneyline(self) -> None:
        """WR-05: a moneyline-only market frame supplies no WP market opinion; it refuses."""
        preds, odds = _make_wp_predictions_with_edges(n_weeks=1, games_per_week=3)
        moneyline_only = odds[["game_id", "ml_home", "ml_away"]]
        blender = MarketBlender(market_probability_slope_beta=_CONVERTER_SLOPE)
        with pytest.raises(MarketProbabilityUnavailable):
            blender.check_weekly_edge_rate(preds, moneyline_only, target="wp")

    def test_the_wp_edge_is_measured_against_the_converted_prelock_spread(self) -> None:
        """WR-05: the edge is |blend - market| with market = sigmoid(slope * spread)."""
        from scipy.special import expit, logit

        merged = pd.DataFrame(
            {"game_id": ["a", "b"], "model_prob": [0.7, 0.4], "spread": [3.0, -6.0]}
        )
        blender = MarketBlender(
            config=BlendConfig(weights=BlendWeights(wp_model_weight=0.5)),
            market_probability_slope_beta=_CONVERTER_SLOPE,
        )
        market = expit(_CONVERTER_SLOPE * np.array([3.0, -6.0]))
        blended = expit(0.5 * logit(np.array([0.7, 0.4])) + 0.5 * logit(market))
        np.testing.assert_allclose(
            blender._compute_edges("wp", merged), np.abs(blended - market)
        )

    def test_a_historical_row_is_measured_against_its_out_of_fold_market(self) -> None:
        """A frame carrying market_prob_oof is read as-is; the serving slope is not used."""
        merged = pd.DataFrame(
            {
                "game_id": ["a"],
                "model_prob": [0.6],
                "spread": [10.0],
                "market_prob_oof": [0.6],
            }
        )
        blender = MarketBlender(market_probability_slope_beta=_CONVERTER_SLOPE)
        np.testing.assert_allclose(blender._compute_edges("wp", merged), [0.0])


# ---------------------------------------------------------------------------
# Test class: Updated artifacts with edge thresholds
# ---------------------------------------------------------------------------


class TestArtifactsWithEdgeThresholds:
    """Tests for save/load artifacts including edge thresholds."""

    @pytest.fixture()
    def sample_tuning_result(self) -> TuningResult:
        """The fixed-weight TuningResult shape (Plan 33.2-24): an outcome-loss record."""
        return TuningResult(
            weights=BlendWeights(
                wp_model_weight=0.55,
                ats_model_weight=0.62,
                ou_model_weight=0.58,
            ),
            objective_by_target={
                "wp": "log_loss",
                "ats": "mean_absolute_error",
                "ou": "mean_absolute_error",
            },
            loss_by_target={"wp": 0.61, "ats": 10.1, "ou": 9.9},
            market_only_loss_by_target={"wp": 0.62, "ats": 10.2, "ou": 10.0},
            model_only_loss_by_target={"wp": 0.63, "ats": 10.3, "ou": 10.1},
            grid_by_target={"wp": [], "ats": [], "ou": []},
            seasons_by_target={"wp": [2021], "ats": [2020, 2021], "ou": [2020, 2021]},
            n_games={"wp": 400, "ats": 400, "ou": 400},
            season_best_weight_by_target={"wp": {}, "ats": {}, "ou": {}},
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
        _write_converter(tmp_path)
        blender = MarketBlender(
            config=config,
            market_probability_artifact_id=_CONVERTER_ID,
            market_probability_slope_beta=_CONVERTER_SLOPE,
        )
        artifact_dir = blender.save_blend_artifacts(
            sample_tuning_result, artifacts_dir=tmp_path, provenance=_provenance()
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
        _write_converter(tmp_path)
        blender = MarketBlender(
            config=config,
            market_probability_artifact_id=_CONVERTER_ID,
            market_probability_slope_beta=_CONVERTER_SLOPE,
        )
        # Plan 33.2-24: saving no longer moves latest.json by default, so the saved
        # directory is named explicitly rather than resolved through the manifest.
        artifact_dir = blender.save_blend_artifacts(
            sample_tuning_result, artifacts_dir=tmp_path, provenance=_provenance()
        )

        loaded = MarketBlender.from_artifacts(
            artifacts_dir=tmp_path, version=artifact_dir.name
        )
        assert loaded.config.edge_thresholds.wp_threshold == pytest.approx(0.04)
        assert loaded.config.edge_thresholds.ats_threshold == pytest.approx(1.75)
        assert loaded.config.edge_thresholds.ou_threshold == pytest.approx(1.50)
