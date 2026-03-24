"""Unit tests for MarketBlender weight tuning via walk-forward grid search.

Tests cover:
- tune_weights returns BlendWeights with all weights in [0.50, 0.70]
- tune_weights uses only seasons in tuning_predictions (temporal isolation)
- tune_weights grid searches weight range [0.50, 0.70] with step 0.01
- tune_weights with perfectly calibrated model returns weight near upper bound
- tune_weights with random model returns weight near lower bound
- tune_weights returns the weight that maximizes mean CLV
- tune_weights logs per-weight CLV for the grid search
- save_blend_artifacts writes blend_weights.json
- save_blend_artifacts includes provenance metadata
- load_blend_artifacts reads back the same BlendWeights
- temporal isolation: tune_weights raises ValueError if season > 2017
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
    TuningResult,
)


# ---------------------------------------------------------------------------
# Helpers: Synthetic tuning data
# ---------------------------------------------------------------------------


def _make_wp_predictions(
    seasons: list[int],
    n_per_season: int = 50,
    model_quality: str = "decent",
    rng_seed: int = 42,
) -> pd.DataFrame:
    """Create synthetic WP predictions for tuning tests.

    model_quality:
        - "decent": model_prob close to actual with some noise
        - "perfect": model_prob tracks actual perfectly
        - "random": model_prob is random noise
    """
    rng = np.random.default_rng(rng_seed)
    rows = []
    game_id_counter = 0
    for season in seasons:
        for week in range(1, n_per_season // 4 + 1):
            for _ in range(4):  # 4 games per week
                game_id_counter += 1
                actual = rng.choice([0, 1])
                if model_quality == "perfect":
                    model_prob = 0.95 if actual == 1 else 0.05
                elif model_quality == "random":
                    model_prob = rng.uniform(0.3, 0.7)
                else:  # decent
                    base = 0.65 if actual == 1 else 0.35
                    model_prob = float(np.clip(base + rng.normal(0, 0.08), 0.05, 0.95))
                rows.append(
                    {
                        "game_id": f"{season}_{week:02d}_{game_id_counter:04d}",
                        "season": season,
                        "week": week,
                        "model_prob": model_prob,
                        "actual": actual,
                    }
                )
    return pd.DataFrame(rows)


def _make_ats_predictions(
    seasons: list[int],
    n_per_season: int = 50,
    rng_seed: int = 42,
) -> pd.DataFrame:
    """Create synthetic ATS predictions."""
    rng = np.random.default_rng(rng_seed)
    rows = []
    game_id_counter = 0
    for season in seasons:
        for week in range(1, n_per_season // 4 + 1):
            for _ in range(4):
                game_id_counter += 1
                actual_margin = rng.normal(0, 14)
                model_spread = actual_margin + rng.normal(0, 5)
                rows.append(
                    {
                        "game_id": f"{season}_{week:02d}_{game_id_counter:04d}",
                        "season": season,
                        "week": week,
                        "model_spread": model_spread,
                        "actual": actual_margin,
                    }
                )
    return pd.DataFrame(rows)


def _make_ou_predictions(
    seasons: list[int],
    n_per_season: int = 50,
    rng_seed: int = 42,
) -> pd.DataFrame:
    """Create synthetic O/U predictions."""
    rng = np.random.default_rng(rng_seed)
    rows = []
    game_id_counter = 0
    for season in seasons:
        for week in range(1, n_per_season // 4 + 1):
            for _ in range(4):
                game_id_counter += 1
                actual_total = rng.normal(45, 7)
                model_total = actual_total + rng.normal(0, 4)
                rows.append(
                    {
                        "game_id": f"{season}_{week:02d}_{game_id_counter:04d}",
                        "season": season,
                        "week": week,
                        "model_total": model_total,
                        "actual": actual_total,
                    }
                )
    return pd.DataFrame(rows)


def _make_tuning_odds(
    predictions_dfs: dict[str, pd.DataFrame],
    rng_seed: int = 42,
) -> pd.DataFrame:
    """Create synthetic odds data aligned with prediction game_ids.

    Generates realistic moneylines, spreads, and totals for all game_ids
    found across any of the prediction DataFrames.
    """
    rng = np.random.default_rng(rng_seed)
    all_game_ids = set()
    for df in predictions_dfs.values():
        all_game_ids.update(df["game_id"].tolist())

    rows = []
    for gid in sorted(all_game_ids):
        # Realistic moneylines (home team slightly favored on average)
        home_ml = rng.choice([-150, -130, -120, -110, 100, 110, 130, 150])
        if home_ml < 0:
            away_ml = abs(home_ml) - 20
        else:
            away_ml = -(home_ml + 20)

        spread = rng.normal(-2.5, 5.0)
        total = rng.normal(45.0, 4.0)

        rows.append(
            {
                "game_id": gid,
                "ml_home": home_ml,
                "ml_away": away_ml,
                "spread": spread,
                "total": total,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Test class: Weight tuning
# ---------------------------------------------------------------------------


class TestTuneWeights:
    """Tests for MarketBlender.tune_weights grid search."""

    @pytest.fixture()
    def tuning_seasons(self) -> list[int]:
        return list(range(2010, 2018))

    @pytest.fixture()
    def tuning_predictions(self, tuning_seasons: list[int]) -> dict[str, pd.DataFrame]:
        return {
            "wp": _make_wp_predictions(tuning_seasons),
            "ats": _make_ats_predictions(tuning_seasons),
            "ou": _make_ou_predictions(tuning_seasons),
        }

    @pytest.fixture()
    def tuning_odds(self, tuning_predictions: dict[str, pd.DataFrame]) -> pd.DataFrame:
        return _make_tuning_odds(tuning_predictions)

    def test_tune_weights_returns_weights_in_range(
        self,
        tuning_predictions: dict[str, pd.DataFrame],
        tuning_odds: pd.DataFrame,
    ) -> None:
        """tune_weights returns BlendWeights with all weights in [0.50, 0.70]."""
        blender = MarketBlender()
        result = blender.tune_weights(tuning_predictions, tuning_odds)

        assert isinstance(result, TuningResult)
        assert isinstance(result.weights, BlendWeights)
        assert 0.50 <= result.weights.wp_model_weight <= 0.70
        assert 0.50 <= result.weights.ats_model_weight <= 0.70
        assert 0.50 <= result.weights.ou_model_weight <= 0.70

    def test_tune_weights_uses_only_tuning_seasons(
        self,
        tuning_predictions: dict[str, pd.DataFrame],
        tuning_odds: pd.DataFrame,
    ) -> None:
        """tune_weights uses only seasons present in tuning_predictions."""
        blender = MarketBlender()
        result = blender.tune_weights(tuning_predictions, tuning_odds)

        # All tuning seasons should be recorded
        assert result.tuning_seasons == list(range(2010, 2018))

    def test_tune_weights_grid_searches_weight_range(
        self,
        tuning_predictions: dict[str, pd.DataFrame],
        tuning_odds: pd.DataFrame,
    ) -> None:
        """tune_weights grid searches [0.50, 0.70] with 0.01 step (21 candidates)."""
        blender = MarketBlender()
        result = blender.tune_weights(tuning_predictions, tuning_odds)

        # per_target_grid should have 21 entries per target
        for target in ("wp", "ats", "ou"):
            grid = result.per_target_grid[target]
            assert len(grid) == 21, f"Expected 21 grid entries for {target}, got {len(grid)}"
            # Each entry is (weight, clv) pair
            weights = [w for w, _ in grid]
            assert min(weights) == pytest.approx(0.50, abs=0.001)
            assert max(weights) == pytest.approx(0.70, abs=0.001)

    def test_tune_weights_perfect_model_prefers_higher_weight(self) -> None:
        """tune_weights with perfectly calibrated model returns weight near upper bound."""
        seasons = list(range(2010, 2018))
        wp_preds = _make_wp_predictions(seasons, model_quality="perfect")
        predictions = {"wp": wp_preds}
        odds = _make_tuning_odds(predictions)

        blender = MarketBlender()
        result = blender.tune_weights(predictions, odds)

        # Perfect model should get a higher weight (trusts model more)
        assert result.weights.wp_model_weight >= 0.60

    def test_tune_weights_random_model_prefers_lower_weight(self) -> None:
        """tune_weights with random model returns weight near lower bound."""
        seasons = list(range(2010, 2018))
        wp_preds = _make_wp_predictions(seasons, model_quality="random")
        predictions = {"wp": wp_preds}
        odds = _make_tuning_odds(predictions)

        blender = MarketBlender()
        result = blender.tune_weights(predictions, odds)

        # Random model should get a lower weight (trusts market more)
        assert result.weights.wp_model_weight <= 0.65

    def test_tune_weights_maximizes_mean_clv(
        self,
        tuning_predictions: dict[str, pd.DataFrame],
        tuning_odds: pd.DataFrame,
    ) -> None:
        """tune_weights returns weights that maximize mean CLV across tuning predictions."""
        blender = MarketBlender()
        result = blender.tune_weights(tuning_predictions, tuning_odds)

        # For each target, the optimal weight should have the best CLV in the grid
        for target in result.per_target_grid:
            grid = result.per_target_grid[target]
            best_clv = max(clv for _, clv in grid)
            optimal_clv = result.per_target_clv[target]
            assert optimal_clv == pytest.approx(best_clv, abs=1e-10)

    def test_tune_weights_records_game_counts(
        self,
        tuning_predictions: dict[str, pd.DataFrame],
        tuning_odds: pd.DataFrame,
    ) -> None:
        """tune_weights records number of games per target in TuningResult."""
        blender = MarketBlender()
        result = blender.tune_weights(tuning_predictions, tuning_odds)

        for target in ("wp", "ats", "ou"):
            assert target in result.n_games
            assert result.n_games[target] > 0

    def test_tune_weights_updates_blender_config(
        self,
        tuning_predictions: dict[str, pd.DataFrame],
        tuning_odds: pd.DataFrame,
    ) -> None:
        """tune_weights updates self.config.weights with optimal weights."""
        blender = MarketBlender()
        result = blender.tune_weights(tuning_predictions, tuning_odds)

        # The blender's own config should now reflect the tuned weights
        assert blender.config.weights.wp_model_weight == result.weights.wp_model_weight
        assert blender.config.weights.ats_model_weight == result.weights.ats_model_weight
        assert blender.config.weights.ou_model_weight == result.weights.ou_model_weight


class TestTemporalIsolation:
    """Tests for temporal isolation in tune_weights."""

    def test_temporal_isolation_raises_on_holdout_season(self) -> None:
        """tune_weights raises ValueError if any prediction season > 2017."""
        seasons_with_leak = [2015, 2016, 2017, 2022]  # 2022 is holdout
        predictions = {"wp": _make_wp_predictions(seasons_with_leak)}
        odds = _make_tuning_odds(predictions)

        blender = MarketBlender()
        with pytest.raises(ValueError, match="Holdout data detected"):
            blender.tune_weights(predictions, odds)

    def test_temporal_isolation_allows_tuning_seasons(self) -> None:
        """tune_weights allows all seasons <= 2017."""
        seasons = [2010, 2015, 2017]
        predictions = {"wp": _make_wp_predictions(seasons)}
        odds = _make_tuning_odds(predictions)

        blender = MarketBlender()
        # Should not raise
        result = blender.tune_weights(predictions, odds)
        assert isinstance(result, TuningResult)


# ---------------------------------------------------------------------------
# Test class: Artifact save/load
# ---------------------------------------------------------------------------


class TestBlendArtifacts:
    """Tests for save_blend_artifacts and from_artifacts."""

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

    def test_save_blend_artifacts_creates_json(
        self,
        sample_tuning_result: TuningResult,
        tmp_path: Path,
    ) -> None:
        """save_blend_artifacts writes blend_weights.json to artifacts directory."""
        blender = MarketBlender()
        artifact_dir = blender.save_blend_artifacts(
            sample_tuning_result, artifacts_dir=tmp_path
        )

        weights_file = artifact_dir / "blend_weights.json"
        assert weights_file.exists()

        data = json.loads(weights_file.read_text())
        assert "weights" in data
        assert data["weights"]["wp"] == pytest.approx(0.55)
        assert data["weights"]["ats"] == pytest.approx(0.62)
        assert data["weights"]["ou"] == pytest.approx(0.58)

    def test_save_blend_artifacts_includes_provenance(
        self,
        sample_tuning_result: TuningResult,
        tmp_path: Path,
    ) -> None:
        """save_blend_artifacts includes provenance metadata."""
        blender = MarketBlender()
        artifact_dir = blender.save_blend_artifacts(
            sample_tuning_result, artifacts_dir=tmp_path
        )

        data = json.loads((artifact_dir / "blend_weights.json").read_text())
        assert "tuning_seasons" in data
        assert data["tuning_seasons"] == list(range(2010, 2018))
        assert "tuned_at" in data
        assert "per_target_clv" in data
        assert "n_games" in data
        assert "weight_range" in data
        assert "weight_step" in data

    def test_save_blend_artifacts_updates_latest_json(
        self,
        sample_tuning_result: TuningResult,
        tmp_path: Path,
    ) -> None:
        """save_blend_artifacts updates latest.json with blend key."""
        blender = MarketBlender()
        artifact_dir = blender.save_blend_artifacts(
            sample_tuning_result, artifacts_dir=tmp_path
        )

        latest_path = tmp_path / "latest.json"
        assert latest_path.exists()
        manifest = json.loads(latest_path.read_text())
        assert "blend" in manifest
        assert manifest["blend"] == artifact_dir.name

    def test_from_artifacts_loads_weights(
        self,
        sample_tuning_result: TuningResult,
        tmp_path: Path,
    ) -> None:
        """from_artifacts reads back the same BlendWeights that were saved."""
        blender = MarketBlender()
        blender.save_blend_artifacts(sample_tuning_result, artifacts_dir=tmp_path)

        loaded = MarketBlender.from_artifacts(artifacts_dir=tmp_path)
        assert loaded.config.weights.wp_model_weight == pytest.approx(0.55)
        assert loaded.config.weights.ats_model_weight == pytest.approx(0.62)
        assert loaded.config.weights.ou_model_weight == pytest.approx(0.58)
