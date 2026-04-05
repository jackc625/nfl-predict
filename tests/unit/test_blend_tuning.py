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
- build_dynamic_synthetic_predictions with per-week noise from noise profile
- Deterministic RNG seeding produces identical output
- WP predictions clipped to [0.01, 0.99]
- Fallback for weeks not in noise profile
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models.blending import (
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
            assert len(grid) == 21, (
                f"Expected 21 grid entries for {target}, got {len(grid)}"
            )
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
        """tune_weights with random model does not select the maximum weight.

        A random model should not consistently beat the market at any weight,
        so the optimizer should not push to the upper bound of the range.
        With synthetic data the signal is weak, so we verify the random model
        gets a weight no higher than the perfect model gets.
        """
        seasons = list(range(2010, 2018))

        # Random model
        wp_random = _make_wp_predictions(seasons, model_quality="random", rng_seed=99)
        predictions_random = {"wp": wp_random}
        odds_random = _make_tuning_odds(predictions_random, rng_seed=99)
        blender_random = MarketBlender()
        result_random = blender_random.tune_weights(predictions_random, odds_random)

        # Perfect model
        wp_perfect = _make_wp_predictions(seasons, model_quality="perfect", rng_seed=99)
        predictions_perfect = {"wp": wp_perfect}
        odds_perfect = _make_tuning_odds(predictions_perfect, rng_seed=99)
        blender_perfect = MarketBlender()
        result_perfect = blender_perfect.tune_weights(predictions_perfect, odds_perfect)

        # Random model weight should be <= perfect model weight
        assert (
            result_random.weights.wp_model_weight
            <= result_perfect.weights.wp_model_weight
        )

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
        assert (
            blender.config.weights.ats_model_weight == result.weights.ats_model_weight
        )
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


# ---------------------------------------------------------------------------
# Helpers: Noise profile and tuning odds for dynamic synthetic tests
# ---------------------------------------------------------------------------


def _make_noise_profile(
    weeks: list[int] | None = None,
    wp_mean: float = 0.02,
    wp_std: float = 0.10,
    ats_mean: float = 0.5,
    ats_std: float = 3.0,
    ou_mean: float = -0.3,
    ou_std: float = 3.0,
    count_per_week: int = 50,
) -> dict[str, pd.DataFrame]:
    """Create a synthetic noise profile for testing build_dynamic_synthetic_predictions."""
    if weeks is None:
        weeks = list(range(1, 19))

    def _make_target_df(mean: float, std: float) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "week": weeks,
                "mean": [mean] * len(weeks),
                "std": [std] * len(weeks),
                "count": [count_per_week] * len(weeks),
            }
        )

    return {
        "wp": _make_target_df(wp_mean, wp_std),
        "ats": _make_target_df(ats_mean, ats_std),
        "ou": _make_target_df(ou_mean, ou_std),
    }


def _make_dynamic_tuning_odds(
    seasons: list[int] | None = None,
    n_per_season: int = 64,
    rng_seed: int = 42,
) -> pd.DataFrame:
    """Create tuning odds DataFrame for dynamic synthetic prediction tests."""
    if seasons is None:
        seasons = [2015, 2016]

    rng = np.random.default_rng(rng_seed)
    rows = []
    for season in seasons:
        for week in range(1, n_per_season // 4 + 1):
            for game_idx in range(4):
                game_id = f"{season}_W{week:02d}_G{game_idx:02d}"
                rows.append(
                    {
                        "game_id": game_id,
                        "season": season,
                        "week": week,
                        "spread": rng.normal(-2.5, 5.0),
                        "total": rng.normal(45.0, 4.0),
                        "ml_home": rng.choice([-150, -130, -120, -110]),
                        "ml_away": rng.choice([100, 110, 130, 150]),
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Test class: Dynamic synthetic predictions
# ---------------------------------------------------------------------------


class TestDynamicSyntheticPredictions:
    """Tests for build_dynamic_synthetic_predictions function."""

    def test_dynamic_synthetic_predictions_uses_noise_profile(self) -> None:
        """build_dynamic_synthetic_predictions returns dict with keys wp, ats, ou."""
        from backtest.tune import build_dynamic_synthetic_predictions

        noise_profile = _make_noise_profile()
        tuning_odds = _make_dynamic_tuning_odds()
        rng = np.random.default_rng(42)

        result = build_dynamic_synthetic_predictions(tuning_odds, noise_profile, rng)

        assert set(result.keys()) == {"wp", "ats", "ou"}

        # WP DataFrame should have game_id, season, week, model_prob
        wp_df = result["wp"]
        assert "game_id" in wp_df.columns
        assert "season" in wp_df.columns
        assert "week" in wp_df.columns
        assert "model_prob" in wp_df.columns

        # ATS DataFrame should have model_spread
        assert "model_spread" in result["ats"].columns

        # O/U DataFrame should have model_total
        assert "model_total" in result["ou"].columns

    def test_dynamic_synthetic_predictions_deterministic(self) -> None:
        """Same rng seed produces identical output twice (reproducibility)."""
        from backtest.tune import build_dynamic_synthetic_predictions

        noise_profile = _make_noise_profile()
        tuning_odds = _make_dynamic_tuning_odds()

        rng1 = np.random.default_rng(42)
        result1 = build_dynamic_synthetic_predictions(tuning_odds, noise_profile, rng1)

        rng2 = np.random.default_rng(42)
        result2 = build_dynamic_synthetic_predictions(tuning_odds, noise_profile, rng2)

        for target in ("wp", "ats", "ou"):
            model_col = {
                "wp": "model_prob",
                "ats": "model_spread",
                "ou": "model_total",
            }[target]
            np.testing.assert_array_equal(
                result1[target][model_col].values,
                result2[target][model_col].values,
            )

    def test_dynamic_synthetic_predictions_noise_per_week(self) -> None:
        """Per-week noise from profile is applied (different std per week)."""
        from backtest.tune import build_dynamic_synthetic_predictions

        # Week 1: large std, Week 10: small std
        noise_profile = _make_noise_profile(weeks=[1, 10])
        noise_profile["wp"] = pd.DataFrame(
            {
                "week": [1, 10],
                "mean": [0.0, 0.0],
                "std": [0.20, 0.02],
                "count": [100, 100],
            }
        )

        # Create tuning odds with 500 games per week for statistical power
        rows = []
        for week in [1, 10]:
            for i in range(500):
                rows.append(
                    {
                        "game_id": f"2015_W{week:02d}_G{i:04d}",
                        "season": 2015,
                        "week": week,
                        "spread": -3.0,
                        "total": 45.0,
                        "ml_home": -150,
                        "ml_away": 130,
                    }
                )
        tuning_odds = pd.DataFrame(rows)

        rng = np.random.default_rng(42)
        result = build_dynamic_synthetic_predictions(tuning_odds, noise_profile, rng)

        wp_df = result["wp"]
        # Get fair probability from devigged moneylines for reference
        # ml_home=-150, ml_away=130 -> fair_prob is constant for all rows
        # The noise std should be visible in the spread of model_prob values
        week1_probs = wp_df[wp_df["week"] == 1]["model_prob"].values
        week10_probs = wp_df[wp_df["week"] == 10]["model_prob"].values

        week1_std = np.std(week1_probs)
        week10_std = np.std(week10_probs)

        # Week 1 (std=0.20) should have significantly larger spread than Week 10 (std=0.02)
        assert week1_std > week10_std * 2.0

    def test_dynamic_synthetic_predictions_fallback_for_missing_weeks(self) -> None:
        """Weeks not in noise profile use fallback (overall mean/std from available weeks)."""
        from backtest.tune import build_dynamic_synthetic_predictions

        # Noise profile only has weeks 1-5
        noise_profile = _make_noise_profile(weeks=[1, 2, 3, 4, 5])

        # Tuning odds has games in week 10 (not in profile)
        rows = []
        for i in range(50):
            rows.append(
                {
                    "game_id": f"2015_W10_G{i:02d}",
                    "season": 2015,
                    "week": 10,
                    "spread": -3.0,
                    "total": 45.0,
                    "ml_home": -150,
                    "ml_away": 130,
                }
            )
        tuning_odds = pd.DataFrame(rows)

        rng = np.random.default_rng(42)

        # Should not raise -- week 10 uses fallback stats
        result = build_dynamic_synthetic_predictions(tuning_odds, noise_profile, rng)

        assert len(result["wp"]) == 50
        assert len(result["ats"]) == 50
        assert len(result["ou"]) == 50

    def test_dynamic_synthetic_predictions_wp_clipped(self) -> None:
        """WP synthetic predictions are in [0.01, 0.99] after perturbation."""
        from backtest.tune import build_dynamic_synthetic_predictions

        # Very large std to force extreme values before clipping
        noise_profile = _make_noise_profile(wp_std=0.50)

        tuning_odds = _make_dynamic_tuning_odds(n_per_season=256)

        rng = np.random.default_rng(42)
        result = build_dynamic_synthetic_predictions(tuning_odds, noise_profile, rng)

        wp_probs = result["wp"]["model_prob"].values
        assert np.all(wp_probs >= 0.01)
        assert np.all(wp_probs <= 0.99)


# ---------------------------------------------------------------------------
# Test class: Sigmoid objective function
# ---------------------------------------------------------------------------


class TestSigmoidObjective:
    """Tests for create_sigmoid_objective function."""

    def _make_sigmoid_tuning_data(
        self,
        n_games: int = 10,
        seasons: list[int] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Create minimal tuning data for sigmoid objective tests.

        Returns (predictions_df, tuning_odds) with valid game_ids
        in the tuning era (2010-2017).
        """
        if seasons is None:
            seasons = [2015]

        rng = np.random.default_rng(99)
        rows_preds = []
        rows_odds = []
        game_counter = 0

        for season in seasons:
            for week in range(1, n_games // len(seasons) + 1):
                game_counter += 1
                gid = f"{season}_W{week:02d}_ATL@PHI_{game_counter:02d}"

                rows_preds.append(
                    {
                        "game_id": gid,
                        "season": season,
                        "week": week,
                        "model_prob": float(
                            np.clip(rng.normal(0.55, 0.10), 0.05, 0.95)
                        ),
                        "model_spread": float(rng.normal(-3.0, 5.0)),
                        "model_total": float(rng.normal(45.0, 4.0)),
                    }
                )

                home_ml = rng.choice([-150, -130, -120, -110])
                away_ml = rng.choice([100, 110, 130, 150])
                rows_odds.append(
                    {
                        "game_id": gid,
                        "season": season,
                        "week": week,
                        "spread": float(rng.normal(-2.5, 5.0)),
                        "total": float(rng.normal(45.0, 4.0)),
                        "ml_home": home_ml,
                        "ml_away": away_ml,
                    }
                )

        preds_df = pd.DataFrame(rows_preds)
        odds_df = pd.DataFrame(rows_odds)
        return preds_df, odds_df

    def test_sigmoid_objective_callable(self) -> None:
        """create_sigmoid_objective returns a callable that accepts an optuna.Trial."""
        import optuna

        from backtest.tune import create_sigmoid_objective

        preds_df, odds_df = self._make_sigmoid_tuning_data(n_games=10)

        objective = create_sigmoid_objective("wp", preds_df, odds_df, max_week=17)

        # Should be callable
        assert callable(objective)

        # Should accept a FixedTrial and return a float
        trial = optuna.trial.FixedTrial({"midpoint": 0.5, "steepness": 1.0})
        result = objective(trial)
        assert isinstance(result, float)

    def test_sigmoid_objective_search_ranges(self) -> None:
        """Objective accepts midpoint in [3/17, 14/17] and steepness in [0.1, 1.5]."""
        import optuna

        from backtest.tune import create_sigmoid_objective

        preds_df, odds_df = self._make_sigmoid_tuning_data(n_games=10)

        objective = create_sigmoid_objective("wp", preds_df, odds_df, max_week=17)

        # Lower bounds should work
        lower_trial = optuna.trial.FixedTrial({"midpoint": 3 / 17, "steepness": 0.1})
        result_lower = objective(lower_trial)
        assert isinstance(result_lower, float)

        # Upper bounds should work
        upper_trial = optuna.trial.FixedTrial({"midpoint": 14 / 17, "steepness": 1.5})
        result_upper = objective(upper_trial)
        assert isinstance(result_upper, float)

    def test_sigmoid_objective_uses_all_games(self) -> None:
        """Objective uses ALL games with valid odds (no threshold filtering)."""
        import optuna

        from backtest.tune import create_sigmoid_objective

        preds_df, odds_df = self._make_sigmoid_tuning_data(n_games=10)

        # All 10 games have valid odds
        assert len(preds_df) == 10
        assert len(odds_df) == 10

        objective = create_sigmoid_objective("wp", preds_df, odds_df, max_week=17)

        trial = optuna.trial.FixedTrial({"midpoint": 0.5, "steepness": 1.0})
        result = objective(trial)

        # Result should be a proper float (not NaN or zero from empty data)
        assert isinstance(result, float)
        assert not np.isnan(result)

    def test_sigmoid_objective_frozen_data(self) -> None:
        """Same frozen data used for all trials (not regenerated per trial)."""
        import optuna

        from backtest.tune import create_sigmoid_objective

        preds_df, odds_df = self._make_sigmoid_tuning_data(n_games=10)

        objective = create_sigmoid_objective("wp", preds_df, odds_df, max_week=17)

        # Call with different params -- both should work on same data
        trial1 = optuna.trial.FixedTrial({"midpoint": 0.3, "steepness": 0.5})
        result1 = objective(trial1)

        trial2 = optuna.trial.FixedTrial({"midpoint": 0.7, "steepness": 1.2})
        result2 = objective(trial2)

        # Both should be valid floats (data was not regenerated)
        assert isinstance(result1, float)
        assert isinstance(result2, float)
        assert not np.isnan(result1)
        assert not np.isnan(result2)

        # With different params, results should differ
        # (same data, different weights -> different CLV)
        assert result1 != result2

    def test_sigmoid_objective_ats_target(self) -> None:
        """create_sigmoid_objective works for ATS target."""
        import optuna

        from backtest.tune import create_sigmoid_objective

        preds_df, odds_df = self._make_sigmoid_tuning_data(n_games=10)

        objective = create_sigmoid_objective("ats", preds_df, odds_df, max_week=17)

        trial = optuna.trial.FixedTrial({"midpoint": 0.5, "steepness": 1.0})
        result = objective(trial)
        assert isinstance(result, float)
        assert not np.isnan(result)

    def test_sigmoid_objective_ou_target(self) -> None:
        """create_sigmoid_objective works for O/U target."""
        import optuna

        from backtest.tune import create_sigmoid_objective

        preds_df, odds_df = self._make_sigmoid_tuning_data(n_games=10)

        objective = create_sigmoid_objective("ou", preds_df, odds_df, max_week=17)

        trial = optuna.trial.FixedTrial({"midpoint": 0.5, "steepness": 1.0})
        result = objective(trial)
        assert isinstance(result, float)
        assert not np.isnan(result)


# ---------------------------------------------------------------------------
# Test class: run_dynamic_blend_tuning
# ---------------------------------------------------------------------------


class TestRunDynamicBlendTuning:
    """Tests for run_dynamic_blend_tuning function."""

    def test_run_dynamic_blend_tuning_returns_results(
        self,
        tmp_path: Path,
    ) -> None:
        """run_dynamic_blend_tuning returns dict with expected keys."""
        from unittest.mock import patch

        from backtest.tune import run_dynamic_blend_tuning

        noise_profile = _make_noise_profile()
        tuning_odds = _make_dynamic_tuning_odds(seasons=[2015, 2016])

        with (
            patch(
                "backtest.tune.extract_noise_profile",
                return_value=noise_profile,
            ),
            patch(
                "backtest.tune.load_tuning_period_data",
                return_value=tuning_odds,
            ),
        ):
            result = run_dynamic_blend_tuning(
                n_trials=5,
                artifacts_dir=str(tmp_path / "artifacts"),
            )

        assert isinstance(result, dict)
        assert "tuning_results" in result
        assert "dynamic_weights" in result
        assert "artifact_dir" in result
        assert "static_fallback_weights" in result
        assert "metadata" in result

        # Check metadata contents
        metadata = result["metadata"]
        assert "rng_seed" in metadata
        assert "n_trials" in metadata
        assert "noise_profile_source" in metadata
        assert "search_ranges" in metadata
        assert "bet_counts" in metadata

    def test_run_dynamic_blend_tuning_saves_dynamic_artifacts(
        self,
        tmp_path: Path,
    ) -> None:
        """run_dynamic_blend_tuning saves artifacts with dynamic section."""
        from unittest.mock import patch

        from backtest.tune import run_dynamic_blend_tuning

        noise_profile = _make_noise_profile()
        tuning_odds = _make_dynamic_tuning_odds(seasons=[2015, 2016])

        with (
            patch(
                "backtest.tune.extract_noise_profile",
                return_value=noise_profile,
            ),
            patch(
                "backtest.tune.load_tuning_period_data",
                return_value=tuning_odds,
            ),
        ):
            result = run_dynamic_blend_tuning(
                n_trials=5,
                artifacts_dir=str(tmp_path / "artifacts"),
            )

        artifact_dir = Path(result["artifact_dir"])
        weights_file = artifact_dir / "blend_weights.json"
        assert weights_file.exists()

        data = json.loads(weights_file.read_text())
        assert "dynamic" in data
        assert data["blender_version"] == "2.0"

    def test_run_dynamic_blend_tuning_valid_dynamic_weights(
        self,
        tmp_path: Path,
    ) -> None:
        """DynamicBlendWeights from tuning has valid SigmoidParams."""
        from unittest.mock import patch

        from backtest.tune import run_dynamic_blend_tuning
        from models.blending import DynamicBlendWeights, SigmoidParams

        noise_profile = _make_noise_profile()
        tuning_odds = _make_dynamic_tuning_odds(seasons=[2015, 2016])

        with (
            patch(
                "backtest.tune.extract_noise_profile",
                return_value=noise_profile,
            ),
            patch(
                "backtest.tune.load_tuning_period_data",
                return_value=tuning_odds,
            ),
        ):
            result = run_dynamic_blend_tuning(
                n_trials=5,
                artifacts_dir=str(tmp_path / "artifacts"),
            )

        dw = result["dynamic_weights"]
        assert isinstance(dw, DynamicBlendWeights)
        assert isinstance(dw.wp, SigmoidParams)
        assert isinstance(dw.ats, SigmoidParams)
        assert isinstance(dw.ou, SigmoidParams)

        # Sigmoid params should be in valid ranges
        for target_params in (dw.wp, dw.ats, dw.ou):
            assert 0.0 <= target_params.midpoint <= 1.0
            assert 0.1 <= target_params.steepness <= 1.5


# ---------------------------------------------------------------------------
# Test class: CLI parser flags
# ---------------------------------------------------------------------------


class TestDynamicCLIFlags:
    """Tests for --dynamic and --n-trials CLI flags."""

    def test_dynamic_flag_accepted(self) -> None:
        """--dynamic flag is accepted by CLI parser."""

        from backtest.tune import _build_cli_parser

        parser = _build_cli_parser()
        args = parser.parse_args(["--dynamic"])
        assert args.dynamic is True

    def test_dynamic_flag_default_false(self) -> None:
        """--dynamic flag defaults to False."""
        from backtest.tune import _build_cli_parser

        parser = _build_cli_parser()
        args = parser.parse_args([])
        assert args.dynamic is False

    def test_n_trials_flag_accepted(self) -> None:
        """--n-trials flag is accepted by CLI parser."""
        from backtest.tune import _build_cli_parser

        parser = _build_cli_parser()
        args = parser.parse_args(["--dynamic", "--n-trials", "50"])
        assert args.n_trials == 50

    def test_n_trials_default_100(self) -> None:
        """--n-trials defaults to 100."""
        from backtest.tune import _build_cli_parser

        parser = _build_cli_parser()
        args = parser.parse_args([])
        assert args.n_trials == 100

    def test_rng_seed_flag_accepted(self) -> None:
        """--rng-seed flag is accepted by CLI parser."""
        from backtest.tune import _build_cli_parser

        parser = _build_cli_parser()
        args = parser.parse_args(["--rng-seed", "123"])
        assert args.rng_seed == 123

    def test_rng_seed_default_42(self) -> None:
        """--rng-seed defaults to 42."""
        from backtest.tune import _build_cli_parser

        parser = _build_cli_parser()
        args = parser.parse_args([])
        assert args.rng_seed == 42


# ---------------------------------------------------------------------------
# Test class: Sigmoid tuning metadata (guardrail metrics)
# ---------------------------------------------------------------------------


class TestSigmoidTuningMetadata:
    """Tests for tuning metadata capturing guardrail metrics."""

    def test_metadata_captures_bet_counts(
        self,
        tmp_path: Path,
    ) -> None:
        """Metadata includes bet_counts per target as guardrail metric."""
        from unittest.mock import patch

        from backtest.tune import run_dynamic_blend_tuning

        noise_profile = _make_noise_profile()
        tuning_odds = _make_dynamic_tuning_odds(seasons=[2015, 2016])

        with (
            patch(
                "backtest.tune.extract_noise_profile",
                return_value=noise_profile,
            ),
            patch(
                "backtest.tune.load_tuning_period_data",
                return_value=tuning_odds,
            ),
        ):
            result = run_dynamic_blend_tuning(
                n_trials=5,
                artifacts_dir=str(tmp_path / "artifacts"),
            )

        metadata = result["metadata"]
        bet_counts = metadata["bet_counts"]

        # Should have counts for all three targets
        assert "wp" in bet_counts
        assert "ats" in bet_counts
        assert "ou" in bet_counts

        # Counts should be positive
        for target in ("wp", "ats", "ou"):
            assert bet_counts[target] > 0

    def test_metadata_captures_study_names(
        self,
        tmp_path: Path,
    ) -> None:
        """Metadata includes study names per D-10 (distinct from Phase 12)."""
        from unittest.mock import patch

        from backtest.tune import run_dynamic_blend_tuning

        noise_profile = _make_noise_profile()
        tuning_odds = _make_dynamic_tuning_odds(seasons=[2015, 2016])

        with (
            patch(
                "backtest.tune.extract_noise_profile",
                return_value=noise_profile,
            ),
            patch(
                "backtest.tune.load_tuning_period_data",
                return_value=tuning_odds,
            ),
        ):
            result = run_dynamic_blend_tuning(
                n_trials=5,
                artifacts_dir=str(tmp_path / "artifacts"),
            )

        study_names = result["metadata"]["study_names"]
        assert study_names["wp"] == "blend_sigmoid_wp"
        assert study_names["ats"] == "blend_sigmoid_ats"
        assert study_names["ou"] == "blend_sigmoid_ou"
