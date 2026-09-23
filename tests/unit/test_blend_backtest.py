"""Integration-style unit tests for blended backtest pipeline (Plan 07-03).

Tests wiring between MarketBlender and BacktestEngine, BettingSimulator
schema compatibility with blended predictions, report blend delta
rendering, backward compatibility, and JSON export with blending key.

All heavy components (BacktestEngine.run, trainer training) are mocked.
Tests verify the WIRING, not model quality.

DELETED BY RULING (Plan 33.2-24 Task 2, D33.2-10): the ``TestDynamicComparison`` cases that
drove ``backtest.tune._gate_per_target`` (five), ``_generate_comparison_report`` (two) and the
per-target ``mode_by_target`` update (one). They tested the fixed-versus-DYNAMIC comparison on a
closing-line CLV metric; with the week-varying shape retired there is no second mode to gate,
and the metric itself is dead under D33.2-03. The one case that asserted something about the
SURVIVING path -- post-hoc ``blend_predictions`` equals a manual ``blend_wp`` on the bound
converter -- is kept, as ``TestPostHocBlendingEquivalence``.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from backtest.engine import BacktestConfig, BacktestResults
from backtest.report import BacktestReporter
from backtest.run import export_summary_json
from backtest.simulation import BettingSimulator, SimulationResults
from models.blending import BlendConfig, MarketBlender

# ---------------------------------------------------------------------------
# Synthetic data factories
# ---------------------------------------------------------------------------


def _make_wp_predictions(n: int = 50, season: int = 2022) -> pd.DataFrame:
    """Create synthetic WP predictions DataFrame."""
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "game_id": [
                f"{season}_W{(i % 17) + 1:02d}_TEAM{i}@HOME{i}" for i in range(n)
            ],
            "season": season,
            "week": [(i % 17) + 1 for i in range(n)],
            "model_prob": rng.uniform(0.3, 0.7, n),
            "actual": rng.integers(0, 2, n),
        }
    )


def _make_ats_predictions(n: int = 50, season: int = 2022) -> pd.DataFrame:
    """Create synthetic ATS predictions DataFrame."""
    rng = np.random.default_rng(43)
    return pd.DataFrame(
        {
            "game_id": [
                f"{season}_W{(i % 17) + 1:02d}_TEAM{i}@HOME{i}" for i in range(n)
            ],
            "season": season,
            "week": [(i % 17) + 1 for i in range(n)],
            "model_spread": rng.uniform(-10, 10, n),
            "actual": rng.uniform(-20, 20, n),
        }
    )


def _make_ou_predictions(n: int = 50, season: int = 2022) -> pd.DataFrame:
    """Create synthetic O/U predictions DataFrame."""
    rng = np.random.default_rng(44)
    return pd.DataFrame(
        {
            "game_id": [
                f"{season}_W{(i % 17) + 1:02d}_TEAM{i}@HOME{i}" for i in range(n)
            ],
            "season": season,
            "week": [(i % 17) + 1 for i in range(n)],
            "model_total": rng.uniform(35, 55, n),
            "actual": rng.uniform(30, 60, n),
        }
    )


def _make_closing_odds(game_ids: list[str]) -> pd.DataFrame:
    """Create synthetic closing odds DataFrame matching given game_ids."""
    rng = np.random.default_rng(45)
    n = len(game_ids)
    return pd.DataFrame(
        {
            "game_id": game_ids,
            "ml_home": rng.choice([-150, -130, -110, 110, 130, 150], n),
            "ml_away": rng.choice([-150, -130, -110, 110, 130, 150], n),
            "spread": rng.uniform(-7, 7, n),
            "total": rng.uniform(40, 50, n),
        }
    )


#: The production converter fit's slope (``market_probability_20260923_025709``, measured
#: over 1,342 graded 2020-2024 games). Plan 33.2-21: the WP blend takes its market opinion
#: from the pre-lock SPREAD through a BOUND converter, never from a closing moneyline.
_BOUND_CONVERTER_ID = "market_probability_20260923_025709"
_BOUND_CONVERTER_SLOPE = 0.1512435881109338


def _make_bound_blender() -> MarketBlender:
    """A blender with a converter bound to it, as Plan 33.2-24 will write one."""
    return MarketBlender(
        market_probability_artifact_id=_BOUND_CONVERTER_ID,
        market_probability_slope_beta=_BOUND_CONVERTER_SLOPE,
    )


def _make_backtest_results(
    blend_config: BlendConfig | None = None,
    is_blended: bool = False,
) -> BacktestResults:
    """Create synthetic BacktestResults with prediction DataFrames."""
    wp_preds = _make_wp_predictions()
    ats_preds = _make_ats_predictions()
    ou_preds = _make_ou_predictions()

    config = BacktestConfig(
        holdout_seasons=[2022],
        targets=["wp", "ats", "ou"],
        blend_config=blend_config,
    )

    return BacktestResults(
        config=config,
        season_results=[],
        all_predictions={
            "wp": wp_preds,
            "ats": ats_preds,
            "ou": ou_preds,
        },
        all_clv={
            "wp": wp_preds.assign(
                probability_clv=np.random.default_rng(50).uniform(
                    -0.1, 0.1, len(wp_preds)
                ),
                has_closing_odds=True,
            ),
            "ats": ats_preds.assign(
                probability_clv=np.random.default_rng(51).uniform(
                    -0.1, 0.1, len(ats_preds)
                ),
                has_closing_odds=True,
            ),
            "ou": ou_preds.assign(
                probability_clv=np.random.default_rng(52).uniform(
                    -0.1, 0.1, len(ou_preds)
                ),
                has_closing_odds=True,
            ),
        },
        headline_clv={"wp": 0.025, "ats": 0.018, "ou": -0.003},
        odds_coverage={"wp_with_odds": 50, "wp_without_odds": 0},
        covid_annotation={
            "note": "test",
            "impact_on_model": "none",
            "home_win_pct": 0.5,
            "normal_home_win_pct": 0.57,
        },
        era_info={2022: 18},
        is_blended=is_blended,
    )


def _make_simulation_results() -> SimulationResults:
    """Create minimal SimulationResults for testing."""
    from backtest.simulation import SimulationConfig, StrategyResult

    config = SimulationConfig()
    flat = StrategyResult(
        strategy_name="flat_stake",
        total_bets=100,
        winning_bets=55,
        losing_bets=43,
        push_bets=2,
        win_rate=0.55,
        total_wagered=10000.0,
        net_profit=500.0,
        roi=0.05,
        final_bankroll=10500.0,
        max_drawdown=200.0,
        max_drawdown_pct=0.02,
        equity_curve=[10000.0, 10100.0, 10500.0],
        bet_timestamps=["g1", "g2", "g3"],
    )
    kelly = StrategyResult(
        strategy_name="kelly",
        total_bets=100,
        winning_bets=55,
        losing_bets=43,
        push_bets=2,
        win_rate=0.55,
        total_wagered=8000.0,
        net_profit=600.0,
        roi=0.075,
        final_bankroll=10600.0,
        max_drawdown=150.0,
        max_drawdown_pct=0.015,
        equity_curve=[10000.0, 10200.0, 10600.0],
        bet_timestamps=["g1", "g2", "g3"],
    )
    return SimulationResults(
        config=config,
        flat_stake=flat,
        kelly=kelly,
        bet_records=[],
        by_target={},
        by_season={},
    )


# ---------------------------------------------------------------------------
# Test: BacktestConfig with blend_config=None produces is_blended=False
# ---------------------------------------------------------------------------


class TestBacktestConfigBlendField:
    """Verify blend_config field on BacktestConfig."""

    def test_default_blend_config_is_none(self) -> None:
        config = BacktestConfig()
        assert config.blend_config is None

    def test_blend_config_accepts_blend_config(self) -> None:
        bc = BlendConfig()
        config = BacktestConfig(blend_config=bc)
        assert config.blend_config is bc


class TestBacktestResultsIsBlended:
    """Verify is_blended field on BacktestResults."""

    def test_unblended_results_is_blended_false(self) -> None:
        results = _make_backtest_results(blend_config=None, is_blended=False)
        assert results.is_blended is False

    def test_blended_results_is_blended_true(self) -> None:
        bc = BlendConfig()
        results = _make_backtest_results(blend_config=bc, is_blended=True)
        assert results.is_blended is True


# ---------------------------------------------------------------------------
# Test: Blended predictions have modified values vs original
# ---------------------------------------------------------------------------


class TestBlendPredictionsModifiesValues:
    """Verify that blend_predictions actually changes model values."""

    def test_wp_blending_changes_model_prob(self) -> None:
        preds = _make_wp_predictions(n=20)
        game_ids = preds["game_id"].tolist()
        odds = _make_closing_odds(game_ids)

        original_probs = preds["model_prob"].copy()
        blender = _make_bound_blender()
        blended = blender.blend_predictions(preds, odds, "wp")

        # At least some values should have changed
        assert not blended["model_prob"].equals(original_probs), (
            "Blended WP predictions should differ from originals"
        )

    def test_ats_blending_changes_model_spread(self) -> None:
        from models.blending import MarketBlender

        preds = _make_ats_predictions(n=20)
        game_ids = preds["game_id"].tolist()
        odds = _make_closing_odds(game_ids)

        original_spreads = preds["model_spread"].copy()
        blender = MarketBlender()
        blended = blender.blend_predictions(preds, odds, "ats")

        assert not blended["model_spread"].equals(original_spreads), (
            "Blended ATS predictions should differ from originals"
        )

    def test_ou_blending_changes_model_total(self) -> None:
        from models.blending import MarketBlender

        preds = _make_ou_predictions(n=20)
        game_ids = preds["game_id"].tolist()
        odds = _make_closing_odds(game_ids)

        original_totals = preds["model_total"].copy()
        blender = MarketBlender()
        blended = blender.blend_predictions(preds, odds, "ou")

        assert not blended["model_total"].equals(original_totals), (
            "Blended O/U predictions should differ from originals"
        )


# ---------------------------------------------------------------------------
# Test: blend_predictions preserves game_id, season, week columns
# ---------------------------------------------------------------------------


class TestBlendPreservesMetadata:
    """Verify that blending preserves non-model columns."""

    def test_wp_preserves_game_id_season_week(self) -> None:
        preds = _make_wp_predictions(n=15)
        game_ids = preds["game_id"].tolist()
        odds = _make_closing_odds(game_ids)

        blender = _make_bound_blender()
        blended = blender.blend_predictions(preds, odds, "wp")

        pd.testing.assert_series_equal(
            blended["game_id"].reset_index(drop=True),
            preds["game_id"].reset_index(drop=True),
        )
        pd.testing.assert_series_equal(
            blended["season"].reset_index(drop=True),
            preds["season"].reset_index(drop=True),
        )
        pd.testing.assert_series_equal(
            blended["week"].reset_index(drop=True),
            preds["week"].reset_index(drop=True),
        )


# ---------------------------------------------------------------------------
# Test: BettingSimulator accepts blended predictions (schema compatibility)
# ---------------------------------------------------------------------------


class TestSimulatorSchemaCompatibility:
    """Verify BettingSimulator.simulate accepts blended prediction DataFrames."""

    def test_simulator_accepts_blended_wp_predictions(self) -> None:
        preds = _make_wp_predictions(n=20)
        game_ids = preds["game_id"].tolist()
        odds = _make_closing_odds(game_ids)

        blender = _make_bound_blender()
        blended_wp = blender.blend_predictions(preds, odds, "wp")

        # Merge odds columns into predictions (as BettingSimulator expects)
        merged = blended_wp.merge(odds, on="game_id", how="inner")
        merged["has_closing_odds"] = True

        # Create a mock backtest_results with blended predictions
        mock_results = MagicMock()
        mock_results.all_predictions = {"wp": merged}

        simulator = BettingSimulator()
        # This should not raise -- validates schema compatibility
        results = simulator.simulate(mock_results, odds)
        assert isinstance(results, SimulationResults)

    def test_simulator_accepts_blended_ats_predictions(self) -> None:
        from models.blending import MarketBlender

        preds = _make_ats_predictions(n=20)
        game_ids = preds["game_id"].tolist()
        odds = _make_closing_odds(game_ids)

        blender = MarketBlender()
        blended_ats = blender.blend_predictions(preds, odds, "ats")

        merged = blended_ats.merge(odds, on="game_id", how="inner")
        merged["has_closing_odds"] = True

        mock_results = MagicMock()
        mock_results.all_predictions = {"ats": merged}

        simulator = BettingSimulator()
        results = simulator.simulate(mock_results, odds)
        assert isinstance(results, SimulationResults)


# ---------------------------------------------------------------------------
# Test: BacktestReporter.generate with baseline_results produces blend_delta
# ---------------------------------------------------------------------------


class TestReporterBlendDelta:
    """Verify BacktestReporter handles blend delta correctly."""

    def test_generate_with_baseline_includes_blend_section(
        self, tmp_path: Path
    ) -> None:
        blended_results = _make_backtest_results(
            blend_config=BlendConfig(), is_blended=True
        )
        baseline_results = _make_backtest_results(blend_config=None, is_blended=False)
        # Give baseline different CLV values
        baseline_results.headline_clv = {"wp": 0.020, "ats": 0.015, "ou": -0.005}

        sim_results = _make_simulation_results()
        reporter = BacktestReporter(output_dir=tmp_path)
        report_path = reporter.generate(
            blended_results, sim_results, baseline_results=baseline_results
        )

        html_content = report_path.read_text(encoding="utf-8")
        assert "Market Blending Results" in html_content
        assert "Blended CLV" in html_content
        assert "Baseline CLV" in html_content
        assert "Delta" in html_content

    def test_generate_without_baseline_omits_blend_section(
        self, tmp_path: Path
    ) -> None:
        results = _make_backtest_results(blend_config=None, is_blended=False)
        sim_results = _make_simulation_results()
        reporter = BacktestReporter(output_dir=tmp_path)
        report_path = reporter.generate(results, sim_results)

        html_content = report_path.read_text(encoding="utf-8")
        # The blend section header is inside a Jinja2 conditional, so it
        # should not appear as a rendered <h2> when is_blended is False.
        # (The HTML comment text is always present in the template.)
        assert "<h2>Market Blending Results</h2>" not in html_content

    def test_build_blend_delta_values(self) -> None:
        blended = _make_backtest_results(is_blended=True)
        blended.headline_clv = {"wp": 0.030, "ats": 0.020}

        baseline = _make_backtest_results(is_blended=False)
        baseline.headline_clv = {"wp": 0.025, "ats": 0.022}

        reporter = BacktestReporter()
        delta = reporter._build_blend_delta(blended, baseline)

        assert delta["wp"]["blended_clv"] == pytest.approx(0.030)
        assert delta["wp"]["baseline_clv"] == pytest.approx(0.025)
        assert delta["wp"]["delta"] == pytest.approx(0.005)
        assert delta["wp"]["improved"] is True

        assert delta["ats"]["delta"] == pytest.approx(-0.002)
        assert delta["ats"]["improved"] is False


# ---------------------------------------------------------------------------
# Test: run_backtest with blend=False produces same output structure
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    """Verify that blend=False preserves Phase 6 behavior."""

    def test_unblended_results_structure(self) -> None:
        """BacktestResults without blending has expected fields."""
        results = _make_backtest_results(blend_config=None, is_blended=False)
        assert results.is_blended is False
        assert results.config.blend_config is None
        assert "wp" in results.all_predictions
        assert "ats" in results.all_predictions
        assert "ou" in results.all_predictions


# ---------------------------------------------------------------------------
# Test: export_summary_json with blended results contains "blending" key
# ---------------------------------------------------------------------------


class TestExportSummaryJsonBlending:
    """Verify blending key in JSON summary."""

    def test_blended_json_contains_blending_key(self, tmp_path: Path) -> None:
        results = _make_backtest_results(blend_config=BlendConfig(), is_blended=True)
        baseline = _make_backtest_results(blend_config=None, is_blended=False)
        baseline.headline_clv = {"wp": 0.020, "ats": 0.015, "ou": -0.005}

        sim_results = _make_simulation_results()
        json_path = export_summary_json(
            results, sim_results, tmp_path, baseline_results=baseline
        )

        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert "blending" in data
        assert data["blending"]["is_blended"] is True
        assert "blend_delta" in data["blending"]
        assert "wp" in data["blending"]["blend_delta"]

    def test_unblended_json_lacks_blending_key(self, tmp_path: Path) -> None:
        results = _make_backtest_results(blend_config=None, is_blended=False)
        sim_results = _make_simulation_results()
        json_path = export_summary_json(results, sim_results, tmp_path)

        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert "blending" not in data

    def test_blend_delta_values_in_json(self, tmp_path: Path) -> None:
        results = _make_backtest_results(blend_config=BlendConfig(), is_blended=True)
        results.headline_clv = {"wp": 0.030}

        baseline = _make_backtest_results(is_blended=False)
        baseline.headline_clv = {"wp": 0.025}

        sim_results = _make_simulation_results()
        json_path = export_summary_json(
            results, sim_results, tmp_path, baseline_results=baseline
        )

        data = json.loads(json_path.read_text(encoding="utf-8"))
        wp_delta = data["blending"]["blend_delta"]["wp"]
        assert wp_delta["blended_clv"] == pytest.approx(0.030)
        assert wp_delta["baseline_clv"] == pytest.approx(0.025)
        assert wp_delta["delta"] == pytest.approx(0.005)
        assert wp_delta["improved"] is True


# ---------------------------------------------------------------------------
# Test: post-hoc blending is the production blend
# ---------------------------------------------------------------------------


class TestPostHocBlendingEquivalence:
    """Post-hoc ``blend_predictions`` equals a manual ``blend_wp`` on the bound converter."""

    # -- Post-hoc blending equivalence test --

    def test_post_hoc_blending_equivalence(self) -> None:
        """Post-hoc blending via blend_predictions produces the same result as
        manual per-row blending, proving post-hoc is equivalent to production."""
        from models.clv import compute_clv_for_predictions

        rng = np.random.default_rng(99)
        n = 30
        game_ids = [f"2022_W{(i % 17) + 1:02d}_TEAM{i}@HOME{i}" for i in range(n)]

        preds_df = pd.DataFrame(
            {
                "game_id": game_ids,
                "season": 2022,
                "week": [(i % 17) + 1 for i in range(n)],
                "model_prob": rng.uniform(0.3, 0.7, n),
            }
        )

        odds_df = pd.DataFrame(
            {
                "game_id": game_ids,
                "ml_home": rng.choice([-150, -130, -110, 110, 130, 150], n),
                "ml_away": rng.choice([-150, -130, -110, 110, 130, 150], n),
                "spread": rng.uniform(-7, 7, n),
                "total": rng.uniform(40, 50, n),
            }
        )

        blender = _make_bound_blender()

        # Path A: blend_predictions (what run_comparison uses)
        blended_a = blender.blend_predictions(preds_df.copy(), odds_df, "wp")

        # Path B: manual blend_wp call (what engine does internally). Plan 33.2-21: the
        # market opinion is the BOUND converter applied to the pre-lock spread, not a
        # devigged closing moneyline.
        from models.market_probability import market_home_win_probability

        merged = preds_df.copy().merge(odds_df, on="game_id", how="left")
        market_prob = market_home_win_probability(
            merged["spread"].to_numpy(dtype=float), _BOUND_CONVERTER_SLOPE
        )
        blended_manual = blender.blend_wp(
            np.asarray(preds_df["model_prob"].values, dtype=np.float64),
            np.asarray(market_prob, dtype=np.float64),
        )

        # The blended model_prob values should match exactly
        np.testing.assert_array_almost_equal(
            blended_a["model_prob"].values,
            blended_manual,
            decimal=10,
            err_msg="Post-hoc blend_predictions must produce same result as manual blend_wp",
        )

        # Additionally verify CLV computation produces identical results
        clv_cols = [
            "probability_clv",
            "fair_closing_prob",
            "has_closing_odds",
            "line_clv",
            "ml_home",
            "ml_away",
            "spread",
            "total",
        ]
        drop_a = [c for c in clv_cols if c in blended_a.columns]
        clv_a = compute_clv_for_predictions(
            blended_a.drop(columns=drop_a), odds_df, "wp"
        )

        blended_b_df = preds_df.copy()
        blended_b_df["model_prob"] = blended_manual
        clv_b = compute_clv_for_predictions(blended_b_df, odds_df, "wp")

        valid_a = clv_a[clv_a["has_closing_odds"]]
        valid_b = clv_b[clv_b["has_closing_odds"]]
        np.testing.assert_array_almost_equal(
            valid_a["probability_clv"].values,
            valid_b["probability_clv"].values,
            decimal=10,
            err_msg="Post-hoc CLV must match manual CLV computation",
        )
