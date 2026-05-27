"""Tests for betting simulation module (BACK-07 slippage, BACK-08 strategies).

Validates:
- Half-point slippage on ATS/O/U bets (BACK-07)
- Flat-stake and Kelly criterion strategies side by side (BACK-08)
- Standard -110 vig on ATS/O/U bets
- SimulationResults contains both strategy results with equity curves
- BetRecord has all required fields
"""

from __future__ import annotations

from dataclasses import fields
from unittest.mock import MagicMock

import pandas as pd

from backtest.simulation import (
    SLIPPAGE_POINTS,
    STANDARD_VIG_ODDS,
    BetRecord,
    BettingSimulator,
    SimulationConfig,
    SimulationResults,
    StrategyResult,
    apply_slippage_spread,
    apply_slippage_total,
)

# ---------------------------------------------------------------------------
# Slippage tests (BACK-07)
# ---------------------------------------------------------------------------


class TestSlippageConstants:
    """Verify simulation constants are correct."""

    def test_slippage_constant(self) -> None:
        assert SLIPPAGE_POINTS == 0.5

    def test_standard_vig_constant(self) -> None:
        assert STANDARD_VIG_ODDS == -110


class TestSlippageSpread:
    """Spread slippage moves the line against the bettor."""

    def test_slippage_spread_home_cover(self) -> None:
        """Home cover: spread becomes more negative (harder to cover)."""
        assert apply_slippage_spread(-3.0, "home_cover") == -3.5

    def test_slippage_spread_away_cover(self) -> None:
        """Away cover: spread becomes less negative (away gets fewer points)."""
        assert apply_slippage_spread(-3.0, "away_cover") == -2.5

    def test_slippage_spread_custom_amount(self) -> None:
        """Custom slippage amount works correctly."""
        assert apply_slippage_spread(-3.0, "home_cover", slippage=1.0) == -4.0
        assert apply_slippage_spread(-3.0, "away_cover", slippage=1.0) == -2.0

    def test_slippage_spread_positive_spread(self) -> None:
        """Positive spread (home underdog) also slips correctly."""
        # Home cover with +3.0 spread: becomes +2.5 (harder for home)
        assert apply_slippage_spread(3.0, "home_cover") == 2.5
        # Away cover with +3.0 spread: becomes +3.5 (harder for away)
        assert apply_slippage_spread(3.0, "away_cover") == 3.5


class TestSlippageTotal:
    """Total slippage moves the line against the bettor."""

    def test_slippage_total_over(self) -> None:
        """Over: line goes up (harder to go over)."""
        assert apply_slippage_total(44.0, "over") == 44.5

    def test_slippage_total_under(self) -> None:
        """Under: line goes down (harder to go under)."""
        assert apply_slippage_total(44.0, "under") == 43.5

    def test_slippage_total_custom_amount(self) -> None:
        """Custom slippage amount works correctly."""
        assert apply_slippage_total(44.0, "over", slippage=1.0) == 45.0
        assert apply_slippage_total(44.0, "under", slippage=1.0) == 43.0


# ---------------------------------------------------------------------------
# Strategy tests (BACK-08)
# ---------------------------------------------------------------------------


class TestFlatStakeStrategy:
    """Flat-stake strategy correctness."""

    def test_flat_stake_positive_roi(self) -> None:
        """60% win rate at -110 odds produces positive ROI.

        Expected: 60 wins * $90.91 payout - 40 losses * $100 stake
        = $5454.55 - $4000 = $1454.55 net profit on $10000 wagered
        """
        config = SimulationConfig(
            starting_bankroll=10_000.0,
            flat_stake_amount=100.0,
        )
        sim = BettingSimulator(config=config)

        # Create a synthetic BacktestResults with WP predictions
        # 60 wins and 40 losses
        n_wins = 60
        n_losses = 40
        n_total = n_wins + n_losses

        # Build predictions with strong edge for winners, negative for losers
        game_ids = [f"2021_{i:02d}_HOME_AWAY" for i in range(n_total)]
        model_probs = [0.70] * n_wins + [0.30] * n_losses
        actuals = [1] * n_wins + [0] * n_losses

        predictions_df = pd.DataFrame(
            {
                "game_id": game_ids,
                "season": [2021] * n_total,
                "week": list(range(1, n_total + 1)),
                "model_prob": model_probs,
                "actual": actuals,
            }
        )

        closing_odds_df = pd.DataFrame(
            {
                "game_id": game_ids,
                "ml_home": [-150] * n_total,
                "ml_away": [130] * n_total,
                "spread": [-3.0] * n_total,
                "total": [44.0] * n_total,
            }
        )

        backtest_results = MagicMock()
        backtest_results.all_predictions = {"wp": predictions_df}
        backtest_results.all_clv = {"wp": pd.DataFrame()}

        result = sim.simulate(backtest_results, closing_odds_df)
        assert result.flat_stake.final_bankroll > config.starting_bankroll
        assert result.flat_stake.roi > 0

    def test_flat_stake_negative_roi(self) -> None:
        """45% win rate at -110 odds produces negative ROI."""
        config = SimulationConfig(
            starting_bankroll=10_000.0,
            flat_stake_amount=100.0,
        )
        sim = BettingSimulator(config=config)

        n_wins = 45
        n_losses = 55
        n_total = n_wins + n_losses

        game_ids = [f"2021_{i:02d}_HOME_AWAY" for i in range(n_total)]
        # All predict home strongly so they all get bet
        # Winners: model says 0.70 and home wins
        # Losers: model says 0.70 but home loses
        model_probs = [0.70] * n_total
        actuals = [1] * n_wins + [0] * n_losses

        predictions_df = pd.DataFrame(
            {
                "game_id": game_ids,
                "season": [2021] * n_total,
                "week": list(range(1, n_total + 1)),
                "model_prob": model_probs,
                "actual": actuals,
            }
        )

        closing_odds_df = pd.DataFrame(
            {
                "game_id": game_ids,
                "ml_home": [-150] * n_total,
                "ml_away": [130] * n_total,
                "spread": [-3.0] * n_total,
                "total": [44.0] * n_total,
            }
        )

        backtest_results = MagicMock()
        backtest_results.all_predictions = {"wp": predictions_df}
        backtest_results.all_clv = {"wp": pd.DataFrame()}

        result = sim.simulate(backtest_results, closing_odds_df)
        assert result.flat_stake.final_bankroll < config.starting_bankroll
        assert result.flat_stake.roi < 0


class TestKellyStrategy:
    """Kelly criterion strategy correctness."""

    def test_kelly_no_bet_when_no_edge(self) -> None:
        """Kelly returns zero bet when model_prob <= implied_prob + threshold.

        -110 implies ~52.4% probability. model_prob=0.51 has negative edge.
        """
        from utils.kelly_criterion import KellyCalculator

        calc = KellyCalculator(
            starting_bankroll=10_000.0,
            default_kelly_fraction=0.25,
            confidence_threshold=0.02,
        )
        result = calc.calculate_optimal_bet_size(model_prob=0.51, market_odds=-110)
        assert result.recommended_bet == 0.0

    def test_kelly_bets_with_sufficient_edge(self) -> None:
        """Kelly returns non-zero bet when model_prob >> implied_prob.

        model_prob=0.60, -110 implies ~0.524. Edge ~7.6%.
        """
        from utils.kelly_criterion import KellyCalculator

        calc = KellyCalculator(
            starting_bankroll=10_000.0,
            default_kelly_fraction=0.25,
            confidence_threshold=0.02,
        )
        result = calc.calculate_optimal_bet_size(model_prob=0.60, market_odds=-110)
        assert result.recommended_bet > 0.0


class TestSimulationResults:
    """SimulationResults structure tests."""

    def test_simulation_results_has_both_strategies(self) -> None:
        """simulate() returns SimulationResults with flat_stake and kelly."""
        config = SimulationConfig(
            starting_bankroll=10_000.0,
            flat_stake_amount=100.0,
            min_edge_threshold=0.02,
        )
        sim = BettingSimulator(config=config)

        n_total = 20
        game_ids = [f"2021_{i:02d}_HOME_AWAY" for i in range(n_total)]

        predictions_df = pd.DataFrame(
            {
                "game_id": game_ids,
                "season": [2021] * n_total,
                "week": list(range(1, n_total + 1)),
                "model_prob": [0.70] * n_total,
                "actual": [1, 0] * (n_total // 2),
            }
        )

        closing_odds_df = pd.DataFrame(
            {
                "game_id": game_ids,
                "ml_home": [-150] * n_total,
                "ml_away": [130] * n_total,
                "spread": [-3.0] * n_total,
                "total": [44.0] * n_total,
            }
        )

        backtest_results = MagicMock()
        backtest_results.all_predictions = {"wp": predictions_df}
        backtest_results.all_clv = {"wp": pd.DataFrame()}

        result = sim.simulate(backtest_results, closing_odds_df)

        assert isinstance(result, SimulationResults)
        assert isinstance(result.flat_stake, StrategyResult)
        assert isinstance(result.kelly, StrategyResult)
        assert result.flat_stake.strategy_name == "flat_stake"
        assert result.kelly.strategy_name == "kelly"
        assert len(result.flat_stake.equity_curve) > 0
        assert len(result.kelly.equity_curve) > 0

    def test_bet_record_has_required_fields(self) -> None:
        """BetRecord dataclass has all required fields."""
        required_fields = {
            "game_id",
            "target",
            "bet_side",
            "odds",
            "flat_stake",
            "kelly_stake",
            "outcome",
            "slipped_line",
            "season",
            "week",
            "model_value",
            "market_value",
            "edge",
            "payout_flat",
            "payout_kelly",
        }
        actual_fields = {f.name for f in fields(BetRecord)}
        assert required_fields.issubset(actual_fields), (
            f"Missing fields: {required_fields - actual_fields}"
        )

    def test_simulation_by_target_breakdown(self) -> None:
        """by_target dict has entries for each target with n_bets, win_rate."""
        config = SimulationConfig(starting_bankroll=10_000.0)
        sim = BettingSimulator(config=config)

        n_total = 10
        game_ids = [f"2021_{i:02d}_HOME_AWAY" for i in range(n_total)]

        predictions_df = pd.DataFrame(
            {
                "game_id": game_ids,
                "season": [2021] * n_total,
                "week": list(range(1, n_total + 1)),
                "model_prob": [0.70] * n_total,
                "actual": [1, 0] * (n_total // 2),
            }
        )

        closing_odds_df = pd.DataFrame(
            {
                "game_id": game_ids,
                "ml_home": [-150] * n_total,
                "ml_away": [130] * n_total,
                "spread": [-3.0] * n_total,
                "total": [44.0] * n_total,
            }
        )

        backtest_results = MagicMock()
        backtest_results.all_predictions = {"wp": predictions_df}
        backtest_results.all_clv = {"wp": pd.DataFrame()}

        result = sim.simulate(backtest_results, closing_odds_df)
        assert "wp" in result.by_target
        assert "n_bets" in result.by_target["wp"]
        assert "win_rate" in result.by_target["wp"]
