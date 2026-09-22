"""Tests for betting simulation module (BACK-07 slippage, BACK-08 strategies).

Validates:
- Half-point slippage on ATS/O/U bets (BACK-07)
- Flat-stake and Kelly criterion strategies side by side (BACK-08)
- Standard -110 vig on ATS/O/U bets
- SimulationResults contains both strategy results with equity curves
- BetRecord has all required fields
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import fields
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

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


# ---------------------------------------------------------------------------
# Phase 31, plan 31-10, Task 2 (D31-04, SPEC R10): the spread Kelly defect
#
# THE DEFECT, STATED SO THE TESTS BELOW CAN BE READ AGAINST IT. The simulator sized the spread
# target's Kelly stake off ``implied + edge`` where ``edge`` was ``abs(model_spread -
# closing_spread)`` -- a POINTS DISTANCE added to an implied PROBABILITY. Measured in the live
# cache the sum crossed 1.0 for every strong signal, so ``calculate_kelly_fraction`` returned 0.0
# for 725 of them and staked only the 348 weakest; the published spread Kelly return was computed
# over that inverted selection.
#
# The fix mirrors the totals pattern exactly: an injected-selector path that sizes on the
# calibrated cover probability, and a legacy branch zeroed BY DESIGN. It is not enough to swap the
# probability inline -- Task 3's class guard is what makes a points quantity reaching a probability
# argument impossible for any target.
# ---------------------------------------------------------------------------

_ATS_SD = 13.0
_ATS_BIAS = {2021: 0.0}


def _spread_row_for_p_cover(p_cover: float) -> dict:
    """One ATS prediction row whose calibrated P(home cover) is exactly ``p_cover``.

    Inverted from the chain rather than tuned by hand: with a zero season bias the corrected margin
    IS ``model_spread``, so ``p_cover = 1 - Phi((threshold - model_spread) / sd)`` inverts to
    ``model_spread = threshold + sd * Phi_inv(p_cover)``. The threshold is the SLIPPED STORED
    SPREAD, un-negated (-2.5 for a home-cover bet on a -3.0 stored spread) -- the measured
    convention DEF-31-01 was ruled onto on 2026-09-04, in which the stored spread already IS the
    home margin the bet must clear. The legacy simulator branch keeps its older reading (DEF-31-02).
    """
    from scipy.stats import norm

    cover_threshold = -2.5
    model_spread = cover_threshold + _ATS_SD * float(norm.ppf(p_cover))
    return {
        "game_id": "2021_W01_A@B",
        "season": 2021,
        "week": 1,
        "model_spread": model_spread,
        # The realized home MARGIN. A 10-point home win clears the -2.5 cover threshold, so the
        # home-cover bet wins.
        "actual": 10.0,
        "ml_home": -110,
        "ml_away": -110,
        "spread": -3.0,
        "total": 45.0,
        "has_closing_odds": True,
    }


def _ats_selector(ev_floor_t: float = 0.0, bankroll: float = 10_000.0):
    """A BetSelector registering the SPREAD strategy alone, for injection into the simulator."""
    from backtest.bet_selector import BetSelector
    from backtest.selector_strategies import ATSStrategy

    return BetSelector(
        # The two O/U-shaped constructor arguments are inert here: only the spread strategy is
        # registered, and it carries its OWN frozen SD (a home-margin scale, not a total scale).
        frozen_sd=_ATS_SD,
        season_bias_by_season=_ATS_BIAS,
        ev_floor_t=ev_floor_t,
        bankroll=bankroll,
        strategies=[
            ATSStrategy(frozen_sd=_ATS_SD, season_bias_by_season=_ATS_BIAS),
        ],
    )


def _results_like(frames: dict[str, pd.DataFrame]):
    backtest_results = MagicMock()
    backtest_results.all_predictions = frames
    backtest_results.all_clv = dict.fromkeys(frames, pd.DataFrame())
    return backtest_results


class TestSpreadKellySizing:
    """D31-04: the spread stake comes from the calibrated cover probability or from nothing."""

    def test_injected_selector_sizes_on_the_calibrated_cover_probability(self) -> None:
        """A 0.55 cover probability gets the quarter-Kelly stake, hand-computed here.

        The arithmetic, written out rather than re-read from the calculator: at -110 the net odds
        are ``b = 100/110`` and the implied probability is ``110/210``. The raw Kelly fraction is
        ``(b * p - (1 - p)) / b``; quarter Kelly scales it by 0.25; the confidence adjustment for
        an edge of 0.0262 is 0.6; there is no drawdown so the risk adjustment is 1.0; the 5%
        per-bet ceiling does not bind. On a 10,000 bankroll that is 82.50.
        """
        sim = BettingSimulator(
            SimulationConfig(), ats_bet_selector=_ats_selector(ev_floor_t=0.0)
        )
        frame = pd.DataFrame([_spread_row_for_p_cover(0.55)])
        results = sim.simulate(_results_like({"ats": frame}), pd.DataFrame())

        ats_bets = [r for r in results.bet_records if r.target == "ats"]
        assert len(ats_bets) == 1
        bet = ats_bets[0]

        p = 0.55
        b = 100.0 / 110.0
        raw_kelly = (b * p - (1.0 - p)) / b
        expected = 10_000.0 * min(0.05, raw_kelly * 0.25 * 0.6)
        assert bet.kelly_stake == pytest.approx(expected, abs=1e-9)
        assert bet.kelly_stake == pytest.approx(82.5, abs=1e-6)

        # The number handed to Kelly is a PROBABILITY, not a points distance.
        assert 0.0 < bet.model_value < 1.0
        assert bet.model_value == pytest.approx(0.55, abs=1e-9)
        # And the bet is graded on the margin scale: a 10-point home win clears the -2.5 threshold.
        assert bet.bet_side == "home_cover"
        assert bet.outcome is True

    def test_the_legacy_branch_stakes_exactly_zero_by_design(self) -> None:
        """The SAME row with no selector injected: a flat bet is graded, the Kelly stake is 0.0.

        Zero is the designed outcome, not an accident of this fixture: without the calibrated cover
        probability there is no honest Kelly probability for the spread target, and the quantity the
        legacy branch used to substitute was a points distance.
        """
        sim = BettingSimulator(SimulationConfig())
        frame = pd.DataFrame([_spread_row_for_p_cover(0.55)])
        results = sim.simulate(_results_like({"ats": frame}), pd.DataFrame())

        ats_bets = [r for r in results.bet_records if r.target == "ats"]
        assert len(ats_bets) == 1
        assert ats_bets[0].kelly_stake == 0.0
        # The flat-stake ledger is untouched: the bet is still placed and still graded, which is
        # what keeps the published spread bet POPULATION where it was.
        assert ats_bets[0].flat_stake == SimulationConfig().flat_stake_amount
        assert ats_bets[0].outcome is not None

    def test_the_legacy_branch_carries_an_explaining_comment(self) -> None:
        """The zero is documented at the branch, in the shape the legacy totals branch already uses.

        A zero with no explanation reads as a bug to the next reader, and the next reader's repair
        would be to restore the points-distance expression.
        """
        source = inspect.getsource(BettingSimulator.simulate)
        marker = "the spread Kelly stake here is 0 by design"
        assert marker in source, (
            "the legacy spread branch does not name the reason its stake is zero"
        )


class TestWinnerBranchIsUntouched:
    """D31-04 leaves WP alone; its model value is already the side-correct probability."""

    # RECORDED from the simulator at commit 872f8d7 -- the parent of the Task 2 edit -- over the
    # fixed week below. Exact reprs, compared with ``==`` and no tolerance: the claim is that the
    # winner stakes are BIT-IDENTICAL before and after, and a tolerance would not be that claim.
    #
    # ``2021_W03_K@L`` (model_prob 0.51) is absent because it is inside the LOCKED no-bet band and
    # no bet was placed. ``2021_W03_I@J`` is present with a stake of 0.0: an 0.81 probability
    # against a -400 line is a 0.01 edge, at or below the calculator's own confidence threshold.
    # That zero is a legitimate NO-EDGE Kelly refusal and is NOT the spread defect -- the winner
    # branch consumes a genuine probability both before and after this plan.
    _RECORDED_WP_STAKES = {
        "2021_W01_A@B": 499.99999999999955,
        "2021_W01_C@D": 417.4666666666668,
        "2021_W02_E@F": 436.2981333333331,
        "2021_W02_G@H": 118.4419286250001,
        "2021_W03_I@J": 0.0,
    }
    _RECORDED_WP_FINAL_BANKROLL = 10301.354836785713

    _WP_WEEK = [
        ("2021_W01_A@B", 0.70, -150, 130, 1),
        ("2021_W01_C@D", 0.62, -110, -110, 0),
        ("2021_W02_E@F", 0.30, 140, -160, 0),
        ("2021_W02_G@H", 0.55, -105, -115, 1),
        ("2021_W03_I@J", 0.81, -400, 320, 1),
        ("2021_W03_K@L", 0.51, -110, -110, 0),
    ]

    def _frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "game_id": game_id,
                    "season": 2021,
                    "week": int(game_id.split("_W")[1][:2]),
                    "model_prob": model_prob,
                    "actual": actual,
                    "ml_home": ml_home,
                    "ml_away": ml_away,
                    "spread": -3.0,
                    "total": 45.0,
                    "has_closing_odds": True,
                }
                for game_id, model_prob, ml_home, ml_away, actual in self._WP_WEEK
            ]
        )

    def test_winner_stakes_are_bit_identical_to_the_recorded_run(self) -> None:
        """Every winner stake, and the final Kelly bankroll, compare EQUAL with no tolerance."""
        sim = BettingSimulator(SimulationConfig())
        results = sim.simulate(_results_like({"wp": self._frame()}), pd.DataFrame())

        stakes = {
            r.game_id: r.kelly_stake for r in results.bet_records if r.target == "wp"
        }
        assert stakes == self._RECORDED_WP_STAKES
        assert results.kelly.final_bankroll == self._RECORDED_WP_FINAL_BANKROLL

    def test_winner_stakes_do_not_move_when_a_spread_selector_is_injected(self) -> None:
        """Injecting the spread selector changes the spread target and nothing else."""
        sim = BettingSimulator(SimulationConfig(), ats_bet_selector=_ats_selector())
        results = sim.simulate(_results_like({"wp": self._frame()}), pd.DataFrame())

        stakes = {
            r.game_id: r.kelly_stake for r in results.bet_records if r.target == "wp"
        }
        assert stakes == self._RECORDED_WP_STAKES

    def test_every_winner_kelly_input_is_a_probability(self) -> None:
        """The winner branch's model value is in (0, 1] for every bet it places.

        Measured, not assumed: this is the reason the plan does not touch the branch.
        """
        sim = BettingSimulator(SimulationConfig())
        results = sim.simulate(_results_like({"wp": self._frame()}), pd.DataFrame())
        wp_bets = [r for r in results.bet_records if r.target == "wp"]
        assert wp_bets
        for bet in wp_bets:
            assert 0.0 < bet.model_value <= 1.0


class TestAWinnerBetWithNoPriceIsSkipped:
    """A WP side with no moneyline is NOT placed, NOT graded and recorded as ``no_price``.

    Plan 33.2-08 (owner ruling) deliberately blanked disputed moneylines (e.g.
    ``2024_W17_TEN@JAX``, ``2022_W08_SF@LA``) and a live game can lack a price, so this is
    reachable in real grading. ``_get_wp_odds`` used to reach ``int(nan)`` and crash the whole
    simulation with "cannot convert float NaN to integer"; a default price would invent a market.
    """

    _WP_WEEK = TestWinnerBranchIsUntouched._WP_WEEK

    def _frame(self, plant: dict[str, object] | None = None) -> pd.DataFrame:
        frame = TestWinnerBranchIsUntouched._frame(self)
        if plant:
            target = frame["game_id"] == "2021_W01_A@B"  # model 0.70 -> a HOME bet
            for column, value in plant.items():
                frame[column] = frame[column].astype(object)
                frame.loc[target, column] = value
        return frame

    @staticmethod
    def _run(frame: pd.DataFrame):
        sim = BettingSimulator(SimulationConfig())
        return sim.simulate(_results_like({"wp": frame}), pd.DataFrame())

    @pytest.mark.parametrize("missing", [float("nan"), None], ids=["nan", "none"])
    def test_a_missing_price_on_the_bet_side_is_skipped_by_name(self, missing) -> None:
        from backtest.simulation import NO_PRICE_REASON, SkippedBet

        results = self._run(self._frame({"ml_home": missing}))
        assert "2021_W01_A@B" not in {r.game_id for r in results.bet_records}
        assert results.skipped_bets == [
            SkippedBet(
                game_id="2021_W01_A@B",
                season=2021,
                week=1,
                target="wp",
                bet_side="home",
                reason=NO_PRICE_REASON,
            )
        ]
        assert NO_PRICE_REASON == "no_price"

    def test_every_other_game_is_exactly_as_if_the_unpriced_game_were_absent(
        self,
    ) -> None:
        skipped = self._run(self._frame({"ml_home": float("nan")}))
        frame = self._frame()
        absent = self._run(frame.loc[frame["game_id"] != "2021_W01_A@B"])
        assert skipped.bet_records == absent.bet_records
        assert skipped.kelly.final_bankroll == absent.kelly.final_bankroll
        assert absent.skipped_bets == []

    def test_a_missing_price_on_the_other_side_places_the_bet_unchanged(self) -> None:
        baseline = self._run(self._frame())
        other_side = self._run(self._frame({"ml_away": float("nan")}))
        assert other_side.bet_records == baseline.bet_records
        assert other_side.skipped_bets == []

    def test_the_odds_helper_returns_none_rather_than_a_default(self) -> None:
        sim = BettingSimulator(SimulationConfig())
        assert sim._get_wp_odds("home", float("nan"), 130.0) is None
        assert sim._get_wp_odds("away", -150.0, None) is None
        assert sim._get_wp_odds("away", float("nan"), 130.0) == 130
        assert sim._get_wp_odds("home", -150.0, float("nan")) == -150


class TestNoEdgeReachesAProbabilityArgument:
    """SPEC R10: the defect is removed structurally, for all three target branches."""

    def test_no_branch_adds_an_edge_or_an_implied_probability_to_anything(self) -> None:
        """An AST scan over ``simulate``: no addition mixes an edge with an implied probability.

        Keyed on the OPERANDS rather than on a name: renaming ``kelly_model_prob`` would defeat a
        substring scan while leaving the defect exactly where it was.
        """
        tree = ast.parse(textwrap.dedent(inspect.getsource(BettingSimulator.simulate)))
        forbidden_names = {"edge", "implied", "points_edge", "market_prob"}

        additions = [node for node in ast.walk(tree) if isinstance(node, ast.BinOp)]
        additions = [node for node in additions if isinstance(node.op, ast.Add)]
        for node in additions:
            for operand in (node.left, node.right):
                assert not (
                    isinstance(operand, ast.Name) and operand.id in forbidden_names
                ), (
                    f"simulate() adds {getattr(operand, 'id', operand)!r} to another quantity"
                )
                assert not (
                    isinstance(operand, ast.Call)
                    and isinstance(operand.func, ast.Name)
                    and operand.func.id == "moneyline_to_probability"
                ), "simulate() adds an implied probability to another quantity"

        # Non-vacuity: the scan really did read the function that carries all three branches.
        source = inspect.getsource(BettingSimulator.simulate)
        for target_code in ('"wp"', '"ats"', '"ou"'):
            assert target_code in source

    def test_every_kelly_call_in_simulate_passes_a_bare_probability_name(self) -> None:
        """``model_prob=`` is a plain name, never an arithmetic expression.

        This is the shape the defect had: an expression computed at the call site. A bare name can
        still hold the wrong value, which is why Task 3's guard validates the value as well.
        """
        tree = ast.parse(textwrap.dedent(inspect.getsource(BettingSimulator.simulate)))
        call_sites = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "calculate_optimal_bet_size"
        ]
        assert call_sites, (
            "simulate() no longer sizes any bet -- the scan would pass vacuously"
        )
        for call in call_sites:
            passed = {kw.arg: kw.value for kw in call.keywords}
            assert "model_prob" in passed
            assert isinstance(passed["model_prob"], ast.Name), (
                "a Kelly probability argument is an expression, not a named value"
            )


class TestCachePopulationInjectsNoSelector:
    """D31-04: /betting's published bet POPULATION does not move, so its path injects nothing."""

    def test_the_backtest_runner_constructs_the_simulator_without_a_selector(
        self,
    ) -> None:
        """The call site that writes ``betting_simulation.csv`` passes no selector at all.

        Injecting selectors there would move /betting's winner and spread bet counts, hit rates and
        returns -- reworking that page, which D31-04 rules out. The consequence is that the page's
        spread Kelly figure reads zero, which is recorded rather than removed quietly.
        """
        tree = ast.parse(Path("backtest/run.py").read_text(encoding="utf-8"))
        constructions = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "BettingSimulator"
        ]
        assert constructions, "backtest/run.py no longer builds a BettingSimulator"
        for call in constructions:
            keywords = {kw.arg for kw in call.keywords}
            assert not any(
                name and name.endswith("_bet_selector") for name in keywords
            ), "the cache-population path injects a bet selector"
            # A selector could also arrive positionally; only ``config`` may be passed that way.
            assert len(call.args) <= 1
