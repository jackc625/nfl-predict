"""Betting simulation module for backtest evaluation.

Provides:
- BettingSimulator: Runs flat-stake and Kelly strategies side by side
- SimulationConfig: Configuration for simulation parameters
- SimulationResults: Complete simulation output with strategy comparison
- BetRecord: Individual bet details for both strategies
- StrategyResult: Per-strategy performance summary with equity curve

Key design decisions:
- Half-point slippage on every ATS/O/U bet (D-07, BACK-07)
- Standard -110 vig assumed on ATS/O/U bets
- Flat-stake and quarter-Kelly run side by side (D-08, BACK-08)
- WP uses moneyline directly (no spread slippage)
- Equity curves tracked chronologically for report charts (Plan 03)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from utils import get_logger
from utils.kelly_criterion import KellyCalculator, KellyMode
from utils.probability_utils import moneyline_to_probability

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SLIPPAGE_POINTS: float = 0.5
"""Half-point slippage applied to every simulated ATS/O/U bet (D-07)."""

STANDARD_VIG_ODDS: int = -110
"""Standard vig on ATS/O/U bets."""

DEFAULT_STARTING_BANKROLL: float = 10_000.0
DEFAULT_FLAT_STAKE: float = 100.0
DEFAULT_KELLY_FRACTION: float = 0.25
DEFAULT_MIN_EDGE: float = 0.02


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class SimulationConfig:
    """Configuration for betting simulation.

    Attributes:
        starting_bankroll: Initial bankroll for both strategies.
        flat_stake_amount: Fixed bet size for flat-stake strategy.
        kelly_fraction: Fractional Kelly multiplier (0.25 = quarter Kelly).
        min_edge_threshold: Minimum edge required to place a bet.
        max_kelly_bet_pct: Maximum Kelly bet as percentage of bankroll.
        slippage_points: Half-point slippage amount for ATS/O/U.
        standard_vig_odds: Standard vig odds for ATS/O/U bets.
    """

    starting_bankroll: float = DEFAULT_STARTING_BANKROLL
    flat_stake_amount: float = DEFAULT_FLAT_STAKE
    kelly_fraction: float = DEFAULT_KELLY_FRACTION
    min_edge_threshold: float = DEFAULT_MIN_EDGE
    max_kelly_bet_pct: float = 0.05
    slippage_points: float = SLIPPAGE_POINTS
    standard_vig_odds: int = STANDARD_VIG_ODDS


@dataclass
class BetRecord:
    """Record of a single simulated bet with both strategy stakes.

    Captures all details needed for CSV export in Plan 03 and
    strategy comparison in the report.
    """

    game_id: str
    season: int
    week: int
    target: str  # wp / ats / ou
    bet_side: str  # home/away for WP, home_cover/away_cover for ATS, over/under for O/U
    model_value: float  # model probability or predicted margin/total
    market_value: float  # closing odds implied prob or closing line
    edge: float  # model_value - market_value for WP; computed edge for ATS/O/U
    slipped_line: float | None  # line after slippage for ATS/O/U; None for WP ML
    odds: int  # American odds for the bet
    flat_stake: float  # bet amount under flat-stake strategy
    kelly_stake: float  # bet amount under Kelly strategy
    outcome: bool | None  # True=win, False=loss, None=push
    payout_flat: float  # payout under flat-stake
    payout_kelly: float  # payout under Kelly


@dataclass
class StrategyResult:
    """Performance summary for a single strategy.

    Attributes:
        strategy_name: Identifier ("flat_stake" or "kelly").
        total_bets: Number of bets placed.
        winning_bets: Number of winning bets.
        losing_bets: Number of losing bets.
        push_bets: Number of push (tie) bets.
        win_rate: Winning percentage.
        total_wagered: Total amount wagered.
        net_profit: Net profit/loss.
        roi: Return on investment (net_profit / total_wagered).
        final_bankroll: Ending bankroll value.
        max_drawdown: Maximum peak-to-trough drawdown in dollars.
        max_drawdown_pct: Maximum drawdown as percentage of peak.
        equity_curve: Bankroll value after each bet, chronological.
        bet_timestamps: Game IDs in chronological order for equity curve x-axis.
    """

    strategy_name: str
    total_bets: int
    winning_bets: int
    losing_bets: int
    push_bets: int
    win_rate: float
    total_wagered: float
    net_profit: float
    roi: float
    final_bankroll: float
    max_drawdown: float
    max_drawdown_pct: float
    equity_curve: list[float]
    bet_timestamps: list[str]


@dataclass
class SimulationResults:
    """Complete simulation output with both strategies.

    Attributes:
        config: Simulation configuration used.
        flat_stake: Flat-stake strategy results.
        kelly: Kelly criterion strategy results.
        bet_records: All individual bet records.
        by_target: Breakdown per target (n_bets, win_rate, roi per strategy).
        by_season: Breakdown per season.
    """

    config: SimulationConfig
    flat_stake: StrategyResult
    kelly: StrategyResult
    bet_records: list[BetRecord]
    by_target: dict[str, dict[str, Any]]
    by_season: dict[int, dict[str, Any]]


# ---------------------------------------------------------------------------
# Slippage functions
# ---------------------------------------------------------------------------


def apply_slippage_spread(
    line: float, side: str, slippage: float = SLIPPAGE_POINTS
) -> float:
    """Apply half-point slippage to a spread bet, moving against the bettor.

    Args:
        line: Spread from the home team's perspective (negative = home favored).
        side: "home_cover" or "away_cover".
        slippage: Points of slippage to apply.

    Returns:
        Slipped spread line.

    Examples:
        >>> apply_slippage_spread(-3.0, "home_cover")
        -3.5
        >>> apply_slippage_spread(-3.0, "away_cover")
        -2.5
    """
    if side == "home_cover":
        # Home bettor needs home to cover: make spread more negative (harder)
        return line - slippage
    if side == "away_cover":
        # Away bettor gets fewer points: make spread less negative (harder for away)
        return line + slippage
    msg = f"Unknown spread bet side: '{side}'. Must be 'home_cover' or 'away_cover'."
    raise ValueError(msg)


def apply_slippage_total(
    line: float, side: str, slippage: float = SLIPPAGE_POINTS
) -> float:
    """Apply half-point slippage to an over/under bet, moving against the bettor.

    Args:
        line: Game total line.
        side: "over" or "under".
        slippage: Points of slippage to apply.

    Returns:
        Slipped total line.

    Examples:
        >>> apply_slippage_total(44.0, "over")
        44.5
        >>> apply_slippage_total(44.0, "under")
        43.5
    """
    if side == "over":
        # Over bettor: line goes up (harder to go over)
        return line + slippage
    if side == "under":
        # Under bettor: line goes down (harder to go under)
        return line - slippage
    msg = f"Unknown total bet side: '{side}'. Must be 'over' or 'under'."
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# Payout helpers
# ---------------------------------------------------------------------------


def _calculate_payout(stake: float, odds: int, outcome: bool | None) -> float:
    """Calculate payout for a bet given American odds and outcome.

    Args:
        stake: Amount wagered.
        odds: American odds.
        outcome: True=win, False=loss, None=push.

    Returns:
        Net payout (profit if won, -stake if lost, 0 if push).
    """
    if outcome is None:
        return 0.0  # Push: stake returned, net zero
    if not outcome:
        return -stake  # Loss
    # Win: profit depends on odds
    if odds > 0:
        return stake * (odds / 100)
    return stake * (100 / abs(odds))


# ---------------------------------------------------------------------------
# BettingSimulator
# ---------------------------------------------------------------------------


class BettingSimulator:
    """Simulates flat-stake and Kelly strategies over backtest predictions.

    Takes BacktestResults and closing odds, applies half-point slippage
    per D-07, and runs both strategies side by side per D-08.

    Usage::

        sim = BettingSimulator()
        results = sim.simulate(backtest_results, closing_odds_df)
        print(results.flat_stake.roi, results.kelly.roi)
    """

    def __init__(
        self,
        config: SimulationConfig | None = None,
        ou_bet_selector: Any | None = None,
        ats_bet_selector: Any | None = None,
    ) -> None:
        self.config = config or SimulationConfig()
        self.logger = get_logger(__name__)
        # LOCKED-2 (BET-01): when a BetSelector is injected, the O/U decision -- eligibility, side,
        # threshold, EV admission, AND sizing -- is owned END-TO-END by it (proof == production).
        # When None, the O/U target is skipped (no inline O/U decision logic remains -- the
        # points-distance admission and the BET-02 `implied+edge` Kelly branch were removed; O/U
        # decisions live ONLY in the BetSelector).
        self.ou_bet_selector = ou_bet_selector
        # D31-04 (Phase 31, plan 31-10): the SPREAD target gains the same injection point, and it
        # is deliberately a SECOND argument rather than a widening of the first. The two selectors
        # carry different per-target fit parameters -- the ATS frozen residual SD is on the
        # home-margin scale and the O/U one is on the total scale -- so one shared selector would
        # be a category error, and a caller wanting both passes the same three-target selector
        # twice by choice rather than by default.
        #
        # WHEN NONE, THE SPREAD TARGET IS STILL GRADED and its Kelly stake is 0 BY DESIGN. It is
        # NOT skipped the way O/U is: the /betting page reads this ledger, and dropping the spread
        # rows would move a published bet POPULATION, which D31-04 rules out. What changes is the
        # published spread Kelly return, which becomes zero, and that is stated in the R11 readout
        # rather than removed quietly.
        self.ats_bet_selector = ats_bet_selector

    # -- Bet side determination -----------------------------------------------

    def _determine_bet_side_wp(self, model_prob: float) -> str | None:
        """Determine WP bet side based on model probability edge.

        Returns "home" if model likes home, "away" if model likes away,
        or None if edge is insufficient.
        """
        threshold = self.config.min_edge_threshold
        if model_prob > 0.5 + threshold:
            return "home"
        if model_prob < 0.5 - threshold:
            return "away"
        return None

    def _determine_bet_side_ats(
        self, model_spread: float, closing_spread: float
    ) -> str | None:
        """Determine ATS bet side based on model vs closing spread.

        If model thinks home wins by more than market (model_spread < closing_spread),
        bet home_cover. If model thinks home wins by less, bet away_cover.
        """
        threshold = self.config.min_edge_threshold
        if model_spread < closing_spread - threshold:
            return "home_cover"
        if model_spread > closing_spread + threshold:
            return "away_cover"
        return None

    def _determine_bet_side_ou(
        self, model_total: float, closing_total: float
    ) -> str | None:
        """Determine O/U bet side based on model vs closing total.

        If model predicts more points than market, bet over.
        If model predicts fewer points, bet under.
        """
        threshold = self.config.min_edge_threshold
        if model_total > closing_total + threshold:
            return "over"
        if model_total < closing_total - threshold:
            return "under"
        return None

    # -- Outcome resolution ---------------------------------------------------

    def _resolve_wp_outcome(self, bet_side: str, actual_home_win: int) -> bool | None:
        """Resolve WP moneyline bet outcome."""
        if bet_side == "home":
            return actual_home_win == 1
        if bet_side == "away":
            return actual_home_win == 0
        return None

    def _resolve_ats_outcome(
        self, bet_side: str, actual_margin: float, slipped_line: float
    ) -> bool | None:
        """Resolve ATS spread bet outcome.

        actual_margin is home_score - away_score.
        slipped_line is the spread from home perspective after slippage.
        Home covers if actual_margin > slipped_line (home beat spread).
        """
        # Push detection
        if abs(actual_margin - slipped_line) < 1e-9:
            return None

        home_covers = actual_margin > slipped_line
        if bet_side == "home_cover":
            return home_covers
        if bet_side == "away_cover":
            return not home_covers
        return None

    def _resolve_ou_outcome(
        self, bet_side: str, actual_total: float, slipped_line: float
    ) -> bool | None:
        """Resolve O/U bet outcome."""
        # Push detection
        if abs(actual_total - slipped_line) < 1e-9:
            return None

        went_over = actual_total > slipped_line
        if bet_side == "over":
            return went_over
        if bet_side == "under":
            return not went_over
        return None

    # -- Odds helpers ---------------------------------------------------------

    def _get_wp_odds(self, bet_side: str, ml_home: float, ml_away: float) -> int:
        """Get American odds for a WP moneyline bet."""
        if bet_side == "home":
            return int(ml_home)
        return int(ml_away)

    # -- O/U routing (LOCKED-2: decisions owned by the BetSelector) ------------

    def _select_ou_decisions(self, merged: pd.DataFrame) -> dict[str, dict[str, Any]]:
        """Build the per-game O/U decision lookup from BetSelector.select() (LOCKED-2, BET-01).

        Passes the merged O/U prediction frame (carrying game_id, season, week, model_total, the
        closing ``total``, and the ``actual`` label) to the injected BetSelector and returns a
        ``{game_id -> selected-record}`` map for the bets the selector SELECTED. The simulator then
        grades ONLY those games, with the calibrated ``calibrated_p_side`` / ``kelly_stake`` /
        ``slipped_line`` / ``outcome`` the selector supplied. The eligibility, side, EV-floor
        threshold, EV admission, and sizing are ALL the selector's -- the simulator adds no inline
        O/U decision logic (proof == production).

        Args:
            merged: The chronologically-sorted, odds-merged O/U prediction frame.

        Returns:
            A dict keyed by game_id of the selector's selected per-bet records (empty when nothing
            was selected). Rows lacking ``model_total`` are dropped before selection.
        """
        if "model_total" not in merged.columns:
            return {}

        candidate_cols = ["game_id", "season", "week", "model_total"]
        candidates = merged[merged["model_total"].notna()].copy()
        if candidates.empty:
            return {}

        # The BetSelector expects ``closing_total`` (and ``actual`` for grading); map from the
        # merged frame's ``total`` / ``actual`` columns without mutating the simulator's row schema.
        candidates["closing_total"] = candidates["total"].astype(float)
        if "actual" in candidates.columns:
            candidates["actual"] = candidates["actual"].astype(float)
        keep = [*candidate_cols, "closing_total"]
        if "actual" in candidates.columns:
            keep.append("actual")

        result = self.ou_bet_selector.select(candidates[keep])
        return {rec["game_id"]: rec for rec in result.selected}

    # -- ATS routing (D31-04: decisions owned by the injected BetSelector) -----

    def _select_ats_decisions(self, merged: pd.DataFrame) -> dict[str, dict[str, Any]]:
        """Build the per-game ATS decision lookup from BetSelector.select() (D31-04, SPEC R10).

        The direct analogue of ``_select_ou_decisions``. Passes the merged ATS prediction frame
        (carrying game_id, season, week, the predicted home MARGIN ``model_spread``, the closing
        ``spread`` and the realized margin ``actual``) to the injected BetSelector, and returns a
        ``{game_id -> selected-record}`` map. The side, the EV admission and the sizing are ALL the
        selector's; the simulator adds no inline spread decision logic.

        THE SIGN HANDLING HAPPENS INSIDE ``ATSStrategy``, NOT HERE. ``model_spread`` and ``spread``
        are BOTH home MARGINS (DEF-31-01, ruled 2026-09-04); the strategy negates only into the two
        LOCKED helpers written in the opposite line convention and negates back out. Converting
        anything here as well would apply a negation twice.

        Rows are TAGGED with their target code, unlike the O/U path: an injected selector may
        register more than one strategy, and an untagged row is ambiguous the moment it does.

        Args:
            merged: The chronologically-sorted, odds-merged ATS prediction frame.

        Returns:
            A dict keyed by game_id of the selector's selected per-bet records (empty when nothing
            was selected). Rows lacking ``model_spread`` are dropped before selection.
        """
        selector = self.ats_bet_selector
        if selector is None or "model_spread" not in merged.columns:
            return {}

        candidates = merged[merged["model_spread"].notna()].copy()
        if candidates.empty:
            return {}

        # Map from the merged frame's ``spread`` / ``actual`` columns onto the names the strategy
        # declares, without mutating the simulator's own row schema.
        candidates["closing_spread"] = candidates["spread"].astype(float)
        candidates["target"] = "ats"
        keep = ["game_id", "season", "week", "model_spread", "closing_spread", "target"]
        if "actual" in candidates.columns:
            candidates["actual"] = candidates["actual"].astype(float)
            keep.append("actual")

        result = selector.select(candidates[keep])
        return {rec["game_id"]: rec for rec in result.selected}

    # -- Main simulation ------------------------------------------------------

    def simulate(
        self,
        backtest_results: Any,  # BacktestResults (loose typing for testability)
        closing_odds_df: pd.DataFrame,
    ) -> SimulationResults:
        """Run betting simulation over backtest predictions.

        For each target in backtest_results.all_predictions:
        1. Merge predictions with closing odds on game_id.
        2. Sort chronologically (season, week).
        3. For each game: determine bet side, apply slippage, resolve outcome.
        4. Track flat-stake and Kelly bankrolls in parallel.

        Args:
            backtest_results: BacktestResults (or mock with all_predictions dict).
            closing_odds_df: DataFrame with game_id, ml_home, ml_away, spread, total.

        Returns:
            SimulationResults with both strategies, bet records, and breakdowns.
        """
        config = self.config

        # Initialize Kelly calculator for Kelly strategy
        kelly_calc = KellyCalculator(
            starting_bankroll=config.starting_bankroll,
            max_bet_pct=config.max_kelly_bet_pct,
            default_kelly_fraction=config.kelly_fraction,
            confidence_threshold=config.min_edge_threshold,
        )

        # Flat-stake bankroll tracking (manual)
        flat_bankroll = config.starting_bankroll
        flat_peak = config.starting_bankroll
        flat_max_dd = 0.0
        flat_total_wagered = 0.0
        flat_net_profit = 0.0
        flat_wins = 0
        flat_losses = 0
        flat_pushes = 0

        # Kelly equity curve and bet tracking
        kelly_equity: list[float] = [config.starting_bankroll]
        flat_equity: list[float] = [config.starting_bankroll]
        bet_game_ids: list[str] = []

        all_bet_records: list[BetRecord] = []

        for target, preds_df in backtest_results.all_predictions.items():
            if preds_df.empty:
                continue

            # Merge predictions with closing odds (skip if already merged via CLV)
            odds_cols = {"ml_home", "ml_away", "spread", "total"}
            if odds_cols.issubset(preds_df.columns):
                merged = preds_df[preds_df["has_closing_odds"]].copy()
            else:
                merged = preds_df.merge(closing_odds_df, on="game_id", how="inner")
            if merged.empty:
                self.logger.warning(
                    "No matching closing odds for target", target=target
                )
                continue

            # Extract week from game_id (format: YYYY_WXX_AWAY@HOME)
            if "week" not in merged.columns:
                merged["week"] = (
                    merged["game_id"].str.extract(r"_W(\d+)_")[0].astype(int)
                )

            # Sort chronologically
            merged = merged.sort_values(["season", "week"]).reset_index(drop=True)

            # LOCKED-2 (BET-01): when a BetSelector is injected, route the ENTIRE O/U decision --
            # eligibility, side, EV-floor threshold, EV admission, and sizing -- through
            # BetSelector.select() so proof == production. The simulator then grades ONLY the bets
            # it returned, with the calibrated p_side it supplied for Kelly. When no BetSelector is
            # injected (the Phase-26 DIAGNOSIS consumers that grade every game at
            # min_edge_threshold=0.0), the simulator retains the LOCKED side/grading path for
            # backward-compatible O/U GRADING -- but the BET-02 points-distance Kelly bug is removed
            # in BOTH paths (the legacy O/U path no longer sizes Kelly off `implied + edge`). WP/ATS
            # are unaffected (out of scope).
            ou_decisions: dict[str, dict[str, Any]] = {}
            if target == "ou" and self.ou_bet_selector is not None:
                ou_decisions = self._select_ou_decisions(merged)
                if not ou_decisions:
                    continue

            # D31-04: the same routing for the SPREAD target. Unlike O/U, an empty decision set
            # does NOT skip the target -- the legacy branch below still grades every spread bet at
            # a flat stake, because /betting's published spread bet POPULATION must not move.
            ats_decisions: dict[str, dict[str, Any]] = {}
            if target == "ats" and self.ats_bet_selector is not None:
                ats_decisions = self._select_ats_decisions(merged)

            for _, row in merged.iterrows():
                game_id = row["game_id"]
                season = int(row["season"])
                week = int(row["week"])

                bet_side: str | None = None
                model_value: float = 0.0
                market_value: float = 0.0
                edge: float = 0.0
                slipped_line: float | None = None
                odds: int = STANDARD_VIG_ODDS
                outcome: bool | None = None
                # The selector's record for this game, when one owns the decision. It is the ONE
                # signal the sizing block below reads to tell an owned bet from a legacy one, so
                # the two questions "was this decided by the selector?" and "which target is this?"
                # cannot drift apart.
                selector_decision: dict[str, Any] | None = None

                if target == "wp":
                    model_prob = float(row["model_prob"])
                    bet_side = self._determine_bet_side_wp(model_prob)
                    if bet_side is None:
                        continue

                    ml_home = float(row["ml_home"])
                    ml_away = float(row["ml_away"])
                    odds = self._get_wp_odds(bet_side, ml_home, ml_away)

                    # Market implied probability for the side we're betting
                    market_prob = moneyline_to_probability(odds)
                    model_value = (
                        model_prob if bet_side == "home" else (1.0 - model_prob)
                    )
                    market_value = market_prob
                    edge = model_value - market_value

                    # Resolve outcome
                    actual = int(row["actual"])
                    outcome = self._resolve_wp_outcome(bet_side, actual)

                elif target == "ats" and ats_decisions.get(game_id) is not None:
                    # D31-04 routing: the ENTIRE spread decision is owned by the injected
                    # BetSelector -- side, EV admission and sizing -- and the calibrated P(cover)
                    # it supplied is what Kelly consumes. The strategy already handed the two
                    # line-convention helpers their arguments and converted back (DEF-31-01), so
                    # nothing is converted again here.
                    selector_decision = ats_decisions[game_id]

                    bet_side = selector_decision["bet_side"]
                    model_value = selector_decision[
                        "calibrated_p_side"
                    ]  # the calibrated P(cover) for the side bet
                    market_value = selector_decision["closing_spread"]
                    # WR-06 (as for O/U): on the selector path `edge` carries the per-bet EV, NOT
                    # the points/probability edge BetRecord.edge documents. Same warning applies --
                    # if these rows ever flow into `betting_bets` / the edge-bucket UI, the EV
                    # scale will misclassify them.
                    edge = selector_decision["per_bet_ev"]
                    slipped_line = selector_decision["slipped_line"]
                    # The price the selector actually judged and sized the bet at, so the payout
                    # below cannot diverge from the stake above it.
                    odds = int(selector_decision["selected_odds"])
                    outcome = selector_decision["outcome"]

                elif target == "ats":
                    # Legacy path (no BetSelector injected): grade the spread bet so /betting's
                    # published bet POPULATION does not move (D31-04). The side and grading here
                    # are the PRE-EXISTING convention and are deliberately left alone; see the
                    # note in `backtest/ats_ev_chain.py`'s module docstring on the two sign
                    # conventions, and `ATSStrategy`, which converts at both seams.
                    if "model_spread" not in row.index:
                        continue
                    model_spread = float(row["model_spread"])
                    closing_spread = float(row["spread"])
                    bet_side = self._determine_bet_side_ats(
                        model_spread, closing_spread
                    )
                    if bet_side is None:
                        continue

                    slipped_line = apply_slippage_spread(
                        closing_spread, bet_side, config.slippage_points
                    )

                    model_value = model_spread
                    market_value = closing_spread
                    edge = abs(model_spread - closing_spread)
                    odds = config.standard_vig_odds

                    actual_margin = float(row["actual"])
                    outcome = self._resolve_ats_outcome(
                        bet_side, actual_margin, slipped_line
                    )

                elif target == "ou" and self.ou_bet_selector is not None:
                    # LOCKED-2 monetization path: the ENTIRE O/U decision is owned by the
                    # BetSelector (eligibility, side, threshold, EV admission, sizing). Grade ONLY a
                    # game the selector returned; the calibrated p_side it supplied drives Kelly (the
                    # BET-02 fix -- NO inline `implied + edge` for O/U). The LOCKED grading helpers
                    # (apply_slippage_total / _resolve_ou_outcome) are reused INSIDE the selector,
                    # unchanged (D-18).
                    decision = ou_decisions.get(game_id)
                    if decision is None:
                        continue  # not selected by the BetSelector -> no O/U bet (proof==production)
                    selector_decision = decision

                    bet_side = decision["bet_side"]
                    model_value = decision[
                        "calibrated_p_side"
                    ]  # the calibrated P(side) for Kelly
                    market_value = decision["closing_total"]
                    # WR-06: for the O/U+selector path `edge` carries the per-bet EV, NOT the
                    # points/probability edge BetRecord.edge documents. This is the selector's
                    # native decision value and is NOT load-bearing here: the selector's consumer is
                    # the `ou_bet_list` cache table (which stores per_bet_ev in its own column), not
                    # `betting_bets`. WARNING -- if O/U-via-selector rows ever flow into
                    # `betting_bets` / the edge-bucket UI (api/cache._compute_confidence, D-15/16/20),
                    # the EV scale will misclassify them; store the points/prob edge or make those
                    # consumers target-aware before that happens.
                    edge = decision[
                        "per_bet_ev"
                    ]  # per-bet EV (NOT a points distance -- see WR-06 note above)
                    slipped_line = decision["slipped_line"]
                    odds = config.standard_vig_odds
                    outcome = decision["outcome"]

                elif target == "ou":
                    # Legacy DIAGNOSIS path (no BetSelector injected): grade O/U with the LOCKED
                    # side/slippage/outcome convention (D-18) for backward-compatible grading (the
                    # Phase-26 divergence harness reads the flat-stake win-rate at
                    # min_edge_threshold=0.0). The BET-02 bug is removed: Kelly is NOT sized off the
                    # points distance here (see the sizing block below).
                    if "model_total" not in row.index:
                        continue
                    model_total = float(row["model_total"])
                    closing_total = float(row["total"])
                    bet_side = self._determine_bet_side_ou(model_total, closing_total)
                    if bet_side is None:
                        continue

                    slipped_line = apply_slippage_total(
                        closing_total, bet_side, config.slippage_points
                    )

                    model_value = model_total
                    market_value = closing_total
                    edge = abs(model_total - closing_total)
                    odds = config.standard_vig_odds

                    actual_total = float(row["actual"])
                    outcome = self._resolve_ou_outcome(
                        bet_side, actual_total, slipped_line
                    )

                else:
                    continue

                # -- Flat-stake sizing --
                flat_bet = config.flat_stake_amount
                payout_flat = _calculate_payout(flat_bet, odds, outcome)

                # -- Kelly sizing --
                #
                # THE ONE RULE, FOR ALL THREE TARGETS: the number handed to Kelly is a calibrated
                # PROBABILITY or there is no stake. There is no third case, and in particular no
                # branch derives an "equivalent model probability" from a points distance -- that
                # expression is what SPEC R10 removes, and Task 3's class guard is what keeps it
                # removed for any target added later.
                if selector_decision is not None:
                    # A selector owns this decision end to end (LOCKED-2 for O/U, D31-04 for the
                    # spread target). The stake already went through the LOCKED-order Kelly + cap
                    # pipeline on the calibrated P(side); use it directly.
                    kelly_bet = selector_decision["kelly_stake"]
                elif target in ("ou", "ats"):
                    # Legacy paths, BOTH zeroed BY DESIGN. Without a calibrated P(side) there is no
                    # honest Kelly probability for a line target, and the quantity these branches
                    # used to substitute was a POINTS DISTANCE added to an implied probability. For
                    # O/U that removal is Phase 27's (the Phase-26 divergence harness reads the
                    # flat-stake win-rate, which is unaffected). For the spread target it is
                    # D31-04's: measured in the live cache, `implied + abs(model_spread -
                    # closing_spread)` crossed 1.0 for every strong signal, so Kelly staked 348
                    # weak signals and refused 725 strong ones, and the published spread Kelly
                    # return was computed over that inverted selection. So:
                    # the spread Kelly stake here is 0 by design.
                    # The bet is still placed and graded at a flat stake, so /betting's published
                    # bet POPULATION does not move; what moves is its spread Kelly return, which
                    # becomes zero and is stated in the R11 readout rather than removed quietly.
                    kelly_bet = 0.0
                else:
                    # WP, unchanged and deliberately so: `model_value` is ALREADY the side-correct
                    # probability, computed a few lines above from the deployed calibrated model,
                    # so Kelly consumes a genuine probability for the side actually bet. This is
                    # measured (every winner bet's model_value is in (0, 1]), not assumed.
                    kelly_result = kelly_calc.calculate_optimal_bet_size(
                        model_prob=model_value,
                        market_odds=odds,
                        mode=KellyMode.FRACTIONAL,
                    )
                    kelly_bet = kelly_result.recommended_bet
                payout_kelly = _calculate_payout(kelly_bet, odds, outcome)

                # -- Update flat-stake bankroll --
                flat_bankroll += payout_flat
                flat_total_wagered += flat_bet
                flat_net_profit += payout_flat
                if outcome is True:
                    flat_wins += 1
                elif outcome is False:
                    flat_losses += 1
                else:
                    flat_pushes += 1

                flat_peak = max(flat_peak, flat_bankroll)
                dd = flat_peak - flat_bankroll
                flat_max_dd = max(flat_max_dd, dd)

                # -- Update Kelly bankroll --
                if kelly_bet > 0 and outcome is not None:
                    kelly_calc.update_bankroll(kelly_bet, outcome, odds=odds)
                elif kelly_bet > 0 and outcome is None:
                    # Push: no change for Kelly (stake returned)
                    pass

                # -- Record equity and bet --
                flat_equity.append(flat_bankroll)
                kelly_equity.append(kelly_calc.bankroll_state.current_balance)
                bet_game_ids.append(game_id)

                all_bet_records.append(
                    BetRecord(
                        game_id=game_id,
                        season=season,
                        week=week,
                        target=target,
                        bet_side=bet_side,
                        model_value=model_value,
                        market_value=market_value,
                        edge=edge,
                        slipped_line=slipped_line,
                        odds=odds,
                        flat_stake=flat_bet,
                        kelly_stake=kelly_bet,
                        outcome=outcome,
                        payout_flat=payout_flat,
                        payout_kelly=payout_kelly,
                    )
                )

        # -- Build StrategyResult for flat-stake --
        flat_total_bets = flat_wins + flat_losses + flat_pushes
        flat_result = StrategyResult(
            strategy_name="flat_stake",
            total_bets=flat_total_bets,
            winning_bets=flat_wins,
            losing_bets=flat_losses,
            push_bets=flat_pushes,
            win_rate=flat_wins / flat_total_bets if flat_total_bets > 0 else 0.0,
            total_wagered=flat_total_wagered,
            net_profit=flat_net_profit,
            roi=flat_net_profit / flat_total_wagered if flat_total_wagered > 0 else 0.0,
            final_bankroll=flat_bankroll,
            max_drawdown=flat_max_dd,
            max_drawdown_pct=flat_max_dd / flat_peak if flat_peak > 0 else 0.0,
            equity_curve=flat_equity,
            bet_timestamps=bet_game_ids,
        )

        # -- Build StrategyResult for Kelly --
        kelly_state = kelly_calc.bankroll_state

        kelly_total_bets = kelly_state.total_bets
        kelly_result_obj = StrategyResult(
            strategy_name="kelly",
            total_bets=kelly_total_bets,
            winning_bets=kelly_state.winning_bets,
            losing_bets=kelly_state.losing_bets,
            push_bets=flat_pushes,  # Kelly doesn't track pushes separately (push = no update)
            win_rate=kelly_state.win_rate,
            total_wagered=kelly_state.total_wagered,
            net_profit=kelly_state.net_profit,
            roi=kelly_state.roi,
            final_bankroll=kelly_state.current_balance,
            max_drawdown=kelly_state.max_drawdown,
            max_drawdown_pct=(
                kelly_state.max_drawdown / kelly_state.peak_balance
                if kelly_state.peak_balance > 0
                else 0.0
            ),
            equity_curve=kelly_equity,
            bet_timestamps=bet_game_ids,
        )

        # -- Build breakdowns --
        by_target = self._build_target_breakdown(all_bet_records)
        by_season = self._build_season_breakdown(all_bet_records)

        self.logger.info(
            "Simulation complete",
            total_bets=len(all_bet_records),
            flat_roi=flat_result.roi,
            kelly_roi=kelly_result_obj.roi,
        )

        return SimulationResults(
            config=self.config,
            flat_stake=flat_result,
            kelly=kelly_result_obj,
            bet_records=all_bet_records,
            by_target=by_target,
            by_season=by_season,
        )

    # -- Breakdown builders ---------------------------------------------------

    @staticmethod
    def _build_target_breakdown(
        records: list[BetRecord],
    ) -> dict[str, dict[str, Any]]:
        """Build per-target breakdown with n_bets, win_rate, roi per strategy."""
        breakdown: dict[str, dict[str, Any]] = {}
        targets = {r.target for r in records}

        for target in sorted(targets):
            target_records = [r for r in records if r.target == target]
            n_bets = len(target_records)
            wins = sum(1 for r in target_records if r.outcome is True)
            losses = sum(1 for r in target_records if r.outcome is False)
            pushes = sum(1 for r in target_records if r.outcome is None)

            flat_wagered = sum(r.flat_stake for r in target_records)
            flat_profit = sum(r.payout_flat for r in target_records)
            kelly_wagered = sum(r.kelly_stake for r in target_records)
            kelly_profit = sum(r.payout_kelly for r in target_records)

            breakdown[target] = {
                "n_bets": n_bets,
                "wins": wins,
                "losses": losses,
                "pushes": pushes,
                "win_rate": wins / n_bets if n_bets > 0 else 0.0,
                "flat_roi": flat_profit / flat_wagered if flat_wagered > 0 else 0.0,
                "kelly_roi": kelly_profit / kelly_wagered if kelly_wagered > 0 else 0.0,
            }

        return breakdown

    @staticmethod
    def _build_season_breakdown(
        records: list[BetRecord],
    ) -> dict[int, dict[str, Any]]:
        """Build per-season breakdown with n_bets, win_rate, roi per strategy."""
        breakdown: dict[int, dict[str, Any]] = {}
        seasons = {r.season for r in records}

        for season in sorted(seasons):
            season_records = [r for r in records if r.season == season]
            n_bets = len(season_records)
            wins = sum(1 for r in season_records if r.outcome is True)
            losses = sum(1 for r in season_records if r.outcome is False)
            pushes = sum(1 for r in season_records if r.outcome is None)

            flat_wagered = sum(r.flat_stake for r in season_records)
            flat_profit = sum(r.payout_flat for r in season_records)
            kelly_wagered = sum(r.kelly_stake for r in season_records)
            kelly_profit = sum(r.payout_kelly for r in season_records)

            breakdown[season] = {
                "n_bets": n_bets,
                "wins": wins,
                "losses": losses,
                "pushes": pushes,
                "win_rate": wins / n_bets if n_bets > 0 else 0.0,
                "flat_roi": flat_profit / flat_wagered if flat_wagered > 0 else 0.0,
                "kelly_roi": kelly_profit / kelly_wagered if kelly_wagered > 0 else 0.0,
            }

        return breakdown
