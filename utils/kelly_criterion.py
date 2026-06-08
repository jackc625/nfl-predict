"""
Kelly Criterion implementation for optimal bet sizing with risk management.

This module provides comprehensive Kelly Criterion calculations with
advanced risk management features including drawdown limits, confidence
adjustments, and bankroll management.
"""

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from utils.probability_utils import moneyline_to_probability

# ---------------------------------------------------------------------------
# Honest-sizing exposure controls (Phase 27, BET-03; D27-09/10/11).
#
# These module constants are PRE-REGISTERED and FROZEN. They name the sizing
# discipline the BetSelector (Plan 03) consumes. The helpers below are ADDITIVE:
# the existing KellyCalculator / calculate_optimal_bet_size /
# calculate_simultaneous_kelly interfaces are NOT changed.
# ---------------------------------------------------------------------------

# 1 unit = 1% of bankroll (D27-09).
UNIT_PCT_OF_BANKROLL = 0.01

# Per-week total-exposure cap = 10% of bankroll, pro-rata scaled (D27-10, NEW).
WEEKLY_CAP_PCT = 0.10

# The existing 5%-of-bankroll per-bet cap (KellyCalculator.max_bet_pct),
# named here for the LOCKED cap-order convention (D27-09; rarely-binding net).
PER_BET_CAP_PCT = 0.05

# Same-week SAME-SIDE de-weight rule, FROZEN (D27-11; review tightening #2).
# Documented as a string so the chosen formula is auditable as a constant.
SAME_SIDE_DEWEIGHT = "stake / sqrt(group_size)"

# Full-Kelly is EXCLUDED (ruin risk). The half-Kelly ceiling is a hard clamp
# applied at the END of the pipeline (D27-09).
HALF_KELLY_CEILING = 0.5

# The LOCKED cap order (review tightening #2). The single orchestration helper
# apply_sizing_pipeline applies these four steps in EXACTLY this order; the
# BetSelector calls that helper so the order is enforced in one place.
CAP_ORDER = (
    "kelly_stake",
    "per_bet_5pct_cap",
    "same_side_deweight",
    "weekly_10pct_cap",
)


def _validate_bankroll(bankroll: float) -> None:
    """Raise a named ValueError when the bankroll is not strictly positive.

    Edge-case guard (review tightening #8): an invalid bankroll (<= 0) is a
    hard error, never a silent divide-by-zero or a nonsense unit/cap.
    """
    if not bankroll > 0:
        raise ValueError(f"bankroll must be > 0, got {bankroll!r}")


def _validate_stake(stake: float) -> float:
    """Coerce + validate a single stake, raising a named ValueError on NaN.

    Edge-case guard (review tightening #8): a NaN stake is a hard error rather
    than silently propagating through the pro-rata math.
    """
    value = float(stake)
    if math.isnan(value):
        raise ValueError("stake must not be NaN")
    return value


def unit_size(bankroll: float) -> float:
    """Return the size of one betting unit (1% of bankroll, D27-09).

    Args:
        bankroll: Current bankroll (must be strictly positive).

    Returns:
        The dollar value of one unit (bankroll * UNIT_PCT_OF_BANKROLL).
    """
    _validate_bankroll(bankroll)
    return bankroll * UNIT_PCT_OF_BANKROLL


def clamp_to_half_kelly(fraction: float) -> float:
    """Hard-clamp a Kelly fraction to the half-Kelly ceiling (D27-09).

    Full-Kelly is excluded; this clamp is applied at the END so the effective
    fraction never exceeds HALF_KELLY_CEILING (0.5). A fraction already at or
    below the ceiling is returned unchanged.
    """
    return min(float(fraction), HALF_KELLY_CEILING)


def apply_same_side_deweight(bets: list[dict]) -> list[dict]:
    """De-weight SAME-WEEK SAME-SIDE concentration (D27-11; review #2).

    Uses the FROZEN formula ``stake / sqrt(group_size)``. Bets are grouped by a
    CASE-INSENSITIVELY normalized ``bet_side`` (lower-cased): all UNDERs are one
    group; a high-total OVER and a normal OVER are one "over" group (overs are
    NOT split by ``totals_regime``). Each bet's de-weighted stake is
    ``original_stake / sqrt(len(group))``.

    The covariance / joint-Kelly path is REJECTED (D27-11: overfit-prone on
    limited data, weather features degenerate). There is deliberately NO
    ``correlation_matrix`` argument -- this is SIMPLE same-side grouping only.

    Args:
        bets: A week's bets; each dict carries ``bet_side`` and ``stake``.

    Returns:
        Per-bet records (input order preserved) carrying metadata:
        ``original_stake``, ``deweighted_stake``, ``scale_factor``
        (where ``scale_factor = 1 / sqrt(group_size)``).

    Raises:
        ValueError: if any stake is NaN (review tightening #8).
    """
    if not bets:
        return []

    # Count group sizes by case-insensitively normalized side.
    group_sizes: dict[str, int] = {}
    for bet in bets:
        side = str(bet["bet_side"]).strip().lower()
        group_sizes[side] = group_sizes.get(side, 0) + 1

    records: list[dict] = []
    for bet in bets:
        side = str(bet["bet_side"]).strip().lower()
        original = _validate_stake(bet["stake"])
        factor = 1.0 / math.sqrt(group_sizes[side])
        records.append(
            {
                "bet_side": side,
                "original_stake": original,
                "deweighted_stake": original * factor,
                "scale_factor": factor,
            }
        )
    return records


def apply_weekly_exposure_cap(
    stakes: list[float],
    bankroll: float,
    weekly_cap_pct: float = WEEKLY_CAP_PCT,
) -> list[dict]:
    """Pro-rata scale a week's stakes to the 10% total-exposure cap (D27-10).

    Copies the EXACT pro-rata shape from ``calculate_simultaneous_kelly`` (lines
    531-539): ``max_total = bankroll * weekly_cap_pct``; if ``sum(stakes) <=
    max_total`` the stakes are returned UNCHANGED (scale_factor 1.0); otherwise
    ``scale = max_total / sum(stakes)`` scales every stake. Ratios are preserved
    and NO bets are dropped (D27-10).

    Because this runs on the ALREADY-de-weighted stakes (the LOCKED cap order),
    any room freed by de-weighting is naturally reflected -- the smaller summed
    input divides under the cap (review #2 Gemini interaction-order note).

    Args:
        stakes: The (already-de-weighted) per-bet stakes for one week.
        bankroll: Current bankroll (must be strictly positive).
        weekly_cap_pct: Fraction of bankroll allowed across the week.

    Returns:
        Per-bet records (input order preserved) carrying metadata:
        ``deweighted_stake`` (the input), ``weekly_scaled_stake``,
        ``scale_factor``.

    Raises:
        ValueError: if bankroll <= 0 or any stake is NaN (review #8).
    """
    _validate_bankroll(bankroll)
    if not stakes:
        return []

    validated = [_validate_stake(s) for s in stakes]
    total = sum(validated)
    max_total = bankroll * weekly_cap_pct

    # Guard the zero-sum (and the under-cap no-op) before dividing.
    if total <= max_total or total == 0:
        scale = 1.0
    else:
        scale = max_total / total

    return [
        {
            "deweighted_stake": stake,
            "weekly_scaled_stake": stake * scale,
            "scale_factor": scale,
        }
        for stake in validated
    ]


def apply_sizing_pipeline(bets: list[dict], bankroll: float) -> list[dict]:
    """Apply the four sizing steps in the LOCKED CAP_ORDER (review #2).

    Order (CAP_ORDER): calibrated Kelly stake -> 5% per-bet cap -> same-side
    de-weight -> 10% weekly cap. This single orchestration helper is what the
    BetSelector (Plan 03) consumes, so the order is enforced in ONE place; the
    BetSelector does NOT re-order the steps.

    Each input bet carries a ``bet_side`` and a ``stake`` (the calibrated Kelly
    stake -- step 1, already computed upstream). The half-Kelly ceiling is a
    hard clamp on the fraction; here stakes arrive as dollar amounts, so the
    fraction clamp is enforced via ``clamp_to_half_kelly`` on the implied
    bankroll fraction before the per-bet cap.

    Args:
        bets: A week's bets; each dict carries ``bet_side`` and ``stake``.
        bankroll: Current bankroll (must be strictly positive).

    Returns:
        Per-bet records (input order preserved) carrying the full metadata set:
        ``original_stake``, ``deweighted_stake``, ``weekly_scaled_stake``,
        ``scale_factor`` (the COMBINED de-weight x weekly factor).

    Raises:
        ValueError: if bankroll <= 0 or any stake is NaN (review #8).
    """
    _validate_bankroll(bankroll)
    if not bets:
        return []

    per_bet_cap = bankroll * PER_BET_CAP_PCT

    # Step 1 (kelly_stake) is the provided stake. Validate + clamp the implied
    # fraction to the half-Kelly ceiling, then apply Step 2 (5% per-bet cap).
    capped_bets: list[dict] = []
    for bet in bets:
        original = _validate_stake(bet["stake"])
        clamped_fraction = clamp_to_half_kelly(original / bankroll)
        clamped_stake = clamped_fraction * bankroll
        capped_stake = min(clamped_stake, per_bet_cap)
        capped_bets.append({"bet_side": bet["bet_side"], "stake": capped_stake})

    # Step 3 (same_side_deweight): operate on the per-bet-capped stakes.
    deweighted = apply_same_side_deweight(capped_bets)

    # Step 4 (weekly_10pct_cap): operate on the already-de-weighted stakes.
    weekly = apply_weekly_exposure_cap(
        [rec["deweighted_stake"] for rec in deweighted], bankroll
    )

    # Merge: original_stake is the pre-cap Kelly stake; scale_factor is the
    # COMBINED de-weight x weekly factor relative to that original.
    records: list[dict] = []
    for bet, dw, wk in zip(bets, deweighted, weekly, strict=True):
        original = float(bet["stake"])
        final_stake = wk["weekly_scaled_stake"]
        combined = final_stake / original if original != 0 else 1.0
        records.append(
            {
                "bet_side": dw["bet_side"],
                "original_stake": original,
                "deweighted_stake": dw["deweighted_stake"],
                "weekly_scaled_stake": final_stake,
                "scale_factor": combined,
            }
        )
    return records


class KellyMode(Enum):
    """Kelly sizing modes."""

    FULL = "full"
    FRACTIONAL = "fractional"
    CONFIDENCE_ADJUSTED = "confidence_adjusted"
    CONSERVATIVE = "conservative"


@dataclass
class BankrollState:
    """Container for current bankroll state and metrics."""

    current_balance: float
    starting_balance: float
    peak_balance: float
    current_drawdown: float
    max_drawdown: float
    total_bets: int
    winning_bets: int
    losing_bets: int
    total_wagered: float
    net_profit: float
    roi: float

    @property
    def win_rate(self) -> float:
        """Calculate current win rate."""
        if self.total_bets == 0:
            return 0.0
        return self.winning_bets / self.total_bets

    @property
    def drawdown_pct(self) -> float:
        """Calculate current drawdown as percentage."""
        if self.peak_balance <= 0:
            return 0.0
        return self.current_drawdown / self.peak_balance


@dataclass
class KellyResult:
    """Container for Kelly sizing calculation results."""

    recommended_bet: float
    kelly_fraction: float
    confidence_adjustment: float
    risk_adjustment: float
    final_fraction: float
    reasoning: str
    bankroll_pct: float
    units: float


class KellyCalculator:
    """
    Advanced Kelly Criterion calculator with risk management features.

    Features:
    - Full and fractional Kelly calculations
    - Confidence-based adjustments
    - Drawdown limits and bankroll protection
    - Portfolio-level risk management
    - Unit sizing with configurable unit values
    """

    def __init__(
        self,
        starting_bankroll: float = 10000.0,
        max_bet_pct: float = 0.05,
        max_drawdown_pct: float = 0.20,
        base_unit_size: float = 100.0,
        default_kelly_fraction: float = 0.25,
        confidence_threshold: float = 0.02,
    ):
        """
        Initialize Kelly Calculator.

        Args:
            starting_bankroll: Initial bankroll amount
            max_bet_pct: Maximum bet as percentage of bankroll
            max_drawdown_pct: Maximum allowed drawdown before reducing sizing
            base_unit_size: Base unit size for bet recommendations
            default_kelly_fraction: Default fractional Kelly multiplier
            confidence_threshold: Minimum edge required for betting
        """
        self.starting_bankroll = starting_bankroll
        self.max_bet_pct = max_bet_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.base_unit_size = base_unit_size
        self.default_kelly_fraction = default_kelly_fraction
        self.confidence_threshold = confidence_threshold

        # Initialize bankroll state
        self.bankroll_state = BankrollState(
            current_balance=starting_bankroll,
            starting_balance=starting_bankroll,
            peak_balance=starting_bankroll,
            current_drawdown=0.0,
            max_drawdown=0.0,
            total_bets=0,
            winning_bets=0,
            losing_bets=0,
            total_wagered=0.0,
            net_profit=0.0,
            roi=0.0,
        )

        # Risk adjustment parameters
        self.drawdown_scaling = {
            0.05: 1.0,  # 0-5% drawdown: no adjustment
            0.10: 0.8,  # 5-10% drawdown: 20% reduction
            0.15: 0.6,  # 10-15% drawdown: 40% reduction
            0.20: 0.4,  # 15-20% drawdown: 60% reduction
        }

    def calculate_kelly_fraction(
        self, win_probability: float, odds: int, mode: KellyMode = KellyMode.FRACTIONAL
    ) -> float:
        """
        Calculate raw Kelly fraction for a bet.

        Args:
            win_probability: Probability of winning the bet (0-1)
            odds: American odds for the bet
            mode: Kelly calculation mode

        Returns:
            Kelly fraction (percentage of bankroll to bet)
        """
        if win_probability <= 0 or win_probability >= 1:
            return 0.0

        # Convert American odds to decimal odds
        decimal_odds = odds / 100 + 1 if odds > 0 else 100 / abs(odds) + 1

        # Kelly formula: f* = (bp - q) / b
        # where b = decimal_odds - 1, p = win_prob, q = lose_prob
        b = decimal_odds - 1  # Net odds
        p = win_probability
        q = 1 - p

        if b <= 0:
            return 0.0

        kelly_fraction = (b * p - q) / b

        # Apply mode-specific adjustments
        if mode == KellyMode.FULL:
            return max(kelly_fraction, 0.0)
        if mode == KellyMode.FRACTIONAL:
            return max(kelly_fraction * self.default_kelly_fraction, 0.0)
        if mode == KellyMode.CONFIDENCE_ADJUSTED:
            confidence = abs(p - 0.5) * 2  # Scale 0-1
            adjustment = 0.5 + (confidence * 0.5)  # Scale 0.5-1.0
            return max(kelly_fraction * self.default_kelly_fraction * adjustment, 0.0)
        if mode == KellyMode.CONSERVATIVE:
            return max(kelly_fraction * 0.1, 0.0)  # Very conservative 10% Kelly
        return max(kelly_fraction * self.default_kelly_fraction, 0.0)

    def calculate_confidence_adjustment(
        self, model_prob: float, market_prob: float, sample_size: int | None = None
    ) -> float:
        """
        Calculate confidence adjustment based on edge size and sample size.

        Args:
            model_prob: Model's probability estimate
            market_prob: Market's implied probability
            sample_size: Sample size for model estimation

        Returns:
            Confidence adjustment factor (0-1)
        """
        edge = abs(model_prob - market_prob)

        # Base confidence from edge size
        if edge < 0.01:
            base_confidence = 0.1
        elif edge < 0.02:
            base_confidence = 0.3
        elif edge < 0.05:
            base_confidence = 0.6
        elif edge < 0.10:
            base_confidence = 0.8
        else:
            base_confidence = 1.0

        # Adjust for sample size if provided
        if sample_size is not None:
            if sample_size < 100:
                sample_adjustment = 0.5
            elif sample_size < 500:
                sample_adjustment = 0.7
            elif sample_size < 1000:
                sample_adjustment = 0.9
            else:
                sample_adjustment = 1.0

            base_confidence *= sample_adjustment

        return min(base_confidence, 1.0)

    def calculate_risk_adjustment(self) -> float:
        """
        Calculate risk adjustment based on current bankroll state.

        Returns:
            Risk adjustment factor (0-1)
        """
        current_dd_pct = self.bankroll_state.drawdown_pct

        # Find appropriate scaling factor
        risk_adjustment = 1.0
        for dd_threshold, scaling in sorted(self.drawdown_scaling.items()):
            if current_dd_pct >= dd_threshold:
                risk_adjustment = scaling

        # Additional adjustment for win rate
        if self.bankroll_state.total_bets > 10:
            if self.bankroll_state.win_rate < 0.4:
                risk_adjustment *= 0.7
            elif self.bankroll_state.win_rate < 0.45:
                risk_adjustment *= 0.85

        return risk_adjustment

    def calculate_optimal_bet_size(
        self,
        model_prob: float,
        market_odds: int,
        market_prob: float | None = None,
        confidence_level: float | None = None,
        mode: KellyMode = KellyMode.FRACTIONAL,
    ) -> KellyResult:
        """
        Calculate optimal bet size with all risk adjustments applied.

        Args:
            model_prob: Model's probability estimate
            market_odds: American odds
            market_prob: Market's probability (will calculate if not provided)
            confidence_level: Override confidence adjustment
            mode: Kelly calculation mode

        Returns:
            KellyResult with detailed sizing information
        """
        if market_prob is None:
            market_prob = moneyline_to_probability(market_odds)

        edge = model_prob - market_prob

        # Early exit for negative or insufficient edge
        if edge <= self.confidence_threshold:
            return KellyResult(
                recommended_bet=0.0,
                kelly_fraction=0.0,
                confidence_adjustment=0.0,
                risk_adjustment=1.0,
                final_fraction=0.0,
                reasoning="Edge too small or negative",
                bankroll_pct=0.0,
                units=0.0,
            )

        # Calculate raw Kelly fraction
        raw_kelly = self.calculate_kelly_fraction(model_prob, market_odds, mode)

        # Calculate confidence adjustment
        if confidence_level is not None:
            confidence_adj = confidence_level
        else:
            confidence_adj = self.calculate_confidence_adjustment(
                model_prob, market_prob
            )

        # Calculate risk adjustment based on bankroll state
        risk_adj = self.calculate_risk_adjustment()

        # Apply all adjustments
        adjusted_kelly = raw_kelly * confidence_adj * risk_adj

        # Apply maximum bet constraint
        max_fraction = min(self.max_bet_pct, adjusted_kelly)

        # Calculate actual bet size
        bet_size = max_fraction * self.bankroll_state.current_balance

        # Calculate units
        units = bet_size / self.base_unit_size if self.base_unit_size > 0 else 0.0

        # Generate reasoning
        reasoning_parts = []
        if raw_kelly > 0:
            reasoning_parts.append(f"Raw Kelly: {raw_kelly:.1%}")
        if confidence_adj < 1.0:
            reasoning_parts.append(f"Confidence adj: {confidence_adj:.1%}")
        if risk_adj < 1.0:
            reasoning_parts.append(f"Risk adj: {risk_adj:.1%}")
        if max_fraction < adjusted_kelly:
            reasoning_parts.append(f"Capped at {self.max_bet_pct:.1%}")

        reasoning = (
            " | ".join(reasoning_parts) if reasoning_parts else "Standard sizing"
        )

        return KellyResult(
            recommended_bet=bet_size,
            kelly_fraction=raw_kelly,
            confidence_adjustment=confidence_adj,
            risk_adjustment=risk_adj,
            final_fraction=max_fraction,
            reasoning=reasoning,
            bankroll_pct=max_fraction,
            units=units,
        )

    def update_bankroll(
        self,
        bet_amount: float,
        outcome: bool,
        payout: float | None = None,
        odds: int | None = None,
    ) -> None:
        """
        Update bankroll state after a bet result.

        Args:
            bet_amount: Amount wagered
            outcome: True if won, False if lost
            payout: Total payout received (if won)
            odds: American odds (used to calculate payout if not provided)
        """
        if payout is None and odds is not None:
            if outcome:
                if odds > 0:
                    payout = bet_amount * (1 + odds / 100)
                else:
                    payout = bet_amount * (1 + 100 / abs(odds))
            else:
                payout = 0.0
        elif payout is None:
            payout = bet_amount if outcome else 0.0

        # Calculate profit/loss
        profit = payout - bet_amount

        # Update bankroll
        self.bankroll_state.current_balance += profit
        self.bankroll_state.total_wagered += bet_amount
        self.bankroll_state.net_profit += profit
        self.bankroll_state.total_bets += 1

        if outcome:
            self.bankroll_state.winning_bets += 1
        else:
            self.bankroll_state.losing_bets += 1

        # Update peak balance
        if self.bankroll_state.current_balance > self.bankroll_state.peak_balance:
            self.bankroll_state.peak_balance = self.bankroll_state.current_balance
            self.bankroll_state.current_drawdown = 0.0
        else:
            self.bankroll_state.current_drawdown = (
                self.bankroll_state.peak_balance - self.bankroll_state.current_balance
            )

        # Update max drawdown
        self.bankroll_state.max_drawdown = max(
            self.bankroll_state.max_drawdown, self.bankroll_state.current_drawdown
        )

        # Update ROI
        if self.bankroll_state.total_wagered > 0:
            self.bankroll_state.roi = (
                self.bankroll_state.net_profit / self.bankroll_state.total_wagered
            )

    def simulate_kelly_performance(
        self, scenarios: list[dict], mode: KellyMode = KellyMode.FRACTIONAL
    ) -> dict[str, float]:
        """
        Simulate Kelly performance over multiple betting scenarios.

        Args:
            scenarios: List of dicts with 'model_prob', 'odds', 'outcome' keys
            mode: Kelly calculation mode

        Returns:
            Performance metrics dictionary
        """
        initial_balance = self.bankroll_state.current_balance
        results = []

        for scenario in scenarios:
            model_prob = scenario["model_prob"]
            odds = scenario["odds"]
            outcome = scenario["outcome"]

            kelly_result = self.calculate_optimal_bet_size(model_prob, odds, mode=mode)

            if kelly_result.recommended_bet > 0:
                self.update_bankroll(kelly_result.recommended_bet, outcome, odds=odds)
                results.append(
                    {
                        "bet_size": kelly_result.recommended_bet,
                        "outcome": outcome,
                        "balance": self.bankroll_state.current_balance,
                    }
                )

        final_balance = self.bankroll_state.current_balance
        total_return = (final_balance / initial_balance) - 1

        # Calculate additional metrics
        balance_series = [r["balance"] for r in results]
        if balance_series:
            volatility = np.std([b / initial_balance for b in balance_series])
            max_balance = max(balance_series)
            max_dd = max(0, (max_balance - min(balance_series)) / max_balance)
        else:
            volatility = 0.0
            max_dd = 0.0

        return {
            "total_return": total_return,
            "volatility": volatility,
            "max_drawdown": max_dd,
            "sharpe_ratio": total_return / volatility if volatility > 0 else 0.0,
            "final_balance": final_balance,
            "total_bets": len(results),
            "win_rate": self.bankroll_state.win_rate,
        }

    def get_bankroll_summary(self) -> dict[str, float | int]:
        """Get comprehensive bankroll summary."""
        return {
            "current_balance": self.bankroll_state.current_balance,
            "starting_balance": self.bankroll_state.starting_balance,
            "net_profit": self.bankroll_state.net_profit,
            "total_return": (
                self.bankroll_state.current_balance
                / self.bankroll_state.starting_balance
            )
            - 1,
            "roi": self.bankroll_state.roi,
            "total_bets": self.bankroll_state.total_bets,
            "win_rate": self.bankroll_state.win_rate,
            "current_drawdown": self.bankroll_state.current_drawdown,
            "current_drawdown_pct": self.bankroll_state.drawdown_pct,
            "max_drawdown": self.bankroll_state.max_drawdown,
            "max_drawdown_pct": self.bankroll_state.max_drawdown
            / self.bankroll_state.peak_balance
            if self.bankroll_state.peak_balance > 0
            else 0.0,
            "total_wagered": self.bankroll_state.total_wagered,
            "units_available": self.bankroll_state.current_balance
            / self.base_unit_size,
        }

    def reset_bankroll(self, new_starting_balance: float | None = None) -> None:
        """Reset bankroll to starting state."""
        starting_balance = new_starting_balance or self.starting_bankroll

        self.bankroll_state = BankrollState(
            current_balance=starting_balance,
            starting_balance=starting_balance,
            peak_balance=starting_balance,
            current_drawdown=0.0,
            max_drawdown=0.0,
            total_bets=0,
            winning_bets=0,
            losing_bets=0,
            total_wagered=0.0,
            net_profit=0.0,
            roi=0.0,
        )


def calculate_simultaneous_kelly(
    opportunities: list[dict],
    correlation_matrix: np.ndarray | None = None,
    total_bankroll: float = 10000.0,
    max_total_allocation: float = 0.25,
) -> list[float]:
    """
    Calculate optimal Kelly sizing for simultaneous bets.

    This is a simplified version that adjusts for multiple bets.
    Full covariance optimization would require more complex mathematics.

    Args:
        opportunities: List of dicts with 'model_prob', 'odds', 'edge' keys
        correlation_matrix: Correlation between bet outcomes (optional)
        total_bankroll: Available bankroll
        max_total_allocation: Maximum total allocation across all bets

    Returns:
        List of optimal bet sizes
    """
    if not opportunities:
        return []

    calculator = KellyCalculator(starting_bankroll=total_bankroll)
    individual_results = []

    # Calculate individual Kelly sizes
    for opp in opportunities:
        kelly_result = calculator.calculate_optimal_bet_size(
            model_prob=opp["model_prob"], market_odds=opp["odds"]
        )
        individual_results.append(kelly_result.recommended_bet)

    # Apply simultaneous betting adjustment
    total_individual = sum(individual_results)
    max_total = total_bankroll * max_total_allocation

    if total_individual <= max_total:
        return individual_results
    # Scale down proportionally
    scale_factor = max_total / total_individual
    return [bet * scale_factor for bet in individual_results]


def compare_kelly_modes(
    model_prob: float, market_odds: int, bankroll: float = 10000.0
) -> dict[KellyMode, KellyResult]:
    """
    Compare different Kelly modes for a single bet.

    Args:
        model_prob: Model probability estimate
        market_odds: American odds
        bankroll: Available bankroll

    Returns:
        Dictionary mapping KellyMode to KellyResult
    """
    calculator = KellyCalculator(starting_bankroll=bankroll)
    results = {}

    for mode in KellyMode:
        result = calculator.calculate_optimal_bet_size(
            model_prob=model_prob, market_odds=market_odds, mode=mode
        )
        results[mode] = result

    return results
