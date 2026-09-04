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

# Same-week CORRELATED de-weight rule, FROZEN (D27-11 same-side; EXTENDED by D31-03 with
# same-game cross-target grouping). Documented as a string so the chosen formula is auditable
# as a constant.
#
# There is exactly ONE de-weighting formula. It takes TWO group sizes -- the same-SIDE group
# and the same-GAME group -- and applies the STRICTER of the two: the stake is divided by the
# square root of the LARGER group size, which is the expression the constant below states.
#
# The two groupings are never COMPOSED: a bet sharing both a side and a game with one other bet
# is de-weighted once by 1/sqrt(2), never twice by 1/2. D31-03's finding: the three targets'
# side strings are DISJOINT (home/away, home_cover/away_cover, over/under), so pooling a mixed
# week into same-side grouping alone yields six groups that never mix -- de-weighting would stay
# silently within-target and the correlation that matters (a WP home bet and an ATS home-cover
# bet on the SAME game, close to one leveraged wager) would go entirely uncaptured.
#
# SAME_SIDE_DEWEIGHT and SAME_GAME_DEWEIGHT are the two LEGACY NAMES under which that single
# rule is quoted (D27-11 readouts use the first, D31-03 the second). They are deliberately the
# SAME string, and a test pins their equality, so the rule can never be documented two ways.
SAME_SIDE_DEWEIGHT = "stake / sqrt(max(same_side_group_size, same_game_group_size))"
SAME_GAME_DEWEIGHT = SAME_SIDE_DEWEIGHT

# Full-Kelly is EXCLUDED (ruin risk). The exclusion is enforced UPSTREAM, where the Kelly
# fraction is computed: KellyCalculator applies a 0.25 (quarter-Kelly) fraction by default
# and caps every bet at PER_BET_CAP_PCT (5%) of bankroll. The sizing pipeline below re-applies
# that 5% per-bet cap as its operative per-bet fraction ceiling. There is deliberately NO
# separate half-Kelly clamp: given quarter-Kelly + the 5% cap it could never bind, so it only
# advertised a protection it did not provide (WR-01, removed 2026-06-08; D27-09).

# The LOCKED cap order (review tightening #2). The single orchestration helper
# apply_sizing_pipeline applies these four steps in EXACTLY this order; the
# BetSelector calls that helper so the order is enforced in one place.
CAP_ORDER = (
    "kelly_stake",
    "per_bet_5pct_cap",
    "same_game_and_side_deweight",
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


def _require_game_id(bet: dict) -> str:
    """Return a bet's ``game_id``, raising a NAMED KeyError when it is absent or None.

    No-silent-fallback guard (D31-03, T-31-17), following the D27-07 pattern
    already used at the per-season bias lookup. Falling back to same-side-only
    grouping would silently restore the exact blind spot D31-03 closes, on a
    pooled week, with nothing in the output to signal it. A present-but-None
    ``game_id`` is worse than an absent one -- it would group every
    unidentified bet together and still produce a number -- so it raises too.
    """
    if "game_id" not in bet or bet["game_id"] is None:
        msg = (
            "game_id is required for same-game correlated de-weighting (D31-03): a bet "
            "record without a game_id cannot be grouped by game, and there is NO silent "
            "fallback to same-side-only grouping (T-31-17)"
        )
        raise KeyError(msg)
    return str(bet["game_id"])


def apply_same_game_and_side_deweight(bets: list[dict]) -> list[dict]:
    """De-weight SAME-WEEK correlated concentration (D27-11 side; D31-03 game).

    Uses the FROZEN formula ``SAME_SIDE_DEWEIGHT`` / ``SAME_GAME_DEWEIGHT``
    (one rule, two legacy names)::

        stake / sqrt(max(same_side_group_size, same_game_group_size))

    Two INDEPENDENT group keys feed one factor:

    - The SIDE key is the CASE-INSENSITIVELY normalized ``bet_side``
      (lower-cased): all UNDERs are one group; a high-total OVER and a normal
      OVER are one "over" group (overs are NOT split by ``totals_regime``).
    - The GAME key is ``game_id`` VERBATIM -- a game_id is an exact identifier,
      and case-folding it would silently merge two distinct games.

    The STRICTER of the two groupings applies; they are never COMPOSED, so a
    bet sharing both a side and a game with one other bet is de-weighted once
    by ``1/sqrt(2)``, never twice by ``1/2``.

    The extension is MONOTONE at this step: because ``max(side, game) >= side``,
    no bet's de-weighted stake can be larger than the pre-D31-03 same-side-only
    rule would have produced. (Downstream, the pro-rata 10% weekly cap can
    redistribute room freed here, so an individual FINAL stake may rise even
    though a week's TOTAL exposure cannot -- see ``apply_weekly_exposure_cap``
    and ``TestWeeklyCapRedistribution``.)

    The covariance / joint-Kelly path is REJECTED (D27-11: overfit-prone on
    limited data, weather features degenerate). There is deliberately NO
    ``correlation_matrix`` argument -- this is SIMPLE grouping only.

    Args:
        bets: A week's bets; each dict carries ``game_id``, ``bet_side`` and
            ``stake``.

    Returns:
        Per-bet records (input order preserved) carrying metadata:
        ``game_id``, ``bet_side`` (normalized), ``original_stake``,
        ``deweighted_stake``, ``scale_factor``, ``same_side_group_size``,
        ``same_game_group_size``, and ``binding_group`` -- one of
        ``"same_side"``, ``"same_game"`` or ``"both"`` (the groupings tie,
        including the uncorrelated 1-and-1 case whose factor is exactly 1.0).

    Raises:
        KeyError: if any bet lacks a non-None ``game_id`` (D31-03, T-31-17).
        ValueError: if any stake is NaN (review tightening #8).
    """
    if not bets:
        return []

    # Count both group sizes up front: side by normalized string, game verbatim.
    side_sizes: dict[str, int] = {}
    game_sizes: dict[str, int] = {}
    for bet in bets:
        side = str(bet["bet_side"]).strip().lower()
        game_id = _require_game_id(bet)
        side_sizes[side] = side_sizes.get(side, 0) + 1
        game_sizes[game_id] = game_sizes.get(game_id, 0) + 1

    records: list[dict] = []
    for bet in bets:
        side = str(bet["bet_side"]).strip().lower()
        game_id = _require_game_id(bet)
        original = _validate_stake(bet["stake"])

        side_size = side_sizes[side]
        game_size = game_sizes[game_id]
        # The STRICTER grouping binds. Equal sizes are reported as "both": the
        # two rules agree, so naming either one alone would be arbitrary.
        if side_size > game_size:
            binding_group = "same_side"
        elif game_size > side_size:
            binding_group = "same_game"
        else:
            binding_group = "both"

        factor = 1.0 / math.sqrt(max(side_size, game_size))
        records.append(
            {
                "game_id": game_id,
                "bet_side": side,
                "original_stake": original,
                "deweighted_stake": original * factor,
                "scale_factor": factor,
                "same_side_group_size": side_size,
                "same_game_group_size": game_size,
                "binding_group": binding_group,
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

    Order (CAP_ORDER): calibrated Kelly stake -> 5% per-bet cap -> correlated
    (same-game-and-side) de-weight -> 10% weekly cap. This single orchestration
    helper is what the BetSelector (Plan 27-03) and the pooled cross-target
    selector core (Plan 31-06) consume, so the order is enforced in ONE place;
    callers do NOT re-order the steps. Because the de-weight step is inside this
    seam, pooling a week across targets needs no new sizing call site (D31-03).

    Each input bet carries a ``game_id``, a ``bet_side`` and a ``stake`` (the
    calibrated Kelly stake -- step 1, already computed upstream by
    ``KellyCalculator`` with its quarter-Kelly fraction and 5% per-bet cap).
    Step 2 re-applies the 5% per-bet cap (``PER_BET_CAP_PCT``) here as the
    operative per-bet fraction ceiling; full-Kelly is excluded by that cap
    together with the upstream quarter-Kelly fraction (there is no separate
    half-Kelly clamp -- see the module header).

    Args:
        bets: A week's bets; each dict carries ``game_id``, ``bet_side`` and
            ``stake``.
        bankroll: Current bankroll (must be strictly positive).

    Returns:
        Per-bet records (input order preserved) carrying the full metadata set:
        ``game_id``, ``bet_side``, ``original_stake``, ``deweighted_stake``,
        ``weekly_scaled_stake``, ``scale_factor`` (the COMBINED de-weight x
        weekly factor), plus the D31-03 grouping fields
        ``same_side_group_size``, ``same_game_group_size`` and
        ``binding_group``.

    Raises:
        KeyError: if any bet lacks a non-None ``game_id`` (D31-03, T-31-17).
        ValueError: if bankroll <= 0 or any stake is NaN (review #8).
    """
    _validate_bankroll(bankroll)
    if not bets:
        return []

    per_bet_cap = bankroll * PER_BET_CAP_PCT

    # Step 1 (kelly_stake) is the provided stake. Apply Step 2 (the 5% per-bet cap),
    # the operative per-bet fraction ceiling. ``game_id`` is carried through so step 3
    # can group by game; a bet without one raises here rather than being grouped wrongly.
    capped_bets: list[dict] = []
    for bet in bets:
        original = _validate_stake(bet["stake"])
        capped_stake = min(original, per_bet_cap)
        capped_bets.append(
            {
                "game_id": _require_game_id(bet),
                "bet_side": bet["bet_side"],
                "stake": capped_stake,
            }
        )

    # Step 3 (same_game_and_side_deweight): operate on the per-bet-capped stakes.
    deweighted = apply_same_game_and_side_deweight(capped_bets)

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
                "game_id": dw["game_id"],
                "bet_side": dw["bet_side"],
                "original_stake": original,
                "deweighted_stake": dw["deweighted_stake"],
                "weekly_scaled_stake": final_stake,
                "scale_factor": combined,
                # D31-03 grouping provenance: which groups the bet fell into and
                # which of the two bound, so a published stake can be explained
                # without recomputing the rule.
                "same_side_group_size": dw["same_side_group_size"],
                "same_game_group_size": dw["same_game_group_size"],
                "binding_group": dw["binding_group"],
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
