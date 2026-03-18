"""
Bet Selection and Filtering System for NFL Prediction System.

This module implements sophisticated bet selection logic with configurable filters
for edge thresholds, confidence requirements, position limits, and market disagreement
detection to identify the highest-value betting opportunities.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np

from .betting_utils import BettingResult, BetType
from .logging_config import get_logger

logger = get_logger(__name__)


class FilterReason(Enum):
    """Reasons why a bet was filtered out."""

    EDGE_TOO_LOW = "edge_below_threshold"
    CONFIDENCE_TOO_LOW = "confidence_below_threshold"
    POSITION_LIMIT_REACHED = "position_limit_reached"
    MARKET_DISAGREEMENT_LOW = "market_disagreement_low"
    PROBABILITY_EXTREME = "probability_too_extreme"
    ODDS_UNAVAILABLE = "odds_unavailable"
    GAME_TOO_SOON = "game_too_soon"
    DUPLICATE_BET = "duplicate_bet"


@dataclass
class FilterCriteria:
    """Configuration for bet selection filters."""

    # Edge requirements
    min_edge_threshold: float = 0.02  # 2% minimum edge
    min_edge_moneyline: float = 0.015  # Lower threshold for moneylines (harder to find)
    min_edge_spread: float = 0.02  # Standard threshold for spreads
    min_edge_total: float = 0.025  # Higher threshold for totals (more variance)

    # Confidence requirements
    min_confidence_threshold: float = 0.05  # |prob - 0.5| >= 0.05 (55%+ or 45%- prob)
    min_confidence_moneyline: float = 0.03  # Lower for moneylines
    min_confidence_spread: float = 0.05  # Standard for spreads
    min_confidence_total: float = 0.06  # Higher for totals

    # Position limits
    max_bets_per_week: int = 8
    max_bets_per_game: int = 2
    max_bets_per_bet_type: dict[BetType, int] = field(
        default_factory=lambda: {
            BetType.MONEYLINE: 4,
            BetType.SPREAD: 6,
            BetType.TOTAL: 4,
        }
    )

    # Market disagreement
    min_market_disagreement: float = 0.03  # 3% disagreement with closing line
    closing_line_weight: float = 0.7  # Weight given to closing line vs opening

    # Timing filters
    min_hours_before_kickoff: float = 2.0  # Don't bet within 2 hours of kickoff

    # Quality filters
    max_probability_extreme: float = 0.95  # Don't bet on >95% or <5% probabilities
    min_probability_extreme: float = 0.05


@dataclass
class BetCandidate:
    """A potential bet that has passed initial screening."""

    game_id: str
    bet_type: BetType
    team: str | None  # None for totals
    model_prob: float
    market_prob: float
    edge: float
    confidence: float
    expected_value: float
    market_odds: int
    line_value: float | None  # Spread or total value

    # Market context
    opening_odds: int | None = None
    closing_odds: int | None = None
    market_disagreement: float | None = None

    # Game context
    kickoff_time: datetime | None = None
    home_team: str | None = None
    away_team: str | None = None
    week: int | None = None
    season: int | None = None

    # Selection metadata
    rank: int | None = None
    selected: bool = False
    filter_reasons: list[FilterReason] = field(default_factory=list)


@dataclass
class BetSelectionResult:
    """Result of bet selection process."""

    selected_bets: list[BetCandidate]
    filtered_bets: list[BetCandidate]
    total_candidates: int
    filters_applied: dict[str, int]  # Count of bets filtered by each reason

    # Portfolio metrics
    total_expected_value: float
    average_edge: float
    average_confidence: float
    bet_type_distribution: dict[BetType, int]

    # Timing
    selection_timestamp: datetime = field(default_factory=datetime.now)


class BetSelector:
    """
    Sophisticated bet selection system with configurable filtering.

    Implements multi-stage filtering to identify the highest-value betting
    opportunities while managing risk and position limits.
    """

    def __init__(self, criteria: FilterCriteria | None = None):
        """Initialize bet selector with filtering criteria."""
        self.criteria = criteria or FilterCriteria()
        logger.info(f"BetSelector initialized with criteria: {self.criteria}")

    def select_bets(
        self,
        betting_results: list[BettingResult],
        game_context: dict[str, dict] | None = None,
        closing_lines: dict[str, dict] | None = None,
        current_positions: dict[str, int] | None = None,
    ) -> BetSelectionResult:
        """
        Select optimal bets from a list of betting opportunities.

        Args:
            betting_results: List of BettingResult objects from betting analysis
            game_context: Dict with game metadata (kickoff times, teams, etc.)
            closing_lines: Dict with closing line information for disagreement detection
            current_positions: Dict with current position counts by bet type/team

        Returns:
            BetSelectionResult with selected and filtered bets
        """
        logger.info(f"Starting bet selection from {len(betting_results)} candidates")

        # Convert BettingResults to BetCandidates
        candidates = self._create_candidates(
            betting_results, game_context, closing_lines
        )

        # Apply filters in order of importance
        candidates = self._apply_edge_filters(candidates)
        candidates = self._apply_confidence_filters(candidates)
        candidates = self._apply_timing_filters(candidates)
        candidates = self._apply_quality_filters(candidates)
        candidates = self._apply_market_disagreement_filters(candidates, closing_lines)

        # Apply position limits and select top candidates
        selected_candidates = self._apply_position_limits(candidates, current_positions)

        # Create result summary
        result = self._create_selection_result(candidates, selected_candidates)

        logger.info(
            f"Selected {len(selected_candidates)} bets from {len(betting_results)} candidates"
        )
        return result

    def _create_candidates(
        self,
        betting_results: list[BettingResult],
        game_context: dict[str, dict] | None,
        closing_lines: dict[str, dict] | None,
    ) -> list[BetCandidate]:
        """Convert BettingResults to BetCandidates with additional context."""
        candidates = []

        for result in betting_results:
            # Extract game context if available
            game_id = getattr(result, "game_id", "unknown")
            context = game_context.get(game_id, {}) if game_context else {}

            # Calculate confidence as distance from 50%
            confidence = abs(result.model_prob - 0.5)

            # Get closing line info if available
            closing_info = closing_lines.get(game_id, {}) if closing_lines else {}
            market_disagreement = self._calculate_market_disagreement(
                result, closing_info
            )

            candidate = BetCandidate(
                game_id=game_id,
                bet_type=result.bet_type,
                team=getattr(result, "team", None),
                model_prob=result.model_prob,
                market_prob=result.market_prob,
                edge=result.edge,
                confidence=confidence,
                expected_value=result.expected_value,
                market_odds=getattr(result, "market_odds", 0),
                line_value=getattr(result, "line_value", None),
                opening_odds=getattr(result, "opening_odds", None),
                closing_odds=closing_info.get("closing_odds"),
                market_disagreement=market_disagreement,
                kickoff_time=context.get("kickoff_time"),
                home_team=context.get("home_team"),
                away_team=context.get("away_team"),
                week=context.get("week"),
                season=context.get("season"),
            )

            candidates.append(candidate)

        logger.debug(f"Created {len(candidates)} bet candidates")
        return candidates

    def _apply_edge_filters(self, candidates: list[BetCandidate]) -> list[BetCandidate]:
        """Filter candidates based on minimum edge requirements."""
        filtered = []

        for candidate in candidates:
            # Get bet-type specific edge threshold
            if candidate.bet_type == BetType.MONEYLINE:
                min_edge = self.criteria.min_edge_moneyline
            elif candidate.bet_type == BetType.SPREAD:
                min_edge = self.criteria.min_edge_spread
            elif candidate.bet_type == BetType.TOTAL:
                min_edge = self.criteria.min_edge_total
            else:
                min_edge = self.criteria.min_edge_threshold

            if candidate.edge >= min_edge:
                filtered.append(candidate)
            else:
                candidate.filter_reasons.append(FilterReason.EDGE_TOO_LOW)
                filtered.append(candidate)  # Keep for reporting

        passed_count = sum(
            1 for c in filtered if FilterReason.EDGE_TOO_LOW not in c.filter_reasons
        )
        logger.debug(f"Edge filter: {passed_count}/{len(candidates)} candidates passed")
        return filtered

    def _apply_confidence_filters(
        self, candidates: list[BetCandidate]
    ) -> list[BetCandidate]:
        """Filter candidates based on minimum confidence requirements."""
        for candidate in candidates:
            # Skip if already filtered
            if candidate.filter_reasons:
                continue

            # Get bet-type specific confidence threshold
            if candidate.bet_type == BetType.MONEYLINE:
                min_confidence = self.criteria.min_confidence_moneyline
            elif candidate.bet_type == BetType.SPREAD:
                min_confidence = self.criteria.min_confidence_spread
            elif candidate.bet_type == BetType.TOTAL:
                min_confidence = self.criteria.min_confidence_total
            else:
                min_confidence = self.criteria.min_confidence_threshold

            if candidate.confidence < min_confidence:
                candidate.filter_reasons.append(FilterReason.CONFIDENCE_TOO_LOW)

        passed_count = sum(1 for c in candidates if not c.filter_reasons)
        logger.debug(f"Confidence filter: {passed_count} candidates passed")
        return candidates

    def _apply_timing_filters(
        self, candidates: list[BetCandidate]
    ) -> list[BetCandidate]:
        """Filter candidates based on timing requirements."""
        current_time = datetime.now()

        for candidate in candidates:
            # Skip if already filtered
            if candidate.filter_reasons:
                continue

            # Check if game is too soon
            if candidate.kickoff_time:
                hours_until_kickoff = (
                    candidate.kickoff_time - current_time
                ).total_seconds() / 3600
                if hours_until_kickoff < self.criteria.min_hours_before_kickoff:
                    candidate.filter_reasons.append(FilterReason.GAME_TOO_SOON)

        passed_count = sum(1 for c in candidates if not c.filter_reasons)
        logger.debug(f"Timing filter: {passed_count} candidates passed")
        return candidates

    def _apply_quality_filters(
        self, candidates: list[BetCandidate]
    ) -> list[BetCandidate]:
        """Filter candidates based on probability quality requirements."""
        for candidate in candidates:
            # Skip if already filtered
            if candidate.filter_reasons:
                continue

            # Check for extreme probabilities
            if (
                candidate.model_prob <= self.criteria.min_probability_extreme
                or candidate.model_prob >= self.criteria.max_probability_extreme
            ):
                candidate.filter_reasons.append(FilterReason.PROBABILITY_EXTREME)

            # Check for missing odds
            if candidate.market_odds == 0:
                candidate.filter_reasons.append(FilterReason.ODDS_UNAVAILABLE)

        passed_count = sum(1 for c in candidates if not c.filter_reasons)
        logger.debug(f"Quality filter: {passed_count} candidates passed")
        return candidates

    def _apply_market_disagreement_filters(
        self, candidates: list[BetCandidate], closing_lines: dict[str, dict] | None
    ) -> list[BetCandidate]:
        """Filter candidates based on market disagreement requirements."""
        if not closing_lines:
            logger.debug("No closing lines available, skipping disagreement filter")
            return candidates

        for candidate in candidates:
            # Skip if already filtered
            if candidate.filter_reasons:
                continue

            if candidate.market_disagreement is not None and (
                candidate.market_disagreement < self.criteria.min_market_disagreement
            ):
                candidate.filter_reasons.append(FilterReason.MARKET_DISAGREEMENT_LOW)

        passed_count = sum(1 for c in candidates if not c.filter_reasons)
        logger.debug(f"Market disagreement filter: {passed_count} candidates passed")
        return candidates

    def _apply_position_limits(
        self,
        candidates: list[BetCandidate],
        current_positions: dict[str, int] | None,
    ) -> list[BetCandidate]:
        """Apply position limits and select top candidates."""
        # Filter to only valid candidates
        valid_candidates = [c for c in candidates if not c.filter_reasons]

        # Sort by expected value (descending)
        valid_candidates.sort(key=lambda c: c.expected_value, reverse=True)

        # Track selections
        selected = []
        game_counts = {}
        bet_type_counts = dict.fromkeys(BetType, 0)

        # Add current positions if provided
        if current_positions:
            for bet_type_str, count in current_positions.items():
                if bet_type_str in [bt.value for bt in BetType]:
                    bet_type = BetType(bet_type_str)
                    bet_type_counts[bet_type] = count

        for candidate in valid_candidates:
            # Check weekly limit
            if len(selected) >= self.criteria.max_bets_per_week:
                candidate.filter_reasons.append(FilterReason.POSITION_LIMIT_REACHED)
                continue

            # Check per-game limit
            game_bets = game_counts.get(candidate.game_id, 0)
            if game_bets >= self.criteria.max_bets_per_game:
                candidate.filter_reasons.append(FilterReason.POSITION_LIMIT_REACHED)
                continue

            # Check bet-type specific limit
            bet_type_limit = self.criteria.max_bets_per_bet_type.get(
                candidate.bet_type, float("inf")
            )
            if bet_type_counts[candidate.bet_type] >= bet_type_limit:
                candidate.filter_reasons.append(FilterReason.POSITION_LIMIT_REACHED)
                continue

            # Select this bet
            candidate.selected = True
            candidate.rank = len(selected) + 1
            selected.append(candidate)

            # Update counts
            game_counts[candidate.game_id] = game_counts.get(candidate.game_id, 0) + 1
            bet_type_counts[candidate.bet_type] += 1

        logger.info(
            f"Position limits applied: selected {len(selected)} from {len(valid_candidates)} valid candidates"
        )
        return selected

    def _calculate_market_disagreement(
        self, result: BettingResult, closing_info: dict
    ) -> float | None:
        """Calculate disagreement between our line and market closing line."""
        if not closing_info:
            return None

        closing_prob = closing_info.get("closing_probability")
        if closing_prob is None:
            return None

        # Calculate weighted disagreement (opening + closing)
        if hasattr(result, "opening_prob") and result.opening_prob is not None:
            market_consensus = (
                1 - self.criteria.closing_line_weight
            ) * result.opening_prob + self.criteria.closing_line_weight * closing_prob
        else:
            market_consensus = closing_prob

        disagreement = abs(result.model_prob - market_consensus)
        return disagreement

    def _create_selection_result(
        self,
        all_candidates: list[BetCandidate],
        selected_candidates: list[BetCandidate],
    ) -> BetSelectionResult:
        """Create comprehensive selection result summary."""
        # Count filters applied
        filters_applied = {}
        for reason in FilterReason:
            count = sum(1 for c in all_candidates if reason in c.filter_reasons)
            if count > 0:
                filters_applied[reason.value] = count

        # Calculate portfolio metrics for selected bets
        if selected_candidates:
            total_ev = sum(c.expected_value for c in selected_candidates)
            avg_edge = np.mean([c.edge for c in selected_candidates])
            avg_confidence = np.mean([c.confidence for c in selected_candidates])

            bet_type_dist = {}
            for bet_type in BetType:
                count = sum(1 for c in selected_candidates if c.bet_type == bet_type)
                if count > 0:
                    bet_type_dist[bet_type] = count
        else:
            total_ev = 0.0
            avg_edge = 0.0
            avg_confidence = 0.0
            bet_type_dist = {}

        filtered_candidates = [c for c in all_candidates if c.filter_reasons]

        return BetSelectionResult(
            selected_bets=selected_candidates,
            filtered_bets=filtered_candidates,
            total_candidates=len(all_candidates),
            filters_applied=filters_applied,
            total_expected_value=total_ev,
            average_edge=avg_edge,
            average_confidence=avg_confidence,
            bet_type_distribution=bet_type_dist,
        )

    def get_selection_summary(self, result: BetSelectionResult) -> dict[str, Any]:
        """Generate human-readable summary of bet selection."""
        summary = {
            "selection_overview": {
                "total_candidates": result.total_candidates,
                "selected_bets": len(result.selected_bets),
                "filtered_bets": len(result.filtered_bets),
                "selection_rate": len(result.selected_bets) / result.total_candidates
                if result.total_candidates > 0
                else 0,
            },
            "portfolio_metrics": {
                "total_expected_value": round(result.total_expected_value, 4),
                "average_edge": round(result.average_edge, 4),
                "average_confidence": round(result.average_confidence, 4),
            },
            "bet_distribution": {
                bet_type.value: count
                for bet_type, count in result.bet_type_distribution.items()
            },
            "filter_breakdown": result.filters_applied,
            "top_selections": [
                {
                    "game": f"{c.away_team} @ {c.home_team}"
                    if c.home_team and c.away_team
                    else c.game_id,
                    "bet_type": c.bet_type.value,
                    "edge": round(c.edge, 4),
                    "expected_value": round(c.expected_value, 4),
                    "confidence": round(c.confidence, 4),
                    "rank": c.rank,
                }
                for c in result.selected_bets[:5]  # Top 5
            ],
        }

        return summary


def create_default_criteria() -> FilterCriteria:
    """Create default filtering criteria based on conservative best practices."""
    return FilterCriteria(
        min_edge_threshold=0.02,  # 2% minimum edge
        min_confidence_threshold=0.05,  # 55%+ probability
        max_bets_per_week=6,  # Conservative position sizing
        max_bets_per_game=1,  # Avoid correlation
        min_market_disagreement=0.025,  # 2.5% disagreement
        min_hours_before_kickoff=3.0,  # 3 hours minimum
    )


def create_aggressive_criteria() -> FilterCriteria:
    """Create aggressive filtering criteria for higher volume betting."""
    return FilterCriteria(
        min_edge_threshold=0.015,  # 1.5% minimum edge
        min_confidence_threshold=0.03,  # 53%+ probability
        max_bets_per_week=12,  # Higher volume
        max_bets_per_game=2,  # Allow some correlation
        min_market_disagreement=0.015,  # 1.5% disagreement
        min_hours_before_kickoff=1.0,  # 1 hour minimum
    )
