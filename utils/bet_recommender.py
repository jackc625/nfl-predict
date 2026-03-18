"""
Bet Recommendation Engine for NFL Prediction System.

This module provides sophisticated bet recommendation capabilities that combine
bet selection, Kelly criterion sizing, and bankroll management to generate
optimal betting recommendations with proper risk management.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np

from .bankroll_manager import BankrollManager
from .bet_selector import BetCandidate, BetSelectionResult, BetSelector
from .betting_utils import (
    BettingResult,
    BetType,
)
from .kelly_criterion import KellyCalculator, KellyMode
from .logging_config import get_logger
from .unit_sizing import UnitSizer

logger = get_logger(__name__)


class RecommendationTier(Enum):
    """Tiers for bet recommendations based on confidence and edge."""

    PREMIUM = "premium"  # Highest confidence, best edge
    STRONG = "strong"  # High confidence, good edge
    VALUE = "value"  # Medium confidence, decent edge
    SPECULATIVE = "speculative"  # Lower confidence, high edge


class RecommendationAction(Enum):
    """Recommended actions for bets."""

    BET = "bet"  # Place the bet as recommended
    MONITOR = "monitor"  # Watch for line movement
    PASS = "pass"  # Skip this opportunity
    ALERT = "alert"  # Requires manual review


@dataclass
class UnitRecommendation:
    """Unit size recommendation with rationale."""

    units: float
    kelly_size: float
    confidence_adjustment: float
    risk_adjustment: float
    bankroll_limit: float | None
    reasoning: str


@dataclass
class BetRecommendation:
    """Complete bet recommendation with all decision factors."""

    # Core bet information (required fields first)
    game_id: str
    bet_type: BetType
    team: str | None
    description: str

    # Market information (required fields)
    market_odds: int
    line_value: float | None  # Spread or total value

    # Model assessment (required fields)
    model_prob: float
    market_prob: float
    edge: float
    confidence: float
    expected_value: float

    # Recommendation details (required fields)
    recommendation_tier: RecommendationTier
    action: RecommendationAction
    unit_recommendation: UnitRecommendation
    priority_score: float

    # Optional fields with defaults
    closing_odds: int | None = None
    kickoff_time: datetime | None = None
    hours_until_kickoff: float | None = None
    home_team: str | None = None
    away_team: str | None = None
    week: int | None = None
    season: int | None = None
    market_disagreement: float | None = None
    line_movement: float | None = None
    injury_concerns: list[str] | None = None
    weather_impact: str | None = None
    recommendation_timestamp: datetime = field(default_factory=datetime.now)
    recommendation_id: str | None = None
    notes: str | None = None


@dataclass
class RecommendationPortfolio:
    """Complete portfolio of bet recommendations."""

    recommendations: list[BetRecommendation]
    tier_distribution: dict[RecommendationTier, int]
    total_units_recommended: float
    total_expected_value: float
    portfolio_kelly_size: float

    # Risk metrics
    max_drawdown_risk: float
    correlation_risk_score: float
    diversification_score: float

    # Summary statistics
    average_edge: float
    average_confidence: float
    total_games: int

    # Metadata
    recommendation_timestamp: datetime = field(default_factory=datetime.now)
    bankroll_snapshot: float | None = None
    week_context: dict | None = None


class BetRecommender:
    """
    Comprehensive bet recommendation engine.

    Combines bet selection, Kelly sizing, and risk management to generate
    optimal betting recommendations with proper portfolio construction.
    """

    def __init__(
        self,
        bankroll_manager: BankrollManager,
        bet_selector: BetSelector | None = None,
        kelly_calculator: KellyCalculator | None = None,
        unit_sizer: UnitSizer | None = None,
    ):
        """Initialize recommendation engine with required components."""
        self.bankroll_manager = bankroll_manager
        self.bet_selector = bet_selector or BetSelector()
        self.kelly_calculator = kelly_calculator or KellyCalculator(
            mode=KellyMode.FRACTIONAL
        )
        self.unit_sizer = unit_sizer or UnitSizer()

        logger.info(
            f"BetRecommender initialized with bankroll: ${bankroll_manager.current_bankroll:.2f}"
        )

    def generate_recommendations(
        self,
        betting_results: list[BettingResult],
        game_context: dict[str, dict] | None = None,
        closing_lines: dict[str, dict] | None = None,
        current_positions: dict[str, int] | None = None,
        risk_factors: dict[str, dict] | None = None,
    ) -> RecommendationPortfolio:
        """
        Generate comprehensive betting recommendations.

        Args:
            betting_results: List of betting opportunities from analysis
            game_context: Game metadata (times, teams, weather, etc.)
            closing_lines: Closing line data for disagreement analysis
            current_positions: Current betting positions
            risk_factors: Additional risk information (injuries, etc.)

        Returns:
            RecommendationPortfolio with ranked recommendations
        """
        logger.info(
            f"Generating recommendations from {len(betting_results)} opportunities"
        )

        # Step 1: Select viable bets using filtering criteria
        selection_result = self.bet_selector.select_bets(
            betting_results, game_context, closing_lines, current_positions
        )

        logger.info(
            f"Selected {len(selection_result.selected_bets)} viable bets from filtering"
        )

        # Step 2: Create detailed recommendations for selected bets
        recommendations = []
        for bet_candidate in selection_result.selected_bets:
            recommendation = self._create_recommendation(
                bet_candidate, game_context, risk_factors
            )
            recommendations.append(recommendation)

        # Step 3: Optimize portfolio allocation
        recommendations = self._optimize_portfolio(recommendations)

        # Step 4: Create portfolio summary
        portfolio = self._create_portfolio_summary(recommendations, selection_result)

        logger.info(f"Generated {len(recommendations)} final recommendations")
        return portfolio

    def _create_recommendation(
        self,
        bet_candidate: BetCandidate,
        game_context: dict[str, dict] | None,
        risk_factors: dict[str, dict] | None,
    ) -> BetRecommendation:
        """Create detailed recommendation for a bet candidate."""

        # Get additional context
        context = game_context.get(bet_candidate.game_id, {}) if game_context else {}
        risks = risk_factors.get(bet_candidate.game_id, {}) if risk_factors else {}

        # Calculate unit recommendation
        unit_rec = self._calculate_unit_recommendation(bet_candidate)

        # Determine recommendation tier and action
        tier = self._determine_recommendation_tier(bet_candidate)
        action = self._determine_recommendation_action(bet_candidate, unit_rec)

        # Calculate priority score
        priority_score = self._calculate_priority_score(bet_candidate, tier)

        # Calculate timing information
        hours_until_kickoff = None
        if bet_candidate.kickoff_time:
            hours_until_kickoff = (
                bet_candidate.kickoff_time - datetime.now()
            ).total_seconds() / 3600

        # Create recommendation description
        description = self._create_bet_description(bet_candidate)

        return BetRecommendation(
            game_id=bet_candidate.game_id,
            bet_type=bet_candidate.bet_type,
            team=bet_candidate.team,
            description=description,
            market_odds=bet_candidate.market_odds,
            line_value=bet_candidate.line_value,
            closing_odds=bet_candidate.closing_odds,
            model_prob=bet_candidate.model_prob,
            market_prob=bet_candidate.market_prob,
            edge=bet_candidate.edge,
            confidence=bet_candidate.confidence,
            expected_value=bet_candidate.expected_value,
            recommendation_tier=tier,
            action=action,
            unit_recommendation=unit_rec,
            priority_score=priority_score,
            kickoff_time=bet_candidate.kickoff_time,
            hours_until_kickoff=hours_until_kickoff,
            home_team=bet_candidate.home_team,
            away_team=bet_candidate.away_team,
            week=bet_candidate.week,
            season=bet_candidate.season,
            market_disagreement=bet_candidate.market_disagreement,
            line_movement=context.get("line_movement"),
            injury_concerns=risks.get("injuries"),
            weather_impact=risks.get("weather_impact"),
            recommendation_id=self._generate_recommendation_id(bet_candidate),
        )

    def _calculate_unit_recommendation(
        self, bet_candidate: BetCandidate
    ) -> UnitRecommendation:
        """Calculate optimal unit sizing for a bet."""

        # Calculate Kelly size
        kelly_result = self.kelly_calculator.calculate_optimal_bet_size(
            model_prob=bet_candidate.model_prob,
            market_odds=bet_candidate.market_odds,
            market_prob=bet_candidate.market_prob,
        )

        # Get confidence-based sizing
        confidence_metrics = self.unit_sizer.calculate_confidence_metrics(
            model_prob=bet_candidate.model_prob,
            edge=bet_candidate.edge,
            expected_value=bet_candidate.expected_value,
        )

        unit_rec = self.unit_sizer.recommend_unit_size(
            kelly_size=kelly_result.recommended_size,
            confidence_metrics=confidence_metrics,
            current_bankroll=self.bankroll_manager.current_bankroll,
        )

        # Apply bankroll manager constraints
        max_units = self.bankroll_manager.get_max_bet_size(bet_candidate.bet_type.value)
        constrained_units = min(unit_rec.recommended_units, max_units)

        # Create reasoning
        reasoning_parts = []
        reasoning_parts.append(f"Kelly size: {kelly_result.recommended_size:.2f}")
        reasoning_parts.append(
            f"Confidence adjustment: {confidence_metrics.distance_from_fifty:.3f}"
        )

        if constrained_units < unit_rec.recommended_units:
            reasoning_parts.append(f"Bankroll limit applied: {max_units:.2f}")

        return UnitRecommendation(
            units=constrained_units,
            kelly_size=kelly_result.recommended_size,
            confidence_adjustment=confidence_metrics.distance_from_fifty,
            risk_adjustment=kelly_result.risk_adjustment,
            bankroll_limit=max_units
            if constrained_units < unit_rec.recommended_units
            else None,
            reasoning=" | ".join(reasoning_parts),
        )

    def _determine_recommendation_tier(
        self, bet_candidate: BetCandidate
    ) -> RecommendationTier:
        """Determine recommendation tier based on edge and confidence."""

        edge = bet_candidate.edge
        confidence = bet_candidate.confidence

        # Premium tier: High edge AND high confidence
        if edge >= 0.06 and confidence >= 0.08:
            return RecommendationTier.PREMIUM

        # Strong tier: Good edge OR high confidence
        if edge >= 0.04 and confidence >= 0.06:
            return RecommendationTier.STRONG

        # Value tier: Decent edge with medium confidence
        if edge >= 0.025 and confidence >= 0.04:
            return RecommendationTier.VALUE

        # Speculative: Lower confidence but potentially high edge
        return RecommendationTier.SPECULATIVE

    def _determine_recommendation_action(
        self, bet_candidate: BetCandidate, unit_rec: UnitRecommendation
    ) -> RecommendationAction:
        """Determine recommended action for a bet."""

        # Check if units are too small to be worthwhile
        if unit_rec.units < 0.25:
            return RecommendationAction.MONITOR

        # Check timing constraints
        if bet_candidate.kickoff_time:
            hours_until = (
                bet_candidate.kickoff_time - datetime.now()
            ).total_seconds() / 3600
            if hours_until < 1.0:  # Less than 1 hour
                return RecommendationAction.ALERT

        # Check for extreme probabilities that need review
        if bet_candidate.model_prob > 0.85 or bet_candidate.model_prob < 0.15:
            return RecommendationAction.ALERT

        # Default to betting if all checks pass
        return RecommendationAction.BET

    def _calculate_priority_score(
        self, bet_candidate: BetCandidate, tier: RecommendationTier
    ) -> float:
        """Calculate priority score for ranking recommendations."""

        # Base score from expected value
        base_score = bet_candidate.expected_value * 100

        # Tier multipliers
        tier_multipliers = {
            RecommendationTier.PREMIUM: 1.5,
            RecommendationTier.STRONG: 1.2,
            RecommendationTier.VALUE: 1.0,
            RecommendationTier.SPECULATIVE: 0.8,
        }

        # Edge and confidence bonuses
        edge_bonus = bet_candidate.edge * 50  # Higher edge = higher priority
        confidence_bonus = (
            bet_candidate.confidence * 30
        )  # Higher confidence = higher priority

        # Market disagreement bonus (if available)
        disagreement_bonus = 0
        if bet_candidate.market_disagreement:
            disagreement_bonus = bet_candidate.market_disagreement * 20

        # Calculate final score
        priority_score = (
            base_score + edge_bonus + confidence_bonus + disagreement_bonus
        ) * tier_multipliers[tier]

        return round(priority_score, 2)

    def _create_bet_description(self, bet_candidate: BetCandidate) -> str:
        """Create human-readable description of the bet."""

        game_desc = (
            f"{bet_candidate.away_team} @ {bet_candidate.home_team}"
            if bet_candidate.home_team
            else bet_candidate.game_id
        )

        if bet_candidate.bet_type == BetType.MONEYLINE:
            if bet_candidate.team:
                return f"{bet_candidate.team} ML ({bet_candidate.market_odds:+d}) vs {game_desc}"
            return f"Moneyline ({bet_candidate.market_odds:+d}) - {game_desc}"

        if bet_candidate.bet_type == BetType.SPREAD:
            if bet_candidate.line_value is not None:
                spread_str = f"{bet_candidate.line_value:+.1f}"
                if bet_candidate.team:
                    return f"{bet_candidate.team} {spread_str} ({bet_candidate.market_odds:+d}) vs {game_desc}"
                return f"Spread {spread_str} ({bet_candidate.market_odds:+d}) - {game_desc}"
            return f"Spread ({bet_candidate.market_odds:+d}) - {game_desc}"

        if bet_candidate.bet_type == BetType.TOTAL:
            if bet_candidate.line_value is not None:
                total_str = f"O/U {bet_candidate.line_value:.1f}"
                return f"{total_str} ({bet_candidate.market_odds:+d}) - {game_desc}"
            return f"Total ({bet_candidate.market_odds:+d}) - {game_desc}"

        return f"{bet_candidate.bet_type.value} - {game_desc}"

    def _optimize_portfolio(
        self, recommendations: list[BetRecommendation]
    ) -> list[BetRecommendation]:
        """Optimize portfolio allocation across recommendations."""

        # Sort by priority score
        recommendations.sort(key=lambda x: x.priority_score, reverse=True)

        # Check bankroll constraints and adjust if needed
        total_units = sum(rec.unit_recommendation.units for rec in recommendations)
        available_units = self.bankroll_manager.get_available_units()

        if total_units > available_units:
            logger.warning(
                f"Total recommended units ({total_units:.2f}) exceeds available ({available_units:.2f})"
            )

            # Scale down proportionally or remove lower priority bets
            scale_factor = available_units / total_units
            if scale_factor > 0.7:  # Scale down if not too severe
                for rec in recommendations:
                    original_units = rec.unit_recommendation.units
                    scaled_units = original_units * scale_factor
                    rec.unit_recommendation.units = scaled_units
                    rec.unit_recommendation.reasoning += (
                        f" | Scaled by {scale_factor:.2f} for bankroll"
                    )
            else:
                # Remove lower priority bets
                cumulative_units = 0
                filtered_recommendations = []
                for rec in recommendations:
                    if (
                        cumulative_units + rec.unit_recommendation.units
                        <= available_units
                    ):
                        filtered_recommendations.append(rec)
                        cumulative_units += rec.unit_recommendation.units
                    else:
                        rec.action = RecommendationAction.MONITOR
                        rec.notes = "Removed due to bankroll constraints"
                        filtered_recommendations.append(rec)
                recommendations = filtered_recommendations

        return recommendations

    def _create_portfolio_summary(
        self,
        recommendations: list[BetRecommendation],
        selection_result: BetSelectionResult,
    ) -> RecommendationPortfolio:
        """Create comprehensive portfolio summary."""

        # Calculate tier distribution
        tier_distribution = {}
        for tier in RecommendationTier:
            count = sum(1 for rec in recommendations if rec.recommendation_tier == tier)
            if count > 0:
                tier_distribution[tier] = count

        # Calculate portfolio metrics
        betting_recs = [
            rec for rec in recommendations if rec.action == RecommendationAction.BET
        ]

        total_units = sum(rec.unit_recommendation.units for rec in betting_recs)
        total_ev = sum(
            rec.expected_value * rec.unit_recommendation.units for rec in betting_recs
        )

        avg_edge = np.mean([rec.edge for rec in betting_recs]) if betting_recs else 0.0
        avg_confidence = (
            np.mean([rec.confidence for rec in betting_recs]) if betting_recs else 0.0
        )

        # Calculate portfolio Kelly size
        portfolio_kelly = sum(
            rec.unit_recommendation.kelly_size for rec in betting_recs
        )

        # Risk metrics (simplified)
        max_drawdown_risk = min(
            0.1, total_units * 0.02
        )  # Estimate based on total exposure
        correlation_risk = self._calculate_correlation_risk(recommendations)
        diversification_score = self._calculate_diversification_score(recommendations)

        # Count unique games
        unique_games = len({rec.game_id for rec in recommendations})

        return RecommendationPortfolio(
            recommendations=recommendations,
            tier_distribution=tier_distribution,
            total_units_recommended=total_units,
            total_expected_value=total_ev,
            portfolio_kelly_size=portfolio_kelly,
            max_drawdown_risk=max_drawdown_risk,
            correlation_risk_score=correlation_risk,
            diversification_score=diversification_score,
            average_edge=avg_edge,
            average_confidence=avg_confidence,
            total_games=unique_games,
            bankroll_snapshot=self.bankroll_manager.current_bankroll,
        )

    def _calculate_correlation_risk(
        self, recommendations: list[BetRecommendation]
    ) -> float:
        """Calculate portfolio correlation risk score."""

        # Count bets per game
        game_bet_counts = {}
        for rec in recommendations:
            if rec.action == RecommendationAction.BET:
                game_bet_counts[rec.game_id] = game_bet_counts.get(rec.game_id, 0) + 1

        # Higher correlation risk if multiple bets per game
        max_bets_per_game = max(game_bet_counts.values()) if game_bet_counts else 0
        correlation_risk = min(
            1.0, max_bets_per_game * 0.25
        )  # 0.25 per additional bet per game

        return correlation_risk

    def _calculate_diversification_score(
        self, recommendations: list[BetRecommendation]
    ) -> float:
        """Calculate portfolio diversification score."""

        betting_recs = [
            rec for rec in recommendations if rec.action == RecommendationAction.BET
        ]

        if not betting_recs:
            return 1.0

        # Count bet types
        bet_type_counts = {}
        for rec in betting_recs:
            bet_type_counts[rec.bet_type] = bet_type_counts.get(rec.bet_type, 0) + 1

        # Count unique games
        unique_games = len({rec.game_id for rec in betting_recs})

        # Diversification score based on bet type variety and game spread
        bet_type_variety = len(bet_type_counts) / 3  # 3 possible bet types
        game_spread = min(1.0, unique_games / len(betting_recs))

        diversification_score = (bet_type_variety + game_spread) / 2
        return min(1.0, diversification_score)

    def _generate_recommendation_id(self, bet_candidate: BetCandidate) -> str:
        """Generate unique recommendation ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        game_id = bet_candidate.game_id.replace("_", "")
        bet_type = bet_candidate.bet_type.value[:2].upper()
        return f"REC_{timestamp}_{game_id}_{bet_type}"

    def export_recommendations_json(
        self, portfolio: RecommendationPortfolio, filepath: str
    ) -> None:
        """Export recommendations to JSON format."""

        # Convert to serializable format
        export_data = {
            "portfolio_summary": {
                "total_recommendations": len(portfolio.recommendations),
                "total_units_recommended": portfolio.total_units_recommended,
                "total_expected_value": portfolio.total_expected_value,
                "average_edge": portfolio.average_edge,
                "average_confidence": portfolio.average_confidence,
                "total_games": portfolio.total_games,
                "bankroll_snapshot": portfolio.bankroll_snapshot,
                "recommendation_timestamp": portfolio.recommendation_timestamp.isoformat(),
                "tier_distribution": {
                    tier.value: count
                    for tier, count in portfolio.tier_distribution.items()
                },
                "risk_metrics": {
                    "max_drawdown_risk": portfolio.max_drawdown_risk,
                    "correlation_risk_score": portfolio.correlation_risk_score,
                    "diversification_score": portfolio.diversification_score,
                },
            },
            "recommendations": [],
        }

        for rec in portfolio.recommendations:
            rec_data = {
                "recommendation_id": rec.recommendation_id,
                "game_info": {
                    "game_id": rec.game_id,
                    "home_team": rec.home_team,
                    "away_team": rec.away_team,
                    "week": rec.week,
                    "season": rec.season,
                    "kickoff_time": rec.kickoff_time.isoformat()
                    if rec.kickoff_time
                    else None,
                    "hours_until_kickoff": rec.hours_until_kickoff,
                },
                "bet_details": {
                    "bet_type": rec.bet_type.value,
                    "team": rec.team,
                    "description": rec.description,
                    "market_odds": rec.market_odds,
                    "line_value": rec.line_value,
                    "closing_odds": rec.closing_odds,
                },
                "analysis": {
                    "model_prob": rec.model_prob,
                    "market_prob": rec.market_prob,
                    "edge": rec.edge,
                    "confidence": rec.confidence,
                    "expected_value": rec.expected_value,
                    "market_disagreement": rec.market_disagreement,
                },
                "recommendation": {
                    "tier": rec.recommendation_tier.value,
                    "action": rec.action.value,
                    "priority_score": rec.priority_score,
                    "units": rec.unit_recommendation.units,
                    "kelly_size": rec.unit_recommendation.kelly_size,
                    "reasoning": rec.unit_recommendation.reasoning,
                },
                "risk_factors": {
                    "line_movement": rec.line_movement,
                    "injury_concerns": rec.injury_concerns,
                    "weather_impact": rec.weather_impact,
                },
                "metadata": {
                    "recommendation_timestamp": rec.recommendation_timestamp.isoformat(),
                    "notes": rec.notes,
                },
            }
            export_data["recommendations"].append(rec_data)

        # Write to file
        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2)

        logger.info(
            f"Exported {len(portfolio.recommendations)} recommendations to {filepath}"
        )

    def get_recommendation_summary(
        self, portfolio: RecommendationPortfolio
    ) -> dict[str, Any]:
        """Generate human-readable summary of recommendations."""

        betting_recs = [
            rec
            for rec in portfolio.recommendations
            if rec.action == RecommendationAction.BET
        ]

        summary = {
            "portfolio_overview": {
                "total_recommendations": len(portfolio.recommendations),
                "actionable_bets": len(betting_recs),
                "total_units": round(portfolio.total_units_recommended, 2),
                "total_expected_value": round(portfolio.total_expected_value, 4),
                "bankroll_utilization": round(
                    portfolio.total_units_recommended
                    / portfolio.bankroll_snapshot
                    * 100,
                    1,
                )
                if portfolio.bankroll_snapshot
                else 0,
            },
            "quality_metrics": {
                "average_edge": round(portfolio.average_edge, 4),
                "average_confidence": round(portfolio.average_confidence, 4),
                "portfolio_kelly": round(portfolio.portfolio_kelly_size, 2),
                "diversification_score": round(portfolio.diversification_score, 3),
            },
            "tier_breakdown": {
                tier.value: count for tier, count in portfolio.tier_distribution.items()
            },
            "top_recommendations": [
                {
                    "rank": i + 1,
                    "description": rec.description,
                    "tier": rec.recommendation_tier.value,
                    "action": rec.action.value,
                    "units": round(rec.unit_recommendation.units, 2),
                    "edge": round(rec.edge, 4),
                    "expected_value": round(rec.expected_value, 4),
                    "priority_score": rec.priority_score,
                }
                for i, rec in enumerate(
                    sorted(
                        portfolio.recommendations,
                        key=lambda x: x.priority_score,
                        reverse=True,
                    )[:5]
                )
            ],
        }

        return summary
