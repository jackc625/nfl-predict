"""
Advanced unit sizing system based on edge strength, confidence levels, and risk management.

This module provides sophisticated unit sizing algorithms that consider multiple
factors including model confidence, historical performance, and market conditions.
"""

from typing import Dict, List, Optional, Tuple, Union
import math
import numpy as np
from dataclasses import dataclass
from enum import Enum

from utils.probability_utils import moneyline_to_probability, edge_calculation


class ConfidenceMethod(Enum):
    """Methods for calculating betting confidence."""
    EDGE_BASED = "edge_based"
    VOLATILITY_ADJUSTED = "volatility_adjusted"
    HISTORICAL_PERFORMANCE = "historical_performance"
    MARKET_DISAGREEMENT = "market_disagreement"
    COMBINED = "combined"


class UnitScale(Enum):
    """Unit scaling approaches."""
    LINEAR = "linear"
    LOGARITHMIC = "logarithmic"
    EXPONENTIAL = "exponential"
    THRESHOLD = "threshold"


@dataclass
class ConfidenceMetrics:
    """Container for confidence calculation metrics."""
    edge_confidence: float
    model_confidence: float
    market_confidence: float
    historical_confidence: float
    combined_confidence: float
    confidence_factors: Dict[str, float]

    @property
    def distance_from_fifty(self) -> float:
        """Calculate distance from neutral (50%) for compatibility."""
        return self.combined_confidence


@dataclass
class UnitRecommendation:
    """Container for unit sizing recommendation."""
    recommended_units: float
    base_units: float
    confidence_multiplier: float
    risk_adjustment: float
    final_multiplier: float
    reasoning: str
    dollar_amount: float
    bankroll_percentage: float
    confidence_metrics: ConfidenceMetrics


class UnitSizer:
    """
    Advanced unit sizing system with confidence-based adjustments.

    Features:
    - Edge-based base unit calculation
    - Confidence adjustments from multiple sources
    - Historical performance integration
    - Market condition adjustments
    - Risk-based scaling
    """

    def __init__(
        self,
        base_unit_dollar: float = 100.0,
        max_units: float = 10.0,
        min_units: float = 0.5,
        confidence_scaling: float = 2.0,
        edge_threshold: float = 0.02,
        confidence_threshold: float = 0.6,
        volatility_factor: float = 0.1,
        historical_weight: float = 0.3
    ):
        """
        Initialize unit sizer.

        Args:
            base_unit_dollar: Base unit size in dollars
            max_units: Maximum units for any single bet
            min_units: Minimum units for viable bets
            confidence_scaling: How much confidence affects sizing
            edge_threshold: Minimum edge required for betting
            confidence_threshold: Minimum confidence required for betting
            volatility_factor: Adjustment factor for volatility
            historical_weight: Weight given to historical performance
        """
        self.base_unit_dollar = base_unit_dollar
        self.max_units = max_units
        self.min_units = min_units
        self.confidence_scaling = confidence_scaling
        self.edge_threshold = edge_threshold
        self.confidence_threshold = confidence_threshold
        self.volatility_factor = volatility_factor
        self.historical_weight = historical_weight

        # Historical tracking
        self.bet_history: List[Dict] = []
        self.performance_by_confidence: Dict[str, List[float]] = {}
        self.performance_by_edge: Dict[str, List[float]] = {}

    def calculate_edge_confidence(
        self,
        model_prob: float,
        market_prob: float,
        sample_size: Optional[int] = None
    ) -> float:
        """
        Calculate confidence based on edge size and statistical significance.

        Args:
            model_prob: Model probability estimate
            market_prob: Market probability
            sample_size: Sample size used for model estimate

        Returns:
            Edge-based confidence (0-1)
        """
        edge = abs(model_prob - market_prob)

        # Base confidence from edge magnitude
        if edge >= 0.15:
            base_confidence = 1.0
        elif edge >= 0.10:
            base_confidence = 0.9
        elif edge >= 0.05:
            base_confidence = 0.7
        elif edge >= 0.03:
            base_confidence = 0.5
        elif edge >= 0.02:
            base_confidence = 0.3
        else:
            base_confidence = 0.1

        # Adjust for sample size if available
        if sample_size is not None:
            # Statistical confidence adjustment
            if sample_size >= 1000:
                sample_adjustment = 1.0
            elif sample_size >= 500:
                sample_adjustment = 0.9
            elif sample_size >= 200:
                sample_adjustment = 0.8
            elif sample_size >= 100:
                sample_adjustment = 0.7
            else:
                sample_adjustment = 0.5

            base_confidence *= sample_adjustment

        return min(base_confidence, 1.0)

    def calculate_model_confidence(
        self,
        model_prob: float,
        prediction_interval: Optional[Tuple[float, float]] = None,
        model_accuracy: Optional[float] = None
    ) -> float:
        """
        Calculate confidence based on model characteristics.

        Args:
            model_prob: Model probability estimate
            prediction_interval: Confidence interval for prediction
            model_accuracy: Historical model accuracy

        Returns:
            Model-based confidence (0-1)
        """
        # Distance from 50/50 indicates stronger signal
        distance_from_neutral = abs(model_prob - 0.5) * 2

        base_confidence = distance_from_neutral

        # Adjust for prediction interval width if available
        if prediction_interval is not None:
            interval_width = prediction_interval[1] - prediction_interval[0]
            # Narrower intervals indicate higher confidence
            interval_adjustment = max(0.3, 1.0 - interval_width)
            base_confidence *= interval_adjustment

        # Adjust for historical model accuracy
        if model_accuracy is not None:
            accuracy_adjustment = min(1.0, model_accuracy / 0.6)  # Scale so 60% accuracy = 1.0
            base_confidence *= accuracy_adjustment

        return min(base_confidence, 1.0)

    def calculate_market_confidence(
        self,
        market_prob: float,
        line_movement: Optional[float] = None,
        volume_indicator: Optional[float] = None,
        time_to_game: Optional[float] = None
    ) -> float:
        """
        Calculate confidence based on market characteristics.

        Args:
            market_prob: Market probability
            line_movement: Recent line movement (positive = moved toward our side)
            volume_indicator: Relative betting volume (0-1)
            time_to_game: Hours until game

        Returns:
            Market-based confidence (0-1)
        """
        base_confidence = 0.7  # Moderate baseline

        # Line movement in our favor increases confidence
        if line_movement is not None:
            if line_movement > 0.02:  # Significant movement in our favor
                base_confidence *= 1.2
            elif line_movement < -0.02:  # Movement against us
                base_confidence *= 0.8

        # Lower volume might indicate softer line
        if volume_indicator is not None:
            if volume_indicator < 0.3:  # Low volume
                base_confidence *= 1.1
            elif volume_indicator > 0.8:  # High volume
                base_confidence *= 0.9

        # Confidence generally decreases closer to game time
        if time_to_game is not None:
            if time_to_game < 2:  # Less than 2 hours
                base_confidence *= 0.8
            elif time_to_game > 24:  # More than 24 hours
                base_confidence *= 1.1

        return min(base_confidence, 1.0)

    def calculate_historical_confidence(
        self,
        similar_situations: Optional[List[Dict]] = None,
        recent_performance: Optional[Dict] = None
    ) -> float:
        """
        Calculate confidence based on historical performance.

        Args:
            similar_situations: Historical results in similar situations
            recent_performance: Recent betting performance metrics

        Returns:
            Historical-based confidence (0-1)
        """
        base_confidence = 0.7

        # Adjust based on similar historical situations
        if similar_situations:
            results = [s.get('outcome', False) for s in similar_situations]
            if results:
                success_rate = sum(results) / len(results)
                if success_rate >= 0.6:
                    base_confidence *= 1.2
                elif success_rate <= 0.4:
                    base_confidence *= 0.8

        # Adjust based on recent performance
        if recent_performance:
            recent_roi = recent_performance.get('roi', 0.0)
            recent_accuracy = recent_performance.get('accuracy', 0.5)

            if recent_roi > 0.05 and recent_accuracy > 0.55:
                base_confidence *= 1.1
            elif recent_roi < -0.05 or recent_accuracy < 0.45:
                base_confidence *= 0.9

        return min(base_confidence, 1.0)

    def calculate_combined_confidence(
        self,
        edge_confidence: float,
        model_confidence: float,
        market_confidence: float,
        historical_confidence: float,
        weights: Optional[Dict[str, float]] = None
    ) -> Tuple[float, Dict[str, float]]:
        """
        Calculate combined confidence from all sources.

        Args:
            edge_confidence: Edge-based confidence
            model_confidence: Model-based confidence
            market_confidence: Market-based confidence
            historical_confidence: Historical confidence
            weights: Custom weights for each component

        Returns:
            Tuple of (combined_confidence, factor_breakdown)
        """
        if weights is None:
            weights = {
                'edge': 0.4,
                'model': 0.3,
                'market': 0.2,
                'historical': 0.1
            }

        # Weighted average
        combined = (
            edge_confidence * weights['edge'] +
            model_confidence * weights['model'] +
            market_confidence * weights['market'] +
            historical_confidence * weights['historical']
        )

        # Apply non-linear scaling for extreme values
        if combined >= 0.8:
            combined = 0.8 + (combined - 0.8) * 0.5  # Compress high values
        elif combined <= 0.3:
            combined = combined * 0.8  # Compress low values

        factor_breakdown = {
            'edge_contribution': edge_confidence * weights['edge'],
            'model_contribution': model_confidence * weights['model'],
            'market_contribution': market_confidence * weights['market'],
            'historical_contribution': historical_confidence * weights['historical']
        }

        return min(combined, 1.0), factor_breakdown

    def calculate_base_units_from_edge(
        self,
        edge: float,
        scale_method: UnitScale = UnitScale.LOGARITHMIC
    ) -> float:
        """
        Calculate base units based on edge size.

        Args:
            edge: Betting edge (positive values)
            scale_method: Scaling method to use

        Returns:
            Base unit recommendation
        """
        if edge <= 0:
            return 0.0

        if scale_method == UnitScale.LINEAR:
            # Linear scaling: 2% edge = 1 unit, 10% edge = 5 units
            base_units = edge * 50
        elif scale_method == UnitScale.LOGARITHMIC:
            # Logarithmic scaling: diminishing returns for large edges
            base_units = math.log(1 + edge * 20) * 2
        elif scale_method == UnitScale.EXPONENTIAL:
            # Exponential scaling: accelerating units for large edges
            base_units = (edge * 10) ** 1.5
        elif scale_method == UnitScale.THRESHOLD:
            # Threshold-based: discrete levels
            if edge >= 0.08:
                base_units = 4.0
            elif edge >= 0.05:
                base_units = 3.0
            elif edge >= 0.03:
                base_units = 2.0
            elif edge >= 0.02:
                base_units = 1.0
            else:
                base_units = 0.5
        else:
            # Default to logarithmic
            base_units = math.log(1 + edge * 20) * 2

        return min(base_units, self.max_units)

    def calculate_unit_recommendation(
        self,
        model_prob: float,
        market_odds: int,
        bankroll: float,
        market_prob: Optional[float] = None,
        confidence_overrides: Optional[Dict[str, float]] = None,
        scale_method: UnitScale = UnitScale.LOGARITHMIC,
        **kwargs
    ) -> UnitRecommendation:
        """
        Calculate comprehensive unit recommendation.

        Args:
            model_prob: Model probability estimate
            market_odds: American odds
            bankroll: Available bankroll
            market_prob: Market probability (calculated if not provided)
            confidence_overrides: Manual confidence overrides
            scale_method: Unit scaling method
            **kwargs: Additional parameters for confidence calculations

        Returns:
            UnitRecommendation with detailed breakdown
        """
        # Calculate market probability if not provided
        if market_prob is None:
            market_prob = moneyline_to_probability(market_odds)

        # Calculate edge
        edge = model_prob - market_prob

        # Early exit for insufficient edge
        if edge <= self.edge_threshold:
            return UnitRecommendation(
                recommended_units=0.0,
                base_units=0.0,
                confidence_multiplier=0.0,
                risk_adjustment=1.0,
                final_multiplier=0.0,
                reasoning="Edge below threshold",
                dollar_amount=0.0,
                bankroll_percentage=0.0,
                confidence_metrics=ConfidenceMetrics(0, 0, 0, 0, 0, {})
            )

        # Calculate confidence metrics
        edge_conf = confidence_overrides.get('edge', None) if confidence_overrides else None
        if edge_conf is None:
            edge_conf = self.calculate_edge_confidence(
                model_prob, market_prob, kwargs.get('sample_size')
            )

        model_conf = confidence_overrides.get('model', None) if confidence_overrides else None
        if model_conf is None:
            model_conf = self.calculate_model_confidence(
                model_prob, kwargs.get('prediction_interval'), kwargs.get('model_accuracy')
            )

        market_conf = confidence_overrides.get('market', None) if confidence_overrides else None
        if market_conf is None:
            market_conf = self.calculate_market_confidence(
                market_prob, kwargs.get('line_movement'),
                kwargs.get('volume_indicator'), kwargs.get('time_to_game')
            )

        historical_conf = confidence_overrides.get('historical', None) if confidence_overrides else None
        if historical_conf is None:
            historical_conf = self.calculate_historical_confidence(
                kwargs.get('similar_situations'), kwargs.get('recent_performance')
            )

        # Calculate combined confidence
        combined_conf, factor_breakdown = self.calculate_combined_confidence(
            edge_conf, model_conf, market_conf, historical_conf
        )

        # Create confidence metrics
        confidence_metrics = ConfidenceMetrics(
            edge_confidence=edge_conf,
            model_confidence=model_conf,
            market_confidence=market_conf,
            historical_confidence=historical_conf,
            combined_confidence=combined_conf,
            confidence_factors=factor_breakdown
        )

        # Exit if combined confidence too low
        if combined_conf < self.confidence_threshold:
            return UnitRecommendation(
                recommended_units=0.0,
                base_units=0.0,
                confidence_multiplier=combined_conf,
                risk_adjustment=1.0,
                final_multiplier=0.0,
                reasoning="Confidence below threshold",
                dollar_amount=0.0,
                bankroll_percentage=0.0,
                confidence_metrics=confidence_metrics
            )

        # Calculate base units from edge
        base_units = self.calculate_base_units_from_edge(edge, scale_method)

        # Apply confidence multiplier
        confidence_multiplier = 0.5 + (combined_conf * self.confidence_scaling)
        confidence_multiplier = min(confidence_multiplier, 2.0)  # Cap at 2x

        # Apply risk adjustment (could be from external risk manager)
        risk_adjustment = kwargs.get('risk_adjustment', 1.0)

        # Calculate final multiplier and units
        final_multiplier = confidence_multiplier * risk_adjustment
        recommended_units = base_units * final_multiplier

        # Apply min/max constraints
        if recommended_units < self.min_units:
            if recommended_units > 0:
                recommended_units = 0.0  # Below minimum viable
        recommended_units = min(recommended_units, self.max_units)

        # Calculate dollar amount and bankroll percentage
        dollar_amount = recommended_units * self.base_unit_dollar
        bankroll_percentage = dollar_amount / bankroll if bankroll > 0 else 0.0

        # Generate reasoning
        reasoning_parts = []
        reasoning_parts.append(f"Edge: {edge:.1%}")
        reasoning_parts.append(f"Base: {base_units:.1f}u")
        if confidence_multiplier != 1.0:
            reasoning_parts.append(f"Conf: {confidence_multiplier:.1f}x")
        if risk_adjustment != 1.0:
            reasoning_parts.append(f"Risk: {risk_adjustment:.1f}x")

        reasoning = " | ".join(reasoning_parts)

        return UnitRecommendation(
            recommended_units=recommended_units,
            base_units=base_units,
            confidence_multiplier=confidence_multiplier,
            risk_adjustment=risk_adjustment,
            final_multiplier=final_multiplier,
            reasoning=reasoning,
            dollar_amount=dollar_amount,
            bankroll_percentage=bankroll_percentage,
            confidence_metrics=confidence_metrics
        )

    def record_bet_outcome(
        self,
        model_prob: float,
        market_prob: float,
        edge: float,
        confidence: float,
        units_bet: float,
        outcome: bool,
        profit_loss: float
    ) -> None:
        """
        Record bet outcome for historical analysis.

        Args:
            model_prob: Model probability used
            market_prob: Market probability
            edge: Edge for the bet
            confidence: Confidence level used
            units_bet: Units wagered
            outcome: True if won
            profit_loss: Profit or loss in dollars
        """
        bet_record = {
            'model_prob': model_prob,
            'market_prob': market_prob,
            'edge': edge,
            'confidence': confidence,
            'units_bet': units_bet,
            'outcome': outcome,
            'profit_loss': profit_loss,
            'timestamp': np.datetime64('now')
        }

        self.bet_history.append(bet_record)

        # Update performance tracking by confidence buckets
        conf_bucket = self._get_confidence_bucket(confidence)
        if conf_bucket not in self.performance_by_confidence:
            self.performance_by_confidence[conf_bucket] = []
        self.performance_by_confidence[conf_bucket].append(profit_loss / units_bet)

        # Update performance tracking by edge buckets
        edge_bucket = self._get_edge_bucket(edge)
        if edge_bucket not in self.performance_by_edge:
            self.performance_by_edge[edge_bucket] = []
        self.performance_by_edge[edge_bucket].append(profit_loss / units_bet)

    def _get_confidence_bucket(self, confidence: float) -> str:
        """Get confidence bucket for performance tracking."""
        if confidence >= 0.8:
            return "high"
        elif confidence >= 0.6:
            return "medium"
        else:
            return "low"

    def _get_edge_bucket(self, edge: float) -> str:
        """Get edge bucket for performance tracking."""
        if edge >= 0.05:
            return "large"
        elif edge >= 0.03:
            return "medium"
        else:
            return "small"

    def get_performance_analysis(self) -> Dict[str, any]:
        """Get comprehensive performance analysis."""
        if not self.bet_history:
            return {"message": "No betting history available"}

        total_bets = len(self.bet_history)
        winning_bets = sum(1 for bet in self.bet_history if bet['outcome'])
        win_rate = winning_bets / total_bets

        total_profit = sum(bet['profit_loss'] for bet in self.bet_history)
        avg_profit_per_bet = total_profit / total_bets

        # Performance by confidence level
        conf_performance = {}
        for conf_level, returns in self.performance_by_confidence.items():
            if returns:
                conf_performance[conf_level] = {
                    'avg_return_per_unit': np.mean(returns),
                    'win_rate': sum(1 for r in returns if r > 0) / len(returns),
                    'total_bets': len(returns)
                }

        # Performance by edge level
        edge_performance = {}
        for edge_level, returns in self.performance_by_edge.items():
            if returns:
                edge_performance[edge_level] = {
                    'avg_return_per_unit': np.mean(returns),
                    'win_rate': sum(1 for r in returns if r > 0) / len(returns),
                    'total_bets': len(returns)
                }

        return {
            'total_bets': total_bets,
            'overall_win_rate': win_rate,
            'total_profit': total_profit,
            'avg_profit_per_bet': avg_profit_per_bet,
            'performance_by_confidence': conf_performance,
            'performance_by_edge': edge_performance,
            'recent_performance': self._get_recent_performance()
        }

    def _get_recent_performance(self, lookback: int = 20) -> Dict[str, float]:
        """Get recent performance metrics."""
        if len(self.bet_history) < lookback:
            recent_bets = self.bet_history
        else:
            recent_bets = self.bet_history[-lookback:]

        if not recent_bets:
            return {}

        recent_wins = sum(1 for bet in recent_bets if bet['outcome'])
        recent_roi = sum(bet['profit_loss'] for bet in recent_bets) / len(recent_bets)

        return {
            'recent_win_rate': recent_wins / len(recent_bets),
            'recent_avg_profit': recent_roi,
            'bets_analyzed': len(recent_bets)
        }