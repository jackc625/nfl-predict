"""
API Recommendation Bridge.

This module provides a bridge between the existing sophisticated recommendation
engine and the API services, converting between internal recommendation formats
and API response schemas.
"""

import logging
from typing import Any

import pandas as pd

# Import existing recommendation system
try:
    from .bet_recommender import (
        BetRecommendation as InternalBetRecommendation,
    )
    from .bet_recommender import (
        BetRecommendationEngine,
    )
    from .bet_recommender import (
        RecommendationAction as InternalRecommendationAction,
    )
    from .bet_recommender import (
        RecommendationTier as InternalRecommendationTier,
    )
    from .bet_selector import BetCandidate, BetSelector, FilterCriteria
    from .betting_utils import BetType as InternalBetType

    HAS_RECOMMENDATION_ENGINE = True
except ImportError:
    # Fallback if recommendation engine is not available
    HAS_RECOMMENDATION_ENGINE = False

    # Define placeholder types
    class InternalBetRecommendation:
        pass

    class InternalRecommendationTier:
        pass

    class InternalRecommendationAction:
        pass

    class InternalBetType:
        pass

    class BetCandidate:
        pass

    class FilterCriteria:
        pass

    class BetSelector:
        def select_bets(self, *args, **kwargs):
            return type("Result", (), {"selected_bets": []})()

    class BetRecommendationEngine:
        pass


# Import API schemas
try:
    from api.schemas import BetRecommendation, BetType, RecommendationTier

    HAS_API_SCHEMAS = True
except ImportError:
    # Fallback for testing without full API
    HAS_API_SCHEMAS = False
    from typing import NamedTuple

    class BetType:
        MONEYLINE = "moneyline"
        SPREAD = "spread"
        TOTAL = "total"

    class RecommendationTier:
        HIGH = "high"
        MEDIUM = "medium"
        LOW = "low"

    class BetRecommendation(NamedTuple):
        bet_type: str
        team: str
        recommended_odds: int
        edge: float
        expected_value: float
        confidence: float
        tier: str
        units: float
        description: str


logger = logging.getLogger(__name__)


class APIRecommendationBridge:
    """
    Bridge between internal recommendation engine and API schemas.

    This class handles the conversion between the sophisticated internal
    recommendation system and the simpler API response format.
    """

    def __init__(self):
        """Initialize the recommendation bridge."""
        self.has_engine = HAS_RECOMMENDATION_ENGINE

        if self.has_engine:
            try:
                self.engine = BetRecommendationEngine()
                self.selector = BetSelector()
                logger.info("Recommendation engine initialized successfully")
            except (ValueError, KeyError, TypeError, ImportError) as e:
                logger.warning(f"Failed to initialize recommendation engine: {e}")
                self.has_engine = False
        else:
            logger.warning("Recommendation engine not available, using fallback")

    def generate_game_recommendations(
        self,
        game_id: str,
        predictions: pd.Series,
        market_data: pd.Series,
        min_edge: float = 0.02,
    ) -> list[BetRecommendation]:
        """
        Generate bet recommendations for a single game.

        Args:
            game_id: Game identifier
            predictions: Game predictions data
            market_data: Market odds data
            min_edge: Minimum edge threshold

        Returns:
            List of API-formatted bet recommendations
        """
        if not self.has_engine:
            return self._generate_fallback_recommendations(
                game_id, predictions, market_data, min_edge
            )

        try:
            # Convert to internal format and generate recommendations
            internal_recommendations = self._generate_internal_recommendations(
                game_id, predictions, market_data, min_edge
            )

            # Convert to API format
            api_recommendations = []
            for rec in internal_recommendations:
                api_rec = self._convert_to_api_recommendation(rec)
                if api_rec:
                    api_recommendations.append(api_rec)

            return api_recommendations

        except (ValueError, KeyError, TypeError, ImportError) as e:
            logger.error(f"Error generating recommendations for game {game_id}: {e}")
            return self._generate_fallback_recommendations(
                game_id, predictions, market_data, min_edge
            )

    def get_filtered_recommendations(
        self,
        predictions_df: pd.DataFrame,
        market_df: pd.DataFrame,
        min_edge: float = 0.02,
        has_recommendations_filter: bool | None = None,
    ) -> list[dict[str, Any]]:
        """
        Get filtered recommendations for multiple games.

        Args:
            predictions_df: DataFrame with game predictions
            market_df: DataFrame with market data
            min_edge: Minimum edge threshold
            has_recommendations_filter: Filter for games with/without recommendations

        Returns:
            List of games with recommendation metadata
        """
        if not self.has_engine:
            return self._get_fallback_filtered_recommendations(
                predictions_df, market_df, min_edge, has_recommendations_filter
            )

        try:
            results = []

            for _, pred_row in predictions_df.iterrows():
                game_id = pred_row.get("game_id")
                if not game_id:
                    continue

                # Find corresponding market data
                market_row = market_df[market_df["game_id"] == game_id]
                market_row = pd.Series() if market_row.empty else market_row.iloc[0]

                # Generate recommendations
                recommendations = self.generate_game_recommendations(
                    game_id, pred_row, market_row, min_edge
                )

                has_recs = len(recommendations) > 0
                top_recommendation = recommendations[0] if recommendations else None

                # Apply filter
                if has_recommendations_filter is not None:
                    if has_recommendations_filter != has_recs:
                        continue

                results.append(
                    {
                        "game_id": game_id,
                        "has_recommendations": has_recs,
                        "top_recommendation": top_recommendation,
                        "total_recommendations": len(recommendations),
                    }
                )

            return results

        except (ValueError, KeyError, TypeError, ImportError) as e:
            logger.error(f"Error getting filtered recommendations: {e}")
            return self._get_fallback_filtered_recommendations(
                predictions_df, market_df, min_edge, has_recommendations_filter
            )

    def _generate_internal_recommendations(
        self,
        game_id: str,
        predictions: pd.Series,
        market_data: pd.Series,
        min_edge: float,
    ) -> list[InternalBetRecommendation]:
        """Generate recommendations using internal engine."""

        # Create bet candidates from predictions and market data
        candidates = []

        # Win probability / Moneyline bet
        if "win_probability" in predictions and "home_moneyline" in market_data:
            win_prob = predictions["win_probability"]
            home_ml = market_data.get("home_moneyline")
            market_data.get("away_moneyline")

            if home_ml and win_prob:
                # Calculate edge for home team
                market_prob = self._moneyline_to_probability(home_ml)
                edge = abs(win_prob - market_prob)

                if edge >= min_edge:
                    candidate = BetCandidate(
                        game_id=game_id,
                        bet_type=InternalBetType.MONEYLINE,
                        team=predictions.get("home_team"),
                        model_prob=win_prob,
                        market_prob=market_prob,
                        edge=edge,
                        confidence=abs(win_prob - 0.5),
                        expected_value=edge * 100,  # Simplified EV
                        market_odds=home_ml,
                        line_value=None,
                    )
                    candidates.append(candidate)

        # Apply selection criteria using existing selector
        filter_criteria = FilterCriteria(
            min_edge_threshold=min_edge, min_confidence_threshold=0.05
        )

        # Use selector to filter and rank candidates
        selection_result = self.selector.select_bets(
            candidates, filter_criteria, max_selections=5
        )

        # Convert selected candidates to full recommendations
        # This is a simplified version - the real engine would do more sophisticated analysis
        recommendations = []
        for candidate in selection_result.selected_bets:
            rec = self._create_recommendation_from_candidate(candidate)
            recommendations.append(rec)

        return recommendations

    def _create_recommendation_from_candidate(
        self, candidate: BetCandidate
    ) -> InternalBetRecommendation:
        """Create a full recommendation from a bet candidate."""

        # Determine tier based on edge and confidence
        if candidate.edge >= 0.05 and candidate.confidence >= 0.15:
            tier = InternalRecommendationTier.PREMIUM
        elif candidate.edge >= 0.03 and candidate.confidence >= 0.10:
            tier = InternalRecommendationTier.STRONG
        elif candidate.edge >= 0.02 and candidate.confidence >= 0.05:
            tier = InternalRecommendationTier.VALUE
        else:
            tier = InternalRecommendationTier.SPECULATIVE

        # Create unit recommendation (simplified)
        from .unit_sizing import UnitRecommendation

        unit_rec = UnitRecommendation(
            units=min(candidate.edge * 20, 3.0),  # Simple Kelly-like sizing
            kelly_size=candidate.edge * 25,
            confidence_adjustment=candidate.confidence,
            risk_adjustment=1.0,
            bankroll_limit=None,
            reasoning=f"Edge: {candidate.edge:.1%}, Confidence: {candidate.confidence:.1%}",
        )

        # Create full recommendation
        rec = InternalBetRecommendation(
            game_id=candidate.game_id,
            bet_type=candidate.bet_type,
            team=candidate.team,
            description=self._create_description(candidate),
            market_odds=candidate.market_odds,
            line_value=candidate.line_value,
            model_prob=candidate.model_prob,
            market_prob=candidate.market_prob,
            edge=candidate.edge,
            confidence=candidate.confidence,
            expected_value=candidate.expected_value,
            recommendation_tier=tier,
            action=InternalRecommendationAction.BET,
            unit_recommendation=unit_rec,
            priority_score=candidate.edge * candidate.confidence * 100,
            home_team=candidate.home_team,
            away_team=candidate.away_team,
        )

        return rec

    def _convert_to_api_recommendation(
        self, internal_rec: InternalBetRecommendation
    ) -> BetRecommendation | None:
        """Convert internal recommendation to API format."""
        try:
            # Map internal enums to API enums
            api_bet_type = self._map_bet_type(internal_rec.bet_type)
            api_tier = self._map_recommendation_tier(internal_rec.recommendation_tier)

            api_rec = BetRecommendation(
                bet_type=api_bet_type,
                team=internal_rec.team,
                recommended_odds=internal_rec.market_odds,
                edge=internal_rec.edge,
                expected_value=internal_rec.expected_value,
                confidence=internal_rec.confidence,
                tier=api_tier,
                units=internal_rec.unit_recommendation.units
                if internal_rec.unit_recommendation
                else 1.0,
                description=internal_rec.description,
            )

            return api_rec

        except (ValueError, KeyError, TypeError, ImportError) as e:
            logger.error(f"Error converting recommendation to API format: {e}")
            return None

    def _generate_fallback_recommendations(
        self,
        game_id: str,
        predictions: pd.Series,
        market_data: pd.Series,
        min_edge: float,
    ) -> list[BetRecommendation]:
        """Generate simple fallback recommendations when engine is unavailable."""
        recommendations = []

        try:
            # Simple win probability recommendation
            win_prob = predictions.get("win_probability")
            if win_prob and abs(win_prob - 0.5) >= min_edge:
                edge = abs(win_prob - 0.5)
                team = (
                    predictions.get("home_team")
                    if win_prob > 0.5
                    else predictions.get("away_team")
                )

                rec = BetRecommendation(
                    bet_type=BetType.MONEYLINE,
                    team=team,
                    recommended_odds=-110,  # Standard odds
                    edge=edge,
                    expected_value=edge * 100,
                    confidence=edge,
                    tier=RecommendationTier.MEDIUM
                    if edge >= 0.1
                    else RecommendationTier.LOW,
                    units=min(edge * 10, 2.0),
                    description=f"Model favors {'home' if win_prob > 0.5 else 'away'} team",
                )
                recommendations.append(rec)

        except (ValueError, KeyError, TypeError, ImportError) as e:
            logger.error(f"Error generating fallback recommendations: {e}")

        return recommendations

    def _get_fallback_filtered_recommendations(
        self,
        predictions_df: pd.DataFrame,
        market_df: pd.DataFrame,
        min_edge: float,
        has_recommendations_filter: bool | None,
    ) -> list[dict[str, Any]]:
        """Fallback filtered recommendations."""
        results = []

        for _, pred_row in predictions_df.iterrows():
            game_id = pred_row.get("game_id")
            win_prob = pred_row.get("win_probability", 0.5)

            has_recs = abs(win_prob - 0.5) >= min_edge

            if has_recommendations_filter is not None:
                if has_recommendations_filter != has_recs:
                    continue

            top_rec = None
            if has_recs:
                top_rec = BetRecommendation(
                    bet_type=BetType.MONEYLINE,
                    team=pred_row.get("home_team")
                    if win_prob > 0.5
                    else pred_row.get("away_team"),
                    recommended_odds=-110,
                    edge=abs(win_prob - 0.5),
                    expected_value=abs(win_prob - 0.5) * 100,
                    confidence=abs(win_prob - 0.5),
                    tier=RecommendationTier.MEDIUM,
                    units=1.0,
                    description="Simple model recommendation",
                )

            results.append(
                {
                    "game_id": game_id,
                    "has_recommendations": has_recs,
                    "top_recommendation": top_rec,
                    "total_recommendations": 1 if has_recs else 0,
                }
            )

        return results

    def _create_description(self, candidate: BetCandidate) -> str:
        """Create human-readable description for recommendation."""
        bet_type_desc = {
            InternalBetType.MONEYLINE: "Moneyline",
            InternalBetType.SPREAD: "Point Spread",
            InternalBetType.TOTAL: "Over/Under",
        }.get(candidate.bet_type, "Bet")

        team_desc = f" on {candidate.team}" if candidate.team else ""
        edge_desc = f"Edge: {candidate.edge:.1%}"

        return f"{bet_type_desc}{team_desc} - {edge_desc}"

    @staticmethod
    def _map_bet_type(internal_type: "InternalBetType") -> BetType:
        """Map internal bet type to API bet type."""
        mapping = {
            InternalBetType.MONEYLINE: BetType.MONEYLINE,
            InternalBetType.SPREAD: BetType.SPREAD,
            InternalBetType.TOTAL: BetType.TOTAL,
        }
        return mapping.get(internal_type, BetType.MONEYLINE)

    @staticmethod
    def _map_recommendation_tier(
        internal_tier: "InternalRecommendationTier",
    ) -> RecommendationTier:
        """Map internal recommendation tier to API tier."""
        mapping = {
            InternalRecommendationTier.PREMIUM: RecommendationTier.HIGH,
            InternalRecommendationTier.STRONG: RecommendationTier.HIGH,
            InternalRecommendationTier.VALUE: RecommendationTier.MEDIUM,
            InternalRecommendationTier.SPECULATIVE: RecommendationTier.LOW,
        }
        return mapping.get(internal_tier, RecommendationTier.MEDIUM)

    @staticmethod
    def _moneyline_to_probability(moneyline: int) -> float:
        """Convert American moneyline odds to implied probability."""
        if moneyline == 0:
            return 0.5

        if moneyline > 0:
            return 100 / (moneyline + 100)
        return abs(moneyline) / (abs(moneyline) + 100)


# Create global instance
api_recommendation_bridge = APIRecommendationBridge()
