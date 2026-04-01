#!/usr/bin/env python3
"""
NFL Prediction Pipeline

This module provides a unified prediction pipeline that combines all models:
- Win Probability (WP) model for game outcomes
- Against The Spread (ATS) model for spread betting
- Over/Under (O/U) model for total points betting

The pipeline generates comprehensive predictions including:
- Fair lines and probabilities for all bet types
- Edge calculations vs market lines
- Structured output format for betting analysis
- Confidence metrics and model agreement analysis
- Bet recommendation engine integration

Models are loaded from versioned artifacts (via models.artifacts) rather than
legacy trainer classes. Each artifact contains the raw sklearn/xgboost model,
feature list, metadata, and optional calibrator.
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from models.artifacts import load_model_artifact
from models.train_ats import ResidualDistributionConverter
from models.train_ou import TotalDistributionConverter
from utils import get_logger

logger = get_logger(__name__)


class BetType(Enum):
    """Enumeration of available bet types."""

    MONEYLINE_HOME = "moneyline_home"
    MONEYLINE_AWAY = "moneyline_away"
    SPREAD_HOME = "spread_home"
    SPREAD_AWAY = "spread_away"
    OVER = "over"
    UNDER = "under"


@dataclass
class FairLine:
    """
    Container for fair line calculations.

    Attributes:
        bet_type: Type of bet
        fair_probability: Model-derived fair probability
        fair_odds_american: Fair odds in American format
        fair_odds_decimal: Fair odds in decimal format
        fair_line_value: Fair line value (spread/total)
        confidence: Model confidence in prediction
    """

    bet_type: BetType
    fair_probability: float
    fair_odds_american: int
    fair_odds_decimal: float
    fair_line_value: float | None = None
    confidence: float | None = None


@dataclass
class MarketEdge:
    """
    Container for edge calculations vs market.

    Attributes:
        bet_type: Type of bet
        market_probability: Market implied probability
        fair_probability: Model fair probability
        edge: Edge percentage (fair - market)
        market_odds_american: Market odds in American format
        fair_odds_american: Fair odds in American format
        expected_value: Expected value per unit bet
        kelly_fraction: Optimal Kelly bet size
        confidence_level: Confidence in the edge
    """

    bet_type: BetType
    market_probability: float
    fair_probability: float
    edge: float
    market_odds_american: int
    fair_odds_american: int
    expected_value: float
    kelly_fraction: float
    confidence_level: str


@dataclass
class BetRecommendation:
    """
    Container for bet recommendations.

    Attributes:
        bet_type: Type of bet
        recommendation: Recommendation level (STRONG_BET, BET, LEAN, PASS)
        edge: Edge percentage
        expected_value: Expected value
        kelly_fraction: Recommended bet size
        confidence: Model confidence
        reasoning: Text explanation of recommendation
    """

    bet_type: BetType
    recommendation: str
    edge: float
    expected_value: float
    kelly_fraction: float
    confidence: float
    reasoning: str


# -- Prediction dataclasses (replace legacy WPModelPrediction, ATSModelPrediction,
# OUModelPrediction from train_*.py modules) --


@dataclass
class WPPrediction:
    """Win probability prediction from artifact-loaded model."""

    game_id: str
    home_team: str
    away_team: str
    raw_win_probability: float
    calibrated_win_probability: float | None = None
    prediction_confidence: float | None = None
    feature_importances: dict[str, float] = field(default_factory=dict)


@dataclass
class ATSPrediction:
    """Against the spread prediction from artifact-loaded model."""

    predicted_margin: float
    predicted_spread: float
    cover_probability: float
    confidence: float | None = None
    feature_importances: dict[str, float] = field(default_factory=dict)


@dataclass
class OUPrediction:
    """Over/under prediction from artifact-loaded model."""

    predicted_total: float
    over_probability: float
    under_probability: float
    confidence: float | None = None
    feature_importances: dict[str, float] = field(default_factory=dict)


@dataclass
class UnifiedGamePrediction:
    """
    Container for unified game prediction combining all models.

    Attributes:
        game_id: Unique game identifier
        home_team: Home team abbreviation
        away_team: Away team abbreviation
        prediction_date: When prediction was made

        # Win Probability predictions
        wp_home_probability: Home team win probability
        wp_away_probability: Away team win probability

        # ATS predictions
        predicted_margin: Expected point margin
        predicted_spread: Predicted spread line
        ats_cover_probability: Probability home team covers spread

        # O/U predictions
        predicted_total: Expected total points
        over_probability: Probability total goes over
        under_probability: Probability total goes under

        # Market information
        market_moneyline_home: Market moneyline for home team
        market_moneyline_away: Market moneyline for away team
        market_spread: Market spread line
        market_total: Market total line

        # Fair lines
        fair_lines: Dictionary of fair lines by bet type

        # Edge analysis
        edges: Dictionary of edges by bet type

        # Recommendations
        recommendations: List of bet recommendations

        # Model diagnostics
        model_agreement: Model agreement score
        prediction_confidence: Overall prediction confidence
        feature_importance: Combined feature importance

        # Metadata
        model_versions: Dictionary of model versions used
        metadata: Additional prediction metadata
    """

    game_id: str
    home_team: str
    away_team: str
    prediction_date: datetime

    # Core predictions
    wp_home_probability: float
    wp_away_probability: float
    predicted_margin: float
    predicted_spread: float
    ats_cover_probability: float
    predicted_total: float
    over_probability: float
    under_probability: float

    # Market lines
    market_moneyline_home: int | None = None
    market_moneyline_away: int | None = None
    market_spread: float | None = None
    market_total: float | None = None

    # Analysis
    fair_lines: dict[BetType, FairLine] = field(default_factory=dict)
    edges: dict[BetType, MarketEdge] = field(default_factory=dict)
    recommendations: list[BetRecommendation] = field(default_factory=list)

    # Diagnostics
    model_agreement: float | None = None
    prediction_confidence: float | None = None
    feature_importance: dict[str, float] = field(default_factory=dict)

    # Metadata
    model_versions: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class OddsConverter:
    """Utility class for odds conversions."""

    @staticmethod
    def american_to_decimal(american_odds: int) -> float:
        """Convert American odds to decimal odds."""
        if american_odds > 0:
            return (american_odds / 100) + 1
        return (100 / abs(american_odds)) + 1

    @staticmethod
    def decimal_to_american(decimal_odds: float) -> int:
        """Convert decimal odds to American odds."""
        if decimal_odds >= 2.0:
            return int((decimal_odds - 1) * 100)
        return int(-100 / (decimal_odds - 1))

    @staticmethod
    def american_to_probability(
        american_odds: int, remove_juice: bool = False
    ) -> float:
        """Convert American odds to implied probability."""
        if american_odds > 0:
            prob = 100 / (american_odds + 100)
        else:
            prob = abs(american_odds) / (abs(american_odds) + 100)

        if remove_juice:
            # Simple devig assuming standard juice
            prob = prob * 1.05  # Rough adjustment for typical overround
            prob = min(prob, 0.99)  # Cap at 99%

        return prob

    @staticmethod
    def probability_to_american(probability: float) -> int:
        """Convert probability to American odds."""
        if probability >= 0.5:
            return int(-100 * probability / (1 - probability))
        return int(100 * (1 - probability) / probability)


class EdgeCalculator:
    """Calculator for betting edges and expected values."""

    def __init__(
        self, min_edge_threshold: float = 0.02, max_kelly_fraction: float = 0.25
    ):
        """
        Initialize edge calculator.

        Args:
            min_edge_threshold: Minimum edge to consider for betting
            max_kelly_fraction: Maximum Kelly fraction to recommend
        """
        self.min_edge_threshold = min_edge_threshold
        self.max_kelly_fraction = max_kelly_fraction
        self.odds_converter = OddsConverter()

    def calculate_edge(self, fair_probability: float, market_odds: int) -> MarketEdge:
        """Calculate edge for a betting opportunity."""
        market_probability = self.odds_converter.american_to_probability(market_odds)
        edge = fair_probability - market_probability

        # Expected value calculation
        decimal_odds = self.odds_converter.american_to_decimal(market_odds)
        expected_value = (fair_probability * (decimal_odds - 1)) - (
            1 - fair_probability
        )

        # Kelly criterion
        if edge > 0 and market_probability < 1:
            kelly_fraction = edge / (decimal_odds - 1)
            kelly_fraction = min(kelly_fraction, self.max_kelly_fraction)
        else:
            kelly_fraction = 0.0

        # Confidence level
        if edge > 0.05:
            confidence_level = "HIGH"
        elif edge > 0.03:
            confidence_level = "MEDIUM"
        elif edge > 0.01:
            confidence_level = "LOW"
        else:
            confidence_level = "NONE"

        return MarketEdge(
            bet_type=BetType.MONEYLINE_HOME,  # Will be updated by caller
            market_probability=market_probability,
            fair_probability=fair_probability,
            edge=edge,
            market_odds_american=market_odds,
            fair_odds_american=self.odds_converter.probability_to_american(
                fair_probability
            ),
            expected_value=expected_value,
            kelly_fraction=kelly_fraction,
            confidence_level=confidence_level,
        )


class BetRecommendationEngine:
    """Engine for generating bet recommendations."""

    def __init__(self, min_edge: float = 0.02, min_confidence: float = 0.6):
        """
        Initialize recommendation engine.

        Args:
            min_edge: Minimum edge for bet recommendation
            min_confidence: Minimum confidence for bet recommendation
        """
        self.min_edge = min_edge
        self.min_confidence = min_confidence

    def generate_recommendation(
        self, edge: MarketEdge, confidence: float
    ) -> BetRecommendation:
        """Generate a bet recommendation based on edge and confidence."""

        # Determine recommendation level
        if edge.edge >= 0.05 and confidence >= 0.8:
            recommendation = "STRONG_BET"
            reasoning = (
                f"High edge ({edge.edge:.1%}) with high confidence ({confidence:.1%})"
            )
        elif edge.edge >= 0.03 and confidence >= 0.7:
            recommendation = "BET"
            reasoning = (
                f"Good edge ({edge.edge:.1%}) with solid confidence ({confidence:.1%})"
            )
        elif edge.edge >= 0.01 and confidence >= 0.6:
            recommendation = "LEAN"
            reasoning = f"Small edge ({edge.edge:.1%}) with moderate confidence ({confidence:.1%})"
        else:
            recommendation = "PASS"
            reasoning = (
                f"Insufficient edge ({edge.edge:.1%}) or confidence ({confidence:.1%})"
            )

        return BetRecommendation(
            bet_type=edge.bet_type,
            recommendation=recommendation,
            edge=edge.edge,
            expected_value=edge.expected_value,
            kelly_fraction=edge.kelly_fraction,
            confidence=confidence,
            reasoning=reasoning,
        )


class NFLPredictionPipeline:
    """
    Unified NFL prediction pipeline combining WP, ATS, and O/U models.

    Models are loaded from versioned artifacts (via models.artifacts) and called
    with raw sklearn/xgboost predict methods. This pipeline orchestrates all three
    models to generate comprehensive game predictions with fair lines, edges, and
    betting recommendations.
    """

    def __init__(
        self,
        wp_artifact: dict[str, Any] | None = None,
        ats_artifact: dict[str, Any] | None = None,
        ou_artifact: dict[str, Any] | None = None,
        min_edge_threshold: float = 0.02,
        min_confidence_threshold: float = 0.6,
        max_kelly_fraction: float = 0.25,
    ):
        """
        Initialize the prediction pipeline.

        Args:
            wp_artifact: Loaded WP model artifact dict (from load_model_artifact)
            ats_artifact: Loaded ATS model artifact dict (from load_model_artifact)
            ou_artifact: Loaded O/U model artifact dict (from load_model_artifact)
            min_edge_threshold: Minimum edge for bet consideration
            min_confidence_threshold: Minimum confidence for recommendations
            max_kelly_fraction: Maximum Kelly fraction for bet sizing
        """
        self.wp_artifact = wp_artifact
        self.ats_artifact = ats_artifact
        self.ou_artifact = ou_artifact

        # Initialize utility classes
        self.odds_converter = OddsConverter()
        self.edge_calculator = EdgeCalculator(min_edge_threshold, max_kelly_fraction)
        self.recommendation_engine = BetRecommendationEngine(
            min_edge_threshold, min_confidence_threshold
        )

        # Configuration
        self.min_edge_threshold = min_edge_threshold
        self.min_confidence_threshold = min_confidence_threshold
        self.max_kelly_fraction = max_kelly_fraction

    def load_models(
        self,
        artifacts_dir: Path = Path("artifacts"),
        wp_version: str | None = None,
        ats_version: str | None = None,
        ou_version: str | None = None,
    ) -> None:
        """Load all three model artifacts from versioned artifact storage.

        Args:
            artifacts_dir: Root directory for model artifacts.
            wp_version: Specific WP artifact version (None = latest).
            ats_version: Specific ATS artifact version (None = latest).
            ou_version: Specific O/U artifact version (None = latest).
        """
        self.wp_artifact = load_model_artifact("wp", wp_version, artifacts_dir)
        logger.info("Loaded WP model artifact", version=wp_version or "latest")

        self.ats_artifact = load_model_artifact("ats", ats_version, artifacts_dir)
        logger.info("Loaded ATS model artifact", version=ats_version or "latest")

        self.ou_artifact = load_model_artifact("ou", ou_version, artifacts_dir)
        logger.info("Loaded O/U model artifact", version=ou_version or "latest")

    def _check_models_loaded(self) -> None:
        """Check that all required model artifacts are loaded."""
        missing = []
        if self.wp_artifact is None:
            missing.append("WP model")
        if self.ats_artifact is None:
            missing.append("ATS model")
        if self.ou_artifact is None:
            missing.append("O/U model")
        if missing:
            raise ValueError(f"Missing loaded models: {', '.join(missing)}")

    def _generate_fair_lines(
        self,
        wp_pred: WPPrediction,
        ats_pred: ATSPrediction,
        ou_pred: OUPrediction,
    ) -> dict[BetType, FairLine]:
        """Generate fair lines from model predictions."""
        fair_lines = {}

        # Moneyline fair lines
        wp_prob = wp_pred.calibrated_win_probability or wp_pred.raw_win_probability

        fair_lines[BetType.MONEYLINE_HOME] = FairLine(
            bet_type=BetType.MONEYLINE_HOME,
            fair_probability=wp_prob,
            fair_odds_american=self.odds_converter.probability_to_american(wp_prob),
            fair_odds_decimal=self.odds_converter.american_to_decimal(
                self.odds_converter.probability_to_american(wp_prob)
            ),
            confidence=wp_pred.prediction_confidence,
        )

        fair_lines[BetType.MONEYLINE_AWAY] = FairLine(
            bet_type=BetType.MONEYLINE_AWAY,
            fair_probability=1 - wp_prob,
            fair_odds_american=self.odds_converter.probability_to_american(
                1 - wp_prob
            ),
            fair_odds_decimal=self.odds_converter.american_to_decimal(
                self.odds_converter.probability_to_american(1 - wp_prob)
            ),
            confidence=wp_pred.prediction_confidence,
        )

        # Spread fair lines
        fair_lines[BetType.SPREAD_HOME] = FairLine(
            bet_type=BetType.SPREAD_HOME,
            fair_probability=ats_pred.cover_probability,
            fair_odds_american=self.odds_converter.probability_to_american(
                ats_pred.cover_probability
            ),
            fair_odds_decimal=self.odds_converter.american_to_decimal(
                self.odds_converter.probability_to_american(ats_pred.cover_probability)
            ),
            fair_line_value=ats_pred.predicted_spread,
            confidence=ats_pred.confidence,
        )

        fair_lines[BetType.SPREAD_AWAY] = FairLine(
            bet_type=BetType.SPREAD_AWAY,
            fair_probability=1 - ats_pred.cover_probability,
            fair_odds_american=self.odds_converter.probability_to_american(
                1 - ats_pred.cover_probability
            ),
            fair_odds_decimal=self.odds_converter.american_to_decimal(
                self.odds_converter.probability_to_american(
                    1 - ats_pred.cover_probability
                )
            ),
            fair_line_value=-ats_pred.predicted_spread,
            confidence=ats_pred.confidence,
        )

        # Over/Under fair lines
        fair_lines[BetType.OVER] = FairLine(
            bet_type=BetType.OVER,
            fair_probability=ou_pred.over_probability,
            fair_odds_american=self.odds_converter.probability_to_american(
                ou_pred.over_probability
            ),
            fair_odds_decimal=self.odds_converter.american_to_decimal(
                self.odds_converter.probability_to_american(ou_pred.over_probability)
            ),
            fair_line_value=ou_pred.predicted_total,
            confidence=ou_pred.confidence,
        )

        fair_lines[BetType.UNDER] = FairLine(
            bet_type=BetType.UNDER,
            fair_probability=ou_pred.under_probability,
            fair_odds_american=self.odds_converter.probability_to_american(
                ou_pred.under_probability
            ),
            fair_odds_decimal=self.odds_converter.american_to_decimal(
                self.odds_converter.probability_to_american(ou_pred.under_probability)
            ),
            fair_line_value=ou_pred.predicted_total,
            confidence=ou_pred.confidence,
        )

        return fair_lines

    def _calculate_edges(
        self,
        fair_lines: dict[BetType, FairLine],
        market_moneyline_home: int | None,
        market_moneyline_away: int | None,
        market_spread: float | None,
        market_total: float | None,
    ) -> dict[BetType, MarketEdge]:
        """Calculate edges vs market lines."""
        edges = {}

        # Moneyline edges
        if market_moneyline_home is not None:
            edge = self.edge_calculator.calculate_edge(
                fair_lines[BetType.MONEYLINE_HOME].fair_probability,
                market_moneyline_home,
            )
            edge.bet_type = BetType.MONEYLINE_HOME
            edges[BetType.MONEYLINE_HOME] = edge

        if market_moneyline_away is not None:
            edge = self.edge_calculator.calculate_edge(
                fair_lines[BetType.MONEYLINE_AWAY].fair_probability,
                market_moneyline_away,
            )
            edge.bet_type = BetType.MONEYLINE_AWAY
            edges[BetType.MONEYLINE_AWAY] = edge

        # Spread edges (assuming -110 juice)
        if market_spread is not None:
            spread_odds = -110

            edge = self.edge_calculator.calculate_edge(
                fair_lines[BetType.SPREAD_HOME].fair_probability, spread_odds
            )
            edge.bet_type = BetType.SPREAD_HOME
            edges[BetType.SPREAD_HOME] = edge

            edge = self.edge_calculator.calculate_edge(
                fair_lines[BetType.SPREAD_AWAY].fair_probability, spread_odds
            )
            edge.bet_type = BetType.SPREAD_AWAY
            edges[BetType.SPREAD_AWAY] = edge

        # Total edges (assuming -110 juice)
        if market_total is not None:
            total_odds = -110

            edge = self.edge_calculator.calculate_edge(
                fair_lines[BetType.OVER].fair_probability, total_odds
            )
            edge.bet_type = BetType.OVER
            edges[BetType.OVER] = edge

            edge = self.edge_calculator.calculate_edge(
                fair_lines[BetType.UNDER].fair_probability, total_odds
            )
            edge.bet_type = BetType.UNDER
            edges[BetType.UNDER] = edge

        return edges

    def _calculate_model_agreement(
        self,
        wp_pred: WPPrediction,
        ats_pred: ATSPrediction,
        ou_pred: OUPrediction,
    ) -> float:
        """Calculate agreement score between models."""
        agreements = []

        # WP vs ATS agreement (via implied win probability from margin)
        wp_home_prob = wp_pred.calibrated_win_probability or wp_pred.raw_win_probability

        # Rough conversion from margin to win probability
        if ats_pred.predicted_margin > 0:
            ats_implied_wp = 0.5 + min(abs(ats_pred.predicted_margin) / 40, 0.49)
        else:
            ats_implied_wp = 0.5 - min(abs(ats_pred.predicted_margin) / 40, 0.49)

        wp_ats_agreement = 1 - abs(wp_home_prob - ats_implied_wp)
        agreements.append(wp_ats_agreement)

        # Additional agreement metrics could be added here

        return float(np.mean(agreements))

    def _generate_recommendations(
        self, edges: dict[BetType, MarketEdge], fair_lines: dict[BetType, FairLine]
    ) -> list[BetRecommendation]:
        """Generate betting recommendations."""
        recommendations = []

        for bet_type, edge in edges.items():
            if edge.edge > self.min_edge_threshold:
                confidence = fair_lines[bet_type].confidence or 0.5
                recommendation = self.recommendation_engine.generate_recommendation(
                    edge, confidence
                )
                recommendations.append(recommendation)

        # Sort by edge descending
        recommendations.sort(key=lambda x: x.edge, reverse=True)

        return recommendations

    def predict_games(self, games_data: pd.DataFrame) -> list[UnifiedGamePrediction]:
        """
        Generate unified predictions for multiple games.

        Uses raw model.predict() / model.predict_proba() on feature DataFrames
        with feature columns specified by each artifact's feature_list.

        Args:
            games_data: DataFrame with game data, features, and market lines.
                Must contain all feature columns referenced by the loaded model
                artifacts, plus game_id, home_team, away_team, and optional
                market line columns (market_moneyline_home, market_moneyline_away,
                market_spread, market_total).

        Returns:
            List of unified game predictions.
        """
        self._check_models_loaded()

        logger.info(f"Generating predictions for {len(games_data)} games")

        # Extract models and feature lists from artifacts
        wp_model = self.wp_artifact["model"]
        wp_features = self.wp_artifact["feature_list"]
        wp_calibrator = self.wp_artifact.get("calibrator")

        ats_model = self.ats_artifact["model"]
        ats_features = self.ats_artifact["feature_list"]

        ou_model = self.ou_artifact["model"]
        ou_features = self.ou_artifact["feature_list"]

        # -- WP predictions: predict_proba for classification --
        wp_feature_df = games_data[wp_features].copy()
        wp_raw_probs = wp_model.predict_proba(wp_feature_df)[:, 1]

        # Apply calibrator if available
        wp_calibrated_probs = None
        if wp_calibrator is not None:
            wp_calibrated_probs = wp_calibrator.transform(wp_raw_probs)

        # -- ATS predictions: predict for regression --
        ats_feature_df = games_data[ats_features].copy()
        ats_margins = ats_model.predict(ats_feature_df)

        # Compute cover probabilities using ResidualDistributionConverter
        # Use market_spread column if available, otherwise use predicted margin as spread
        if "market_spread" in games_data.columns:
            spreads = games_data["market_spread"].values
        else:
            spreads = -ats_margins  # Default: predicted margin as implied spread

        # Build a simple converter with a normal distribution fallback
        # The residual_std is embedded in metadata if available, otherwise use a
        # reasonable default (NFL margin std ~13.5 points)
        ats_metadata = self.ats_artifact.get("metadata", {})
        ats_residual_std = ats_metadata.get("residual_std", 13.5)
        ats_converter = ResidualDistributionConverter(distribution_type="normal")
        ats_converter.is_fitted = True
        ats_converter.residual_std = ats_residual_std
        ats_converter.distribution_params = {"loc": 0.0, "scale": ats_residual_std}

        cover_probs = ats_converter.predict_cover_probability(
            np.array(ats_margins), np.array(spreads)
        )

        # -- O/U predictions: predict for regression --
        ou_feature_df = games_data[ou_features].copy()
        ou_totals = ou_model.predict(ou_feature_df)

        # Compute over/under probabilities using TotalDistributionConverter
        if "market_total" in games_data.columns:
            market_totals = games_data["market_total"].values
        else:
            market_totals = ou_totals  # Default: predicted total as market total

        ou_metadata = self.ou_artifact.get("metadata", {})
        ou_residual_std = ou_metadata.get("residual_std", 13.0)
        ou_converter = TotalDistributionConverter(distribution_type="normal")
        ou_converter.is_fitted = True
        ou_converter.residual_std = ou_residual_std
        ou_converter.distribution_params = {"loc": 0.0, "scale": ou_residual_std}

        over_probs, under_probs = ou_converter.predict_over_under_probabilities(
            np.array(ou_totals), np.array(market_totals)
        )

        # -- Build per-game predictions --
        unified_predictions = []

        for i in range(len(games_data)):
            game_data = games_data.iloc[i]

            # Build WP prediction
            raw_prob = float(wp_raw_probs[i])
            cal_prob = float(wp_calibrated_probs[i]) if wp_calibrated_probs is not None else None
            confidence = abs(raw_prob - 0.5) * 2  # Distance from 0.5 scaled to [0, 1]

            wp_pred = WPPrediction(
                game_id=str(game_data.get("game_id", f"game_{i}")),
                home_team=str(game_data.get("home_team", "UNK")),
                away_team=str(game_data.get("away_team", "UNK")),
                raw_win_probability=raw_prob,
                calibrated_win_probability=cal_prob,
                prediction_confidence=confidence,
            )

            # Build ATS prediction
            margin = float(ats_margins[i])
            cover_prob = float(cover_probs[i])
            spread_val = float(spreads[i]) if spreads is not None else -margin

            ats_pred = ATSPrediction(
                predicted_margin=margin,
                predicted_spread=spread_val,
                cover_probability=cover_prob,
                confidence=abs(cover_prob - 0.5) * 2,
            )

            # Build O/U prediction
            total = float(ou_totals[i])
            over_p = float(over_probs[i])
            under_p = float(under_probs[i])

            ou_pred = OUPrediction(
                predicted_total=total,
                over_probability=over_p,
                under_probability=under_p,
                confidence=abs(over_p - 0.5) * 2,
            )

            # Extract market lines
            market_moneyline_home = game_data.get("market_moneyline_home")
            market_moneyline_away = game_data.get("market_moneyline_away")
            market_spread = game_data.get("market_spread")
            market_total = game_data.get("market_total")

            # Generate fair lines
            fair_lines = self._generate_fair_lines(wp_pred, ats_pred, ou_pred)

            # Calculate edges
            edges = self._calculate_edges(
                fair_lines,
                market_moneyline_home,
                market_moneyline_away,
                market_spread,
                market_total,
            )

            # Generate recommendations
            recommendations = self._generate_recommendations(edges, fair_lines)

            # Calculate model agreement
            model_agreement = self._calculate_model_agreement(
                wp_pred, ats_pred, ou_pred
            )

            # Calculate overall confidence
            confidences = [
                wp_pred.prediction_confidence or 0.5,
                ats_pred.confidence or 0.5,
                ou_pred.confidence or 0.5,
            ]
            overall_confidence = float(np.mean(confidences))

            # Combine feature importances
            combined_features: dict[str, float] = {}
            if wp_pred.feature_importances:
                combined_features.update(wp_pred.feature_importances)
            if ats_pred.feature_importances:
                combined_features.update(ats_pred.feature_importances)
            if ou_pred.feature_importances:
                combined_features.update(ou_pred.feature_importances)

            # Build model version info from artifact metadata
            wp_meta = self.wp_artifact.get("metadata", {})
            ats_meta = self.ats_artifact.get("metadata", {})
            ou_meta = self.ou_artifact.get("metadata", {})

            # Create unified prediction
            unified_pred = UnifiedGamePrediction(
                game_id=wp_pred.game_id,
                home_team=wp_pred.home_team,
                away_team=wp_pred.away_team,
                prediction_date=datetime.now(),
                # Core predictions
                wp_home_probability=cal_prob if cal_prob is not None else raw_prob,
                wp_away_probability=1 - (cal_prob if cal_prob is not None else raw_prob),
                predicted_margin=ats_pred.predicted_margin,
                predicted_spread=ats_pred.predicted_spread,
                ats_cover_probability=ats_pred.cover_probability,
                predicted_total=ou_pred.predicted_total,
                over_probability=ou_pred.over_probability,
                under_probability=ou_pred.under_probability,
                # Market lines
                market_moneyline_home=market_moneyline_home,
                market_moneyline_away=market_moneyline_away,
                market_spread=market_spread,
                market_total=market_total,
                # Analysis
                fair_lines=fair_lines,
                edges=edges,
                recommendations=recommendations,
                # Diagnostics
                model_agreement=model_agreement,
                prediction_confidence=overall_confidence,
                feature_importance=combined_features,
                # Metadata
                model_versions={
                    "wp_model": wp_meta.get("version", "artifact"),
                    "ats_model": ats_meta.get("version", "artifact"),
                    "ou_model": ou_meta.get("version", "artifact"),
                },
            )

            unified_predictions.append(unified_pred)

        logger.info(f"Generated {len(unified_predictions)} unified predictions")
        return unified_predictions

    def export_predictions(
        self, predictions: list[UnifiedGamePrediction], format: str = "json"
    ) -> str | pd.DataFrame:
        """
        Export predictions to various formats.

        Args:
            predictions: List of predictions to export
            format: Export format ("json", "csv", "dataframe")

        Returns:
            Exported data in requested format
        """
        if format == "dataframe":
            data = []
            for pred in predictions:
                row = {
                    "game_id": pred.game_id,
                    "home_team": pred.home_team,
                    "away_team": pred.away_team,
                    "prediction_date": pred.prediction_date,
                    # Probabilities
                    "wp_home_probability": pred.wp_home_probability,
                    "predicted_margin": pred.predicted_margin,
                    "ats_cover_probability": pred.ats_cover_probability,
                    "predicted_total": pred.predicted_total,
                    "over_probability": pred.over_probability,
                    # Market lines
                    "market_spread": pred.market_spread,
                    "market_total": pred.market_total,
                    # Diagnostics
                    "model_agreement": pred.model_agreement,
                    "prediction_confidence": pred.prediction_confidence,
                    # Best recommendation
                    "best_recommendation": pred.recommendations[0].bet_type.value
                    if pred.recommendations
                    else None,
                    "best_edge": pred.recommendations[0].edge
                    if pred.recommendations
                    else None,
                }
                data.append(row)

            return pd.DataFrame(data)

        if format == "json":
            # Convert to JSON-serializable format
            json_data = []
            for pred in predictions:
                pred_dict = {
                    "game_id": pred.game_id,
                    "home_team": pred.home_team,
                    "away_team": pred.away_team,
                    "prediction_date": pred.prediction_date.isoformat(),
                    "predictions": {
                        "wp_home_probability": pred.wp_home_probability,
                        "predicted_margin": pred.predicted_margin,
                        "ats_cover_probability": pred.ats_cover_probability,
                        "predicted_total": pred.predicted_total,
                        "over_probability": pred.over_probability,
                    },
                    "market_lines": {
                        "moneyline_home": pred.market_moneyline_home,
                        "moneyline_away": pred.market_moneyline_away,
                        "spread": pred.market_spread,
                        "total": pred.market_total,
                    },
                    "fair_lines": {
                        bet_type.value: {
                            "probability": line.fair_probability,
                            "american_odds": line.fair_odds_american,
                        }
                        for bet_type, line in pred.fair_lines.items()
                    },
                    "edges": {
                        bet_type.value: {
                            "edge": edge.edge,
                            "expected_value": edge.expected_value,
                            "confidence": edge.confidence_level,
                        }
                        for bet_type, edge in pred.edges.items()
                    },
                    "recommendations": [
                        {
                            "bet_type": rec.bet_type.value,
                            "recommendation": rec.recommendation,
                            "edge": rec.edge,
                            "kelly_fraction": rec.kelly_fraction,
                            "reasoning": rec.reasoning,
                        }
                        for rec in pred.recommendations
                    ],
                    "diagnostics": {
                        "model_agreement": pred.model_agreement,
                        "prediction_confidence": pred.prediction_confidence,
                    },
                }
                json_data.append(pred_dict)

            return json.dumps(json_data, indent=2)

        raise ValueError(f"Unsupported export format: {format}")

    def get_pipeline_summary(self) -> dict[str, Any]:
        """Get a summary of the pipeline configuration."""
        return {
            "models_loaded": {
                "wp_model": self.wp_artifact is not None,
                "ats_model": self.ats_artifact is not None,
                "ou_model": self.ou_artifact is not None,
            },
            "configuration": {
                "min_edge_threshold": self.min_edge_threshold,
                "min_confidence_threshold": self.min_confidence_threshold,
                "max_kelly_fraction": self.max_kelly_fraction,
            },
            "supported_bet_types": [bet_type.value for bet_type in BetType],
            "pipeline_version": "2.0.0",
        }


def main():
    """Main function for testing pipeline functionality."""
    logger.info("NFL Prediction Pipeline module loaded successfully")


if __name__ == "__main__":
    main()
