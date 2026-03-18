"""
Closing Line Value (CLV) tracking for betting simulation.

This module implements CLV calculation and tracking to measure the quality
of betting decisions against closing lines, which is considered the most
accurate measure of betting skill.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

import numpy as np

from utils.logging_config import get_logger
from utils.probability_utils import (
    edge_calculation,
    normalize_odds,
)

logger = get_logger(__name__)


class CLVMetric(Enum):
    """Types of CLV metrics."""

    ABSOLUTE_CLV = "absolute"  # Raw difference in odds/probabilities
    PERCENTAGE_CLV = "percentage"  # Percentage difference
    EXPECTED_VALUE_CLV = "expected_value"  # Difference in expected value


class BetOutcome(Enum):
    """Possible bet outcomes."""

    WIN = "win"
    LOSS = "loss"
    PUSH = "push"  # Tie - stake returned


@dataclass
class BetRecord:
    """Record of a single bet for CLV analysis."""

    game_id: str
    bet_type: str
    bet_side: str
    bet_timestamp: datetime

    # Our bet details
    our_odds: int  # American odds when we bet
    our_probability: float  # Implied probability when we bet
    our_decimal_odds: float  # Decimal odds when we bet
    bet_amount: float

    # Model edge details (for edge drift analysis)
    initial_model_probability: float | None = None  # Our model's probability
    initial_edge: float | None = None  # Edge when bet was placed

    # Closing line details
    closing_odds: int | None = None
    closing_probability: float | None = None
    closing_decimal_odds: float | None = None
    closing_timestamp: datetime | None = None
    closing_edge: float | None = None  # Edge at closing

    # Actual outcome (now supports pushes)
    outcome: BetOutcome | None = None
    payout: float | None = None

    # CLV metrics (calculated)
    absolute_clv: float | None = None
    percentage_clv: float | None = None
    ev_clv: float | None = None
    edge_drift: float | None = None  # How edge changed from initial to closing


@dataclass
class CLVSummary:
    """Summary of CLV performance."""

    # Required fields (no defaults) - must come first
    total_bets: int
    bets_with_clv_data: int

    # CLV metrics
    average_absolute_clv: float
    average_percentage_clv: float
    positive_clv_rate: float

    # Expected value impact
    total_ev_from_clv: float
    average_ev_clv: float

    # Performance by CLV buckets
    clv_bucket_performance: dict[str, dict[str, float]]

    # Push handling
    total_wins: int
    total_losses: int
    total_pushes: int
    push_rate: float

    # Time period (required fields)
    start_date: datetime
    end_date: datetime

    # Optional fields (with defaults) - must come last
    # Edge drift analysis
    average_initial_edge: float | None = None
    average_closing_edge: float | None = None
    average_edge_drift: float | None = None
    edge_persistence_rate: float | None = None  # % of bets where edge persisted

    # Statistical significance
    clv_t_statistic: float | None = None
    clv_p_value: float | None = None


class CLVTracker:
    """
    Tracks and analyzes Closing Line Value (CLV) for betting performance.

    CLV measures how much better or worse our betting lines were compared
    to the closing lines, which are considered the most efficient predictor.
    """

    def __init__(self):
        """Initialize CLV tracker."""
        self.bet_records: list[BetRecord] = []
        self.closing_lines: dict[str, dict] = {}  # game_id -> closing line data

        logger.info("CLV Tracker initialized")

    def record_bet(
        self,
        game_id: str,
        bet_type: str,
        bet_side: str,
        our_odds: int,
        bet_amount: float,
        bet_timestamp: datetime | None = None,
        model_probability: float | None = None,
    ) -> str:
        """
        Record a bet for CLV tracking.

        Args:
            game_id: Unique identifier for the game
            bet_type: Type of bet (moneyline, spread, total)
            bet_side: Which side of bet (home, away, over, under)
            our_odds: American odds when we placed the bet
            bet_amount: Amount wagered
            bet_timestamp: When the bet was placed
            model_probability: Our model's probability estimate (for edge drift)

        Returns:
            Bet record ID for reference
        """

        if bet_timestamp is None:
            bet_timestamp = datetime.now()

        # Use centralized normalization for consistency
        our_probability, our_decimal_odds, our_odds_normalized = normalize_odds(
            our_odds, "american"
        )

        # Calculate initial edge if model probability provided
        initial_edge = None
        if model_probability is not None:
            initial_edge = edge_calculation(
                model_probability, our_probability, "additive"
            )

        bet_record = BetRecord(
            game_id=game_id,
            bet_type=bet_type,
            bet_side=bet_side,
            bet_timestamp=bet_timestamp,
            our_odds=our_odds_normalized,
            our_probability=our_probability,
            our_decimal_odds=our_decimal_odds,
            bet_amount=bet_amount,
            initial_model_probability=model_probability,
            initial_edge=initial_edge,
        )

        self.bet_records.append(bet_record)
        bet_id = (
            f"{game_id}_{bet_type}_{bet_side}_{bet_timestamp.strftime('%Y%m%d_%H%M%S')}"
        )

        logger.debug(f"Recorded bet: {bet_id} at {our_odds}")
        return bet_id

    def update_closing_lines(self, closing_lines_data: dict[str, dict]):
        """
        Update closing line data for games.

        Args:
            closing_lines_data: Dict mapping game_id to closing line info
                Expected format:
                {
                    "game_id": {
                        "moneyline_home": -120,
                        "moneyline_away": +105,
                        "spread_home": -110,
                        "spread_away": -110,
                        "total_over": -110,
                        "total_under": -110,
                        "closing_timestamp": datetime
                    }
                }
        """

        self.closing_lines.update(closing_lines_data)
        logger.info(f"Updated closing lines for {len(closing_lines_data)} games")

        # Calculate CLV for existing bets
        self._calculate_clv_for_all_bets()

    def record_bet_outcome(
        self,
        game_id: str,
        bet_type: str,
        bet_side: str,
        outcome: bool | BetOutcome | str,
        actual_value: float | None = None,
        line_value: float | None = None,
    ):
        """
        Record the outcome of a bet with push handling.

        Args:
            game_id: Game identifier
            bet_type: Type of bet
            bet_side: Side of bet
            outcome: Bet outcome (bool, BetOutcome, or string)
            actual_value: Actual game value (for push detection)
            line_value: Line value (for push detection)
        """

        # Convert outcome to BetOutcome enum
        if isinstance(outcome, bool):
            bet_outcome = BetOutcome.WIN if outcome else BetOutcome.LOSS
        elif isinstance(outcome, str):
            bet_outcome = BetOutcome(outcome.lower())
        else:
            bet_outcome = outcome

        # Detect pushes for ATS/O/U if values provided
        if (
            actual_value is not None
            and line_value is not None
            and bet_type in ["spread", "total"]
            and bet_outcome != BetOutcome.PUSH
        ) and abs(actual_value - line_value) < 0.1:  # Within 0.1 points = push
            bet_outcome = BetOutcome.PUSH

        # Find matching bet record
        for bet_record in self.bet_records:
            if (
                bet_record.game_id == game_id
                and bet_record.bet_type == bet_type
                and bet_record.bet_side == bet_side
            ):
                bet_record.outcome = bet_outcome

                # Calculate payout based on outcome
                if bet_outcome == BetOutcome.WIN:
                    # Use decimal odds for cleaner calculation
                    bet_record.payout = bet_record.bet_amount * (
                        bet_record.our_decimal_odds - 1
                    )
                elif bet_outcome == BetOutcome.LOSS:
                    bet_record.payout = -bet_record.bet_amount
                else:  # PUSH
                    bet_record.payout = 0.0  # Stake returned

                logger.debug(
                    f"Recorded outcome for {game_id} {bet_type} {bet_side}: {bet_outcome.value.upper()}"
                )
                return

        logger.warning(
            f"No matching bet record found for {game_id} {bet_type} {bet_side}"
        )

    def _calculate_clv_for_all_bets(self):
        """Calculate CLV for all bets that have closing line data."""

        for bet_record in self.bet_records:
            if bet_record.game_id not in self.closing_lines:
                continue

            closing_data = self.closing_lines[bet_record.game_id]
            self._calculate_clv_for_bet(bet_record, closing_data)

    def _calculate_clv_for_bet(self, bet_record: BetRecord, closing_data: dict):
        """Calculate CLV metrics for a single bet."""

        # Get the appropriate closing line
        closing_odds = self._get_closing_odds_for_bet(bet_record, closing_data)

        if closing_odds is None:
            logger.debug(
                f"No closing odds available for {bet_record.game_id} {bet_record.bet_type} {bet_record.bet_side}"
            )
            return

        # Use centralized normalization for consistency
        closing_probability, closing_decimal_odds, closing_odds_normalized = (
            normalize_odds(closing_odds, "american")
        )

        bet_record.closing_odds = closing_odds_normalized
        bet_record.closing_probability = closing_probability
        bet_record.closing_decimal_odds = closing_decimal_odds
        bet_record.closing_timestamp = closing_data.get("closing_timestamp")

        # Calculate CLV metrics

        # 1. Absolute CLV (difference in implied probability)
        bet_record.absolute_clv = (
            bet_record.closing_probability - bet_record.our_probability
        )

        # 2. Percentage CLV
        if bet_record.our_probability > 0:
            bet_record.percentage_clv = (
                bet_record.closing_probability - bet_record.our_probability
            ) / bet_record.our_probability

        # 3. Expected Value CLV
        # Use closing line as "true" probability for EV calculation
        true_prob = bet_record.closing_probability
        our_ev = (true_prob * (bet_record.our_decimal_odds - 1)) - (1 - true_prob)
        closing_ev = (true_prob * (closing_decimal_odds - 1)) - (1 - true_prob)
        bet_record.ev_clv = our_ev - closing_ev

        # 4. Edge drift calculation
        if bet_record.initial_model_probability is not None:
            bet_record.closing_edge = edge_calculation(
                bet_record.initial_model_probability, closing_probability, "additive"
            )
            if bet_record.initial_edge is not None:
                bet_record.edge_drift = (
                    bet_record.closing_edge - bet_record.initial_edge
                )

        logger.debug(
            f"CLV calculated for {bet_record.game_id}: Absolute={bet_record.absolute_clv:.4f}, "
            f"Percentage={bet_record.percentage_clv:.4f}, EV={bet_record.ev_clv:.4f}, "
            f"Edge drift={bet_record.edge_drift:.4f if bet_record.edge_drift else 'N/A'}"
        )

    def _get_closing_odds_for_bet(
        self, bet_record: BetRecord, closing_data: dict
    ) -> int | None:
        """Get the appropriate closing odds for a bet."""

        if bet_record.bet_type == "moneyline":
            if bet_record.bet_side == "home":
                return closing_data.get("moneyline_home")
            if bet_record.bet_side == "away":
                return closing_data.get("moneyline_away")

        elif bet_record.bet_type == "spread":
            if bet_record.bet_side == "home":
                return closing_data.get("spread_home")
            if bet_record.bet_side == "away":
                return closing_data.get("spread_away")

        elif bet_record.bet_type == "total":
            if bet_record.bet_side == "over":
                return closing_data.get("total_over")
            if bet_record.bet_side == "under":
                return closing_data.get("total_under")

        return None

    def calculate_clv_summary(
        self, start_date: datetime | None = None, end_date: datetime | None = None
    ) -> CLVSummary:
        """
        Calculate comprehensive CLV summary statistics.

        Args:
            start_date: Start date for analysis (optional)
            end_date: End date for analysis (optional)

        Returns:
            CLVSummary with comprehensive CLV metrics
        """

        # Filter bets by date range
        filtered_bets = self.bet_records
        if start_date:
            filtered_bets = [b for b in filtered_bets if b.bet_timestamp >= start_date]
        if end_date:
            filtered_bets = [b for b in filtered_bets if b.bet_timestamp <= end_date]

        # Filter bets with CLV data
        clv_bets = [b for b in filtered_bets if b.absolute_clv is not None]

        # Count outcomes (including pushes)
        wins = sum(1 for b in filtered_bets if b.outcome == BetOutcome.WIN)
        losses = sum(1 for b in filtered_bets if b.outcome == BetOutcome.LOSS)
        pushes = sum(1 for b in filtered_bets if b.outcome == BetOutcome.PUSH)
        push_rate = pushes / len(filtered_bets) if filtered_bets else 0.0

        if not clv_bets:
            logger.warning("No bets with CLV data found")
            return CLVSummary(
                total_bets=len(filtered_bets),
                bets_with_clv_data=0,
                average_absolute_clv=0.0,
                average_percentage_clv=0.0,
                positive_clv_rate=0.0,
                total_ev_from_clv=0.0,
                average_ev_clv=0.0,
                clv_bucket_performance={},
                total_wins=wins,
                total_losses=losses,
                total_pushes=pushes,
                push_rate=push_rate,
                start_date=start_date or min(b.bet_timestamp for b in filtered_bets),
                end_date=end_date or max(b.bet_timestamp for b in filtered_bets),
            )

        # Calculate basic CLV metrics
        absolute_clvs = [b.absolute_clv for b in clv_bets]
        percentage_clvs = [
            b.percentage_clv for b in clv_bets if b.percentage_clv is not None
        ]
        ev_clvs = [b.ev_clv for b in clv_bets if b.ev_clv is not None]

        average_absolute_clv = np.mean(absolute_clvs)
        average_percentage_clv = np.mean(percentage_clvs) if percentage_clvs else 0.0
        positive_clv_rate = sum(1 for clv in absolute_clvs if clv > 0) / len(
            absolute_clvs
        )

        total_ev_from_clv = sum(ev_clvs) if ev_clvs else 0.0
        average_ev_clv = np.mean(ev_clvs) if ev_clvs else 0.0

        # CLV bucket performance
        clv_bucket_performance = self._calculate_bucket_performance(clv_bets)

        # Edge drift analysis
        edge_drift_bets = [b for b in clv_bets if b.edge_drift is not None]
        average_initial_edge = (
            np.mean(
                [b.initial_edge for b in edge_drift_bets if b.initial_edge is not None]
            )
            if edge_drift_bets
            else None
        )
        average_closing_edge = (
            np.mean(
                [b.closing_edge for b in edge_drift_bets if b.closing_edge is not None]
            )
            if edge_drift_bets
            else None
        )
        average_edge_drift = (
            np.mean([b.edge_drift for b in edge_drift_bets])
            if edge_drift_bets
            else None
        )

        # Edge persistence rate (bets where closing edge remained positive)
        edge_persistence_rate = None
        if edge_drift_bets:
            positive_initial = [
                b for b in edge_drift_bets if b.initial_edge and b.initial_edge > 0
            ]
            if positive_initial:
                still_positive = [
                    b for b in positive_initial if b.closing_edge and b.closing_edge > 0
                ]
                edge_persistence_rate = len(still_positive) / len(positive_initial)

        # Statistical significance test
        clv_t_stat, clv_p_value = self._calculate_clv_significance(absolute_clvs)

        return CLVSummary(
            total_bets=len(filtered_bets),
            bets_with_clv_data=len(clv_bets),
            average_absolute_clv=average_absolute_clv,
            average_percentage_clv=average_percentage_clv,
            positive_clv_rate=positive_clv_rate,
            total_ev_from_clv=total_ev_from_clv,
            average_ev_clv=average_ev_clv,
            clv_bucket_performance=clv_bucket_performance,
            total_wins=wins,
            total_losses=losses,
            total_pushes=pushes,
            push_rate=push_rate,
            average_initial_edge=average_initial_edge,
            average_closing_edge=average_closing_edge,
            average_edge_drift=average_edge_drift,
            edge_persistence_rate=edge_persistence_rate,
            start_date=start_date or min(b.bet_timestamp for b in filtered_bets),
            end_date=end_date or max(b.bet_timestamp for b in filtered_bets),
            clv_t_statistic=clv_t_stat,
            clv_p_value=clv_p_value,
        )

    def _calculate_bucket_performance(
        self, clv_bets: list[BetRecord]
    ) -> dict[str, dict[str, float]]:
        """Calculate performance metrics by CLV buckets."""

        # Define CLV buckets
        buckets = {
            "Strong Negative CLV": (-float("inf"), -0.05),
            "Negative CLV": (-0.05, -0.01),
            "Neutral CLV": (-0.01, 0.01),
            "Positive CLV": (0.01, 0.05),
            "Strong Positive CLV": (0.05, float("inf")),
        }

        bucket_performance = {}

        for bucket_name, (min_clv, max_clv) in buckets.items():
            bucket_bets = [
                b
                for b in clv_bets
                if min_clv <= b.absolute_clv < max_clv and b.outcome is not None
            ]

            if bucket_bets:
                wins = sum(1 for b in bucket_bets if b.outcome == BetOutcome.WIN)
                losses = sum(1 for b in bucket_bets if b.outcome == BetOutcome.LOSS)
                pushes = sum(1 for b in bucket_bets if b.outcome == BetOutcome.PUSH)
                total_payout = sum(
                    b.payout for b in bucket_bets if b.payout is not None
                )
                total_wagered = sum(b.bet_amount for b in bucket_bets)

                bucket_performance[bucket_name] = {
                    "count": len(bucket_bets),
                    "wins": wins,
                    "losses": losses,
                    "pushes": pushes,
                    "win_rate": wins / len(bucket_bets),
                    "push_rate": pushes / len(bucket_bets),
                    "roi": total_payout / total_wagered if total_wagered > 0 else 0.0,
                    "average_clv": np.mean([b.absolute_clv for b in bucket_bets]),
                }

        return bucket_performance

    def _calculate_clv_significance(
        self, clv_values: list[float]
    ) -> tuple[float | None, float | None]:
        """Calculate statistical significance of CLV."""

        if len(clv_values) < 10:  # Need reasonable sample size
            return None, None

        try:
            from scipy import stats

            # One-sample t-test against zero (no CLV)
            t_stat, p_value = stats.ttest_1samp(clv_values, 0)
            return t_stat, p_value
        except ImportError:
            logger.warning("scipy not available for CLV significance testing")
            return None, None

    def export_clv_analysis(self, filepath: str, summary: CLVSummary | None = None):
        """Export CLV analysis to JSON file."""

        if summary is None:
            summary = self.calculate_clv_summary()

        export_data = {
            "clv_summary": {
                "total_bets": summary.total_bets,
                "bets_with_clv_data": summary.bets_with_clv_data,
                "average_absolute_clv": summary.average_absolute_clv,
                "average_percentage_clv": summary.average_percentage_clv,
                "positive_clv_rate": summary.positive_clv_rate,
                "total_ev_from_clv": summary.total_ev_from_clv,
                "average_ev_clv": summary.average_ev_clv,
                "start_date": summary.start_date.isoformat(),
                "end_date": summary.end_date.isoformat(),
                "clv_t_statistic": summary.clv_t_statistic,
                "clv_p_value": summary.clv_p_value,
            },
            "bucket_performance": summary.clv_bucket_performance,
            "bet_details": [],
        }

        # Add individual bet details
        for bet in self.bet_records:
            if bet.absolute_clv is not None:
                export_data["bet_details"].append(
                    {
                        "game_id": bet.game_id,
                        "bet_type": bet.bet_type,
                        "bet_side": bet.bet_side,
                        "bet_timestamp": bet.bet_timestamp.isoformat(),
                        "our_odds": bet.our_odds,
                        "closing_odds": bet.closing_odds,
                        "absolute_clv": bet.absolute_clv,
                        "percentage_clv": bet.percentage_clv,
                        "ev_clv": bet.ev_clv,
                        "outcome": bet.outcome.value if bet.outcome else None,
                        "payout": bet.payout,
                        "initial_edge": bet.initial_edge,
                        "closing_edge": bet.closing_edge,
                        "edge_drift": bet.edge_drift,
                    }
                )

        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"CLV analysis exported to {filepath}")

    def get_clv_insights(self, summary: CLVSummary | None = None) -> dict[str, str]:
        """Generate human-readable insights from CLV analysis."""

        if summary is None:
            summary = self.calculate_clv_summary()

        insights = []

        # Overall CLV assessment
        if summary.average_absolute_clv > 0.02:
            insights.append(
                "🟢 Excellent CLV: You're consistently getting better lines than the closing market"
            )
        elif summary.average_absolute_clv > 0.005:
            insights.append(
                "🟡 Positive CLV: You're getting slightly better lines than the market"
            )
        elif summary.average_absolute_clv > -0.005:
            insights.append(
                "🟡 Neutral CLV: Your lines are roughly in line with the market"
            )
        else:
            insights.append(
                "🔴 Negative CLV: You're getting worse lines than the closing market"
            )

        # Positive CLV rate
        if summary.positive_clv_rate > 0.6:
            insights.append(
                f"🟢 High positive CLV rate: {summary.positive_clv_rate:.1%} of bets beat the closing line"
            )
        elif summary.positive_clv_rate > 0.4:
            insights.append(
                f"🟡 Moderate positive CLV rate: {summary.positive_clv_rate:.1%} of bets beat the closing line"
            )
        else:
            insights.append(
                f"🔴 Low positive CLV rate: Only {summary.positive_clv_rate:.1%} of bets beat the closing line"
            )

        # Statistical significance
        if summary.clv_p_value and summary.clv_p_value < 0.05:
            insights.append("📊 Your CLV is statistically significant (p < 0.05)")
        elif summary.clv_p_value:
            insights.append(
                "📊 Your CLV is not statistically significant - need more data or better line shopping"
            )

        # Best performing buckets
        if summary.clv_bucket_performance:
            best_bucket = max(
                summary.clv_bucket_performance.items(), key=lambda x: x[1]["roi"]
            )
            insights.append(
                f"🎯 Best performing CLV bucket: {best_bucket[0]} with {best_bucket[1]['roi']:.2%} ROI"
            )

        return {
            "overall_assessment": insights[0] if insights else "No data available",
            "detailed_insights": insights[1:] if len(insights) > 1 else [],
            "recommendation": self._get_clv_recommendation(summary),
        }

    def _get_clv_recommendation(self, summary: CLVSummary) -> str:
        """Get recommendation based on CLV performance."""

        recommendations = []

        # CLV performance
        if summary.average_absolute_clv > 0.01 and summary.positive_clv_rate > 0.55:
            recommendations.append(
                "Continue current strategy - you're beating the market consistently"
            )
        elif summary.average_absolute_clv > 0:
            recommendations.append(
                "Good CLV but can improve - consider line shopping and bet timing optimization"
            )
        else:
            recommendations.append(
                "Focus on line shopping and bet timing - you're not beating closing lines consistently"
            )

        # Edge drift insights
        if summary.edge_persistence_rate is not None:
            if summary.edge_persistence_rate > 0.7:
                recommendations.append(
                    "Strong edge persistence - your model identifies lasting advantages"
                )
            elif summary.edge_persistence_rate < 0.3:
                recommendations.append(
                    "Low edge persistence - market quickly corrects your identified edges"
                )

        # Push rate insights
        if summary.push_rate > 0.1:
            recommendations.append(
                f"High push rate ({summary.push_rate:.1%}) - consider line shopping for better numbers"
            )

        return "; ".join(recommendations)
