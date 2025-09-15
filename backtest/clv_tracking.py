"""
Closing Line Value (CLV) tracking for betting simulation.

This module implements CLV calculation and tracking to measure the quality
of betting decisions against closing lines, which is considered the most
accurate measure of betting skill.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple, Any
import pandas as pd
import numpy as np
from datetime import datetime
import json

from utils.logging_config import get_logger
from utils.probability_utils import moneyline_to_probability

logger = get_logger(__name__)


class CLVMetric(Enum):
    """Types of CLV metrics."""
    ABSOLUTE_CLV = "absolute"  # Raw difference in odds/probabilities
    PERCENTAGE_CLV = "percentage"  # Percentage difference
    EXPECTED_VALUE_CLV = "expected_value"  # Difference in expected value


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
    bet_amount: float

    # Closing line details
    closing_odds: Optional[int] = None
    closing_probability: Optional[float] = None
    closing_timestamp: Optional[datetime] = None

    # Actual outcome
    won: Optional[bool] = None
    payout: Optional[float] = None

    # CLV metrics (calculated)
    absolute_clv: Optional[float] = None
    percentage_clv: Optional[float] = None
    ev_clv: Optional[float] = None


@dataclass
class CLVSummary:
    """Summary of CLV performance."""

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
    clv_bucket_performance: Dict[str, Dict[str, float]]

    # Time period
    start_date: datetime
    end_date: datetime

    # Statistical significance
    clv_t_statistic: Optional[float] = None
    clv_p_value: Optional[float] = None


class CLVTracker:
    """
    Tracks and analyzes Closing Line Value (CLV) for betting performance.

    CLV measures how much better or worse our betting lines were compared
    to the closing lines, which are considered the most efficient predictor.
    """

    def __init__(self):
        """Initialize CLV tracker."""
        self.bet_records: List[BetRecord] = []
        self.closing_lines: Dict[str, Dict] = {}  # game_id -> closing line data

        logger.info("CLV Tracker initialized")

    def record_bet(
        self,
        game_id: str,
        bet_type: str,
        bet_side: str,
        our_odds: int,
        bet_amount: float,
        bet_timestamp: datetime = None
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

        Returns:
            Bet record ID for reference
        """

        if bet_timestamp is None:
            bet_timestamp = datetime.now()

        our_probability = moneyline_to_probability(our_odds)

        bet_record = BetRecord(
            game_id=game_id,
            bet_type=bet_type,
            bet_side=bet_side,
            bet_timestamp=bet_timestamp,
            our_odds=our_odds,
            our_probability=our_probability,
            bet_amount=bet_amount
        )

        self.bet_records.append(bet_record)
        bet_id = f"{game_id}_{bet_type}_{bet_side}_{bet_timestamp.strftime('%Y%m%d_%H%M%S')}"

        logger.debug(f"Recorded bet: {bet_id} at {our_odds}")
        return bet_id

    def update_closing_lines(self, closing_lines_data: Dict[str, Dict]):
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

    def record_bet_outcome(self, game_id: str, bet_type: str, bet_side: str, won: bool):
        """
        Record the outcome of a bet.

        Args:
            game_id: Game identifier
            bet_type: Type of bet
            bet_side: Side of bet
            won: Whether the bet won
        """

        # Find matching bet record
        for bet_record in self.bet_records:
            if (bet_record.game_id == game_id and
                bet_record.bet_type == bet_type and
                bet_record.bet_side == bet_side):

                bet_record.won = won

                # Calculate payout
                if won:
                    if bet_record.our_odds > 0:
                        bet_record.payout = bet_record.bet_amount * (bet_record.our_odds / 100)
                    else:
                        bet_record.payout = bet_record.bet_amount * (100 / abs(bet_record.our_odds))
                else:
                    bet_record.payout = -bet_record.bet_amount

                logger.debug(f"Recorded outcome for {game_id} {bet_type} {bet_side}: {'WIN' if won else 'LOSS'}")
                return

        logger.warning(f"No matching bet record found for {game_id} {bet_type} {bet_side}")

    def _calculate_clv_for_all_bets(self):
        """Calculate CLV for all bets that have closing line data."""

        for bet_record in self.bet_records:
            if bet_record.game_id not in self.closing_lines:
                continue

            closing_data = self.closing_lines[bet_record.game_id]
            self._calculate_clv_for_bet(bet_record, closing_data)

    def _calculate_clv_for_bet(self, bet_record: BetRecord, closing_data: Dict):
        """Calculate CLV metrics for a single bet."""

        # Get the appropriate closing line
        closing_odds = self._get_closing_odds_for_bet(bet_record, closing_data)

        if closing_odds is None:
            logger.debug(f"No closing odds available for {bet_record.game_id} {bet_record.bet_type} {bet_record.bet_side}")
            return

        bet_record.closing_odds = closing_odds
        bet_record.closing_probability = moneyline_to_probability(closing_odds)
        bet_record.closing_timestamp = closing_data.get('closing_timestamp')

        # Calculate CLV metrics

        # 1. Absolute CLV (difference in implied probability)
        bet_record.absolute_clv = bet_record.closing_probability - bet_record.our_probability

        # 2. Percentage CLV
        if bet_record.our_probability > 0:
            bet_record.percentage_clv = (bet_record.closing_probability - bet_record.our_probability) / bet_record.our_probability

        # 3. Expected Value CLV
        # EV difference based on odds difference
        our_decimal = american_to_decimal(bet_record.our_odds)
        closing_decimal = american_to_decimal(closing_odds)

        # Assuming true probability is somewhere between our line and closing line
        # Use closing line as "true" probability for EV calculation
        true_prob = bet_record.closing_probability

        our_ev = (true_prob * (our_decimal - 1)) - (1 - true_prob)
        closing_ev = (true_prob * (closing_decimal - 1)) - (1 - true_prob)
        bet_record.ev_clv = our_ev - closing_ev

        logger.debug(f"CLV calculated for {bet_record.game_id}: Absolute={bet_record.absolute_clv:.4f}, "
                    f"Percentage={bet_record.percentage_clv:.4f}, EV={bet_record.ev_clv:.4f}")

    def _get_closing_odds_for_bet(self, bet_record: BetRecord, closing_data: Dict) -> Optional[int]:
        """Get the appropriate closing odds for a bet."""

        if bet_record.bet_type == "moneyline":
            if bet_record.bet_side == "home":
                return closing_data.get("moneyline_home")
            elif bet_record.bet_side == "away":
                return closing_data.get("moneyline_away")

        elif bet_record.bet_type == "spread":
            if bet_record.bet_side == "home":
                return closing_data.get("spread_home")
            elif bet_record.bet_side == "away":
                return closing_data.get("spread_away")

        elif bet_record.bet_type == "total":
            if bet_record.bet_side == "over":
                return closing_data.get("total_over")
            elif bet_record.bet_side == "under":
                return closing_data.get("total_under")

        return None

    def calculate_clv_summary(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
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
                start_date=start_date or min(b.bet_timestamp for b in filtered_bets),
                end_date=end_date or max(b.bet_timestamp for b in filtered_bets)
            )

        # Calculate basic CLV metrics
        absolute_clvs = [b.absolute_clv for b in clv_bets]
        percentage_clvs = [b.percentage_clv for b in clv_bets if b.percentage_clv is not None]
        ev_clvs = [b.ev_clv for b in clv_bets if b.ev_clv is not None]

        average_absolute_clv = np.mean(absolute_clvs)
        average_percentage_clv = np.mean(percentage_clvs) if percentage_clvs else 0.0
        positive_clv_rate = sum(1 for clv in absolute_clvs if clv > 0) / len(absolute_clvs)

        total_ev_from_clv = sum(ev_clvs) if ev_clvs else 0.0
        average_ev_clv = np.mean(ev_clvs) if ev_clvs else 0.0

        # CLV bucket performance
        clv_bucket_performance = self._calculate_bucket_performance(clv_bets)

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
            start_date=start_date or min(b.bet_timestamp for b in filtered_bets),
            end_date=end_date or max(b.bet_timestamp for b in filtered_bets),
            clv_t_statistic=clv_t_stat,
            clv_p_value=clv_p_value
        )

    def _calculate_bucket_performance(self, clv_bets: List[BetRecord]) -> Dict[str, Dict[str, float]]:
        """Calculate performance metrics by CLV buckets."""

        # Define CLV buckets
        buckets = {
            "Strong Negative CLV": (-float('inf'), -0.05),
            "Negative CLV": (-0.05, -0.01),
            "Neutral CLV": (-0.01, 0.01),
            "Positive CLV": (0.01, 0.05),
            "Strong Positive CLV": (0.05, float('inf'))
        }

        bucket_performance = {}

        for bucket_name, (min_clv, max_clv) in buckets.items():
            bucket_bets = [
                b for b in clv_bets
                if min_clv <= b.absolute_clv < max_clv and b.won is not None
            ]

            if bucket_bets:
                wins = sum(1 for b in bucket_bets if b.won)
                total_payout = sum(b.payout for b in bucket_bets if b.payout is not None)
                total_wagered = sum(b.bet_amount for b in bucket_bets)

                bucket_performance[bucket_name] = {
                    "count": len(bucket_bets),
                    "win_rate": wins / len(bucket_bets),
                    "roi": total_payout / total_wagered if total_wagered > 0 else 0.0,
                    "average_clv": np.mean([b.absolute_clv for b in bucket_bets])
                }

        return bucket_performance

    def _calculate_clv_significance(self, clv_values: List[float]) -> Tuple[Optional[float], Optional[float]]:
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

    def export_clv_analysis(self, filepath: str, summary: Optional[CLVSummary] = None):
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
                "clv_p_value": summary.clv_p_value
            },
            "bucket_performance": summary.clv_bucket_performance,
            "bet_details": []
        }

        # Add individual bet details
        for bet in self.bet_records:
            if bet.absolute_clv is not None:
                export_data["bet_details"].append({
                    "game_id": bet.game_id,
                    "bet_type": bet.bet_type,
                    "bet_side": bet.bet_side,
                    "bet_timestamp": bet.bet_timestamp.isoformat(),
                    "our_odds": bet.our_odds,
                    "closing_odds": bet.closing_odds,
                    "absolute_clv": bet.absolute_clv,
                    "percentage_clv": bet.percentage_clv,
                    "ev_clv": bet.ev_clv,
                    "won": bet.won,
                    "payout": bet.payout
                })

        with open(filepath, 'w') as f:
            json.dump(export_data, f, indent=2)

        logger.info(f"CLV analysis exported to {filepath}")

    def get_clv_insights(self, summary: Optional[CLVSummary] = None) -> Dict[str, str]:
        """Generate human-readable insights from CLV analysis."""

        if summary is None:
            summary = self.calculate_clv_summary()

        insights = []

        # Overall CLV assessment
        if summary.average_absolute_clv > 0.02:
            insights.append("🟢 Excellent CLV: You're consistently getting better lines than the closing market")
        elif summary.average_absolute_clv > 0.005:
            insights.append("🟡 Positive CLV: You're getting slightly better lines than the market")
        elif summary.average_absolute_clv > -0.005:
            insights.append("🟡 Neutral CLV: Your lines are roughly in line with the market")
        else:
            insights.append("🔴 Negative CLV: You're getting worse lines than the closing market")

        # Positive CLV rate
        if summary.positive_clv_rate > 0.6:
            insights.append(f"🟢 High positive CLV rate: {summary.positive_clv_rate:.1%} of bets beat the closing line")
        elif summary.positive_clv_rate > 0.4:
            insights.append(f"🟡 Moderate positive CLV rate: {summary.positive_clv_rate:.1%} of bets beat the closing line")
        else:
            insights.append(f"🔴 Low positive CLV rate: Only {summary.positive_clv_rate:.1%} of bets beat the closing line")

        # Statistical significance
        if summary.clv_p_value and summary.clv_p_value < 0.05:
            insights.append("📊 Your CLV is statistically significant (p < 0.05)")
        elif summary.clv_p_value:
            insights.append("📊 Your CLV is not statistically significant - need more data or better line shopping")

        # Best performing buckets
        if summary.clv_bucket_performance:
            best_bucket = max(
                summary.clv_bucket_performance.items(),
                key=lambda x: x[1]['roi']
            )
            insights.append(f"🎯 Best performing CLV bucket: {best_bucket[0]} with {best_bucket[1]['roi']:.2%} ROI")

        return {
            "overall_assessment": insights[0] if insights else "No data available",
            "detailed_insights": insights[1:] if len(insights) > 1 else [],
            "recommendation": self._get_clv_recommendation(summary)
        }

    def _get_clv_recommendation(self, summary: CLVSummary) -> str:
        """Get recommendation based on CLV performance."""

        if summary.average_absolute_clv > 0.01 and summary.positive_clv_rate > 0.55:
            return "Continue current strategy - you're beating the market consistently"
        elif summary.average_absolute_clv > 0:
            return "Good CLV but can improve - consider line shopping and bet timing optimization"
        else:
            return "Focus on line shopping and bet timing - you're not beating closing lines consistently"


def american_to_decimal(american_odds: int) -> float:
    """Convert American odds to decimal odds."""
    if american_odds > 0:
        return (american_odds / 100) + 1
    else:
        return (100 / abs(american_odds)) + 1