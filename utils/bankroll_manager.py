"""
Advanced bankroll management system with risk controls and performance tracking.

This module provides comprehensive bankroll management including drawdown limits,
stop-loss mechanisms, unit sizing, and performance analytics.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np

from utils.kelly_criterion import KellyCalculator, KellyMode, KellyResult


class RiskLevel(Enum):
    """Risk management levels."""

    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    CUSTOM = "custom"


class AlertType(Enum):
    """Types of bankroll alerts."""

    DRAWDOWN_WARNING = "drawdown_warning"
    DRAWDOWN_LIMIT = "drawdown_limit"
    LOW_BALANCE = "low_balance"
    POOR_PERFORMANCE = "poor_performance"
    UNIT_SIZE_ADJUSTMENT = "unit_size_adjustment"


@dataclass
class BankrollAlert:
    """Container for bankroll alerts and warnings."""

    alert_type: AlertType
    message: str
    severity: str  # 'info', 'warning', 'critical'
    timestamp: datetime
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class BettingSession:
    """Container for a betting session's data."""

    session_id: str
    start_time: datetime
    end_time: datetime | None
    starting_balance: float
    ending_balance: float | None
    total_bets: int = 0
    winning_bets: int = 0
    total_wagered: float = 0.0
    net_profit: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    max_concurrent_bets: int = 0

    @property
    def win_rate(self) -> float:
        """Calculate session win rate."""
        if self.total_bets == 0:
            return 0.0
        return self.winning_bets / self.total_bets

    @property
    def roi(self) -> float:
        """Calculate session ROI."""
        if self.total_wagered == 0:
            return 0.0
        return self.net_profit / self.total_wagered


class BankrollManager:
    """
    Comprehensive bankroll management system.

    Features:
    - Drawdown limits and stop-loss mechanisms
    - Dynamic unit sizing based on bankroll performance
    - Risk level management with automatic adjustments
    - Session tracking and performance analytics
    - Alert system for risk management
    - Integration with Kelly Criterion calculations
    """

    def __init__(
        self,
        starting_bankroll: float,
        risk_level: RiskLevel = RiskLevel.MODERATE,
        max_drawdown_pct: float = 0.20,
        stop_loss_pct: float = 0.15,
        unit_size_pct: float = 0.01,
        min_unit_size: float = 25.0,
        max_unit_size: float = 500.0,
        rebalance_frequency: int = 50,  # bets
        confidence_threshold: float = 0.02,
    ):
        """
        Initialize bankroll manager.

        Args:
            starting_bankroll: Initial bankroll amount
            risk_level: Risk management level
            max_drawdown_pct: Maximum drawdown before stopping
            stop_loss_pct: Stop loss threshold
            unit_size_pct: Unit size as percentage of bankroll
            min_unit_size: Minimum unit size in dollars
            max_unit_size: Maximum unit size in dollars
            rebalance_frequency: How often to rebalance (in number of bets)
            confidence_threshold: Minimum edge required for betting
        """
        self.starting_bankroll = starting_bankroll
        self.risk_level = risk_level
        self.max_drawdown_pct = max_drawdown_pct
        self.stop_loss_pct = stop_loss_pct
        self.unit_size_pct = unit_size_pct
        self.min_unit_size = min_unit_size
        self.max_unit_size = max_unit_size
        self.rebalance_frequency = rebalance_frequency
        self.confidence_threshold = confidence_threshold

        # Initialize Kelly calculator with appropriate settings
        kelly_fraction = self._get_kelly_fraction_for_risk_level()
        initial_unit_size = starting_bankroll * unit_size_pct
        initial_unit_size = max(min_unit_size, min(max_unit_size, initial_unit_size))

        self.kelly_calculator = KellyCalculator(
            starting_bankroll=starting_bankroll,
            max_bet_pct=self._get_max_bet_pct_for_risk_level(),
            max_drawdown_pct=max_drawdown_pct,
            base_unit_size=initial_unit_size,
            default_kelly_fraction=kelly_fraction,
            confidence_threshold=confidence_threshold,
        )

        # Tracking variables
        self.alerts: list[BankrollAlert] = []
        self.sessions: list[BettingSession] = []
        self.current_session: BettingSession | None = None
        self.bets_since_rebalance = 0
        self.is_stopped = False
        self.stop_reason: str | None = None

        # Performance tracking
        self.daily_balances: list[tuple[datetime, float]] = [
            (datetime.now(), starting_bankroll)
        ]
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.best_streak = 0
        self.worst_streak = 0

    def _get_kelly_fraction_for_risk_level(self) -> float:
        """Get Kelly fraction based on risk level."""
        risk_settings = {
            RiskLevel.CONSERVATIVE: 0.1,
            RiskLevel.MODERATE: 0.25,
            RiskLevel.AGGRESSIVE: 0.5,
            RiskLevel.CUSTOM: 0.25,
        }
        return risk_settings.get(self.risk_level, 0.25)

    def _get_max_bet_pct_for_risk_level(self) -> float:
        """Get maximum bet percentage based on risk level."""
        risk_settings = {
            RiskLevel.CONSERVATIVE: 0.02,
            RiskLevel.MODERATE: 0.05,
            RiskLevel.AGGRESSIVE: 0.10,
            RiskLevel.CUSTOM: 0.05,
        }
        return risk_settings.get(self.risk_level, 0.05)

    @property
    def current_bankroll(self) -> float:
        """Get current bankroll balance."""
        return self.kelly_calculator.bankroll_state.current_balance

    def calculate_current_unit_size(self) -> float:
        """Calculate current unit size based on bankroll."""
        current_balance = self.kelly_calculator.bankroll_state.current_balance
        calculated_unit = current_balance * self.unit_size_pct

        # Apply min/max constraints
        unit_size = max(self.min_unit_size, min(self.max_unit_size, calculated_unit))

        return unit_size

    def get_max_bet_size(self, bet_type: str) -> float:
        """Get maximum bet size for a given bet type."""
        base_max = self.calculate_current_unit_size() * 5  # Max 5 units
        return min(base_max, self.max_unit_size * 2)

    def get_available_units(self) -> float:
        """Get available units for betting."""
        current_balance = self.current_bankroll
        return current_balance / self.calculate_current_unit_size()

    def start_session(self, session_id: str | None = None) -> str:
        """Start a new betting session."""
        if session_id is None:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        current_balance = self.kelly_calculator.bankroll_state.current_balance

        self.current_session = BettingSession(
            session_id=session_id,
            start_time=datetime.now(),
            end_time=None,
            starting_balance=current_balance,
            ending_balance=None,
        )

        return session_id

    def end_session(self) -> BettingSession | None:
        """End the current betting session."""
        if self.current_session is None:
            return None

        current_balance = self.kelly_calculator.bankroll_state.current_balance
        self.current_session.end_time = datetime.now()
        self.current_session.ending_balance = current_balance

        # Archive the session
        self.sessions.append(self.current_session)
        completed_session = self.current_session
        self.current_session = None

        return completed_session

    def calculate_bet_size(
        self,
        model_prob: float,
        market_odds: int,
        confidence_override: float | None = None,
        mode: KellyMode = KellyMode.FRACTIONAL,
    ) -> KellyResult:
        """
        Calculate optimal bet size with all risk management applied.

        Args:
            model_prob: Model probability estimate
            market_odds: American odds
            confidence_override: Manual confidence override
            mode: Kelly calculation mode

        Returns:
            KellyResult with recommended bet size
        """
        # Check if betting is stopped
        if self.is_stopped:
            return KellyResult(
                recommended_bet=0.0,
                kelly_fraction=0.0,
                confidence_adjustment=0.0,
                risk_adjustment=0.0,
                final_fraction=0.0,
                reasoning=f"Betting stopped: {self.stop_reason}",
                bankroll_pct=0.0,
                units=0.0,
            )

        # Check drawdown limits
        self._check_risk_limits()

        if self.is_stopped:
            return KellyResult(
                recommended_bet=0.0,
                kelly_fraction=0.0,
                confidence_adjustment=0.0,
                risk_adjustment=0.0,
                final_fraction=0.0,
                reasoning=f"Betting stopped: {self.stop_reason}",
                bankroll_pct=0.0,
                units=0.0,
            )

        # Update unit size if needed
        self._update_unit_size_if_needed()

        # Calculate Kelly result
        kelly_result = self.kelly_calculator.calculate_optimal_bet_size(
            model_prob=model_prob,
            market_odds=market_odds,
            confidence_level=confidence_override,
            mode=mode,
        )

        return kelly_result

    def record_bet_result(
        self,
        bet_amount: float,
        outcome: bool,
        odds: int,
        payout: float | None = None,
    ) -> None:
        """
        Record the result of a bet and update all tracking.

        Args:
            bet_amount: Amount wagered
            outcome: True if won, False if lost
            odds: American odds
            payout: Total payout if won
        """
        # Update Kelly calculator
        self.kelly_calculator.update_bankroll(bet_amount, outcome, payout, odds)

        # Update session tracking
        if self.current_session is not None:
            self.current_session.total_bets += 1
            self.current_session.total_wagered += bet_amount

            if outcome:
                self.current_session.winning_bets += 1
                profit = (payout or 0) - bet_amount
                self.current_session.net_profit += profit
                self.current_session.largest_win = max(
                    self.current_session.largest_win, profit
                )
            else:
                loss = bet_amount
                self.current_session.net_profit -= loss
                self.current_session.largest_loss = max(
                    self.current_session.largest_loss, loss
                )

        # Update streak tracking
        if outcome:
            self.consecutive_wins += 1
            self.consecutive_losses = 0
            self.best_streak = max(self.best_streak, self.consecutive_wins)
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0
            self.worst_streak = max(self.worst_streak, self.consecutive_losses)

        # Update daily balance tracking
        current_balance = self.kelly_calculator.bankroll_state.current_balance
        self.daily_balances.append((datetime.now(), current_balance))

        # Increment rebalance counter
        self.bets_since_rebalance += 1

        # Check for performance-based alerts
        self._check_performance_alerts()

    def _check_risk_limits(self) -> None:
        """Check if any risk limits have been exceeded."""
        current_dd_pct = self.kelly_calculator.bankroll_state.drawdown_pct

        # Check stop loss
        if current_dd_pct >= self.stop_loss_pct:
            self._stop_betting(f"Stop loss triggered at {current_dd_pct:.1%} drawdown")
            return

        # Check maximum drawdown
        if current_dd_pct >= self.max_drawdown_pct:
            self._stop_betting(f"Maximum drawdown limit reached: {current_dd_pct:.1%}")
            return

        # Check drawdown warnings
        if current_dd_pct >= 0.1 and current_dd_pct < self.max_drawdown_pct:
            self._add_alert(
                AlertType.DRAWDOWN_WARNING,
                f"Drawdown warning: {current_dd_pct:.1%}",
                "warning",
                {"drawdown_pct": current_dd_pct},
            )

        # Check low balance
        current_balance = self.kelly_calculator.bankroll_state.current_balance
        if current_balance < self.starting_bankroll * 0.5:
            self._add_alert(
                AlertType.LOW_BALANCE,
                f"Low balance warning: ${current_balance:.2f}",
                "warning",
                {
                    "current_balance": current_balance,
                    "starting_balance": self.starting_bankroll,
                },
            )

    def _check_performance_alerts(self) -> None:
        """Check for performance-based alerts."""
        total_bets = self.kelly_calculator.bankroll_state.total_bets

        # Only check after minimum number of bets
        if total_bets < 20:
            return

        win_rate = self.kelly_calculator.bankroll_state.win_rate

        # Poor performance alert
        if win_rate < 0.4 and total_bets >= 50:
            self._add_alert(
                AlertType.POOR_PERFORMANCE,
                f"Low win rate: {win_rate:.1%} over {total_bets} bets",
                "warning",
                {"win_rate": win_rate, "total_bets": total_bets},
            )

        # Excessive consecutive losses
        if self.consecutive_losses >= 8:
            self._add_alert(
                AlertType.POOR_PERFORMANCE,
                f"Consecutive losses: {self.consecutive_losses}",
                "critical",
                {"consecutive_losses": self.consecutive_losses},
            )

    def _update_unit_size_if_needed(self) -> None:
        """Update unit size if rebalance is needed."""
        if self.bets_since_rebalance >= self.rebalance_frequency:
            old_unit_size = self.kelly_calculator.base_unit_size
            new_unit_size = self.calculate_current_unit_size()

            if abs(new_unit_size - old_unit_size) / old_unit_size > 0.1:  # 10% change
                self.kelly_calculator.base_unit_size = new_unit_size
                self._add_alert(
                    AlertType.UNIT_SIZE_ADJUSTMENT,
                    f"Unit size adjusted: ${old_unit_size:.2f} -> ${new_unit_size:.2f}",
                    "info",
                    {"old_unit_size": old_unit_size, "new_unit_size": new_unit_size},
                )

            self.bets_since_rebalance = 0

    def _stop_betting(self, reason: str) -> None:
        """Stop betting with given reason."""
        self.is_stopped = True
        self.stop_reason = reason

        self._add_alert(
            AlertType.DRAWDOWN_LIMIT,
            f"Betting stopped: {reason}",
            "critical",
            {"reason": reason},
        )

    def _add_alert(
        self,
        alert_type: AlertType,
        message: str,
        severity: str,
        data: dict | None = None,
    ) -> None:
        """Add an alert to the tracking system."""
        alert = BankrollAlert(
            alert_type=alert_type,
            message=message,
            severity=severity,
            timestamp=datetime.now(),
            data=data or {},
        )
        self.alerts.append(alert)

    def resume_betting(self, reason: str = "Manual resume") -> bool:
        """
        Resume betting if conditions allow.

        Args:
            reason: Reason for resuming

        Returns:
            True if resumed successfully
        """
        if not self.is_stopped:
            return True

        # Check if conditions are safe to resume
        current_dd_pct = self.kelly_calculator.bankroll_state.drawdown_pct

        if current_dd_pct >= self.stop_loss_pct:
            return False

        self.is_stopped = False
        self.stop_reason = None

        self._add_alert(
            AlertType.DRAWDOWN_WARNING,
            f"Betting resumed: {reason}",
            "info",
            {"reason": reason, "current_drawdown": current_dd_pct},
        )

        return True

    def get_performance_summary(self) -> dict[str, Any]:
        """Get comprehensive performance summary."""
        kelly_summary = self.kelly_calculator.get_bankroll_summary()

        # Calculate additional metrics
        sessions_summary = {}
        if self.sessions:
            session_returns = [s.roi for s in self.sessions if s.roi is not None]
            if session_returns:
                sessions_summary = {
                    "total_sessions": len(self.sessions),
                    "avg_session_roi": np.mean(session_returns),
                    "best_session_roi": max(session_returns),
                    "worst_session_roi": min(session_returns),
                    "session_win_rate": len([r for r in session_returns if r > 0])
                    / len(session_returns),
                }

        # Recent alerts
        recent_alerts = [
            {
                "type": alert.alert_type.value,
                "message": alert.message,
                "severity": alert.severity,
                "timestamp": alert.timestamp.isoformat(),
            }
            for alert in self.alerts[-10:]  # Last 10 alerts
        ]

        return {
            **kelly_summary,
            "risk_level": self.risk_level.value,
            "is_stopped": self.is_stopped,
            "stop_reason": self.stop_reason,
            "consecutive_wins": self.consecutive_wins,
            "consecutive_losses": self.consecutive_losses,
            "best_streak": self.best_streak,
            "worst_streak": self.worst_streak,
            "current_unit_size": self.calculate_current_unit_size(),
            "bets_until_rebalance": self.rebalance_frequency
            - self.bets_since_rebalance,
            "total_alerts": len(self.alerts),
            **sessions_summary,
            "recent_alerts": recent_alerts,
        }

    def export_performance_data(self) -> dict[str, Any]:
        """Export all performance data for analysis."""
        return {
            "bankroll_summary": self.get_performance_summary(),
            "kelly_state": {
                "current_balance": self.kelly_calculator.bankroll_state.current_balance,
                "starting_balance": self.kelly_calculator.bankroll_state.starting_balance,
                "peak_balance": self.kelly_calculator.bankroll_state.peak_balance,
                "current_drawdown": self.kelly_calculator.bankroll_state.current_drawdown,
                "max_drawdown": self.kelly_calculator.bankroll_state.max_drawdown,
                "total_bets": self.kelly_calculator.bankroll_state.total_bets,
                "winning_bets": self.kelly_calculator.bankroll_state.winning_bets,
                "total_wagered": self.kelly_calculator.bankroll_state.total_wagered,
                "net_profit": self.kelly_calculator.bankroll_state.net_profit,
                "roi": self.kelly_calculator.bankroll_state.roi,
            },
            "sessions": [
                {
                    "session_id": session.session_id,
                    "start_time": session.start_time.isoformat(),
                    "end_time": session.end_time.isoformat()
                    if session.end_time
                    else None,
                    "starting_balance": session.starting_balance,
                    "ending_balance": session.ending_balance,
                    "total_bets": session.total_bets,
                    "winning_bets": session.winning_bets,
                    "win_rate": session.win_rate,
                    "total_wagered": session.total_wagered,
                    "net_profit": session.net_profit,
                    "roi": session.roi,
                    "largest_win": session.largest_win,
                    "largest_loss": session.largest_loss,
                }
                for session in self.sessions
            ],
            "daily_balances": [
                {"date": dt.isoformat(), "balance": balance}
                for dt, balance in self.daily_balances
            ],
            "alerts": [
                {
                    "type": alert.alert_type.value,
                    "message": alert.message,
                    "severity": alert.severity,
                    "timestamp": alert.timestamp.isoformat(),
                    "data": alert.data,
                }
                for alert in self.alerts
            ],
            "settings": {
                "starting_bankroll": self.starting_bankroll,
                "risk_level": self.risk_level.value,
                "max_drawdown_pct": self.max_drawdown_pct,
                "stop_loss_pct": self.stop_loss_pct,
                "unit_size_pct": self.unit_size_pct,
                "min_unit_size": self.min_unit_size,
                "max_unit_size": self.max_unit_size,
                "rebalance_frequency": self.rebalance_frequency,
                "confidence_threshold": self.confidence_threshold,
            },
        }
