"""FTMO-inspired risk evaluation over backtest equity curves."""

from __future__ import annotations

from typing import Sequence, TYPE_CHECKING

from research_lab.backend.core.risk_guard.models import FtmoRiskConfig, FtmoRiskSummary, FtmoRiskEvent, FtmoBreachType

if TYPE_CHECKING:
    from research_lab.backend.core.backtests.models import BacktestBar


class FtmoRiskGuard:
    """Evaluate a backtest against FTMO-like loss constraints."""

    def __init__(self, config: FtmoRiskConfig):
        self.config = config

    def evaluate(self, bars: Sequence["BacktestBar"], returns: Sequence[float]) -> FtmoRiskSummary:
        """Evaluate bar-aligned returns against configured FTMO limits.

        Args:
            bars: Sequence of bars (timestamped, UTC expected) aligned to returns.
            returns: Per-bar returns matching the bars sequence.

        Raises:
            ValueError: if the lengths of bars and returns differ or sequences are empty.
        """

        if len(bars) != len(returns):
            raise ValueError("bars and returns must have identical length for risk evaluation.")
        if not bars:
            raise ValueError("bars and returns must not be empty.")

        # Sort by timestamp to ensure chronological evaluation
        paired = sorted(zip(bars, returns), key=lambda p: p[0].ts)

        equity = self.config.initial_equity
        total_limit = max(self.config.max_total_loss_pct - self.config.safety_buffer_pct, 0.0)
        daily_limit = max(self.config.max_daily_loss_pct - self.config.safety_buffer_pct, 0.0)

        worst_total_dd = 0.0
        worst_daily_dd = 0.0
        first_breach: FtmoRiskEvent | None = None

        current_day = paired[0][0].ts.date()
        day_start_equity = equity
        day_min_equity = equity

        for bar, ret in paired:
            if bar.ts.date() != current_day:
                current_day = bar.ts.date()
                day_start_equity = equity
                day_min_equity = equity

            equity *= 1.0 + ret
            day_min_equity = min(day_min_equity, equity)

            total_drawdown_pct = (self.config.initial_equity - equity) / self.config.initial_equity
            daily_drawdown_pct = (day_start_equity - day_min_equity) / self.config.initial_equity

            worst_total_dd = max(worst_total_dd, total_drawdown_pct)
            worst_daily_dd = max(worst_daily_dd, daily_drawdown_pct)

            breach_type: FtmoBreachType = FtmoBreachType.NONE
            breach_drawdown = 0.0
            if total_drawdown_pct >= total_limit:
                breach_type = FtmoBreachType.TOTAL
                breach_drawdown = total_drawdown_pct
            elif daily_drawdown_pct >= daily_limit:
                breach_type = FtmoBreachType.DAILY
                breach_drawdown = daily_drawdown_pct

            if breach_type != FtmoBreachType.NONE and first_breach is None:
                first_breach = FtmoRiskEvent(
                    ts=bar.ts,
                    breach_type=breach_type,
                    equity=equity,
                    drawdown_pct=breach_drawdown,
                    day=bar.ts.date(),
                )

        summary = FtmoRiskSummary(
            passed=first_breach is None,
            first_breach=first_breach,
            worst_daily_drawdown_pct=worst_daily_dd,
            worst_total_drawdown_pct=worst_total_dd,
            config=self.config,
        )
        return summary
