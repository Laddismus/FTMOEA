from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Optional

from afts_pro.exec.position_models import AccountState
from afts_pro.risk.base_policy import BaseRiskPolicy, RiskDecision

logger = logging.getLogger(__name__)


class FtmoRiskPolicy(BaseRiskPolicy):
    """
    FTMO-like drawdown constraints with soft/hard buffers.
    """

    def __init__(
        self,
        *,
        initial_balance: float,
        total_dd_hard_stop_pct: float = 0.085,
        daily_soft_dd_pct: float = 0.035,
        daily_hard_dd_pct: float = 0.04,
        include_unrealized: bool = True,
    ) -> None:
        super().__init__(name="FTMO")
        self.initial_balance = initial_balance
        self.total_dd_hard_stop_pct = total_dd_hard_stop_pct
        self.daily_soft_dd_pct = daily_soft_dd_pct
        self.daily_hard_dd_pct = daily_hard_dd_pct
        self.include_unrealized = include_unrealized
        self.daily_start_equity: Optional[float] = None
        self.last_day: Optional[date] = None

    def on_new_day(self, account_state: AccountState, ts: datetime) -> None:
        self.daily_start_equity = self._get_equity(account_state)
        self.last_day = ts.date()
        logger.debug(
            "FTMO reset daily start equity: %.4f for date %s",
            self.daily_start_equity,
            self.last_day,
        )

    def evaluate(self, *, account_state: AccountState, ts: datetime) -> RiskDecision:
        self._update_daily_start_if_needed(account_state, ts)
        equity = self._get_equity(account_state)
        total_dd = self._compute_total_dd(equity)
        daily_soft_limit, daily_hard_limit = self._compute_daily_limits()

        daily_dd_soft_breached = equity <= daily_soft_limit
        daily_dd_hard_breached = equity <= daily_hard_limit
        total_dd_hard_breached = total_dd >= self.total_dd_hard_stop_pct

        meta = {
            "equity": equity,
            "total_dd": total_dd,
            "total_dd_hard_stop_pct": self.total_dd_hard_stop_pct,
            "daily_start_equity": self.daily_start_equity,
            "daily_soft_limit": daily_soft_limit,
            "daily_hard_limit": daily_hard_limit,
        }

        if total_dd_hard_breached:
            logger.error("FTMO total hard stop breached: total_dd=%.4f", total_dd)
            return RiskDecision(
                allow_new_orders=False,
                hard_stop_trading=True,
                reason="FTMO_TOTAL_HARD_STOP",
                meta=meta,
            )

        if daily_dd_hard_breached:
            logger.error("FTMO daily hard stop breached: equity=%.4f limit=%.4f", equity, daily_hard_limit)
            return RiskDecision(
                allow_new_orders=False,
                hard_stop_trading=True,
                reason="FTMO_DAILY_HARD_STOP",
                meta=meta,
            )

        if daily_dd_soft_breached:
            logger.warning("FTMO daily soft stop breached: equity=%.4f limit=%.4f", equity, daily_soft_limit)
            return RiskDecision(
                allow_new_orders=False,
                hard_stop_trading=False,
                reason="FTMO_DAILY_SOFT_STOP",
                meta=meta,
            )

        return RiskDecision(
            allow_new_orders=True,
            hard_stop_trading=False,
            reason=None,
            meta=meta,
        )

    def _get_equity(self, account_state: AccountState) -> float:
        if self.include_unrealized:
            return account_state.equity
        return account_state.balance

    def _update_daily_start_if_needed(self, account_state: AccountState, ts: datetime) -> None:
        if self.last_day != ts.date() or self.daily_start_equity is None:
            self.on_new_day(account_state, ts)

    def _compute_total_dd(self, equity: float) -> float:
        return (self.initial_balance - equity) / self.initial_balance

    def _compute_daily_limits(self) -> tuple[float, float]:
        soft_abs = self.initial_balance * self.daily_soft_dd_pct
        hard_abs = self.initial_balance * self.daily_hard_dd_pct
        daily_start = self.daily_start_equity or self.initial_balance
        daily_soft_limit = daily_start - soft_abs
        daily_hard_limit = daily_start - hard_abs
        return daily_soft_limit, daily_hard_limit
