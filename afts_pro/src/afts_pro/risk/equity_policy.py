from __future__ import annotations

import logging
from datetime import datetime

from afts_pro.exec.position_models import AccountState
from afts_pro.risk.base_policy import BaseRiskPolicy, RiskDecision

logger = logging.getLogger(__name__)


class EquityMaxDdPolicy(BaseRiskPolicy):
    """
    Equity-based maximum drawdown policy with optional HWM and equity basis.
    """

    def __init__(
        self,
        *,
        max_dd_pct: float,
        initial_balance: float,
        include_unrealized: bool = True,
        use_hwm: bool = True,
        equity_basis: str = "full",
    ) -> None:
        super().__init__(name="EQUITY_MAX_DD")
        self.max_dd_pct = max_dd_pct
        self.initial_balance = initial_balance
        self.include_unrealized = include_unrealized
        self.use_hwm = use_hwm
        self.equity_basis = equity_basis
        self.hwm_equity: float = initial_balance

    def on_new_day(self, account_state: AccountState, ts: datetime) -> None:
        return None

    def evaluate(self, *, account_state: AccountState, ts: datetime) -> RiskDecision:
        equity = self._get_equity(account_state)
        self._update_hwm(equity)
        if self.use_hwm:
            dd = (self.hwm_equity - equity) / self.hwm_equity
        else:
            dd = (self.initial_balance - equity) / self.initial_balance

        meta = {
            "equity": equity,
            "hwm_equity": self.hwm_equity,
            "dd": dd,
            "max_dd_pct": self.max_dd_pct,
            "equity_basis": self.equity_basis,
            "use_hwm": self.use_hwm,
        }

        if dd >= self.max_dd_pct:
            logger.error("Equity max DD breach: dd=%.4f", dd)
            return RiskDecision(
                allow_new_orders=False,
                hard_stop_trading=True,
                reason="EQUITY_MAX_DD_BREACH",
                meta=meta,
            )

        return RiskDecision(
            allow_new_orders=True,
            hard_stop_trading=False,
            reason=None,
            meta=meta,
        )

    def _get_equity(self, account_state: AccountState) -> float:
        if self.equity_basis == "full":
            return account_state.equity
        return account_state.balance

    def _update_hwm(self, equity: float) -> None:
        if self.use_hwm:
            self.hwm_equity = max(self.hwm_equity, equity)
