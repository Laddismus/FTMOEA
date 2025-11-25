from __future__ import annotations

import logging
from datetime import datetime

from afts_pro.exec.position_models import AccountState
from afts_pro.risk.base_policy import BaseRiskPolicy, RiskDecision

logger = logging.getLogger(__name__)


class ApexRiskPolicy(BaseRiskPolicy):
    """
    Trailing drawdown policy similar to Apex challenges.
    """

    def __init__(
        self,
        *,
        initial_balance: float,
        trailing_dd_pct: float = 0.04,
        include_unrealized: bool = True,
    ) -> None:
        super().__init__(name="APEX")
        self.initial_balance = initial_balance
        self.trailing_dd_pct = trailing_dd_pct
        self.include_unrealized = include_unrealized
        self.live_ath_equity = initial_balance

    def on_new_day(self, account_state: AccountState, ts: datetime) -> None:
        # No daily resets needed for trailing DD.
        return None

    def evaluate(self, *, account_state: AccountState, ts: datetime) -> RiskDecision:
        equity = self._get_equity(account_state)
        self.live_ath_equity = max(self.live_ath_equity, equity)
        trailing_floor = self.live_ath_equity * (1.0 - self.trailing_dd_pct)

        meta = {
            "equity": equity,
            "live_ath_equity": self.live_ath_equity,
            "trailing_floor": trailing_floor,
        }

        if equity < trailing_floor:
            logger.error("Apex trailing DD breach: equity=%.4f floor=%.4f", equity, trailing_floor)
            return RiskDecision(
                allow_new_orders=False,
                hard_stop_trading=True,
                reason="APEX_TRAILING_DD_BREACH",
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
