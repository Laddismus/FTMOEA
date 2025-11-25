from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from afts_pro.exec.position_models import AccountState

logger = logging.getLogger(__name__)


class BehaviourDecision(BaseModel):
    allow_new_orders: bool
    hard_block_trading: bool = False
    reason: Optional[str] = Field(default=None)
    meta: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}


class TradeStats(BaseModel):
    date: date
    trades_closed_today: int = 0
    losses_closed_today: int = 0
    wins_closed_today: int = 0
    consecutive_losses: int = 0
    last_trade_pnl: float = 0.0
    realized_pnl_today: float = 0.0

    model_config = {"populate_by_name": True}


class BaseBehaviourGuard:
    name: str

    def __init__(self, name: str) -> None:
        self.name = name

    def on_trade_closed(
        self,
        *,
        trade_pnl: float,
        ts: datetime,
        stats: TradeStats,
        account_state: AccountState,
    ) -> None:
        """
        Hook invoked when a trade closes or is reduced with realized PnL.
        """
        return None

    def before_new_orders(
        self,
        *,
        ts: datetime,
        stats: TradeStats,
        account_state: AccountState,
    ) -> BehaviourDecision:
        """
        Decide if new orders are allowed before order construction.
        """
        return BehaviourDecision(allow_new_orders=True)
