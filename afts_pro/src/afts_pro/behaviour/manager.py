from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List, Optional

from afts_pro.behaviour.base_guard import BaseBehaviourGuard, BehaviourDecision, TradeStats
from afts_pro.exec.position_models import AccountState

logger = logging.getLogger(__name__)


class BehaviourManager:
    def __init__(self, guards: List[BaseBehaviourGuard], tz: Optional[timezone] = None) -> None:
        self.guards = guards
        self.tz = tz
        now = datetime.now(tz=tz).date()
        self.stats = TradeStats(date=now)

    def _ensure_stats_date(self, ts: datetime) -> None:
        if self.stats.date != ts.date():
            logger.debug("Behaviour stats reset for new day: %s", ts.date())
            self.stats = TradeStats(date=ts.date())

    def on_trade_closed(self, trade_pnl: float, ts: datetime, account_state: AccountState) -> None:
        self._ensure_stats_date(ts)
        for guard in self.guards:
            guard.on_trade_closed(trade_pnl=trade_pnl, ts=ts, stats=self.stats, account_state=account_state)

    def before_new_orders(self, ts: datetime, account_state: AccountState) -> BehaviourDecision:
        self._ensure_stats_date(ts)

        decisions = []
        for guard in self.guards:
            decision = guard.before_new_orders(ts=ts, stats=self.stats, account_state=account_state)
            logger.debug("Behaviour guard decision | guard=%s | decision=%s", guard.name, decision)
            decisions.append((guard.name, decision))

        # Hard block check
        for name, decision in decisions:
            if decision.hard_block_trading:
                logger.info("Behaviour hard block by guard=%s reason=%s", name, decision.reason)
                return BehaviourDecision(
                    allow_new_orders=False,
                    hard_block_trading=True,
                    reason="BEHAVIOUR_HARD_BLOCK",
                    meta={"guards": decisions},
                )

        blocking = [(n, d) for n, d in decisions if not d.allow_new_orders]
        if blocking:
            logger.info("Behaviour soft block triggered by guards=%s", [n for n, _ in blocking])
            return BehaviourDecision(
                allow_new_orders=False,
                hard_block_trading=False,
                reason="BEHAVIOUR_SOFT_BLOCK",
                meta={"guards": blocking},
            )

        return BehaviourDecision(allow_new_orders=True)
