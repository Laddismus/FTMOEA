from __future__ import annotations

import logging
from datetime import datetime
from datetime import time
from typing import List, Optional

from pydantic import BaseModel

from afts_pro.behaviour.base_guard import BaseBehaviourGuard, BehaviourDecision, TradeStats
from afts_pro.exec.position_models import AccountState

logger = logging.getLogger(__name__)


class MaxTradesPerDayGuard(BaseBehaviourGuard):
    def __init__(self, max_trades_per_day: int) -> None:
        super().__init__(name="MaxTradesPerDay")
        self.max_trades_per_day = max_trades_per_day

    def before_new_orders(
        self, *, ts: datetime, stats: TradeStats, account_state: AccountState
    ) -> BehaviourDecision:
        if stats.trades_closed_today >= self.max_trades_per_day:
            meta = {
                "trades_closed_today": stats.trades_closed_today,
                "max_trades_per_day": self.max_trades_per_day,
            }
            return BehaviourDecision(
                allow_new_orders=False,
                hard_block_trading=False,
                reason="MAX_TRADES_PER_DAY_REACHED",
                meta=meta,
            )
        return BehaviourDecision(allow_new_orders=True)


class MaxConsecutiveLossesGuard(BaseBehaviourGuard):
    def __init__(self, max_consecutive_losses: int) -> None:
        super().__init__(name="MaxConsecutiveLosses")
        self.max_consecutive_losses = max_consecutive_losses

    def on_trade_closed(
        self,
        *,
        trade_pnl: float,
        ts: datetime,
        stats: TradeStats,
        account_state: AccountState,
    ) -> None:
        if trade_pnl < 0:
            stats.consecutive_losses += 1
            stats.losses_closed_today += 1
        elif trade_pnl > 0:
            stats.consecutive_losses = 0
            stats.wins_closed_today += 1
        stats.trades_closed_today += 1
        stats.realized_pnl_today += trade_pnl
        stats.last_trade_pnl = trade_pnl

    def before_new_orders(
        self, *, ts: datetime, stats: TradeStats, account_state: AccountState
    ) -> BehaviourDecision:
        if stats.consecutive_losses >= self.max_consecutive_losses:
            meta = {
                "consecutive_losses": stats.consecutive_losses,
                "max_consecutive_losses": self.max_consecutive_losses,
            }
            return BehaviourDecision(
                allow_new_orders=False,
                hard_block_trading=False,
                reason="MAX_CONSEC_LOSSES_REACHED",
                meta=meta,
            )
        return BehaviourDecision(allow_new_orders=True)


class CooldownAfterLossGuard(BaseBehaviourGuard):
    def __init__(self, cooldown_minutes: int) -> None:
        super().__init__(name="CooldownAfterLoss")
        self.cooldown_minutes = cooldown_minutes
        self.last_loss_ts: datetime | None = None

    def on_trade_closed(
        self,
        *,
        trade_pnl: float,
        ts: datetime,
        stats: TradeStats,
        account_state: AccountState,
    ) -> None:
        if trade_pnl < 0:
            self.last_loss_ts = ts

    def before_new_orders(
        self, *, ts: datetime, stats: TradeStats, account_state: AccountState
    ) -> BehaviourDecision:
        if self.last_loss_ts is None:
            return BehaviourDecision(allow_new_orders=True)

        delta_minutes = (ts - self.last_loss_ts).total_seconds() / 60.0
        if delta_minutes < self.cooldown_minutes:
            meta = {
                "minutes_since_last_loss": delta_minutes,
                "cooldown_minutes": self.cooldown_minutes,
            }
            return BehaviourDecision(
                allow_new_orders=False,
                hard_block_trading=False,
                reason="COOLDOWN_AFTER_LOSS_ACTIVE",
                meta=meta,
            )

        return BehaviourDecision(allow_new_orders=True)


class DailyPnLGuard(BaseBehaviourGuard):
    def __init__(
        self,
        *,
        initial_balance: float,
        max_daily_loss_pct_initial: float,
        lock_in_profit_pct_initial: float = 0.0,
    ) -> None:
        super().__init__(name="DailyPnL")
        self.initial_balance = initial_balance
        self.max_daily_loss_pct_initial = max_daily_loss_pct_initial
        self.lock_in_profit_pct_initial = lock_in_profit_pct_initial

    def before_new_orders(
        self, *, ts: datetime, stats: TradeStats, account_state: AccountState
    ) -> BehaviourDecision:
        daily_realized = stats.realized_pnl_today
        daily_loss = min(daily_realized, 0.0)
        daily_loss_abs = abs(daily_loss)
        max_daily_loss_abs = self.initial_balance * self.max_daily_loss_pct_initial

        if daily_loss_abs >= max_daily_loss_abs:
            return BehaviourDecision(
                allow_new_orders=False,
                hard_block_trading=False,
                reason="DAILY_REALIZED_LOSS_GUARD",
                meta={
                    "daily_realized_pnl": daily_realized,
                    "max_daily_loss_abs": max_daily_loss_abs,
                },
            )

        if self.lock_in_profit_pct_initial > 0.0:
            target_profit_abs = self.initial_balance * self.lock_in_profit_pct_initial
            if daily_realized >= target_profit_abs:
                return BehaviourDecision(
                    allow_new_orders=False,
                    hard_block_trading=False,
                    reason="DAILY_PROFIT_LOCKIN_GUARD",
                    meta={
                        "daily_realized_pnl": daily_realized,
                        "lockin_profit_abs": target_profit_abs,
                    },
                )

        return BehaviourDecision(allow_new_orders=True)


class DailyProfitTargetGuard(BaseBehaviourGuard):
    def __init__(
        self,
        *,
        initial_balance: float,
        target_profit_pct_initial: float,
        mode: str = "soft_stop",
    ) -> None:
        super().__init__(name="DailyProfitTarget")
        self.initial_balance = initial_balance
        self.target_profit_pct_initial = target_profit_pct_initial
        self.mode = mode

    def before_new_orders(
        self, *, ts: datetime, stats: TradeStats, account_state: AccountState
    ) -> BehaviourDecision:
        daily_realized = stats.realized_pnl_today
        target_abs = self.initial_balance * self.target_profit_pct_initial

        if daily_realized >= target_abs:
            hard = self.mode == "hard_stop"
            reason = "DAILY_PROFIT_TARGET_HARD" if hard else "DAILY_PROFIT_TARGET_SOFT"
            return BehaviourDecision(
                allow_new_orders=False,
                hard_block_trading=hard,
                reason=reason,
                meta={"daily_realized_pnl": daily_realized, "target_abs": target_abs},
            )
        return BehaviourDecision(allow_new_orders=True)


class MaxOpenPositionsGuard(BaseBehaviourGuard):
    def __init__(self, max_open_positions: int) -> None:
        super().__init__(name="MaxOpenPositions")
        self.max_open_positions = max_open_positions

    def before_new_orders(
        self, *, ts: datetime, stats: TradeStats, account_state: AccountState
    ) -> BehaviourDecision:
        open_positions = len([p for p in account_state.positions.values() if p.qty > 0])
        if open_positions >= self.max_open_positions:
            return BehaviourDecision(
                allow_new_orders=False,
                hard_block_trading=False,
                reason="MAX_OPEN_POSITIONS_REACHED",
                meta={"open_positions": open_positions, "max_open_positions": self.max_open_positions},
            )
        return BehaviourDecision(allow_new_orders=True)


class SessionTimeWindowConfig(BaseModel):
    name: str
    start_time: time
    end_time: time
    weekdays: Optional[List[int]] = None

    model_config = {"populate_by_name": True}


class SessionTimeWindowGuard(BaseBehaviourGuard):
    def __init__(self, windows: List[SessionTimeWindowConfig]) -> None:
        super().__init__(name="SessionTimeWindow")
        self.windows = windows

    def before_new_orders(
        self, *, ts: datetime, stats: TradeStats, account_state: AccountState
    ) -> BehaviourDecision:
        ts_time = ts.time()
        ts_weekday = ts.weekday()

        for window in self.windows:
            within_time = window.start_time <= ts_time < window.end_time
            weekday_ok = window.weekdays is None or ts_weekday in window.weekdays
            if within_time and weekday_ok:
                return BehaviourDecision(allow_new_orders=True)

        return BehaviourDecision(
            allow_new_orders=False,
            hard_block_trading=False,
            reason="OUTSIDE_SESSION_WINDOW",
            meta={"ts": ts.isoformat(), "windows": [w.name for w in self.windows]},
        )


class BigLossCooldownGuard(BaseBehaviourGuard):
    def __init__(self, *, initial_balance: float, big_loss_pct_initial: float, cooldown_minutes: int) -> None:
        super().__init__(name="BigLossCooldown")
        self.initial_balance = initial_balance
        self.big_loss_pct_initial = big_loss_pct_initial
        self.cooldown_minutes = cooldown_minutes
        self.last_big_loss_ts: datetime | None = None

    def on_trade_closed(
        self,
        *,
        trade_pnl: float,
        ts: datetime,
        stats: TradeStats,
        account_state: AccountState,
    ) -> None:
        if trade_pnl < 0:
            if abs(trade_pnl) >= self.initial_balance * self.big_loss_pct_initial:
                self.last_big_loss_ts = ts

    def before_new_orders(
        self, *, ts: datetime, stats: TradeStats, account_state: AccountState
    ) -> BehaviourDecision:
        if self.last_big_loss_ts is None:
            return BehaviourDecision(allow_new_orders=True)

        delta_minutes = (ts - self.last_big_loss_ts).total_seconds() / 60.0
        if delta_minutes < self.cooldown_minutes:
            return BehaviourDecision(
                allow_new_orders=False,
                hard_block_trading=False,
                reason="BIG_LOSS_COOLDOWN_ACTIVE",
                meta={
                    "minutes_since_big_loss": delta_minutes,
                    "cooldown_minutes": self.cooldown_minutes,
                },
            )

        return BehaviourDecision(allow_new_orders=True)
