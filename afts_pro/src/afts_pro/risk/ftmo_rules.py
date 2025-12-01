from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional


@dataclass
class FtmoRiskConfig:
    initial_equity: Optional[float] = None
    max_daily_loss_pct: float = 5.0
    max_overall_loss_pct: float = 10.0
    daily_soft_stop_pct: float = 3.0
    daily_hard_stop_pct: float = 4.0
    safety_overall_loss_pct: float = 8.5
    enforce_safety: bool = True
    close_positions_on_breach: bool = True


@dataclass
class FtmoDayState:
    day: date
    daily_start_equity: float
    daily_realized_pnl: float = 0.0


@dataclass
class FtmoRiskState:
    initial_equity: float
    current_equity: float
    day_state: FtmoDayState


class FtmoRiskEngine:
    def __init__(self, cfg: FtmoRiskConfig):
        self.cfg = cfg
        self.state: Optional[FtmoRiskState] = None

    def ensure_initialized(self, equity: float, now: datetime) -> None:
        if self.state is None:
            init_eq = self.cfg.initial_equity if self.cfg.initial_equity is not None else equity
            self.state = FtmoRiskState(
                initial_equity=init_eq,
                current_equity=equity,
                day_state=FtmoDayState(day=now.date(), daily_start_equity=init_eq),
            )

    def on_new_equity(self, equity: float, realized_pnl: float, now: datetime) -> None:
        self.ensure_initialized(equity, now)
        assert self.state is not None
        # Handle day reset
        if now.date() != self.state.day_state.day:
            self.state.day_state = FtmoDayState(day=now.date(), daily_start_equity=equity, daily_realized_pnl=realized_pnl)
        else:
            self.state.day_state.daily_realized_pnl = realized_pnl
        self.state.current_equity = equity

    def current_overall_loss_pct(self) -> float:
        if self.state is None:
            return 0.0
        loss = self.state.initial_equity - self.state.current_equity
        return (loss / self.state.initial_equity) * 100 if self.state.initial_equity else 0.0

    def current_daily_loss_pct(self) -> float:
        if self.state is None:
            return 0.0
        loss = self.state.day_state.daily_start_equity - self.state.current_equity
        return (loss / self.state.day_state.daily_start_equity) * 100 if self.state.day_state.daily_start_equity else 0.0

    def is_overall_safety_breached(self) -> bool:
        return self.current_overall_loss_pct() >= self.cfg.safety_overall_loss_pct

    def is_daily_safety_breached(self) -> bool:
        return self.current_daily_loss_pct() >= self.cfg.daily_hard_stop_pct

    def is_daily_soft_breached(self) -> bool:
        return self.current_daily_loss_pct() >= self.cfg.daily_soft_stop_pct

    def is_daily_hard_breached(self) -> bool:
        return self.current_daily_loss_pct() >= self.cfg.daily_hard_stop_pct

    def can_open_new_trade(self) -> bool:
        if not self.cfg.enforce_safety:
            return True
        if self.is_daily_soft_breached():
            return False
        if self.is_overall_safety_breached():
            return False
        return True

    def should_force_close_all_positions(self) -> bool:
        if not self.cfg.enforce_safety or not self.cfg.close_positions_on_breach:
            return False
        if self.is_daily_hard_breached():
            return True
        if self.is_overall_safety_breached():
            return True
        return False
