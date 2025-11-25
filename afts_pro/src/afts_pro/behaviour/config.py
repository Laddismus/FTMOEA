from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field

from afts_pro.behaviour.base_guard import BaseBehaviourGuard
from afts_pro.behaviour.guards import (
    BigLossCooldownGuard,
    CooldownAfterLossGuard,
    DailyPnLGuard,
    DailyProfitTargetGuard,
    MaxConsecutiveLossesGuard,
    MaxOpenPositionsGuard,
    MaxTradesPerDayGuard,
    SessionTimeWindowConfig,
    SessionTimeWindowGuard,
)

logger = logging.getLogger(__name__)


class BehaviourGuardConfig(BaseModel):
    enabled: bool = True
    params: Dict[str, Any] = Field(default_factory=dict)


class BehaviourConfig(BaseModel):
    enabled: bool = True
    max_trades_per_day: Optional[BehaviourGuardConfig] = None
    max_consecutive_losses: Optional[BehaviourGuardConfig] = None
    cooldown_after_loss: Optional[BehaviourGuardConfig] = None
    daily_pnl: Optional[BehaviourGuardConfig] = None
    daily_profit_target: Optional[BehaviourGuardConfig] = None
    max_open_positions: Optional[BehaviourGuardConfig] = None
    session_time_window: Optional[BehaviourGuardConfig] = None
    big_loss_cooldown: Optional[BehaviourGuardConfig] = None


def load_behaviour_config(path: str) -> BehaviourConfig:
    with open(path, "r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}
    cfg = data.get("behaviour", data)
    return BehaviourConfig(**cfg)


def create_guards_from_config(config: BehaviourConfig, *, initial_balance: float) -> List[BaseBehaviourGuard]:
    guards: List[BaseBehaviourGuard] = []

    if not config.enabled:
        logger.info("Behaviour guards disabled via config.")
        return guards

    if config.max_trades_per_day and config.max_trades_per_day.enabled:
        guards.append(
            MaxTradesPerDayGuard(
                max_trades_per_day=int(config.max_trades_per_day.params["max_trades_per_day"])
            )
        )

    if config.max_consecutive_losses and config.max_consecutive_losses.enabled:
        guards.append(
            MaxConsecutiveLossesGuard(
                max_consecutive_losses=int(config.max_consecutive_losses.params["max_consecutive_losses"])
            )
        )

    if config.cooldown_after_loss and config.cooldown_after_loss.enabled:
        guards.append(
            CooldownAfterLossGuard(
                cooldown_minutes=int(config.cooldown_after_loss.params["cooldown_minutes"])
            )
        )

    if config.daily_pnl and config.daily_pnl.enabled:
        params = config.daily_pnl.params
        guards.append(
            DailyPnLGuard(
                initial_balance=initial_balance,
                max_daily_loss_pct_initial=float(params["max_daily_loss_pct_initial"]),
                lock_in_profit_pct_initial=float(params.get("lock_in_profit_pct_initial", 0.0)),
            )
        )

    if config.daily_profit_target and config.daily_profit_target.enabled:
        params = config.daily_profit_target.params
        guards.append(
            DailyProfitTargetGuard(
                initial_balance=initial_balance,
                target_profit_pct_initial=float(params["target_profit_pct_initial"]),
                mode=str(params.get("mode", "soft_stop")),
            )
        )

    if config.max_open_positions and config.max_open_positions.enabled:
        guards.append(
            MaxOpenPositionsGuard(
                max_open_positions=int(config.max_open_positions.params["max_open_positions"])
            )
        )

    if config.session_time_window and config.session_time_window.enabled:
        params = config.session_time_window.params
        windows = [SessionTimeWindowConfig(**w) for w in params.get("windows", [])]
        guards.append(SessionTimeWindowGuard(windows=windows))

    if config.big_loss_cooldown and config.big_loss_cooldown.enabled:
        params = config.big_loss_cooldown.params
        guards.append(
            BigLossCooldownGuard(
                initial_balance=initial_balance,
                big_loss_pct_initial=float(params["big_loss_pct_initial"]),
                cooldown_minutes=int(params["cooldown_minutes"]),
            )
        )

    return guards
