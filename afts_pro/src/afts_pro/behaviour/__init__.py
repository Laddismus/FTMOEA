"""
Behaviour guards to protect trading discipline.
"""

from .base_guard import BaseBehaviourGuard, BehaviourDecision, TradeStats
from .guards import (
    BigLossCooldownGuard,
    CooldownAfterLossGuard,
    DailyPnLGuard,
    DailyProfitTargetGuard,
    MaxConsecutiveLossesGuard,
    MaxOpenPositionsGuard,
    MaxTradesPerDayGuard,
    SessionTimeWindowGuard,
)
from .manager import BehaviourManager
from .config import BehaviourConfig, create_guards_from_config, load_behaviour_config

__all__ = [
    "BehaviourDecision",
    "BaseBehaviourGuard",
    "TradeStats",
    "MaxTradesPerDayGuard",
    "MaxConsecutiveLossesGuard",
    "CooldownAfterLossGuard",
    "DailyPnLGuard",
    "DailyProfitTargetGuard",
    "MaxOpenPositionsGuard",
    "SessionTimeWindowGuard",
    "BigLossCooldownGuard",
    "BehaviourManager",
    "BehaviourConfig",
    "load_behaviour_config",
    "create_guards_from_config",
]
