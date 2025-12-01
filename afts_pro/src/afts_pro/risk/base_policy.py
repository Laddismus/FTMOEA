from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field

from afts_pro.exec.position_models import AccountState

logger = logging.getLogger(__name__)


class RiskDecision(BaseModel):
    allow_new_orders: bool
    hard_stop_trading: bool
    reason: Optional[str] = Field(default=None)
    meta: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}


class BaseRiskPolicy(ABC):
    name: str

    def __init__(self, name: str = "risk_policy") -> None:
        self.name = name

    def on_new_day(self, account_state: AccountState, ts: datetime) -> None:  # pragma: no cover - default no-op
        """
        Hook to reset daily counters; invoked when a new trading day starts.
        """
        logger.debug("on_new_day default no-op for policy %s", self.name)

    @abstractmethod
    def evaluate(self, *, account_state: AccountState, ts: datetime) -> RiskDecision:
        """
        Evaluate current risk state and return a decision.
        """
