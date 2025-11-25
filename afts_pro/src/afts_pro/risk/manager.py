from __future__ import annotations

import logging
from datetime import datetime

from afts_pro.exec.position_models import AccountState
from afts_pro.risk.base_policy import BaseRiskPolicy, RiskDecision

logger = logging.getLogger(__name__)


class RiskManager:
    """
    Thin wrapper that delegates to a risk policy.
    """

    def __init__(self, policy: BaseRiskPolicy) -> None:
        self._policy = policy

    def on_bar(self, account_state: AccountState, ts: datetime) -> RiskDecision:
        return self._policy.evaluate(account_state=account_state, ts=ts)

    def before_new_orders(self, account_state: AccountState, ts: datetime) -> RiskDecision:
        return self._policy.evaluate(account_state=account_state, ts=ts)
