from datetime import datetime

from afts_pro.exec.position_models import AccountState
from afts_pro.risk.base_policy import RiskDecision, BaseRiskPolicy
from afts_pro.risk.ftmo_rules import FtmoRiskEngine, FtmoRiskConfig
from afts_pro.risk.manager import RiskManager


class DummyPolicy(BaseRiskPolicy):
    name: str = "dummy"

    def evaluate(self, account_state: AccountState, ts: datetime) -> RiskDecision:
        return RiskDecision(allow_new_orders=True, hard_stop_trading=False, reason=None, meta={})

    def on_new_day(self, account_state: AccountState, ts: datetime) -> None:
        return None


def test_soft_stop_sets_block_flag_but_no_force_flatten():
    ftmo = FtmoRiskEngine(FtmoRiskConfig(initial_equity=100000.0, daily_soft_stop_pct=3.0, daily_hard_stop_pct=4.0))
    now = datetime(2025, 1, 1, 9, 0)
    ftmo.on_new_equity(97000.0, 0.0, now)
    manager = RiskManager(DummyPolicy(), ftmo_engine=ftmo)
    decision = manager.before_new_orders(AccountState(balance=0, equity=97000.0, realized_pnl=0.0, unrealized_pnl=0.0, fees_total=0.0), now)
    assert decision.allow_new_orders is False
    assert decision.meta.get("ftmo_blocked") is True
    assert decision.meta.get("ftmo_force_flatten") is None


def test_hard_stop_sets_force_flatten_flag():
    ftmo = FtmoRiskEngine(FtmoRiskConfig(initial_equity=100000.0, daily_soft_stop_pct=3.0, daily_hard_stop_pct=4.0))
    now = datetime(2025, 1, 1, 9, 0)
    ftmo.on_new_equity(95900.0, 0.0, now)
    manager = RiskManager(DummyPolicy(), ftmo_engine=ftmo)
    decision = manager.before_new_orders(AccountState(balance=0, equity=95900.0, realized_pnl=0.0, unrealized_pnl=0.0, fees_total=0.0), now)
    assert decision.meta.get("ftmo_force_flatten") is True
