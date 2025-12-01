from datetime import datetime

from afts_pro.risk.base_policy import BaseRiskPolicy, RiskDecision
from afts_pro.risk.ftmo_plus import (
    FtmoPlusConfig,
    FtmoPlusEngine,
    NewsWindowConfig,
    ProfitTargetConfig,
    RollingRiskConfig,
    SessionRiskConfig,
    LossVelocityConfig,
    RiskStageConfig,
    ExposureCapsConfig,
    SpreadGuardConfig,
    TimeFenceConfig,
)
from afts_pro.risk.ftmo_rules import FtmoRiskConfig, FtmoRiskEngine
from afts_pro.risk.manager import RiskManager


class DummyPolicy(BaseRiskPolicy):
    def evaluate(self, account_state, ts):
        return RiskDecision(allow_new_orders=True, hard_stop_trading=False, reason=None, meta={})


class DummyAccount:
    def __init__(self):
        self.equity = 100000.0
        self.realized_pnl = -500.0
        self.trading_days_count = 5
        self.completed_trades_count = 8
        self.last_equity = 100000.0
        self.last_trade_slippage_pips = None

    def last_n_trade_pnls(self, n):
        return [1.0, -0.5, 0.8]


def _cfg():
    return FtmoPlusConfig(
        sessions=[SessionRiskConfig(name="London", start_time="08:00", end_time="17:00", max_session_loss_pct=1.5)],
        rolling=RollingRiskConfig(),
        loss_velocity=LossVelocityConfig(),
        stages=RiskStageConfig(),
        exposure_caps=ExposureCapsConfig(),
        spread_guard=SpreadGuardConfig(),
        news_windows=[
            NewsWindowConfig(name="Event", start_datetime="2025-03-07T13:20:00", end_datetime="2025-03-07T13:50:00")
        ],
        time_fences=[TimeFenceConfig(name="Allow", daily_start_time="08:00", daily_end_time="22:00")],
        profit_target=ProfitTargetConfig(hard_lock_pct=5.0, soft_lock_pct=3.0),
    )


def test_news_block_sets_meta_and_blocks_entry():
    ftmo_cfg = FtmoRiskConfig(initial_equity=100000.0)
    ftmo_engine = FtmoRiskEngine(ftmo_cfg)
    plus_engine = FtmoPlusEngine(_cfg())
    mgr = RiskManager(policy=DummyPolicy(), ftmo_engine=ftmo_engine, ftmo_plus_engine=plus_engine)
    account = DummyAccount()
    now = datetime.fromisoformat("2025-03-07T13:30:00")
    decision = mgr.before_new_orders(account, now)
    assert decision.allow_new_orders is False
    assert decision.meta.get("ftmo_plus_blocked_news") is True


def test_profit_hard_lock_blocks_entry():
    ftmo_cfg = FtmoRiskConfig(initial_equity=100000.0)
    ftmo_engine = FtmoRiskEngine(ftmo_cfg)
    plus_engine = FtmoPlusEngine(_cfg())
    mgr = RiskManager(policy=DummyPolicy(), ftmo_engine=ftmo_engine, ftmo_plus_engine=plus_engine)
    account = DummyAccount()
    account.equity = 106000.0  # 6% progress > hard lock 5%
    now = datetime(2025, 3, 7, 12, 0, 0)
    decision = mgr.before_new_orders(account, now)
    assert decision.allow_new_orders is False
    assert decision.meta.get("ftmo_plus_blocked_profit_hard") is True


def test_circuit_breaker_sets_force_flatten():
    ftmo_cfg = FtmoRiskConfig(initial_equity=100000.0)
    ftmo_engine = FtmoRiskEngine(ftmo_cfg)
    plus_engine = FtmoPlusEngine(_cfg())
    mgr = RiskManager(policy=DummyPolicy(), ftmo_engine=ftmo_engine, ftmo_plus_engine=plus_engine)
    account = DummyAccount()
    account.last_equity = 100000.0
    account.equity = 97000.0  # 3% instant loss
    now = datetime(2025, 3, 7, 12, 0, 0)
    decision = mgr.before_new_orders(account, now)
    assert decision.meta.get("ftmo_force_flatten") is True
    assert decision.meta.get("ftmo_plus_circuit_triggered") is True


def test_stability_kpis_populated():
    ftmo_cfg = FtmoRiskConfig(initial_equity=100000.0)
    ftmo_engine = FtmoRiskEngine(ftmo_cfg)
    plus_engine = FtmoPlusEngine(_cfg())
    mgr = RiskManager(policy=DummyPolicy(), ftmo_engine=ftmo_engine, ftmo_plus_engine=plus_engine)
    account = DummyAccount()
    now = datetime(2025, 3, 7, 12, 0, 0)
    decision = mgr.before_new_orders(account, now)
    assert "ftmo_plus_pf" in decision.meta
    assert "ftmo_plus_winrate" in decision.meta
    assert "ftmo_plus_pnl_std" in decision.meta
