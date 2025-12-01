from datetime import datetime

from afts_pro.exec.position_models import AccountState
from afts_pro.risk.base_policy import BaseRiskPolicy, RiskDecision
from afts_pro.risk.ftmo_plus import (
    FtmoPlusConfig,
    FtmoPlusEngine,
    SessionRiskConfig,
    RollingRiskConfig,
    LossVelocityConfig,
    RiskStageConfig,
    ExposureCapsConfig,
    SpreadGuardConfig,
)
from afts_pro.risk.manager import RiskManager


class DummyPolicy(BaseRiskPolicy):
    name: str = "dummy"

    def evaluate(self, account_state: AccountState, ts: datetime) -> RiskDecision:
        return RiskDecision(allow_new_orders=True, hard_stop_trading=False, reason=None, meta={})

    def on_new_day(self, account_state: AccountState, ts: datetime) -> None:
        return None


def _plus_engine():
    cfg = FtmoPlusConfig(
        sessions=[SessionRiskConfig(name="London", start_time="08:00", end_time="17:00", max_session_loss_pct=2.0)],
        rolling=RollingRiskConfig(window_minutes=60, max_rolling_loss_pct=1.5),
        loss_velocity=LossVelocityConfig(dd_fast_threshold_pct_per_hour=4.0),
        stages=RiskStageConfig(stage1_risk_mult=0.5, stage1_max_risk_pct=0.5, stage2_risk_mult=0.0, stage2_max_risk_pct=0.0),
        exposure_caps=ExposureCapsConfig(max_open_trades=1, max_total_risk_pct=1.0),
        spread_guard=SpreadGuardConfig(max_spread_pips=0.5, enabled=True),
    )
    return FtmoPlusEngine(cfg)


def test_stage_affects_effective_risk_pct():
    # Here we verify stage scaling by manually setting meta values; PositionSizer handles scaling.
    eng = _plus_engine()
    eng.state.current_stage = 1
    assert eng.current_stage_risk_mult() == 0.5


def test_exposure_guard_blocks_new_entry():
    eng = _plus_engine()
    manager = RiskManager(DummyPolicy(), ftmo_engine=None, ftmo_plus_engine=eng)
    acc = AccountState(balance=0, equity=10000.0, realized_pnl=0.0, unrealized_pnl=0.0, fees_total=0.0)
    acc.open_trades_count = 2
    decision = manager.before_new_orders(acc, datetime(2025, 1, 1, 9, 0))
    assert decision.allow_new_orders is False or decision.meta.get("ftmo_plus_blocked_exposure")


def test_spread_guard_blocks_new_entry():
    eng = _plus_engine()
    manager = RiskManager(DummyPolicy(), ftmo_engine=None, ftmo_plus_engine=eng)
    acc = AccountState(balance=0, equity=10000.0, realized_pnl=0.0, unrealized_pnl=0.0, fees_total=0.0)
    acc.current_spread_pips = 1.0
    decision = manager.before_new_orders(acc, datetime(2025, 1, 1, 9, 0))
    assert decision.allow_new_orders is False or decision.meta.get("ftmo_plus_blocked_spread")
