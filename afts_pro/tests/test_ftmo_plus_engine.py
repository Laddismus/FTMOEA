from datetime import datetime, timedelta

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


def _cfg():
    return FtmoPlusConfig(
        sessions=[SessionRiskConfig(name="London", start_time="08:00", end_time="17:00", max_session_loss_pct=1.5)],
        rolling=RollingRiskConfig(window_minutes=60, max_rolling_loss_pct=1.5),
        loss_velocity=LossVelocityConfig(dd_fast_threshold_pct_per_hour=4.0),
        stages=RiskStageConfig(),
        exposure_caps=ExposureCapsConfig(),
        spread_guard=SpreadGuardConfig(),
    )


def test_rolling_loss_pct_within_window():
    cfg = _cfg()
    eng = FtmoPlusEngine(cfg)
    now = datetime(2025, 1, 1, 9, 0)
    eng.update_rolling_equity(now, 100.0)
    eng.update_rolling_equity(now + timedelta(minutes=30), 98.5)
    loss = eng.rolling_loss_pct()
    assert loss > 0


def test_loss_velocity_classification():
    cfg = _cfg()
    eng = FtmoPlusEngine(cfg)
    now = datetime(2025, 1, 1, 9, 0)
    eng.update_rolling_equity(now, 100.0)
    eng.update_rolling_equity(now + timedelta(minutes=30), 97.0)
    vel = eng.loss_velocity_pct_per_hour()
    assert vel > 0


def test_stage_escalation_on_high_rolling_dd_or_velocity():
    cfg = _cfg()
    eng = FtmoPlusEngine(cfg)
    eng.update_stage(
        ftmo_daily_loss_pct=0.0,
        ftmo_overall_loss_pct=0.0,
        rolling_loss_pct=2.0,
        loss_velocity_pct_per_hour=0.0,
        session_loss_pct=None,
        num_recent_trades=10,
    )
    assert eng.state.current_stage >= 1


def test_exposure_caps_block_excess_positions():
    cfg = _cfg()
    eng = FtmoPlusEngine(cfg)
    assert eng.exposure_allows_new_trade(open_trades_count=5, total_open_risk_pct=0.0) is False


def test_spread_guard_blocks_when_spread_too_high():
    cfg = _cfg()
    eng = FtmoPlusEngine(cfg)
    assert eng.spread_allows_new_trade(spread_pips=1.0) is False
