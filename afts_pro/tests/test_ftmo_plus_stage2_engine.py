from datetime import datetime

from afts_pro.risk.ftmo_plus import (
    CircuitBreakerConfig,
    FtmoPlusConfig,
    FtmoPlusEngine,
    LossVelocityConfig,
    NewsWindowConfig,
    PerformanceStabilityConfig,
    ProfitTargetConfig,
    RollingRiskConfig,
    SessionRiskConfig,
    SpreadGuardConfig,
    RiskStageConfig,
    ExposureCapsConfig,
    TimeFenceConfig,
)


def _cfg():
    return FtmoPlusConfig(
        sessions=[SessionRiskConfig(name="London", start_time="08:00", end_time="17:00", max_session_loss_pct=1.5)],
        rolling=RollingRiskConfig(),
        loss_velocity=LossVelocityConfig(),
        stages=RiskStageConfig(),
        exposure_caps=ExposureCapsConfig(),
        spread_guard=SpreadGuardConfig(),
        news_windows=[
            NewsWindowConfig(
                name="NFP",
                start_datetime="2025-03-07T13:20:00",
                end_datetime="2025-03-07T13:50:00",
            )
        ],
        time_fences=[TimeFenceConfig(name="Regular", daily_start_time="08:00", daily_end_time="22:00")],
        profit_target=ProfitTargetConfig(),
        stability=PerformanceStabilityConfig(),
        circuit_breaker=CircuitBreakerConfig(),
    )


def test_news_window_blocks_when_inside():
    cfg = _cfg()
    eng = FtmoPlusEngine(cfg)
    now = datetime.fromisoformat("2025-03-07T13:30:00")
    assert eng.is_in_news_window(now) is True


def test_time_fence_allow_only_behavior():
    cfg = _cfg()
    eng = FtmoPlusEngine(cfg)
    inside = datetime.fromisoformat("2025-03-07T12:00:00")
    outside = datetime.fromisoformat("2025-03-07T23:00:00")
    assert eng.is_allowed_by_time_fence(inside) is True
    assert eng.is_allowed_by_time_fence(outside) is False


def test_profit_target_progress_and_lock():
    cfg = _cfg()
    eng = FtmoPlusEngine(cfg)
    progress = eng.profit_target_progress_pct(current_equity=110000.0, initial_equity=100000.0)
    assert progress == 10.0
    assert eng.is_profit_soft_lock(progress * 10) is True
    assert eng.is_profit_hard_lock(progress * 10) is True


def test_stability_kpis_basic():
    cfg = _cfg()
    eng = FtmoPlusEngine(cfg)
    kpis = eng.compute_stability_kpis([1.0, -1.0, 2.0, -0.5, 1.0])
    assert kpis["profit_factor"] > 0
    assert 0 <= kpis["winrate"] <= 1
    assert kpis["pnl_std"] >= 0


def test_circuit_breaker_triggers_on_instant_loss():
    cfg = _cfg()
    eng = FtmoPlusEngine(cfg)
    now = datetime(2025, 1, 1, 10, 0, 0)
    assert (
        eng.check_circuit_breaker(
            last_equity=100000.0, current_equity=97500.0, last_trade_slippage_pips=None, now=now
        )
        is True
    )
    assert eng.is_circuit_breaker_active(now) is True
