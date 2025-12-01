from datetime import datetime, timedelta

from afts_pro.risk.ftmo_rules import FtmoRiskConfig, FtmoRiskEngine


def test_initialization_uses_current_equity_if_not_set():
    engine = FtmoRiskEngine(FtmoRiskConfig(initial_equity=None))
    now = datetime(2025, 1, 1, 9, 0)
    engine.ensure_initialized(100000.0, now)
    assert engine.state is not None
    assert engine.state.initial_equity == 100000.0


def test_daily_reset_changes_day_state():
    engine = FtmoRiskEngine(FtmoRiskConfig(initial_equity=None))
    now = datetime(2025, 1, 1, 9, 0)
    engine.on_new_equity(100000.0, 0.0, now)
    next_day = now + timedelta(days=1)
    engine.on_new_equity(98000.0, 0.0, next_day)
    assert engine.state is not None
    assert engine.state.day_state.day == next_day.date()
    assert engine.current_daily_loss_pct() == 0.0


def test_overall_safety_trigger():
    engine = FtmoRiskEngine(FtmoRiskConfig(initial_equity=100000.0, safety_overall_loss_pct=8.5))
    now = datetime(2025, 1, 1, 9, 0)
    engine.on_new_equity(91500.0, 0.0, now)
    assert engine.is_overall_safety_breached() is True


def test_daily_safety_trigger():
    engine = FtmoRiskEngine(FtmoRiskConfig(initial_equity=100000.0, daily_hard_stop_pct=4.0, daily_soft_stop_pct=3.0))
    now = datetime(2025, 1, 1, 9, 0)
    engine.on_new_equity(96000.0, 0.0, now)
    assert engine.is_daily_hard_breached() is True


def test_can_open_new_trade_false_when_safety_breached():
    engine = FtmoRiskEngine(FtmoRiskConfig(initial_equity=100000.0, safety_overall_loss_pct=8.5))
    now = datetime(2025, 1, 1, 9, 0)
    engine.on_new_equity(91500.0, 0.0, now)
    assert engine.can_open_new_trade() is False
