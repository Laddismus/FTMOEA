from datetime import datetime

from afts_pro.risk.ftmo_rules import FtmoRiskConfig, FtmoRiskEngine


def test_soft_stop_blocks_new_entries_but_does_not_force_close():
    engine = FtmoRiskEngine(FtmoRiskConfig(daily_soft_stop_pct=3.0, daily_hard_stop_pct=4.0, initial_equity=100000.0))
    now = datetime(2025, 1, 1, 9, 0)
    engine.on_new_equity(97000.0, 0.0, now)  # 3% loss
    assert engine.is_daily_soft_breached() is True
    assert engine.should_force_close_all_positions() is False
    assert engine.can_open_new_trade() is False


def test_hard_stop_triggers_force_close():
    engine = FtmoRiskEngine(FtmoRiskConfig(daily_soft_stop_pct=3.0, daily_hard_stop_pct=4.0, initial_equity=100000.0))
    now = datetime(2025, 1, 1, 9, 0)
    engine.on_new_equity(95900.0, 0.0, now)  # 4.1% loss
    assert engine.is_daily_hard_breached() is True
    assert engine.should_force_close_all_positions() is True


def test_overall_safety_also_triggers_force_close():
    engine = FtmoRiskEngine(FtmoRiskConfig(safety_overall_loss_pct=8.5, initial_equity=100000.0))
    now = datetime(2025, 1, 1, 9, 0)
    engine.on_new_equity(91500.0, 0.0, now)
    assert engine.should_force_close_all_positions() is True
