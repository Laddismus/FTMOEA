from datetime import datetime, timedelta, timezone, date

import pytest

from research_lab.backend.core.backtests.models import BacktestBar
from research_lab.backend.core.risk_guard.ftmo_guard import FtmoRiskGuard
from research_lab.backend.core.risk_guard.models import FtmoRiskConfig, FtmoBreachType


def _bars(n: int, start: datetime | None = None, step_minutes: int = 60) -> list[BacktestBar]:
    start_ts = start or datetime.now(timezone.utc)
    return [BacktestBar(ts=start_ts + timedelta(minutes=i * step_minutes), open=1, high=1, low=1, close=1) for i in range(n)]


def test_ftmo_guard_no_breach() -> None:
    bars = _bars(3)
    returns = [0.01, -0.02, 0.01]
    guard = FtmoRiskGuard(FtmoRiskConfig(max_daily_loss_pct=0.05, max_total_loss_pct=0.1))

    summary = guard.evaluate(bars, returns)

    assert summary.passed is True
    assert summary.first_breach is None
    assert summary.worst_daily_drawdown_pct < 0.05
    assert summary.worst_total_drawdown_pct < 0.1


def test_ftmo_guard_daily_breach() -> None:
    bars = _bars(2)
    returns = [-0.06, 0.0]
    guard = FtmoRiskGuard(FtmoRiskConfig(max_daily_loss_pct=0.05, max_total_loss_pct=0.1))

    summary = guard.evaluate(bars, returns)

    assert summary.passed is False
    assert summary.first_breach is not None
    assert summary.first_breach.breach_type == FtmoBreachType.DAILY
    assert summary.worst_daily_drawdown_pct == pytest.approx(0.06, rel=1e-6)
    assert summary.worst_total_drawdown_pct < 0.1


def test_ftmo_guard_total_breach_across_days() -> None:
    base = datetime(2024, 1, 1, tzinfo=timezone.utc)
    bars = [
        BacktestBar(ts=base, open=1, high=1, low=1, close=1),
        BacktestBar(ts=base + timedelta(days=1), open=1, high=1, low=1, close=1),
        BacktestBar(ts=base + timedelta(days=2), open=1, high=1, low=1, close=1),
    ]
    returns = [-0.04, -0.04, -0.04]
    guard = FtmoRiskGuard(FtmoRiskConfig(max_daily_loss_pct=0.05, max_total_loss_pct=0.1))

    summary = guard.evaluate(bars, returns)

    assert summary.passed is False
    assert summary.first_breach is not None
    assert summary.first_breach.breach_type == FtmoBreachType.TOTAL
    assert summary.first_breach.day == date(2024, 1, 3)
    assert summary.worst_total_drawdown_pct >= 0.1
