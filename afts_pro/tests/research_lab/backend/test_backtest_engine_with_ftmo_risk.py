from datetime import datetime, timezone, timedelta

from research_lab.backend.core.backtests.engine import RollingKpiBacktestEngine
from research_lab.backend.core.backtests.models import BacktestRequest, BacktestBar
from research_lab.backend.core.risk_guard.models import FtmoRiskConfig, FtmoBreachType


def test_engine_includes_ftmo_risk_summary() -> None:
    engine = RollingKpiBacktestEngine()
    start = datetime.now(timezone.utc)
    bars = [
        BacktestBar(ts=start, open=1, high=1.1, low=0.9, close=1.0),
        BacktestBar(ts=start + timedelta(minutes=1), open=1, high=1.0, low=0.8, close=0.8),
    ]
    request = BacktestRequest(
        mode="graph",
        returns=[-0.06, 0.0],
        bars=bars,
        window=2,
        ftmo_risk=FtmoRiskConfig(max_daily_loss_pct=0.05, max_total_loss_pct=0.1),
    )

    result = engine.run_backtest(request)

    assert result.ftmo_risk_summary is not None
    summary = result.ftmo_risk_summary
    assert summary.passed is False
    assert summary.first_breach is not None
    assert summary.first_breach.breach_type == FtmoBreachType.DAILY
