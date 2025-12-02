from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from research_lab.backend.core.backtests.engine import RollingKpiBacktestEngine
from research_lab.backend.core.backtests.models import BacktestRequest, BacktestBar, PythonStrategyRef

STRATEGY_CODE = """
from research_lab.backend.core.python_strategies.interface import BasePythonStrategy
from research_lab.backend.core.backtests.models import BacktestBar, BacktestPositionState, TradingAction, PositionSide

class BarExecStrategy(BasePythonStrategy):
    strategy_key = "bar.exec"
    strategy_name = "Bar Exec Strategy"
    strategy_version = "1.0.0"

    def on_bar_trade(self, bar: BacktestBar, state: BacktestPositionState) -> TradingAction:
        if state.side == PositionSide.FLAT:
            return TradingAction.ENTER_LONG
        return TradingAction.EXIT
"""


def test_engine_python_mode_with_bars(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module_path = tmp_path / "bar_exec_strategy.py"
    module_path.write_text(STRATEGY_CODE, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))

    engine = RollingKpiBacktestEngine()
    request = BacktestRequest(
        mode="python",
        python_strategy=PythonStrategyRef(module_path="bar_exec_strategy", class_name="BarExecStrategy"),
        bars=[
            BacktestBar(ts=datetime.now(timezone.utc), open=1, high=1.1, low=0.9, close=1.0),
            BacktestBar(ts=datetime.now(timezone.utc) + timedelta(minutes=1), open=1.0, high=1.2, low=1.0, close=1.2),
            BacktestBar(ts=datetime.now(timezone.utc) + timedelta(minutes=2), open=1.2, high=1.3, low=1.1, close=1.1),
        ],
        window=2,
    )

    result = engine.run_backtest(request)

    assert result.strategy_metadata is not None
    assert result.strategy_metadata["type"] == "python"
    assert result.kpi_summary.trade_count == len(request.bars)
    assert result.kpi_summary.total_return == pytest.approx((1.2 - 1.0) / 1.0)
    assert result.trades is not None
    assert len(result.trades) == 1
