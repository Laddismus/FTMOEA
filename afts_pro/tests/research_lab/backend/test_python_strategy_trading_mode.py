from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from research_lab.backend.core.backtests.models import BacktestBar, BacktestPositionState, TradingAction, PositionSide
from research_lab.backend.core.strategy_execution.python_executor import PythonStrategyExecutorAdapter
from research_lab.backend.core.strategy_execution.context import SimpleExecutionContext
from research_lab.backend.core.python_strategies.loader import import_strategy_class


STRATEGY_CODE = """
from research_lab.backend.core.python_strategies.interface import BasePythonStrategy
from research_lab.backend.core.backtests.models import BacktestBar, BacktestPositionState, TradingAction, PositionSide

class MyTradingStrategy(BasePythonStrategy):
    strategy_key = "test.trading_strategy"
    strategy_name = "Trading Strategy"
    strategy_version = "1.0.0"
    def on_bar_trade(self, bar: BacktestBar, state: BacktestPositionState) -> TradingAction:
        if state.side == PositionSide.FLAT:
            return TradingAction.ENTER_LONG
        # exit on last bar
        return TradingAction.EXIT
"""


def test_trading_strategy_runs_bar_loop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module_path = tmp_path / "trading_strategy.py"
    module_path.write_text(STRATEGY_CODE, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))

    strategy_cls = import_strategy_class("trading_strategy", "MyTradingStrategy")
    executor = PythonStrategyExecutorAdapter(strategy_cls)
    executor.initialize(SimpleExecutionContext(config={}, params={}))

    bars = [
        BacktestBar(ts=datetime.now(timezone.utc), open=1.0, high=1.1, low=0.9, close=1.0),
        BacktestBar(ts=datetime.now(timezone.utc) + timedelta(minutes=1), open=1.0, high=1.2, low=1.0, close=1.2),
    ]

    returns = executor.run_bar_loop(bars)

    assert len(returns) == len(bars)
    assert returns[0] == 0.0
    assert returns[1] > 0
    assert len(executor.trades) == 1
    trade = executor.trades[0]
    assert trade.entry_price == 1.0
    assert trade.exit_price == 1.2


STRATEGY_CODE_SHORT = """
from research_lab.backend.core.python_strategies.interface import BasePythonStrategy
from research_lab.backend.core.backtests.models import BacktestBar, BacktestPositionState, TradingAction, PositionSide

class MyShortStrategy(BasePythonStrategy):
    strategy_key = "test.short_strategy"
    strategy_name = "Short Strategy"
    strategy_version = "1.0.0"
    def on_bar_trade(self, bar: BacktestBar, state: BacktestPositionState) -> TradingAction:
        if state.side == PositionSide.FLAT:
            return TradingAction.ENTER_SHORT
        return TradingAction.EXIT
"""


def test_trading_strategy_short_with_fees(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module_path = tmp_path / "short_strategy.py"
    module_path.write_text(STRATEGY_CODE_SHORT, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))

    strategy_cls = import_strategy_class("short_strategy", "MyShortStrategy")
    executor = PythonStrategyExecutorAdapter(strategy_cls, cost_model=None)
    executor.initialize(SimpleExecutionContext(config={}, params={"size": 1.0}))

    bars = [
        BacktestBar(ts=datetime.now(timezone.utc), open=1.0, high=1.1, low=0.9, close=1.0),
        BacktestBar(ts=datetime.now(timezone.utc) + timedelta(minutes=1), open=1.0, high=1.0, low=0.8, close=0.8),
    ]

    returns = executor.run_bar_loop(bars)
    assert returns[-1] > 0  # profitable short
    assert executor.trades and executor.trades[0].side == PositionSide.SHORT
