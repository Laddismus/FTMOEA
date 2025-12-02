from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from research_lab.backend.core.backtests.models import BacktestBar
from research_lab.backend.core.strategy_execution.python_executor import PythonStrategyExecutorAdapter
from research_lab.backend.core.strategy_execution.context import SimpleExecutionContext
from research_lab.backend.core.python_strategies.loader import import_strategy_class


STRATEGY_CODE = """
from research_lab.backend.core.python_strategies.interface import BasePythonStrategy
from research_lab.backend.core.backtests.models import BacktestBar

class MyBarStrategy(BasePythonStrategy):
    strategy_key = "test.bar_strategy"
    strategy_name = "Bar Strategy"
    strategy_version = "1.0.0"

    def __init__(self):
        super().__init__()
        self.last_close = None

    def on_bar(self, bar: BacktestBar) -> float:
        if self.last_close is None:
            self.last_close = bar.close
            return 0.0
        ret = (bar.close - self.last_close) / self.last_close
        self.last_close = bar.close
        return ret
"""


def test_python_strategy_bar_loop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module_path = tmp_path / "bar_strategy.py"
    module_path.write_text(STRATEGY_CODE, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))

    strategy_cls = import_strategy_class("bar_strategy", "MyBarStrategy")
    executor = PythonStrategyExecutorAdapter(strategy_cls)
    executor.initialize(SimpleExecutionContext(config={}, params={}))

    bars = [
        BacktestBar(ts=datetime.now(timezone.utc), open=1, high=1.1, low=0.9, close=1.0),
        BacktestBar(ts=datetime.now(timezone.utc) + timedelta(minutes=1), open=1.0, high=1.2, low=1.0, close=1.2),
        BacktestBar(ts=datetime.now(timezone.utc) + timedelta(minutes=2), open=1.2, high=1.3, low=1.1, close=1.1),
    ]

    returns = executor.run_bar_loop(bars)
    assert returns[0] == 0.0
    assert pytest.approx(returns[1], rel=1e-6) == 0.2
    assert pytest.approx(returns[2], rel=1e-6) == -0.0833333
