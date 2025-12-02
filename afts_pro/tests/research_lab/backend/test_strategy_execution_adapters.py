from pathlib import Path

import pytest

from research_lab.backend.core.strategy_execution.context import SimpleExecutionContext
from research_lab.backend.core.strategy_execution.graph_executor import GraphStrategyExecutorAdapter
from research_lab.backend.core.strategy_execution.python_executor import PythonStrategyExecutorAdapter
from research_lab.backend.core.python_strategies.interface import BasePythonStrategy


def test_graph_strategy_executor_metadata() -> None:
    engine_config = {"strategy": {"id": "g1", "name": "Graph One"}, "execution": {"mode": "strategy_graph_v1"}}
    dsl = {"id": "g1", "name": "Graph One"}
    executor = GraphStrategyExecutorAdapter(engine_config=engine_config, dsl=dsl)
    ctx = SimpleExecutionContext(config={"foo": "bar"}, params={"p": 1})

    executor.initialize(ctx)
    meta = executor.get_metadata()
    assert meta["type"] == "graph"
    assert meta["engine_mode"] == "strategy_graph_v1"
    assert meta["graph_id"] == "g1"
    assert meta["params"] == {"p": 1}


STRATEGY_CODE = """
from research_lab.backend.core.python_strategies.interface import BasePythonStrategy

class AdapterStrategy(BasePythonStrategy):
    strategy_key = "adapter.test"
    strategy_name = "Adapter Test Strategy"
"""


def test_python_strategy_executor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module_path = tmp_path / "adapter_strategy.py"
    module_path.write_text(STRATEGY_CODE, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))

    import sys
    if "adapter_strategy" in sys.modules:
        sys.modules.pop("adapter_strategy")

    from research_lab.backend.core.python_strategies.loader import import_strategy_class

    strategy_cls = import_strategy_class("adapter_strategy", "AdapterStrategy")
    executor = PythonStrategyExecutorAdapter(strategy_cls)
    ctx = SimpleExecutionContext(config={"foo": "bar"}, params={"risk": 1.0})

    executor.initialize(ctx)
    meta = executor.get_metadata()
    assert meta["type"] == "python"
    assert meta["name"] == "Adapter Test Strategy"
    assert meta["params"]["risk"] == 1.0
