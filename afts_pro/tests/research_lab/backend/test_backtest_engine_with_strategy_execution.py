from pathlib import Path

import pytest

from research_lab.backend.core.backtests.engine import RollingKpiBacktestEngine
from research_lab.backend.core.backtests.models import BacktestRequest, StrategyGraphRef, PythonStrategyRef

STRATEGY_CODE = """
from research_lab.backend.core.python_strategies.interface import BasePythonStrategy

class ExecTestStrategy(BasePythonStrategy):
    strategy_key = "exec.test"
    strategy_name = "Exec Test Strategy"
"""


def test_engine_with_graph_executor() -> None:
    engine = RollingKpiBacktestEngine()
    graph_ref = StrategyGraphRef(engine_config={"strategy": {"id": "g1"}, "execution": {"mode": "strategy_graph_v1"}}, dsl={"id": "g1"})
    request = BacktestRequest(mode="graph", graph=graph_ref, returns=[1, -1, 2], window=2, strategy_params={"alpha": 1})

    result = engine.run_backtest(request)

    assert result.strategy_metadata is not None
    assert result.strategy_metadata["type"] == "graph"
    assert result.strategy_metadata["graph_id"] == "g1"


def test_engine_with_python_executor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module_path = tmp_path / "exec_strategy.py"
    module_path.write_text(STRATEGY_CODE, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))

    from research_lab.backend.core.python_strategies.registry import PythonStrategyRegistry
    from research_lab.backend.core.python_strategies.models import PythonStrategyMetadata

    registry = PythonStrategyRegistry(registry_dir=tmp_path / "reg")
    registry.register_strategy(
        PythonStrategyMetadata(
            key="exec.test",
            name="Exec Test Strategy",
            version="1.0.0",
            module_path="exec_strategy",
            class_name="ExecTestStrategy",
            description=None,
            tags=[],
            params_schema={},
        )
    )

    engine = RollingKpiBacktestEngine()
    engine.strategy_registry = registry
    request = BacktestRequest(
        mode="python",
        python_strategy=PythonStrategyRef(key="exec.test"),
        returns=[0.5, -0.2],
        window=2,
        strategy_params={"risk": 0.5},
    )

    result = engine.run_backtest(request)
    assert result.strategy_metadata is not None
    assert result.strategy_metadata["type"] == "python"
    assert result.strategy_metadata["name"] == "Exec Test Strategy"
