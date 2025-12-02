import pytest

from research_lab.backend.core.backtests.engine import RollingKpiBacktestEngine
from research_lab.backend.core.backtests.models import BacktestRequest, StrategyGraphRef, PythonStrategyRef


def test_backtest_engine_graph_mode() -> None:
    engine = RollingKpiBacktestEngine()
    returns = [1, -1, 2, -2, 3, -3]
    request = BacktestRequest(mode="graph", graph=StrategyGraphRef(graph_id="g1"), returns=returns, window=3)

    result = engine.run_backtest(request)

    assert result.mode == "graph"
    assert result.graph is not None
    assert result.kpi_summary.total_return == sum(returns)
    assert result.kpi_summary.trade_count == len(returns)
    assert result.kpi_summary.win_rate >= 0
    assert result.kpi_summary.max_drawdown >= 0


def test_backtest_engine_python_mode() -> None:
    engine = RollingKpiBacktestEngine()
    # ensure registry contains the strategy reference
    from research_lab.backend.core.python_strategies.models import PythonStrategyMetadata
    engine.strategy_registry.register_strategy(
        PythonStrategyMetadata(
            key="strat1",
            name="Strat One",
            version="1.0.0",
            module_path="math",
            class_name="sqrt",  # intentionally invalid class; execution will fail before usage due to interface check
            description=None,
            tags=[],
            params_schema={},
        )
    )
    returns = [0.5, 0.5, -0.2]
    request = BacktestRequest(
        mode="python",
        python_strategy=PythonStrategyRef(key="strat1"),
        returns=returns,
        window=2,
    )

    with pytest.raises(TypeError):
        engine.run_backtest(request)
