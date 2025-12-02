"""Backtest engine interfaces and stub implementation."""

from __future__ import annotations

import statistics
import uuid
from typing import Protocol

import numpy as np

from research_lab.backend.core.analytics.kpi_engine import RollingKpiEngine
from research_lab.backend.core.backtests.models import (
    BacktestEngineDetail,
    BacktestKpiSummary,
    BacktestRequest,
    BacktestResult,
    BacktestBar,
)
from research_lab.backend.core.risk_guard import FtmoRiskGuard
from research_lab.backend.core.strategy_execution import (
    SimpleExecutionContext,
    GraphStrategyExecutorAdapter,
    PythonStrategyExecutorAdapter,
)
from research_lab.backend.core.python_strategies.loader import import_strategy_class
from research_lab.backend.core.python_strategies.registry import PythonStrategyRegistry


class BacktestEngineInterface(Protocol):
    """Protocol for backtest engines."""

    def run_backtest(self, request: BacktestRequest) -> BacktestResult:
        """Execute a backtest and return the result."""


class RollingKpiBacktestEngine(BacktestEngineInterface):
    """Stub engine that computes KPIs over provided returns using rolling windows."""

    def __init__(self) -> None:
        self.kpi_engine = RollingKpiEngine()
        self.strategy_registry = PythonStrategyRegistry()

    def run_backtest(self, request: BacktestRequest) -> BacktestResult:
        executor = self._build_executor(request)
        ctx = SimpleExecutionContext(config=request.metadata, params=request.strategy_params)
        executor.initialize(ctx)

        returns_series: list[float]
        if request.mode == "python" and request.bars:
            returns_series = executor.run_bar_loop(request.bars)
        else:
            if not request.returns:
                raise ValueError("returns must not be empty when bars are not provided.")
            returns_series = request.returns

        returns = np.asarray(returns_series, dtype=float)
        if returns.size == 0:
            raise ValueError("returns must not be empty")
        window_kpis = self.kpi_engine.compute_rolling_kpis(returns.tolist(), request.window)

        total_return = float(np.sum(returns))
        mean_return = float(np.mean(returns))
        std_return = float(np.std(returns))

        positives = returns[returns > 0]
        negatives = returns[returns < 0]
        sum_pos = float(np.sum(positives)) if positives.size else 0.0
        sum_neg = float(np.sum(negatives)) if negatives.size else 0.0
        if sum_neg == 0.0:
            profit_factor = sum_pos if sum_pos > 0 else 0.0
        else:
            profit_factor = sum_pos / abs(sum_neg)
        win_rate = float(len(positives)) / float(len(returns))

        equity_curve = np.cumsum(returns)
        peak = -np.inf
        max_dd = 0.0
        for value in equity_curve:
            peak = max(peak, value)
            drawdown = peak - value
            max_dd = max(max_dd, drawdown)

        kpi_summary = BacktestKpiSummary(
            total_return=total_return,
            mean_return=mean_return,
            std_return=std_return,
            profit_factor=profit_factor,
            win_rate=win_rate,
            max_drawdown=float(max_dd),
            trade_count=len(returns),
        )

        engine_detail = BacktestEngineDetail(window_kpis=[kpi.model_dump() for kpi in window_kpis])

        strategy_metadata = executor.get_metadata()
        ftmo_risk_summary = None
        if request.ftmo_risk is not None and request.bars:
            guard = FtmoRiskGuard(config=request.ftmo_risk)
            ftmo_risk_summary = guard.evaluate(request.bars, returns_series)

        return BacktestResult(
            id=str(uuid.uuid4()),
            mode=request.mode,
            graph=request.graph,
            python_strategy=request.python_strategy,
            kpi_summary=kpi_summary,
            engine_detail=engine_detail,
            metadata=request.metadata,
            strategy_metadata=strategy_metadata,
            trades=getattr(executor, "trades", None),
            ftmo_risk_summary=ftmo_risk_summary,
        )

    def _build_executor(self, request: BacktestRequest):
        if request.mode == "graph":
            engine_config = (request.graph.engine_config if request.graph else None) or {
                "strategy": {"id": request.graph.graph_id if request.graph else None},
                "execution": {"mode": "strategy_graph_v1"},
            }
            dsl = request.graph.dsl if request.graph else {}
            return GraphStrategyExecutorAdapter(engine_config=engine_config, dsl=dsl)
        if request.mode == "python":
            if request.python_strategy is None:
                raise ValueError("python_strategy reference is required for mode 'python'")
            if request.python_strategy.key:
                meta = self.strategy_registry.get_strategy(request.python_strategy.key)
                if meta is None:
                    raise ValueError(f"Python strategy with key '{request.python_strategy.key}' not found in registry.")
                module_path = meta.module_path
                class_name = meta.class_name
            else:
                module_path = request.python_strategy.module_path
                class_name = request.python_strategy.class_name
            if not module_path or not class_name:
                raise ValueError("module_path and class_name are required for python strategy execution.")
            strategy_cls = import_strategy_class(module_path, class_name)
            return PythonStrategyExecutorAdapter(strategy_cls, cost_model=request.cost_model)
        raise ValueError(f"Unsupported backtest mode '{request.mode}'")


__all__ = ["BacktestEngineInterface", "RollingKpiBacktestEngine"]
