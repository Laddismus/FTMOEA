"""Adapter for Python strategy execution."""

from __future__ import annotations

from typing import Any, Dict, Optional, Type, Sequence

from research_lab.backend.core.strategy_execution.interface import StrategyExecutor, StrategyExecutionContext
from research_lab.backend.core.python_strategies.interface import PythonStrategyInterface, BasePythonStrategy
from research_lab.backend.core.python_strategies.loader import extract_metadata
from research_lab.backend.core.backtests.models import (
    BacktestBar,
    BacktestPositionState,
    TradingAction,
    BacktestTrade,
    PositionSide,
    BacktestCostModel,
)


class PythonStrategyExecutorAdapter(StrategyExecutor):
    """StrategyExecutor adapter for Python strategy classes."""

    def __init__(self, strategy_cls: Type[PythonStrategyInterface], cost_model: Optional[BacktestCostModel] = None) -> None:
        self.strategy_cls = strategy_cls
        self.strategy_instance: Optional[PythonStrategyInterface] = None
        self._metadata: Dict[str, Any] = {}
        self.trades: list[BacktestTrade] = []
        self.cost_model = cost_model or BacktestCostModel()

    def initialize(self, ctx: StrategyExecutionContext) -> None:
        self.strategy_instance = self.strategy_cls()
        self.strategy_instance.initialize(ctx.params)
        meta = extract_metadata(self.strategy_cls).model_dump()
        meta.update({"type": "python", "config": ctx.config, "params": ctx.params})
        self._metadata = meta

    def get_metadata(self) -> Dict[str, Any]:
        return self._metadata

    def run_bar_loop(self, bars: Sequence[BacktestBar]) -> list[float]:
        """Execute strategy on each bar and return per-bar returns."""

        if self.strategy_instance is None:
            raise RuntimeError("Strategy must be initialized before running bar loop.")
        if self._has_custom_on_bar_trade():
            return self._run_trading_mode(bars)
        returns: list[float] = []
        for bar in bars:
            ret = self.strategy_instance.on_bar(bar)
            if not isinstance(ret, (float, int)):
                raise TypeError("on_bar must return a numeric value.")
            returns.append(float(ret))
        return returns

    def _has_custom_on_bar_trade(self) -> bool:
        base_method = BasePythonStrategy.on_bar_trade
        current_method = type(self.strategy_instance).on_bar_trade  # type: ignore[arg-type]
        return current_method is not base_method

    def _run_trading_mode(self, bars: Sequence[BacktestBar]) -> list[float]:
        state = BacktestPositionState()
        returns: list[float] = []
        self.trades = []
        size = 1.0
        if self.strategy_instance and hasattr(self.strategy_instance, "params"):
            size = getattr(self.strategy_instance, "params", {}).get("size", 1.0) or 1.0
        for idx, bar in enumerate(bars):
            action = self.strategy_instance.on_bar_trade(bar, state)  # type: ignore[arg-type]
            bar_return = 0.0
            if state.side == PositionSide.FLAT:
                if action == TradingAction.ENTER_LONG:
                    state.side = PositionSide.LONG
                    state.size = float(size)
                    state.entry_price = self._apply_slippage(bar.close, is_entry=True)
                    state.entry_ts = bar.ts
                    state.costs_accrued += self._fee_cost(state.entry_price, state.size)
                elif action == TradingAction.ENTER_SHORT:
                    state.side = PositionSide.SHORT
                    state.size = float(size)
                    state.entry_price = self._apply_slippage(bar.close, is_entry=True)
                    state.entry_ts = bar.ts
                    state.costs_accrued += self._fee_cost(state.entry_price, state.size)
            elif state.side == PositionSide.LONG:
                if action == TradingAction.EXIT and state.entry_price is not None:
                    exit_price = self._apply_slippage(bar.close, is_entry=False)
                    gross_ret = (exit_price - state.entry_price) / state.entry_price
                    fees = self._fee_cost(exit_price, state.size)
                    net_ret = (gross_ret * state.size) - fees - state.costs_accrued
                    bar_return = net_ret
                    state.equity *= 1.0 + net_ret
                    self.trades.append(
                        BacktestTrade(
                            entry_ts=state.entry_ts or (bars[idx - 1].ts if idx > 0 else bar.ts),
                            exit_ts=bar.ts,
                            entry_price=state.entry_price,
                            exit_price=exit_price,
                            pnl=(exit_price - state.entry_price) * state.size,
                            return_=net_ret,
                            gross_return=gross_ret,
                            net_return=net_ret,
                            fees=fees + state.costs_accrued,
                            side=PositionSide.LONG,
                            size=state.size,
                        )
                    )
                    state.side = PositionSide.FLAT
                    state.size = 0.0
                    state.entry_price = None
                    state.entry_ts = None
                    state.costs_accrued = 0.0
            elif state.side == PositionSide.SHORT:
                if action == TradingAction.EXIT and state.entry_price is not None:
                    exit_price = self._apply_slippage(bar.close, is_entry=False)
                    gross_ret = (state.entry_price - exit_price) / state.entry_price
                    fees = self._fee_cost(exit_price, state.size)
                    net_ret = (gross_ret * state.size) - fees - state.costs_accrued
                    bar_return = net_ret
                    state.equity *= 1.0 + net_ret
                    self.trades.append(
                        BacktestTrade(
                            entry_ts=state.entry_ts or (bars[idx - 1].ts if idx > 0 else bar.ts),
                            exit_ts=bar.ts,
                            entry_price=state.entry_price,
                            exit_price=exit_price,
                            pnl=(state.entry_price - exit_price) * state.size,
                            return_=net_ret,
                            gross_return=gross_ret,
                            net_return=net_ret,
                            fees=fees + state.costs_accrued,
                            side=PositionSide.SHORT,
                            size=state.size,
                        )
                    )
                    state.side = PositionSide.FLAT
                    state.size = 0.0
                    state.entry_price = None
                    state.entry_ts = None
                    state.costs_accrued = 0.0
            returns.append(bar_return)
        return returns

    def _apply_slippage(self, price: float, is_entry: bool) -> float:
        if self.cost_model.slippage_rate <= 0:
            return price
        delta = price * self.cost_model.slippage_rate
        return price + delta if is_entry else price - delta

    def _fee_cost(self, price: float, size: float) -> float:
        if self.cost_model.fee_rate <= 0:
            return 0.0
        notional = price * size
        return notional * self.cost_model.fee_rate


__all__ = ["PythonStrategyExecutorAdapter"]
