"""Interfaces for Python-based strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from research_lab.backend.core.backtests.models import BacktestBar, BacktestPositionState, TradingAction


class PythonStrategyInterface(ABC):
    """Minimal interface for Python strategies executable by AFTS Core."""

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata describing this strategy."""

    @abstractmethod
    def initialize(self, params: Optional[Dict[str, Any]] = None) -> None:
        """Initialize internal state with provided parameters."""

    def on_bar(self, bar: "BacktestBar") -> float:
        """Process a single bar and return per-bar return (override in concrete strategies)."""

        raise NotImplementedError

    def on_bar_trade(self, bar: "BacktestBar", state: "BacktestPositionState") -> "TradingAction":
        """Optional trading hook to emit trading actions."""

        from research_lab.backend.core.backtests.models import TradingAction

        return TradingAction.HOLD


class BasePythonStrategy(PythonStrategyInterface):
    """Base class implementing metadata via class attributes."""

    strategy_key: str = ""
    strategy_name: str = ""
    strategy_version: str = "1.0.0"
    strategy_description: str | None = None
    strategy_tags: list[str] = []
    strategy_params_schema: dict[str, Any] = {}

    def __init__(self) -> None:
        self.params: dict[str, Any] = {}

    def get_metadata(self) -> Dict[str, Any]:
        return {
            "key": self.strategy_key or f"{self.__class__.__module__}.{self.__class__.__name__}",
            "name": self.strategy_name or self.__class__.__name__,
            "version": self.strategy_version,
            "description": self.strategy_description,
            "tags": self.strategy_tags,
            "params_schema": self.strategy_params_schema,
            "module_path": self.__class__.__module__,
            "class_name": self.__class__.__name__,
        }

    def initialize(self, params: Optional[Dict[str, Any]] = None) -> None:
        self.params = params or {}

    def on_bar(self, bar: "BacktestBar") -> float:
        return 0.0

    def on_bar_trade(self, bar: "BacktestBar", state: "BacktestPositionState") -> "TradingAction":
        from research_lab.backend.core.backtests.models import TradingAction

        return TradingAction.HOLD


__all__ = ["PythonStrategyInterface", "BasePythonStrategy"]
