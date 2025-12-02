"""Interfaces for strategy execution within backtests."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Protocol


class StrategyExecutionContext(Protocol):
    """Lightweight execution context placeholder."""

    config: Dict[str, Any]
    params: Dict[str, Any]


class StrategyExecutor(ABC):
    """Abstract base for strategy executors used by the Backtest Engine."""

    @abstractmethod
    def initialize(self, ctx: StrategyExecutionContext) -> None:
        """Initialize the underlying strategy implementation with the given context."""

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """Return metadata about the executed strategy."""


__all__ = ["StrategyExecutor", "StrategyExecutionContext"]
