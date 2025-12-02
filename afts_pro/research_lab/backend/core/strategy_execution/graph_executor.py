"""Adapter for graph-based strategies."""

from __future__ import annotations

from typing import Any, Dict, Optional

from research_lab.backend.core.strategy_execution.interface import StrategyExecutor, StrategyExecutionContext


class GraphStrategyExecutorAdapter(StrategyExecutor):
    """StrategyExecutor adapter for graph-based strategies."""

    def __init__(self, engine_config: Dict[str, Any], dsl: Optional[Dict[str, Any]] = None) -> None:
        self.engine_config = engine_config
        self.dsl = dsl or {}
        self._metadata: Dict[str, Any] = {}

    def initialize(self, ctx: StrategyExecutionContext) -> None:
        graph_info = self.engine_config.get("strategy", {})
        self._metadata = {
            "type": "graph",
            "engine_mode": self.engine_config.get("execution", {}).get("mode", "strategy_graph_v1"),
            "graph_id": graph_info.get("id") or self.dsl.get("id"),
            "graph_name": graph_info.get("name") or self.dsl.get("name"),
            "params": ctx.params,
            "config": ctx.config,
        }

    def get_metadata(self) -> Dict[str, Any]:
        return self._metadata


__all__ = ["GraphStrategyExecutorAdapter"]
