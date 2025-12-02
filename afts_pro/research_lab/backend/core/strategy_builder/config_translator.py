"""Translate strategy graphs into engine configuration dictionaries."""

from __future__ import annotations

from typing import Any, Dict

from research_lab.backend.core.strategy_builder.models import StrategyGraph


class StrategyConfigTranslator:
    """Produce generic engine-ready configs from a strategy graph."""

    def to_engine_config(self, graph: StrategyGraph) -> Dict[str, Any]:
        """Return an engine configuration dict derived from the strategy graph."""

        return {
            "strategy": {
                "id": graph.id,
                "name": graph.name,
                "description": graph.description,
                "nodes": [node.model_dump() for node in graph.nodes],
                "edges": [edge.model_dump() for edge in graph.edges],
                "metadata": graph.metadata,
            },
            "execution": {
                "mode": "strategy_graph_v1",
            },
        }


__all__ = ["StrategyConfigTranslator"]
