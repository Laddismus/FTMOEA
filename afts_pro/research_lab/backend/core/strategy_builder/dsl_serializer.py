"""Serializer for Strategy Graphs into DSL dictionaries and YAML."""

from __future__ import annotations

from typing import Any, Dict

import yaml

from research_lab.backend.core.strategy_builder.models import StrategyGraph


class StrategyDslSerializer:
    """Transform strategy graphs into DSL representations."""

    VERSION = 1

    def to_dict(self, graph: StrategyGraph) -> Dict[str, Any]:
        """Return a stable DSL representation as a dictionary."""

        return {
            "version": self.VERSION,
            "id": graph.id,
            "name": graph.name,
            "description": graph.description,
            "nodes": [
                {
                    "id": node.id,
                    "type": node.type,
                    "params": node.params,
                }
                for node in graph.nodes
            ],
            "edges": [
                {
                    "id": edge.id,
                    "from": {"node": edge.from_node, "port": edge.from_port},
                    "to": {"node": edge.to_node, "port": edge.to_port},
                }
                for edge in graph.edges
            ],
            "metadata": graph.metadata,
        }

    def to_yaml(self, graph: StrategyGraph) -> str:
        """Return a YAML string for the DSL representation."""

        return yaml.safe_dump(self.to_dict(graph), sort_keys=False)


__all__ = ["StrategyDslSerializer"]
