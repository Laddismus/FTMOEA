"""Graph validation and ordering utilities for strategy graphs."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List, Set

from research_lab.backend.core.strategy_builder.models import (
    StrategyEdge,
    StrategyGraph,
    StrategyNode,
    ValidationIssue,
)
from research_lab.backend.core.strategy_builder.node_catalog import NodeCatalog


class GraphEngine:
    """Validates and orders strategy graphs."""

    def __init__(self, catalog: NodeCatalog) -> None:
        self.catalog = catalog

    def validate_graph(self, graph: StrategyGraph) -> List[ValidationIssue]:
        """Validate structural constraints of the graph."""

        issues: List[ValidationIssue] = []
        node_ids: Set[str] = set()
        for node in graph.nodes:
            if node.id in node_ids:
                issues.append(ValidationIssue(code="duplicate_node", message=f"Duplicate node id '{node.id}'", node_id=node.id))
            node_ids.add(node.id)

            spec = self.catalog.get_node(node.type)
            if spec is None:
                issues.append(ValidationIssue(code="unknown_node_type", message=f"Unknown node type '{node.type}'", node_id=node.id))

        for edge in graph.edges:
            if edge.from_node not in node_ids:
                issues.append(ValidationIssue(code="edge_from_missing", message=f"Edge references missing from_node '{edge.from_node}'"))
            if edge.to_node not in node_ids:
                issues.append(ValidationIssue(code="edge_to_missing", message=f"Edge references missing to_node '{edge.to_node}'"))

            from_spec = self.catalog.get_node(self._node_type(graph, edge.from_node))
            to_spec = self.catalog.get_node(self._node_type(graph, edge.to_node))
            if from_spec and edge.from_port not in from_spec.outputs:
                issues.append(
                    ValidationIssue(
                        code="invalid_from_port",
                        message=f"Port '{edge.from_port}' not in outputs of node '{edge.from_node}'",
                        node_id=edge.from_node,
                    )
                )
            if to_spec and edge.to_port not in to_spec.inputs:
                issues.append(
                    ValidationIssue(
                        code="invalid_to_port",
                        message=f"Port '{edge.to_port}' not in inputs of node '{edge.to_node}'",
                        node_id=edge.to_node,
                    )
                )

        if self._has_cycle(graph):
            issues.append(ValidationIssue(code="cycle_detected", message="Graph contains a cycle."))

        return issues

    def topological_sort(self, graph: StrategyGraph) -> List[StrategyNode]:
        """Return nodes in topological order; raise ValueError if graph has a cycle."""

        incoming_counts: Dict[str, int] = {node.id: 0 for node in graph.nodes}
        adjacency: Dict[str, List[str]] = defaultdict(list)
        for edge in graph.edges:
            adjacency[edge.from_node].append(edge.to_node)
            incoming_counts[edge.to_node] = incoming_counts.get(edge.to_node, 0) + 1

        queue = deque([node_id for node_id, count in incoming_counts.items() if count == 0])
        ordered: List[str] = []

        while queue:
            current = queue.popleft()
            ordered.append(current)
            for neighbor in adjacency[current]:
                incoming_counts[neighbor] -= 1
                if incoming_counts[neighbor] == 0:
                    queue.append(neighbor)

        if len(ordered) != len(graph.nodes):
            raise ValueError("Graph contains a cycle; topological sort failed.")

        node_map = {node.id: node for node in graph.nodes}
        return [node_map[node_id] for node_id in ordered]

    def _has_cycle(self, graph: StrategyGraph) -> bool:
        """Detect cycle using DFS."""

        adjacency: Dict[str, List[str]] = defaultdict(list)
        for edge in graph.edges:
            adjacency[edge.from_node].append(edge.to_node)

        visited: Set[str] = set()
        stack: Set[str] = set()

        def dfs(node_id: str) -> bool:
            if node_id in stack:
                return True
            if node_id in visited:
                return False
            visited.add(node_id)
            stack.add(node_id)
            for neighbor in adjacency.get(node_id, []):
                if dfs(neighbor):
                    return True
            stack.remove(node_id)
            return False

        for node in graph.nodes:
            if dfs(node.id):
                return True
        return False

    def _node_type(self, graph: StrategyGraph, node_id: str) -> str | None:
        for node in graph.nodes:
            if node.id == node_id:
                return node.type
        return None


__all__ = ["GraphEngine"]
