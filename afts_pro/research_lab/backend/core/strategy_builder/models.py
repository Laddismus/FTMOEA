"""Strategy Builder domain models."""

from __future__ import annotations

from typing import Any, List, Optional, Literal

from pydantic import BaseModel, Field

NodeCategory = Literal["source", "indicator", "condition", "logic", "signal", "risk", "filter", "meta"]
NodeStage = Literal["entry", "filter", "risk", "exit", "meta"]


class NodeParamDefinition(BaseModel):
    """Parameter definition for a node type."""

    name: str
    dtype: str
    required: bool = True
    default: Optional[Any] = None


class NodeSpec(BaseModel):
    """Node specification describing available node types."""

    type: str
    category: NodeCategory
    stage: NodeStage
    description: str
    inputs: List[str] = Field(default_factory=list)
    outputs: List[str] = Field(default_factory=list)
    params: List[NodeParamDefinition] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    version: str = "1.0.0"


class StrategyNode(BaseModel):
    """A node instance within a strategy graph."""

    id: str
    type: str
    params: dict[str, Any] = Field(default_factory=dict)


class StrategyEdge(BaseModel):
    """Directional edge connecting two strategy nodes."""

    id: str
    from_node: str
    from_port: str
    to_node: str
    to_port: str


class StrategyGraph(BaseModel):
    """Full strategy graph composed of nodes and edges."""

    id: str
    name: str
    description: Optional[str] = None
    nodes: List[StrategyNode] = Field(min_length=1)
    edges: List[StrategyEdge]
    metadata: dict[str, Any] = Field(default_factory=dict)


class ValidationIssue(BaseModel):
    """Represents a validation issue detected in a graph."""

    code: str
    message: str
    node_id: Optional[str] = None


__all__ = [
    "NodeParamDefinition",
    "NodeSpec",
    "StrategyNode",
    "StrategyEdge",
    "StrategyGraph",
    "ValidationIssue",
    "NodeCategory",
    "NodeStage",
]
