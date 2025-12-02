"""Strategy Builder endpoints."""

from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import yaml

from research_lab.backend import settings as settings_module
from research_lab.backend.core.strategy_builder import (
    GraphEngine,
    NodeCatalog,
    StrategyConfigTranslator,
    StrategyDslSerializer,
    StrategyEdge,
    StrategyGraph,
    StrategyNode,
    ValidationIssue,
    NodeSpec,
)

router = APIRouter(prefix="/strategy-builder", tags=["strategy-builder"])

catalog = NodeCatalog()
engine = GraphEngine(catalog)
serializer = StrategyDslSerializer()
translator = StrategyConfigTranslator()
get_settings_fn = settings_module.get_settings


class StrategyNodeIn(BaseModel):
    id: str
    type: str
    params: dict = {}


class StrategyEdgeIn(BaseModel):
    id: str
    from_node: str
    from_port: str
    to_node: str
    to_port: str


class StrategyGraphIn(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    nodes: List[StrategyNodeIn]
    edges: List[StrategyEdgeIn]
    metadata: dict = {}

    def to_domain(self) -> StrategyGraph:
        return StrategyGraph(
            id=self.id,
            name=self.name,
            description=self.description,
            metadata=self.metadata,
            nodes=[StrategyNode.model_validate(node.model_dump()) for node in self.nodes],
            edges=[StrategyEdge.model_validate(edge.model_dump()) for edge in self.edges],
        )


class ValidationResponse(BaseModel):
    valid: bool
    issues: List[ValidationIssue]


class CompileResponse(BaseModel):
    dsl: dict
    engine_config: dict


class SaveRequest(StrategyGraphIn):
    target_id: Optional[str] = None


class SaveResponse(BaseModel):
    dsl_path: str
    engine_config: dict


@router.get("/nodes", response_model=List[NodeSpec])
def list_nodes() -> List[NodeSpec]:
    """Return available node specifications."""

    return catalog.list_nodes()


@router.post("/validate", response_model=ValidationResponse)
def validate_graph(payload: StrategyGraphIn) -> ValidationResponse:
    """Validate a strategy graph against the catalog."""

    domain_graph = payload.to_domain()
    issues = engine.validate_graph(domain_graph)
    return ValidationResponse(valid=len(issues) == 0, issues=issues)


@router.post("/compile", response_model=CompileResponse)
def compile_graph(payload: StrategyGraphIn) -> CompileResponse:
    """Validate and compile a strategy graph into DSL and engine config."""

    domain_graph = payload.to_domain()
    issues = engine.validate_graph(domain_graph)
    if issues:
        raise HTTPException(status_code=400, detail=[issue.model_dump() for issue in issues])

    dsl = serializer.to_dict(domain_graph)
    engine_config = translator.to_engine_config(domain_graph)
    return CompileResponse(dsl=dsl, engine_config=engine_config)


@router.post("/compile-and-save", response_model=SaveResponse)
def compile_and_save(payload: SaveRequest) -> SaveResponse:
    """Validate, compile, and persist a strategy DSL YAML."""

    domain_graph = payload.to_domain()
    issues = engine.validate_graph(domain_graph)
    if issues:
        raise HTTPException(status_code=400, detail=[issue.model_dump() for issue in issues])

    dsl = serializer.to_dict(domain_graph)
    yaml_str = serializer.to_yaml(domain_graph)
    settings = get_settings_fn()
    strategies_dir = settings.strategies_dir
    strategies_dir.mkdir(parents=True, exist_ok=True)
    target_name = payload.target_id or domain_graph.id
    target_path: Path = strategies_dir / f"{target_name}.yaml"
    target_path.write_text(yaml_str, encoding="utf-8")

    engine_config = translator.to_engine_config(domain_graph)
    return SaveResponse(dsl_path=str(target_path), engine_config=engine_config)


__all__ = ["router"]
