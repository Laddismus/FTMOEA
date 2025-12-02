import pytest
from pydantic import ValidationError

from research_lab.backend.core.strategy_builder.models import StrategyGraph, StrategyNode, StrategyEdge


def test_strategy_models_instantiation() -> None:
    node = StrategyNode(id="n1", type="price_source", params={"symbol": "EURUSD"})
    edge = StrategyEdge(id="e1", from_node="n1", from_port="price", to_node="n2", to_port="input")

    graph = StrategyGraph(id="g1", name="Test", nodes=[node], edges=[edge])

    assert graph.id == "g1"
    assert graph.nodes[0].params["symbol"] == "EURUSD"


def test_strategy_graph_requires_nodes() -> None:
    with pytest.raises(ValidationError):
        StrategyGraph(id="g1", name="Invalid", nodes=[], edges=[])
