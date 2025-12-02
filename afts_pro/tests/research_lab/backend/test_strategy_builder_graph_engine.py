import pytest

from research_lab.backend.core.strategy_builder.graph_engine import GraphEngine
from research_lab.backend.core.strategy_builder.models import StrategyEdge, StrategyGraph, StrategyNode
from research_lab.backend.core.strategy_builder.node_catalog import NodeCatalog


def build_valid_graph():
    nodes = [
        StrategyNode(id="source", type="price_source", params={"symbol": "EURUSD", "timeframe": "M5"}),
        StrategyNode(id="sma", type="indicator_sma", params={"length": 10, "field": "close"}),
        StrategyNode(id="condition", type="condition_greater_than"),
        StrategyNode(id="signal", type="signal_long"),
    ]
    edges = [
        StrategyEdge(id="e1", from_node="source", from_port="close", to_node="sma", to_port="source"),
        StrategyEdge(id="e2", from_node="sma", from_port="sma", to_node="condition", to_port="left"),
        StrategyEdge(id="e3", from_node="source", from_port="close", to_node="condition", to_port="right"),
        StrategyEdge(id="e4", from_node="condition", from_port="condition", to_node="signal", to_port="condition"),
    ]
    return StrategyGraph(id="g1", name="Valid Graph", nodes=nodes, edges=edges)


def test_validate_graph_success():
    catalog = NodeCatalog()
    engine = GraphEngine(catalog)
    graph = build_valid_graph()

    issues = engine.validate_graph(graph)

    assert issues == []
    ordered = engine.topological_sort(graph)
    assert ordered[0].id == "source"
    assert ordered[-1].id == "signal"


def test_validate_graph_errors():
    catalog = NodeCatalog()
    engine = GraphEngine(catalog)
    nodes = [StrategyNode(id="n1", type="price_source", params={"symbol": "X", "timeframe": "M1"}), StrategyNode(id="n2", type="price_source", params={"symbol": "Y", "timeframe": "M5"})]
    edges = [StrategyEdge(id="e1", from_node="n1", from_port="invalid", to_node="missing", to_port="y")]
    graph = StrategyGraph(id="g2", name="Invalid", nodes=nodes, edges=edges)

    issues = engine.validate_graph(graph)
    codes = {issue.code for issue in issues}
    assert "edge_to_missing" in codes
    assert "invalid_from_port" in codes


def test_cycle_detection_in_topological_sort():
    catalog = NodeCatalog()
    engine = GraphEngine(catalog)
    nodes = [
        StrategyNode(id="a", type="price_source"),
        StrategyNode(id="b", type="indicator_sma"),
    ]
    edges = [
        StrategyEdge(id="e1", from_node="a", from_port="price", to_node="b", to_port="price"),
        StrategyEdge(id="e2", from_node="b", from_port="sma", to_node="a", to_port="price"),
    ]
    graph = StrategyGraph(id="g3", name="Cycle", nodes=nodes, edges=edges)

    issues = engine.validate_graph(graph)
    assert any(issue.code == "cycle_detected" for issue in issues)
    with pytest.raises(ValueError):
        engine.topological_sort(graph)
