from collections import Counter

from research_lab.backend.core.strategy_builder.node_catalog import NodeCatalog, ALLOWED_DTYPES
from research_lab.backend.core.strategy_builder.graph_engine import GraphEngine
from research_lab.backend.core.strategy_builder.models import StrategyGraph, StrategyNode, StrategyEdge


CORE_NODE_TYPES = {
    "price_source",
    "htf_price_source",
    "indicator_sma",
    "indicator_ema",
    "indicator_rsi",
    "indicator_atr",
    "condition_greater_than",
    "condition_less_than",
    "condition_cross_over",
    "condition_cross_under",
    "logic_and",
    "logic_or",
    "logic_not",
    "signal_long",
    "signal_short",
    "signal_flat",
    "risk_fixed_sl_tp",
    "risk_atr_sl_tp",
    "filter_session",
    "filter_volatility",
}

ALLOWED_CATEGORIES = {"source", "indicator", "condition", "logic", "signal", "risk", "filter", "meta"}
ALLOWED_STAGES = {"entry", "filter", "risk", "exit", "meta"}


def test_node_catalog_contains_core_v1_nodes() -> None:
    catalog = NodeCatalog()
    node_types = {node.type for node in catalog.list_nodes()}
    assert CORE_NODE_TYPES.issubset(node_types)


def test_node_specs_have_valid_structure() -> None:
    catalog = NodeCatalog()
    for spec in catalog.list_nodes():
        assert spec.category in ALLOWED_CATEGORIES
        assert spec.stage in ALLOWED_STAGES
        assert spec.description
        assert len(spec.inputs) == len(set(spec.inputs))
        assert len(spec.outputs) == len(set(spec.outputs))
        param_names = [param.name for param in spec.params]
        assert len(param_names) == len(set(param_names))
        for param in spec.params:
            assert param.dtype in ALLOWED_DTYPES


def test_example_trend_follow_strategy_graph_is_valid() -> None:
    catalog = NodeCatalog()
    engine = GraphEngine(catalog)

    nodes = [
        StrategyNode(id="src", type="price_source", params={"symbol": "EURUSD", "timeframe": "M5"}),
        StrategyNode(id="ema_fast", type="indicator_ema", params={"length": 10, "field": "close"}),
        StrategyNode(id="ema_slow", type="indicator_ema", params={"length": 30, "field": "close"}),
        StrategyNode(id="cross", type="condition_cross_over"),
        StrategyNode(id="sig_long", type="signal_long"),
    ]
    edges = [
        StrategyEdge(id="e1", from_node="src", from_port="close", to_node="ema_fast", to_port="source"),
        StrategyEdge(id="e2", from_node="src", from_port="close", to_node="ema_slow", to_port="source"),
        StrategyEdge(id="e3", from_node="ema_fast", from_port="ema", to_node="cross", to_port="fast"),
        StrategyEdge(id="e4", from_node="ema_slow", from_port="ema", to_node="cross", to_port="slow"),
        StrategyEdge(id="e5", from_node="cross", from_port="condition", to_node="sig_long", to_port="condition"),
    ]

    graph = StrategyGraph(id="trend1", name="Trend Follow", nodes=nodes, edges=edges)

    issues = engine.validate_graph(graph)
    assert issues == []
    ordered = engine.topological_sort(graph)
    assert ordered[0].id == "src"


def test_example_risk_and_filter_extension_is_valid() -> None:
    catalog = NodeCatalog()
    engine = GraphEngine(catalog)

    nodes = [
        StrategyNode(id="src", type="price_source", params={"symbol": "EURUSD", "timeframe": "M15"}),
        StrategyNode(id="atr", type="indicator_atr", params={"length": 14}),
        StrategyNode(id="signal", type="signal_long"),
        StrategyNode(id="risk", type="risk_atr_sl_tp", params={"sl_atr_mult": 1.5, "tp_atr_mult": 3.0}),
        StrategyNode(id="filter", type="filter_session", params={"session": "london"}),
    ]
    edges = [
        StrategyEdge(id="e1", from_node="src", from_port="close", to_node="atr", to_port="source"),
        StrategyEdge(id="e2", from_node="src", from_port="close", to_node="signal", to_port="condition"),
        StrategyEdge(id="e3", from_node="signal", from_port="signal_long", to_node="risk", to_port="signal"),
        StrategyEdge(id="e4", from_node="atr", from_port="atr", to_node="risk", to_port="atr"),
        StrategyEdge(id="e5", from_node="risk", from_port="risk_profile", to_node="filter", to_port="signal"),
    ]

    graph = StrategyGraph(id="risk_filter", name="Risk + Filter", nodes=nodes, edges=edges)
    issues = engine.validate_graph(graph)
    assert issues == []
    engine.topological_sort(graph)
