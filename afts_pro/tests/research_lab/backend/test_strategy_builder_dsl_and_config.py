from research_lab.backend.core.strategy_builder.dsl_serializer import StrategyDslSerializer
from research_lab.backend.core.strategy_builder.config_translator import StrategyConfigTranslator
from research_lab.backend.core.strategy_builder.models import StrategyGraph, StrategyNode, StrategyEdge


def build_graph() -> StrategyGraph:
    nodes = [
        StrategyNode(id="source", type="price_source", params={"symbol": "EURUSD", "timeframe": "M5"}),
        StrategyNode(id="signal", type="signal_long"),
    ]
    edges = [
        StrategyEdge(id="e1", from_node="source", from_port="close", to_node="signal", to_port="condition"),
    ]
    return StrategyGraph(id="g1", name="Example", description="Test graph", nodes=nodes, edges=edges, metadata={"stage": "dev"})


def test_dsl_serializer_outputs_structure():
    graph = build_graph()
    serializer = StrategyDslSerializer()

    dsl = serializer.to_dict(graph)

    assert dsl["version"] == 1
    assert dsl["id"] == graph.id
    assert len(dsl["nodes"]) == 2
    assert dsl["edges"][0]["from"]["node"] == "source"


def test_config_translator_builds_engine_config():
    graph = build_graph()
    translator = StrategyConfigTranslator()

    config = translator.to_engine_config(graph)

    assert config["strategy"]["id"] == graph.id
    assert config["strategy"]["name"] == graph.name
    assert len(config["strategy"]["nodes"]) == 2
    assert config["execution"]["mode"] == "strategy_graph_v1"
