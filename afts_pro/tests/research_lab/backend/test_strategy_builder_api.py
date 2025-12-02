from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from research_lab.backend.app.main import app
from research_lab.backend.core.strategy_builder.node_catalog import NodeCatalog


def build_valid_payload():
    return {
        "id": "g1",
        "name": "Valid Graph",
        "nodes": [
            {"id": "source", "type": "price_source", "params": {"symbol": "EURUSD", "timeframe": "M5"}},
            {"id": "sma", "type": "indicator_sma", "params": {"length": 10, "field": "close"}},
            {"id": "condition", "type": "condition_greater_than", "params": {"strict": True}},
            {"id": "signal", "type": "signal_long", "params": {}},
        ],
        "edges": [
            {"id": "e1", "from_node": "source", "from_port": "close", "to_node": "sma", "to_port": "source"},
            {"id": "e2", "from_node": "sma", "from_port": "sma", "to_node": "condition", "to_port": "left"},
            {"id": "e3", "from_node": "source", "from_port": "close", "to_node": "condition", "to_port": "right"},
            {"id": "e4", "from_node": "condition", "from_port": "condition", "to_node": "signal", "to_port": "condition"},
        ],
        "metadata": {},
    }


def test_list_nodes_endpoint_returns_catalog() -> None:
    client = TestClient(app)
    response = client.get("/api/strategy-builder/nodes")
    assert response.status_code == 200
    node_types = {node["type"] for node in response.json()}
    expected = {spec.type for spec in NodeCatalog().list_nodes()}
    assert expected.issubset(node_types)


def test_validate_endpoint_valid_graph() -> None:
    client = TestClient(app)
    payload = build_valid_payload()
    response = client.post("/api/strategy-builder/validate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is True
    assert data["issues"] == []


def test_validate_endpoint_invalid_graph() -> None:
    client = TestClient(app)
    payload = build_valid_payload()
    payload["nodes"][0]["type"] = "unknown"
    response = client.post("/api/strategy-builder/validate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is False
    assert len(data["issues"]) > 0


def test_compile_endpoint_handles_valid_and_invalid_graphs() -> None:
    client = TestClient(app)
    payload = build_valid_payload()
    resp_ok = client.post("/api/strategy-builder/compile", json=payload)
    assert resp_ok.status_code == 200
    assert "dsl" in resp_ok.json()
    assert "engine_config" in resp_ok.json()

    payload_bad = build_valid_payload()
    payload_bad["edges"][0]["from_node"] = "missing"
    resp_bad = client.post("/api/strategy-builder/compile", json=payload_bad)
    assert resp_bad.status_code == 400


def test_compile_and_save_writes_yaml(tmp_path, monkeypatch) -> None:
    from research_lab.backend import settings as settings_module
    from research_lab.backend.app.api import strategy_builder as strategy_builder_module

    # Override settings for temp strategies dir
    monkeypatch.setattr(strategy_builder_module, "get_settings_fn", lambda: settings_module.ResearchSettings(strategies_dir=tmp_path))

    client = TestClient(app)
    payload = build_valid_payload()
    response = client.post("/api/strategy-builder/compile-and-save", json=payload)
    assert response.status_code == 200
    data = response.json()
    saved_path = data["dsl_path"]
    assert tmp_path.as_posix() in saved_path.replace("\\", "/")

    loaded = yaml.safe_load(Path(saved_path).read_text(encoding="utf-8"))
    assert loaded["id"] == payload["id"]
