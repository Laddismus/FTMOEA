from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from research_lab.backend.app.main import app
from research_lab.backend.app.api import python_strategies as api_module
from research_lab.backend.core.python_strategies.registry import PythonStrategyRegistry


STRATEGY_CODE = """
from research_lab.backend.core.python_strategies.interface import BasePythonStrategy

class MyTestStrategy(BasePythonStrategy):
    strategy_key = "api.test.strategy"
    strategy_name = "API Test Strategy"
    strategy_version = "0.1.0"
"""


def setup_strategy_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module_path = tmp_path / "my_strategy.py"
    module_path.write_text(STRATEGY_CODE, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    # swap registry to use temp dir
    api_module.registry = PythonStrategyRegistry(registry_dir=tmp_path)


def test_validate_and_register_strategy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    setup_strategy_module(tmp_path, monkeypatch)
    client = TestClient(app)

    request_body = {"module_path": "my_strategy", "class_name": "MyTestStrategy"}

    resp_validate = client.post("/api/python-strategies/validate-import", json=request_body)
    assert resp_validate.status_code == 200
    data = resp_validate.json()
    assert data["valid"] is True
    assert data["metadata"]["key"] != ""

    resp_register = client.post("/api/python-strategies/register", json=request_body)
    assert resp_register.status_code == 200
    meta = resp_register.json()
    assert meta["key"] == "api.test.strategy"

    resp_list = client.get("/api/python-strategies")
    assert resp_list.status_code == 200
    strategies = resp_list.json()["strategies"]
    assert any(entry["key"] == "api.test.strategy" for entry in strategies)
