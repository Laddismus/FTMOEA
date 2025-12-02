import pytest
from fastapi.testclient import TestClient

from research_lab.backend.app.main import app
from research_lab.backend.app.api import backtests as backtests_module
from research_lab.backend.core.backtests import BacktestService, RollingKpiBacktestEngine
from research_lab.backend.core.backtests.persistence import BacktestPersistence
from research_lab.backend.core.job_runner import InMemoryJobRunner


def override_service(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    service = BacktestService(
        job_runner=InMemoryJobRunner(),
        engine=RollingKpiBacktestEngine(),
        persistence=BacktestPersistence(tmp_path),
    )
    backtests_module._service = service


def test_run_sync_and_fetch_persisted_result(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    override_service(tmp_path, monkeypatch)
    client = TestClient(app)

    payload = {"mode": "graph", "returns": [1.0, -0.5, 0.5], "window": 2}
    resp = client.post("/api/backtests/run-sync", json=payload)
    assert resp.status_code == 200
    result = resp.json()
    run_id = result["id"]

    fetch = client.get(f"/api/backtests/runs/{run_id}")
    assert fetch.status_code == 200
    fetched = fetch.json()
    assert fetched["id"] == run_id


def test_list_runs_endpoint(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    override_service(tmp_path, monkeypatch)
    client = TestClient(app)

    payload = {"mode": "graph", "returns": [0.1, 0.2], "window": 1}
    client.post("/api/backtests/run-sync", json=payload)
    client.post("/api/backtests/run-sync", json=payload)

    list_resp = client.get("/api/backtests/runs")
    assert list_resp.status_code == 200
    runs = list_resp.json()["runs"]
    assert len(runs) >= 2


def test_get_run_404(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    override_service(tmp_path, monkeypatch)
    client = TestClient(app)

    resp = client.get("/api/backtests/runs/nonexistent")
    assert resp.status_code == 404
