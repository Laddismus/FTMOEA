import pytest
from fastapi.testclient import TestClient

from research_lab.backend.app.main import app


def test_run_sync_endpoint() -> None:
    client = TestClient(app)
    payload = {"mode": "graph", "returns": [1, -1, 2], "window": 2}
    response = client.post("/api/backtests/run-sync", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["kpi_summary"]["total_return"] == 2


def test_submit_and_get_job() -> None:
    client = TestClient(app)
    payload = {"mode": "graph", "returns": [0.5, -0.1, 0.2], "window": 2}
    submit_resp = client.post("/api/backtests/submit", json=payload)
    assert submit_resp.status_code == 200
    job_id = submit_resp.json()["job_id"]

    status_resp = client.get(f"/api/backtests/jobs/{job_id}")
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert status_data["status"] == "completed"
    assert status_data["result"]["kpi_summary"]["total_return"] == pytest.approx(sum(payload["returns"]))
