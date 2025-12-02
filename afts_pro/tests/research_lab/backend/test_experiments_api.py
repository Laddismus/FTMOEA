from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from research_lab.backend.app.main import app
from research_lab.backend.app.api import experiments as experiments_router
from research_lab.backend.core.backtests.engine import RollingKpiBacktestEngine
from research_lab.backend.core.backtests.persistence import BacktestPersistence
from research_lab.backend.core.backtests.service import BacktestService
from research_lab.backend.core.job_runner import InMemoryJobRunner
from research_lab.backend.core.experiments.persistence import ExperimentPersistence
from research_lab.backend.core.experiments.service import ExperimentService
from research_lab.backend.core.experiments.scoring import ExperimentScorer


def setup_router(tmp_path: Path) -> None:
    job_runner = InMemoryJobRunner()
    backtest_service = BacktestService(
        job_runner=job_runner,
        engine=RollingKpiBacktestEngine(),
        persistence=BacktestPersistence(tmp_path / "backtests"),
    )
    experiment_service = ExperimentService(
        backtest_service=backtest_service,
        experiment_persistence=ExperimentPersistence(tmp_path / "experiments"),
        scorer=ExperimentScorer(backtest_persistence=backtest_service.persistence),
    )
    experiments_router._experiment_service = experiment_service


def test_experiments_api_flow(tmp_path: Path, monkeypatch) -> None:
    setup_router(tmp_path)
    client = TestClient(app)
    base_backtest = {
        "mode": "graph",
        "returns": [0.1, -0.05],
        "window": 2,
    }
    payload = {
        "name": "api-exp",
        "strategy": {"mode": "graph"},
        "base_backtest": base_backtest,
        "param_grid": [{"values": {"size": 1.0}}, {"values": {"size": 2.0}}],
        "tags": [],
        "metadata": {},
    }

    resp = client.post("/api/experiments", json=payload)
    assert resp.status_code == 200
    exp = resp.json()
    exp_id = exp["id"]

    launch_resp = client.post(f"/api/experiments/{exp_id}/launch")
    assert launch_resp.status_code == 200

    list_resp = client.get("/api/experiments")
    assert list_resp.status_code == 200
    assert any(item["id"] == exp_id for item in list_resp.json())

    detail_resp = client.get(f"/api/experiments/{exp_id}")
    assert detail_resp.status_code == 200
    detail = detail_resp.json()
    assert detail["config"]["id"] == exp_id
    assert len(detail["runs"]) == 2
    assert any(run["status"] == "completed" for run in detail["runs"])

    leaderboard_resp = client.get(f"/api/experiments/{exp_id}/leaderboard")
    assert leaderboard_resp.status_code == 200
    lb = leaderboard_resp.json()
    assert lb["experiment_id"] == exp_id
    assert "runs" in lb
