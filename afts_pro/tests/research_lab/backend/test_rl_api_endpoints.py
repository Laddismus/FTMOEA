from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from research_lab.backend.app.main import app
from research_lab.backend.app.api import rl as rl_router
from research_lab.backend.core.job_runner import InMemoryJobRunner
from research_lab.backend.core.rl.reward_verifier import RLRewardVerifier
from research_lab.backend.core.rl.runner import RLRunner
from research_lab.backend.core.rl.service import RLService
from research_lab.backend.core.rl.policy_loader import RLPolicyLoader


def setup_router(tmp_path: Path) -> None:
    rl_router._job_runner = InMemoryJobRunner()
    rl_router._reward_verifier = RLRewardVerifier()
    rl_router._rl_runner = RLRunner(policies_dir=tmp_path / "policies", verifier=rl_router._reward_verifier)
    rl_router._policy_loader = RLPolicyLoader(policies_dir=tmp_path / "policies")
    rl_router._rl_service = RLService(job_runner=rl_router._job_runner, rl_runner=rl_router._rl_runner)


def test_rl_api_flow(tmp_path: Path, monkeypatch) -> None:
    setup_router(tmp_path)
    client = TestClient(app)
    run_body = {
        "config": {"env": {"env_id": "AFTS-v0"}, "algo": "sac", "total_timesteps": 3000},
        "reward_check": {"min_avg_reward": 0.5},
    }

    sync_resp = client.post("/api/rl/runs/sync", json=run_body)
    assert sync_resp.status_code == 200
    sync_data = sync_resp.json()
    assert "metrics" in sync_data

    submit_resp = client.post("/api/rl/runs", json=run_body)
    assert submit_resp.status_code == 200
    job_info = submit_resp.json()
    job_id = job_info["job_id"]

    status_resp = client.get(f"/api/rl/jobs/{job_id}")
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert status_data["status"] == "completed"
    assert status_data["result"] is not None

    policies_resp = client.get("/api/rl/policies")
    assert policies_resp.status_code == 200

    reward_check_resp = client.post(
        "/api/rl/reward-check",
        json={
            "metrics": sync_data["metrics"],
            "config": {"min_avg_reward": 0.0},
        },
    )
    assert reward_check_resp.status_code == 200
