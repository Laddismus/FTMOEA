from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from research_lab.backend.app.main import app
from research_lab.backend.app.api import rl_experiments as rl_exp_router
from research_lab.backend.core.job_runner import InMemoryJobRunner
from research_lab.backend.core.rl.reward_verifier import RLRewardVerifier
from research_lab.backend.core.rl.runner import RLRunner
from research_lab.backend.core.rl.service import RLService
from research_lab.backend.core.rl_experiments.persistence import RlExperimentPersistence
from research_lab.backend.core.rl_experiments.scoring import RlExperimentScorer
from research_lab.backend.core.rl_experiments.service import RlExperimentService


def setup_router(tmp_path: Path) -> None:
    rl_exp_router._job_runner = InMemoryJobRunner()
    rl_exp_router._reward_verifier = RLRewardVerifier()
    rl_exp_router._rl_runner = RLRunner(policies_dir=tmp_path / "policies", verifier=rl_exp_router._reward_verifier)
    rl_exp_router._rl_service = RLService(job_runner=rl_exp_router._job_runner, rl_runner=rl_exp_router._rl_runner)
    rl_exp_router._rl_experiment_persistence = RlExperimentPersistence(tmp_path / "experiments")
    rl_exp_router._rl_experiment_scorer = RlExperimentScorer(rl_service=rl_exp_router._rl_service)
    rl_exp_router._rl_experiment_service = RlExperimentService(
        rl_service=rl_exp_router._rl_service,
        persistence=rl_exp_router._rl_experiment_persistence,
        scorer=rl_exp_router._rl_experiment_scorer,
    )


def test_rl_experiments_api_flow(tmp_path: Path, monkeypatch) -> None:
    setup_router(tmp_path)
    client = TestClient(app)
    payload = {
        "name": "rl-api-exp",
        "env": {"env_id": "AFTS-v0"},
        "algo": "sac",
        "base_training": {"env": {"env_id": "AFTS-v0"}, "algo": "sac", "total_timesteps": 2000},
        "param_grid": [{"values": {"learning_rate": 0.0003}}, {"values": {"learning_rate": 0.0001}}],
    }

    resp = client.post("/api/rl-experiments", json=payload)
    assert resp.status_code == 200
    exp = resp.json()
    exp_id = exp["id"]

    launch_resp = client.post(f"/api/rl-experiments/{exp_id}/launch")
    assert launch_resp.status_code == 200

    detail_resp = client.get(f"/api/rl-experiments/{exp_id}")
    assert detail_resp.status_code == 200
    detail = detail_resp.json()
    assert len(detail["runs"]) == 2

    leaderboard_resp = client.get(f"/api/rl-experiments/{exp_id}/leaderboard")
    assert leaderboard_resp.status_code == 200
    lb = leaderboard_resp.json()
    assert lb["experiment_id"] == exp_id
