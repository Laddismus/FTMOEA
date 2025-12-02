from datetime import datetime, timezone
from pathlib import Path

from research_lab.backend.core.job_runner import InMemoryJobRunner
from research_lab.backend.core.rl.models import RLRunRequest, RLTrainingConfig, RLEnvRef, RLAlgo
from research_lab.backend.core.rl.runner import RLRunner
from research_lab.backend.core.rl.service import RLService
from research_lab.backend.core.rl.reward_verifier import RLRewardVerifier


def build_service(tmp_path: Path) -> RLService:
    return RLService(
        job_runner=InMemoryJobRunner(),
        rl_runner=RLRunner(policies_dir=tmp_path, verifier=RLRewardVerifier()),
    )


def make_request() -> RLRunRequest:
    return RLRunRequest(
        id="run-test",
        created_at=datetime.now(timezone.utc),
        config=RLTrainingConfig(env=RLEnvRef(env_id="AFTS-v0"), algo=RLAlgo.PPO, total_timesteps=2000),
        reward_check=None,
        notes=None,
        tags=[],
    )


def test_run_sync(tmp_path: Path) -> None:
    service = build_service(tmp_path)
    req = make_request()
    result = service.run_sync(req)
    assert result.id == req.id
    assert result.metrics.episode_rewards


def test_submit_and_get_job_result(tmp_path: Path) -> None:
    service = build_service(tmp_path)
    req = make_request()
    job_id = service.submit_job(req)
    status, result, error = service.get_job_result(job_id)
    assert status.value == "completed"
    assert result is not None
    assert error is None
