from datetime import datetime, timezone
from pathlib import Path

from research_lab.backend.core.job_runner import InMemoryJobRunner
from research_lab.backend.core.rl.models import RLTrainingConfig, RLEnvRef, RLAlgo
from research_lab.backend.core.rl.reward_verifier import RLRewardVerifier
from research_lab.backend.core.rl.runner import RLRunner
from research_lab.backend.core.rl.service import RLService
from research_lab.backend.core.rl_experiments.models import (
    RlExperimentConfig,
    RlExperimentParamPoint,
    RlExperimentStatus,
)
from research_lab.backend.core.rl_experiments.persistence import RlExperimentPersistence
from research_lab.backend.core.rl_experiments.scoring import RlExperimentScorer
from research_lab.backend.core.rl_experiments.service import RlExperimentService


def build_service(tmp_path: Path) -> RlExperimentService:
    job_runner = InMemoryJobRunner()
    rl_service = RLService(job_runner=job_runner, rl_runner=RLRunner(policies_dir=tmp_path / "policies", verifier=RLRewardVerifier()))
    persistence = RlExperimentPersistence(tmp_path / "experiments")
    scorer = RlExperimentScorer(rl_service=rl_service)
    return RlExperimentService(rl_service=rl_service, persistence=persistence, scorer=scorer)


def _config(exp_id: str) -> RlExperimentConfig:
    return RlExperimentConfig(
        id=exp_id,
        name="rl-exp",
        description=None,
        created_at=datetime.now(timezone.utc),
        env=RLEnvRef(env_id="AFTS-v0"),
        algo=RLAlgo.SAC,
        base_training=RLTrainingConfig(env=RLEnvRef(env_id="AFTS-v0"), algo=RLAlgo.SAC, total_timesteps=2000),
        param_grid=[RlExperimentParamPoint(values={"learning_rate": 3e-4}), RlExperimentParamPoint(values={"learning_rate": 1e-4})],
        reward_checks=[],
        tags=[],
        metadata={},
    )


def test_create_and_launch(tmp_path: Path) -> None:
    service = build_service(tmp_path)
    cfg = _config("exp1")
    service.create_experiment(cfg)

    loaded = service.get_experiment("exp1")
    assert loaded is not None
    config, runs = loaded
    assert len(runs) == len(cfg.param_grid)
    assert all(r.status == RlExperimentStatus.PENDING for r in runs)

    config2, running = service.launch_experiment("exp1")
    assert config2.id == cfg.id
    assert all(r.status == RlExperimentStatus.RUNNING for r in running)

    _, refreshed = service.refresh_status("exp1")
    assert any(r.status == RlExperimentStatus.COMPLETED for r in refreshed)


def test_get_leaderboard(tmp_path: Path) -> None:
    service = build_service(tmp_path)
    cfg = _config("exp2")
    service.create_experiment(cfg)
    service.launch_experiment("exp2")
    service.refresh_status("exp2")

    leaderboard = service.get_leaderboard("exp2")
    assert leaderboard is not None
    assert leaderboard.experiment_id == "exp2"
