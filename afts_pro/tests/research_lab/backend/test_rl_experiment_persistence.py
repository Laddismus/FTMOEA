from datetime import datetime, timezone
from pathlib import Path

from research_lab.backend.core.rl.models import RLEnvRef, RLAlgo, RLTrainingConfig, RLRewardCheckConfig
from research_lab.backend.core.rl_experiments.models import (
    RlExperimentConfig,
    RlExperimentParamPoint,
    RlExperimentRunStatus,
    RlExperimentStatus,
)
from research_lab.backend.core.rl_experiments.persistence import RlExperimentPersistence


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
        reward_checks=[RLRewardCheckConfig(min_avg_reward=0.5)],
        tags=[],
        metadata={},
    )


def test_save_and_load(tmp_path: Path) -> None:
    persistence = RlExperimentPersistence(tmp_path)
    config = _config("exp1")
    runs = [
        RlExperimentRunStatus(run_id="r1", status=RlExperimentStatus.PENDING, job_id=None, error=None, rl_run_id=None),
        RlExperimentRunStatus(run_id="r2", status=RlExperimentStatus.RUNNING, job_id="job2", error=None, rl_run_id=None),
    ]

    persistence.save_experiment(config, runs)
    loaded = persistence.load_experiment("exp1")
    assert loaded is not None
    cfg, loaded_runs = loaded
    assert cfg.id == config.id
    assert len(loaded_runs) == 2


def test_list_experiments(tmp_path: Path) -> None:
    persistence = RlExperimentPersistence(tmp_path)
    persistence.save_experiment(_config("exp1"), [])
    persistence.save_experiment(_config("exp2"), [])

    configs = persistence.list_experiments()
    assert {c.id for c in configs} == {"exp1", "exp2"}
