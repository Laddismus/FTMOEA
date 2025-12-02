from datetime import datetime, timezone
from pathlib import Path

from research_lab.backend.core.job_runner import InMemoryJobRunner
from research_lab.backend.core.rl.models import RLRunRequest, RLTrainingConfig, RLEnvRef, RLAlgo
from research_lab.backend.core.rl.reward_verifier import RLRewardVerifier
from research_lab.backend.core.rl.runner import RLRunner
from research_lab.backend.core.rl.service import RLService
from research_lab.backend.core.governance.models import ModelType, ModelStage
from research_lab.backend.core.governance.registry import GovernanceRegistry
from research_lab.backend.core.governance.service import GovernanceService
from research_lab.backend.core.backtests.persistence import BacktestPersistence
from research_lab.backend.core.experiments.persistence import ExperimentPersistence
from research_lab.backend.core.rl_experiments.persistence import RlExperimentPersistence


def test_register_from_rl(tmp_path: Path) -> None:
    job_runner = InMemoryJobRunner()
    rl_service = RLService(job_runner=job_runner, rl_runner=RLRunner(policies_dir=tmp_path / "policies", verifier=RLRewardVerifier()))
    # submit a job to ensure result is present
    run_request = RLRunRequest(
        id="rl-run-1",
        created_at=datetime.now(timezone.utc),
        config=RLTrainingConfig(env=RLEnvRef(env_id="AFTS-v0"), algo=RLAlgo.SAC, total_timesteps=1000),
        reward_check=None,
        notes=None,
        tags=[],
    )
    job_id = rl_service.submit_job(run_request)

    service = GovernanceService(
        registry=GovernanceRegistry(tmp_path / "gov"),
        backtest_persistence=BacktestPersistence(tmp_path / "bt"),
        experiment_persistence=ExperimentPersistence(tmp_path / "exp"),
        rl_service=rl_service,
        rl_experiment_persistence=RlExperimentPersistence(tmp_path / "rlexp"),
    )

    entry = service.register_from_rl_run(name="rl-model", rl_run_id=job_id)
    assert entry.type == ModelType.RL_POLICY
    assert entry.stage == ModelStage.CANDIDATE
    assert entry.rl.mean_return is not None
