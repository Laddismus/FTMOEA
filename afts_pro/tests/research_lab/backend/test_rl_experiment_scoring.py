from datetime import datetime, timezone
from pathlib import Path

import pytest

from research_lab.backend.core.rl.models import (
    RLTrainingConfig,
    RLEnvRef,
    RLAlgo,
    RLTrainingMetrics,
    RLRewardMetricPoint,
    RLRewardCheckResult,
    RLRunResult,
    RLRunStatus,
)
from research_lab.backend.core.rl_experiments.models import (
    RlExperimentConfig,
    RlExperimentParamPoint,
    RlExperimentRunStatus,
    RlExperimentStatus,
)
from research_lab.backend.core.rl_experiments.scoring import RlExperimentScorer


class FakeRLService:
    def __init__(self, results):
        self.results = results

    def get_job_result(self, job_id):
        result = self.results.get(job_id)
        return result["status"], result["result"], None


def _result(result_id: str, mean: float, std: float, reward_pass: bool) -> RLRunResult:
    metrics = RLTrainingMetrics(
        episode_rewards=[mean - std, mean, mean + std],
        avg_reward=mean,
        max_reward=mean + std,
        min_reward=mean - std,
        reward_curve=[RLRewardMetricPoint(step=1, value=mean)],
    )
    reward_check = RLRewardCheckResult(passed=reward_pass, avg_reward=mean, last_n_avg_reward=mean, reason=None if reward_pass else "fail")
    return RLRunResult(
        id=result_id,
        config=RLTrainingConfig(env=RLEnvRef(env_id="AFTS-v0"), algo=RLAlgo.SAC, total_timesteps=1000),
        metrics=metrics,
        reward_check_result=reward_check,
        policy_ref=None,
        created_at=datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
    )


def test_rl_leaderboard_scoring() -> None:
    results = {
        "job1": {"status": RLRunStatus.COMPLETED, "result": _result("run1", 1.0, 0.1, True)},
        "job2": {"status": RLRunStatus.COMPLETED, "result": _result("run2", 0.5, 0.05, False)},
        "job3": {"status": RLRunStatus.COMPLETED, "result": _result("run3", 1.2, 0.2, True)},
    }
    scorer = RlExperimentScorer(rl_service=FakeRLService(results))
    config = RlExperimentConfig(
        id="exp",
        name="exp",
        description=None,
        created_at=datetime.now(timezone.utc),
        env=RLEnvRef(env_id="AFTS-v0"),
        algo=RLAlgo.SAC,
        base_training=RLTrainingConfig(env=RLEnvRef(env_id="AFTS-v0"), algo=RLAlgo.SAC, total_timesteps=1000),
        param_grid=[
            RlExperimentParamPoint(values={"lr": 1e-3}),
            RlExperimentParamPoint(values={"lr": 1e-4}),
            RlExperimentParamPoint(values={"lr": 5e-4}),
        ],
        reward_checks=[],
        tags=[],
        metadata={},
    )
    runs = [
        RlExperimentRunStatus(run_id="r1", job_id="job1", status=RlExperimentStatus.COMPLETED, error=None, rl_run_id="run1"),
        RlExperimentRunStatus(run_id="r2", job_id="job2", status=RlExperimentStatus.COMPLETED, error=None, rl_run_id="run2"),
        RlExperimentRunStatus(run_id="r3", job_id="job3", status=RlExperimentStatus.COMPLETED, error=None, rl_run_id="run3"),
    ]

    leaderboard = scorer.build_leaderboard("exp", config, runs)

    assert leaderboard.total_runs == 3
    assert leaderboard.pass_rate == pytest.approx(2 / 3)
    assert leaderboard.runs[0].run_id == "r3"
    assert leaderboard.runs[0].rank == 1
