from datetime import datetime, timezone
from pathlib import Path

import pytest

from research_lab.backend.core.rl.models import (
    RLRunRequest,
    RLTrainingConfig,
    RLEnvRef,
    RLAlgo,
    RLRewardCheckConfig,
)
from research_lab.backend.core.rl.reward_verifier import RLRewardVerifier
from research_lab.backend.core.rl.runner import RLRunner


def test_rl_runner_generates_metrics(tmp_path: Path) -> None:
    runner = RLRunner(policies_dir=tmp_path, verifier=RLRewardVerifier())
    request = RLRunRequest(
        id="run1",
        created_at=datetime.now(timezone.utc),
        config=RLTrainingConfig(env=RLEnvRef(env_id="AFTS-v0"), algo=RLAlgo.SAC, total_timesteps=5000, seed=42),
        reward_check=RLRewardCheckConfig(min_avg_reward=0.5),
        notes=None,
        tags=[],
    )

    result = runner.run(request)

    assert result.metrics.episode_rewards
    assert result.metrics.avg_reward == pytest.approx(sum(result.metrics.episode_rewards) / len(result.metrics.episode_rewards))
    assert result.reward_check_result is not None
    assert result.policy_ref is not None
    assert result.policy_ref.metadata.get("simulated") is True
