"""Simulated RL runner for generating deterministic training metrics."""

from __future__ import annotations

import math
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from research_lab.backend.core.rl.models import (
    RLPolicyRef,
    RLRewardMetricPoint,
    RLRewardCheckResult,
    RLRunRequest,
    RLRunResult,
    RLTrainingMetrics,
)
from research_lab.backend.core.rl.reward_verifier import RLRewardVerifier
from research_lab.backend.core.utils.datetime import ensure_utc_datetime


class RLRunner:
    """Stub RL runner that simulates training metrics."""

    def __init__(self, policies_dir: Path, verifier: RLRewardVerifier | None = None) -> None:
        self.policies_dir = policies_dir
        self.policies_dir.mkdir(parents=True, exist_ok=True)
        self.verifier = verifier or RLRewardVerifier()

    def run(self, request: RLRunRequest) -> RLRunResult:
        """Simulate an RL training run based on the request."""

        rng = random.Random(request.config.seed)
        episodes = min(max(request.config.total_timesteps // 1000, 5), 50)
        base = max(1.0, math.log(max(request.config.total_timesteps, 10)))
        episode_rewards: List[float] = []
        for _ in range(episodes):
            noise = rng.gauss(0, 0.2)
            episode_rewards.append(base + noise)

        avg_reward = sum(episode_rewards) / len(episode_rewards)
        max_reward = max(episode_rewards)
        min_reward = min(episode_rewards)

        step_size = max(request.config.total_timesteps // episodes, 1)
        reward_curve = [
            RLRewardMetricPoint(step=(i + 1) * step_size, value=episode_rewards[i]) for i in range(len(episode_rewards))
        ]

        metrics = RLTrainingMetrics(
            episode_rewards=episode_rewards,
            avg_reward=avg_reward,
            max_reward=max_reward,
            min_reward=min_reward,
            reward_curve=reward_curve,
        )

        reward_check_result: RLRewardCheckResult | None = None
        if request.reward_check is not None:
            reward_check_result = self.verifier.verify(metrics=metrics, config=request.reward_check)

        policy_key = f"{request.config.env.env_id}_{request.config.algo.value}_{request.id}"
        policy_ref = RLPolicyRef(
            key=policy_key,
            algo=request.config.algo,
            path=str(self.policies_dir / f"{policy_key}.bin"),
            created_at=datetime.now(timezone.utc),
            metadata={"simulated": True},
        )

        completed_at = datetime.now(timezone.utc)
        return RLRunResult(
            id=request.id,
            config=request.config,
            metrics=metrics,
            reward_check_result=reward_check_result,
            policy_ref=policy_ref,
            created_at=ensure_utc_datetime(request.created_at),
            completed_at=completed_at,
        )


__all__ = ["RLRunner"]
