"""Leaderboard scoring for RL experiments."""

from __future__ import annotations

import statistics
from typing import List

from research_lab.backend.core.rl.service import RLService
from research_lab.backend.core.rl.models import RLRunStatus
from research_lab.backend.core.rl_experiments.models import (
    RlExperimentConfig,
    RlExperimentLeaderboard,
    RlExperimentRunScore,
    RlExperimentRunStatus,
    RlExperimentStatus,
)


class RlExperimentScorer:
    """Build leaderboards for RL experiments using stored run results."""

    def __init__(self, rl_service: RLService) -> None:
        self.rl_service = rl_service

    def build_leaderboard(
        self,
        experiment_id: str,
        config: RlExperimentConfig,
        runs: List[RlExperimentRunStatus],
    ) -> RlExperimentLeaderboard:
        scored_runs: list[RlExperimentRunScore] = []
        passes = 0

        param_base = config.base_training.hyperparams or {}

        for idx, run in enumerate(runs):
            if run.status != RlExperimentStatus.COMPLETED or run.job_id is None:
                continue
            status, result, _ = self.rl_service.get_job_result(run.job_id)
            if result is None or status != RLRunStatus.COMPLETED:
                continue
            metrics = result.metrics
            params = dict(param_base)
            if idx < len(config.param_grid):
                params.update(config.param_grid[idx].values)
            std_return = None
            if metrics.episode_rewards:
                try:
                    std_return = statistics.pstdev(metrics.episode_rewards)
                except statistics.StatisticsError:
                    std_return = 0.0
            mean_return = metrics.avg_reward
            max_return = metrics.max_reward
            steps = metrics.reward_curve[-1].step if metrics.reward_curve else None
            reward_passed = result.reward_check_result.passed if result.reward_check_result else None
            failed_checks = []
            if result.reward_check_result and not result.reward_check_result.passed and result.reward_check_result.reason:
                failed_checks.append(result.reward_check_result.reason)

            composite_score = mean_return
            if std_return is not None:
                composite_score = mean_return - 0.1 * std_return
            if reward_passed is False:
                composite_score = composite_score - 1.0

            if reward_passed:
                passes += 1

            scored_runs.append(
                RlExperimentRunScore(
                    run_id=run.run_id,
                    rl_run_id=result.id,
                    params=params,
                    mean_return=mean_return,
                    std_return=std_return,
                    max_return=max_return,
                    steps=steps,
                    reward_checks_passed=reward_passed,
                    failed_checks=failed_checks,
                    composite_score=composite_score,
                )
            )

        scored_runs.sort(
            key=lambda s: (
                0 if s.reward_checks_passed else 1,
                -(s.composite_score if s.composite_score is not None else float("-inf")),
            )
        )
        for idx, run in enumerate(scored_runs, start=1):
            run.rank = idx

        total_runs = len(scored_runs)
        pass_rate = passes / total_runs if total_runs else None

        return RlExperimentLeaderboard(
            experiment_id=experiment_id,
            name=config.name,
            created_at=config.created_at,
            total_runs=total_runs,
            passes=passes,
            pass_rate=pass_rate,
            runs=scored_runs,
        )


__all__ = ["RlExperimentScorer"]
