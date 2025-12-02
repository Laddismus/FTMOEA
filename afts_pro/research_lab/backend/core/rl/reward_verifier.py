"""Reward verification logic for RL runs."""

from __future__ import annotations

from research_lab.backend.core.rl.models import RLRewardCheckConfig, RLRewardCheckResult, RLTrainingMetrics


class RLRewardVerifier:
    """Verify reward metrics against configured thresholds."""

    def verify(self, metrics: RLTrainingMetrics, config: RLRewardCheckConfig) -> RLRewardCheckResult:
        """Evaluate metrics against threshold configuration."""

        avg_reward = metrics.avg_reward
        last_n_avg = None
        reason = None

        if metrics.episode_rewards and config.window > 0:
            last_rewards = metrics.episode_rewards[-config.window :]
            last_n_avg = sum(last_rewards) / len(last_rewards)

        passed = True
        if config.min_avg_reward is not None and avg_reward < config.min_avg_reward:
            passed = False
            reason = f"avg_reward {avg_reward:.4f} below min_avg_reward {config.min_avg_reward:.4f}"
        if config.min_last_n_avg_reward is not None:
            if last_n_avg is None or last_n_avg < config.min_last_n_avg_reward:
                passed = False
                reason = reason or f"last_n_avg_reward below threshold {config.min_last_n_avg_reward:.4f}"

        return RLRewardCheckResult(
            passed=passed,
            avg_reward=avg_reward,
            last_n_avg_reward=last_n_avg,
            reason=reason,
        )


__all__ = ["RLRewardVerifier"]
