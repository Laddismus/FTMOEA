from research_lab.backend.core.rl.models import RLTrainingMetrics, RLRewardMetricPoint, RLRewardCheckConfig
from research_lab.backend.core.rl.reward_verifier import RLRewardVerifier


def _metrics() -> RLTrainingMetrics:
    rewards = [1.0, 1.5, 2.0, 2.5]
    return RLTrainingMetrics(
        episode_rewards=rewards,
        avg_reward=sum(rewards) / len(rewards),
        max_reward=max(rewards),
        min_reward=min(rewards),
        reward_curve=[RLRewardMetricPoint(step=i + 1, value=r) for i, r in enumerate(rewards)],
    )


def test_reward_verifier_fails_on_avg() -> None:
    verifier = RLRewardVerifier()
    config = RLRewardCheckConfig(min_avg_reward=3.0)
    result = verifier.verify(metrics=_metrics(), config=config)
    assert result.passed is False
    assert result.reason is not None


def test_reward_verifier_fails_on_last_n() -> None:
    verifier = RLRewardVerifier()
    config = RLRewardCheckConfig(min_last_n_avg_reward=2.4, window=2)
    result = verifier.verify(metrics=_metrics(), config=config)
    assert result.passed is False


def test_reward_verifier_passes_thresholds() -> None:
    verifier = RLRewardVerifier()
    config = RLRewardCheckConfig(min_avg_reward=1.0, min_last_n_avg_reward=1.5, window=2)
    result = verifier.verify(metrics=_metrics(), config=config)
    assert result.passed is True
