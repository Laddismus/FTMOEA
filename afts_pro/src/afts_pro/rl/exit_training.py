from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from afts_pro.rl.exit_agent import ExitAgent
from afts_pro.rl.replay_buffer import ReplayBuffer
from afts_pro.rl.env import RLTradingEnv

logger = logging.getLogger(__name__)


@dataclass
class ExitTrainConfig:
    total_episodes: int = 10
    max_steps_per_episode: int = 500
    replay_capacity: int = 5000
    batch_size: int = 32
    log_interval: int = 1
    save_every_n_episodes: int = 5


@dataclass
class ExitTrainSummary:
    episodes: int
    returns: List[float]
    mean_return: float
    best_return: float
    checkpoint_dir: str | None = None


def train_exit_agent(env: RLTradingEnv, agent: ExitAgent, train_config: ExitTrainConfig, checkpoint_dir: Path | None = None) -> ExitTrainSummary:
    replay = ReplayBuffer(capacity=train_config.replay_capacity, obs_dim=agent.obs_dim)
    episode_returns: List[float] = []
    checkpoint_dir = checkpoint_dir or Path("models/exit_agent")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(train_config.total_episodes):
        obs, _ = env.reset(seed=ep)
        ep_return = 0.0
        for step in range(train_config.max_steps_per_episode):
            action = agent.act(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            replay.add(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done)
            obs = next_obs
            ep_return += reward
            if replay.size >= train_config.batch_size:
                batch = replay.sample(train_config.batch_size)
                agent.train_on_batch(batch)
            if done:
                break
        episode_returns.append(ep_return)
        if (ep + 1) % train_config.log_interval == 0:
            logger.info("ExitAgent Episode %d/%d | return=%.4f", ep + 1, train_config.total_episodes, ep_return)
        if (ep + 1) % train_config.save_every_n_episodes == 0:
            agent.save(checkpoint_dir)

    mean_ret = float(np.mean(episode_returns)) if episode_returns else 0.0
    best_ret = float(np.max(episode_returns)) if episode_returns else 0.0
    summary = ExitTrainSummary(
        episodes=train_config.total_episodes,
        returns=episode_returns,
        mean_return=mean_ret,
        best_return=best_ret,
        checkpoint_dir=str(checkpoint_dir),
    )
    summary_path = checkpoint_dir / "exit_train_summary.json"
    summary_path.write_text(json.dumps(summary.__dict__, indent=2))
    return summary
