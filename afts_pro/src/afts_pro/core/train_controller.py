from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from afts_pro.rl.env import RLTradingEnv, load_env_config
from afts_pro.rl.risk_agent import RiskAgent, RiskAgentConfig
from afts_pro.rl.exit_agent import ExitAgent, ExitAgentConfig
from afts_pro.rl.risk_training import TrainLoopConfig, train_risk_agent
from afts_pro.rl.exit_training import ExitTrainConfig, train_exit_agent
from afts_pro.rl.types import ActionSpec, RLObsSpec

logger = logging.getLogger(__name__)


@dataclass
class TrainJobConfig:
    agent_type: str
    env_config_path: str
    agent_config_path: str
    output_dir: str
    seed: Optional[int] = None
    resume_from: Optional[str] = None
    post_analysis: bool = False


@dataclass
class TrainJobResult:
    agent_type: str
    output_dir: str
    episodes: int
    mean_return: Optional[float]
    best_return: Optional[float]
    extra: Dict[str, Any]


class TrainController:
    """
    Orchestrates TRAIN mode runs: env + agent + training loop.
    """

    def __init__(self) -> None:
        logger.info("TrainController initialized")

    def run_train_job(self, job_cfg: TrainJobConfig) -> TrainJobResult:
        logger.info("TRAIN JOB START | agent_type=%s | env_cfg=%s | agent_cfg=%s", job_cfg.agent_type, job_cfg.env_config_path, job_cfg.agent_config_path)
        env_cfg = load_env_config(job_cfg.env_config_path)
        env_cfg["env_type"] = job_cfg.agent_type
        env = RLTradingEnv(env_cfg, event_stream=self._build_dummy_stream())
        obs_spec = RLObsSpec(shape=env.obs_spec.shape, dtype="float32", as_dict=False)

        output_dir = Path(job_cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if job_cfg.agent_type == "risk":
            raw_cfg = yaml.safe_load(Path(job_cfg.agent_config_path).read_text()) or {}
            raw_cfg.pop("n_actions", None)
            agent_cfg = RiskAgentConfig(**raw_cfg)
            agent = RiskAgent(agent_cfg, obs_spec=obs_spec, action_spec=ActionSpec(action_type="continuous", shape=(1,)))
            train_kwargs = {}
            if isinstance(agent_cfg.train, dict):
                allowed = {"total_episodes", "max_steps_per_episode", "replay_capacity", "batch_size", "log_interval", "save_every_n_episodes"}
                train_kwargs = {k: v for k, v in agent_cfg.train.items() if k in allowed}
            train_cfg = TrainLoopConfig(**train_kwargs)
            summary = train_risk_agent(env, agent, train_cfg, checkpoint_dir=output_dir)
            episodes = summary.episodes
            mean_ret = summary.mean_return
            best_ret = summary.best_return
        elif job_cfg.agent_type == "exit":
            raw_cfg = yaml.safe_load(Path(job_cfg.agent_config_path).read_text()) or {}
            agent_cfg = ExitAgentConfig(**{k: v for k, v in raw_cfg.items() if k in ExitAgentConfig().__dict__ or k in {"action_mode", "n_actions", "exploration_epsilon", "sl_tighten_factor", "partial_close_fraction", "train"}})
            agent = ExitAgent(agent_cfg, obs_spec=obs_spec, action_spec=ActionSpec(action_type="discrete", num_actions=agent_cfg.n_actions))
            train_kwargs = {}
            if isinstance(agent_cfg.train, dict):
                allowed = {"total_episodes", "max_steps_per_episode", "replay_capacity", "batch_size", "log_interval", "save_every_n_episodes"}
                train_kwargs = {k: v for k, v in agent_cfg.train.items() if k in allowed}
            train_cfg = ExitTrainConfig(**train_kwargs)
            summary = train_exit_agent(env, agent, train_cfg, checkpoint_dir=output_dir)
            episodes = summary.episodes
            mean_ret = summary.mean_return
            best_ret = summary.best_return
        else:
            raise ValueError(f"Unsupported agent_type: {job_cfg.agent_type}")

        job_snapshot = {
            "agent_type": job_cfg.agent_type,
            "env_config_path": job_cfg.env_config_path,
            "agent_config_path": job_cfg.agent_config_path,
            "output_dir": job_cfg.output_dir,
            "seed": job_cfg.seed,
        }
        (output_dir / "train_job_config.yaml").write_text(yaml.safe_dump(job_snapshot))
        result = TrainJobResult(
            agent_type=job_cfg.agent_type,
            output_dir=str(output_dir),
            episodes=episodes,
            mean_return=mean_ret,
            best_return=best_ret,
            extra={"checkpoint_dir": str(output_dir)},
        )
        logger.info("TRAIN JOB DONE | agent_type=%s | mean_return=%.4f | best_return=%.4f", job_cfg.agent_type, mean_ret, best_ret)
        return result

    def _build_dummy_stream(self, length: int = 500) -> list[Dict[str, Any]]:
        return [{"equity": 1.0, "drawdown": 0.0, "dd_remaining": 1.0, "features": [0.0, 0.0, 0.0, 0.0]} for _ in range(length)]
