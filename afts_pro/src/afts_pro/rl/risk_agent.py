from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml

from afts_pro.rl.types import ActionSpec, RLObsSpec

logger = logging.getLogger(__name__)


@dataclass
class RiskAgentConfig:
    action_mode: str = "continuous"
    min_risk_pct: float = 0.1
    max_risk_pct: float = 3.0
    exploration_epsilon: float = 0.1
    train: Dict[str, Any] = field(
        default_factory=lambda: {
            "total_episodes": 10,
            "max_steps_per_episode": 500,
            "batch_size": 32,
            "learning_rate": 5e-4,
            "gamma": 0.99,
            "replay_capacity": 10000,
            "log_interval": 1,
            "save_every_n_episodes": 5,
        }
    )


class RiskAgent:
    """
    Simple risk control agent producing risk percentage actions.
    """

    def __init__(self, config: RiskAgentConfig, obs_spec: RLObsSpec, action_spec: ActionSpec):
        self.config = config
        self.obs_dim = obs_spec.shape[0]
        self.action_spec = action_spec
        rng = np.random.default_rng()
        self.weights = rng.normal(0, 0.1, size=(self.obs_dim,))
        self.bias = 0.0
        logger.info("RiskAgent initialized | obs_dim=%d | action_mode=%s", self.obs_dim, config.action_mode)

    def _squash(self, x: float) -> float:
        sig = 1.0 / (1.0 + np.exp(-x))
        risk = self.config.min_risk_pct + sig * (self.config.max_risk_pct - self.config.min_risk_pct)
        return float(risk)

    def act(self, obs: np.ndarray, deterministic: bool = False) -> float:
        """
        Map observation to a risk percentage within configured bounds.
        """
        obs = np.asarray(obs, dtype=np.float32)
        base_action = self._squash(float(np.dot(obs, self.weights) + self.bias))
        if deterministic:
            return base_action
        if np.random.rand() < self.config.exploration_epsilon:
            rand_action = np.random.uniform(self.config.min_risk_pct, self.config.max_risk_pct)
            return float(rand_action)
        return base_action

    def train_on_batch(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Placeholder training step; returns dummy stats.
        """
        # TODO: implement proper optimization (e.g., policy gradient).
        return {"loss": 0.0}

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        cfg_path = path / "risk_agent_config.yaml"
        weights_path = path / "risk_agent_weights.npz"
        with cfg_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(self.config.__dict__, fh)
        np.savez(weights_path, weights=self.weights, bias=self.bias)
        logger.info("RiskAgent saved to %s", path)

    def load(self, path: Path) -> None:
        cfg_path = path / "risk_agent_config.yaml"
        weights_path = path / "risk_agent_weights.npz"
        if cfg_path.exists():
            data = yaml.safe_load(cfg_path.read_text())
            self.config = RiskAgentConfig(**data)
        if weights_path.exists():
            data = np.load(weights_path)
            self.weights = data["weights"]
            self.bias = float(data["bias"])
        logger.info("RiskAgent loaded from %s", path)
