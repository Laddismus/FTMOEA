from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

from afts_pro.rl.types import ActionSpec, RLObsSpec

logger = logging.getLogger(__name__)


@dataclass
class ExitAgentConfig:
    action_mode: str = "discrete"
    n_actions: int = 6
    exploration_epsilon: float = 0.1
    sl_tighten_factor: float = 0.2
    partial_close_fraction: float = 0.3
    train: Dict[str, Any] = field(
        default_factory=lambda: {
            "total_episodes": 10,
            "max_steps_per_episode": 500,
            "batch_size": 32,
            "learning_rate": 5e-4,
            "gamma": 0.99,
            "replay_capacity": 5000,
            "log_interval": 1,
        }
    )


class ExitAgent:
    """
    Simple exit decision agent producing discrete actions.

    Action semantics (docstring reference):
      0 = noop
      1 = tighten_sl
      2 = move_sl_to_be
      3 = trail_sl
      4 = partial_close
      5 = full_close
    """

    def __init__(self, config: ExitAgentConfig, obs_spec: RLObsSpec, action_spec: ActionSpec):
        self.config = config
        self.obs_dim = obs_spec.shape[0]
        self.action_spec = action_spec
        rng = np.random.default_rng()
        self.weights = rng.normal(0, 0.1, size=(self.obs_dim, config.n_actions))
        self.bias = np.zeros((config.n_actions,), dtype=np.float32)
        logger.info("ExitAgent initialized | obs_dim=%d | n_actions=%d", self.obs_dim, config.n_actions)

    def act(self, obs: np.ndarray, deterministic: bool = False) -> int:
        obs = np.asarray(obs, dtype=np.float32)
        logits = obs @ self.weights + self.bias
        if not deterministic and np.random.rand() < self.config.exploration_epsilon:
            action = int(np.random.randint(0, self.config.n_actions))
            return action
        action = int(np.argmax(logits))
        action = max(0, min(self.config.n_actions - 1, action))
        return action

    def train_on_batch(self, batch: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Placeholder training; returns dummy stats.
        """
        return {"loss": 0.0}

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        cfg_path = path / "exit_agent_config.yaml"
        weights_path = path / "exit_agent_weights.npz"
        with cfg_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(self.config.__dict__, fh)
        np.savez(weights_path, weights=self.weights, bias=self.bias)
        logger.info("ExitAgent saved to %s", path)

    def load(self, path: Path) -> None:
        cfg_path = path / "exit_agent_config.yaml"
        weights_path = path / "exit_agent_weights.npz"
        if cfg_path.exists():
            data = yaml.safe_load(cfg_path.read_text())
            self.config = ExitAgentConfig(**data)
        if weights_path.exists():
            data = np.load(weights_path)
            self.weights = data["weights"]
            self.bias = data["bias"]
        logger.info("ExitAgent loaded from %s", path)
