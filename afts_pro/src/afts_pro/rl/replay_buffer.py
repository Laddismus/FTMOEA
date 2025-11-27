from __future__ import annotations

import numpy as np
from typing import Dict, Any


class ReplayBuffer:
    """
    Simple ring-buffer replay storage for RL training.
    """

    def __init__(self, capacity: int, obs_dim: int) -> None:
        self.capacity = capacity
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity,), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self._ptr = 0
        self.size = 0

    def add(self, obs: np.ndarray, action: float, reward: float, next_obs: np.ndarray, done: bool) -> None:
        idx = self._ptr % self.capacity
        self.obs[idx] = obs
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_obs[idx] = next_obs
        self.dones[idx] = float(done)
        self._ptr = (self._ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, Any]:
        if self.size < batch_size:
            raise ValueError("Not enough samples in buffer to draw a batch.")
        idxs = np.random.choice(self.size, size=batch_size, replace=False)
        return {
            "obs": self.obs[idxs],
            "actions": self.actions[idxs],
            "rewards": self.rewards[idxs],
            "next_obs": self.next_obs[idxs],
            "dones": self.dones[idxs],
        }
