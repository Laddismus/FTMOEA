"""Domain models for RL experimentation in the Research Lab."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, field_validator

from research_lab.backend.core.utils.datetime import ensure_utc_datetime


class RLEnvRef(BaseModel):
    """Reference to an RL environment."""

    env_id: str
    version: Optional[str] = None


class RLAlgo(str, Enum):
    """Supported RL algorithm identifiers."""

    SAC = "sac"
    PPO = "ppo"
    DQN = "dqn"
    CUSTOM = "custom"


class RLPolicyRef(BaseModel):
    """Reference to a stored RL policy artifact."""

    key: str
    algo: RLAlgo
    path: str
    created_at: datetime
    metadata: Dict[str, Any] = {}

    @field_validator("created_at", mode="before")
    @classmethod
    def _ensure_created_at_timezone(cls, value: datetime | str) -> datetime:
        return ensure_utc_datetime(value)


class RLTrainingConfig(BaseModel):
    """Configuration for an RL training run."""

    env: RLEnvRef
    algo: RLAlgo
    total_timesteps: int
    seed: Optional[int] = None
    gamma: Optional[float] = None
    learning_rate: Optional[float] = None
    batch_size: Optional[int] = None
    hyperparams: Dict[str, Any] = {}


class RLRewardMetricPoint(BaseModel):
    """A single reward metric point over training steps."""

    step: int
    value: float


class RLTrainingMetrics(BaseModel):
    """Aggregated metrics from an RL run."""

    episode_rewards: List[float]
    avg_reward: float
    max_reward: float
    min_reward: float
    reward_curve: List[RLRewardMetricPoint]


class RLRewardCheckConfig(BaseModel):
    """Threshold configuration for reward verification."""

    min_avg_reward: Optional[float] = None
    min_last_n_avg_reward: Optional[float] = None
    window: int = 10


class RLRewardCheckResult(BaseModel):
    """Outcome of a reward verification."""

    passed: bool
    avg_reward: float
    last_n_avg_reward: Optional[float] = None
    reason: Optional[str] = None


class RLRunRequest(BaseModel):
    """Request payload for running RL training."""

    id: str
    created_at: datetime
    config: RLTrainingConfig
    reward_check: Optional[RLRewardCheckConfig] = None
    notes: Optional[str] = None
    tags: List[str] = []

    @field_validator("created_at", mode="before")
    @classmethod
    def _ensure_created_at_timezone(cls, value: datetime | str) -> datetime:
        return ensure_utc_datetime(value)


class RLRunStatus(str, Enum):
    """Lifecycle status for an RL run/job."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class RLRunResult(BaseModel):
    """Result payload of an RL training run."""

    id: str
    config: RLTrainingConfig
    metrics: RLTrainingMetrics
    reward_check_result: Optional[RLRewardCheckResult] = None
    policy_ref: Optional[RLPolicyRef] = None
    created_at: datetime
    completed_at: datetime

    @field_validator("created_at", "completed_at", mode="before")
    @classmethod
    def _ensure_result_timestamps_timezone(cls, value: datetime | str) -> datetime:
        return ensure_utc_datetime(value)
