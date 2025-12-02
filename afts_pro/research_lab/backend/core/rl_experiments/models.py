"""Pydantic models for RL experiments and leaderboards."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, field_validator

from research_lab.backend.core.rl.models import (
    RLEnvRef,
    RLAlgo,
    RLTrainingConfig,
    RLRunResult,
    RLRewardCheckConfig,
)
from research_lab.backend.core.utils.datetime import ensure_utc_datetime


class RlExperimentStatus(str, Enum):
    """Lifecycle status for RL experiments and runs."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class RlExperimentParamPoint(BaseModel):
    """Concrete hyperparameter point for an RL experiment grid."""

    values: Dict[str, Any]


class RlExperimentConfig(BaseModel):
    """Configuration of an RL experiment."""

    id: str
    name: str
    description: Optional[str] = None
    created_at: datetime
    env: RLEnvRef
    algo: RLAlgo
    base_training: RLTrainingConfig
    param_grid: List[RlExperimentParamPoint]
    reward_checks: List[RLRewardCheckConfig] = []
    tags: List[str] = []
    metadata: Dict[str, Any] = {}

    @field_validator("created_at", mode="before")
    @classmethod
    def _ensure_created_at_timezone(cls, value: datetime | str) -> datetime:
        return ensure_utc_datetime(value)


class RlExperimentRunStatus(BaseModel):
    """Tracks status of a single RL run within an experiment."""

    run_id: str
    job_id: Optional[str] = None
    status: RlExperimentStatus
    error: Optional[str] = None
    rl_run_id: Optional[str] = None


class RlExperimentRunScore(BaseModel):
    """Scorecard for a single RL experiment run."""

    run_id: str
    rl_run_id: Optional[str] = None
    params: Dict[str, Any]
    mean_return: float
    std_return: Optional[float] = None
    max_return: Optional[float] = None
    steps: Optional[int] = None
    reward_checks_passed: Optional[bool] = None
    failed_checks: List[str] = []
    composite_score: Optional[float] = None
    rank: Optional[int] = None


class RlExperimentLeaderboard(BaseModel):
    """Leaderboard view of experiment runs."""

    experiment_id: str
    name: str
    created_at: datetime
    total_runs: int
    passes: int
    pass_rate: Optional[float]
    runs: List[RlExperimentRunScore]

    @field_validator("created_at", mode="before")
    @classmethod
    def _ensure_leaderboard_created_at_timezone(cls, value: datetime | str) -> datetime:
        return ensure_utc_datetime(value)

