"""Pydantic models for research experiments and orchestration."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, field_validator

from research_lab.backend.core.backtests.models import BacktestRequest, PythonStrategyRef, StrategyGraphRef
from research_lab.backend.core.utils.datetime import ensure_utc_datetime


class ExperimentStatus(str, Enum):
    """Lifecycle status for an experiment or run."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ExperimentStrategyRef(BaseModel):
    """Strategy reference used by an experiment."""

    mode: str  # "python" or "graph"
    python_strategy: Optional[PythonStrategyRef] = None
    graph: Optional[StrategyGraphRef] = None


class ExperimentParamPoint(BaseModel):
    """Concrete parameter point for a sweep/grid."""

    values: Dict[str, Any]


class ExperimentConfig(BaseModel):
    """Configuration for an experiment and its backtests."""

    id: str
    name: str
    description: str | None = None
    created_at: datetime
    strategy: ExperimentStrategyRef
    base_backtest: BacktestRequest
    param_grid: List[ExperimentParamPoint]
    tags: List[str] = []
    metadata: Dict[str, Any] = {}

    @field_validator("created_at", mode="before")
    @classmethod
    def _ensure_created_at_timezone(cls, value: datetime | str) -> datetime:
        return ensure_utc_datetime(value)


class ExperimentRunStatus(BaseModel):
    """Tracks status of a single backtest run belonging to an experiment."""

    run_id: str
    backtest_id: Optional[str] = None
    job_id: Optional[str] = None
    status: ExperimentStatus
    error: Optional[str] = None


class ExperimentSummary(BaseModel):
    """Summary view for listing experiments."""

    id: str
    name: str
    status: ExperimentStatus
    created_at: datetime
    total_runs: int
    completed_runs: int
    failed_runs: int
    best_total_return: Optional[float] = None
    best_pf: Optional[float] = None
    ftmo_pass_rate: Optional[float] = None

    @field_validator("created_at", mode="before")
    @classmethod
    def _ensure_summary_created_at_timezone(cls, value: datetime | str) -> datetime:
        return ensure_utc_datetime(value)


class ExperimentRunScore(BaseModel):
    """Scorecard for a single experiment run."""

    run_id: str
    backtest_id: str
    params: Dict[str, Any]
    total_return: float
    profit_factor: Optional[float] = None
    max_drawdown: Optional[float] = None
    ftmo_passed: Optional[bool] = None
    ftmo_breach_type: Optional[str] = None
    rank: Optional[int] = None


class ExperimentLeaderboard(BaseModel):
    """Leaderboard view for an experiment."""

    experiment_id: str
    name: str
    created_at: datetime
    total_runs: int
    ftmo_pass_rate: Optional[float] = None
    runs: List[ExperimentRunScore]

    @field_validator("created_at", mode="before")
    @classmethod
    def _ensure_leaderboard_created_at_timezone(cls, value: datetime | str) -> datetime:
        return ensure_utc_datetime(value)
