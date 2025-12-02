"""Domain models for the governance and promotion layer."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, field_validator

from research_lab.backend.core.utils.datetime import ensure_utc_datetime


class ModelType(str, Enum):
    """Supported model artifact types."""

    BACKTEST_STRATEGY = "backtest_strategy"
    RL_POLICY = "rl_policy"
    HYBRID = "hybrid"


class ModelStage(str, Enum):
    """Promotion stages for governed models."""

    IDEA = "idea"
    CANDIDATE = "candidate"
    QUALIFIED = "qualified"
    APPROVED = "approved"
    ARCHIVED = "archived"


class GovernanceTag(BaseModel):
    """Simple key/value tag for classification."""

    key: str
    value: str


class BacktestLink(BaseModel):
    """Links a governance entry to backtest artifacts."""

    backtest_id: Optional[str] = None
    experiment_id: Optional[str] = None
    experiment_run_id: Optional[str] = None


class RlLink(BaseModel):
    """Links a governance entry to RL artifacts."""

    rl_run_id: Optional[str] = None
    rl_experiment_id: Optional[str] = None
    rl_experiment_run_id: Optional[str] = None


class KpiSnapshot(BaseModel):
    """Backtest KPI snapshot for governance."""

    total_return: Optional[float] = None
    profit_factor: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    trade_count: Optional[int] = None


class FtmoSnapshot(BaseModel):
    """FTMO risk snapshot for governance."""

    passed: Optional[bool] = None
    first_breach_type: Optional[str] = None
    worst_daily_drawdown_pct: Optional[float] = None
    worst_total_drawdown_pct: Optional[float] = None


class RlSnapshot(BaseModel):
    """RL metrics snapshot for governance."""

    mean_return: Optional[float] = None
    std_return: Optional[float] = None
    max_return: Optional[float] = None
    steps: Optional[int] = None
    reward_checks_passed: Optional[bool] = None
    failed_checks: List[str] = []


class GovernanceScore(BaseModel):
    """Composite governance score."""

    composite_score: Optional[float] = None
    notes: Optional[str] = None


class ModelEntry(BaseModel):
    """Complete governance entry representing a promoted model."""

    id: str
    name: str
    type: ModelType
    stage: ModelStage
    created_at: datetime
    updated_at: datetime
    backtest_link: Optional[BacktestLink] = None
    rl_link: Optional[RlLink] = None
    kpi: KpiSnapshot = KpiSnapshot()
    ftmo: FtmoSnapshot = FtmoSnapshot()
    rl: RlSnapshot = RlSnapshot()
    score: GovernanceScore = GovernanceScore()
    tags: List[GovernanceTag] = []
    metadata: Dict[str, Any] = {}

    @field_validator("created_at", "updated_at", mode="before")
    @classmethod
    def _ensure_timestamps_timezone(cls, value: datetime | str) -> datetime:
        return ensure_utc_datetime(value)


class ModelEntrySummary(BaseModel):
    """Lightweight summary for listing governance entries."""

    id: str
    name: str
    type: ModelType
    stage: ModelStage
    created_at: datetime
    updated_at: datetime
    total_return: Optional[float] = None
    profit_factor: Optional[float] = None
    mean_return: Optional[float] = None
    ftmo_passed: Optional[bool] = None

    @field_validator("created_at", "updated_at", mode="before")
    @classmethod
    def _ensure_summary_timestamps_timezone(cls, value: datetime | str) -> datetime:
        return ensure_utc_datetime(value)


__all__ = [
    "ModelType",
    "ModelStage",
    "GovernanceTag",
    "BacktestLink",
    "RlLink",
    "KpiSnapshot",
    "FtmoSnapshot",
    "RlSnapshot",
    "GovernanceScore",
    "ModelEntry",
    "ModelEntrySummary",
]
