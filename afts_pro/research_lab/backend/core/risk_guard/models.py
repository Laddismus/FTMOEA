"""Models for FTMO-inspired risk guard evaluation."""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, field_validator

from research_lab.backend.core.utils.datetime import ensure_utc_datetime


class FtmoRiskConfig(BaseModel):
    """Configuration for FTMO-like risk limits."""

    initial_equity: float = 1.0
    max_daily_loss_pct: float = 0.05
    max_total_loss_pct: float = 0.10
    safety_buffer_pct: float = 0.0


class FtmoBreachType(str, Enum):
    """Type of FTMO rule breach."""

    NONE = "none"
    DAILY = "daily"
    TOTAL = "total"


class FtmoRiskEvent(BaseModel):
    """Captures the first breach event during evaluation."""

    ts: datetime
    breach_type: FtmoBreachType
    equity: float
    drawdown_pct: float
    day: date

    @field_validator("ts", mode="before")
    @classmethod
    def _ensure_ts_timezone(cls, value: datetime | str) -> datetime:
        return ensure_utc_datetime(value)


class FtmoRiskSummary(BaseModel):
    """Summary of FTMO risk evaluation for a backtest run."""

    passed: bool
    first_breach: Optional[FtmoRiskEvent] = None
    worst_daily_drawdown_pct: float
    worst_total_drawdown_pct: float
    config: FtmoRiskConfig
