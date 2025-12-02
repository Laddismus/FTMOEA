"""Pydantic models for backtest requests and results."""

from __future__ import annotations

from typing import Any, Optional
from typing import Literal
from datetime import datetime, timezone
from enum import Enum

from pydantic import BaseModel, Field, field_validator
from research_lab.backend.core.risk_guard.models import FtmoRiskConfig, FtmoRiskSummary
from research_lab.backend.core.utils.datetime import ensure_utc_datetime


BacktestMode = Literal["graph", "python"]


class PositionSide(str, Enum):
    FLAT = "flat"
    LONG = "long"
    SHORT = "short"


class TradingAction(str, Enum):
    HOLD = "hold"
    ENTER_LONG = "enter_long"
    ENTER_SHORT = "enter_short"
    EXIT = "exit"


class StrategyGraphRef(BaseModel):
    """Reference to a strategy graph/DSL."""

    graph_id: Optional[str] = None
    dsl: Optional[dict[str, Any]] = None
    engine_config: Optional[dict[str, Any]] = None


class PythonStrategyRef(BaseModel):
    """Reference to a Python strategy."""

    key: Optional[str] = None
    module_path: Optional[str] = None
    class_name: Optional[str] = None


class BacktestRequest(BaseModel):
    """Request payload for running a backtest."""

    mode: BacktestMode
    graph: Optional[StrategyGraphRef] = None
    python_strategy: Optional[PythonStrategyRef] = None
    returns: Optional[list[float]] = None
    bars: Optional[list["BacktestBar"]] = None
    window: int = 50
    strategy_params: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    cost_model: Optional["BacktestCostModel"] = None
    ftmo_risk: Optional[FtmoRiskConfig] = None


class BacktestKpiSummary(BaseModel):
    """Aggregated KPI summary for a backtest."""

    total_return: float
    mean_return: float
    std_return: float
    profit_factor: float
    win_rate: float
    max_drawdown: float
    trade_count: int


class BacktestEngineDetail(BaseModel):
    """Engine-level details for a backtest execution."""

    window_kpis: list[dict[str, Any]] = Field(default_factory=list)


class BacktestResult(BaseModel):
    """Full backtest result payload."""

    id: str
    created_at: Optional[datetime] = None
    mode: BacktestMode
    graph: Optional[StrategyGraphRef] = None
    python_strategy: Optional[PythonStrategyRef] = None
    kpi_summary: BacktestKpiSummary
    engine_detail: BacktestEngineDetail
    metadata: dict[str, Any] = Field(default_factory=dict)
    strategy_metadata: Optional[dict[str, Any]] = None
    trades: Optional[list["BacktestTrade"]] = None
    ftmo_risk_summary: Optional[FtmoRiskSummary] = None

    @field_validator("created_at", mode="before")
    @classmethod
    def _ensure_created_at_timezone(cls, value: datetime | str | None) -> datetime | None:
        if value is None:
            return None
        return ensure_utc_datetime(value)


class BacktestIndexEntry(BaseModel):
    """Lightweight view for listing backtest runs."""

    id: str
    created_at: datetime
    mode: BacktestMode
    metadata: dict[str, Any] = Field(default_factory=dict)
    kpi_summary: Optional[BacktestKpiSummary] = None

    @field_validator("created_at", mode="before")
    @classmethod
    def _ensure_index_created_at_timezone(cls, value: datetime | str) -> datetime:
        return ensure_utc_datetime(value)


class BacktestTrade(BaseModel):
    """Represents an executed trade for journaling."""

    entry_ts: datetime
    exit_ts: datetime
    entry_price: float
    exit_price: float
    pnl: float
    return_: float
    gross_return: Optional[float] = None
    net_return: Optional[float] = None
    fees: Optional[float] = None
    side: Optional[PositionSide] = None
    size: Optional[float] = None

    @field_validator("entry_ts", "exit_ts", mode="before")
    @classmethod
    def _ensure_trade_timestamps_timezone(cls, value: datetime | str) -> datetime:
        return ensure_utc_datetime(value)


class BacktestPositionState(BaseModel):
    """Represents the current trading position state."""

    side: PositionSide = PositionSide.FLAT
    size: float = 0.0
    entry_price: Optional[float] = None
    entry_ts: Optional[datetime] = None
    equity: float = 1.0
    costs_accrued: float = 0.0

    @field_validator("entry_ts", mode="before")
    @classmethod
    def _ensure_position_entry_ts_timezone(cls, value: datetime | str | None) -> datetime | None:
        if value is None:
            return None
        return ensure_utc_datetime(value)


class BacktestBar(BaseModel):
    """Simple OHLCV bar representation for backtests."""

    ts: datetime
    open: float
    high: float
    low: float
    close: float
    volume: Optional[float] = None

    @field_validator("ts", mode="before")
    @classmethod
    def _ensure_bar_timestamp_timezone(cls, value: datetime | str) -> datetime:
        return ensure_utc_datetime(value)


class BacktestCostModel(BaseModel):
    """Cost model for backtests."""

    fee_rate: float = 0.0
    slippage_rate: float = 0.0


__all__ = [
    "BacktestMode",
    "StrategyGraphRef",
    "PythonStrategyRef",
    "BacktestRequest",
    "BacktestKpiSummary",
    "BacktestEngineDetail",
    "BacktestResult",
    "BacktestIndexEntry",
    "BacktestTrade",
    "BacktestPositionState",
    "BacktestBar",
    "BacktestCostModel",
    "PositionSide",
    "TradingAction",
    "ensure_utc_datetime",
    "FtmoRiskConfig",
    "FtmoRiskSummary",
]

# Resolve forward references
BacktestRequest.model_rebuild()
BacktestResult.model_rebuild()
