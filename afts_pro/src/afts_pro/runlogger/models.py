from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class RunMeta(BaseModel):
    run_id: str
    mode: str
    profile_name: str
    started_at: datetime
    finished_at: Optional[datetime] = None
    symbol: str
    timeframe: str
    seed: Optional[int] = None

    model_config = {"populate_by_name": True}


class TradeRecord(BaseModel):
    trade_id: str
    symbol: str
    side: str
    entry_timestamp: datetime
    exit_timestamp: datetime
    entry_price: float
    exit_price: float
    size: float
    realized_pnl: float
    fees: float
    max_favourable_excursion: Optional[float] = None
    max_adverse_excursion: Optional[float] = None
    tags: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}


class EquityPoint(BaseModel):
    timestamp: datetime
    equity: float
    balance: float
    unrealized_pnl: float
    realized_pnl_cum: float
    max_equity_to_date: float
    drawdown_abs: float
    drawdown_pct: float

    model_config = {"populate_by_name": True}


class MetricsSnapshot(BaseModel):
    profit_factor: Optional[float] = None
    winrate: Optional[float] = None
    avg_win: Optional[float] = None
    avg_loss: Optional[float] = None
    expectancy_per_trade: Optional[float] = None
    num_trades: int = 0
    max_drawdown_abs: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    cagr_simulated: Optional[float] = None
    sharpe_like_basic: Optional[float] = None
    additional: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}
