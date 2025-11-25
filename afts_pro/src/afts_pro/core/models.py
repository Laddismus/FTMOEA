from datetime import datetime
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field


class ApplicationMetadata(BaseModel):
    """
    Minimal Pydantic model placeholder.
    """

    name: str = Field(default="AFTS-PRO")
    version: str = Field(default="0.0.1")


class PositionState(BaseModel):
    """
    Placeholder for position-related state.
    """

    size: float = Field(default=0.0)
    average_entry_price: Optional[float] = Field(default=None)
    leverage: Optional[float] = Field(default=None)


class MarketState(BaseModel):
    """
    Represents a single OHLCV bar enriched with optional metadata.
    """

    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    extras: Dict[str, Any] = Field(default_factory=dict)
    regime: Optional[int] = Field(default=None)
    features: Dict[str, Any] = Field(default_factory=dict)
    position: Optional[PositionState] = Field(default=None)


class StrategyDecision(BaseModel):
    """
    Output contract for strategies when processing a bar.
    """

    action: Literal["none", "entry", "manage", "exit"] = Field(default="none")
    side: Optional[Literal["long", "short"]] = Field(default=None)
    confidence: float = Field(default=1.0)
    update: Dict[str, Any] = Field(default_factory=dict)
    meta: Dict[str, Any] = Field(default_factory=dict)
