from datetime import datetime
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class OHLCV(BaseModel):
    open: float
    high: float
    low: float
    close: float
    volume: float


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

    timestamp: Optional[datetime] = Field(default=None)
    symbol: str
    open: float = Field(default=0.0)
    high: float = Field(default=0.0)
    low: float = Field(default=0.0)
    close: float = Field(default=0.0)
    volume: float = Field(default=0.0)
    ohlcv: Optional[OHLCV] = Field(default=None)
    extras: Dict[str, Any] = Field(default_factory=dict)
    regime: Optional[int] = Field(default=None)
    features: Dict[str, Any] = Field(default_factory=dict)
    position: Optional[PositionState] = Field(default=None)

    @model_validator(mode="after")
    def _populate_from_ohlcv(self):
        if self.ohlcv:
            self.open = self.ohlcv.open
            self.high = self.ohlcv.high
            self.low = self.ohlcv.low
            self.close = self.ohlcv.close
            self.volume = self.ohlcv.volume
        return self


class StrategyDecision(BaseModel):
    """
    Output contract for strategies when processing a bar.
    """

    action: Literal["none", "entry", "manage", "exit"] = Field(default="none")
    side: Optional[Literal["long", "short"]] = Field(default=None)
    confidence: float = Field(default=1.0)
    update: Dict[str, Any] = Field(default_factory=dict)
    meta: Dict[str, Any] = Field(default_factory=dict)
