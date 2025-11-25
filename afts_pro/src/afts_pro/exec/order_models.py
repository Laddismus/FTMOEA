from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_MARKET = "STOP_MARKET"
    STOP_LIMIT = "STOP_LIMIT"


class OrderStatus(str, Enum):
    NEW = "NEW"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELED = "CANCELED"


class TimeInForce(str, Enum):
    GTC = "GTC"
    IOC = "IOC"
    FOK = "FOK"


class Order(BaseModel):
    id: str
    client_order_id: Optional[str] = Field(default=None)
    symbol: str
    side: OrderSide
    type: OrderType
    qty: float
    price: Optional[float] = Field(default=None)
    stop_price: Optional[float] = Field(default=None)
    reduce_only: bool = Field(default=False)
    is_sl: bool = Field(default=False)
    is_tp: bool = Field(default=False)
    time_in_force: TimeInForce = Field(default=TimeInForce.GTC)
    status: OrderStatus = Field(default=OrderStatus.NEW)
    created_at: datetime
    updated_at: datetime

    model_config = {
        "populate_by_name": True,
        "arbitrary_types_allowed": True,
    }
