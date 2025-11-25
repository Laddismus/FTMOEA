from __future__ import annotations

from enum import Enum
from typing import Dict

from pydantic import BaseModel, Field

from afts_pro.exec.order_models import Order


class PositionSide(str, Enum):
    LONG = "LONG"
    SHORT = "SHORT"


class Position(BaseModel):
    symbol: str
    side: PositionSide
    qty: float
    entry_price: float
    realized_pnl: float
    unrealized_pnl: float
    avg_entry_fees: float

    model_config = {
        "populate_by_name": True,
        "arbitrary_types_allowed": True,
    }


class AccountState(BaseModel):
    balance: float
    equity: float
    realized_pnl: float
    unrealized_pnl: float
    fees_total: float
    positions: Dict[str, Position] = Field(default_factory=dict)
    open_orders: Dict[str, Order] = Field(default_factory=dict)

    model_config = {
        "populate_by_name": True,
        "arbitrary_types_allowed": True,
    }
