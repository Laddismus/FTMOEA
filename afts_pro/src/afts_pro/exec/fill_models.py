from __future__ import annotations

from datetime import datetime

from typing import Dict, Any

from pydantic import BaseModel, Field

from afts_pro.exec.order_models import OrderSide


class Fill(BaseModel):
    order_id: str
    trade_id: str
    symbol: str
    side: OrderSide
    qty: float
    price: float
    fee: float
    fee_asset: str
    timestamp: datetime
    meta: Dict[str, Any] = Field(default_factory=dict)

    model_config = {
        "populate_by_name": True,
        "arbitrary_types_allowed": True,
    }
