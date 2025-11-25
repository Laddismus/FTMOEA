from __future__ import annotations

from typing import Literal

from afts_pro.config.base_models import BaseConfigModel


class ExecutionConfig(BaseConfigModel):
    slippage_model: str
    max_slippage_pct: float
    fee_model: str
    taker_fee_pct: float
    maker_fee_pct: float
    fill_timing: Literal["bar_open", "bar_close"]
    allow_partial_fills: bool
    tick_size: float
    min_notional: float
