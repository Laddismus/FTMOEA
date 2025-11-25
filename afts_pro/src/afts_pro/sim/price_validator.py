from __future__ import annotations

import logging
from datetime import datetime
from typing import Optional

from afts_pro.core import MarketState
from afts_pro.exec.order_models import Order

logger = logging.getLogger(__name__)


class PriceValidator:
    """
    Validates bar ordering and prevents retro-fills.
    """

    def validate_bar_sequence(self, prev_bar: Optional[MarketState], curr_bar: MarketState) -> None:
        if prev_bar is None:
            return

        if curr_bar.timestamp <= prev_bar.timestamp:
            logger.error(
                "Bar timestamp ordering violated: prev=%s curr=%s",
                prev_bar.timestamp.isoformat(),
                curr_bar.timestamp.isoformat(),
            )
            raise ValueError("Bar sequence timestamp violation.")

    def validate_fill_timing(self, order: Order, fill_ts: datetime) -> None:
        if fill_ts < order.created_at:
            logger.error(
                "Retro-fill detected: order_created=%s fill_ts=%s order_id=%s",
                order.created_at.isoformat(),
                fill_ts.isoformat(),
                order.id,
            )
            raise ValueError("Fill timestamp precedes order creation.")
