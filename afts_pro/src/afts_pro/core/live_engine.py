from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from afts_pro.broker.base import LivePrice, LivePosition


@dataclass
class LiveOrder:
    kind: str  # "entry" | "exit" | "modify"
    symbol: str
    side: Optional[str] = None
    size: Optional[float] = None
    sl: Optional[float] = None
    tp: Optional[float] = None


class LiveEngine:
    """
    Minimal live engine wrapper. For now returns stub orders.
    """

    def __init__(self, symbol: str):
        self.symbol = symbol

    def process_live_tick(self, price: LivePrice, position: LivePosition | None) -> List[LiveOrder]:
        # Stub: return no orders. Later hook to strategy pipeline.
        return []
