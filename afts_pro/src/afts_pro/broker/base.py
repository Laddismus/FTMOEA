from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

Side = Literal["long", "short"]


@dataclass
class LivePrice:
    symbol: str
    bid: float
    ask: float
    timestamp: float  # epoch seconds


@dataclass
class LivePosition:
    symbol: str
    size: float
    entry_price: float
    side: Side
    sl_price: float | None = None
    tp_price: float | None = None


@dataclass
class LiveOrderResult:
    order_id: str
    status: Literal["accepted", "rejected"]
    reason: str | None = None


class BrokerClient(Protocol):
    """
    Minimal interface for live trading.
    """

    def get_price(self, symbol: str) -> LivePrice:
        ...

    def get_position(self, symbol: str) -> LivePosition | None:
        ...

    def send_entry_order(self, symbol: str, side: Side, size: float, sl: float | None, tp: float | None) -> LiveOrderResult:
        ...

    def send_exit_order(self, symbol: str, size: float) -> LiveOrderResult:
        ...

    def modify_sl_tp(self, symbol: str, sl: float | None, tp: float | None) -> LiveOrderResult:
        ...
