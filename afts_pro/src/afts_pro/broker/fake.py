from __future__ import annotations

import time
from dataclasses import dataclass, field

from afts_pro.broker.base import BrokerClient, LivePrice, LivePosition, LiveOrderResult, Side


@dataclass
class FakeBrokerState:
    prices: dict[str, LivePrice] = field(default_factory=dict)
    positions: dict[str, LivePosition] = field(default_factory=dict)
    next_order_id: int = 1


class FakeBroker(BrokerClient):
    def __init__(self, state: FakeBrokerState | None = None):
        self.state = state or FakeBrokerState()

    def get_price(self, symbol: str) -> LivePrice:
        price = self.state.prices.get(symbol)
        if price is None:
            now = time.time()
            price = LivePrice(symbol=symbol, bid=100.0, ask=100.5, timestamp=now)
            self.state.prices[symbol] = price
        return price

    def get_position(self, symbol: str) -> LivePosition | None:
        return self.state.positions.get(symbol)

    def send_entry_order(self, symbol: str, side: Side, size: float, sl: float | None, tp: float | None) -> LiveOrderResult:
        price = self.get_price(symbol)
        entry = price.ask if side == "long" else price.bid
        pos = self.state.positions.get(symbol)
        if pos is None:
            pos = LivePosition(symbol=symbol, size=size, entry_price=entry, side=side, sl_price=sl, tp_price=tp)
        else:
            total_size = pos.size + size
            avg_price = (pos.entry_price * pos.size + entry * size) / total_size
            pos.size = total_size
            pos.entry_price = avg_price
            if sl is not None:
                pos.sl_price = sl
            if tp is not None:
                pos.tp_price = tp
        self.state.positions[symbol] = pos
        oid = f"FAKE-{self.state.next_order_id}"
        self.state.next_order_id += 1
        return LiveOrderResult(order_id=oid, status="accepted")

    def send_exit_order(self, symbol: str, size: float) -> LiveOrderResult:
        pos = self.state.positions.get(symbol)
        if pos is None:
            return LiveOrderResult(order_id="NA", status="rejected", reason="no_position")
        remaining = pos.size - size
        if remaining <= 0:
            self.state.positions.pop(symbol, None)
        else:
            pos.size = remaining
        oid = f"FAKE-{self.state.next_order_id}"
        self.state.next_order_id += 1
        return LiveOrderResult(order_id=oid, status="accepted")

    def modify_sl_tp(self, symbol: str, sl: float | None, tp: float | None) -> LiveOrderResult:
        pos = self.state.positions.get(symbol)
        if pos is None:
            return LiveOrderResult(order_id="NA", status="rejected", reason="no_position")
        if sl is not None:
            pos.sl_price = sl
        if tp is not None:
            pos.tp_price = tp
        oid = f"FAKE-{self.state.next_order_id}"
        self.state.next_order_id += 1
        return LiveOrderResult(order_id=oid, status="accepted")
