from __future__ import annotations

import logging
from typing import Dict, List, Optional

from afts_pro.core import MarketState
from afts_pro.exec.fill_models import Fill
from afts_pro.exec.order_models import Order, OrderSide, OrderStatus, OrderType
from afts_pro.exec.position_models import AccountState, PositionSide

logger = logging.getLogger(__name__)


class SimFillEngine:
    """
    Bar-based fill simulation for deterministic backtests.

    Note: Orders generated on bar t should typically become active and fill no sooner than bar t+1;
    timing is controlled by the engine loop.
    """

    def __init__(
        self,
        fee_rate: float = 0.0004,
        slippage_ticks: float = 0.0,
        tick_size: float = 0.1,
        slippage_pct: float = 0.0,
    ) -> None:
        self.fee_rate = fee_rate
        self.slippage_ticks = slippage_ticks
        self.tick_size = tick_size
        self.slippage_pct = slippage_pct

    def process_bar(
        self,
        *,
        account_state: AccountState,
        open_orders: Dict[str, Order],
        market_state: MarketState,
        last_bar: Optional[MarketState] = None,
    ) -> List[Fill]:
        fills: List[Fill] = []
        if not open_orders:
            return fills

        for order_id, order in list(open_orders.items()):
            fill = self._try_fill_order(order, market_state, last_bar, account_state)
            if fill:
                fills.append(fill)
                order.status = OrderStatus.FILLED
                open_orders.pop(order_id, None)
        return fills

    def _try_fill_order(
        self,
        order: Order,
        bar: MarketState,
        last_bar: Optional[MarketState],
        account_state: AccountState,
    ) -> Optional[Fill]:
        if order.type == OrderType.MARKET:
            price = self._apply_slippage(bar.open, order.side)
            qty = self._respect_reduce_only(order, account_state, bar.symbol)
            if qty <= 0:
                logger.debug("Reduce-only market order skipped; no position to reduce.")
                return None
            return self._build_fill(order, bar, qty, price)

        if order.type == OrderType.LIMIT:
            if bar.low <= (order.price or 0) <= bar.high:
                qty = self._respect_reduce_only(order, account_state, bar.symbol)
                if qty <= 0:
                    logger.debug("Reduce-only limit order skipped; no position to reduce.")
                    return None
                price = self._apply_slippage(order.price or bar.open, order.side)
                return self._build_fill(order, bar, qty, price)
            return None

        if order.type == OrderType.STOP_MARKET:
            return self._try_fill_stop_market(order, bar, last_bar, account_state)

        return None

    def _try_fill_stop_market(
        self,
        order: Order,
        bar: MarketState,
        last_bar: Optional[MarketState],
        account_state: AccountState,
    ) -> Optional[Fill]:
        if order.stop_price is None:
            return None

        triggered = False
        price = None

        if order.side == OrderSide.BUY:
            if bar.high >= order.stop_price:
                triggered = True
                price = max(order.stop_price, bar.open)
        else:
            if bar.low <= order.stop_price:
                triggered = True
                price = min(order.stop_price, bar.open)

        if not triggered:
            return None

        qty = self._respect_reduce_only(order, account_state, bar.symbol)
        if qty <= 0:
            logger.debug("Reduce-only stop-market order skipped; no position to reduce.")
            return None

        price = self._apply_slippage(price or bar.open, order.side)
        return self._build_fill(order, bar, qty, price)

    def _apply_slippage(self, price: float, side: OrderSide) -> float:
        if self.slippage_pct > 0:
            adjustment = price * self.slippage_pct
            return price + adjustment if side == OrderSide.BUY else price - adjustment
        if self.slippage_ticks <= 0:
            return price
        adjustment = self.slippage_ticks * self.tick_size
        return price + adjustment if side == OrderSide.BUY else price - adjustment

    def _respect_reduce_only(self, order: Order, account_state: AccountState, symbol: str) -> float:
        if not order.reduce_only:
            return order.qty
        position = account_state.positions.get(symbol)
        if position is None:
            return 0.0
        same_side = OrderSide.BUY if position.side == PositionSide.LONG else OrderSide.SELL
        if order.side == same_side:
            return 0.0
        qty = order.qty if order.qty > 0 else position.qty
        return min(qty, position.qty)

    def _build_fill(self, order: Order, bar: MarketState, qty: float, price: float) -> Fill:
        fee = abs(qty * price) * self.fee_rate
        return Fill(
            order_id=order.id,
            trade_id=order.id,
            symbol=bar.symbol,
            side=order.side,
            qty=qty,
            price=price,
            fee=fee,
            fee_asset="USD",
            timestamp=bar.timestamp,
            meta={
                "is_sl": order.is_sl,
                "is_tp": order.is_tp,
                "reduce_only": order.reduce_only,
                "order_type": order.type.value,
            },
        )
