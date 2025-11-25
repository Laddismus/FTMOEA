from __future__ import annotations

import logging
from typing import Dict, Optional

from pydantic import BaseModel, Field

from afts_pro.exec.fill_models import Fill
from afts_pro.exec.order_models import OrderSide
from afts_pro.exec.position_models import AccountState, Position, PositionSide

logger = logging.getLogger(__name__)


class PositionEvent(BaseModel):
    symbol: str
    event_type: str
    realized_pnl_delta: float = Field(default=0.0)

    model_config = {"populate_by_name": True}


class PositionManager:
    """
    Applies fills to maintain positions and account state.
    """

    def apply_fill(self, fill: Fill, account_state: AccountState) -> PositionEvent:
        account_state.fees_total += fill.fee
        position = account_state.positions.get(fill.symbol)

        if position is None:
            self._open_position(fill, account_state)
            self.update_unrealized_pnl(account_state, market_price=fill.price)
            return PositionEvent(symbol=fill.symbol, event_type="OPENED")

        if self._is_same_side(position.side, fill.side):
            self._increase_position(position, fill)
            self.update_unrealized_pnl(account_state, market_price=fill.price)
            return PositionEvent(symbol=fill.symbol, event_type="INCREASED")

        pnl_delta = self._reduce_position(position, fill, account_state)
        self.update_unrealized_pnl(account_state, market_price=fill.price)

        if position.qty == 0:
            return PositionEvent(symbol=fill.symbol, event_type="CLOSED", realized_pnl_delta=pnl_delta)
        return PositionEvent(symbol=fill.symbol, event_type="REDUCED", realized_pnl_delta=pnl_delta)

    def update_unrealized_pnl(self, account_state: AccountState, market_price: float) -> None:
        total_unrealized = 0.0
        for position in account_state.positions.values():
            pos_unrealized = (
                (market_price - position.entry_price) * position.qty
                if position.side == PositionSide.LONG
                else (position.entry_price - market_price) * position.qty
            )
            position.unrealized_pnl = pos_unrealized
            total_unrealized += pos_unrealized
        account_state.unrealized_pnl = total_unrealized
        account_state.equity = account_state.balance + account_state.realized_pnl + account_state.unrealized_pnl

    def _open_position(self, fill: Fill, account_state: AccountState) -> None:
        side = PositionSide.LONG if fill.side == OrderSide.BUY else PositionSide.SHORT
        position = Position(
            symbol=fill.symbol,
            side=side,
            qty=fill.qty,
            entry_price=fill.price,
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            avg_entry_fees=fill.fee / fill.qty if fill.qty else 0.0,
        )
        account_state.positions[fill.symbol] = position
        logger.debug("Opened position: %s", position)

    def _increase_position(self, position: Position, fill: Fill) -> None:
        old_qty = position.qty
        total_cost = position.entry_price * old_qty + fill.price * fill.qty
        new_qty = old_qty + fill.qty
        position.entry_price = total_cost / new_qty if new_qty else position.entry_price
        position.qty = new_qty
        total_fees = position.avg_entry_fees * old_qty + fill.fee
        position.avg_entry_fees = total_fees / new_qty if new_qty else position.avg_entry_fees
        logger.debug("Increased position: %s", position)

    def _reduce_position(self, position: Position, fill: Fill, account_state: AccountState) -> float:
        if fill.qty > position.qty:
            raise ValueError("Position flip attempted; not allowed in v1.")

        pnl = self._calculate_pnl(position, fill)
        position.qty -= fill.qty
        position.realized_pnl += pnl
        account_state.realized_pnl += pnl

        if position.qty == 0:
            self._close_position(position, account_state)
        else:
            logger.debug("Reduced position: %s | PnL=%.4f", position, pnl)
        return pnl

    def _close_position(self, position: Position, account_state: AccountState) -> None:
        logger.debug("Closed position: %s", position)
        account_state.positions.pop(position.symbol, None)

    def _calculate_pnl(self, position: Position, fill: Fill) -> float:
        if position.side == PositionSide.LONG:
            return (fill.price - position.entry_price) * fill.qty - fill.fee
        return (position.entry_price - fill.price) * fill.qty - fill.fee

    def _is_same_side(self, position_side: PositionSide, fill_side: OrderSide) -> bool:
        return (position_side == PositionSide.LONG and fill_side == OrderSide.BUY) or (
            position_side == PositionSide.SHORT and fill_side == OrderSide.SELL
        )
