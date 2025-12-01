from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import List, Sequence

from afts_pro.core import MarketState, StrategyDecision
from afts_pro.config.asset_config import AssetSpec
from afts_pro.exec.order_models import (
    Order,
    OrderSide,
    OrderStatus,
    OrderType,
    TimeInForce,
)
from afts_pro.exec.position_models import AccountState

logger = logging.getLogger(__name__)


class OrderBuilder:
    """
    Translates StrategyDecisions into executable Orders.
    """

    def __init__(self, asset_specs: dict[str, AssetSpec] | None = None, use_position_sizer: bool = False) -> None:
        self.asset_specs = asset_specs or {}
        self.use_position_sizer = use_position_sizer

    def build_entry_orders(
        self,
        decision: StrategyDecision,
        market_state: MarketState,
        account_state: AccountState,
    ) -> List[Order]:
        if decision.action != "entry" or decision.side is None:
            return []

        side = OrderSide.BUY if decision.side == "long" else OrderSide.SELL
        qty_override = None
        if self.use_position_sizer and decision.update.get("position_size") is not None:
            qty_override = float(decision.update.get("position_size"))
        order = self._build_order(
            symbol=market_state.symbol,
            side=side,
            order_type=OrderType.MARKET,
            qty=qty_override if qty_override is not None else self._default_qty(account_state, market_state.symbol),
            price=market_state.close,
            reduce_only=False,
        )
        logger.debug("Entry order built: %s", order)
        return [order]

    def build_manage_orders(
        self,
        decision: StrategyDecision,
        market_state: MarketState,
        account_state: AccountState,
    ) -> List[Order]:
        if decision.action != "manage":
            return []

        orders: List[Order] = []
        updates = decision.update or {}

        if "new_sl" in updates or "sl_price" in updates:
            price_val = float(updates.get("new_sl", updates.get("sl_price")))
            orders.append(
                self._build_sl_order(
                    symbol=market_state.symbol,
                    side=decision.side or "long",
                    price=price_val,
                )
            )
        if "new_tp" in updates:
            orders.append(
                self._build_tp_order(
                    symbol=market_state.symbol,
                    side=decision.side or "long",
                    price=float(updates["new_tp"]),
                )
            )
        if updates.get("trail_sl_to") == "BE":
            be_price = self._get_entry_price(account_state, market_state.symbol)
            if be_price is not None:
                orders.append(
                    self._build_sl_order(
                        symbol=market_state.symbol,
                        side=decision.side or "long",
                        price=be_price,
                    )
                )
        if "trail_sl_pct" in updates:
            pct = float(updates["trail_sl_pct"])
            trail_price = self._get_entry_price(account_state, market_state.symbol)
            if trail_price is not None:
                trail_price = trail_price * (1 + pct) if (decision.side or "long") == "long" else trail_price * (1 - pct)
                orders.append(
                    self._build_sl_order(
                        symbol=market_state.symbol,
                        side=decision.side or "long",
                        price=trail_price,
                    )
                )
        if "rr_target" in updates:
            logger.debug("rr_target provided but not yet implemented: %s", updates["rr_target"])
        if "close_pct" in updates:
            pct = float(updates["close_pct"])
            orders.append(
                self._build_partial_close_order(
                    symbol=market_state.symbol,
                    side=decision.side or "long",
                    pct=pct,
                    account_state=account_state,
                )
            )
        meta_partial = decision.meta.get("exit_partial_close_fraction") if decision.meta else None
        if meta_partial is not None:
            orders.append(
                self._build_partial_close_order(
                    symbol=market_state.symbol,
                    side=decision.side or "long",
                    pct=float(meta_partial),
                    account_state=account_state,
                )
            )

        if orders:
            logger.debug("Manage orders built: %s", orders)
        return orders

    def build_exit_orders(
        self,
        decision: StrategyDecision,
        market_state: MarketState,
        account_state: AccountState,
    ) -> List[Order]:
        if decision.action != "exit":
            return []

        orders: List[Order] = []
        pct_meta = decision.meta.get("exit_partial_close_fraction") if decision.meta else None
        pct = float(pct_meta) if pct_meta is not None else 1.0
        orders.append(
            self._build_partial_close_order(
                symbol=market_state.symbol,
                side=decision.side or "long",
                pct=pct,
                account_state=account_state,
            )
        )
        logger.debug("Exit orders built: %s", orders)
        return orders

    def _build_sl_order(self, symbol: str, side: str, price: float) -> Order:
        exit_side = OrderSide.SELL if side == "long" else OrderSide.BUY
        return self._build_order(
            symbol=symbol,
            side=exit_side,
            order_type=OrderType.STOP_MARKET,
            qty=0.0,  # Placeholder until sizing logic is added.
            stop_price=price,
            reduce_only=True,
            is_sl=True,
        )

    def _build_tp_order(self, symbol: str, side: str, price: float) -> Order:
        exit_side = OrderSide.SELL if side == "long" else OrderSide.BUY
        return self._build_order(
            symbol=symbol,
            side=exit_side,
            order_type=OrderType.LIMIT,
            qty=0.0,
            price=price,
            reduce_only=True,
            is_tp=True,
        )

    def _build_partial_close_order(
        self,
        symbol: str,
        side: str,
        pct: float,
        account_state: AccountState,
    ) -> Order:
        position = account_state.positions.get(symbol)
        qty = (position.qty * pct) if position else 0.0
        exit_side = OrderSide.SELL if side == "long" else OrderSide.BUY
        return self._build_order(
            symbol=symbol,
            side=exit_side,
            order_type=OrderType.MARKET,
            qty=qty,
            reduce_only=True,
        )

    def _build_order(
        self,
        symbol: str,
        side: OrderSide,
        order_type: OrderType,
        qty: float,
        price: float | None = None,
        stop_price: float | None = None,
        reduce_only: bool = False,
        is_sl: bool = False,
        is_tp: bool = False,
    ) -> Order:
        now = datetime.now(timezone.utc)
        return Order(
            id=str(uuid.uuid4()),
            client_order_id=None,
            symbol=symbol,
            side=side,
            type=order_type,
            qty=qty,
            price=price,
            stop_price=stop_price,
            reduce_only=reduce_only,
            is_sl=is_sl,
            is_tp=is_tp,
            time_in_force=TimeInForce.GTC,
            status=OrderStatus.NEW,
            created_at=now,
            updated_at=now,
        )

    def _default_qty(self, account_state: AccountState, symbol: str) -> float:
        # Placeholder sizing leveraging asset config if available.
        # Defaults to a small fixed size when no spec is present.
        # For now ignore account_state sizing logic.
        spec = self.asset_specs.get(symbol)
        if spec:
            return max(spec.min_qty, 0.01)
        return 0.01

    def _get_entry_price(self, account_state: AccountState, symbol: str) -> float | None:
        position = account_state.positions.get(symbol)
        return position.entry_price if position else None
