import pandas as pd

from afts_pro.core.models import StrategyDecision, MarketState
from afts_pro.exec.order_builder import OrderBuilder
from afts_pro.exec.position_models import AccountState, Position, PositionSide


def _account(position_qty: float = 0.0):
    acc = AccountState(balance=1000.0, equity=1000.0, realized_pnl=0.0, unrealized_pnl=0.0, fees_total=0.0)
    if position_qty:
        acc.positions["ETH"] = Position(
            symbol="ETH",
            side=PositionSide.LONG,
            qty=position_qty,
            entry_price=100.0,
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            avg_entry_fees=0.0,
        )
    return acc


def _market(price: float = 100.0) -> MarketState:
    return MarketState(timestamp=pd.Timestamp.utcnow(), symbol="ETH", open=price, high=price, low=price, close=price, volume=0.0)


def test_builds_entry_order_with_given_size():
    builder = OrderBuilder(use_position_sizer=True)
    decision = StrategyDecision(action="entry", side="long", confidence=1.0, update={"position_size": 0.5})
    orders = builder.build_entry_orders(decision, _market(), _account())
    assert orders
    assert orders[0].qty == 0.5


def test_builds_modify_sl_order():
    builder = OrderBuilder()
    decision = StrategyDecision(action="manage", side="long", confidence=1.0, update={"sl_price": 95.0})
    orders = builder.build_manage_orders(decision, _market(), _account(position_qty=1.0))
    assert any(o.is_sl for o in orders)


def test_builds_partial_close_order():
    builder = OrderBuilder()
    decision = StrategyDecision(action="manage", side="long", confidence=1.0, meta={"exit_partial_close_fraction": 0.4}, update={})
    orders = builder.build_manage_orders(decision, _market(), _account(position_qty=1.0))
    assert orders
    assert orders[0].qty == 0.4
