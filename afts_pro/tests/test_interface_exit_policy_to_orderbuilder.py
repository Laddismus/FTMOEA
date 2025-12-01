import pandas as pd

from afts_pro.core.models import StrategyDecision, MarketState
from afts_pro.exec.exit_policy import ExitPolicyApplier, ExitPolicyConfig, ExitAction
from afts_pro.exec.order_builder import OrderBuilder
from afts_pro.exec.position_models import AccountState, Position, PositionSide


def _market() -> MarketState:
    return MarketState(timestamp=pd.Timestamp.utcnow(), symbol="ETH", open=100, high=100, low=100, close=100, volume=0.0)


def _account_with_position(qty: float = 1.0) -> AccountState:
    acc = AccountState(balance=1000, equity=1000, realized_pnl=0, unrealized_pnl=0, fees_total=0)
    acc.positions["ETH"] = Position(
        symbol="ETH",
        side=PositionSide.LONG,
        qty=qty,
        entry_price=100.0,
        realized_pnl=0.0,
        unrealized_pnl=0.0,
        avg_entry_fees=0.0,
    )
    return acc


def test_exit_action_full_close_creates_full_exit_order():
    applier = ExitPolicyApplier(ExitPolicyConfig())
    decision = StrategyDecision(action="exit", side="long", confidence=1.0, meta={"exit_action": ExitAction.FULL_CLOSE})
    applier.apply(ExitAction.FULL_CLOSE, _account_with_position().positions["ETH"], _market(), decision, atr=None)
    ob = OrderBuilder()
    orders = ob.build_exit_orders(decision, _market(), _account_with_position())
    assert orders and orders[0].qty == 1.0


def test_exit_action_partial_close_respects_fraction():
    applier = ExitPolicyApplier(ExitPolicyConfig(partial_close_fraction=0.3))
    decision = StrategyDecision(action="exit", side="long", confidence=1.0, meta={"exit_action": ExitAction.PARTIAL_CLOSE})
    applier.apply(ExitAction.PARTIAL_CLOSE, _account_with_position().positions["ETH"], _market(), decision, atr=None)
    ob = OrderBuilder()
    orders = ob.build_exit_orders(decision, _market(), _account_with_position())
    assert orders and orders[0].qty == 0.3
