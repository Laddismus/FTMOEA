import pandas as pd
from pathlib import Path

from afts_pro.core import StrategyDecision
from afts_pro.core.models import MarketState
from afts_pro.exec.exit_policy import ExitPolicyApplier, ExitPolicyConfig, ExitAction
from afts_pro.exec.position_models import Position, PositionSide, AccountState


def _market(ts_price: float = 100.0) -> MarketState:
    return MarketState(timestamp=pd.Timestamp.utcnow(), symbol="ETH", open=ts_price, high=ts_price, low=ts_price, close=ts_price, volume=0.0)


def _position(entry: float = 100.0, sl: float = 95.0, side: PositionSide = PositionSide.LONG) -> Position:
    pos = Position(symbol="ETH", side=side, qty=1.0, entry_price=entry, realized_pnl=0.0, unrealized_pnl=0.0, avg_entry_fees=0.0)
    return pos


def test_tighten_sl_never_looser():
    cfg = ExitPolicyConfig(allow_looser_sl=False, tighten_factor_atr=0.5, tighten_min_distance_pct=0.05)
    applier = ExitPolicyApplier(cfg)
    decision = StrategyDecision(action="manage", side="long", confidence=1.0)
    decision.meta["current_sl"] = 95.0
    pos = _position()
    market = _market(100.0)
    applier.apply(ExitAction.TIGHTEN_SL, pos, market, decision, atr=2.0)
    assert decision.update["sl_price"] >= 95.0
    assert decision.update["sl_price"] <= market.close


def test_move_sl_to_be_sets_entry_price():
    cfg = ExitPolicyConfig(be_offset_ticks=0.0)
    applier = ExitPolicyApplier(cfg)
    decision = StrategyDecision(action="manage", side="long", confidence=1.0)
    pos = _position(entry=100.0, sl=95.0)
    market = _market(105.0)
    applier.apply(ExitAction.MOVE_SL_TO_BE, pos, market, decision, atr=None)
    assert decision.update["sl_price"] == 100.0


def test_trail_sl_follows_price_up_only():
    cfg = ExitPolicyConfig(trail_factor_atr=1.0, allow_looser_sl=False)
    applier = ExitPolicyApplier(cfg)
    decision = StrategyDecision(action="manage", side="long", confidence=1.0)
    decision.meta["current_sl"] = 95.0
    pos = _position()
    market_up = _market(110.0)
    applier.apply(ExitAction.TRAIL_SL, pos, market_up, decision, atr=2.0)
    first_sl = decision.update["sl_price"]
    market_down = _market(108.0)
    decision.meta["current_sl"] = first_sl
    applier.apply(ExitAction.TRAIL_SL, pos, market_down, decision, atr=2.0)
    assert decision.update["sl_price"] == first_sl


def test_partial_close_fraction_applied():
    cfg = ExitPolicyConfig(partial_close_fraction=0.3)
    applier = ExitPolicyApplier(cfg)
    decision = StrategyDecision(action="manage", side="long", confidence=1.0)
    pos = _position()
    market = _market(100.0)
    applier.apply(ExitAction.PARTIAL_CLOSE, pos, market, decision, atr=None)
    assert decision.meta.get("exit_partial_close_fraction") == 0.3


def test_full_close_closes_entire_position():
    cfg = ExitPolicyConfig()
    applier = ExitPolicyApplier(cfg)
    decision = StrategyDecision(action="manage", side="long", confidence=1.0)
    pos = _position()
    market = _market(100.0)
    applier.apply(ExitAction.FULL_CLOSE, pos, market, decision, atr=None)
    assert decision.meta.get("exit_full_close") is True


def test_no_position_no_exit():
    cfg = ExitPolicyConfig()
    applier = ExitPolicyApplier(cfg)
    decision = StrategyDecision(action="manage", side="long", confidence=1.0)
    market = _market(100.0)
    applier.apply(ExitAction.FULL_CLOSE, None, market, decision, atr=None)
    assert decision.meta == {}
