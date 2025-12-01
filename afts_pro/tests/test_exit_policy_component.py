import pandas as pd

from afts_pro.core.models import MarketState, StrategyDecision
from afts_pro.exec.exit_policy import ExitPolicyApplier, ExitPolicyConfig, ExitAction
from afts_pro.exec.position_models import Position, PositionSide


def _market(price: float) -> MarketState:
    return MarketState(timestamp=pd.Timestamp.utcnow(), symbol="ETH", open=price, high=price, low=price, close=price, volume=0.0)


def _position(entry: float = 100.0, side: PositionSide = PositionSide.LONG) -> Position:
    return Position(symbol="ETH", side=side, qty=1.0, entry_price=entry, realized_pnl=0.0, unrealized_pnl=0.0, avg_entry_fees=0.0)


def test_tighten_sl_only_moves_sl_closer():
    cfg = ExitPolicyConfig(tighten_factor_atr=0.5, tighten_min_distance_pct=0.05, allow_looser_sl=False)
    applier = ExitPolicyApplier(cfg)
    decision = StrategyDecision(action="manage", side="long", confidence=1.0, meta={"current_sl": 95.0})
    applier.apply(ExitAction.TIGHTEN_SL, _position(), _market(110.0), decision, atr=2.0)
    assert 95.0 <= decision.update["sl_price"] < 110.0


def test_move_sl_to_be_exact():
    cfg = ExitPolicyConfig(be_offset_ticks=0.0)
    applier = ExitPolicyApplier(cfg)
    decision = StrategyDecision(action="manage", side="long", confidence=1.0)
    applier.apply(ExitAction.MOVE_SL_TO_BE, _position(entry=100.0), _market(105.0), decision, atr=None)
    assert decision.update["sl_price"] == 100.0


def test_trail_sl_never_looser():
    cfg = ExitPolicyConfig(trail_factor_atr=1.0, allow_looser_sl=False)
    applier = ExitPolicyApplier(cfg)
    decision = StrategyDecision(action="manage", side="long", confidence=1.0, meta={"current_sl": 95.0})
    pos = _position()
    applier.apply(ExitAction.TRAIL_SL, pos, _market(110.0), decision, atr=2.0)
    first_sl = decision.update["sl_price"]
    applier.apply(ExitAction.TRAIL_SL, pos, _market(108.0), decision, atr=2.0)
    assert decision.update["sl_price"] == first_sl


def test_partial_close_calculates_fraction():
    cfg = ExitPolicyConfig(partial_close_fraction=0.25)
    applier = ExitPolicyApplier(cfg)
    decision = StrategyDecision(action="manage", side="long", confidence=1.0)
    applier.apply(ExitAction.PARTIAL_CLOSE, _position(), _market(100.0), decision, atr=None)
    assert decision.meta["exit_partial_close_fraction"] == 0.25


def test_full_close_sets_full_exit_flag():
    applier = ExitPolicyApplier(ExitPolicyConfig())
    decision = StrategyDecision(action="manage", side="long", confidence=1.0)
    applier.apply(ExitAction.FULL_CLOSE, _position(), _market(100.0), decision, atr=None)
    assert decision.meta.get("exit_full_close") is True
