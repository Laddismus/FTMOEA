import numpy as np
from pathlib import Path

import pandas as pd
from afts_pro.exec.position_sizer import PositionSizer, PositionSizerConfig
from afts_pro.core.models import StrategyDecision, MarketState
from afts_pro.exec.position_models import AccountState


def _account_state(equity: float = 10000.0):
    return AccountState(balance=equity, equity=equity, realized_pnl=0.0, unrealized_pnl=0.0, fees_total=0.0)


def test_risk_agent_influences_position_size():
    cfg = PositionSizerConfig()
    sizer = PositionSizer(cfg)
    state = MarketState(timestamp=pd.Timestamp.utcnow(), symbol="ETH", open=100, high=100, low=100, close=100, volume=0.0)
    decision = StrategyDecision(action="entry", side="long", confidence=1.0)
    res_high = sizer.compute_position_size(
        symbol=state.symbol,
        side=decision.side,
        entry_price=state.close,
        sl_price=99.0,
        equity=10000.0,
        agent_risk_pct=2.0,
    )
    res_low = sizer.compute_position_size(
        symbol=state.symbol,
        side=decision.side,
        entry_price=state.close,
        sl_price=99.0,
        equity=10000.0,
        agent_risk_pct=1.0,
    )
    assert res_high.size > res_low.size


def test_pipeline_without_risk_agent_uses_fixed_fallback():
    cfg = PositionSizerConfig(base_risk_mode="fixed", fixed_risk_pct=0.5)
    sizer = PositionSizer(cfg)
    state = MarketState(timestamp=pd.Timestamp.utcnow(), symbol="ETH", open=100, high=100, low=100, close=100, volume=0.0)
    res = sizer.compute_position_size(
        symbol=state.symbol,
        side="long",
        entry_price=state.close,
        sl_price=99.0,
        equity=10000.0,
        agent_risk_pct=None,
    )
    assert res.effective_risk_pct == 0.5
    assert res.size > 0
