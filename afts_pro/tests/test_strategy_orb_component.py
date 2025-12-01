import pandas as pd

from afts_pro.core.models import MarketState
from afts_pro.core.strategy_orb import ORBConfig, ORBStrategy, SessionConfig


def _bar(ts: str, high: float, low: float, close: float) -> MarketState:
    return MarketState(timestamp=pd.Timestamp(ts), symbol="EURUSD", open=close, high=high, low=low, close=close, volume=0.0)


def test_orb_builds_range_and_triggers_long_breakout():
    cfg = ORBConfig(range_minutes=15, min_range_pips=0.5, breakout_buffer_pips=0.1, max_entries_per_day=1, range_rr_sl_mult=1.0, range_rr_tp_mult=2.0)
    session = SessionConfig(session_start="08:00", session_end="18:00")
    orb = ORBStrategy(cfg, session)
    bar1 = _bar("2024-01-01 08:00", high=1.1010, low=1.1000, close=1.1005)
    dec1 = orb.on_bar(bar1)
    assert dec1.action == "none"
    bar2 = _bar("2024-01-01 08:16", high=1.1013, low=1.1005, close=1.1013)
    dec2 = orb.on_bar(bar2)
    assert dec2.action == "entry"
    assert dec2.side == "long"
    assert "sl_price" in dec2.update and "tp_price" in dec2.update


def test_orb_short_breakout_works_symmetrically():
    cfg = ORBConfig(range_minutes=15, min_range_pips=0.5, breakout_buffer_pips=0.1, max_entries_per_day=1, range_rr_sl_mult=1.0, range_rr_tp_mult=2.0)
    session = SessionConfig(session_start="08:00", session_end="18:00")
    orb = ORBStrategy(cfg, session)
    bar1 = _bar("2024-01-01 08:00", high=1.1010, low=1.1000, close=1.1005)
    orb.on_bar(bar1)
    bar2 = _bar("2024-01-01 08:16", high=1.1008, low=1.0990, close=1.0990)
    dec2 = orb.on_bar(bar2)
    assert dec2.action == "entry"
    assert dec2.side == "short"


def test_orb_ignores_too_small_range():
    cfg = ORBConfig(range_minutes=15, min_range_pips=5.0, breakout_buffer_pips=0.1, max_entries_per_day=1, range_rr_sl_mult=1.0, range_rr_tp_mult=2.0)
    session = SessionConfig(session_start="08:00", session_end="18:00")
    orb = ORBStrategy(cfg, session)
    bar1 = _bar("2024-01-01 08:00", high=1.1002, low=1.1000, close=1.1001)
    orb.on_bar(bar1)
    bar2 = _bar("2024-01-01 08:16", high=1.1004, low=1.1001, close=1.1003)
    dec2 = orb.on_bar(bar2)
    assert dec2.action == "none"


def test_orb_respects_max_entries_per_day():
    cfg = ORBConfig(range_minutes=15, min_range_pips=0.5, breakout_buffer_pips=0.1, max_entries_per_day=1, range_rr_sl_mult=1.0, range_rr_tp_mult=2.0)
    session = SessionConfig(session_start="08:00", session_end="18:00")
    orb = ORBStrategy(cfg, session)
    bar1 = _bar("2024-01-01 08:00", high=1.1010, low=1.1000, close=1.1005)
    orb.on_bar(bar1)
    bar2 = _bar("2024-01-01 08:16", high=1.1013, low=1.1005, close=1.1013)
    orb.on_bar(bar2)
    bar3 = _bar("2024-01-01 08:20", high=1.1020, low=1.1015, close=1.1020)
    dec3 = orb.on_bar(bar3)
    assert dec3.action == "none"
