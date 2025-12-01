from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, time
from typing import Dict, Optional

from afts_pro.core.models import StrategyDecision
from afts_pro.core import MarketState
from afts_pro.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


@dataclass
class ORBConfig:
    range_minutes: int
    min_range_pips: float
    breakout_buffer_pips: float
    max_entries_per_day: int
    allow_long: bool = True
    allow_short: bool = True
    atr_sl_mult: float = 1.0
    atr_tp_mult: float = 2.0
    range_rr_sl_mult: float = 1.0
    range_rr_tp_mult: float = 2.0


@dataclass
class SessionConfig:
    session_start: str
    session_end: str

    def contains(self, ts: datetime) -> bool:
        t = ts.time()
        start = time.fromisoformat(self.session_start)
        end = time.fromisoformat(self.session_end)
        return start <= t <= end


@dataclass
class ORBState:
    day: str
    range_high: Optional[float] = None
    range_low: Optional[float] = None
    range_finalized: bool = False
    entries_today: int = 0


class ORBStrategy(BaseStrategy):
    """
    Simple ORB strategy operating on MarketState bars.
    """

    def __init__(self, config: ORBConfig, session: SessionConfig, symbol: str = "ORB"):
        super().__init__(symbol=symbol)
        self.cfg = config
        self.session = session
        self.state_by_symbol: Dict[str, ORBState] = {}

    def _reset_state_if_new_day(self, symbol: str, ts: datetime) -> ORBState:
        day_str = ts.date().isoformat()
        state = self.state_by_symbol.get(symbol)
        if state is None or state.day != day_str:
            state = ORBState(day=day_str)
            self.state_by_symbol[symbol] = state
        return state

    def on_bar(self, bar: MarketState, features: Optional[Any] = None, atr: Optional[float] = None) -> StrategyDecision:
        state = self._reset_state_if_new_day(bar.symbol, bar.timestamp)
        if not self.session.contains(bar.timestamp):
            return StrategyDecision(action="none", side=None, confidence=0.0)

        minutes_since_session_start = (
            bar.timestamp.hour * 60 + bar.timestamp.minute
            - time.fromisoformat(self.session.session_start).hour * 60
            - time.fromisoformat(self.session.session_start).minute
        )

        if minutes_since_session_start < self.cfg.range_minutes:
            # building range
            state.range_high = max(state.range_high or bar.high, bar.high)
            state.range_low = min(state.range_low or bar.low, bar.low)
            return StrategyDecision(action="none", side=None, confidence=0.0)

        if state.range_high is None or state.range_low is None:
            return StrategyDecision(action="none", side=None, confidence=0.0)

        state.range_finalized = True
        range_pips = (state.range_high - state.range_low) * 10000
        if range_pips < self.cfg.min_range_pips:
            return StrategyDecision(action="none", side=None, confidence=0.0)

        long_level = state.range_high + self.cfg.breakout_buffer_pips / 10000
        short_level = state.range_low - self.cfg.breakout_buffer_pips / 10000

        # Entry signals
        if bar.high >= long_level and self.cfg.allow_long and state.entries_today < self.cfg.max_entries_per_day:
            state.entries_today += 1
            sl = bar.close - (self.cfg.range_rr_sl_mult * (state.range_high - state.range_low))
            tp = bar.close + (self.cfg.range_rr_tp_mult * (state.range_high - state.range_low))
            decision = StrategyDecision(
                action="entry",
                side="long",
                confidence=1.0,
                update={"sl_price": sl, "tp_price": tp},
                meta={"orb_range": (state.range_low, state.range_high), "orb_breakout": "long"},
            )
            return decision
        if bar.low <= short_level and self.cfg.allow_short and state.entries_today < self.cfg.max_entries_per_day:
            state.entries_today += 1
            sl = bar.close + (self.cfg.range_rr_sl_mult * (state.range_high - state.range_low))
            tp = bar.close - (self.cfg.range_rr_tp_mult * (state.range_high - state.range_low))
            decision = StrategyDecision(
                action="entry",
                side="short",
                confidence=1.0,
                update={"sl_price": sl, "tp_price": tp},
                meta={"orb_range": (state.range_low, state.range_high), "orb_breakout": "short"},
            )
            return decision

        return StrategyDecision(action="none", side=None, confidence=0.0)
