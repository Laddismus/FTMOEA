from datetime import date
import logging
from typing import Optional

from typing import Optional

from afts_pro.core import MarketState, StrategyDecision
from afts_pro.features.state import FeatureBundle
from afts_pro.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class OrbStrategy(BaseStrategy):
    """
    Simple Opening Range Breakout strategy.
    """

    def __init__(self, symbol: str) -> None:
        super().__init__(symbol=symbol)
        self._current_day: Optional[date] = None
        self._orb_high: Optional[float] = None
        self._orb_low: Optional[float] = None

    def on_bar(self, market_state: MarketState, features: Optional[FeatureBundle] = None) -> StrategyDecision:
        if market_state.symbol != self.symbol:
            return StrategyDecision(action="none", side=None, confidence=0.0)

        bar_day = market_state.timestamp.date()
        if self._current_day != bar_day:
            # First bar of the day establishes the ORB range.
            self._current_day = bar_day
            self._orb_high = market_state.high
            self._orb_low = market_state.low
            logger.debug(
                "ORB_RANGE | date=%s | high=%.4f | low=%.4f",
                bar_day,
                self._orb_high,
                self._orb_low,
            )
            return StrategyDecision(action="none", side=None, confidence=0.0)

        if self._orb_high is None or self._orb_low is None:
            return StrategyDecision(action="none", side=None, confidence=0.0)

        if market_state.close > self._orb_high:
            logger.debug(
                "ORB_ENTRY | ts=%s | direction=long | close=%.4f | orb_high=%.4f | orb_low=%.4f",
                market_state.timestamp.isoformat(),
                market_state.close,
                self._orb_high,
                self._orb_low,
            )
            return StrategyDecision(action="entry", side="long", confidence=1.0)
        if market_state.close < self._orb_low:
            logger.debug(
                "ORB_ENTRY | ts=%s | direction=short | close=%.4f | orb_high=%.4f | orb_low=%.4f",
                market_state.timestamp.isoformat(),
                market_state.close,
                self._orb_high,
                self._orb_low,
            )
            return StrategyDecision(action="entry", side="short", confidence=1.0)

        return StrategyDecision(action="none", side=None, confidence=0.0)
