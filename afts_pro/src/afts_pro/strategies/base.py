import abc

from typing import Optional

from afts_pro.core import MarketState, StrategyDecision
from afts_pro.features.state import FeatureBundle


class BaseStrategy(abc.ABC):
    """
    Base strategy contract: produces a StrategyDecision for each bar.
    """

    def __init__(self, symbol: str) -> None:
        self.symbol = symbol

    @abc.abstractmethod
    def on_bar(self, market_state: MarketState, features: Optional[FeatureBundle] = None) -> StrategyDecision:
        raise NotImplementedError
