import logging
import random
from typing import Iterable, List, Optional

from afts_pro.core import MarketState, StrategyDecision
from afts_pro.features.state import FeatureBundle
from afts_pro.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class DummyMLStrategy(BaseStrategy):
    """
    Simple heuristic ML placeholder that reacts to prior decisions.
    """

    def __init__(self, symbol: str) -> None:
        super().__init__(symbol=symbol)
        self._rng = random.Random()
        self._prior_decisions: List[StrategyDecision] = []

    def set_prior_decisions(self, decisions: Iterable[StrategyDecision]) -> None:
        self._prior_decisions = list(decisions)

    def on_bar(self, market_state: MarketState, features: Optional[FeatureBundle] = None) -> StrategyDecision:
        has_entry = any(decision.action == "entry" for decision in self._prior_decisions)
        action = "manage" if has_entry else "none"
        confidence = float(self._rng.uniform(0.3, 0.9))
        meta = {"model": "dummy_ml_v0"}
        side: Optional[str] = None

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "DummyML decision | ts=%s | action=%s | confidence=%.3f | meta=%s",
                market_state.timestamp.isoformat(),
                action,
                confidence,
                meta,
            )
            if features and features.model is not None:
                logger.debug(
                    "DummyML got model feature vector of length %d", len(features.model.values)
                )

        return StrategyDecision(action=action, side=side, confidence=confidence, meta=meta)
