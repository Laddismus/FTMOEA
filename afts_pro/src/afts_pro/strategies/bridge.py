from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional, Sequence

from afts_pro.core import MarketState, StrategyDecision
from afts_pro.features.state import FeatureBundle
from afts_pro.strategies.base import BaseStrategy

logger = logging.getLogger(__name__)


class StrategyBridge:
    """
    Orchestrates multiple strategies and merges their decisions.
    """

    PRIORITY_ORDER = ["entry", "manage", "exit", "none"]

    def __init__(self, strategies: Sequence[BaseStrategy], asset_specs: dict | None = None) -> None:
        self._strategies = list(strategies)
        self.asset_specs = asset_specs or {}

    def on_bar(
        self,
        market_state: MarketState,
        features: Optional[FeatureBundle] = None,
    ) -> StrategyDecision:
        decisions: List[StrategyDecision] = []

        for strategy in self._strategies:
            if hasattr(strategy, "set_prior_decisions"):
                # type: ignore[attr-defined]
                strategy.set_prior_decisions(decisions)
            decision = strategy.on_bar(market_state, features=features)
            decisions.append(decision)

        merged = self._merge_decisions(decisions)
        merged.meta.setdefault("strategies", [])
        merged.meta["strategies"].extend(
            [
                {
                    "name": strategy.__class__.__name__.lower(),
                    "action": decision.action,
                    "side": decision.side,
                    "confidence": decision.confidence,
                }
                for strategy, decision in zip(self._strategies, decisions)
            ]
        )

        if logger.isEnabledFor(logging.DEBUG):
            feature_summary = None
            if features is not None:
                feature_summary = {
                    "raw_keys": list(features.raw.values.keys()),
                    "model_len": len(features.model.values) if features.model else 0,
                }
            logger.debug(
                "StrategyBridge decisions | ts=%s | decisions=%s | merged=%s | features=%s",
                market_state.timestamp.isoformat(),
                [
                    {
                        "strategy": strategy.__class__.__name__.lower(),
                        "action": decision.action,
                        "side": decision.side,
                        "confidence": decision.confidence,
                    }
                    for strategy, decision in zip(self._strategies, decisions)
                ],
                {
                    "action": merged.action,
                    "side": merged.side,
                    "confidence": merged.confidence,
                },
                feature_summary,
            )

        return merged

    def _merge_decisions(self, decisions: Iterable[StrategyDecision]) -> StrategyDecision:
        best_decision: StrategyDecision | None = None
        best_priority = len(self.PRIORITY_ORDER)

        for decision in decisions:
            priority = self._decision_priority(decision.action)

            if priority < best_priority:
                best_priority = priority
                best_decision = decision
            elif (
                best_decision is not None
                and decision.action == "entry"
                and best_decision.action == "entry"
                and decision.confidence > best_decision.confidence
            ):
                best_decision = decision

        if best_decision is None:
            return StrategyDecision(action="none", side=None, confidence=0.0)

        if best_decision.action == "none":
            return StrategyDecision(action="none", side=None, confidence=0.0, meta=dict(best_decision.meta))

        return best_decision

    def _decision_priority(self, action: str) -> int:
        try:
            return self.PRIORITY_ORDER.index(action)
        except ValueError:
            return len(self.PRIORITY_ORDER)
