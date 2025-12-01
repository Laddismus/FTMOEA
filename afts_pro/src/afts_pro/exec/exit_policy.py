from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from afts_pro.core import MarketState, StrategyDecision
from afts_pro.exec.position_models import Position, PositionSide

logger = logging.getLogger(__name__)


@dataclass
class ExitPolicyConfig:
    tighten_factor_atr: float = 0.5
    tighten_min_distance_pct: float = 0.1
    trail_factor_atr: float = 1.0
    be_offset_ticks: float = 0.0
    partial_close_fraction: float = 0.3
    allow_looser_sl: bool = False


class ExitAction:
    NONE = 0
    TIGHTEN_SL = 1
    MOVE_SL_TO_BE = 2
    TRAIL_SL = 3
    PARTIAL_CLOSE = 4
    FULL_CLOSE = 5


class ExitPolicyApplier:
    """
    Maps ExitAgent actions into StrategyDecision mutations (SL/TP/size).
    """

    def __init__(self, cfg: ExitPolicyConfig):
        self.cfg = cfg

    def apply(
        self,
        exit_action: Optional[int],
        position: Optional[Position],
        market: MarketState,
        decision: StrategyDecision,
        atr: Optional[float] = None,
    ) -> StrategyDecision:
        if exit_action is None or position is None:
            return decision

        if exit_action == ExitAction.TIGHTEN_SL:
            self._apply_tighten_sl(position, market, decision, atr)
        elif exit_action == ExitAction.MOVE_SL_TO_BE:
            self._apply_move_to_be(position, decision)
        elif exit_action == ExitAction.TRAIL_SL:
            self._apply_trail_sl(position, market, decision, atr)
        elif exit_action == ExitAction.PARTIAL_CLOSE:
            self._apply_partial_close(decision)
        elif exit_action == ExitAction.FULL_CLOSE:
            self._apply_full_close(decision)
        return decision

    def _current_sl(self, decision: StrategyDecision, position: Position) -> float:
        if "current_sl" in decision.meta:
            return float(decision.meta["current_sl"])
        # fallback: entry - 2% for long, entry +2% for short
        sign = -1 if position.side == PositionSide.LONG else 1
        return position.entry_price * (1 + sign * 0.02)

    def _apply_tighten_sl(self, position: Position, market: MarketState, decision: StrategyDecision, atr: Optional[float]) -> None:
        current_sl = self._current_sl(decision, position)
        atr_val = atr or decision.meta.get("atr", 0.0) if decision.meta else 0.0
        tighten_dist = atr_val * self.cfg.tighten_factor_atr if atr_val else abs(market.close * self.cfg.tighten_min_distance_pct)
        if position.side == PositionSide.LONG:
            new_sl = max(current_sl, market.close - tighten_dist)
            if not self.cfg.allow_looser_sl:
                new_sl = max(new_sl, current_sl)
        else:
            new_sl = min(current_sl, market.close + tighten_dist)
            if not self.cfg.allow_looser_sl:
                new_sl = min(new_sl, current_sl)
        decision.update["sl_price"] = new_sl
        decision.meta["current_sl"] = new_sl
        logger.info("RL ExitAction=tighten_sl applied (old_sl=%.4f new_sl=%.4f)", current_sl, new_sl)

    def _apply_move_to_be(self, position: Position, decision: StrategyDecision) -> None:
        offset = self.cfg.be_offset_ticks
        if position.side == PositionSide.LONG:
            new_sl = position.entry_price - offset
        else:
            new_sl = position.entry_price + offset
        decision.update["sl_price"] = new_sl
        decision.meta["current_sl"] = new_sl
        logger.info("RL ExitAction=move_sl_to_be applied (new_sl=%.4f)", new_sl)

    def _apply_trail_sl(self, position: Position, market: MarketState, decision: StrategyDecision, atr: Optional[float]) -> None:
        atr_val = atr or decision.meta.get("atr", 0.0) if decision.meta else 0.0
        trail_dist = atr_val * self.cfg.trail_factor_atr if atr_val else abs(market.close * self.cfg.tighten_min_distance_pct)
        current_sl = self._current_sl(decision, position)
        if position.side == PositionSide.LONG:
            proposed = market.close - trail_dist
            new_sl = max(current_sl, proposed) if not self.cfg.allow_looser_sl else proposed
        else:
            proposed = market.close + trail_dist
            new_sl = min(current_sl, proposed) if not self.cfg.allow_looser_sl else proposed
        decision.update["sl_price"] = new_sl
        decision.meta["current_sl"] = new_sl
        logger.info("RL ExitAction=trail_sl applied (old_sl=%.4f new_sl=%.4f)", current_sl, new_sl)

    def _apply_partial_close(self, decision: StrategyDecision) -> None:
        decision.meta["exit_partial_close_fraction"] = self.cfg.partial_close_fraction
        logger.info("RL ExitAction=partial_close applied (fraction=%.2f)", self.cfg.partial_close_fraction)

    def _apply_full_close(self, decision: StrategyDecision) -> None:
        decision.meta["exit_full_close"] = True
        logger.info("RL ExitAction=full_close applied")
