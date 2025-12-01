from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional

logger = logging.getLogger(__name__)


@dataclass
class PositionSizerConfig:
    base_risk_mode: str = "agent"  # agent | fixed | hybrid
    fixed_risk_pct: float = 0.5
    min_risk_pct: float = 0.0
    max_risk_pct: float = 3.0
    max_risk_per_trade_pct: float = 3.0
    max_risk_per_day_pct: Optional[float] = None
    default_sl_atr_factor: float = 1.5
    cost_per_unit: Optional[float] = None
    hybrid_offset_pct: float = 0.0  # if base_risk_mode == hybrid


@dataclass
class PositionSizingResult:
    size: float
    effective_risk_pct: float
    capped_by: List[str] = field(default_factory=list)


class PositionSizer:
    """
    Converts risk_pct into position size using SL distance and risk caps.
    """

    def __init__(self, cfg: PositionSizerConfig):
        self.cfg = cfg

    def _clamp_risk_pct(self, risk_pct: float) -> tuple[float, List[str]]:
        capped = []
        if risk_pct < self.cfg.min_risk_pct:
            capped.append("min_risk_pct")
            risk_pct = self.cfg.min_risk_pct
        if risk_pct > self.cfg.max_risk_pct:
            capped.append("max_risk_pct")
            risk_pct = self.cfg.max_risk_pct
        return risk_pct, capped

    def compute_position_size(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        sl_price: Optional[float],
        equity: float,
        agent_risk_pct: Optional[float],
        daily_realized_pnl: Optional[float] = None,
        atr: Optional[float] = None,
        ftmo_stage_mult: float = 1.0,
        ftmo_stage_max_risk_pct: Optional[float] = None,
    ) -> PositionSizingResult:
        caps: List[str] = []
        # Determine risk pct
        if self.cfg.base_risk_mode == "agent":
            base_risk_pct = agent_risk_pct if agent_risk_pct is not None else self.cfg.fixed_risk_pct
        elif self.cfg.base_risk_mode == "fixed":
            base_risk_pct = self.cfg.fixed_risk_pct
        else:  # hybrid
            base_risk_pct = (agent_risk_pct or 0.0) + self.cfg.hybrid_offset_pct
        base_risk_pct, clamp_caps = self._clamp_risk_pct(base_risk_pct)
        caps.extend(clamp_caps)
        base_risk_pct *= ftmo_stage_mult
        if ftmo_stage_max_risk_pct is not None:
            base_risk_pct = min(base_risk_pct, ftmo_stage_max_risk_pct)

        # SL distance
        risk_distance = None
        if sl_price is not None:
            risk_distance = abs(entry_price - sl_price)
        elif atr is not None:
            risk_distance = atr * self.cfg.default_sl_atr_factor
            caps.append("default_sl_atr")
        else:
            caps.append("missing_sl_info")
            return PositionSizingResult(size=0.0, effective_risk_pct=0.0, capped_by=caps)

        if risk_distance <= 0:
            caps.append("invalid_sl_distance")
            return PositionSizingResult(size=0.0, effective_risk_pct=0.0, capped_by=caps)

        risk_amount = equity * (base_risk_pct / 100.0)
        max_trade_risk = equity * (self.cfg.max_risk_per_trade_pct / 100.0)
        if risk_amount > max_trade_risk:
            caps.append("max_risk_per_trade")
            risk_amount = max_trade_risk
        if self.cfg.max_risk_per_day_pct is not None and daily_realized_pnl is not None:
            # crude day cap: ensure total daily loss plus new risk not above cap
            max_day_loss = equity * (self.cfg.max_risk_per_day_pct / 100.0)
            potential_loss = -daily_realized_pnl + risk_amount
            if potential_loss > max_day_loss:
                caps.append("max_risk_per_day")
                risk_amount = max(max_day_loss - (-daily_realized_pnl), 0.0)

        contract_value_per_unit = self.cfg.cost_per_unit or 1.0
        risk_per_unit = risk_distance * contract_value_per_unit
        if risk_per_unit <= 0:
            caps.append("invalid_risk_per_unit")
            return PositionSizingResult(size=0.0, effective_risk_pct=0.0, capped_by=caps)
        size = risk_amount / risk_per_unit
        size = max(size, 0.0)
        return PositionSizingResult(size=size, effective_risk_pct=base_risk_pct if risk_amount > 0 else 0.0, capped_by=caps)
