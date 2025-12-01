from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from afts_pro.core.strategy_orb import ORBConfig, SessionConfig


@dataclass
class StrategyProfileConfig:
    name: str
    symbol: str
    timeframe: str
    session: SessionConfig
    orb: Optional[ORBConfig] = None


def load_strategy_profile(path: str) -> StrategyProfileConfig:
    data = yaml.safe_load(Path(path).read_text())
    session_cfg = data.get("session", {})
    session = SessionConfig(session_start=session_cfg.get("session_start", "00:00"), session_end=session_cfg.get("session_end", "23:59"))
    orb_cfg = data.get("orb")
    orb = None
    if orb_cfg:
        orb = ORBConfig(
            range_minutes=int(orb_cfg.get("range_minutes", 15)),
            min_range_pips=float(orb_cfg.get("min_range_pips", 1.0)),
            breakout_buffer_pips=float(orb_cfg.get("breakout_buffer_pips", 0.0)),
            max_entries_per_day=int(orb_cfg.get("max_entries_per_day", 1)),
            allow_long=bool(data.get("filters", {}).get("allow_long", True)),
            allow_short=bool(data.get("filters", {}).get("allow_short", True)),
            atr_sl_mult=float(data.get("sl_tp", {}).get("atr_sl_mult", 1.0)),
            atr_tp_mult=float(data.get("sl_tp", {}).get("atr_tp_mult", 2.0)),
            range_rr_sl_mult=float(data.get("sl_tp", {}).get("range_rr_sl_mult", 1.0)),
            range_rr_tp_mult=float(data.get("sl_tp", {}).get("range_rr_tp_mult", 2.0)),
        )
    return StrategyProfileConfig(
        name=data.get("name", Path(path).stem),
        symbol=data.get("symbol", ""),
        timeframe=data.get("timeframe", ""),
        session=session,
        orb=orb,
    )
