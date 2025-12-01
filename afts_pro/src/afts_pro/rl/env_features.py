from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class FtmoFeatureConfig:
    include_daily_dd_pct: bool = True
    include_overall_dd_pct: bool = True
    include_stage: bool = True
    include_stage_one_hot: bool = True
    max_stage: int = 2
    include_rolling_dd_pct: bool = True
    include_loss_velocity: bool = True
    include_profit_progress_pct: bool = True
    include_session_one_hot: bool = True
    sessions: List[str] = field(default_factory=lambda: ["Asia", "London", "NewYork"])
    include_news_flag: bool = True
    include_time_fence_flag: bool = True
    include_spread: bool = True
    spread_clip_pips: float = 2.0
    include_stability_kpis: bool = True
    stability_pf_clip: float = 3.0
    stability_winrate_clip: float = 1.0
    stability_pnl_std_clip: float = 5.0
    include_circuit_active_flag: bool = True


@dataclass
class EnvFeatureConfig:
    base_price_features: bool = True
    base_pnl_features: bool = True
    ftmo: FtmoFeatureConfig = field(default_factory=FtmoFeatureConfig)
