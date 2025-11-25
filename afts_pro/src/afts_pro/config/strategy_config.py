from __future__ import annotations

from typing import Any, Dict, List

from afts_pro.config.base_models import BaseConfigModel


class StrategyConfig(BaseConfigModel):
    enabled_strategies: List[str]
    strategy_params: Dict[str, Dict[str, Any]]
