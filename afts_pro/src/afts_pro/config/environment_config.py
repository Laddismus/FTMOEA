from __future__ import annotations

from typing import Literal, Optional

from afts_pro.config.base_models import BaseConfigModel


class EnvironmentConfig(BaseConfigModel):
    mode: Literal["sim", "train", "live"]
    timezone: str
    warmup_bars: int
    live_api_enabled: bool
    live_api_key: Optional[str] = None
    live_api_secret: Optional[str] = None
    config_hot_reload_enabled: bool = False
    config_hot_reload_interval_bars: int = 0
    config_hot_reload_scope: list[Literal["behaviour", "strategy", "execution"]] = ["behaviour", "strategy"]
