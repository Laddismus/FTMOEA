from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from afts_pro.behaviour.config import BehaviourConfig as GuardBehaviourConfig
from afts_pro.behaviour.config import create_guards_from_config, load_behaviour_config as _load_guard_config
from afts_pro.config.base_models import BaseConfigModel

logger = logging.getLogger(__name__)


class BehaviourConfig(BaseConfigModel):
    enabled: bool = True
    guards: GuardBehaviourConfig


def load_behaviour_config(path: str) -> BehaviourConfig:
    cfg = _load_guard_config(path)
    enabled = True
    if hasattr(cfg, "enabled"):
        enabled = bool(getattr(cfg, "enabled"))
    behaviour_cfg = BehaviourConfig(enabled=enabled, guards=cfg)
    logger.info("Loaded behaviour config | path=%s | enabled=%s", path, enabled)
    return behaviour_cfg


def create_guards(config: BehaviourConfig, *, initial_balance: float):
    if not config.enabled:
        logger.info("Behaviour layer disabled via config.")
        return []
    return create_guards_from_config(config.guards, initial_balance=initial_balance)
