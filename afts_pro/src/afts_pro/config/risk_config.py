from __future__ import annotations

from afts_pro.config.base_models import BaseConfigModel


class RiskConfig(BaseConfigModel):
    policy_type: str
    policy_path: str
