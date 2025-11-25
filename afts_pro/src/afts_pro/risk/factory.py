from __future__ import annotations

import logging
from typing import Any, Dict

import yaml

from afts_pro.risk.apex_policy import ApexRiskPolicy
from afts_pro.risk.base_policy import BaseRiskPolicy
from afts_pro.risk.equity_policy import EquityMaxDdPolicy
from afts_pro.risk.ftmo_policy import FtmoRiskPolicy

logger = logging.getLogger(__name__)


def load_risk_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as fp:
        return yaml.safe_load(fp) or {}


def create_risk_policy_from_config(config_path: str) -> BaseRiskPolicy:
    config = load_risk_config(config_path)
    cfg_type = (config.get("type") or "").lower()

    if cfg_type == "ftmo":
        return FtmoRiskPolicy(
            initial_balance=float(config["initial_balance"]),
            total_dd_hard_stop_pct=float(config.get("total_dd_hard_stop_pct", 0.085)),
            daily_soft_dd_pct=float(config.get("daily_soft_dd_pct", 0.035)),
            daily_hard_dd_pct=float(config.get("daily_hard_dd_pct", 0.04)),
            include_unrealized=bool(config.get("include_unrealized", True)),
        )
    if cfg_type == "apex":
        return ApexRiskPolicy(
            initial_balance=float(config["initial_balance"]),
            trailing_dd_pct=float(config.get("trailing_dd_pct", 0.04)),
            include_unrealized=bool(config.get("include_unrealized", True)),
        )
    if cfg_type == "equity":
        return EquityMaxDdPolicy(
            initial_balance=float(config["initial_balance"]),
            max_dd_pct=float(config["max_dd_pct"]),
            include_unrealized=bool(config.get("include_unrealized", True)),
            use_hwm=bool(config.get("use_hwm", True)),
            equity_basis=str(config.get("equity_basis", "full")),
        )

    raise ValueError(f"Unsupported risk policy type: {cfg_type}")
