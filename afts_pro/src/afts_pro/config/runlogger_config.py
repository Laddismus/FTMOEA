from __future__ import annotations

import logging
from typing import Dict, Optional

from pydantic import BaseModel, Field

from afts_pro.config.loader import load_yaml

logger = logging.getLogger(__name__)


class RunLoggerConfig(BaseModel):
    enabled: bool = Field(default=True)
    base_dir: str = Field(default="runs")
    filename_patterns: Dict[str, str] = Field(
        default_factory=lambda: {
            "config_snapshot": "config_used.yaml",
            "log_capture": "logs.txt",
            "trades": "trades.parquet",
            "equity_curve": "equity_curve.parquet",
            "positions": "positions.parquet",
            "metrics": "metrics.json",
        }
    )
    retention: Dict[str, Optional[int]] = Field(default_factory=lambda: {"keep_last_n_runs": None})
    include: Dict[str, bool] = Field(
        default_factory=lambda: {
            "config_snapshot": True,
            "trades": True,
            "equity_curve": True,
            "positions": True,
            "metrics": True,
        }
    )

    model_config = {"populate_by_name": True}


def load_runlogger_config(path: str) -> RunLoggerConfig:
    data = load_yaml(path)
    cfg = data.get("runlogger", data)
    config = RunLoggerConfig(**cfg)
    logger.info("RUNLOGGER_CONFIG | path=%s | enabled=%s | base_dir=%s", path, config.enabled, config.base_dir)
    return config
