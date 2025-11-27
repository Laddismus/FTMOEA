from __future__ import annotations

import logging
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from afts_pro.config.loader import load_yaml

logger = logging.getLogger(__name__)


class ExtraDatasetConfig(BaseModel):
    name: str
    dataset_type: str
    enabled: bool = True
    final_file_template: str
    value_columns: List[str]
    column_map: Dict[str, str] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}


class ExtrasConfig(BaseModel):
    enabled: bool = False
    base_dir: str = "data"
    final_pattern: str = "final/extras/{symbol}_{dataset}.parquet"
    stage_pattern: str = "stage/{symbol}/extras/{dataset}/{year}-{month}/*.parquet"
    timezone: str = "UTC"
    timestamp_column: str = "timestamp"
    symbol_column: str = "symbol"
    datasets: List[ExtraDatasetConfig] = Field(default_factory=list)
    symbol_overrides: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}

    def get_enabled_datasets(self) -> List[ExtraDatasetConfig]:
        return [d for d in self.datasets if d.enabled]


def load_extras_config(path: str) -> ExtrasConfig:
    data = load_yaml(path)
    extras_data = data.get("extras", data)
    cfg = ExtrasConfig(**extras_data)
    logger.info(
        "EXTRAS_CONFIG | path=%s | enabled=%s | datasets=%s",
        path,
        cfg.enabled,
        [d.name for d in cfg.get_enabled_datasets()],
    )
    return cfg
