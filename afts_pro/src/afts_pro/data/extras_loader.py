from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import pandas as pd
from pydantic import BaseModel

from afts_pro.config.extras_config import ExtraDatasetConfig, ExtrasConfig

logger = logging.getLogger(__name__)


class ExtrasSeries(BaseModel):
    symbol: str
    dataset: str
    df: pd.DataFrame

    class Config:
        arbitrary_types_allowed = True


def _build_final_path(config: ExtrasConfig, symbol: str, dataset_cfg: ExtraDatasetConfig) -> Path:
    base_dir = Path(config.base_dir)
    # Prefer dataset template if provided.
    rel = dataset_cfg.final_file_template.format(symbol=symbol, dataset=dataset_cfg.name)
    path_rel = Path(config.final_pattern.format(symbol=symbol, dataset=dataset_cfg.name))
    if dataset_cfg.final_file_template:
        path_rel = Path("final/extras") / rel
    return base_dir / path_rel


class ExtrasLoader:
    def __init__(self, config: ExtrasConfig) -> None:
        self.config = config

    def load_for_symbol(self, symbol: str) -> Dict[str, ExtrasSeries]:
        result: Dict[str, ExtrasSeries] = {}
        if not self.config.enabled:
            logger.info("ExtrasLoader disabled, returning empty extras for %s", symbol)
            return result

        for ds_cfg in self.config.get_enabled_datasets():
            path = _build_final_path(self.config, symbol, ds_cfg)
            if not path.exists():
                logger.warning("Extras file missing for symbol=%s dataset=%s path=%s", symbol, ds_cfg.name, path)
                continue

            try:
                df = pd.read_parquet(path)
            except Exception as e:  # pragma: no cover - IO path
                logger.error("Failed to load extras for %s/%s from %s: %s", symbol, ds_cfg.name, path, e)
                continue

            if ds_cfg.column_map:
                df = df.rename(columns=ds_cfg.column_map)

            ts_col = self.config.timestamp_column
            if ts_col not in df.columns:
                if "time" in df.columns:
                    df = df.rename(columns={"time": ts_col})
                else:
                    logger.error("No timestamp column found in extras %s (expected '%s')", path, ts_col)
                    continue

            df = df.sort_values(ts_col).reset_index(drop=True)

            series = ExtrasSeries(symbol=symbol, dataset=ds_cfg.name, df=df)
            result[ds_cfg.name] = series

        return result

    def has_dataset(self, symbol: str, dataset_name: str) -> bool:
        ds_cfgs = {d.name for d in self.config.get_enabled_datasets()}
        if dataset_name not in ds_cfgs:
            return False
        path = _build_final_path(self.config, symbol, next(d for d in self.config.get_enabled_datasets() if d.name == dataset_name))
        return path.exists()
