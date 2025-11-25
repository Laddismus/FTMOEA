from __future__ import annotations

from typing import Dict

from afts_pro.config.base_models import BaseConfigModel


class AssetSpec(BaseConfigModel):
    symbol: str
    base_asset: str
    quote_asset: str
    min_qty: float
    max_qty: float
    qty_step: float
    price_step: float


class AssetConfig(BaseConfigModel):
    assets: Dict[str, AssetSpec]
