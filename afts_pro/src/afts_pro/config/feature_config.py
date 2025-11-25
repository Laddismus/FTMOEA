from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from afts_pro.config.loader import load_yaml

logger = logging.getLogger(__name__)


class RawFeatureDef(BaseModel):
    name: str
    calculator: str
    params: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}


class ZScoreScalingParams(BaseModel):
    means: Dict[str, float] = Field(default_factory=dict)
    stds: Dict[str, float] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}


class MinMaxScalingParams(BaseModel):
    mins: Dict[str, float] = Field(default_factory=dict)
    maxs: Dict[str, float] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}


class ModelScalingConfig(BaseModel):
    type: str = "none"
    params: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}

    def as_zscore(self) -> Optional[ZScoreScalingParams]:
        if self.type != "zscore":
            return None
        params = self.params.get("zscore", {})
        return ZScoreScalingParams(**params)

    def as_minmax(self) -> Optional[MinMaxScalingParams]:
        if self.type != "minmax":
            return None
        params = self.params.get("minmax", {})
        return MinMaxScalingParams(**params)


class ModelFeaturesConfig(BaseModel):
    enabled: bool = False
    feature_order: List[str] = Field(default_factory=list)
    scaling: ModelScalingConfig = Field(default_factory=ModelScalingConfig)

    model_config = {"populate_by_name": True}

    def get_feature_order(self) -> List[str]:
        return self.feature_order or []

    def scaling_type(self) -> str:
        return self.scaling.type


class FeatureConfig(BaseModel):
    enabled: bool = True
    raw_features: List[RawFeatureDef] = Field(default_factory=list)
    model_features: ModelFeaturesConfig = Field(default_factory=ModelFeaturesConfig)

    model_config = {"populate_by_name": True}

    def get_raw_feature_names(self) -> List[str]:
        return [f.name for f in self.raw_features]

    def get_model_feature_order(self) -> List[str]:
        return list(self.model_features.feature_order)


def load_feature_config(path: str) -> FeatureConfig:
    data = load_yaml(path)
    features_data = data.get("features", data)
    cfg = FeatureConfig(**features_data)
    logger.info(
        "FEATURE_CONFIG | path=%s | raw_features=%d | model_enabled=%s",
        path,
        len(cfg.raw_features),
        cfg.model_features.enabled,
    )
    return cfg
