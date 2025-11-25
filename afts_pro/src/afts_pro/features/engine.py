from __future__ import annotations

import logging
from typing import Dict, Optional

from afts_pro.config.feature_config import FeatureConfig, RawFeatureDef
from afts_pro.core import MarketState
from afts_pro.features.base_calculator import BaseFeatureCalculator
from afts_pro.features.simple_calculators import (
    ATRCalculator,
    CloseReturnCalculator,
    EMACalculator,
    RollingVolCalculator,
    RSICalculator,
    TrendScoreCalculator,
    VolatilityScoreCalculator,
)
from afts_pro.features.state import FeatureBundle, ModelFeatureVector, RawFeatureState

logger = logging.getLogger(__name__)


CALCULATOR_REGISTRY: Dict[str, type[BaseFeatureCalculator]] = {
    "close_return": CloseReturnCalculator,
    "rolling_vol": RollingVolCalculator,
    "atr": ATRCalculator,
    "ema": EMACalculator,
    "rsi": RSICalculator,
    "volatility_score": VolatilityScoreCalculator,
    "trend_score": TrendScoreCalculator,
}


class FeatureEngine:
    def __init__(self, config: FeatureConfig) -> None:
        self.config = config
        self.calculators: Dict[str, BaseFeatureCalculator] = {}
        for feature_def in config.raw_features:
            calculator_cls = CALCULATOR_REGISTRY.get(feature_def.calculator)
            if not calculator_cls:
                logger.error("Unknown calculator=%s for feature=%s", feature_def.calculator, feature_def.name)
                continue
            self.calculators[feature_def.name] = calculator_cls(feature_def.name, **feature_def.params)
        logger.info(
            "FeatureEngine initialized | raw_features=%s | model_enabled=%s",
            [f.name for f in config.raw_features],
            self.config.model_features.enabled,
        )

    def update(self, bar: MarketState) -> FeatureBundle:
        for calc in self.calculators.values():
            calc.update(bar)

        raw_values: Dict[str, float] = {}
        for name, calc in self.calculators.items():
            val = calc.current_value()
            raw_values[name] = 0.0 if val is None else float(val)

        raw_state = RawFeatureState(values=raw_values)
        model_vector: Optional[ModelFeatureVector] = None

        if self.config.model_features.enabled:
            order = self.config.model_features.get_feature_order()
            vec = raw_state.to_vector(order)
            scaling = self.config.model_features.scaling
            scaled = vec
            if scaling.type == "zscore":
                params = scaling.params.get("zscore", {})
                means = params.get("means", {})
                stds = params.get("stds", {})
                scaled = []
                for idx, fname in enumerate(order):
                    v = vec[idx]
                    mean = means.get(fname, 0.0)
                    std = stds.get(fname, 1.0)
                    if std <= 0:
                        scaled.append(0.0)
                    else:
                        scaled.append((v - mean) / std)
            elif scaling.type == "minmax":
                params = scaling.params.get("minmax", {})
                mins = params.get("mins", {})
                maxs = params.get("maxs", {})
                scaled = []
                for idx, fname in enumerate(order):
                    v = vec[idx]
                    mn = mins.get(fname, 0.0)
                    mx = maxs.get(fname, 1.0)
                    if mx <= mn:
                        scaled.append(0.0)
                    else:
                        s = (v - mn) / (mx - mn)
                        if s < 0.0:
                            s = 0.0
                        elif s > 1.0:
                            s = 1.0
                        scaled.append(s)
            elif scaling.type != "none":
                logger.error("Unknown scaling.type '%s', falling back to raw vector", scaling.type)

            model_vector = ModelFeatureVector(values=scaled)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "MODEL_FEATURES | type=%s | order=%s | first_values=%s",
                    scaling.type,
                    order,
                    scaled[:3],
                )

        return FeatureBundle(raw=raw_state, model=model_vector)
