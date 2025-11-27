from __future__ import annotations

import logging
from typing import Dict, Optional

from afts_pro.config.feature_config import FeatureConfig
from afts_pro.core import MarketState
from afts_pro.data.extras_loader import ExtrasSeries
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
from afts_pro.features.state import ExtrasSnapshot, FeatureBundle, ModelFeatureVector, RawFeatureState

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
        self._extras: Dict[str, ExtrasSeries] = {}
        self._extras_cursors: Dict[str, int] = {}
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

    def attach_extras(self, extras_by_dataset: Dict[str, ExtrasSeries]) -> None:
        self._extras = extras_by_dataset or {}
        self._extras_cursors = {name: 0 for name in self._extras.keys()}
        if self._extras:
            logger.info("FeatureEngine extras attached | datasets=%s", list(self._extras.keys()))
        else:
            logger.info("FeatureEngine extras attached | no datasets")

    def _snapshot_extras(self, bar: MarketState) -> Optional[ExtrasSnapshot]:
        if not self._extras:
            return None
        snapshot: Dict[str, Dict[str, float]] = {}
        for ds_name, series in self._extras.items():
            df = series.df
            ts_column = df.columns[0]
            cursor = self._extras_cursors.get(ds_name, 0)
            while cursor < len(df) and pd.to_datetime(df.iloc[cursor][ts_column], utc=True) <= bar.timestamp:
                cursor += 1
            if cursor == 0:
                self._extras_cursors[ds_name] = cursor
                continue
            use_idx = cursor - 1
            self._extras_cursors[ds_name] = cursor
            row = df.iloc[use_idx]
            values: Dict[str, float] = {}
            for col in df.columns:
                if col == ts_column:
                    continue
                try:
                    values[col] = float(row[col])
                except Exception:
                    continue
            if values:
                snapshot[ds_name] = values

        if not snapshot:
            return None
        return ExtrasSnapshot(values=snapshot)

    def update(self, bar: MarketState) -> FeatureBundle:
        extras_snapshot = self._snapshot_extras(bar)

        for calc in self.calculators.values():
            calc.update(bar, extras=extras_snapshot)

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

        return FeatureBundle(raw=raw_state, model=model_vector, extras=extras_snapshot)
