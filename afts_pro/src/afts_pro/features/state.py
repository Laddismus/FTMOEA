from __future__ import annotations

import logging
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class RawFeatureState(BaseModel):
    values: Dict[str, float] = Field(default_factory=dict)

    model_config = {"populate_by_name": True}

    def get(self, name: str, default: Optional[float] = None) -> Optional[float]:
        return self.values.get(name, default)

    def to_vector(self, order: List[str]) -> List[float]:
        vector: List[float] = []
        for key in order:
            val = self.values.get(key)
            vector.append(0.0 if val is None else float(val))
        return vector


class ModelFeatureVector(BaseModel):
    values: List[float] = Field(default_factory=list)

    model_config = {"populate_by_name": True}

    def as_array(self) -> List[float]:
        return list(self.values)


class FeatureBundle(BaseModel):
    raw: RawFeatureState
    model: Optional[ModelFeatureVector] = None

    model_config = {"populate_by_name": True}
