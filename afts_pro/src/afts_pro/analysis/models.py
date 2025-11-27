from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class RollingKpiResult:
    df: pd.DataFrame
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MonteCarloResult:
    summary: Dict[str, Any]
    distribution: np.ndarray


@dataclass
class DriftResult:
    drift_points: List[pd.Timestamp]
    segments: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class RegimeResult:
    regimes: pd.Series
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QuantConfig:
    rolling: Dict[str, Any]
    monte_carlo: Dict[str, Any]
    drift: Dict[str, Any]
    regimes: Dict[str, Any]
    output: Dict[str, Any]
