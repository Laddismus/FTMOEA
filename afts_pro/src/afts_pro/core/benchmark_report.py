from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class BenchmarkReport:
    kpis: Dict[str, float]
    ftmo: Dict[str, float | bool]
    rl_train: Dict[str, float]
    score: float
    checkpoint_path: str
    comments: List[str] = field(default_factory=list)


@dataclass
class BenchmarkComparison:
    best_checkpoint: str
    ranked: List[BenchmarkReport]
