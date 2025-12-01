from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class QAConfig:
    enable_e2e: bool = True
    enable_train_smoke: bool = True
    enable_lab_smoke: bool = True
    enable_quant_smoke: bool = True
    enable_pytest_smoke: bool = False
    pytest_args: List[str] = field(default_factory=lambda: ["-q", "tests"])
