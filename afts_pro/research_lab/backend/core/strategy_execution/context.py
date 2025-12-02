"""Simple execution context implementation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict


@dataclass
class SimpleExecutionContext:
    config: Dict[str, Any]
    params: Dict[str, Any]


__all__ = ["SimpleExecutionContext"]
