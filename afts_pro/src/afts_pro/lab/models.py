from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class LabExperiment:
    """
    Definition of a single LAB experiment.
    """

    id: str
    name: str
    mode: str
    base_profile: str
    params: Dict[str, Any] = field(default_factory=dict)
    seed: Optional[int] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LabResult:
    """
    Result of a LAB experiment.
    """

    experiment_id: str
    run_id: str
    run_path: str
    metrics: Dict[str, Any]
    params: Dict[str, Any]
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LabSweepDefinition:
    """
    Sweep definition for grid/random search.
    """

    id: str
    type: str
    params: Dict[str, Any]
    max_experiments: Optional[int] = None
    seed: Optional[int] = None


@dataclass
class RunResult:
    """
    Result of a backtest run used by LAB.
    """

    run_id: str
    run_path: str
    metrics: Dict[str, Any]
