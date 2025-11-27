from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass
class RLObsSpec:
    """
    Specification for observations.
    """

    shape: Tuple[int, ...]
    dtype: str = "float32"
    as_dict: bool = False


@dataclass
class ActionSpec:
    """
    Specification for actions.
    """

    action_type: str  # "discrete" | "continuous"
    num_actions: Optional[int] = None
    shape: Optional[Tuple[int, ...]] = None


@dataclass
class RewardSpec:
    """
    Specification for reward components and weights.
    """

    weight_equity_delta: float = 1.0
    weight_drawdown_delta: float = -1.0
    weight_stage_progress: float = 0.0
    weight_mfe_mae: float = 0.0


@dataclass
class RLContext:
    """
    Context for a run/episode.
    """

    run_id: str
    episode_id: str
    seed: Optional[int] = None
    meta: Dict[str, Any] | None = None
