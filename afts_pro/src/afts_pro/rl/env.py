from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import yaml

from afts_pro.exec.position_models import AccountState
from afts_pro.features.state import FeatureBundle
from afts_pro.rl.types import ActionSpec, RewardSpec, RLContext, RLObsSpec
from afts_pro.rl.reward import RewardCalculator, RewardConfig, RewardContext

logger = logging.getLogger(__name__)


RLObservation = Any


@dataclass
class RLStepResult:
    observation: RLObservation
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any]


class RLBaseEnv(ABC):
    """
    Abstract gym-like environment definition.
    """

    action_spec: ActionSpec
    obs_spec: RLObsSpec

    @abstractmethod
    def reset(self, seed: int | None = None, options: Dict | None = None) -> Tuple[RLObservation, Dict[str, Any]]:
        ...

    @abstractmethod
    def step(self, action: Any) -> Tuple[RLObservation, float, bool, bool, Dict[str, Any]]:
        ...

    def close(self) -> None:  # pragma: no cover - stub
        logger.debug("Env close called (no-op).")

    def render(self) -> None:  # pragma: no cover - stub
        logger.debug("Env render called (no-op).")


def load_env_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _safe_div(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return num / den


class RewardCalculator:
    def __init__(self, reward_spec: RewardSpec, equity_norm: float = 1.0) -> None:
        self.spec = reward_spec
        self.equity_norm = equity_norm or 1.0
        self.prev_equity: Optional[float] = None
        self.prev_dd: Optional[float] = None

    def compute(self, ctx: RewardContext) -> float:
        eq_delta = 0.0
        dd_delta = 0.0
        if self.prev_equity is not None:
            eq_delta = _safe_div(ctx.equity_t - self.prev_equity, self.equity_norm)
        if self.prev_dd is not None:
            dd_delta = _safe_div(ctx.dd_t - self.prev_dd, self.equity_norm)

        self.prev_equity = ctx.equity_t
        self.prev_dd = ctx.dd_t

        reward = (
            self.spec.weight_equity_delta * eq_delta
            + self.spec.weight_drawdown_delta * dd_delta
            + self.spec.weight_stage_progress * ctx.stage_progress
            + self.spec.weight_mfe_usage * (ctx.mfe_t - abs(ctx.mae_t))
        )
        return float(reward)


class RLTradingEnv(RLBaseEnv):
    """
    Minimal gym-like environment for AFTS-PRO RL.
    Uses a provided event stream and optional hooks to map actions to core.
    """

    def __init__(
        self,
        config: Dict[str, Any],
        event_stream: Iterable[Dict[str, Any]],
        action_spec: Optional[ActionSpec] = None,
        obs_spec: Optional[RLObsSpec] = None,
        apply_action_to_pipeline: Optional[Callable[[Any, Dict[str, Any]], None]] = None,
    ) -> None:
        self.config = config
        self.event_stream_source = list(event_stream)
        self._cursor = 0
        self._rng = np.random.default_rng()
        self.apply_action_to_pipeline = apply_action_to_pipeline
        self.action_spec = action_spec or ActionSpec(action_type="discrete", num_actions=3)
        obs_len = self._infer_obs_length(config)
        self.obs_spec = obs_spec or RLObsSpec(shape=(obs_len,), dtype="float32", as_dict=False)
        reward_cfg = config.get("reward", {})
        reward_profile_name = config.get("env_type", "risk")
        profile_cfg = config.get("reward_profiles", {}).get(reward_profile_name, reward_cfg)
        reward_config = RewardConfig(
            weight_equity_delta=profile_cfg.get("weight_equity_delta", 1.0),
            weight_drawdown_delta=profile_cfg.get("weight_drawdown_delta", -1.0),
            weight_stage_progress=profile_cfg.get("weight_stage_progress", 0.0),
            weight_mfe_usage=profile_cfg.get("weight_mfe_usage", 0.0),
            weight_mae_penalty=profile_cfg.get("weight_mae_penalty", 0.0),
            weight_time_under_water=profile_cfg.get("weight_time_under_water", 0.0),
            clip_min=profile_cfg.get("clip_min"),
            clip_max=profile_cfg.get("clip_max"),
        )
        self._reward_calc = RewardCalculator(reward_config, equity_norm=1.0)
        self._start_equity = 1.0
        self._max_steps = config.get("episode", {}).get("max_steps", 0)
        self._episode_mode = config.get("episode", {}).get("mode", "fixed_bars")
        self._step_count = 0
        self._terminated = False
        self._truncated = False
        self._prev_event: Dict[str, Any] | None = None

    def _infer_obs_length(self, cfg: Dict[str, Any]) -> int:
        obs_cfg = cfg.get("observation", {})
        include_position = obs_cfg.get("include_position_state", True)
        include_risk = obs_cfg.get("include_risk_state", True)
        include_features = obs_cfg.get("include_features", True)
        length = 0
        if include_features:
            length += len(obs_cfg.get("feature_names", [])) or 4
        if include_position:
            length += 3
        if include_risk:
            length += 3
        return length

    def reset(self, seed: int | None = None, options: Dict | None = None) -> Tuple[RLObservation, Dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._cursor = 0
        self._step_count = 0
        self._terminated = False
        self._truncated = False
        first_event = self._get_current_event()
        self._start_equity = float(first_event.get("equity", 1.0)) or 1.0
        obs = self._build_observation(first_event)
        self._prev_event = first_event
        return obs, {"reset": True}

    def _get_current_event(self) -> Dict[str, Any]:
        if self._cursor >= len(self.event_stream_source):
            return {}
        return self.event_stream_source[self._cursor]

    def _advance_event(self) -> Dict[str, Any]:
        self._cursor += 1
        return self._get_current_event()

    def step(self, action: Any) -> Tuple[RLObservation, float, bool, bool, Dict[str, Any]]:
        if self._terminated or self._truncated:
            raise RuntimeError("Environment already done; call reset().")
        if self.apply_action_to_pipeline:
            self.apply_action_to_pipeline(action, {"step": self._step_count})
        prev_event = self._get_current_event()
        current_event = self._advance_event()
        self._step_count += 1
        if not current_event:
            self._terminated = True
            current_event = prev_event
        obs = self._build_observation(current_event)
        reward = self._compute_reward(prev_event, current_event)
        terminated = self._terminated
        truncated = False
        if self._max_steps and self._step_count >= self._max_steps:
            truncated = True
            self._truncated = True
        info: Dict[str, Any] = {"step": self._step_count}
        return obs, reward, terminated, truncated, info

    def _build_observation(self, event: Dict[str, Any]) -> np.ndarray:
        obs_cfg = self.config.get("observation", {})
        vector: List[float] = []
        if obs_cfg.get("include_features", True):
            features = event.get("features") or []
            vector.extend(features)
        if obs_cfg.get("include_position_state", True):
            position = event.get("position_state") or {}
            vector.append(float(position.get("side", 0.0)))
            vector.append(float(position.get("size_norm", 0.0)))
            vector.append(float(position.get("unrealized_norm", 0.0)))
        if obs_cfg.get("include_risk_state", True):
            equity = float(event.get("equity", 0.0))
            dd = float(event.get("drawdown", 0.0))
            dd_remaining = float(event.get("dd_remaining", 0.0))
            norm_mode = obs_cfg.get("equity_norm_mode", "start_equity")
            denom = self._start_equity if norm_mode == "start_equity" else 1.0
            denom = denom or 1.0
            vector.append(equity / denom)
            vector.append(dd / denom)
            vector.append(dd_remaining / denom if denom else dd_remaining)
        arr = np.array(vector, dtype=np.float32)
        if np.any(~np.isfinite(arr)):
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr

    def _compute_reward(self, prev_event: Dict[str, Any], current_event: Dict[str, Any]) -> float:
        ctx = RewardContext(
            equity_t=float(current_event.get("equity", 0.0)),
            equity_prev=float(prev_event.get("equity", 0.0)) if prev_event else float(current_event.get("equity", 0.0)),
            dd_t=float(current_event.get("drawdown", 0.0)),
            dd_prev=float(prev_event.get("drawdown", 0.0)) if prev_event else float(current_event.get("drawdown", 0.0)),
            mfe_t=float(current_event.get("mfe", 0.0)),
            mae_t=float(current_event.get("mae", 0.0)),
            unrealized_pnl_t=float(current_event.get("unrealized_pnl", 0.0)),
            stage_index=current_event.get("stage_index"),
            stage_progress=float(current_event.get("stage_progress", 0.0)),
            position_open=bool(current_event.get("position_open", False)),
            step_index_in_trade=current_event.get("step_index_in_trade"),
            time_under_water=float(current_event.get("time_under_water", 0.0)),
        )
        if self._reward_calc.prev_equity is None:
            self._reward_calc.prev_equity = ctx.equity_prev
        if self._reward_calc.prev_dd is None:
            self._reward_calc.prev_dd = ctx.dd_prev
        return self._reward_calc.compute(ctx)
