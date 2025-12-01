from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np

from afts_pro.core import MarketState, StrategyDecision
from afts_pro.exec.position_models import AccountState, Position
from afts_pro.rl.exit_agent import ExitAgent
from afts_pro.rl.risk_agent import RiskAgent
from afts_pro.rl.types import RLObsSpec
from afts_pro.rl.env_features import EnvFeatureConfig, FtmoFeatureConfig

logger = logging.getLogger(__name__)


@dataclass
class ObservationBuilder:
    """
    Builds RL observations outside of RLTradingEnv.
    """

    obs_spec: RLObsSpec
    feature_config: Optional[EnvFeatureConfig] = None

    def _clip_scale(self, x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    def build(
        self,
        market_state: MarketState,
        position: Optional[Position],
        account_state: AccountState,
        decision_meta: Optional[Dict[str, Any]] = None,
        feature_bundle: Optional[Any] = None,
    ) -> np.ndarray:
        cfg = self.feature_config or EnvFeatureConfig(ftmo=FtmoFeatureConfig())
        vector: list[float] = []
        meta = decision_meta or {}
        if cfg.base_price_features and feature_bundle and feature_bundle.raw:
            for key in sorted(feature_bundle.raw.values.keys()):
                vector.append(float(feature_bundle.raw.values.get(key, 0.0)))
        # Position block
        side_val = 0.0
        qty = 0.0
        unrealized = 0.0
        if position:
            side_val = 1.0 if position.side.value.lower() == "long" else -1.0
            qty = float(position.qty)
            unrealized = float(position.unrealized_pnl)
        vector.extend([side_val, qty, unrealized])

        # PnL / equity
        if cfg.base_pnl_features:
            equity = float(getattr(account_state, "equity", 0.0))
            drawdown = float(getattr(account_state, "unrealized_pnl", 0.0))
            vector.extend([equity, drawdown])

        fcfg = cfg.ftmo
        # FTMO core
        if fcfg.include_daily_dd_pct:
            dd = float(meta.get("ftmo_daily_loss_pct", 0.0))
            vector.append(self._clip_scale(dd / 10.0, -1.0, 1.0))
        if fcfg.include_overall_dd_pct:
            odd = float(meta.get("ftmo_overall_loss_pct", 0.0))
            vector.append(self._clip_scale(odd / 20.0, -1.0, 1.0))
        stage = int(meta.get("ftmo_plus_stage", 0))
        if fcfg.include_stage:
            vector.append(self._clip_scale(stage / max(fcfg.max_stage, 1), 0.0, 1.0))
        if fcfg.include_stage_one_hot:
            one_hot = [0.0] * (fcfg.max_stage + 1)
            if 0 <= stage <= fcfg.max_stage:
                one_hot[stage] = 1.0
            vector.extend(one_hot)
        if fcfg.include_rolling_dd_pct:
            rdd = float(meta.get("ftmo_plus_rolling_loss_pct", 0.0))
            vector.append(self._clip_scale(rdd / 5.0, -1.0, 1.0))
        if fcfg.include_loss_velocity:
            vel = float(meta.get("ftmo_plus_loss_velocity", 0.0))
            vector.append(self._clip_scale(vel / 10.0, -1.0, 1.0))
        if fcfg.include_profit_progress_pct:
            prog = float(meta.get("ftmo_plus_profit_progress_pct", 0.0))
            vector.append(self._clip_scale(prog / 20.0, 0.0, 1.0))
        if fcfg.include_session_one_hot and fcfg.sessions:
            sess_vec = [0.0] * len(fcfg.sessions)
            sess_name = meta.get("ftmo_plus_session_name")
            if sess_name in fcfg.sessions:
                sess_vec[fcfg.sessions.index(sess_name)] = 1.0
            vector.extend(sess_vec)
        if fcfg.include_news_flag:
            vector.append(1.0 if meta.get("ftmo_plus_blocked_news", False) else 0.0)
        if fcfg.include_time_fence_flag:
            vector.append(1.0 if meta.get("ftmo_plus_blocked_time", False) else 0.0)
        if fcfg.include_spread:
            spread = float(getattr(account_state, "current_spread_pips", 0.0) or 0.0)
            spread = min(spread, fcfg.spread_clip_pips)
            vector.append(self._clip_scale(spread / max(fcfg.spread_clip_pips, 1e-6), 0.0, 1.0))
        if fcfg.include_stability_kpis:
            pf = float(meta.get("ftmo_plus_pf", 1.0))
            winr = float(meta.get("ftmo_plus_winrate", 0.5))
            std = float(meta.get("ftmo_plus_pnl_std", 0.0))
            vector.append(self._clip_scale(pf / fcfg.stability_pf_clip, 0.0, 1.0))
            vector.append(self._clip_scale(winr / fcfg.stability_winrate_clip, 0.0, 1.0))
            vector.append(self._clip_scale(std / fcfg.stability_pnl_std_clip, 0.0, 1.0))
        if fcfg.include_circuit_active_flag:
            vector.append(1.0 if meta.get("ftmo_plus_blocked_circuit", False) else 0.0)

        arr = np.array(vector, dtype=np.float32)
        arr = np.clip(arr, -1.0, 1.0)
        if arr.shape[0] < self.obs_spec.shape[0]:
            padding = np.zeros(self.obs_spec.shape[0] - arr.shape[0], dtype=np.float32)
            arr = np.concatenate([arr, padding])
        elif arr.shape[0] > self.obs_spec.shape[0]:
            arr = arr[: self.obs_spec.shape[0]]
        return arr


class RLInferenceHook:
    """
    Applies RL agent inference to StrategyDecision in SIM pipeline.
    """

    def __init__(
        self,
        risk_agent: Optional[RiskAgent],
        exit_agent: Optional[ExitAgent],
        obs_builder: ObservationBuilder,
    ) -> None:
        self.risk_agent = risk_agent
        self.exit_agent = exit_agent
        self.obs_builder = obs_builder

    def compute_actions(
        self,
        state: MarketState,
        account_state: AccountState,
        position: Optional[Position],
        decision_meta: Optional[Dict[str, Any]] = None,
        feature_bundle: Optional[Any] = None,
    ) -> Dict[str, Any]:
        obs = self.obs_builder.build(state, position, account_state, decision_meta, feature_bundle)
        risk_pct = None
        exit_action = None
        agent_meta: Dict[str, Any] = {}
        if self.risk_agent:
            risk_pct = self.risk_agent.act(obs, deterministic=True)
            agent_meta["risk_agent"] = {"version": "v1"}
        if self.exit_agent:
            exit_action = self.exit_agent.act(obs, deterministic=True)
            agent_meta["exit_agent"] = {"version": "v1"}
        logger.debug("RL inference | obs_shape=%s | risk_pct=%s | exit_action=%s", obs.shape, risk_pct, exit_action)
        return {"risk_pct": risk_pct, "exit_action": exit_action, "raw_obs": obs, "agent_meta": agent_meta}

    def apply_to_decision(self, decision: StrategyDecision, agent_actions: Dict[str, Any]) -> None:
        if agent_actions.get("risk_pct") is not None:
            decision.update["risk_pct"] = agent_actions["risk_pct"]
        if agent_actions.get("exit_action") is not None:
            decision.meta["exit_action"] = agent_actions["exit_action"]
        meta = agent_actions.get("agent_meta", {})
        if meta:
            decision.meta["agent_meta"] = meta
