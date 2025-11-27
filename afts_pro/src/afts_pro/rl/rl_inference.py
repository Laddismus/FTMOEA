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

logger = logging.getLogger(__name__)


@dataclass
class ObservationBuilder:
    """
    Builds RL observations outside of RLTradingEnv.
    """

    obs_spec: RLObsSpec

    def build(
        self,
        market_state: MarketState,
        position: Optional[Position],
        account_state: AccountState,
        feature_bundle: Optional[Any] = None,
    ) -> np.ndarray:
        vector = []
        if feature_bundle and feature_bundle.raw:
            # Preserve deterministic ordering
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

        # Risk block
        equity = float(getattr(account_state, "equity", 0.0))
        drawdown = float(getattr(account_state, "unrealized_pnl", 0.0))
        vector.extend([equity, drawdown])

        arr = np.array(vector, dtype=np.float32)
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
        feature_bundle: Optional[Any] = None,
    ) -> Dict[str, Any]:
        obs = self.obs_builder.build(state, position, account_state, feature_bundle)
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
