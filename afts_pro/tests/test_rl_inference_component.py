import numpy as np
import pandas as pd

from afts_pro.core.models import MarketState, StrategyDecision
from afts_pro.exec.position_models import AccountState
from afts_pro.rl.rl_inference import RLInferenceHook, ObservationBuilder
from afts_pro.rl.types import RLObsSpec, ActionSpec
from afts_pro.rl.risk_agent import RiskAgent, RiskAgentConfig
from afts_pro.rl.exit_agent import ExitAgent, ExitAgentConfig
from afts_pro.rl.env import RLTradingEnv


class DummyRisk(RiskAgent):
    def __init__(self, val: float, spec: RLObsSpec):
        super().__init__(RiskAgentConfig(exploration_epsilon=0.0), spec, ActionSpec(action_type="continuous", shape=(1,)))
        self.val = val

    def act(self, obs, deterministic: bool = False) -> float:
        return self.val


class DummyExit(ExitAgent):
    def __init__(self, val: int, spec: RLObsSpec):
        super().__init__(ExitAgentConfig(exploration_epsilon=0.0), spec, ActionSpec(action_type="discrete", num_actions=6))
        self.val = val

    def act(self, obs, deterministic: bool = False) -> int:
        return self.val


def test_observation_builder_matches_env_shape():
    cfg = {"observation": {"include_features": True, "include_position_state": True, "include_risk_state": True}}
    env = RLTradingEnv(cfg, event_stream=[{"equity": 1.0, "drawdown": 0.0, "features": [0.0, 0.0, 0.0, 0.0]}])
    obs_env, _ = env.reset(seed=42)
    obs_spec = RLObsSpec(shape=obs_env.shape, dtype="float32", as_dict=False)
    builder = ObservationBuilder(obs_spec)
    ms = MarketState(timestamp=pd.Timestamp.utcnow(), symbol="ETH", open=0, high=0, low=0, close=0, volume=0)
    acc = AccountState(balance=100, equity=100, realized_pnl=0, unrealized_pnl=0, fees_total=0)
    obs = builder.build(ms, None, acc, None)
    assert obs.shape == obs_env.shape


def test_inference_hook_returns_expected_actions():
    obs_spec = RLObsSpec(shape=(4,), dtype="float32", as_dict=False)
    builder = ObservationBuilder(obs_spec)
    risk = DummyRisk(1.5, obs_spec)
    exit_a = DummyExit(3, obs_spec)
    hook = RLInferenceHook(risk_agent=risk, exit_agent=exit_a, obs_builder=builder)
    ms = MarketState(timestamp=pd.Timestamp.utcnow(), symbol="ETH", open=0, high=0, low=0, close=0, volume=0)
    acc = AccountState(balance=100, equity=100, realized_pnl=0, unrealized_pnl=0, fees_total=0)
    actions = hook.compute_actions(ms, acc, None, None)
    assert actions["risk_pct"] == 1.5
    assert actions["exit_action"] == 3


def test_apply_to_decision_sets_meta_and_update():
    obs_spec = RLObsSpec(shape=(4,), dtype="float32", as_dict=False)
    builder = ObservationBuilder(obs_spec)
    risk = DummyRisk(1.0, obs_spec)
    exit_a = DummyExit(2, obs_spec)
    hook = RLInferenceHook(risk_agent=risk, exit_agent=exit_a, obs_builder=builder)
    ms = MarketState(timestamp=pd.Timestamp.utcnow(), symbol="ETH", open=0, high=0, low=0, close=0, volume=0)
    acc = AccountState(balance=100, equity=100, realized_pnl=0, unrealized_pnl=0, fees_total=0)
    decision = StrategyDecision(action="entry", side="long", confidence=1.0)
    actions = hook.compute_actions(ms, acc, None, None)
    hook.apply_to_decision(decision, actions)
    assert decision.update.get("risk_pct") == 1.0
    assert decision.meta.get("exit_action") == 2
