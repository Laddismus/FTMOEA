import numpy as np

import pandas as pd
from afts_pro.core.models import StrategyDecision, MarketState
from afts_pro.exec.position_models import AccountState, Position, PositionSide
from afts_pro.rl.rl_inference import RLInferenceHook, ObservationBuilder
from afts_pro.rl.types import RLObsSpec, ActionSpec
from afts_pro.rl.exit_agent import ExitAgentConfig, ExitAgent
from afts_pro.rl.risk_agent import RiskAgentConfig, RiskAgent
from afts_pro.rl.env import RLTradingEnv


class DummyRiskAgent(RiskAgent):
    def __init__(self, value: float, obs_spec: RLObsSpec):
        super().__init__(RiskAgentConfig(), obs_spec, ActionSpec(action_type="continuous", shape=(1,)))
        self._value = value

    def act(self, obs, deterministic: bool = False) -> float:
        return self._value


class DummyExitAgent(ExitAgent):
    def __init__(self, value: int, obs_spec: RLObsSpec):
        super().__init__(ExitAgentConfig(), obs_spec, ActionSpec(action_type="discrete", num_actions=6))
        self._value = value

    def act(self, obs, deterministic: bool = False) -> int:
        return self._value


def _account_state():
    return AccountState(balance=100.0, equity=100.0, realized_pnl=0.0, unrealized_pnl=0.0, fees_total=0.0)


def test_risk_agent_influence_on_decision():
    obs_spec = RLObsSpec(shape=(6,), dtype="float32", as_dict=False)
    builder = ObservationBuilder(obs_spec)
    risk_agent = DummyRiskAgent(0.5, obs_spec)
    hook = RLInferenceHook(risk_agent=risk_agent, exit_agent=None, obs_builder=builder)
    ms = MarketState(timestamp=pd.Timestamp.utcnow(), symbol="ETH", open=0, high=0, low=0, close=0, volume=0)
    decision = StrategyDecision(action="none", side=None, confidence=0.0)
    actions = hook.compute_actions(ms, _account_state(), None, None)
    hook.apply_to_decision(decision, actions)
    assert decision.update.get("risk_pct") == 0.5


def test_exit_agent_influence_on_decision():
    obs_spec = RLObsSpec(shape=(6,), dtype="float32", as_dict=False)
    builder = ObservationBuilder(obs_spec)
    exit_agent = DummyExitAgent(3, obs_spec)
    hook = RLInferenceHook(risk_agent=None, exit_agent=exit_agent, obs_builder=builder)
    ms = MarketState(timestamp=pd.Timestamp.utcnow(), symbol="ETH", open=0, high=0, low=0, close=0, volume=0)
    decision = StrategyDecision(action="none", side=None, confidence=0.0)
    actions = hook.compute_actions(ms, _account_state(), None, None)
    hook.apply_to_decision(decision, actions)
    assert decision.meta.get("exit_action") == 3


def test_observation_builder_matches_env_shapes():
    cfg = {"observation": {"include_features": True, "include_position_state": True, "include_risk_state": True}}
    env = RLTradingEnv(cfg, event_stream=[{"equity": 1.0, "drawdown": 0.0, "dd_remaining": 1.0, "features": [0.0, 0.0, 0.0, 0.0]}])
    obs_env, _ = env.reset()
    obs_spec = RLObsSpec(shape=obs_env.shape, dtype="float32", as_dict=False)
    builder = ObservationBuilder(obs_spec)
    ms = MarketState(timestamp=pd.Timestamp.utcnow(), symbol="ETH", open=0, high=0, low=0, close=0, volume=0)
    acc = _account_state()
    pos = Position(symbol="ETH", side=PositionSide.LONG, qty=1.0, entry_price=100.0, realized_pnl=0.0, unrealized_pnl=0.0, avg_entry_fees=0.0)
    obs_built = builder.build(ms, pos, acc, None)
    assert obs_built.shape == obs_env.shape


def test_rl_inference_hook_position_state_none():
    obs_spec = RLObsSpec(shape=(6,), dtype="float32", as_dict=False)
    builder = ObservationBuilder(obs_spec)
    risk_agent = DummyRiskAgent(0.4, obs_spec)
    exit_agent = DummyExitAgent(2, obs_spec)
    hook = RLInferenceHook(risk_agent=risk_agent, exit_agent=exit_agent, obs_builder=builder)
    ms = MarketState(timestamp=pd.Timestamp.utcnow(), symbol="ETH", open=0, high=0, low=0, close=0, volume=0)
    decision = StrategyDecision(action="none", side=None, confidence=0.0)
    actions = hook.compute_actions(ms, _account_state(), None, None)
    hook.apply_to_decision(decision, actions)
    assert decision.meta.get("exit_action") == 2
    assert decision.update.get("risk_pct") == 0.4


def test_rl_inference_disabled_does_nothing():
    obs_spec = RLObsSpec(shape=(4,), dtype="float32", as_dict=False)
    builder = ObservationBuilder(obs_spec)
    hook = RLInferenceHook(risk_agent=None, exit_agent=None, obs_builder=builder)
    ms = MarketState(timestamp=None, symbol="ETH", open=0, high=0, low=0, close=0, volume=0)  # type: ignore[arg-type]
    decision = StrategyDecision(action="none", side=None, confidence=0.0)
    actions = hook.compute_actions(ms, _account_state(), None, None)
    hook.apply_to_decision(decision, actions)
    assert decision.update == {}
    assert decision.meta == {}
