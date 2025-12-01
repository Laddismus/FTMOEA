import pandas as pd

from afts_pro.core.models import MarketState, StrategyDecision
from afts_pro.rl.rl_inference import RLInferenceHook, ObservationBuilder
from afts_pro.rl.types import RLObsSpec, ActionSpec
from afts_pro.rl.risk_agent import RiskAgent, RiskAgentConfig
from afts_pro.rl.exit_agent import ExitAgent, ExitAgentConfig


class DummyRisk(RiskAgent):
    def __init__(self, val: float, spec: RLObsSpec):
        super().__init__(RiskAgentConfig(exploration_epsilon=0.0, min_risk_pct=0.1, max_risk_pct=2.0), spec, ActionSpec(action_type="continuous", shape=(1,)))
        self.val = val

    def act(self, obs, deterministic: bool = False) -> float:
        return self.val


class DummyExit(ExitAgent):
    def __init__(self, val: int, spec: RLObsSpec):
        super().__init__(ExitAgentConfig(exploration_epsilon=0.0, n_actions=6), spec, ActionSpec(action_type="discrete", num_actions=6))
        self.val = val

    def act(self, obs, deterministic: bool = False) -> int:
        return self.val


def test_strategy_signal_flows_through_risk_and_rl():
    obs_spec = RLObsSpec(shape=(4,), dtype="float32", as_dict=False)
    builder = ObservationBuilder(obs_spec)
    risk_agent = DummyRisk(1.0, obs_spec)
    exit_agent = DummyExit(3, obs_spec)
    hook = RLInferenceHook(risk_agent=risk_agent, exit_agent=exit_agent, obs_builder=builder)
    ms = MarketState(timestamp=pd.Timestamp.utcnow(), symbol="ETH", open=100, high=100, low=100, close=100, volume=0)
    decision = StrategyDecision(action="entry", side="long", confidence=1.0)
    actions = hook.compute_actions(ms, None, None, None)  # account_state optional for this test
    hook.apply_to_decision(decision, actions)
    assert decision.side == "long"
    assert decision.update.get("risk_pct") == 1.0
    assert decision.meta.get("exit_action") == 3
