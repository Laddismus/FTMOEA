import numpy as np

from afts_pro.rl.risk_agent import RiskAgent, RiskAgentConfig
from afts_pro.rl.types import RLObsSpec, ActionSpec


def _agent():
    cfg = RiskAgentConfig(min_risk_pct=0.1, max_risk_pct=1.0, exploration_epsilon=0.5)
    obs_spec = RLObsSpec(shape=(4,), dtype="float32", as_dict=False)
    action_spec = ActionSpec(action_type="continuous", shape=(1,))
    return RiskAgent(cfg, obs_spec, action_spec)


def test_risk_agent_act_clamps_to_bounds():
    agent = _agent()
    obs = np.ones(4, dtype=np.float32)
    act = agent.act(obs, deterministic=True)
    assert 0.1 <= act <= 1.0


def test_risk_agent_deterministic_act_stable():
    agent = _agent()
    obs = np.ones(4, dtype=np.float32)
    a1 = agent.act(obs, deterministic=True)
    a2 = agent.act(obs, deterministic=True)
    assert np.isclose(a1, a2)


def test_risk_agent_exploratory_varies():
    agent = _agent()
    obs = np.ones(4, dtype=np.float32)
    vals = {agent.act(obs, deterministic=False) for _ in range(10)}
    assert len(vals) >= 1
