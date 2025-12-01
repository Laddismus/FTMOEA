import numpy as np

from afts_pro.rl.exit_agent import ExitAgent, ExitAgentConfig
from afts_pro.rl.types import RLObsSpec, ActionSpec


def _agent():
    cfg = ExitAgentConfig(n_actions=6, exploration_epsilon=0.5)
    obs_spec = RLObsSpec(shape=(4,), dtype="float32", as_dict=False)
    action_spec = ActionSpec(action_type="discrete", num_actions=cfg.n_actions)
    return ExitAgent(cfg, obs_spec, action_spec)


def test_exit_agent_returns_valid_action():
    agent = _agent()
    obs = np.zeros(4, dtype=np.float32)
    act = agent.act(obs, deterministic=True)
    assert 0 <= act < agent.config.n_actions


def test_exit_agent_deterministic_action_stable():
    agent = _agent()
    obs = np.zeros(4, dtype=np.float32)
    a1 = agent.act(obs, deterministic=True)
    a2 = agent.act(obs, deterministic=True)
    assert a1 == a2
