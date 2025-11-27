import numpy as np
from pathlib import Path

from afts_pro.rl.risk_agent import RiskAgent, RiskAgentConfig
from afts_pro.rl.types import RLObsSpec, ActionSpec
from afts_pro.rl.risk_training import train_risk_agent, TrainLoopConfig


class DummyEnv:
    def __init__(self, obs_dim: int = 4):
        self.obs_dim = obs_dim
        self.obs_spec = RLObsSpec(shape=(obs_dim,), dtype="float32", as_dict=False)
        self._step = 0

    def reset(self, seed=None, options=None):
        self._step = 0
        return np.zeros(self.obs_dim, dtype=np.float32), {}

    def step(self, action):
        self._step += 1
        obs = np.full(self.obs_dim, self._step, dtype=np.float32)
        reward = 0.1
        terminated = self._step >= 3
        truncated = False
        return obs, reward, terminated, truncated, {}


def _agent(obs_dim: int = 4) -> RiskAgent:
    cfg = RiskAgentConfig()
    obs_spec = RLObsSpec(shape=(obs_dim,), dtype="float32", as_dict=False)
    action_spec = ActionSpec(action_type="continuous", shape=(1,))
    return RiskAgent(cfg, obs_spec, action_spec)


def test_risk_agent_action_range():
    agent = _agent()
    obs = np.zeros(4, dtype=np.float32)
    action = agent.act(obs, deterministic=True)
    assert agent.config.min_risk_pct <= action <= agent.config.max_risk_pct


def test_risk_agent_deterministic_vs_exploratory():
    agent = _agent()
    obs = np.ones(4, dtype=np.float32)
    deterministic_action = agent.act(obs, deterministic=True)
    exploratory_actions = {agent.act(obs, deterministic=False) for _ in range(10)}
    assert deterministic_action in exploratory_actions
    assert len(exploratory_actions) >= 1


def test_train_loop_runs_one_episode(tmp_path):
    env = DummyEnv()
    agent = _agent()
    train_cfg = TrainLoopConfig(total_episodes=1, max_steps_per_episode=5, replay_capacity=10, batch_size=2, log_interval=1, save_every_n_episodes=1)
    summary = train_risk_agent(env, agent, train_cfg, checkpoint_dir=tmp_path / "ckpt")
    assert summary.episodes == 1
    assert summary.returns
    assert (tmp_path / "ckpt" / "train_summary.json").exists()


def test_risk_agent_save_load_roundtrip(tmp_path):
    agent = _agent()
    obs = np.ones(4, dtype=np.float32)
    action_before = agent.act(obs, deterministic=True)
    save_path = tmp_path / "model"
    agent.save(save_path)

    agent2 = _agent()
    agent2.load(save_path)
    action_after = agent2.act(obs, deterministic=True)
    assert np.isclose(action_before, action_after)
