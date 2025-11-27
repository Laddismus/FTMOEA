import numpy as np
from pathlib import Path

from afts_pro.rl.exit_agent import ExitAgent, ExitAgentConfig
from afts_pro.rl.types import RLObsSpec, ActionSpec
from afts_pro.rl.exit_training import train_exit_agent, ExitTrainConfig


class DummyExitEnv:
    def __init__(self, obs_dim: int = 4, n_actions: int = 6):
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.obs_spec = RLObsSpec(shape=(obs_dim,), dtype="float32", as_dict=False)

    def reset(self, seed=None, options=None):
        return np.zeros(self.obs_dim, dtype=np.float32), {}

    def step(self, action):
        obs = np.ones(self.obs_dim, dtype=np.float32)
        reward = 0.1
        terminated = True
        truncated = False
        return obs, reward, terminated, truncated, {}


def _agent(obs_dim: int = 4) -> ExitAgent:
    cfg = ExitAgentConfig()
    obs_spec = RLObsSpec(shape=(obs_dim,), dtype="float32", as_dict=False)
    action_spec = ActionSpec(action_type="discrete", num_actions=cfg.n_actions)
    return ExitAgent(cfg, obs_spec, action_spec)


def test_exit_agent_action_range():
    agent = _agent()
    obs = np.zeros(4, dtype=np.float32)
    action = agent.act(obs, deterministic=True)
    assert 0 <= action < agent.config.n_actions


def test_exit_agent_deterministic_vs_exploratory():
    agent = _agent()
    obs = np.ones(4, dtype=np.float32)
    deterministic_action = agent.act(obs, deterministic=True)
    exploratory_actions = {agent.act(obs, deterministic=False) for _ in range(10)}
    assert deterministic_action in exploratory_actions
    assert all(0 <= a < agent.config.n_actions for a in exploratory_actions)


def test_exit_training_runs_one_episode(tmp_path):
    env = DummyExitEnv()
    agent = _agent()
    train_cfg = ExitTrainConfig(total_episodes=1, max_steps_per_episode=2, replay_capacity=10, batch_size=2, log_interval=1, save_every_n_episodes=1)
    summary = train_exit_agent(env, agent, train_cfg, checkpoint_dir=tmp_path / "ckpt")
    assert summary.episodes == 1
    assert summary.returns
    assert (tmp_path / "ckpt" / "exit_train_summary.json").exists()


def test_exit_agent_save_load_roundtrip(tmp_path):
    agent = _agent()
    obs = np.ones(4, dtype=np.float32)
    action_before = agent.act(obs, deterministic=True)
    save_path = tmp_path / "model"
    agent.save(save_path)

    agent2 = _agent()
    agent2.load(save_path)
    action_after = agent2.act(obs, deterministic=True)
    assert np.isclose(action_before, action_after)
