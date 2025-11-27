import numpy as np

from afts_pro.rl.env import RLTradingEnv


def _dummy_config():
    return {
        "episode": {"max_steps": 3, "mode": "fixed_bars"},
        "reward": {
            "weight_equity_delta": 1.0,
            "weight_drawdown_delta": -1.0,
            "weight_stage_progress": 0.5,
            "weight_mfe_mae": 0.3,
        },
        "observation": {
            "include_features": True,
            "include_position_state": True,
            "include_risk_state": True,
            "equity_norm_mode": "start_equity",
        },
    }


def _event_stream():
    return [
        {
            "equity": 100.0,
            "drawdown": 0.0,
            "dd_remaining": 5.0,
            "features": [0.1, 0.2, 0.3, 0.4],
            "position_state": {"side": 1, "size_norm": 0.5, "unrealized_norm": 0.0},
            "stage_progress": 0.0,
        },
        {
            "equity": 101.0,
            "drawdown": 0.1,
            "dd_remaining": 4.9,
            "features": [0.2, 0.3, 0.4, 0.5],
            "position_state": {"side": 1, "size_norm": 0.6, "unrealized_norm": 0.05},
            "stage_progress": 0.1,
        },
        {
            "equity": 99.5,
            "drawdown": 0.5,
            "dd_remaining": 4.5,
            "features": [0.3, 0.4, 0.5, 0.6],
            "position_state": {"side": -1, "size_norm": 0.4, "unrealized_norm": -0.1},
            "stage_progress": 0.2,
        },
    ]


def test_rl_env_reset_step_shapes():
    env = RLTradingEnv(config=_dummy_config(), event_stream=_event_stream())
    obs, info = env.reset(seed=42)
    assert "reset" in info
    first_shape = obs.shape
    step_obs, reward, terminated, truncated, info = env.step(0)
    assert step_obs.shape == first_shape
    assert isinstance(reward, float)
    assert terminated is False
    assert truncated in {False, True}


def test_rl_env_reward_signs():
    env = RLTradingEnv(config=_dummy_config(), event_stream=_event_stream())
    env.reset(seed=123)
    obs, reward, terminated, truncated, info = env.step(0)
    assert reward > 0  # equity increased, small drawdown
    obs, reward2, terminated, truncated, info = env.step(0)
    assert reward2 < reward  # equity fell and dd worsened


def test_rl_env_determinism_with_seed():
    cfg = _dummy_config()
    stream = _event_stream()
    env1 = RLTradingEnv(config=cfg, event_stream=stream)
    env2 = RLTradingEnv(config=cfg, event_stream=stream)
    obs1, _ = env1.reset(seed=7)
    obs2, _ = env2.reset(seed=7)
    assert np.allclose(obs1, obs2)
    actions = [0, 1]
    traj1 = []
    traj2 = []
    for a in actions:
        traj1.append(env1.step(a))
        traj2.append(env2.step(a))
    for s1, s2 in zip(traj1, traj2):
        for i in range(2):  # obs, reward
            if i == 0:
                assert np.allclose(s1[i], s2[i])
            else:
                assert s1[i] == s2[i]
