import numpy as np

from afts_pro.rl.env import RLTradingEnv


def _env_cfg(env_type: str):
    return {
        "env_type": env_type,
        "observation": {"include_features": True, "include_position_state": True, "include_risk_state": True},
        "reward_profiles": {
            "risk": {"weight_equity_delta": 1.0, "weight_drawdown_delta": -1.0},
            "exit": {"weight_equity_delta": 0.5, "weight_mfe_usage": 1.0, "weight_mae_penalty": -1.0},
        },
    }


def test_reset_produces_consistent_initial_state():
    cfg = _env_cfg("risk")
    stream = [{"equity": 100.0, "drawdown": 0.0, "features": [0, 0, 0, 0]}]
    env = RLTradingEnv(cfg, event_stream=stream)
    obs1, _ = env.reset(seed=123)
    obs2, _ = env.reset(seed=123)
    assert np.allclose(obs1, obs2)


def test_step_changes_reward_sign():
    cfg = _env_cfg("risk")
    stream = [
        {"equity": 100.0, "drawdown": 0.0, "features": [0, 0, 0, 0]},
        {"equity": 105.0, "drawdown": 0.0, "features": [0, 0, 0, 0]},
    ]
    env = RLTradingEnv(cfg, event_stream=stream)
    env.reset()
    _, reward, *_ = env.step(0)
    assert reward > 0


def test_reward_profile_switch_changes_magnitude():
    stream = [
        {"equity": 100.0, "drawdown": 0.0, "features": [0, 0, 0, 0], "mfe": 5.0, "mae": 1.0, "unrealized_pnl": 4.0},
        {"equity": 101.0, "drawdown": 0.0, "features": [0, 0, 0, 0], "mfe": 5.0, "mae": 1.0, "unrealized_pnl": 3.0},
    ]
    env_risk = RLTradingEnv(_env_cfg("risk"), event_stream=stream)
    env_exit = RLTradingEnv(_env_cfg("exit"), event_stream=stream)
    env_risk.reset()
    env_exit.reset()
    _, reward_risk, *_ = env_risk.step(0)
    _, reward_exit, *_ = env_exit.step(0)
    assert reward_risk != reward_exit
