import numpy as np
import pandas as pd

from afts_pro.rl.env import RLTradingEnv
from afts_pro.rl.reward import RewardConfig


def _env_config(env_type: str):
    return {
        "env_type": env_type,
        "observation": {"include_features": True, "include_position_state": True, "include_risk_state": True},
        "reward_profiles": {
            "risk": {"weight_equity_delta": 1.0, "weight_drawdown_delta": -2.0, "weight_stage_progress": 0.1},
            "exit": {"weight_equity_delta": 0.5, "weight_mfe_usage": 1.0, "weight_mae_penalty": -1.0},
        },
    }


def test_risk_env_uses_risk_profile():
    cfg = _env_config("risk")
    stream = [
        {"equity": 100.0, "drawdown": 0.0, "features": [0.0, 0.0, 0.0, 0.0]},
        {"equity": 101.0, "drawdown": 0.1, "features": [0.0, 0.0, 0.0, 0.0]},
    ]
    env = RLTradingEnv(cfg, event_stream=stream)
    env.reset()
    _, reward, *_ = env.step(0)
    assert reward != 0


def test_exit_env_uses_exit_profile():
    cfg = _env_config("exit")
    stream = [
        {"equity": 100.0, "drawdown": 0.0, "features": [0.0, 0.0, 0.0, 0.0], "mfe": 5.0, "unrealized_pnl": 4.0},
        {"equity": 100.0, "drawdown": 0.0, "features": [0.0, 0.0, 0.0, 0.0], "mfe": 5.0, "unrealized_pnl": 1.0},
    ]
    env = RLTradingEnv(cfg, event_stream=stream)
    env.reset()
    _, reward, *_ = env.step(0)
    assert reward != 0


def test_reward_signals_in_synthetic_trade():
    cfg = _env_config("exit")
    stream_good_exit = [
        {"equity": 100.0, "drawdown": 0.0, "features": [0, 0, 0, 0], "mfe": 5.0, "mae": 1.0, "unrealized_pnl": 4.0},
        {"equity": 104.0, "drawdown": 0.0, "features": [0, 0, 0, 0], "mfe": 5.0, "mae": 1.0, "unrealized_pnl": 4.5},
    ]
    env_good = RLTradingEnv(cfg, event_stream=stream_good_exit)
    env_good.reset()
    _, reward_good, *_ = env_good.step(0)

    stream_bad_exit = [
        {"equity": 100.0, "drawdown": 0.0, "features": [0, 0, 0, 0], "mfe": 5.0, "mae": 1.0, "unrealized_pnl": 4.0},
        {"equity": 101.0, "drawdown": 0.0, "features": [0, 0, 0, 0], "mfe": 5.0, "mae": 3.0, "unrealized_pnl": 0.5},
    ]
    env_bad = RLTradingEnv(cfg, event_stream=stream_bad_exit)
    env_bad.reset()
    _, reward_bad, *_ = env_bad.step(0)

    assert reward_good > reward_bad
