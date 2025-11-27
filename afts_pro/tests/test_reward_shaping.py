import numpy as np
from afts_pro.rl.reward import RewardCalculator, RewardConfig, RewardContext, compute_mfe_usage_score


def test_equity_delta_positive_reward():
    cfg = RewardConfig(weight_equity_delta=1.0, weight_drawdown_delta=0.0)
    calc = RewardCalculator(cfg)
    ctx = RewardContext(
        equity_t=110.0,
        equity_prev=100.0,
        dd_t=0.0,
        dd_prev=0.0,
        mfe_t=0.0,
        mae_t=0.0,
        unrealized_pnl_t=0.0,
    )
    reward = calc.compute(ctx)
    assert reward > 0


def test_drawdown_penalized():
    cfg = RewardConfig(weight_equity_delta=0.0, weight_drawdown_delta=-1.0)
    calc = RewardCalculator(cfg)
    ctx = RewardContext(
        equity_t=100.0,
        equity_prev=100.0,
        dd_t=10.0,
        dd_prev=5.0,
        mfe_t=0.0,
        mae_t=0.0,
        unrealized_pnl_t=0.0,
    )
    reward = calc.compute(ctx)
    assert reward < 0


def test_mfe_usage_reward():
    cfg = RewardConfig(weight_mfe_usage=1.0)
    calc = RewardCalculator(cfg)
    ctx_good = RewardContext(
        equity_t=100.0,
        equity_prev=100.0,
        dd_t=0.0,
        dd_prev=0.0,
        mfe_t=10.0,
        mae_t=0.0,
        unrealized_pnl_t=8.0,
    )
    ctx_bad = RewardContext(
        equity_t=100.0,
        equity_prev=100.0,
        dd_t=0.0,
        dd_prev=0.0,
        mfe_t=10.0,
        mae_t=0.0,
        unrealized_pnl_t=1.0,
    )
    reward_good = calc.compute(ctx_good)
    reward_bad = calc.compute(ctx_bad)
    assert reward_good > reward_bad


def test_mae_penalty():
    cfg = RewardConfig(weight_mae_penalty=1.0)
    calc = RewardCalculator(cfg)
    ctx = RewardContext(
        equity_t=100.0,
        equity_prev=100.0,
        dd_t=0.0,
        dd_prev=0.0,
        mfe_t=0.0,
        mae_t=5.0,
        unrealized_pnl_t=0.0,
    )
    reward = calc.compute(ctx)
    assert reward < 0


def test_time_under_water_penalty():
    cfg = RewardConfig(weight_time_under_water=1.0)
    calc = RewardCalculator(cfg)
    ctx = RewardContext(
        equity_t=100.0,
        equity_prev=100.0,
        dd_t=0.0,
        dd_prev=0.0,
        mfe_t=0.0,
        mae_t=0.0,
        unrealized_pnl_t=0.0,
        time_under_water=5.0,
    )
    reward = calc.compute(ctx)
    assert reward < 0


def test_clipping():
    cfg = RewardConfig(weight_equity_delta=10.0, clip_max=1.0)
    calc = RewardCalculator(cfg)
    ctx = RewardContext(
        equity_t=200.0,
        equity_prev=100.0,
        dd_t=0.0,
        dd_prev=0.0,
        mfe_t=0.0,
        mae_t=0.0,
        unrealized_pnl_t=0.0,
    )
    reward = calc.compute(ctx)
    assert reward <= 1.0
