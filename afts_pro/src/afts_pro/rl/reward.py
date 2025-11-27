from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class RewardConfig:
    weight_equity_delta: float = 1.0
    weight_drawdown_delta: float = -1.0
    weight_stage_progress: float = 0.0
    weight_mfe_usage: float = 0.0
    weight_mae_penalty: float = 0.0
    weight_time_under_water: float = 0.0
    clip_min: Optional[float] = None
    clip_max: Optional[float] = None


@dataclass
class RewardContext:
    equity_t: float
    equity_prev: float
    dd_t: float
    dd_prev: float
    mfe_t: float = 0.0
    mae_t: float = 0.0
    unrealized_pnl_t: float = 0.0
    stage_index: Optional[int] = None
    stage_progress: float = 0.0
    position_open: bool = False
    step_index_in_trade: Optional[int] = None
    time_under_water: float = 0.0  # seconds or bars elapsed underwater


def _safe_div(num: float, den: float) -> float:
    if den == 0:
        return 0.0
    return num / den


def compute_mfe_usage_score(mfe: float, unrealized: float) -> float:
    """
    Score high if unrealized/realized pnl captures a significant portion of MFE.
    """
    if mfe <= 0:
        return 0.0
    return _safe_div(unrealized, mfe)


def compute_mae_penalty_score(mae: float) -> float:
    """
    Penalize large adverse excursion.
    """
    return -abs(mae)


def compute_time_under_water_score(time_under_water: float) -> float:
    """
    Penalize long time under water (negative reward).
    """
    return -time_under_water


class RewardCalculator:
    """
    Computes shaped RL reward based on configurable weights.
    """

    def __init__(self, cfg: RewardConfig):
        self.cfg = cfg

    def compute(self, ctx: RewardContext) -> float:
        eq_delta = ctx.equity_t - ctx.equity_prev
        dd_delta = ctx.dd_t - ctx.dd_prev
        equity_norm = max(abs(ctx.equity_prev), 1e-9)
        dd_norm = max(abs(ctx.dd_prev) + 1e-9, 1e-9)
        r_equity = self.cfg.weight_equity_delta * _safe_div(eq_delta, equity_norm)
        r_dd = self.cfg.weight_drawdown_delta * _safe_div(dd_delta, dd_norm)
        r_stage = self.cfg.weight_stage_progress * ctx.stage_progress
        mfe_score = compute_mfe_usage_score(ctx.mfe_t, ctx.unrealized_pnl_t)
        r_mfe = self.cfg.weight_mfe_usage * mfe_score
        mae_score = compute_mae_penalty_score(ctx.mae_t)
        r_mae = self.cfg.weight_mae_penalty * mae_score
        tuw_score = compute_time_under_water_score(ctx.time_under_water)
        r_tuw = self.cfg.weight_time_under_water * tuw_score

        reward = r_equity + r_dd + r_stage + r_mfe + r_mae + r_tuw
        if self.cfg.clip_min is not None:
            reward = max(self.cfg.clip_min, reward)
        if self.cfg.clip_max is not None:
            reward = min(self.cfg.clip_max, reward)
        return float(reward)
