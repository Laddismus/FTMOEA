from __future__ import annotations

import math
from typing import List, Optional, Sequence

from afts_pro.runlogger.models import EquityPoint, MetricsSnapshot, TradeRecord


def _safe_div(num: float, den: float) -> Optional[float]:
    if den == 0:
        return None
    return num / den


def compute_profit_factor(trades: Sequence[TradeRecord]) -> Optional[float]:
    gross_win = sum(t.realized_pnl for t in trades if t.realized_pnl > 0)
    gross_loss = sum(t.realized_pnl for t in trades if t.realized_pnl < 0)
    return _safe_div(gross_win, abs(gross_loss))


def compute_winrate(trades: Sequence[TradeRecord]) -> Optional[float]:
    if not trades:
        return None
    wins = sum(1 for t in trades if t.realized_pnl > 0)
    return wins / len(trades)


def compute_avg_win_loss(trades: Sequence[TradeRecord]) -> tuple[Optional[float], Optional[float]]:
    wins = [t.realized_pnl for t in trades if t.realized_pnl > 0]
    losses = [abs(t.realized_pnl) for t in trades if t.realized_pnl < 0]
    avg_win = sum(wins) / len(wins) if wins else None
    avg_loss = sum(losses) / len(losses) if losses else None
    return avg_win, avg_loss


def compute_expectancy_per_trade(trades: Sequence[TradeRecord]) -> Optional[float]:
    if not trades:
        return None
    total = sum(t.realized_pnl for t in trades)
    return total / len(trades)


def compute_max_drawdown(equity_points: Sequence[EquityPoint]) -> tuple[Optional[float], Optional[float]]:
    max_equity = None
    max_dd_abs = 0.0
    max_dd_pct = 0.0
    for pt in equity_points:
        if max_equity is None or pt.equity > max_equity:
            max_equity = pt.equity
        if max_equity and pt.equity < max_equity:
            dd_abs = max_equity - pt.equity
            dd_pct = dd_abs / max_equity if max_equity else 0.0
            if dd_abs > max_dd_abs:
                max_dd_abs = dd_abs
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct
    if max_equity is None:
        return None, None
    return max_dd_abs, max_dd_pct


def compute_basic_sharpe_like(equity_points: Sequence[EquityPoint]) -> Optional[float]:
    if len(equity_points) < 2:
        return None
    returns: List[float] = []
    last = equity_points[0].equity
    for pt in equity_points[1:]:
        if last != 0:
            returns.append((pt.equity - last) / last)
        last = pt.equity
    if not returns:
        return None
    mean_ret = sum(returns) / len(returns)
    variance = sum((r - mean_ret) ** 2 for r in returns) / len(returns)
    std_ret = math.sqrt(variance)
    if std_ret == 0:
        return None
    sharpe_like = mean_ret / std_ret * math.sqrt(len(returns))
    return sharpe_like


def build_metrics_snapshot(trades: Sequence[TradeRecord], equity_points: Sequence[EquityPoint]) -> MetricsSnapshot:
    profit_factor = compute_profit_factor(trades)
    winrate = compute_winrate(trades)
    avg_win, avg_loss = compute_avg_win_loss(trades)
    expectancy = compute_expectancy_per_trade(trades)
    max_dd_abs, max_dd_pct = compute_max_drawdown(equity_points)
    sharpe_like = compute_basic_sharpe_like(equity_points)
    snapshot = MetricsSnapshot(
        profit_factor=profit_factor,
        winrate=winrate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        expectancy_per_trade=expectancy,
        num_trades=len(trades),
        max_drawdown_abs=max_dd_abs,
        max_drawdown_pct=max_dd_pct,
        sharpe_like_basic=sharpe_like,
    )
    return snapshot
