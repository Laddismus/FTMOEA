"""Rolling KPI computations for research analytics."""

from __future__ import annotations

import itertools
from typing import Sequence

import numpy as np

from research_lab.backend.core.analytics.models import RollingKpiWindow


class RollingKpiEngine:
    """Compute rolling KPIs over sequences of returns."""

    def compute_rolling_kpis(self, returns: Sequence[float], window: int) -> list[RollingKpiWindow]:
        """Compute KPIs for each sliding window of the given returns.

        Args:
            returns: Sequence of numeric returns.
            window: Size of the rolling window.

        Returns:
            List of RollingKpiWindow ordered by start index.
        """

        if window <= 0:
            raise ValueError("window must be positive")
        if window > len(returns):
            raise ValueError("window must be <= length of returns")

        results: list[RollingKpiWindow] = []
        for start in range(0, len(returns) - window + 1):
            slice_returns = np.asarray(returns[start : start + window], dtype=float)
            positive = slice_returns[slice_returns > 0]
            negative = slice_returns[slice_returns < 0]

            sum_pos = float(np.sum(positive)) if positive.size else 0.0
            sum_neg = float(np.sum(negative)) if negative.size else 0.0
            profit_factor = float("inf") if sum_neg == 0.0 and sum_pos > 0 else 0.0 if sum_pos == 0 else sum_pos / abs(sum_neg)

            win_rate = float(len(positive)) / float(window)
            avg_win = float(np.mean(positive)) if positive.size else 0.0
            avg_loss = float(np.mean(negative)) if negative.size else 0.0

            max_drawdown = self._compute_max_drawdown(slice_returns)

            results.append(
                RollingKpiWindow(
                    start_index=start,
                    end_index=start + window - 1,
                    profit_factor=profit_factor,
                    win_rate=win_rate,
                    avg_win=avg_win,
                    avg_loss=avg_loss,
                    max_drawdown=max_drawdown,
                )
            )
        return results

    @staticmethod
    def _compute_max_drawdown(returns: np.ndarray) -> float:
        equity_curve = np.cumsum(returns)
        peak = -np.inf
        max_dd = 0.0
        for value in equity_curve:
            peak = max(peak, value)
            drawdown = peak - value
            max_dd = max(max_dd, drawdown)
        return float(max_dd)


__all__ = ["RollingKpiEngine"]
