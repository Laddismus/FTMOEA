import math

import pytest

from research_lab.backend.core.analytics.kpi_engine import RollingKpiEngine


def test_compute_rolling_kpis() -> None:
    returns = [1, -1, 2, -2, 3, -3]
    window = 3
    engine = RollingKpiEngine()

    windows = engine.compute_rolling_kpis(returns, window)

    assert len(windows) == len(returns) - window + 1

    first = windows[0]
    assert first.start_index == 0
    assert first.end_index == 2
    assert first.profit_factor == pytest.approx(3.0)
    assert first.win_rate == pytest.approx(2 / 3)
    assert first.avg_win == pytest.approx(1.5)
    assert first.avg_loss == pytest.approx(-1.0)
    assert first.max_drawdown == pytest.approx(1.0)

    second = windows[1]
    assert second.start_index == 1
    assert second.end_index == 3
    assert second.avg_loss < 0  # sanity check for negative avg loss


def test_compute_rolling_kpis_handles_no_losses_or_wins() -> None:
    engine = RollingKpiEngine()
    all_wins = [1, 2, 3]
    all_losses = [-1, -2, -3]

    wins_window = engine.compute_rolling_kpis(all_wins, window=2)[0]
    assert math.isinf(wins_window.profit_factor)
    assert wins_window.avg_loss == 0.0

    losses_window = engine.compute_rolling_kpis(all_losses, window=2)[0]
    assert losses_window.profit_factor == 0.0
    assert losses_window.avg_win == 0.0
