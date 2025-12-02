from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from research_lab.backend.core.backtests.engine import RollingKpiBacktestEngine
from research_lab.backend.core.backtests.models import BacktestRequest, BacktestBar, PythonStrategyRef

STRATEGY_CODE = """
from research_lab.backend.core.python_strategies.interface import BasePythonStrategy
from research_lab.backend.core.backtests.models import BacktestBar, BacktestPositionState, TradingAction, PositionSide

class FeeStrategy(BasePythonStrategy):
    strategy_key = "fee.strat"
    strategy_name = "Fee Strategy"
    strategy_version = "1.0.0"
    def on_bar_trade(self, bar: BacktestBar, state: BacktestPositionState) -> TradingAction:
        if state.side == PositionSide.FLAT:
            return TradingAction.ENTER_LONG
        return TradingAction.EXIT
"""


def test_engine_trading_state_with_costs_and_size(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module_path = tmp_path / "fee_strategy.py"
    module_path.write_text(STRATEGY_CODE, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))

    engine = RollingKpiBacktestEngine()
    request = BacktestRequest(
        mode="python",
        python_strategy=PythonStrategyRef(module_path="fee_strategy", class_name="FeeStrategy"),
        bars=[
            BacktestBar(ts=datetime.now(timezone.utc), open=1.0, high=1.0, low=1.0, close=1.0),
            BacktestBar(ts=datetime.now(timezone.utc) + timedelta(minutes=1), open=1.0, high=1.5, low=1.0, close=1.5),
        ],
        window=2,
        strategy_params={"size": 2.0},
        cost_model={"fee_rate": 0.001, "slippage_rate": 0.0},
    )

    result = engine.run_backtest(request)

    assert result.trades is not None and len(result.trades) == 1
    trade = result.trades[0]
    assert trade.size == 2.0
    assert trade.fees is not None and trade.fees > 0
    # net return should be less than gross due to fees
    assert trade.net_return is not None and trade.gross_return is not None
    assert trade.net_return < trade.gross_return * trade.size
