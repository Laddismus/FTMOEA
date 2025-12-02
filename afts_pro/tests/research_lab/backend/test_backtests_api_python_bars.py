from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from research_lab.backend.app.main import app


STRATEGY_CODE = """
from research_lab.backend.core.python_strategies.interface import BasePythonStrategy
from research_lab.backend.core.backtests.models import BacktestBar, BacktestPositionState, TradingAction, PositionSide

class ApiBarStrategy(BasePythonStrategy):
    strategy_key = "api.bar"
    strategy_name = "API Bar Strategy"
    strategy_version = "1.0.0"
    def on_bar_trade(self, bar: BacktestBar, state: BacktestPositionState) -> TradingAction:
        if state.side == PositionSide.FLAT:
            return TradingAction.ENTER_LONG
        return TradingAction.EXIT
"""


def test_api_python_bars_backtest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module_path = tmp_path / "api_bar_strategy.py"
    module_path.write_text(STRATEGY_CODE, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))

    client = TestClient(app)
    bars = [
        {"ts": datetime.now(timezone.utc).isoformat(), "open": 1, "high": 1.1, "low": 0.9, "close": 1.0},
        {"ts": (datetime.now(timezone.utc) + timedelta(minutes=1)).isoformat(), "open": 1.0, "high": 1.2, "low": 1.0, "close": 1.2},
    ]
    payload = {
        "mode": "python",
        "python_strategy": {"module_path": "api_bar_strategy", "class_name": "ApiBarStrategy"},
        "bars": bars,
        "window": 2,
        "cost_model": {"fee_rate": 0.001, "slippage_rate": 0.0},
        "strategy_params": {"size": 1.0},
    }

    resp = client.post("/api/backtests/run-sync", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["strategy_metadata"]["type"] == "python"
    assert "kpi_summary" in data
    assert data["trades"] is not None
    assert data["trades"][0]["fees"] >= 0
