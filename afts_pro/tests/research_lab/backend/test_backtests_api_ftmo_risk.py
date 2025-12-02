from datetime import datetime, timezone, timedelta

from fastapi.testclient import TestClient

from research_lab.backend.app.main import app


def test_api_backtest_with_ftmo_risk_breach() -> None:
    client = TestClient(app)
    base = datetime.now(timezone.utc)
    bars = [
        {"ts": base.isoformat(), "open": 1, "high": 1.1, "low": 0.9, "close": 1.0},
        {"ts": (base + timedelta(minutes=1)).isoformat(), "open": 1, "high": 1.0, "low": 0.8, "close": 0.8},
    ]
    payload = {
        "mode": "graph",
        "returns": [-0.06, 0.0],
        "bars": bars,
        "window": 2,
        "ftmo_risk": {"initial_equity": 1.0, "max_daily_loss_pct": 0.05, "max_total_loss_pct": 0.1, "safety_buffer_pct": 0.0},
    }

    resp = client.post("/api/backtests/run-sync", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["ftmo_risk_summary"] is not None
    assert data["ftmo_risk_summary"]["passed"] is False
    assert data["ftmo_risk_summary"]["first_breach"]["breach_type"] == "daily"
