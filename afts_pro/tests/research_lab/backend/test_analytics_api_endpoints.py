from fastapi.testclient import TestClient

from research_lab.backend.app.main import app


def test_stats_endpoint() -> None:
    client = TestClient(app)
    response = client.post("/api/analytics/stats", json={"series": [1, 2, 3, 4]})
    assert response.status_code == 200
    data = response.json()
    assert "mean" in data and data["mean"] > 0
    assert data["quantiles"]["0.5"] == 2.5


def test_kpis_endpoint() -> None:
    client = TestClient(app)
    payload = {"returns": [1, -1, 2, -2, 3], "window": 3}
    response = client.post("/api/analytics/kpis", json=payload)
    assert response.status_code == 200
    windows = response.json()["windows"]
    assert len(windows) == 3
    assert windows[0]["profit_factor"] > 0


def test_drift_endpoint() -> None:
    client = TestClient(app)
    payload = {"base_window": [1, 1, 1, 1], "current_window": [1, 1, 1.1, 1], "threshold": 5.0}
    response = client.post("/api/analytics/drift", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "drift_detected" in data


def test_regimes_endpoint() -> None:
    client = TestClient(app)
    features = [[0.0, 0.0], [0.1, -0.1], [10.0, 10.0], [10.1, 9.9]]
    payload = {"features": features, "n_clusters": 2}
    response = client.post("/api/analytics/regimes", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert len(data["labels"]) == len(features)
