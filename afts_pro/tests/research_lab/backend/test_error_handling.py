from fastapi.testclient import TestClient

from research_lab.backend.app.main import app


client = TestClient(app)


def test_http_exception_envelope_includes_request_id() -> None:
    resp = client.get("/api/debug/http-error")
    assert resp.status_code == 400
    data = resp.json()
    assert "request_id" in data and data["request_id"]
    assert data["error_code"] == "bad_request"
    assert data["detail"] == "forced error"
    assert resp.headers.get("X-Request-ID") == data["request_id"]


def test_generic_exception_envelope() -> None:
    resp = client.get("/api/debug/generic-error")
    assert resp.status_code == 500
    data = resp.json()
    assert data["error_code"] == "internal_error"
    assert data["request_id"]
