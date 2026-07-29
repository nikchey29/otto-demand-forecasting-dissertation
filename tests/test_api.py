from fastapi.testclient import TestClient

from otto_forecasting.api import app


def test_api_starts_in_degraded_mode_without_artifacts(monkeypatch, tmp_path):
    monkeypatch.setenv("OTTO_ARTIFACT_DIR", str(tmp_path / "missing"))
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "degraded"
        forecast = client.post("/forecast", json={"history": []})
        assert forecast.status_code == 503
