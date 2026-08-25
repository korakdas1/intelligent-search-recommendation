"""System API contracts. Database ping is mocked so these stay unit tests."""

from fastapi.testclient import TestClient
import pytest


def test_health_ok(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["service"] == "intelligent-search-recommendation-engine"
    assert body["version"] == "0.1.0"


def test_ready_ok_when_database_ok(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("app.api.system.ping_database", lambda: True)
    monkeypatch.setattr("app.api.system.probe_semantic_artifacts", lambda: "ok")
    monkeypatch.setattr("app.api.system.probe_ltr_artifacts", lambda: "skipped")
    monkeypatch.setattr("app.api.system.probe_cf_artifacts", lambda: "skipped")
    response = client.get("/ready")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ready"
    assert body["checks"]["application"] == "ok"
    assert body["checks"]["database"] == "ok"
    assert body["checks"]["semantic_index"] == "ok"
    assert body["checks"]["ltr_ranker"] == "skipped"
    assert body["checks"]["cf_model"] == "skipped"


def test_ready_not_ready_when_database_down(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("app.api.system.ping_database", lambda: False)
    monkeypatch.setattr("app.api.system.probe_semantic_artifacts", lambda: "ok")
    response = client.get("/ready")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["checks"]["application"] == "ok"
    assert body["checks"]["database"] == "unavailable"


def test_ready_not_ready_when_semantic_unavailable(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("app.api.system.ping_database", lambda: True)
    monkeypatch.setattr("app.api.system.probe_semantic_artifacts", lambda: "unavailable")
    response = client.get("/ready")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["checks"]["semantic_index"] == "unavailable"


def test_health_ok_when_semantic_unavailable(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("app.api.system.ping_database", lambda: False)
    monkeypatch.setattr("app.api.system.probe_semantic_artifacts", lambda: "unavailable")
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_unknown_route_is_404(client: TestClient) -> None:
    response = client.get("/this-route-does-not-exist")
    assert response.status_code == 404


def test_health_does_not_require_database(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("app.api.system.ping_database", lambda: False)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
