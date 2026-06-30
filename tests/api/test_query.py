import pytest
from fastapi.testclient import TestClient
from src.api.app import create_app


@pytest.fixture
def client():
    app = create_app()
    return TestClient(app)


def test_health_endpoint(client):
    resp = client.get("/health")
    assert resp.status_code in (200, 503)


def test_metrics_endpoint(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200


def test_query_validation(client):
    resp = client.post("/api/v1/query", json={})
    assert resp.status_code == 422


def test_query_empty(client):
    resp = client.post("/api/v1/query", json={"query": ""})
    assert resp.status_code in (200, 500)
