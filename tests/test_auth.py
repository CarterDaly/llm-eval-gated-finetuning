"""Tests for API key authentication middleware."""
import pytest

from gated_finetune.app import create_app


@pytest.fixture
def authed_client(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret-key")
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture
def open_client():
    """Client with no API_KEY configured — auth disabled."""
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# ---------------------------------------------------------------------------
# Auth enabled
# ---------------------------------------------------------------------------

def test_missing_key_returns_401(authed_client):
    resp = authed_client.get("/users")
    assert resp.status_code == 401
    assert resp.get_json()["error"] == "Unauthorized"


def test_wrong_key_returns_401(authed_client):
    resp = authed_client.get("/users", headers={"X-API-Key": "wrong"})
    assert resp.status_code == 401


def test_correct_key_passes(authed_client):
    resp = authed_client.get("/users", headers={"X-API-Key": "secret-key"})
    assert resp.status_code == 200


def test_health_endpoint_always_public(authed_client):
    resp = authed_client.get("/")
    assert resp.status_code == 200


def test_metrics_endpoint_always_public(authed_client):
    resp = authed_client.get("/metrics")
    # 200 or 204 — just must not be 401
    assert resp.status_code != 401


def test_post_endpoint_requires_key(authed_client):
    resp = authed_client.post("/users", json={"name": "Alice", "email": "a@b.com"})
    assert resp.status_code == 401


def test_post_endpoint_passes_with_key(authed_client):
    resp = authed_client.post(
        "/users",
        json={"name": "Alice", "email": "a@b.com"},
        headers={"X-API-Key": "secret-key"},
    )
    assert resp.status_code == 201


# ---------------------------------------------------------------------------
# Auth disabled (no API_KEY set)
# ---------------------------------------------------------------------------

def test_no_api_key_env_allows_all(open_client):
    resp = open_client.get("/users")
    assert resp.status_code == 200


def test_no_api_key_env_ignores_header(open_client):
    resp = open_client.get("/users", headers={"X-API-Key": "anything"})
    assert resp.status_code == 200
