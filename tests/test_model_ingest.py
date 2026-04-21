"""Tests for POST /models and GET /models model ingest endpoint."""
import io
import pickle

import pytest

from data5580_hw.app import create_app
from data5580_hw.services.database.database_client import db
from data5580_hw.services.database.prediction import ModelSQL


class _DummyModel:
    feature_names_in_ = ["a", "b"]

    def predict(self, _x):
        return ["yes"]


def _pkl_bytes():
    buf = io.BytesIO()
    pickle.dump(_DummyModel(), buf)
    buf.seek(0)
    return buf.read()


@pytest.fixture
def app():
    app = create_app()
    app.config["TESTING"] = True
    yield app


@pytest.fixture
def client(app):
    with app.test_client() as c:
        yield c


# ---------------------------------------------------------------------------
# POST /models
# ---------------------------------------------------------------------------

def test_ingest_creates_model(client, tmp_path, monkeypatch):
    monkeypatch.setenv("MODELS_DIR", str(tmp_path))
    # Need to reload the module so the env var takes effect
    import data5580_hw.controllers.model as mc
    monkeypatch.setattr(mc, "_MODELS_DIR", str(tmp_path))

    resp = client.post(
        "/models",
        data={
            "file": (io.BytesIO(_pkl_bytes()), "model.pkl"),
            "name": "hire-v1",
            "version": "1",
            "features": "a,b",
        },
        content_type="multipart/form-data",
    )
    assert resp.status_code == 201
    body = resp.get_json()
    assert body["model_name"] == "hire-v1"
    assert body["model_version"] == "1"
    assert body["feature_names"] == ["a", "b"]
    assert body["model_id"] == "model-hire-v1-1"


def test_ingest_missing_file(client):
    resp = client.post(
        "/models",
        data={"name": "x", "version": "1", "features": "a"},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400
    assert "file" in resp.get_json()["error"]


def test_ingest_missing_name(client, tmp_path, monkeypatch):
    import data5580_hw.controllers.model as mc
    monkeypatch.setattr(mc, "_MODELS_DIR", str(tmp_path))

    resp = client.post(
        "/models",
        data={"file": (io.BytesIO(_pkl_bytes()), "model.pkl"), "version": "1", "features": "a"},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400
    assert "name" in resp.get_json()["error"]


def test_ingest_missing_version(client, tmp_path, monkeypatch):
    import data5580_hw.controllers.model as mc
    monkeypatch.setattr(mc, "_MODELS_DIR", str(tmp_path))

    resp = client.post(
        "/models",
        data={"file": (io.BytesIO(_pkl_bytes()), "model.pkl"), "name": "x", "features": "a"},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400
    assert "version" in resp.get_json()["error"]


def test_ingest_missing_features(client, tmp_path, monkeypatch):
    import data5580_hw.controllers.model as mc
    monkeypatch.setattr(mc, "_MODELS_DIR", str(tmp_path))

    resp = client.post(
        "/models",
        data={"file": (io.BytesIO(_pkl_bytes()), "model.pkl"), "name": "x", "version": "1"},
        content_type="multipart/form-data",
    )
    assert resp.status_code == 400
    assert "features" in resp.get_json()["error"]


def test_ingest_duplicate_returns_409(client, tmp_path, monkeypatch):
    import data5580_hw.controllers.model as mc
    monkeypatch.setattr(mc, "_MODELS_DIR", str(tmp_path))

    for _ in range(2):
        resp = client.post(
            "/models",
            data={
                "file": (io.BytesIO(_pkl_bytes()), "model.pkl"),
                "name": "dup-model",
                "version": "1",
                "features": "a,b",
            },
            content_type="multipart/form-data",
        )

    assert resp.status_code == 409
    assert "already exists" in resp.get_json()["error"]


def test_ingest_optional_fields(client, tmp_path, monkeypatch):
    import data5580_hw.controllers.model as mc
    monkeypatch.setattr(mc, "_MODELS_DIR", str(tmp_path))

    resp = client.post(
        "/models",
        data={
            "file": (io.BytesIO(_pkl_bytes()), "model.pkl"),
            "name": "full-model",
            "version": "2",
            "features": "x,y,z",
            "type": "tree",
            "mlflow_run_id": "run-abc",
        },
        content_type="multipart/form-data",
    )
    assert resp.status_code == 201
    body = resp.get_json()
    assert body["feature_names"] == ["x", "y", "z"]


# ---------------------------------------------------------------------------
# GET /models
# ---------------------------------------------------------------------------

def test_list_models_empty(client):
    resp = client.get("/models")
    assert resp.status_code == 200
    assert resp.get_json() == []


def test_list_models_returns_registered(client, app, tmp_path, monkeypatch):
    import data5580_hw.controllers.model as mc
    monkeypatch.setattr(mc, "_MODELS_DIR", str(tmp_path))

    client.post(
        "/models",
        data={
            "file": (io.BytesIO(_pkl_bytes()), "model.pkl"),
            "name": "list-test",
            "version": "1",
            "features": "a,b",
        },
        content_type="multipart/form-data",
    )

    resp = client.get("/models")
    assert resp.status_code == 200
    models = resp.get_json()
    assert len(models) == 1
    assert models[0]["model_name"] == "list-test"
