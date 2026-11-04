"""Tests for POST /models and GET /models."""
import io
import pickle

import pytest

import gated_finetune.controllers.model as model_controller_module
from gated_finetune.app import create_app


class _DummyModel:
    feature_names_in_ = ["a", "b"]

    def predict(self, _x):
        return ["yes"]


def _pkl_bytes():
    return pickle.dumps(_DummyModel())


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture
def models_dir(tmp_path, monkeypatch):
    target = tmp_path / "models"
    monkeypatch.setattr(model_controller_module, "_MODELS_DIR", str(target))
    return target


def _form(**overrides):
    form = {
        "file": (io.BytesIO(_pkl_bytes()), "model.pkl"),
        "name": "demo-v1",
        "version": "1",
        "features": "a,b",
        "mlflow_run_id": "run-abc",
    }
    form.update(overrides)
    return {k: v for k, v in form.items() if v is not None}


def _post(client, **overrides):
    return client.post("/models", data=_form(**overrides), content_type="multipart/form-data")


# ---------------------------------------------------------------------------
# POST /models
# ---------------------------------------------------------------------------

def test_ingest_creates_model(client, models_dir):
    resp = _post(client)
    assert resp.status_code == 201
    body = resp.get_json()
    assert body["model_name"] == "demo-v1"
    assert body["model_version"] == "1"
    assert body["feature_names"] == ["a", "b"]
    assert body["model_id"] == "model-demo-v1-1"
    assert (models_dir / "demo-v1" / "1" / "model.pkl").is_file()


def test_ingest_accepts_json_feature_list(client, models_dir):
    resp = _post(client, features='["x", "y", "z"]', type="tree")
    assert resp.status_code == 201
    assert resp.get_json()["feature_names"] == ["x", "y", "z"]


@pytest.mark.parametrize("field", ["file", "name", "version", "features", "mlflow_run_id"])
def test_ingest_missing_required_field(client, models_dir, field):
    resp = _post(client, **{field: None})
    assert resp.status_code == 400
    assert field in resp.get_json()["error"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("name", "../../escaped"),
        ("name", "/etc"),
        ("name", "a/b"),
        ("name", ".hidden"),
        ("version", ".."),
        ("version", "1/../../x"),
    ],
)
def test_ingest_rejects_unsafe_names(client, models_dir, tmp_path, field, value):
    resp = _post(client, **{field: value})
    assert resp.status_code == 400
    assert f"Invalid {field}" in resp.get_json()["error"]
    assert not any(tmp_path.rglob("model.pkl"))


def test_ingest_duplicate_returns_409(client, models_dir):
    _post(client, name="dup-model")
    resp = _post(client, name="dup-model")
    assert resp.status_code == 409
    assert "already exists" in resp.get_json()["error"]


# ---------------------------------------------------------------------------
# GET /models
# ---------------------------------------------------------------------------

def test_list_models_empty(client):
    resp = client.get("/models")
    assert resp.status_code == 200
    assert resp.get_json() == []


def test_list_models_returns_registered(client, models_dir):
    _post(client, name="list-test")

    resp = client.get("/models")
    assert resp.status_code == 200
    models = resp.get_json()
    assert len(models) == 1
    assert models[0]["model_name"] == "list-test"
    assert models[0]["mlflow_run_id"] == "run-abc"
