"""Tests for registering LLMs (base model + optional LoRA adapter) on POST /models."""
import pytest

from gated_finetune.app import create_app
from gated_finetune.tasks.prediction_tasks import _query_model

REVISION = "a" * 40
SHA256 = "b" * 64


@pytest.fixture
def app():
    app = create_app()
    app.config["TESTING"] = True
    return app


@pytest.fixture
def client(app):
    with app.test_client() as c:
        yield c


def _body(**overrides):
    body = {
        "kind": "llm",
        "name": "fc-bot",
        "version": "v1",
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "base_model_revision": REVISION,
    }
    body.update(overrides)
    return {k: v for k, v in body.items() if v is not None}


def _register(client, **overrides):
    return client.post("/models", json=_body(**overrides))


def test_register_base_model(client):
    resp = _register(client)
    assert resp.status_code == 201
    body = resp.get_json()
    assert body["model_id"] == "model-fc-bot-v1"
    assert body["kind"] == "llm"
    assert body["model_type"] == "base"
    assert body["base_model"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert body["base_model_revision"] == REVISION
    assert body["adapter_uri"] is None
    # Greedy decoding by default, so eval runs repeat
    assert body["generation_config"] == {"temperature": 0.0, "top_p": 1.0, "max_tokens": 512, "seed": None}


def test_register_adapter_with_parent(client):
    assert _register(client).status_code == 201
    resp = _register(
        client,
        version="v2",
        adapter_uri="runs:/run-123/adapter",
        adapter_sha256=SHA256,
        mlflow_run_id="run-123",
        parent_model_id="model-fc-bot-v1",
        generation_config={"temperature": 0.7, "seed": 7},
    )
    assert resp.status_code == 201
    body = resp.get_json()
    assert body["model_type"] == "lora"
    assert body["adapter_uri"] == "runs:/run-123/adapter"
    assert body["adapter_sha256"] == SHA256
    assert body["parent_model_id"] == "model-fc-bot-v1"
    assert body["generation_config"]["temperature"] == 0.7
    assert body["generation_config"]["seed"] == 7


@pytest.mark.parametrize("missing", ["name", "version", "base_model", "base_model_revision"])
def test_register_missing_field_returns_400(client, missing):
    resp = _register(client, **{missing: None})
    assert resp.status_code == 400
    assert missing in resp.get_json()["error"]


@pytest.mark.parametrize("field,value", [
    ("name", "../evil"),
    ("version", "v1/../../x"),
    ("base_model", "not-a-repo-id"),
    ("base_model_revision", "main"),         # branches move; only a commit hash pins the weights
    ("base_model_revision", "A" * 40),       # git hashes are lowercase hex
    ("adapter_sha256", "xyz"),
])
def test_register_invalid_field_returns_400(client, field, value):
    overrides = {field: value}
    if field == "adapter_sha256":
        overrides["adapter_uri"] = "runs:/run-123/adapter"
    resp = _register(client, **overrides)
    assert resp.status_code == 400
    body = resp.get_json()
    assert body["error"] == "Invalid request body"
    assert any(d.startswith(field) for d in body["details"])


@pytest.mark.parametrize("overrides", [
    {"adapter_uri": "runs:/run-123/adapter"},
    {"adapter_sha256": SHA256},
])
def test_register_adapter_uri_and_hash_required_together(client, overrides):
    resp = _register(client, **overrides)
    assert resp.status_code == 400


def test_register_rejects_out_of_range_generation_config(client):
    resp = _register(client, generation_config={"temperature": 5})
    assert resp.status_code == 400


def test_register_wrong_kind_returns_400(client):
    resp = _register(client, kind="classical")
    assert resp.status_code == 400


def test_register_duplicate_returns_409(client):
    assert _register(client).status_code == 201
    resp = _register(client)
    assert resp.status_code == 409
    assert resp.get_json()["model_id"] == "model-fc-bot-v1"


def test_register_unknown_parent_returns_400(client):
    resp = _register(client, parent_model_id="model-nope-v1")
    assert resp.status_code == 400
    assert "Unknown parent" in resp.get_json()["error"]


@pytest.mark.parametrize("overrides", [
    {"base_model": "Qwen/Qwen2.5-0.5B-Instruct"},
    {"base_model_revision": "c" * 40},
])
def test_register_parent_on_different_base_returns_400(client, overrides):
    assert _register(client).status_code == 201
    resp = _register(client, version="v2", parent_model_id="model-fc-bot-v1", **overrides)
    assert resp.status_code == 400
    assert "same base_model" in resp.get_json()["error"]


def test_list_models_shows_kind_specific_fields(client):
    _register(client)
    models = client.get("/models").get_json()
    assert len(models) == 1
    llm = models[0]
    assert llm["kind"] == "llm"
    assert llm["base_model_revision"] == REVISION
    assert "feature_names" not in llm


def test_classical_prediction_endpoint_refuses_llm(client):
    _register(client)
    resp = client.post("/predictions/model/fc-bot/version/v1", json={"features": {"a": 1}})
    assert resp.status_code == 400
    assert "/generations" in resp.get_json()["error"]


def test_classical_prediction_task_refuses_llm(app, client):
    _register(client)
    with app.app_context():
        result, error = _query_model("fc-bot", "v1")
    assert result is None
    assert error["status_code"] == 400
