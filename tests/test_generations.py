"""Tests for POST /generations/... and GET /generations/<id>. The backend is faked."""
import pytest

import gated_finetune.controllers.generation as generation_controller_module
from gated_finetune.app import create_app
from gated_finetune.services.database.database_client import db
from gated_finetune.services.database.generation import GenerationSQL
from gated_finetune.services.errors import GenerationBackendError
from gated_finetune.services.generation_service import GenerationResult

URL = "/generations/model/fc-bot/version/v1"
MESSAGES = [{"role": "user", "content": "Weather in Paris?"}]
TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]


@pytest.fixture
def app():
    app = create_app()
    app.config["TESTING"] = True
    return app


@pytest.fixture
def client(app):
    with app.test_client() as c:
        yield c


@pytest.fixture
def llm(client):
    resp = client.post("/models", json={
        "kind": "llm",
        "name": "fc-bot",
        "version": "v1",
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "base_model_revision": "a" * 40,
        "generation_config": {"max_tokens": 128},
    })
    assert resp.status_code == 201
    return resp.get_json()


class _FakeService:
    """Stands in for generation_service; records calls and returns or raises."""

    def __init__(self):
        self.calls = []
        self.outcome = GenerationResult(
            output_text="<tool_call>{}</tool_call>",
            finish_reason="stop",
            prompt_tokens=300,
            completion_tokens=200,
            served_model="Qwen/Qwen2.5-1.5B-Instruct",
        )

    def backend_name(self):
        return "vllm"

    def generate(self, model, messages, tools, params):
        self.calls.append({"model_id": model.id, "messages": messages, "tools": tools, "params": params})
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


@pytest.fixture
def fake_service(monkeypatch):
    service = _FakeService()
    monkeypatch.setattr(generation_controller_module, "generation_service", service)
    return service


class _FakeTask:
    """Stands in for the embedding task; records what was queued."""

    def __init__(self):
        self.queued = []
        self.fail = False

    def apply_async(self, args, retry):
        if self.fail:
            raise ConnectionError("redis is down")
        self.queued.append({"args": args, "retry": retry})


@pytest.fixture(autouse=True)
def fake_embedding_task(monkeypatch):
    task = _FakeTask()
    monkeypatch.setattr(generation_controller_module, "embed_generation_prompt", task)
    return task


@pytest.fixture
def fake_clock(monkeypatch):
    """perf_counter returns 10.0 then 10.25, so the backend call takes 250 ms."""
    ticks = iter([10.0, 10.25])
    monkeypatch.setattr(generation_controller_module.time, "perf_counter", lambda: next(ticks))


def _stored(app, generation_id):
    with app.app_context():
        return db.session.get(GenerationSQL, generation_id)


def test_generation_is_returned_and_logged(app, client, llm, fake_service, fake_clock, monkeypatch):
    monkeypatch.setenv("LLM_COST_PER_1K_TOKENS", "0.002")
    resp = client.post(URL, json={"messages": MESSAGES, "tools": TOOLS, "tags": {"slice": "weather"}})
    assert resp.status_code == 201
    body = resp.get_json()
    assert body["output_text"] == "<tool_call>{}</tool_call>"
    assert body["usage"] == {"prompt_tokens": 300, "completion_tokens": 200}
    assert body["latency_ms"] == pytest.approx(250.0)
    assert body["cost_usd"] == pytest.approx(0.001)  # 500 tokens at $0.002 / 1k

    row = _stored(app, body["generation_id"])
    assert row.model_id == llm["model_id"]
    assert row.messages == MESSAGES
    assert row.tools == TOOLS
    assert row.backend == "vllm"
    assert row.served_model == "Qwen/Qwen2.5-1.5B-Instruct"
    assert row.finish_reason == "stop"
    assert row.cost_per_1k_tokens == 0.002
    assert row.source == "live"
    assert row.example_id is None
    assert row.tags == {"slice": "weather"}
    assert row.error is None


def test_params_merge_model_config_with_request_overrides(client, llm, fake_service):
    client.post(URL, json={"messages": MESSAGES, "params": {"temperature": 0.7}})
    # max_tokens from the model's generation_config, temperature from the request,
    # everything else from the GenerationParams defaults
    assert fake_service.calls[0]["params"] == {"temperature": 0.7, "top_p": 1.0, "max_tokens": 128, "seed": None}


def test_cost_is_null_without_a_configured_rate(client, llm, fake_service):
    body = client.post(URL, json={"messages": MESSAGES}).get_json()
    assert body["cost_usd"] is None


def test_eval_traffic_stores_example_id(app, client, llm, fake_service):
    body = client.post(URL, json={"messages": MESSAGES, "source": "eval", "example_id": "golden-0042"}).get_json()
    row = _stored(app, body["generation_id"])
    assert (row.source, row.example_id) == ("eval", "golden-0042")


@pytest.mark.parametrize("source", ["eval", "replay"])
def test_dataset_traffic_without_example_id_returns_400(client, llm, fake_service, source):
    resp = client.post(URL, json={"messages": MESSAGES, "source": source})
    assert resp.status_code == 400
    assert fake_service.calls == []


def test_backend_failure_returns_502_and_is_logged(app, client, llm, fake_service, fake_clock):
    fake_service.outcome = GenerationBackendError("vLLM request failed: ConnectionError")
    resp = client.post(URL, json={"messages": MESSAGES})
    assert resp.status_code == 502
    body = resp.get_json()
    assert body["error"] == "vLLM request failed: ConnectionError"

    row = _stored(app, body["generation_id"])
    assert row.output_text is None
    assert row.error == "vLLM request failed: ConnectionError"
    assert row.latency_ms == pytest.approx(250.0)


@pytest.mark.parametrize("payload", [
    {},
    {"messages": []},
    {"messages": [{"role": "robot", "content": "hi"}]},
    {"messages": MESSAGES, "tools": [{"type": "function", "function": {"name": "bad name!"}}]},
    {"messages": MESSAGES, "tools": [{"type": "retrieval"}]},
    {"messages": MESSAGES, "params": {"max_tokens": 0}},
    {"messages": MESSAGES, "source": "prod"},
])
def test_invalid_request_returns_400(client, llm, fake_service, payload):
    resp = client.post(URL, json=payload)
    assert resp.status_code == 400
    assert fake_service.calls == []


def test_unknown_model_returns_404(client, fake_service):
    resp = client.post("/generations/model/nope/version/v1", json={"messages": MESSAGES})
    assert resp.status_code == 404


def test_classical_model_returns_400(app, client, fake_service):
    from gated_finetune.services.database.prediction import ModelSQL
    with app.app_context():
        db.session.add(ModelSQL(id="model-tab-1", kind="classical", model_type="tree", model_name="tab", model_version="1"))
        db.session.commit()
    resp = client.post("/generations/model/tab/version/1", json={"messages": MESSAGES})
    assert resp.status_code == 400
    assert "/predictions" in resp.get_json()["error"]


def test_get_generation_includes_request(client, llm, fake_service):
    created = client.post(URL, json={"messages": MESSAGES, "tools": TOOLS}).get_json()
    resp = client.get(f"/generations/{created['generation_id']}")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["messages"] == MESSAGES
    assert body["tools"] == TOOLS
    assert body["output_text"] == created["output_text"]


def test_get_unknown_generation_returns_404(client):
    assert client.get("/generations/does-not-exist").status_code == 404


def test_generation_requires_api_key_when_set(llm, fake_service, monkeypatch):
    # API_KEY is read when the app is created, so build a second app with it set
    monkeypatch.setenv("API_KEY", "k")
    authed_app = create_app()
    authed_app.config["TESTING"] = True
    with authed_app.test_client() as authed:
        assert authed.post(URL, json={"messages": MESSAGES}).status_code == 401
        assert authed.post(URL, json={"messages": MESSAGES}, headers={"X-API-Key": "k"}).status_code == 201


def test_prompt_embedding_is_queued_after_the_row_is_stored(client, llm, fake_service, fake_embedding_task):
    body = client.post(URL, json={"messages": MESSAGES}).get_json()
    assert fake_embedding_task.queued == [{"args": [body["generation_id"]], "retry": False}]


def test_failed_generations_are_embedded_too(client, llm, fake_service, fake_embedding_task):
    fake_service.outcome = GenerationBackendError("down")
    body = client.post(URL, json={"messages": MESSAGES}).get_json()
    assert fake_embedding_task.queued[0]["args"] == [body["generation_id"]]


def test_broker_outage_does_not_fail_the_request(app, client, llm, fake_service, fake_embedding_task):
    fake_embedding_task.fail = True
    resp = client.post(URL, json={"messages": MESSAGES})
    assert resp.status_code == 201
    assert _stored(app, resp.get_json()["generation_id"]).prompt_embedding is None


def test_invalid_request_queues_nothing(client, llm, fake_service, fake_embedding_task):
    client.post(URL, json={"messages": []})
    assert fake_embedding_task.queued == []


def test_get_generation_shows_embedding_fields(client, llm, fake_service):
    created = client.post(URL, json={"messages": MESSAGES}).get_json()
    body = client.get(f"/generations/{created['generation_id']}").get_json()
    assert body["prompt_embedding"] is None  # the (faked) task hasn't run
    assert body["embedding_model"] is None
