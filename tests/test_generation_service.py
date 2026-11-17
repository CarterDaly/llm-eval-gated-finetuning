"""Tests for the vLLM backend and backend selection. HTTP is always faked."""
import pytest
import requests

import gated_finetune.services.generation_service as gs
from gated_finetune.services.database.prediction import ModelSQL
from gated_finetune.services.errors import GenerationBackendError

PARAMS = {"temperature": 0.0, "top_p": 1.0, "max_tokens": 64, "seed": None}
MESSAGES = [{"role": "user", "content": "Weather in Paris?"}]
TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]


def _model(adapter_uri=None):
    return ModelSQL(
        id="model-fc-bot-v2",
        kind="llm",
        base_model="Qwen/Qwen2.5-1.5B-Instruct",
        base_model_revision="a" * 40,
        adapter_uri=adapter_uri,
    )


class _FakeResponse:
    def __init__(self, status_code=200, json_body=None, text=""):
        self.status_code = status_code
        self._json = json_body
        self.text = text

    def json(self):
        if self._json is None:
            raise ValueError("no json")
        return self._json


def _completion(content='<tool_call>\n{"name": "get_weather"}\n</tool_call>', model="Qwen/Qwen2.5-1.5B-Instruct"):
    return {
        "model": model,
        "choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 40, "completion_tokens": 12},
    }


@pytest.fixture
def fake_post(monkeypatch):
    calls = []
    state = {"response": _FakeResponse(json_body=_completion())}

    def post(url, json, headers, timeout):
        calls.append({"url": url, "json": json, "headers": headers, "timeout": timeout})
        if isinstance(state["response"], Exception):
            raise state["response"]
        return state["response"]

    monkeypatch.setattr(gs.requests, "post", post)
    return calls, state


def test_base_model_request_and_result(fake_post):
    calls, _ = fake_post
    result = gs.VLLMBackend().generate(_model(), MESSAGES, None, PARAMS)

    call = calls[0]
    assert call["url"] == "http://localhost:8000/v1/chat/completions"
    assert call["json"]["model"] == "Qwen/Qwen2.5-1.5B-Instruct"
    assert call["json"]["messages"] == MESSAGES
    assert call["json"]["temperature"] == 0.0
    assert call["json"]["max_tokens"] == 64
    assert "seed" not in call["json"]
    assert "tools" not in call["json"]
    assert call["headers"] == {}

    assert result.output_text.startswith("<tool_call>")
    assert result.finish_reason == "stop"
    assert (result.prompt_tokens, result.completion_tokens) == (40, 12)
    assert result.served_model == "Qwen/Qwen2.5-1.5B-Instruct"


def test_adapter_model_is_requested_by_model_id(fake_post):
    calls, _ = fake_post
    gs.VLLMBackend().generate(_model(adapter_uri="runs:/r/adapter"), MESSAGES, None, PARAMS)
    assert calls[0]["json"]["model"] == "model-fc-bot-v2"


def test_tools_are_sent_without_server_side_parsing(fake_post):
    calls, _ = fake_post
    gs.VLLMBackend().generate(_model(), MESSAGES, TOOLS, PARAMS)
    assert calls[0]["json"]["tools"] == TOOLS
    assert calls[0]["json"]["tool_choice"] == "none"


def test_seed_and_config_from_env(fake_post, monkeypatch):
    calls, _ = fake_post
    monkeypatch.setenv("VLLM_BASE_URL", "http://gpu-box:8000/")
    monkeypatch.setenv("VLLM_API_KEY", "secret")
    monkeypatch.setenv("VLLM_TIMEOUT", "5")
    gs.VLLMBackend().generate(_model(), MESSAGES, None, {**PARAMS, "seed": 7})
    assert calls[0]["url"] == "http://gpu-box:8000/v1/chat/completions"
    assert calls[0]["headers"] == {"Authorization": "Bearer secret"}
    assert calls[0]["timeout"] == 5.0
    assert calls[0]["json"]["seed"] == 7


def test_null_content_becomes_empty_string(fake_post):
    _, state = fake_post
    state["response"] = _FakeResponse(json_body=_completion(content=None))
    assert gs.VLLMBackend().generate(_model(), MESSAGES, None, PARAMS).output_text == ""


@pytest.mark.parametrize("response,match", [
    (requests.ConnectionError("refused"), "request failed: ConnectionError"),
    (requests.Timeout("slow"), "request failed: Timeout"),
    (_FakeResponse(status_code=404, text='{"message": "model not found"}'), "HTTP 404: .*model not found"),
    (_FakeResponse(status_code=200, json_body=None), "unexpected response"),
    (_FakeResponse(status_code=200, json_body={"choices": []}), "unexpected response"),
])
def test_failures_raise_backend_error(fake_post, response, match):
    _, state = fake_post
    state["response"] = response
    with pytest.raises(GenerationBackendError, match=match):
        gs.VLLMBackend().generate(_model(), MESSAGES, None, PARAMS)


def test_service_defaults_to_vllm(fake_post):
    service = gs.GenerationService()
    assert service.backend_name() == "vllm"
    assert service.generate(_model(), MESSAGES, None, PARAMS).output_text


def test_service_unknown_backend_raises(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "nope")
    with pytest.raises(GenerationBackendError, match="Unknown LLM_BACKEND 'nope'"):
        gs.GenerationService().generate(_model(), MESSAGES, None, PARAMS)
