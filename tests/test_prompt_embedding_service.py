"""Tests for the prompt embedding service. The fastembed model is always faked."""
import numpy as np
import pytest

from gated_finetune.services.errors import PromptEmbeddingError
from gated_finetune.services.prompt_embedding_service import PromptEmbeddingService, prompt_text


class _FakeModel:
    def __init__(self, fail=False):
        self.fail = fail
        self.texts = []

    def embed(self, texts):
        if self.fail:
            raise RuntimeError("onnx exploded")
        self.texts.extend(texts)
        return iter([np.array([0.6, 0.8], dtype=np.float32) for _ in texts])


def _service(model):
    service = PromptEmbeddingService()
    service._model = model
    return service


def test_prompt_text_keeps_only_user_messages_in_order():
    messages = [
        {"role": "system", "content": "You are a tool-calling assistant."},
        {"role": "user", "content": "Weather in Paris?"},
        {"role": "assistant", "content": "<tool_call>...</tool_call>"},
        {"role": "tool", "content": "{\"temp\": 12}"},
        {"role": "user", "content": "And in Rome?"},
    ]
    assert prompt_text(messages) == "Weather in Paris?\nAnd in Rome?"


def test_embed_returns_plain_floats():
    model = _FakeModel()
    vector = _service(model).embed([{"role": "user", "content": "hi"}])
    assert vector == pytest.approx([0.6, 0.8])
    assert all(type(x) is float for x in vector)  # JSON-serializable, not numpy
    assert model.texts == ["hi"]


@pytest.mark.parametrize("messages", [
    [{"role": "system", "content": "only a system prompt"}],
    [{"role": "user", "content": "   "}],
])
def test_embed_without_user_text_raises(messages):
    with pytest.raises(PromptEmbeddingError, match="no user message"):
        _service(_FakeModel()).embed(messages)


def test_model_failure_raises_prompt_embedding_error():
    with pytest.raises(PromptEmbeddingError, match="Failed to compute"):
        _service(_FakeModel(fail=True)).embed([{"role": "user", "content": "hi"}])


def test_model_id_records_model_and_fastembed_version():
    model_id = PromptEmbeddingService().model_id
    assert model_id.startswith("BAAI/bge-small-en-v1.5 (fastembed ")


def test_model_is_not_loaded_until_first_use():
    assert PromptEmbeddingService()._model is None
