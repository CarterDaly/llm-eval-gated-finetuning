"""Tests for the background prompt-embedding task."""
from datetime import datetime, timezone

import pytest

import gated_finetune.services.prompt_embedding_service as pes
from gated_finetune.app import create_app
from gated_finetune.services.database.database_client import db
from gated_finetune.services.database.generation import GenerationSQL
from gated_finetune.services.database.prediction import ModelSQL
from gated_finetune.services.errors import PromptEmbeddingError
from gated_finetune.tasks.generation_tasks import embed_generation_prompt


class _FakeEmbeddingService:
    model_id = "fake-embedder"

    def __init__(self):
        self.calls = 0
        self.fail = False

    def embed(self, messages):
        self.calls += 1
        if self.fail:
            raise PromptEmbeddingError("Request has no user message to embed")
        return [0.1, 0.2, 0.3]


@pytest.fixture
def app():
    app = create_app()  # binds itself to the Celery instance the task uses
    app.config["TESTING"] = True
    return app


@pytest.fixture
def fake_embedder(monkeypatch):
    service = _FakeEmbeddingService()
    monkeypatch.setattr(pes, "prompt_embedding_service", service)
    return service


@pytest.fixture
def generation_id(app):
    with app.app_context():
        db.session.add(ModelSQL(id="model-fc-bot-v1", kind="llm", model_type="base", model_name="fc-bot", model_version="v1"))
        db.session.add(GenerationSQL(
            id="gen-1",
            model_id="model-fc-bot-v1",
            messages=[{"role": "user", "content": "Weather in Paris?"}],
            params={},
            backend="vllm",
            tags={},
            created=datetime.now(timezone.utc),
        ))
        db.session.commit()
    return "gen-1"


def _run(generation_id):
    # Calling the task runs its body in this process. (Not .apply(): that creates
    # and caches Celery's global result backend, which leaks into test_tasks.py.)
    return embed_generation_prompt(generation_id)


def _row(app, generation_id):
    with app.app_context():
        return db.session.get(GenerationSQL, generation_id)


def test_embedding_is_stored_with_model_id(app, fake_embedder, generation_id):
    result = _run(generation_id)
    assert result == {"generation_id": "gen-1", "status": "embedded", "dim": 3}
    row = _row(app, generation_id)
    assert row.prompt_embedding == [0.1, 0.2, 0.3]
    assert row.embedding_model == "fake-embedder"


def test_already_embedded_generation_is_not_recomputed(app, fake_embedder, generation_id):
    _run(generation_id)
    assert _run(generation_id)["status"] == "already_embedded"
    assert fake_embedder.calls == 1


def test_unknown_generation_is_reported(app, fake_embedder):
    assert _run("nope")["status"] == "not_found"
    assert fake_embedder.calls == 0


def test_embedding_failure_leaves_row_without_embedding(app, fake_embedder, generation_id):
    fake_embedder.fail = True
    result = _run(generation_id)
    assert result["status"] == "failed"
    assert "no user message" in result["error"]
    row = _row(app, generation_id)
    assert row.prompt_embedding is None
    assert row.embedding_model is None


def test_task_does_not_use_the_result_backend():
    # Queueing a task that stores results makes the API wait on the result
    # backend, so a Redis outage would stall every generation request.
    assert embed_generation_prompt.ignore_result is True
