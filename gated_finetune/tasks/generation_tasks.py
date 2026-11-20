import logging

from gated_finetune.celery_app import celery
from gated_finetune.services.errors import PromptEmbeddingError

logger = logging.getLogger(__name__)


# ignore_result: the outcome is written to the generation row, so there is nothing
# to store in the result backend. It also keeps the API from subscribing to the
# result backend when queueing the task, which blocks the request for about 20 s
# of reconnect attempts when Redis is down.
@celery.task(bind=True, name="generation_tasks.embed_generation_prompt", ignore_result=True)
def embed_generation_prompt(self, generation_id: str) -> dict:
    """Compute and store the prompt embedding of one generation.

    Runs after the API has responded, so the embedding model's time never counts
    toward the request's latency. Failed backend calls are embedded too: their
    prompts are still part of the traffic distribution.
    """
    from gated_finetune.services.database.database_client import db
    from gated_finetune.services.database.generation import GenerationSQL
    from gated_finetune.services.prompt_embedding_service import prompt_embedding_service

    with self.app.flask_app.app_context():
        row = db.session.get(GenerationSQL, generation_id)
        if row is None:
            logger.warning("prompt_embedding_missing_generation id=%s", generation_id)
            return {"generation_id": generation_id, "status": "not_found"}
        if row.prompt_embedding is not None:
            return {"generation_id": generation_id, "status": "already_embedded"}

        try:
            vector = prompt_embedding_service.embed(row.messages)
        except PromptEmbeddingError as exc:
            logger.warning("prompt_embedding_failed id=%s error=%s", generation_id, exc)
            return {"generation_id": generation_id, "status": "failed", "error": str(exc)}

        row.prompt_embedding = vector
        row.embedding_model = prompt_embedding_service.model_id
        db.session.commit()
        return {"generation_id": generation_id, "status": "embedded", "dim": len(vector)}
