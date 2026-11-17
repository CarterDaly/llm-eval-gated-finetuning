import logging
import os
import time
from datetime import datetime, timezone
from typing import Optional

from flask import jsonify

from gated_finetune.controllers.common import parse_body
from gated_finetune.models.base import get_id
from gated_finetune.models.llm import GenerationParams, GenerationRequest
from gated_finetune.services.database.database_client import db
from gated_finetune.services.database.generation import GenerationSQL
from gated_finetune.services.database.prediction import ModelSQL
from gated_finetune.services.errors import GenerationBackendError
from gated_finetune.services.generation_service import generation_service

logger = logging.getLogger(__name__)


def _cost_rate() -> Optional[float]:
    """Configured USD per 1k tokens (prompt + completion), or None if unset."""
    raw = os.environ.get("LLM_COST_PER_1K_TOKENS", "").strip()
    return float(raw) if raw else None


def _cost_usd(rate: Optional[float], prompt_tokens: Optional[int], completion_tokens: Optional[int]):
    if rate is None or prompt_tokens is None or completion_tokens is None:
        return None
    return (prompt_tokens + completion_tokens) / 1000 * rate


def _resolve_params(model_sql: ModelSQL, body: GenerationRequest) -> dict:
    """The model's generation_config, overridden by the fields the request set."""
    params = GenerationParams.model_validate(model_sql.generation_config or {}).model_dump()
    if body.params is not None:
        params.update(body.params.model_dump(exclude_unset=True))
    return params


def _generation_json(row: GenerationSQL) -> dict:
    return {
        "generation_id": row.id,
        "model_id": row.model_id,
        "output_text": row.output_text,
        "finish_reason": row.finish_reason,
        "error": row.error,
        "backend": row.backend,
        "served_model": row.served_model,
        "params": row.params,
        "usage": {"prompt_tokens": row.prompt_tokens, "completion_tokens": row.completion_tokens},
        "latency_ms": row.latency_ms,
        "cost_usd": row.cost_usd,
        "source": row.source,
        "example_id": row.example_id,
        "tags": row.tags,
        "created": row.created.isoformat(),
    }


class GenerationController:

    @staticmethod
    def create_generation(model_name: str, model_version: str):
        body, error = parse_body(GenerationRequest)
        if error:
            return error

        model_sql = (
            db.session.query(ModelSQL)
            .filter(ModelSQL.model_name == model_name, ModelSQL.model_version == model_version)
            .first()
        )
        if model_sql is None:
            return jsonify({"error": "Model not found"}), 404
        if model_sql.kind != "llm":
            return jsonify({"error": "This model is not an LLM; send requests to /predictions instead"}), 400

        messages = [m.model_dump() for m in body.messages]
        tools = [t.model_dump(exclude_none=True) for t in body.tools] if body.tools else None
        params = _resolve_params(model_sql, body)
        rate = _cost_rate()

        row = GenerationSQL(
            id=get_id(),
            model_id=model_sql.id,
            messages=messages,
            tools=tools,
            params=params,
            backend=generation_service.backend_name(),
            cost_per_1k_tokens=rate,
            source=body.source,
            example_id=body.example_id,
            tags=body.tags,
            created=datetime.now(timezone.utc),
        )

        # Latency covers the backend call only (network + queueing + generation),
        # not request parsing or the database write.
        started = time.perf_counter()
        try:
            result = generation_service.generate(model_sql, messages, tools, params)
        except GenerationBackendError as exc:
            row.latency_ms = (time.perf_counter() - started) * 1000
            row.error = str(exc)
            logger.warning("generation_backend_failed model=%s error=%s", model_sql.id, exc)
            status = 502
        else:
            row.latency_ms = (time.perf_counter() - started) * 1000
            row.output_text = result.output_text
            row.finish_reason = result.finish_reason
            row.served_model = result.served_model
            row.prompt_tokens = result.prompt_tokens
            row.completion_tokens = result.completion_tokens
            row.cost_usd = _cost_usd(rate, result.prompt_tokens, result.completion_tokens)
            status = 201

        db.session.add(row)
        try:
            db.session.commit()
        except Exception:
            logger.exception("generation_persist_failed")
            db.session.rollback()
            return jsonify({"error": "Failed to store generation"}), 500

        return jsonify(_generation_json(row)), status

    @staticmethod
    def get_generation(generation_id: str):
        row = db.session.get(GenerationSQL, generation_id)
        if row is None:
            return jsonify({"error": "Generation not found"}), 404
        data = _generation_json(row)
        data.update({"messages": row.messages, "tools": row.tools})
        return jsonify(data), 200


generation_controller = GenerationController()
