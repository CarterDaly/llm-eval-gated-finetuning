import logging

from celery.result import AsyncResult
from flask import jsonify, request

from gated_finetune.celery_app import celery
from gated_finetune.controllers.common import parse_body
from gated_finetune.models.prediction import PredictionRequest
from gated_finetune.models.task import TaskStatus
from gated_finetune.tasks.prediction_tasks import run_long_prediction, run_short_prediction

logger = logging.getLogger(__name__)

_TASK_WAIT_TIMEOUT = 30


def _result_response(async_result: AsyncResult):
    """Build a TaskStatus dict from an AsyncResult, handling all states."""
    status = async_result.status
    result = None
    error = None

    if status == "SUCCESS":
        raw = async_result.result
        if isinstance(raw, dict) and "error" in raw:
            error = raw["error"]
            status = "FAILURE"
        else:
            result = raw
    elif status == "FAILURE":
        error = str(async_result.result)

    return TaskStatus(
        task_id=async_result.id,
        status=status,
        result=result,
        error=error,
    )


class TaskController:

    @staticmethod
    def submit_prediction(model_name: str, model_version: str):
        """Submit a prediction task (short or long) based on query params.

        Query params:
            task_type: "short" | "long"  (default: "long")
            wait:      "true"            (if present, block until complete)
        """
        prediction_request, error = parse_body(PredictionRequest)
        if error:
            return error

        task_type = request.args.get("task_type", "long").lower()
        wait = request.args.get("wait", "false").lower() == "true"

        kwargs = {
            "model_name": model_name,
            "model_version": model_version,
            "features": prediction_request.features,
            "tags": prediction_request.tags,
            "actual": prediction_request.actual,
            "threshold": prediction_request.threshold,
        }

        if task_type == "short":
            async_result = run_short_prediction.delay(**kwargs)
        else:
            kwargs["umap_config"] = prediction_request.umap.model_dump() if prediction_request.umap else None
            async_result = run_long_prediction.delay(**kwargs)

        if wait:
            try:
                async_result.get(timeout=_TASK_WAIT_TIMEOUT, propagate=False)
            except Exception as exc:
                logger.exception("task_wait_failed")
                return jsonify({"error": str(exc)}), 500

        task_status = _result_response(async_result)
        http_status = 201 if task_status.status in {"SUCCESS", "PENDING", "STARTED"} else 500
        return jsonify(task_status.model_dump()), http_status

    @staticmethod
    def get_task_status(task_id: str):
        """Query Celery for the status of a submitted task."""
        async_result = celery.AsyncResult(task_id)
        task_status = _result_response(async_result)
        return jsonify(task_status.model_dump()), 200


task_controller = TaskController()
