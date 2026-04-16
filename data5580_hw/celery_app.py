import os
import logging

from celery import Celery
from celery.signals import worker_ready, worker_shutdown, task_failure, task_retry

logger = logging.getLogger(__name__)

celery = Celery("data5580_hw")


def init_celery(flask_app):
    """Configure Celery with Redis broker/backend and bind Flask app context."""
    celery.conf.update(
        broker_url=os.environ.get("CELERY_BROKER_URL", "redis://localhost:6379/0"),
        result_backend=os.environ.get("CELERY_RESULT_BACKEND", "redis://localhost:6379/0"),
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        task_track_started=True,
        task_time_limit=300,
        task_soft_time_limit=270,
        worker_max_tasks_per_child=100,
    )
    celery.flask_app = flask_app
    return celery


@worker_ready.connect
def on_worker_ready(**kwargs):
    logger.info("celery_worker_ready")


@worker_shutdown.connect
def on_worker_shutdown(**kwargs):
    logger.info("celery_worker_shutdown")


@task_failure.connect
def on_task_failure(task_id, exception, **kwargs):
    logger.error("celery_task_failed task_id=%s error=%s", task_id, str(exception))


@task_retry.connect
def on_task_retry(request, reason, **kwargs):
    logger.warning("celery_task_retry task_id=%s reason=%s", request.id, str(reason))
