"""Celery worker entry point: `celery -A gated_finetune.worker worker`.

Tasks use the SQLAlchemy session, so they run inside a Flask app context and the
worker builds the same app the API does (create_app binds it to the Celery
instance). Importing the tasks module is what registers the tasks; a worker
started from celery_app alone rejects every job as unregistered.
"""
from gated_finetune.app import create_app
from gated_finetune.celery_app import celery  # noqa: F401  (located by `-A`)
from gated_finetune.tasks import prediction_tasks  # noqa: F401  (registers tasks)

flask_app = create_app()
