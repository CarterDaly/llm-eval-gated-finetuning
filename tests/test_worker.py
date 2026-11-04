"""The worker entry point must give Celery both the tasks and a Flask app."""
import importlib


def test_worker_registers_tasks_and_binds_flask_app():
    worker = importlib.import_module("gated_finetune.worker")

    registered = set(worker.celery.tasks)
    assert "prediction_tasks.run_short_prediction" in registered
    assert "prediction_tasks.run_long_prediction" in registered
    assert worker.celery.flask_app is not None
