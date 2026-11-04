import shutil
from pathlib import Path

import flask_migrate
import pytest

from gated_finetune.app import create_app
from gated_finetune.services.explainer_service import explainer_service
from gated_finetune.services.mlflow_service import mlflow_service
from gated_finetune.services.model_service import model_service
from gated_finetune.services.umap_service import umap_service

# Env vars that point the app at a real external service or change request
# handling. A developer's shell (or a sourced .env) must not leak into tests:
# Celery, for example, reads CELERY_RESULT_BACKEND itself and would override
# the in-memory backend that test_tasks.py configures.
_EXTERNAL_ENV_VARS = (
    "API_KEY",
    "ARIZE_API_KEY",
    "ARIZE_SPACE_ID",
    "ARIZE_ENVIRONMENT",
    "CELERY_BROKER_URL",
    "CELERY_RESULT_BACKEND",
    "MLFLOW_TRACKING_URI",
    "OLLAMA_BASE_URL",
    "OLLAMA_MODEL",
    "MODELS_DIR",
)


@pytest.fixture(autouse=True)
def _hermetic_env(monkeypatch):
    for name in _EXTERNAL_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ARIZE_ENABLED", "false")


MIGRATIONS_DIR = str(Path(__file__).resolve().parents[1] / "migrations")


@pytest.fixture(scope="session")
def _migrated_template_db(tmp_path_factory):
    """Build the schema once, the way production does: `alembic upgrade head`.

    Tests then run against exactly the schema the migrations produce, not a
    separate create_all() schema that could drift from it.
    """
    path = tmp_path_factory.mktemp("schema") / "template.db"
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("DATABASE_URL", f"sqlite:///{path}")
        with create_app().app_context():
            flask_migrate.upgrade(directory=MIGRATIONS_DIR)
    return path


@pytest.fixture(autouse=True)
def _isolated_database(monkeypatch, tmp_path, _migrated_template_db):
    """Give every test its own copy of the migrated database."""
    db_path = tmp_path / "test.db"
    shutil.copy(_migrated_template_db, db_path)
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")


@pytest.fixture(autouse=True)
def _clear_service_caches():
    """Service singletons cache loaded models by (name, version); reset them so
    tests don't depend on each other's model names."""
    yield
    model_service._model_cache.clear()
    explainer_service._explainer_cache.clear()
    umap_service._umap_cache.clear()


@pytest.fixture(autouse=True)
def _patch_log_embedding_once(monkeypatch):
    """Prevent tests from calling real MLflow for embedding logging."""
    monkeypatch.setattr(
        mlflow_service,
        "log_embedding_once",
        lambda **_kw: "embeddings/test/1/first_inference.json",
    )
