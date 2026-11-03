import pytest

from sqlalchemy.orm import sessionmaker, Session

from gated_finetune.services.database.database_client import db
from gated_finetune.services.mlflow_service import mlflow_service


@pytest.fixture(autouse=True)
def _isolated_database(monkeypatch, tmp_path):
    """Point every app created during a test at its own throwaway SQLite file.

    init_db no longer drops tables on startup, so sharing instance/data.db
    would leak rows between tests (and pollute the dev database).
    """
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'test.db'}")


@pytest.fixture(autouse=True)
def _patch_log_embedding_once(monkeypatch):
    """Prevent tests from calling real MLflow for embedding logging.

    Any test that also patches log_shap_once separately still works fine —
    this fixture just ensures log_embedding_once is always a no-op.
    """
    monkeypatch.setattr(
        mlflow_service,
        "log_embedding_once",
        lambda **_kw: "embeddings/test/1/first_inference.json",
    )


@pytest.fixture
def db_instance(scope="session"):
    """
    Create a DB Instance
    """
    yield db


@pytest.fixture
def session(db_instance, scope="session"):
    """
    Create a Session, close after test session, uses `db_instance` fixture
    """
    session = Session(db_instance.engine)
    yield session
    session.close()


@pytest.fixture
def db_instance_empty(db_instance, session, scope="function"):
    """
    Create an Empty DB Instance, uses `db_instance` and `session` fixtures
    """
    # Clear DB before test function
    db_instance.delete_all_tasks(session=session)
    yield db_instance

    # Clear DB after test function
    db_instance.delete_all_tasks(session=session)