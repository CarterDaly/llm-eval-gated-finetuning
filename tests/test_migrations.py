"""The Alembic migrations are the only source of the database schema."""
import flask_migrate
from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext

from gated_finetune.app import create_app
from gated_finetune.services.database.database_client import db
from tests.conftest import MIGRATIONS_DIR


def test_migrations_match_models(tmp_path, monkeypatch):
    """Upgrading an empty DB to head must give exactly the schema the models
    declare. Changing a model without generating a migration fails here."""
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'fresh.db'}")
    app = create_app()
    with app.app_context():
        flask_migrate.upgrade(directory=MIGRATIONS_DIR)
        with db.engine.connect() as conn:
            diff = compare_metadata(MigrationContext.configure(conn), db.metadata)
    assert diff == []


def test_app_startup_does_not_create_tables(tmp_path, monkeypatch):
    """create_app must not touch the schema: concurrent gunicorn workers would
    race, and tables created outside Alembic break the first upgrade."""
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'untouched.db'}")
    app = create_app()
    with app.app_context():
        with db.engine.connect() as conn:
            assert db.inspect(conn).get_table_names() == []
