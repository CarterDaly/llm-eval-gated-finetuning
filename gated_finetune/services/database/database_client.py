from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import MetaData
from sqlalchemy.orm import DeclarativeBase


class BaseModel(DeclarativeBase):
    # Deterministic constraint names, so later migrations can alter or drop
    # constraints by name (SQLite otherwise leaves them unnamed).
    metadata = MetaData(naming_convention={
        "ix": "ix_%(column_0_label)s",
        "uq": "uq_%(table_name)s_%(column_0_name)s",
        "ck": "ck_%(table_name)s_%(constraint_name)s",
        "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
        "pk": "pk_%(table_name)s",
    })


db = SQLAlchemy(model_class=BaseModel)
# render_as_batch: SQLite can't ALTER most things in place, so Alembic rebuilds
# the table instead. It is a no-op on Postgres.
migrate = Migrate(render_as_batch=True)


def init_db(app) -> None:
    """Bind SQLAlchemy and Flask-Migrate to the app.

    The schema is owned by the Alembic migrations in migrations/ and is applied
    with `flask db upgrade`, never at app startup: several gunicorn workers
    creating tables at once race each other, and tables created outside Alembic
    make the first `upgrade` fail.
    """
    db.init_app(app)
    migrate.init_app(app, db)

    # Import the models so their tables are registered on db.metadata, which
    # Alembic compares against when autogenerating a migration.
    from gated_finetune.services.database import prediction  # noqa: F401
