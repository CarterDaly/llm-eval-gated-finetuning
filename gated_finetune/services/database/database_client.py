from flask_migrate import Migrate
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase


class BaseModel(DeclarativeBase):

    pass

db = SQLAlchemy(model_class=BaseModel)
migrate = Migrate()


def init_db(app) -> None:
    db.init_app(app)
    migrate.init_app(app, db)

    with app.app_context():
        from gated_finetune.services.database import user_model  # noqa: F401
        from gated_finetune.services.database import prediction  # noqa: F401

        database_url = app.config.get("SQLALCHEMY_DATABASE_URI", "")
        if database_url.startswith("sqlite"):
            # Dev: create tables if missing, but never drop existing data
            db.create_all()
            db.session.commit()
        else:
            # Production (Postgres etc.): create tables only if they don't exist yet.
            # Schema evolution is managed through Flask-Migrate / Alembic.
            db.create_all()
            db.session.commit()
