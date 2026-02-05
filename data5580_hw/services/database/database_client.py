from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase


class BaseModel(DeclarativeBase):

    pass

db = SQLAlchemy(model_class=BaseModel)


def init_db(app) -> None:

    db.init_app(app)

    with app.app_context():
        from data5580_hw.services.database import user_model  # noqa: F401
        try:
            db.drop_all()
        except:
            pass
        finally:
            db.create_all()
            db.session.commit()
