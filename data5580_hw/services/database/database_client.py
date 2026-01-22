from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase


class BaseModel(DeclarativeBase):

    pass

db = SQLAlchemy(model_class=BaseModel)


def init_db(app) -> None:

    db.init_app(app)

    with app.app_context():
        try:
            db.drop_all()
        except:
            pass
        finally:
            db.create_all()
            db.session.commit()
