from sqlalchemy import UniqueConstraint

from data5580_hw.services.database.database_client import db


class UserSQL(db.Model):
    __tablename__ = "users"
    id = db.Column(db.String, primary_key=True)
    name = db.Column(db.String)
    email = db.Column(db.String)

    UniqueConstraint(email)