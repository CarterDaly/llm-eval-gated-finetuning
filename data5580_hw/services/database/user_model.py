from data5580_hw.services.database.database_client import db


class UserSQL(db.Model):
    __tablename__ = "users"
    id = db.Column(db.String, primary_key=True)
    name = db.Column(db.String, nullable=False)
    email = db.Column(db.String, unique=True, nullable=False)
