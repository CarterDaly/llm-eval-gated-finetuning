
# Standardlibrary
import uuid
import logging
import pydantic

# Installed
from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.exc import IntegrityError
from pydantic import BaseModel, Field, EmailStr

# Local
from data5580_hw.services.databases.database_client import db, init_db
from data5580_hw.controllers.user import user_controller

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)

# def get_id():
#     return str(uuid.uuid4().hex)

# class User(BaseModel):
#     name: str
#     email: EmailStr

#     id: str = Field(default_factory=get_id)

def create_app():
    app = Flask(__name__)

    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///data.db"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    init_db(app)


    @app.route("/", methods=["GET"])
    def home():
        return jsonify({"message": "Hello, Flask!"})

    # @app.route("/add", methods=["POST"])
    # def add():
    #     data = request.get_json()
    #     a = data.get("a")
    #     b = data.get("b")

    #     if a is None or b is None:
    #         return jsonify({"error": "Missing values"}), 400

    #     return jsonify({"result": a + b})

    @app.route("/user/<user_id>", methods=["GET"])
    def get_user_by_id(user_id: str):

        return user_controller.get_user_by_id(user_id)

    @app.route("/user", methods=["POST"])
    def create_user():

        user = User.model_validate(request.get_json(force=True))

        user_sql = UserSQL(name = user.name, email=user.email, id=user.id)

        db.session.add(user_sql)

        try:
            db.session.commit()
        except IntegrityError:
            return jsonify({"error": "Email already exists"}), 400

        logging.info(f"User created {user.id} for {user.email}")

        return user.model_dump_json(), 200

    return app




if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)