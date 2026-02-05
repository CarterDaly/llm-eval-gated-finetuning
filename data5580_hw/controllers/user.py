import logging

from flask import jsonify, request

from sqlalchemy.exc import IntegrityError

from data5580_hw.services.database.database_client import db
from data5580_hw.services.database.user_model import UserSQL
from pydantic import ValidationError

from data5580_hw.models.user import User, UserUpdate



logger = logging.getLogger(__name__)


def _missing_fields_message(fields: list[str]) -> str:
    return "Missing required fields: {}".format(", ".join(sorted(fields)))


def _validation_error_response(error: ValidationError):
    errors = error.errors()
    missing = [err["loc"][0] for err in errors if err.get("type") == "missing"]
    if missing:
        return jsonify({"error": _missing_fields_message(missing)}), 400
    if any("email" in err.get("loc", []) for err in errors):
        return jsonify({"error": "Invalid email format"}), 400
    return jsonify({"error": "Invalid user details"}), 400


def _user_payload(user_sql: UserSQL) -> dict:
    return {"id": user_sql.id, "name": user_sql.name, "email": user_sql.email}


class UserController:

    @staticmethod
    def create_user() -> tuple[str, int]:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({"error": _missing_fields_message(["name", "email"])}), 400
        data.pop("id", None)

        try:
            user = User.model_validate(data)
        except ValidationError as exc:
            return _validation_error_response(exc)

        existing_user = db.session.query(UserSQL).filter(UserSQL.email == user.email).first()
        if existing_user:
            return jsonify({"error": "Email is already in use"}), 400

        user_sql = UserSQL(name=user.name, email=user.email, id=user.id)

        db.session.add(user_sql)

        try:
            db.session.commit()
        except IntegrityError:
            db.session.rollback()
            return jsonify({"error": "Email is already in use"}), 400

        logger.info("user created, %s for %s", user.id, user.email)

        return jsonify({"message": "User created", "user": _user_payload(user_sql)}), 201

    @staticmethod
    def get_user_by_id(user_id: str) -> tuple[str, int]:
        logger.debug("Got user_id %s", user_id)

        user_sql = db.session.query(UserSQL).filter(UserSQL.id == user_id).first()
        if user_sql is None:
            return jsonify({"error": "User not found"}), 404

        logger.info("user fetched, %s for %s", user_sql.id, user_sql.email)

        return jsonify({"user": _user_payload(user_sql)}), 200

    @staticmethod
    def update_user(user_id: str) -> tuple[str, int]:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({"error": _missing_fields_message(["name", "email"])}), 400

        try:
            update = UserUpdate.model_validate(data)
        except ValidationError as exc:
            return _validation_error_response(exc)

        if update.name is None and update.email is None:
            return jsonify({"error": _missing_fields_message(["name", "email"])}), 400

        user_sql = db.session.query(UserSQL).filter(UserSQL.id == user_id).first()
        if user_sql is None:
            return jsonify({"error": "User not found"}), 404

        if update.email and update.email != user_sql.email:
            existing_user = (
                db.session.query(UserSQL)
                .filter(UserSQL.email == update.email, UserSQL.id != user_id)
                .first()
            )
            if existing_user:
                return jsonify({"error": "Email is already in use"}), 400
            user_sql.email = update.email

        if update.name is not None:
            user_sql.name = update.name

        try:
            db.session.commit()
        except IntegrityError:
            db.session.rollback()
            return jsonify({"error": "Email is already in use"}), 400

        return jsonify({"message": "User updated", "user": _user_payload(user_sql)}), 200

    @staticmethod
    def replace_user(user_id: str) -> tuple[str, int]:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({"error": _missing_fields_message(["name", "email"])}), 400
        data.pop("id", None)

        try:
            user = User.model_validate(data)
        except ValidationError as exc:
            return _validation_error_response(exc)

        user_sql = db.session.query(UserSQL).filter(UserSQL.id == user_id).first()
        if user_sql is None:
            return jsonify({"error": "User not found"}), 404

        if user.email != user_sql.email:
            existing_user = (
                db.session.query(UserSQL)
                .filter(UserSQL.email == user.email, UserSQL.id != user_id)
                .first()
            )
            if existing_user:
                return jsonify({"error": "Email is already in use"}), 400

        user_sql.name = user.name
        user_sql.email = user.email

        try:
            db.session.commit()
        except IntegrityError:
            db.session.rollback()
            return jsonify({"error": "Email is already in use"}), 400

        payload = _user_payload(user_sql)
        return (
            jsonify({"message": "User details updated successfully.", "user": payload}),
            200,
        )

    @staticmethod
    def delete_user(user_id: str) -> tuple[str, int]:
        user_sql = db.session.query(UserSQL).filter(UserSQL.id == user_id).first()
        if user_sql is None:
            return jsonify({"error": "User not found"}), 404

        db.session.delete(user_sql)
        db.session.commit()

        return jsonify({"message": "User deleted"}), 200

    @staticmethod
    def list_users() -> tuple[str, int]:
        users = db.session.query(UserSQL).all()
        return jsonify({"users": [_user_payload(user_sql) for user_sql in users]}), 200





user_controller = UserController()
