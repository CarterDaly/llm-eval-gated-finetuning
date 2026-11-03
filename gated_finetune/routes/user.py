from flask import Blueprint

from gated_finetune.controllers.user import user_controller

user = Blueprint("user", __name__)


@user.route("/users/<user_id>", methods=["GET"])
def get_user_by_id(user_id: str):
    return user_controller.get_user_by_id(user_id)


@user.route("/users", methods=["POST"])
def create_user():
    return user_controller.create_user()


@user.route("/users/<user_id>", methods=["PATCH"])
def update_user(user_id: str):
    return user_controller.update_user(user_id=user_id)


@user.route("/users/<user_id>", methods=["PUT"])
def replace_user(user_id: str):
    return user_controller.replace_user(user_id=user_id)


@user.route("/users/<user_id>", methods=["DELETE"])
def delete_user(user_id: str):
    return user_controller.delete_user(user_id=user_id)


@user.route("/users", methods=["GET"])
def list_users():
    return user_controller.list_users()
