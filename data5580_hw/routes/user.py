from flask import Blueprint

from data5580_hw.controllers.user import user_controller

user = Blueprint("user", __name__)


@user.route("/user/<user_id>", methods=["GET"])
def get_user_by_id(user_id: str):
    return user_controller.get_user_by_id(user_id)


@user.route("/user", methods=["POST"])
def create_user():
    return user_controller.create_user()


@user.route("/user/<user_id>", methods=["PATCH"])
def update_user(user_id: str):
    return user_controller.update_user(user_id=user_id)
