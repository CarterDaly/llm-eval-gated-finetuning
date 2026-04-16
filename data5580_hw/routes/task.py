from flask import Blueprint

from data5580_hw.controllers.task import task_controller

task = Blueprint("task", __name__)


@task.route(
    "/tasks/predictions/model/<model_name>/version/<model_version>",
    methods=["POST"],
)
def submit_prediction(model_name: str, model_version: str):
    return task_controller.submit_prediction(model_name, model_version)


@task.route("/tasks/<task_id>", methods=["GET"])
def get_task_status(task_id: str):
    return task_controller.get_task_status(task_id)
