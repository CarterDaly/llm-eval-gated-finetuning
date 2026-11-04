from flask import Blueprint

from gated_finetune.controllers.prediction import prediction_controller


prediction = Blueprint("prediction", __name__)


@prediction.route("/predictions/model/<model_name>/version/<model_version>", methods=["POST"])
def create_prediction(model_name: str, model_version: str):
    return prediction_controller.create_prediction(model_name, model_version)


@prediction.route("/predictions/<prediction_id>", methods=["GET"])
def get_prediction_by_id(prediction_id: str):
    return prediction_controller.get_prediction_by_id(prediction_id)


@prediction.route("/predictions/<prediction_id>/explainer", methods=["GET"])
def get_prediction_explainer(prediction_id: str):
    return prediction_controller.get_prediction_explainer(prediction_id)


@prediction.route("/predictions/<prediction_id>/actual", methods=["PATCH"])
def update_actual(prediction_id: str):
    return prediction_controller.update_actual(prediction_id)


@prediction.route("/predictions/actuals", methods=["PATCH"])
def update_actuals_batch():
    return prediction_controller.update_actuals_batch()
