from flask import Blueprint

from data5580_hw.controllers.prediction import prediction_controller



prediction = Blueprint("prediction", __name__)


@prediction.route("/model/<model_id>/version/<version>", methods=["POST"])
def predict(model_name: str, version: str):

    response = prediction_controller.create_prediction()

    return response

@