from flask import Blueprint

from gated_finetune.controllers.model import model_controller

model = Blueprint("model", __name__)


@model.route("/models", methods=["POST"])
def ingest_model():
    return model_controller.ingest_model()


@model.route("/models", methods=["GET"])
def list_models():
    return model_controller.list_models()
