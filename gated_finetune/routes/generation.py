from flask import Blueprint

from gated_finetune.controllers.generation import generation_controller

generation = Blueprint("generation", __name__)


@generation.route("/generations/model/<model_name>/version/<model_version>", methods=["POST"])
def create_generation(model_name: str, model_version: str):
    return generation_controller.create_generation(model_name, model_version)


@generation.route("/generations/<generation_id>", methods=["GET"])
def get_generation(generation_id: str):
    return generation_controller.get_generation(generation_id)
