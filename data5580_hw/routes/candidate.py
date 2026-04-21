from flask import Blueprint

from data5580_hw.controllers.candidate import candidate_controller

candidate = Blueprint("candidate", __name__)


@candidate.route("/candidates/<github_username>/score", methods=["POST"])
def score_candidate(github_username: str):
    return candidate_controller.score_candidate(github_username)
