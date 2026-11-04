from flask import Blueprint

from gated_finetune.monitoring import render_metrics

metrics = Blueprint("metrics", __name__)


@metrics.route("/metrics", methods=["GET"])
def get_metrics():
    body, content_type = render_metrics()
    return body, 200, {"Content-Type": content_type}
