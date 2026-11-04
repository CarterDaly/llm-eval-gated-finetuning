import hmac
import logging
import os

# Installed
from flask import Flask, jsonify, request

# Local
from gated_finetune.celery_app import init_celery
from gated_finetune.routes import init_blueprints
from gated_finetune.services.database.database_client import init_db
from gated_finetune.monitoring import init_metrics

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)

# Paths that are always public (health probe + Prometheus scrape)
_PUBLIC_PATHS = {"/", "/metrics"}


def _init_auth(app: Flask) -> None:
    """Register API key middleware when API_KEY env var is set.

    If API_KEY is not configured the middleware is skipped entirely so
    local development works without any extra setup.
    """
    api_key = os.environ.get("API_KEY", "").strip()
    if not api_key:
        return

    @app.before_request
    def _check_api_key():
        if request.path in _PUBLIC_PATHS:
            return None
        provided = request.headers.get("X-API-Key", "")
        # Constant-time comparison so response timing doesn't leak the key.
        if not hmac.compare_digest(provided.encode(), api_key.encode()):
            return jsonify({"error": "Unauthorized"}), 401


def create_app():
    app = Flask(__name__)

    database_url = os.environ.get("DATABASE_URL", "sqlite:///data.db")
    app.config["SQLALCHEMY_DATABASE_URI"] = database_url
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    init_db(app)
    init_celery(app)
    _init_auth(app)

    @app.route("/", methods=["GET"])
    def home():
        return jsonify({"status": "ok"})

    init_blueprints(app)
    init_metrics(app)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
