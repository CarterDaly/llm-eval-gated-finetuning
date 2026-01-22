import os
import tempfile

os.environ['PROMETHEUS_MULTIPROC_DIR'] = tempfile.mkdtemp()

# Standardlibrary
import  logging

# Installed
from flask import Flask, jsonify

# Local
from data5580_hw.routes import init_blueprints
from data5580_hw.services.database.database_client import init_db
from data5580_hw.monitoring import init_metrics

logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger(__name__)


def create_app():
    app = Flask(__name__)

    app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///data.db"
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    init_db(app)

    @app.route("/", methods=["GET"])
    def home():
        return jsonify({"message": "Hello, Flask!"})

    init_blueprints(app)
    init_metrics(app)

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
