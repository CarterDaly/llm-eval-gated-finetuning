import json
import logging
import os
import re
from datetime import datetime, timezone

from flask import jsonify, request

from gated_finetune.services.database.database_client import db
from gated_finetune.services.database.prediction import ModelSQL

logger = logging.getLogger(__name__)

_MODELS_DIR = os.environ.get("MODELS_DIR", "models")

# name and version become directory names under MODELS_DIR. Restricting them to
# this alphabet (and requiring a leading letter/digit) rules out "..", "/" and
# absolute paths, so an upload can never be written outside MODELS_DIR.
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

# Every model must link to the MLflow run that produced it: predictions log
# their first SHAP values and embedding to that run, and fail without it.
_REQUIRED_FIELDS = ("name", "version", "features", "mlflow_run_id")


def _error(message: str, status: int):
    return jsonify({"error": message}), status


def _read_ingest_form():
    """Return (fields, None) or (None, error_response)."""
    fields = {key: (request.form.get(key) or "").strip() for key in _REQUIRED_FIELDS}
    for key in _REQUIRED_FIELDS:
        if not fields[key]:
            return None, _error(f"Missing required field: {key}", 400)
    for key in ("name", "version"):
        if not _SAFE_NAME.match(fields[key]):
            return None, _error(
                f"Invalid {key}: use letters, digits, '.', '_' or '-', starting with a letter or digit",
                400,
            )
    return fields, None


def _parse_feature_names(raw: str) -> list[str]:
    """Accept a JSON list or a comma-separated string."""
    try:
        names = json.loads(raw)
        if isinstance(names, list):
            return names
    except json.JSONDecodeError:
        pass
    return [name.strip() for name in raw.split(",") if name.strip()]


class ModelController:

    @staticmethod
    def ingest_model():
        """Accept a pickled model file + metadata and register it in the DB.

        Expects multipart/form-data with:
          - file          : the pickled model artifact (required)
          - name          : model name (required)
          - version       : model version (required)
          - features      : comma-separated or JSON list of feature names (required)
          - mlflow_run_id : the MLflow run that produced the model (required)
          - type          : model type, default "generic" (optional)

        The artifact is unpickled at prediction time, which executes code from
        the file. Only ingest models you trust, and set API_KEY when exposed.
        """
        model_file = request.files.get("file")
        if model_file is None:
            return _error("Missing required field: file", 400)

        fields, error = _read_ingest_form()
        if error:
            return error

        name, version = fields["name"], fields["version"]
        feature_names = _parse_feature_names(fields["features"])
        model_type = (request.form.get("type") or "generic").strip()

        existing = (
            db.session.query(ModelSQL)
            .filter(ModelSQL.model_name == name, ModelSQL.model_version == version)
            .first()
        )
        if existing:
            return jsonify({"error": f"Model '{name}' version '{version}' already exists", "model_id": existing.id}), 409

        dest_dir = os.path.join(_MODELS_DIR, name, version)
        os.makedirs(dest_dir, exist_ok=True)
        artifact_path = os.path.join(dest_dir, "model.pkl")
        model_file.save(artifact_path)

        now = datetime.now(timezone.utc)
        model_sql = ModelSQL(
            id=f"model-{name}-{version}",
            model_type=model_type,
            model_name=name,
            model_version=version,
            artifact_path=artifact_path,
            feature_names=feature_names,
            mlflow_run_id=fields["mlflow_run_id"],
            created=now,
            updated=now,
        )
        db.session.add(model_sql)
        try:
            db.session.commit()
        except Exception:
            logger.exception("model_ingest_persist_failed")
            db.session.rollback()
            return _error("Failed to register model", 500)

        return jsonify({
            "model_id": model_sql.id,
            "model_name": name,
            "model_version": version,
            "artifact_path": artifact_path,
            "feature_names": feature_names,
        }), 201

    @staticmethod
    def list_models():
        """Return all registered models."""
        models = db.session.query(ModelSQL).all()
        return jsonify([
            {
                "model_id": m.id,
                "model_name": m.model_name,
                "model_version": m.model_version,
                "model_type": m.model_type,
                "feature_names": m.feature_names,
                "artifact_path": m.artifact_path,
                "mlflow_run_id": m.mlflow_run_id,
                "created": m.created.isoformat() if m.created else None,
            }
            for m in models
        ]), 200


model_controller = ModelController()
