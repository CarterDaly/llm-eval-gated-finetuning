import logging
import os
from datetime import datetime, timezone

from flask import jsonify, request

from data5580_hw.services.database.database_client import db
from data5580_hw.services.database.prediction import ModelSQL

logger = logging.getLogger(__name__)

_MODELS_DIR = os.environ.get("MODELS_DIR", "models")


class ModelController:

    @staticmethod
    def ingest_model():
        """Accept a pickled model file + metadata and register it in the DB.

        Expects multipart/form-data with:
          - file     : the pickled model artifact (required)
          - name     : model name (required)
          - version  : model version (required)
          - type     : model type, default "generic" (optional)
          - features : comma-separated feature names (required)
          - mlflow_run_id : optional MLflow run id
        """
        model_file = request.files.get("file")
        if model_file is None:
            return jsonify({"error": "Missing required field: file"}), 400

        name = (request.form.get("name") or "").strip()
        version = (request.form.get("version") or "").strip()
        features_raw = (request.form.get("features") or "").strip()

        if not name:
            return jsonify({"error": "Missing required field: name"}), 400
        if not version:
            return jsonify({"error": "Missing required field: version"}), 400
        if not features_raw:
            return jsonify({"error": "Missing required field: features"}), 400

        feature_names = [f.strip() for f in features_raw.split(",") if f.strip()]
        model_type = (request.form.get("type") or "generic").strip()
        mlflow_run_id = (request.form.get("mlflow_run_id") or "").strip() or None

        # Check for duplicate
        existing = (
            db.session.query(ModelSQL)
            .filter(ModelSQL.model_name == name, ModelSQL.model_version == version)
            .first()
        )
        if existing:
            return jsonify({"error": f"Model '{name}' version '{version}' already exists", "model_id": existing.id}), 409

        # Persist file
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
            mlflow_run_id=mlflow_run_id,
            created=now,
            updated=now,
        )
        db.session.add(model_sql)
        try:
            db.session.commit()
        except Exception:
            logger.exception("model_ingest_persist_failed")
            db.session.rollback()
            return jsonify({"error": "Failed to register model"}), 500

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
