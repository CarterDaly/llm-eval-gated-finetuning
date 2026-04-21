"""Candidate scoring controller.

Orchestrates the full hiring-platform pipeline:
  GitHub username → behavioural features → prediction pipeline
  → explore/exploit adjustment → stored result with hiring metadata.

Nothing in the downstream prediction pipeline knows about GitHub or candidates;
it only sees a normalised feature vector, exactly as designed.
"""
import logging
from datetime import datetime, timezone

from flask import jsonify, request
from pydantic import ValidationError

from data5580_hw.models.candidate import CandidateScoreRequest
from data5580_hw.models.prediction import Explanations, Model, Prediction, UmapConfig
from data5580_hw.services.arize_service import arize_service
from data5580_hw.services.connectors.github_connector import GitHubConnector, GitHubConnectorError, github_connector
from data5580_hw.services.database.database_client import db
from data5580_hw.services.database.prediction import ExplanationSQL, ModelSQL, PredictionSQL
from data5580_hw.services.errors import (
    ArizeLogError,
    EmbeddingError,
    ExplanationError,
    InferenceError,
    MlflowLogError,
    ModelLoadError,
)
from data5580_hw.services.exploration_service import exploration_service
from data5580_hw.services.explainer_service import explainer_service
from data5580_hw.services.mlflow_service import mlflow_service
from data5580_hw.services.model_service import model_service
from data5580_hw.services.umap_service import umap_service

logger = logging.getLogger(__name__)


def _parse_score_request():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return None, (jsonify({"error": "Request body must be JSON"}), 400)
    try:
        return CandidateScoreRequest.model_validate(data), None
    except ValidationError as exc:
        errors = exc.errors()
        missing = [str(err["loc"][0]) for err in errors if err.get("type") == "missing"]
        if missing:
            return None, (jsonify({"error": "Missing required fields: {}".format(", ".join(sorted(missing)))}), 400)
        if any("umap" in err.get("loc", ()) for err in errors):
            return None, (jsonify({"error": "Invalid UMAP configuration"}), 400)
        return None, (jsonify({"error": "Invalid request body"}), 400)


def _get_model(model_name: str, model_version: str):
    model_sql = (
        db.session.query(ModelSQL)
        .filter(ModelSQL.model_name == model_name, ModelSQL.model_version == model_version)
        .first()
    )
    if model_sql is None:
        return None, None, (jsonify({"error": "Model not found"}), 404)

    model = Model(
        id=model_sql.id,
        name=model_sql.model_name,
        version=model_sql.model_version,
        type=model_sql.model_type,
        artifact_path=model_sql.artifact_path,
        feature_names=model_sql.feature_names,
        mlflow_run_id=model_sql.mlflow_run_id,
    )
    return model, model_sql, None


def _count_feedback(model_id: str) -> int:
    """Return number of predictions for this model that have received a ground-truth label."""
    return (
        db.session.query(PredictionSQL)
        .filter(PredictionSQL.model_id == model_id, PredictionSQL.actual.isnot(None))
        .count()
    )


def _to_float_score(label) -> float:
    """Coerce a model label to a float score in [0, 1] for exploration adjustment."""
    try:
        score = float(label)
        return max(0.0, min(1.0, score))
    except (TypeError, ValueError):
        return 0.5


class CandidateController:

    @staticmethod
    def score_candidate(github_username: str):
        score_request, error = _parse_score_request()
        if error:
            return error

        # 1. Extract behavioural features from GitHub
        try:
            features = github_connector.extract_features(github_username)
        except GitHubConnectorError as exc:
            logger.exception("github_feature_extraction_failed username=%s", github_username)
            return jsonify({"error": str(exc)}), 502

        # 2. Load model
        model, model_sql, error = _get_model(score_request.model_name, score_request.model_version)
        if error:
            return error

        try:
            model_service.load_model(model)
        except ModelLoadError as exc:
            logger.exception("model_load_failed")
            return jsonify({"error": str(exc)}), 500

        # 3. Validate features against model's expected feature names
        if model.feature_names:
            missing = set(model.feature_names) - set(features.keys())
            if missing:
                return jsonify({"error": "Missing required features: {}".format(", ".join(sorted(missing)))}), 400

        # 4. Build prediction domain object
        prediction = Prediction(
            features=features,
            candidate_id=github_username,
            role_context=score_request.role_context,
            model=model,
        )

        # 5. Run inference
        try:
            raw_label = model_service.create_inference(model, prediction)
        except InferenceError as exc:
            logger.exception("inference_failed")
            return jsonify({"error": str(exc)}), 500

        # 6. Apply explore/exploit adjustment
        n_feedback = _count_feedback(model_sql.id)
        raw_score = _to_float_score(raw_label)
        prediction.label = exploration_service.adjust_score(raw_score, n_feedback)

        # 7. SHAP explanation
        try:
            prediction.explanations = explainer_service.create_explanation(model, prediction)
        except ExplanationError as exc:
            logger.exception("explanation_failed")
            return jsonify({"error": str(exc)}), 400
        shap_values = _extract_shap(prediction.explanations)

        # 8. MLflow logging
        try:
            artifact_path = mlflow_service.log_shap_once(
                run_id=model_sql.mlflow_run_id or "",
                model_name=model_sql.model_name,
                model_version=model_sql.model_version,
                shap_values=shap_values,
                prediction_id=prediction.id,
                existing_artifact_path=model_sql.shap_artifact_path,
                tracking_uri=model_sql.mlflow_tracking_uri,
            )
        except MlflowLogError as exc:
            logger.exception("mlflow_log_failed")
            return jsonify({"error": str(exc)}), 500

        if artifact_path:
            model_sql.shap_artifact_path = artifact_path
            model_sql.shap_logged_at = datetime.now(timezone.utc)
            db.session.add(model_sql)

        # 9. UMAP embedding
        try:
            umap_config = score_request.umap or UmapConfig()
            embedding = umap_service.create_embedding(model, prediction, umap_config)
        except EmbeddingError as exc:
            logger.exception("embedding_failed")
            return jsonify({"error": str(exc)}), 400

        # 9b. MLflow embedding logging (idempotent — first inference only)
        try:
            embedding_coords = {k: v for k, v in embedding.items() if k != "kind"}
            emb_artifact_path = mlflow_service.log_embedding_once(
                run_id=model_sql.mlflow_run_id or "",
                model_name=model_sql.model_name,
                model_version=model_sql.model_version,
                embedding=embedding_coords,
                prediction_id=prediction.id,
                existing_artifact_path=model_sql.embedding_artifact_path,
                tracking_uri=model_sql.mlflow_tracking_uri,
            )
        except MlflowLogError as exc:
            logger.exception("mlflow_embedding_log_failed")
            return jsonify({"error": str(exc)}), 500

        if emb_artifact_path:
            model_sql.embedding_artifact_path = emb_artifact_path
            model_sql.embedding_logged_at = datetime.now(timezone.utc)
            db.session.add(model_sql)

        prediction.embeddings = {k: v for k, v in embedding.items() if k != "kind"}

        # 10. Persist to DB
        prediction_sql = PredictionSQL.from_prediction(prediction, model)
        db.session.add(prediction_sql)
        for explanation in prediction.explanations.explanations:
            db.session.add(ExplanationSQL.from_explanation(explanation, prediction_sql.id))
        try:
            db.session.commit()
        except Exception:
            logger.exception("candidate_prediction_persist_failed")
            db.session.rollback()
            return jsonify({"error": "Failed to store prediction"}), 500

        # 11. Arize (non-fatal)
        try:
            arize_service.log_inference(
                prediction_id=prediction.id,
                model_name=model.name,
                model_version=model.version,
                prediction_label=prediction.label,
                features=features,
                actual_label=None,
                shap_values=shap_values,
                embedding=prediction.embeddings,
                timestamp=prediction.created,
            )
        except ArizeLogError:
            logger.exception("arize_inference_log_failed")

        return (
            jsonify(
                {
                    "prediction_id": prediction.id,
                    "candidate_id": github_username,
                    "fit_score": prediction.label,
                    "shap_values": shap_values,
                    "embeddings": {k: v for k, v in embedding.items() if k != "kind"},
                    "features": features,
                    "exploration_epsilon": exploration_service.compute_epsilon(n_feedback),
                }
            ),
            201,
        )


def _extract_shap(explanations: Explanations | None) -> dict:
    if not explanations:
        return {}
    for explanation in explanations.explanations:
        if explanation.name == "shap_values":
            return explanation.values
    return {}


candidate_controller = CandidateController()
