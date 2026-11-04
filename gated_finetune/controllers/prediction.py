import logging
from datetime import datetime, timezone

from flask import jsonify

from gated_finetune.controllers.common import missing_fields_message, parse_body
from gated_finetune.models.prediction import (
    ActualUpdateRequest,
    BatchActualUpdateRequest,
    Explanations,
    Model,
    Prediction,
    PredictionRequest,
    UmapConfig,
)
from gated_finetune.services.database.database_client import db
from gated_finetune.services.database.prediction import ExplanationSQL, ModelSQL, PredictionSQL
from gated_finetune.services.arize_service import arize_service
from gated_finetune.services.errors import (
    ArizeLogError,
    EmbeddingError,
    ExplanationError,
    InferenceError,
    LLMError,
    MlflowLogError,
    ModelLoadError,
)
from gated_finetune.services.llm_service import llm_service
from gated_finetune.services.explainer_service import explainer_service
from gated_finetune.services.mlflow_service import mlflow_service
from gated_finetune.services.model_service import model_service
from gated_finetune.services.umap_service import umap_service

logger = logging.getLogger(__name__)


def _extract_shap_values(explanations: Explanations | None) -> dict:
    if not explanations:
        return {}
    for explanation in explanations.explanations:
        if explanation.name == "shap_values":
            return explanation.values
    return {}


def _extract_embeddings(prediction: Prediction) -> dict:
    if not prediction.embeddings:
        return {}
    return {"kind": "inputs", **prediction.embeddings}


def _json_safe(value):
    if hasattr(value, "item"):
        return value.item()
    return value


def _get_model_record(model_name: str, model_version: str):
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


def _load_model_or_error(model: Model):
    try:
        model_service.load_model(model)
    except ModelLoadError as exc:
        logger.exception("model_load_failed")
        return jsonify({"error": str(exc)}), 500
    return None


def _validate_feature_presence(model: Model, prediction_request: PredictionRequest):
    if not model.feature_names:
        return None
    missing = set(model.feature_names) - set(prediction_request.features.keys())
    if missing:
        return jsonify({"error": missing_fields_message(list(missing))}), 400
    return None


def _run_inference(model: Model, prediction: Prediction):
    try:
        prediction.label = model_service.create_inference(model, prediction)
    except InferenceError as exc:
        logger.exception("inference_failed")
        return jsonify({"error": str(exc)}), 500
    return None


def _run_explanation(model: Model, prediction: Prediction):
    try:
        prediction.explanations = explainer_service.create_explanation(model, prediction)
    except ExplanationError as exc:
        logger.exception("explanation_failed")
        return None, (jsonify({"error": str(exc)}), 400)
    return _extract_shap_values(prediction.explanations), None


def _log_shap_values(model_sql: ModelSQL, shap_values: dict, prediction_id: str):
    try:
        artifact_path = mlflow_service.log_shap_once(
            run_id=model_sql.mlflow_run_id or "",
            model_name=model_sql.model_name,
            model_version=model_sql.model_version,
            shap_values=shap_values,
            prediction_id=prediction_id,
            existing_artifact_path=model_sql.shap_artifact_path,
        )
    except MlflowLogError as exc:
        logger.exception("mlflow_log_failed")
        return None, (jsonify({"error": str(exc)}), 500)

    if artifact_path:
        model_sql.shap_artifact_path = artifact_path
        model_sql.shap_logged_at = datetime.now(timezone.utc)
        db.session.add(model_sql)

    return artifact_path, None


def _run_embedding(model: Model, prediction: Prediction, prediction_request: PredictionRequest):
    try:
        umap_config = prediction_request.umap or UmapConfig()
        embedding = umap_service.create_embedding(model, prediction, umap_config)
    except EmbeddingError as exc:
        logger.exception("embedding_failed")
        return None, (jsonify({"error": str(exc)}), 400)
    return embedding, None


def _log_embedding_values(model_sql: ModelSQL, embedding: dict, prediction_id: str):
    coords = {k: v for k, v in embedding.items() if k != "kind"}
    try:
        artifact_path = mlflow_service.log_embedding_once(
            run_id=model_sql.mlflow_run_id or "",
            model_name=model_sql.model_name,
            model_version=model_sql.model_version,
            embedding=coords,
            prediction_id=prediction_id,
            existing_artifact_path=model_sql.embedding_artifact_path,
        )
    except MlflowLogError as exc:
        logger.exception("mlflow_embedding_log_failed")
        return None, (jsonify({"error": str(exc)}), 500)

    if artifact_path:
        model_sql.embedding_artifact_path = artifact_path
        model_sql.embedding_logged_at = datetime.now(timezone.utc)
        db.session.add(model_sql)

    return artifact_path, None


def _explain_and_log(model: Model, model_sql: ModelSQL, prediction: Prediction):
    """Compute SHAP values and log them to MLflow on the model's first inference."""
    shap_values, error = _run_explanation(model, prediction)
    if error:
        return None, error
    _, error = _log_shap_values(model_sql, shap_values, prediction.id)
    if error:
        return None, error
    return shap_values, None


def _embed_and_log(model: Model, model_sql: ModelSQL, prediction: Prediction, prediction_request: PredictionRequest):
    """Compute the UMAP embedding and log it to MLflow on the model's first inference."""
    embedding, error = _run_embedding(model, prediction, prediction_request)
    if error:
        return None, error
    _, error = _log_embedding_values(model_sql, embedding, prediction.id)
    if error:
        return None, error
    return embedding, None


def _log_arize_inference(prediction: Prediction, model: Model) -> None:
    """Submit inference to Arize; failures are non-fatal and logged locally."""
    try:
        arize_service.log_inference(
            prediction_id=prediction.id,
            model_name=model.name,
            model_version=model.version,
            prediction_label=prediction.label,
            features=prediction.features,
            actual_label=prediction.actual,
            shap_values=_extract_shap_values(prediction.explanations),
            timestamp=prediction.created,
        )
    except ArizeLogError:
        logger.exception("arize_inference_log_failed")


def _log_arize_actual(prediction_sql, actual_label) -> None:
    """Submit updated actual to Arize; failures are non-fatal and logged locally."""
    try:
        arize_service.log_actual(
            prediction_id=prediction_sql.id,
            model_name=prediction_sql.model.model_name,
            model_version=prediction_sql.model.model_version,
            actual_label=actual_label,
        )
    except ArizeLogError:
        logger.exception("arize_actual_log_failed")


def _persist_prediction(prediction: Prediction, model: Model):
    prediction_sql = PredictionSQL.from_prediction(prediction, model)
    db.session.add(prediction_sql)

    for explanation in prediction.explanations.explanations:
        db.session.add(ExplanationSQL.from_explanation(explanation, prediction_sql.id))

    try:
        db.session.commit()
    except Exception:
        logger.exception("prediction_persist_failed")
        db.session.rollback()
        return None, (jsonify({"error": "Failed to store prediction"}), 500)

    return prediction_sql, None


class PredictionController:

    @staticmethod
    def create_prediction(model_name: str, model_version: str) -> tuple[str, int]:
        prediction_request, error = parse_body(PredictionRequest)
        if error:
            return error

        model, model_sql, error = _get_model_record(model_name, model_version)
        if error:
            return error

        error = _load_model_or_error(model)
        if error:
            return error

        error = _validate_feature_presence(model, prediction_request)
        if error:
            return error

        prediction = Prediction(
            features=prediction_request.features,
            tags=prediction_request.tags,
            actual=prediction_request.actual,
            threshold=prediction_request.threshold,
            model=model,
        )

        error = _run_inference(model, prediction)
        if error:
            return error

        shap_values, error = _explain_and_log(model, model_sql, prediction)
        if error:
            return error

        embedding, error = _embed_and_log(model, model_sql, prediction, prediction_request)
        if error:
            return error

        prediction.embeddings = {k: v for k, v in embedding.items() if k != "kind"}

        _, error = _persist_prediction(prediction, model)
        if error:
            return error

        _log_arize_inference(prediction, model)

        return (
            jsonify(
                {
                    "prediction_id": prediction.id,
                    "label": _json_safe(prediction.label),
                    "shap_values": shap_values,
                    "embeddings": _extract_embeddings(prediction),
                }
            ),
            201,
        )

    @staticmethod
    def get_prediction_by_id(prediction_id: str) -> tuple[str, int]:
        prediction_sql = (
            db.session.query(PredictionSQL)
            .filter(PredictionSQL.id == prediction_id)
            .first()
        )

        if prediction_sql is None:
            return jsonify({"error": "Prediction not found"}), 404

        prediction = prediction_sql.to_prediction()
        return (
            jsonify(
                {
                    "prediction_id": prediction.id,
                    "label": _json_safe(prediction.label),
                    "shap_values": _extract_shap_values(prediction.explanations),
                    "embeddings": _extract_embeddings(prediction),
                }
            ),
            200,
        )

    @staticmethod
    def get_prediction_explainer(prediction_id: str) -> tuple[str, int]:
        prediction_sql = (
            db.session.query(PredictionSQL)
            .filter(PredictionSQL.id == prediction_id)
            .first()
        )

        if prediction_sql is None:
            return jsonify({"error": "Prediction not found"}), 404

        prediction = prediction_sql.to_prediction()

        all_predictions_sql = (
            db.session.query(PredictionSQL)
            .filter(PredictionSQL.model_id == prediction_sql.model_id)
            .all()
        )
        all_predictions = [p.to_prediction() for p in all_predictions_sql]

        nearest = llm_service.find_nearest_predictions(prediction, all_predictions)

        try:
            summary = llm_service.generate_summary(prediction, nearest)
        except LLMError as exc:
            logger.exception("llm_summary_failed")
            return jsonify({"error": str(exc)}), 500

        return (
            jsonify(
                {
                    "prediction_id": prediction.id,
                    "label": _json_safe(prediction.label),
                    "shap_values": _extract_shap_values(prediction.explanations),
                    "embeddings": _extract_embeddings(prediction),
                    "summary": summary,
                }
            ),
            200,
        )

    @staticmethod
    def update_actual(prediction_id: str) -> tuple[str, int]:
        body, error = parse_body(ActualUpdateRequest)
        if error:
            return error

        prediction_sql = (
            db.session.query(PredictionSQL)
            .filter(PredictionSQL.id == prediction_id)
            .first()
        )

        if prediction_sql is None:
            return jsonify({"error": "Prediction not found"}), 404

        prediction_sql.actual = body.actual
        try:
            db.session.commit()
        except Exception:
            db.session.rollback()
            return jsonify({"error": "Failed to update prediction"}), 500

        _log_arize_actual(prediction_sql, body.actual)

        return jsonify({"message": "Prediction updated", "prediction_id": prediction_id}), 200

    @staticmethod
    def update_actuals_batch() -> tuple[str, int]:
        body, error = parse_body(BatchActualUpdateRequest)
        if error:
            return error

        updated = []
        not_found = []

        for item in body.updates:
            prediction_sql = (
                db.session.query(PredictionSQL)
                .filter(PredictionSQL.id == item.prediction_id)
                .first()
            )
            if prediction_sql is None:
                not_found.append(item.prediction_id)
                continue
            prediction_sql.actual = item.actual
            updated.append(item.prediction_id)

        if updated:
            try:
                db.session.commit()
            except Exception:
                db.session.rollback()
                return jsonify({"error": "Failed to update predictions"}), 500

            for pid in updated:
                prediction_sql = db.session.query(PredictionSQL).filter(PredictionSQL.id == pid).first()
                if prediction_sql:
                    _log_arize_actual(prediction_sql, prediction_sql.actual)

        return jsonify({"updated": updated, "not_found": not_found}), 200


prediction_controller = PredictionController()
