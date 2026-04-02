import logging
from datetime import datetime, timezone

from flask import jsonify, request

from pydantic import ValidationError

from data5580_hw.models.prediction import Model, Prediction, PredictionRequest, Explanations, UmapConfig
from data5580_hw.services.database.database_client import db
from data5580_hw.services.database.prediction import ExplanationSQL, ModelSQL, PredictionSQL
from data5580_hw.services.arize_service import arize_service
from data5580_hw.services.errors import (
    ArizeLogError,
    EmbeddingError,
    ExplanationError,
    InferenceError,
    MlflowLogError,
    ModelLoadError,
)
from data5580_hw.services.explainer_service import explainer_service
from data5580_hw.services.mlflow_service import mlflow_service
from data5580_hw.services.model_service import model_service
from data5580_hw.services.umap_service import umap_service



logger = logging.getLogger(__name__)


def _missing_fields_message(fields: list[str]) -> str:
    return "Missing required fields: {}".format(", ".join(sorted(fields)))


def _validation_error_response(error: ValidationError):
    errors = error.errors()
    missing = [err["loc"][0] for err in errors if err.get("type") == "missing"]
    if missing:
        return jsonify({"error": _missing_fields_message(missing)}), 400
    if any("umap" in err.get("loc", []) for err in errors):
        return jsonify({"error": "Invalid UMAP configuration"}), 400
    return jsonify({"error": "Invalid prediction input"}), 400


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


def _get_prediction_request():
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return None, (jsonify({"error": _missing_fields_message(["features"])}), 400)
    try:
        return PredictionRequest.model_validate(data), None
    except ValidationError as exc:
        return None, _validation_error_response(exc)


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
        return jsonify({"error": _missing_fields_message(list(missing))}), 400
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
            tracking_uri=model_sql.mlflow_tracking_uri,
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
        prediction_request, error = _get_prediction_request()
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

        shap_values, error = _run_explanation(model, prediction)
        if error:
            return error

        _, error = _log_shap_values(model_sql, shap_values, prediction.id)
        if error:
            return error

        embedding, error = _run_embedding(model, prediction, prediction_request)
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
    def update_actual(prediction_id: str) -> tuple[str, int]:
        data = request.get_json(silent=True)
        if not isinstance(data, dict) or "actual" not in data:
            return jsonify({"error": _missing_fields_message(["actual"])}), 400

        prediction_sql = (
            db.session.query(PredictionSQL)
            .filter(PredictionSQL.id == prediction_id)
            .first()
        )

        if prediction_sql is None:
            return jsonify({"error": "Prediction not found"}), 404

        prediction_sql.actual = data["actual"]
        try:
            db.session.commit()
        except Exception:
            db.session.rollback()
            return jsonify({"error": "Failed to update prediction"}), 500

        _log_arize_actual(prediction_sql, data["actual"])

        return jsonify({"message": "Prediction updated", "prediction_id": prediction_id}), 200

prediction_controller = PredictionController()
