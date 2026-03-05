import logging

from flask import jsonify, request

from pydantic import ValidationError

from data5580_hw.models.prediction import Model, Prediction, PredictionRequest, Explanations
from data5580_hw.services.database.database_client import db
from data5580_hw.services.database.prediction import ExplanationSQL, ModelSQL, PredictionSQL
from data5580_hw.services.errors import ExplanationError, InferenceError, ModelLoadError
from data5580_hw.services.explainer_service import explainer_service
from data5580_hw.services.model_service import model_service



logger = logging.getLogger(__name__)


def _missing_fields_message(fields: list[str]) -> str:
    return "Missing required fields: {}".format(", ".join(sorted(fields)))


def _validation_error_response(error: ValidationError):
    errors = error.errors()
    missing = [err["loc"][0] for err in errors if err.get("type") == "missing"]
    if missing:
        return jsonify({"error": _missing_fields_message(missing)}), 400
    return jsonify({"error": "Invalid prediction input"}), 400


def _extract_shap_values(explanations: Explanations | None) -> dict:
    if not explanations:
        return {}
    for explanation in explanations.explanations:
        if explanation.name == "shap_values":
            return explanation.values
    return {}


def _json_safe(value):
    if hasattr(value, "item"):
        return value.item()
    return value


class PredictionController:

    @staticmethod
    def create_prediction(model_name: str, model_version: str) -> tuple[str, int]:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            return jsonify({"error": _missing_fields_message(["features"])}), 400

        try:
            prediction_request = PredictionRequest.model_validate(data)
        except ValidationError as exc:
            return _validation_error_response(exc)

        model_sql = (
            db.session.query(ModelSQL)
            .filter(ModelSQL.model_name == model_name, ModelSQL.model_version == model_version)
            .first()
        )
        if model_sql is None:
            return jsonify({"error": "Model not found"}), 404

        model = Model(
            id=model_sql.id,
            name=model_sql.model_name,
            version=model_sql.model_version,
            type=model_sql.model_type,
            artifact_path=model_sql.artifact_path,
            feature_names=model_sql.feature_names,
        )

        try:
            model_service.load_model(model)
        except ModelLoadError as exc:
            logger.exception("model_load_failed")
            return jsonify({"error": str(exc)}), 500

        if model.feature_names:
            missing = set(model.feature_names) - set(prediction_request.features.keys())
            if missing:
                return jsonify({"error": _missing_fields_message(list(missing))}), 400

        prediction = Prediction(
            features=prediction_request.features,
            tags=prediction_request.tags,
            actual=prediction_request.actual,
            threshold=prediction_request.threshold,
            model=model,
        )

        try:
            prediction.label = model_service.create_inference(model, prediction)
        except InferenceError as exc:
            logger.exception("inference_failed")
            return jsonify({"error": str(exc)}), 500

        try:
            prediction.explanations = explainer_service.create_explanation(model, prediction)
        except ExplanationError as exc:
            logger.exception("explanation_failed")
            return jsonify({"error": str(exc)}), 400

        prediction_sql = PredictionSQL.from_prediction(prediction, model)
        db.session.add(prediction_sql)

        for explanation in prediction.explanations.explanations:
            db.session.add(ExplanationSQL.from_explanation(explanation, prediction_sql.id))

        try:
            db.session.commit()
        except Exception:
            logger.exception("prediction_persist_failed")
            db.session.rollback()
            return jsonify({"error": "Failed to store prediction"}), 500

        return (
            jsonify(
                {
                    "prediction_id": prediction.id,
                    "label": _json_safe(prediction.label),
                    "shap_values": _extract_shap_values(prediction.explanations),
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

        return jsonify({"message": "Prediction updated", "prediction_id": prediction_id}), 200

prediction_controller = PredictionController()
