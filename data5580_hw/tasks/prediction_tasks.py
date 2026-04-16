import logging
from datetime import datetime, timezone

from data5580_hw.celery_app import celery
from data5580_hw.models.prediction import (
    Explanations,
    Model,
    Prediction,
    UmapConfig,
)
from data5580_hw.services.errors import (
    ArizeLogError,
    EmbeddingError,
    ExplanationError,
    InferenceError,
    MlflowLogError,
    ModelLoadError,
)

logger = logging.getLogger(__name__)


def _query_model(model_name: str, model_version: str):
    """Return (ModelSQL, None) or (None, error_dict)."""
    from data5580_hw.services.database.database_client import db
    from data5580_hw.services.database.prediction import ModelSQL

    model_sql = (
        db.session.query(ModelSQL)
        .filter(ModelSQL.model_name == model_name, ModelSQL.model_version == model_version)
        .first()
    )
    if model_sql is None:
        return None, {"error": "Model not found", "status_code": 404}

    model = Model(
        id=model_sql.id,
        name=model_sql.model_name,
        version=model_sql.model_version,
        type=model_sql.model_type,
        artifact_path=model_sql.artifact_path,
        feature_names=model_sql.feature_names,
        mlflow_run_id=model_sql.mlflow_run_id,
    )
    return (model, model_sql), None


def _load_model(model: Model):
    from data5580_hw.services.model_service import model_service

    try:
        model_service.load_model(model)
    except ModelLoadError as exc:
        logger.exception("task_model_load_failed")
        return {"error": str(exc), "status_code": 500}
    return None


def _do_inference(model: Model, prediction: Prediction):
    from data5580_hw.services.model_service import model_service

    try:
        prediction.label = model_service.create_inference(model, prediction)
    except InferenceError as exc:
        logger.exception("task_inference_failed")
        return {"error": str(exc), "status_code": 500}
    return None


def _do_explanation(model: Model, prediction: Prediction):
    from data5580_hw.services.explainer_service import explainer_service

    try:
        prediction.explanations = explainer_service.create_explanation(model, prediction)
    except ExplanationError as exc:
        logger.exception("task_explanation_failed")
        return None, {"error": str(exc), "status_code": 400}
    shap_values = _extract_shap(prediction.explanations)
    return shap_values, None


def _do_mlflow(model_sql, shap_values: dict, prediction_id: str):
    from data5580_hw.services.database.database_client import db
    from data5580_hw.services.mlflow_service import mlflow_service

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
        logger.exception("task_mlflow_log_failed")
        return {"error": str(exc), "status_code": 500}

    if artifact_path:
        model_sql.shap_artifact_path = artifact_path
        model_sql.shap_logged_at = datetime.now(timezone.utc)
        db.session.add(model_sql)
    return None


def _do_embedding(model: Model, prediction: Prediction, umap_config: dict | None):
    from data5580_hw.services.umap_service import umap_service

    try:
        cfg = UmapConfig(**(umap_config or {}))
        embedding = umap_service.create_embedding(model, prediction, cfg)
    except EmbeddingError as exc:
        logger.exception("task_embedding_failed")
        return None, {"error": str(exc), "status_code": 400}
    return embedding, None


def _do_persist(prediction: Prediction, model: Model):
    from data5580_hw.services.database.database_client import db
    from data5580_hw.services.database.prediction import ExplanationSQL, PredictionSQL

    prediction_sql = PredictionSQL.from_prediction(prediction, model)
    db.session.add(prediction_sql)

    for explanation in prediction.explanations.explanations:
        db.session.add(ExplanationSQL.from_explanation(explanation, prediction_sql.id))

    try:
        db.session.commit()
    except Exception:
        logger.exception("task_prediction_persist_failed")
        db.session.rollback()
        return {"error": "Failed to store prediction", "status_code": 500}
    return None


def _do_arize(prediction: Prediction, model: Model) -> None:
    from data5580_hw.services.arize_service import arize_service

    try:
        arize_service.log_inference(
            prediction_id=prediction.id,
            model_name=model.name,
            model_version=model.version,
            prediction_label=prediction.label,
            features=prediction.features,
            actual_label=prediction.actual,
            shap_values=_extract_shap(prediction.explanations),
            timestamp=prediction.created,
        )
    except ArizeLogError:
        logger.exception("task_arize_inference_log_failed")


def _extract_shap(explanations: Explanations | None) -> dict:
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


@celery.task(bind=True, name="prediction_tasks.run_short_prediction", max_retries=3, default_retry_delay=5)
def run_short_prediction(
    self,
    model_name: str,
    model_version: str,
    features: dict,
    tags: dict | None = None,
    actual=None,
    threshold: float | None = None,
):
    """Short-running task: model inference and DB persist only (no SHAP/UMAP/MLflow/Arize)."""
    flask_app = self.app.flask_app
    with flask_app.app_context():
        result, err = _query_model(model_name, model_version)
        if err:
            return err

        model, _model_sql = result

        err = _load_model(model)
        if err:
            return err

        prediction = Prediction(
            features=features,
            tags=tags or {},
            actual=actual,
            threshold=threshold,
            model=model,
            explanations=Explanations(explanations=[]),
        )

        err = _do_inference(model, prediction)
        if err:
            return err

        err = _do_persist(prediction, model)
        if err:
            return err

        logger.info("short_prediction_complete prediction_id=%s", prediction.id)
        return {
            "prediction_id": prediction.id,
            "label": _json_safe(prediction.label),
        }


@celery.task(
    bind=True,
    name="prediction_tasks.run_long_prediction",
    max_retries=3,
    default_retry_delay=10,
    time_limit=300,
    soft_time_limit=270,
)
def run_long_prediction(
    self,
    model_name: str,
    model_version: str,
    features: dict,
    tags: dict | None = None,
    actual=None,
    threshold: float | None = None,
    umap_config: dict | None = None,
):
    """Long-running task: full pipeline (inference + SHAP + MLflow + UMAP + Arize + DB persist)."""
    flask_app = self.app.flask_app
    with flask_app.app_context():
        result, err = _query_model(model_name, model_version)
        if err:
            return err

        model, model_sql = result

        err = _load_model(model)
        if err:
            return err

        prediction = Prediction(
            features=features,
            tags=tags or {},
            actual=actual,
            threshold=threshold,
            model=model,
        )

        err = _do_inference(model, prediction)
        if err:
            return err

        shap_values, err = _do_explanation(model, prediction)
        if err:
            return err

        err = _do_mlflow(model_sql, shap_values, prediction.id)
        if err:
            return err

        embedding, err = _do_embedding(model, prediction, umap_config)
        if err:
            return err

        prediction.embeddings = {k: v for k, v in embedding.items() if k != "kind"}

        err = _do_persist(prediction, model)
        if err:
            return err

        _do_arize(prediction, model)

        logger.info("long_prediction_complete prediction_id=%s", prediction.id)
        return {
            "prediction_id": prediction.id,
            "label": _json_safe(prediction.label),
            "shap_values": shap_values,
            "embeddings": {k: v for k, v in embedding.items() if k != "kind"},
        }
