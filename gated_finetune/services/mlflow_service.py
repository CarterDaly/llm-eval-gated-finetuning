import os
from typing import Any, Dict, Optional

from gated_finetune.services.errors import MlflowLogError


class MlflowService:
    """Logs per-model artifacts to the MLflow tracking server.

    The server is chosen only by MLFLOW_TRACKING_URI, which MLflow reads itself.
    It is required: without it MLflow silently falls back to a local ./mlflow.db,
    and the artifact would end up somewhere nobody looks.
    """

    def log_shap_once(
        self,
        *,
        run_id: str,
        model_name: str,
        model_version: str,
        shap_values: Dict[str, float],
        prediction_id: str,
        existing_artifact_path: Optional[str] = None,
    ) -> Optional[str]:
        """Log SHAP values from a model's first inference; a no-op once logged."""
        return self._log_first_inference(
            kind="shap",
            label="SHAP values",
            values_key="shap_values",
            values=shap_values,
            run_id=run_id,
            model_name=model_name,
            model_version=model_version,
            prediction_id=prediction_id,
            existing_artifact_path=existing_artifact_path,
        )

    def log_embedding_once(
        self,
        *,
        run_id: str,
        model_name: str,
        model_version: str,
        embedding: Dict[str, float],
        prediction_id: str,
        existing_artifact_path: Optional[str] = None,
    ) -> Optional[str]:
        """Log UMAP coordinates from a model's first inference; a no-op once logged."""
        return self._log_first_inference(
            kind="embeddings",
            label="embedding",
            values_key="embedding",
            values=embedding,
            run_id=run_id,
            model_name=model_name,
            model_version=model_version,
            prediction_id=prediction_id,
            existing_artifact_path=existing_artifact_path,
        )

    def _log_first_inference(
        self,
        *,
        kind: str,
        label: str,
        values_key: str,
        values: Dict[str, Any],
        run_id: str,
        model_name: str,
        model_version: str,
        prediction_id: str,
        existing_artifact_path: Optional[str],
    ) -> Optional[str]:
        if existing_artifact_path:
            return None
        if not run_id:
            raise MlflowLogError("MLflow run id is missing for model.")
        if not os.getenv("MLFLOW_TRACKING_URI"):
            raise MlflowLogError("MLFLOW_TRACKING_URI is not set.")

        mlflow = self._import_mlflow()
        artifact_path = f"{kind}/{model_name}/{model_version}/first_inference.json"
        payload = {
            "model_name": model_name,
            "model_version": model_version,
            "prediction_id": prediction_id,
            values_key: values,
        }

        try:
            mlflow.tracking.MlflowClient().log_dict(run_id, payload, artifact_path)
        except Exception as exc:
            raise MlflowLogError(f"Failed to log {label} to MLflow.") from exc

        return artifact_path

    @staticmethod
    def _import_mlflow():
        # Imported lazily: mlflow takes seconds to import and most requests
        # never reach this path (it only runs on a model's first inference).
        try:
            import mlflow  # type: ignore
        except Exception as exc:
            raise MlflowLogError("MLflow is not installed.") from exc
        return mlflow


mlflow_service = MlflowService()
