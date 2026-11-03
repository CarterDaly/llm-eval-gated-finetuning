import os
from typing import Dict, Optional

from gated_finetune.services.errors import MlflowLogError


class MlflowService:
    def log_shap_once(
        self,
        *,
        run_id: str,
        model_name: str,
        model_version: str,
        shap_values: Dict[str, float],
        prediction_id: str,
        existing_artifact_path: Optional[str] = None,
        tracking_uri: Optional[str] = None,
    ) -> Optional[str]:
        if existing_artifact_path:
            return None
        if not run_id:
            raise MlflowLogError("MLflow run id is missing for model.")

        mlflow = self._import_mlflow()
        uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
        if uri:
            mlflow.set_tracking_uri(uri)

        artifact_path = f"shap/{model_name}/{model_version}/first_inference.json"
        payload = {
            "model_name": model_name,
            "model_version": model_version,
            "prediction_id": prediction_id,
            "shap_values": shap_values,
        }

        try:
            client = mlflow.tracking.MlflowClient()
            client.log_dict(run_id, payload, artifact_path)
        except Exception as exc:
            raise MlflowLogError("Failed to log SHAP values to MLflow.") from exc

        return artifact_path

    def log_embedding_once(
        self,
        *,
        run_id: str,
        model_name: str,
        model_version: str,
        embedding: Dict[str, float],
        prediction_id: str,
        existing_artifact_path: Optional[str] = None,
        tracking_uri: Optional[str] = None,
    ) -> Optional[str]:
        """Log UMAP embedding coordinates to MLflow on the first inference.

        Mirrors the idempotency pattern of log_shap_once: if an artifact path
        is already stored on the model record, this call is a no-op.
        """
        if existing_artifact_path:
            return None
        if not run_id:
            raise MlflowLogError("MLflow run id is missing for model.")

        mlflow = self._import_mlflow()
        uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI")
        if uri:
            mlflow.set_tracking_uri(uri)

        artifact_path = f"embeddings/{model_name}/{model_version}/first_inference.json"
        payload = {
            "model_name": model_name,
            "model_version": model_version,
            "prediction_id": prediction_id,
            "embedding": embedding,
        }

        try:
            client = mlflow.tracking.MlflowClient()
            client.log_dict(run_id, payload, artifact_path)
        except Exception as exc:
            raise MlflowLogError("Failed to log embedding to MLflow.") from exc

        return artifact_path

    @staticmethod
    def _import_mlflow():
        try:
            import mlflow  # type: ignore
        except Exception as exc:
            raise MlflowLogError("MLflow is not installed.") from exc
        return mlflow


mlflow_service = MlflowService()
