import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional, Union

from data5580_hw.services.errors import ArizeLogError

logger = logging.getLogger(__name__)


class ArizeService:
    """Adapter for the Arize model monitoring platform.

    Wraps the Arize Python SDK to log inferences and actuals.
    Controlled via environment variables:
      ARIZE_API_KEY      - Arize API key (required when enabled)
      ARIZE_SPACE_KEY    - Arize space key (required when enabled)
      ARIZE_ENVIRONMENT  - production | staging | development (default: development)
      ARIZE_ENABLED      - true | false  (default: true)
    """

    def log_inference(
        self,
        *,
        prediction_id: str,
        model_name: str,
        model_version: str,
        prediction_label: Union[int, float, str],
        features: Dict[str, Any],
        actual_label: Optional[Union[int, float, str]] = None,
        shap_values: Optional[Dict[str, float]] = None,
        timestamp: Optional[datetime] = None,
        environment: Optional[str] = None,
    ) -> bool:
        """Log a model inference to Arize.

        Returns True if the record was submitted, False if monitoring is
        disabled.  Raises ArizeLogError on configuration or API errors.
        """
        if not self._is_enabled():
            return False

        api_key, space_key = self._get_credentials()
        Client, ModelTypes, Environments = self._import_arize()

        env_name = environment or os.getenv("ARIZE_ENVIRONMENT", "development")
        arize_env = self._resolve_environment(Environments, env_name)
        client = Client(space_key=space_key, api_key=api_key)

        kwargs: Dict[str, Any] = dict(
            prediction_id=prediction_id,
            model_id=model_name,
            model_version=model_version,
            model_type=ModelTypes.SCORE_CATEGORICAL,
            environment=arize_env,
            prediction_label=prediction_label,
            features=features,
        )
        if actual_label is not None:
            kwargs["actual_label"] = actual_label
        if shap_values:
            kwargs["shap_values"] = shap_values
        if timestamp is not None:
            kwargs["prediction_timestamp"] = int(timestamp.timestamp())

        try:
            response = client.log(**kwargs)
        except Exception as exc:
            raise ArizeLogError(f"Failed to log inference to Arize: {exc}") from exc

        if response.status_code != 200:
            raise ArizeLogError(
                f"Arize returned status {response.status_code}: {response.text}"
            )

        return True

    def log_actual(
        self,
        *,
        prediction_id: str,
        model_name: str,
        model_version: str,
        actual_label: Union[int, float, str],
        environment: Optional[str] = None,
    ) -> bool:
        """Log a ground-truth actual value to Arize for a prior prediction.

        Returns True if submitted, False if monitoring is disabled.
        Raises ArizeLogError on configuration or API errors.
        """
        if not self._is_enabled():
            return False

        api_key, space_key = self._get_credentials()
        Client, ModelTypes, Environments = self._import_arize()

        env_name = environment or os.getenv("ARIZE_ENVIRONMENT", "development")
        arize_env = self._resolve_environment(Environments, env_name)
        client = Client(space_key=space_key, api_key=api_key)

        try:
            response = client.log(
                prediction_id=prediction_id,
                model_id=model_name,
                model_version=model_version,
                model_type=ModelTypes.SCORE_CATEGORICAL,
                environment=arize_env,
                actual_label=actual_label,
            )
        except Exception as exc:
            raise ArizeLogError(f"Failed to log actual to Arize: {exc}") from exc

        if response.status_code != 200:
            raise ArizeLogError(
                f"Arize returned status {response.status_code}: {response.text}"
            )

        return True

    @staticmethod
    def _is_enabled() -> bool:
        return os.getenv("ARIZE_ENABLED", "true").lower() not in ("false", "0", "no")

    def _get_credentials(self):
        api_key = os.getenv("ARIZE_API_KEY")
        space_key = os.getenv("ARIZE_SPACE_KEY")
        if not api_key:
            raise ArizeLogError("ARIZE_API_KEY environment variable is not set.")
        if not space_key:
            raise ArizeLogError("ARIZE_SPACE_KEY environment variable is not set.")
        return api_key, space_key

    @staticmethod
    def _resolve_environment(Environments, env_name: str):
        mapping = {
            "production": Environments.PRODUCTION,
            "staging": Environments.STAGING,
            "development": Environments.TRAINING,
        }
        return mapping.get(env_name.lower(), Environments.PRODUCTION)

    @staticmethod
    def _import_arize():
        try:
            from arize.api import Client  # type: ignore
            from arize.utils.types import Environments, ModelTypes  # type: ignore
        except Exception as exc:
            raise ArizeLogError("Arize SDK is not installed.") from exc
        return Client, ModelTypes, Environments


arize_service = ArizeService()
