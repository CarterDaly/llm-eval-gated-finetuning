from typing import Any, Dict, Tuple

from data5580_hw.models.prediction import Prediction, Model, Explanation, Explanations
from data5580_hw.services.errors import ExplanationError


class ExplainerService:

    def __init__(self) -> None:
        self._explainer_cache: Dict[Tuple[str, str], Any] = {}

    def create_explanation(self, model: Model, prediction: Prediction) -> Explanations:
        if model._model is None:
            raise ExplanationError("Model is not loaded.")

        frame = prediction.get_pandas_frame_of_inputs(feature_names=model.feature_names)
        explainer = model._explainer or self._get_or_create_explainer(model, frame)
        model._explainer = explainer

        try:
            shap_result = explainer(frame)
        except Exception as exc:
            raise ExplanationError("Failed to compute SHAP values.") from exc

        feature_names = prediction.get_feature_names(model.feature_names)
        shap_values = self._to_shap_mapping(shap_result, feature_names)

        return Explanations(
            explanations=[
                Explanation(name="shap_values", values=shap_values)
            ]
        )

    def _get_or_create_explainer(self, model: Model, frame):
        cache_key = (model.name, model.version)
        if cache_key in self._explainer_cache:
            return self._explainer_cache[cache_key]

        shap = self._import_shap()
        try:
            explainer = self._build_explainer(shap, model, frame)
        except Exception as exc:
            raise ExplanationError("Unsupported model type for SHAP.") from exc

        self._explainer_cache[cache_key] = explainer
        return explainer

    @staticmethod
    def _import_shap():
        try:
            import shap  # type: ignore
        except Exception as exc:
            raise ExplanationError("SHAP is not installed.") from exc
        return shap

    @staticmethod
    def _build_explainer(shap, model: Model, frame):
        model_type = (model.type or "generic").lower()
        if model_type == "tree":
            return shap.TreeExplainer(model._model)
        if model_type == "linear":
            return shap.LinearExplainer(model._model, frame)
        if model_type == "deep":
            return shap.DeepExplainer(model._model, frame)
        if model_type == "kernel":
            return shap.KernelExplainer(model._model.predict, frame)
        return shap.Explainer(model._model, frame)

    @staticmethod
    def _to_shap_mapping(shap_result, feature_names: list[str]) -> Dict[str, float]:
        try:
            import numpy as np
        except Exception as exc:
            raise ExplanationError("NumPy is required for SHAP value parsing.") from exc

        values = getattr(shap_result, "values", shap_result)
        if isinstance(values, list):
            values = values[0]

        arr = np.asarray(values)
        if arr.ndim == 1:
            vector = arr
        elif arr.ndim == 2:
            vector = arr[0]
        elif arr.ndim == 3:
            vector = arr[0][0]
        else:
            raise ExplanationError("Unsupported SHAP output shape.")

        if len(vector) != len(feature_names):
            raise ExplanationError("SHAP output does not match feature count.")

        return {name: float(value) for name, value in zip(feature_names, vector)}
    

explainer_service = ExplainerService()
