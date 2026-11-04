import os
import pickle
from typing import Any, Dict, Tuple, Union

from gated_finetune.models.prediction import Prediction, Model
from gated_finetune.services.errors import InferenceError, ModelLoadError


class ModelService:

    def __init__(self) -> None:
        self._model_cache: Dict[Tuple[str, str], Dict[str, Any]] = {}

    def load_model(self, model: Model) -> Model:
        cache_key = (model.name, model.version)
        if cache_key in self._model_cache:
            cached = self._model_cache[cache_key]
            model._model = cached["model"]
            model.type = model.type or cached.get("type")
            model.feature_names = model.feature_names or cached.get("feature_names")
            return model

        if not model.artifact_path:
            raise ModelLoadError("Model artifact path is missing.")
        if not os.path.isfile(model.artifact_path):
            raise ModelLoadError(f"Model artifact not found at '{model.artifact_path}'.")

        try:
            with open(model.artifact_path, "rb") as handle:
                loaded_model = pickle.load(handle)
        except Exception as exc:
            raise ModelLoadError("Failed to load model artifact.") from exc

        model._model = loaded_model
        model.feature_names = model.feature_names or self._infer_feature_names(loaded_model)
        model.type = model.type or self._infer_model_type(loaded_model)

        self._model_cache[cache_key] = {
            "model": loaded_model,
            "type": model.type,
            "feature_names": model.feature_names,
        }

        return model

    def create_inference(self, model: Model, prediction: Prediction) -> Union[str, int, float]:
        if model._model is None:
            raise InferenceError("Model is not loaded.")

        frame = prediction.get_pandas_frame_of_inputs(feature_names=model.feature_names)
        try:
            output = model._model.predict(frame)
        except Exception as exc:
            raise InferenceError("Failed to run inference.") from exc

        try:
            return output[0]
        except Exception:
            return output

    @staticmethod
    def _infer_feature_names(model_handle: Any) -> list[str] | None:
        for attr in ("feature_names_in_", "feature_names_"):
            names = getattr(model_handle, attr, None)
            if names is not None:
                return list(names)
        return None

    @staticmethod
    def _infer_model_type(model_handle: Any) -> str:
        class_name = model_handle.__class__.__name__.lower()
        if hasattr(model_handle, "tree_") or hasattr(model_handle, "estimators_"):
            return "tree"
        if hasattr(model_handle, "coef_"):
            return "linear"
        if "xgb" in class_name or "lgbm" in class_name or "randomforest" in class_name:
            return "tree"
        if "linear" in class_name or "logistic" in class_name:
            return "linear"
        if "torch" in class_name or "keras" in class_name or "tensorflow" in class_name:
            return "deep"
        return "generic"


model_service = ModelService()
