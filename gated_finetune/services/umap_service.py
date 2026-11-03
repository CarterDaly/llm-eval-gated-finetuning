from typing import Any, Dict, Tuple

from gated_finetune.models.prediction import Prediction, Model, UmapConfig
from gated_finetune.services.errors import EmbeddingError


class UmapService:

    def __init__(self) -> None:
        self._umap_cache: Dict[Tuple[str, str], Any] = {}

    def create_embedding(self, model: Model, prediction: Prediction, config: UmapConfig) -> Dict[str, Any]:
        frame = prediction.get_pandas_frame_of_inputs(feature_names=model.feature_names)
        umap_model = model._umap or self._get_or_create_umap(model, frame, config)
        model._umap = umap_model

        try:
            embedding = umap_model.transform(frame)
        except Exception as exc:
            raise EmbeddingError("Failed to compute UMAP embedding.") from exc

        coords = embedding[0]
        axis_names = self._axis_names(config.n_components)
        return {"kind": "inputs", **{axis_names[i]: float(coords[i]) for i in range(config.n_components)}}

    def _get_or_create_umap(self, model: Model, frame, config: UmapConfig):
        cache_key = (
            model.name,
            model.version,
            config.n_neighbors,
            config.min_dist,
            config.n_components,
            config.metric,
            config.random_state,
            config.n_jobs,
        )
        if cache_key in self._umap_cache:
            return self._umap_cache[cache_key]

        umap = self._import_umap()
        try:
            import pandas as pd
            min_rows = config.n_neighbors + 1
            fit_frame = frame if len(frame) >= min_rows else pd.concat(
                [frame] * min_rows, ignore_index=True
            ).iloc[:min_rows]
            umap_kwargs = dict(
                n_neighbors=config.n_neighbors,
                min_dist=config.min_dist,
                n_components=config.n_components,
                metric=config.metric,
                random_state=config.random_state,
            )
            if config.n_jobs is not None:
                umap_kwargs["n_jobs"] = config.n_jobs
            umap_model = umap.UMAP(**umap_kwargs)
            umap_model.fit(fit_frame)
        except Exception as exc:
            raise EmbeddingError("Failed to fit UMAP model.") from exc

        self._umap_cache[cache_key] = umap_model
        return umap_model

    @staticmethod
    def _import_umap():
        try:
            import umap  # type: ignore
        except Exception as exc:
            raise EmbeddingError("umap-learn is not installed.") from exc
        return umap

    @staticmethod
    def _axis_names(n_components: int) -> list[str]:
        if n_components < 2:
            raise EmbeddingError("n_components must be >= 2.")
        base = ["x", "y", "z", "w"]
        if n_components <= len(base):
            return base[:n_components]
        extra = [f"c{i+1}" for i in range(n_components - len(base))]
        return base + extra


umap_service = UmapService()
