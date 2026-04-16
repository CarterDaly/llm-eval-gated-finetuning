from typing import Any, Dict, List, Optional, Union

from datetime import datetime

from pydantic import BaseModel, Field, PrivateAttr, field_validator

from data5580_hw.models.base import LocalBaseModel


class UmapConfig(BaseModel):
    n_neighbors: int = 15
    min_dist: float = 0.1
    n_components: int = 2
    metric: str = "euclidean"
    random_state: Optional[int] = 42
    n_jobs: Optional[int] = None

    @field_validator("n_neighbors")
    @classmethod
    def _validate_n_neighbors(cls, value: int) -> int:
        if value < 2:
            raise ValueError("n_neighbors must be >= 2.")
        return value

    @field_validator("min_dist")
    @classmethod
    def _validate_min_dist(cls, value: float) -> float:
        if value < 0:
            raise ValueError("min_dist must be >= 0.")
        return value

    @field_validator("n_components")
    @classmethod
    def _validate_n_components(cls, value: int) -> int:
        if value < 2:
            raise ValueError("n_components must be >= 2.")
        return value


class PredictionRequest(BaseModel):
    features: Dict[str, Union[int, float, str]]
    tags: Dict[str, Union[int, float, str]] = Field(default_factory=dict)
    actual: Optional[Union[int, float, str]] = None
    threshold: Optional[float] = None
    umap: Optional[UmapConfig] = None


class ActualUpdateRequest(BaseModel):
    actual: Union[int, float, str]


class Explanation(LocalBaseModel):
    name: str
    values: Dict[str, float]


class Explanations(LocalBaseModel):
    explanations: List[Explanation] = Field(default_factory=list)

    @classmethod
    def from_explanations(cls, explanation_rows: Optional[List[Any]]) -> "Explanations":
        if not explanation_rows:
            return cls(explanations=[])
        return cls(
            explanations=[
                Explanation(id=row.id, name=row.name, values=row.values)
                for row in explanation_rows
            ]
        )


class Model(LocalBaseModel):
    name: str
    version: str
    type: Optional[str] = None
    artifact_path: Optional[str] = None
    feature_names: Optional[List[str]] = None
    mlflow_run_id: Optional[str] = None

    _model: Any = PrivateAttr(default=None)
    _explainer: Any = PrivateAttr(default=None)
    _umap: Any = PrivateAttr(default=None)


class Prediction(LocalBaseModel):
    features: Dict[str, Union[int, float, str]]
    tags: Dict[str, Union[int, float, str]] = Field(default_factory=dict)
    explanations: Optional[Explanations] = None
    label: Optional[Union[int, float, str]] = None
    actual: Optional[Union[int, float, str]] = None
    threshold: Optional[float] = None
    embeddings: Optional[Dict[str, float]] = None

    model: Optional[Model] = None

    created: datetime = Field(default_factory=datetime.now)
    updated: datetime = Field(default_factory=datetime.now)

    def get_feature_names(self, override: Optional[List[str]] = None) -> List[str]:
        if override:
            return [name for name in override if name in self.features]
        return list(self.features.keys())

    def get_pandas_frame_of_inputs(self, feature_names: Optional[List[str]] = None):
        names = self.get_feature_names(feature_names)
        row = {name: self.features.get(name) for name in names}
        try:
            import pandas as pd
        except ImportError:
            return [list(row.values())]
        return pd.DataFrame([row])
