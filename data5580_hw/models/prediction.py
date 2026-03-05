from typing import Any, Dict, List, Optional, Union

from datetime import datetime

from pydantic import BaseModel, Field, PrivateAttr

from data5580_hw.models.base import LocalBaseModel


class PredictionRequest(BaseModel):
    features: Dict[str, Union[int, float, str]]
    tags: Dict[str, Union[int, float, str]] = Field(default_factory=dict)
    actual: Optional[Union[int, float, str]] = None
    threshold: Optional[float] = None


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

    _model: Any = PrivateAttr(default=None)
    _explainer: Any = PrivateAttr(default=None)


class Prediction(LocalBaseModel):
    features: Dict[str, Union[int, float, str]]
    tags: Dict[str, Union[int, float, str]] = Field(default_factory=dict)
    explanations: Optional[Explanations] = None
    label: Optional[Union[int, float, str]] = None
    actual: Optional[Union[int, float, str]] = None
    threshold: Optional[float] = None

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
