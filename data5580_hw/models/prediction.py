from typing import Dict, Union, Optional

from datetime import datetime

from pydantic import Field

from data5580_hw.models.base import LocalBaseModel


class Model(BaseModel):
    name: str
    version: str
    id: str = Field(default_factory=get_id)

class Prediction(BaseModel):
    features: Dict
    tags: Dict
    label: Union[int, float, str]
    actual: Union[int, float, str]
    threshold: Optional[float]

    model: Model

    created: Field(default_factory=datetime.now)
    updated: Field(default_factory=datetime.now)
    id: str = Field(default_factory=get_id)