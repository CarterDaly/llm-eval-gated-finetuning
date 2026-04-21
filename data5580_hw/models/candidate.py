from typing import Optional

from pydantic import BaseModel

from data5580_hw.models.prediction import UmapConfig


class CandidateScoreRequest(BaseModel):
    model_name: str
    model_version: str
    role_context: str
    umap: Optional[UmapConfig] = None
