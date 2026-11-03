from typing import List, Optional

from pydantic import BaseModel


class ModelIngestRequest(BaseModel):
    model_name: str
    model_version: str
    model_type: str = "generic"
    feature_names: List[str]
    mlflow_run_id: Optional[str] = None
