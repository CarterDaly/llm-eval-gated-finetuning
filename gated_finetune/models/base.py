import uuid
from pydantic import BaseModel, Field


def get_id():
    return uuid.uuid4().hex


class LocalBaseModel(BaseModel):

    id: str = Field(default_factory=get_id)
