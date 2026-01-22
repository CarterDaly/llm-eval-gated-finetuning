import uuid

from pydantic import BaseModel, Field, EmailStr


def get_id():
    return uuid.uuid4().hex


class User(BaseModel):
    name: str
    email: EmailStr

    id: str = Field(default_factory=get_id)