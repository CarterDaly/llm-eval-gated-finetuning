import uuid

from pydantic import BaseModel, Field, EmailStr
from data5580_hw.models.base import LocalBaseModel, get_id


class User(BaseModel):
    name: str
    email: EmailStr

    id: str = Field(default_factory=get_id)


class UserUpdate(BaseModel):
    name: str | None = None
    email: EmailStr | None = None
