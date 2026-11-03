from typing import Any, Optional

from pydantic import BaseModel


class TaskStatus(BaseModel):
    task_id: str
    status: str
    result: Optional[Any] = None
    error: Optional[str] = None
