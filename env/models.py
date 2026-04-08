from pydantic import BaseModel
from typing import List, Optional, Literal

Priority = Literal["low","medium","high"]
Category = Literal["spam","work","support","sales"]

class Email(BaseModel):
    id: str
    subject: str
    body: str
    priority: Priority
    category: Category
    handled: bool = False

class Observation(BaseModel):
    emails: List[Email]
    step: int

class Action(BaseModel):
    action_type: str
    email_id: Optional[str] = None
    predicted_category: Optional[str] = None
    predicted_priority: Optional[str] = None
    response_text: Optional[str] = None