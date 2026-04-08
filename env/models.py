from pydantic import BaseModel
from typing import List, Optional, Literal

Priority = Literal["low", "medium", "high"]
Category = Literal["spam", "work", "support", "sales", "other"]
ActionType = Literal["classify", "prioritize", "respond", "delete", "noop"]

class Email(BaseModel):
    id: str
    subject: str
    body: str
    priority: Priority
    category: Category
    handled: bool = False

class Observation(BaseModel):
    emails: List[Email]
    step_count: int

class Action(BaseModel):
    action_type: ActionType
    email_id: Optional[str] = None
    predicted_category: Optional[Category] = None
    predicted_priority: Optional[Priority] = None
    response_text: Optional[str] = None

class StepResult(BaseModel):
    observation: Observation
    reward: float
    done: bool
    info: dict