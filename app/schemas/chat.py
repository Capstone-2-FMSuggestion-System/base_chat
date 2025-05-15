from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime


class MessageModel(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[int] = None


class ChatResponse(BaseModel):
    conversation_id: int
    user_message: Dict[str, str]
    assistant_message: Dict[str, str]


class NewChatResponse(BaseModel):
    conversation_id: int
    created_at: datetime

    class Config:
        orm_mode = True


class ChatContentResponse(BaseModel):
    conversation_id: int
    messages: List[MessageModel]
    has_summary: Optional[bool] = False
    summary: Optional[str] = None
    summary_updated_at: Optional[datetime] = None

    class Config:
        orm_mode = True 