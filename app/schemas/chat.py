from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from datetime import datetime


class MessageModel(BaseModel):
    role: str
    content: str


class ProductDisplayModel(BaseModel):
    id: int
    name: str
    price: float
    original_price: float
    description: Optional[str] = None
    image: Optional[str] = None
    unit: Optional[str] = None
    stock_quantity: int
    category_id: Optional[int] = None
    discount_percentage: Optional[float] = None

    @property
    def discount_percentage(self) -> Optional[float]:
        if self.original_price > 0 and self.price < self.original_price:
            return round((1 - self.price / self.original_price) * 100, 1)
        return None


class ProductSearchResponse(BaseModel):
    conversation_id: int
    search_query: str
    found_products: List[ProductDisplayModel]
    total_found: int
    ingredients_processed: List[str]
    ingredients_not_found: List[str]


class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[int] = None


class ChatResponse(BaseModel):
    conversation_id: int
    user_message: Dict[str, str]
    assistant_message: Dict[str, str]
    has_summary: Optional[bool] = False
    summary: Optional[str] = None
    health_data: Optional[Dict[str, Any]] = None
    limit_reached: bool = False
    message_count: int = 0
    available_products: Optional[List[Dict[str, Any]]] = []


class NewChatResponse(BaseModel):
    conversation_id: int
    created_at: datetime
    welcome_message: Optional[str] = None

    class Config:
        orm_mode = True


class ChatContentResponse(BaseModel):
    conversation_id: int
    messages: List[MessageModel] 
    has_summary: Optional[bool] = False
    summary: Optional[str] = None
    available_products: Optional[List[Dict[str, Any]]] = []

    class Config:
        orm_mode = True 