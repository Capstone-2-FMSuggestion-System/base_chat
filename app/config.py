import os
import logging
from pydantic_settings import BaseSettings
from typing import Optional, Dict, Any, List

# Thiết lập cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
)

class Settings(BaseSettings):
    # Database
    DB_USER: str
    DB_PASSWORD: str
    DB_HOST: str
    DB_PORT: int
    DB_NAME: str
    SQLALCHEMY_DATABASE_URL: Optional[str] = None
    
    # Redis
    REDIS_HOST: str
    REDIS_PORT: int
    REDIS_DB: int
    REDIS_PASSWORD: Optional[str] = None
    
    # LLM Service
    LLAMA_CPP_URL: str
    OLLAMA_URL: str
    LLM_SERVICE_TYPE: str
    LLM_MAX_TOKENS: int = 300
    MEDICHAT_MODEL: str = "monotykamary/medichat-llama3:8b_q4_K_M"
    OLLAMA_TIMEOUT_SECONDS: float = 300.0  # ⭐ TIMEOUT LINH HOẠT CHO OLLAMA (5 phút)
    
    # Chat History Management
    MAX_HISTORY_MESSAGES: int = 30
    SUMMARY_THRESHOLD: int = 5  # Tóm tắt khi có nhiều hơn số tin nhắn này chưa được tóm tắt
    
    # Gemini API
    GEMINI_API_KEY: str = ""
    GEMINI_API_KEYS_LIST: Optional[str] = None  # For API key load balancing
    GEMINI_API_URL: str = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite:generateContent"
    GEMINI_MODEL: str = "gemini-2.0-flash-lite"
    GEMINI_MAX_PROMPT_LENGTH: int = 900  # Giới hạn độ dài prompt cho Medichat
    GEMINI_MAX_PROMPT_WORDS_WITH_CONTEXT: int = 500  # ⭐ GIỚI HẠN TỪ CHO PROMPT MEDICHAT VỚI CONTEXT
    
    # Pinecone Vector Database
    PRODUCT_DB_PINECONE_API_KEY: str = ""
    RECIPE_DB_PINECONE_API_KEY: str = ""
    PINECONE_INDEX_NAME: str = "product-index"
    PINECONE_ENVIRONMENT: str = "gcp-starter"
    
    # Prompt Templates
    MEDICHAT_SYSTEM_PROMPT: str = """Bạn là chuyên gia dinh dưỡng, chỉ cung cấp gợi ý món ăn. Khi người dùng chia sẻ tình trạng sức khỏe, hãy gợi ý 2-3 món ăn phù hợp bằng tiếng Việt, mỗi món kèm 1 câu giải thích ngắn gọn. KHÔNG đề cập đến thuốc."""
    
    # Chat Flow Settings
    VALID_SCOPES: List[str] = [
        "sức khỏe",
        "dinh dưỡng",
        "món ăn",
        "bệnh lý",
        "chế độ ăn uống",
        "dị ứng",
        "thực phẩm"
    ]
    
    # Security
    SECRET_KEY: str
    ALGORITHM: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int
    BACKEND_AUTH_VERIFY_URL: str
    
    # API
    API_HOST: str
    API_PORT: int
    DEBUG_MODE: bool
    API_TIMEOUT: int
    
    # Backend URL
    BACKEND_URL: str = "http://localhost:8000"
    
    def __init__(self, **data):
        super().__init__(**data)
        if not self.SQLALCHEMY_DATABASE_URL:
            self.SQLALCHEMY_DATABASE_URL = f"mysql+pymysql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}?charset=utf8mb4"
    
    class Config:
        env_file = ".env"


settings = Settings() 