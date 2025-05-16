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
    
    # Chat History Management
    MAX_HISTORY_MESSAGES: int = 30
    SUMMARY_THRESHOLD: int = 5  # Tóm tắt khi có nhiều hơn số tin nhắn này chưa được tóm tắt
    
    # Gemini API
    GEMINI_API_KEY: str = ""
    GEMINI_API_URL: str = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite:generateContent"
    
    # Security
    SECRET_KEY: str
    ALGORITHM: str
    ACCESS_TOKEN_EXPIRE_MINUTES: int
    
    # API
    API_HOST: str
    API_PORT: int
    DEBUG_MODE: bool
    API_TIMEOUT: int
    
    def __init__(self, **data):
        super().__init__(**data)
        if not self.SQLALCHEMY_DATABASE_URL:
            self.SQLALCHEMY_DATABASE_URL = f"mysql+pymysql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}?charset=utf8mb4"
    
    class Config:
        env_file = ".env"


settings = Settings() 