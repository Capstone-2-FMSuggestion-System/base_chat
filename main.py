"""
File khởi động chính cho Medical AI Chat API
File này là điểm vào của ứng dụng, gọi đến module app.main
"""
import uvicorn
from app.config import settings

if __name__ == "__main__":
    # Khởi động ứng dụng FastAPI từ app/main.py
    uvicorn.run(
        "app.main:app", 
        host=settings.API_HOST, 
        port=settings.API_PORT,
        reload=settings.DEBUG_MODE
    ) 