import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from app.api.chat import router as chat_router
from app.api.auth import router as auth_router
from app.config import settings
from app.db.database import engine, Base

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

# Tạo bảng trong cơ sở dữ liệu
Base.metadata.create_all(bind=engine)

# Khởi tạo ứng dụng FastAPI
app = FastAPI(
    title="Medical AI Chatbot API",
    description="API cho hệ thống trò chuyện y tế thông minh",
    version="1.0.0"
)

# Thiết lập CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Trong môi trường production nên hạn chế các nguồn
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Type", "Content-Length", "Access-Control-Allow-Origin"],
)

# Đăng ký router
app.include_router(auth_router, prefix="/api", tags=["auth"])
app.include_router(chat_router, prefix="/api", tags=["chat"])


@app.get("/", tags=["health"])
async def root():
    """Kiểm tra trạng thái API"""
    return {"message": "Medical AI Chat API đang hoạt động"}


if __name__ == "__main__":
    uvicorn.run(
        "app.main:app", 
        host=settings.API_HOST, 
        port=settings.API_PORT,
        reload=settings.DEBUG_MODE
    ) 