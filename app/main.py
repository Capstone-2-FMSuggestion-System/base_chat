import logging
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from typing import Dict, Any

from app.api.chat import router as chat_router
from app.api.auth import router as auth_router
from app.config import settings
from app.db.database import engine, Base
from app.services.llm_service_factory import LLMServiceFactory, LLMServiceType

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

# Tạo instance logger
logger = logging.getLogger(__name__)

# Khởi tạo LLM Service Factory
llm_factory = LLMServiceFactory()

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


@app.get("/api/llm/status", tags=["llm"])
async def get_llm_status():
    """Lấy trạng thái dịch vụ LLM và mô hình"""
    try:
        # Lấy service hiện tại
        active_service = await llm_factory.initialize()
        
        # Chuẩn bị thông tin phản hồi
        status = {
            "llm_service": active_service,
            "service_available": active_service is not None,
        }
        
        # Thêm thông tin về mô hình nếu đang sử dụng Ollama
        if active_service == "ollama" and hasattr(llm_factory, "model_status"):
            status.update({
                "model_name": llm_factory.model_status.get("model_name"),
                "model_available": llm_factory.model_status.get("available", False),
                "model_message": llm_factory.model_status.get("message", "")
            })
            
        return status
    except Exception as e:
        logger.error(f"Lỗi khi kiểm tra trạng thái LLM: {str(e)}")
        return {
            "llm_service": "unavailable",
            "service_available": False,
            "error": str(e)
        }


@app.on_event("startup")
async def startup_event():
    """Khởi động các dịch vụ và kiểm tra cấu hình khi ứng dụng khởi động"""
    logger.info("Khởi động ứng dụng và kiểm tra các dịch vụ...")
    try:
        # Khởi tạo và kiểm tra dịch vụ LLM
        active_service = await llm_factory.initialize()
        logger.info(f"Dịch vụ LLM đã khởi động: {active_service}")
        
        # Kiểm tra và thông báo về mô hình Ollama nếu cần
        if active_service == "ollama" and hasattr(llm_factory, "model_status"):
            model_name = llm_factory.model_status.get("model_name")
            model_available = llm_factory.model_status.get("available", False)
            model_message = llm_factory.model_status.get("message", "")
            
            if not model_available:
                logger.warning(f"CẢNH BÁO: {model_message}")
                logger.warning(f"Sử dụng lệnh sau để tải mô hình: ollama pull {model_name}")
            else:
                logger.info(f"Mô hình LLM: {model_name} - Trạng thái: Sẵn sàng")
    except Exception as e:
        logger.error(f"Lỗi khi khởi động dịch vụ LLM: {str(e)}")
        logger.error("Hệ thống có thể hoạt động không ổn định. Vui lòng kiểm tra kết nối đến dịch vụ LLM.") 