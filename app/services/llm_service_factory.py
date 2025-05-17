"""
Factory để quản lý các dịch vụ LLM, hỗ trợ tự động chuyển đổi và fallback
giữa các dịch vụ LLM khác nhau như LlamaClient và OllamaClient.
"""
import logging
import asyncio
import httpx
from typing import Dict, List, Optional, Union, AsyncGenerator, Tuple
from enum import Enum

from app.services.llama_client import LlamaClient
from app.services.ollama_client import OllamaClient
from app.config import settings

logger = logging.getLogger(__name__)

class LLMServiceType(Enum):
    LLAMA_CPP = "llama_cpp"
    OLLAMA = "ollama"
    AUTO = "auto"

class LLMServiceFactory:
    """
    Factory để tạo và quản lý việc truy cập tới các LLM service khác nhau.
    Hỗ trợ tự động chuyển đổi giữa các service khi cần.
    """
    
    def __init__(
        self,
        service_type: Union[str, LLMServiceType] = None,
        llama_url: Optional[str] = None,
        ollama_url: Optional[str] = None,
        default_model: Optional[str] = None
    ):
        """
        Khởi tạo factory với loại service mong muốn.
        
        Args:
            service_type: Loại service (llama_cpp, ollama, hoặc auto để tự động chọn)
            llama_url: URL tùy chọn cho llama.cpp API
            ollama_url: URL tùy chọn cho Ollama API
            default_model: Tên model mặc định khi sử dụng Ollama
        """
        # Xử lý loại dịch vụ
        if service_type is None:
            service_type = getattr(settings, "LLM_SERVICE_TYPE", "auto")
            
        if isinstance(service_type, str):
            try:
                self.service_type = LLMServiceType(service_type.lower())
            except ValueError:
                logger.warning(f"Loại service không hợp lệ: {service_type}, sử dụng AUTO")
                self.service_type = LLMServiceType.AUTO
        else:
            self.service_type = service_type
            
        # Khởi tạo URLs từ tham số hoặc cấu hình
        self.llama_url = llama_url or getattr(settings, "LLAMA_CPP_URL", "http://localhost:8080")
        self.ollama_url = ollama_url or getattr(settings, "OLLAMA_URL", "http://localhost:11434")
        self.default_model = default_model or getattr(settings, "MEDICHAT_MODEL", "medichat-llama3:8b_q4_K_M")
        
        # Khởi tạo các biến trạng thái
        self._llama_client = None
        self._ollama_client = None
        self._active_service = None
        
        # Lưu trữ thông tin về trạng thái mô hình
        self.model_status = {}
        
        logger.info(f"Khởi tạo LLMServiceFactory với mode: {self.service_type.value}")
    
    async def _check_llama_availability(self) -> bool:
        """Kiểm tra xem llama.cpp API có khả dụng không"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.llama_url}/health", timeout=5.0)
                return response.status_code == 200
        except Exception as e:
            logger.warning(f"Không thể kết nối đến llama.cpp API: {str(e)}")
            return False
    
    async def _check_ollama_availability(self) -> bool:
        """Kiểm tra xem Ollama API có khả dụng không"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.ollama_url}/api/version", timeout=5.0)
                return response.status_code == 200
        except Exception as e:
            logger.warning(f"Không thể kết nối đến Ollama API: {str(e)}")
            return False
            
    async def check_ollama_models(self) -> Tuple[bool, str]:
        """
        Kiểm tra xem các mô hình Ollama cần thiết đã được cài đặt chưa
        
        Returns:
            Tuple[bool, str]: Trạng thái và thông báo
        """
        if self._ollama_client is None:
            self._ollama_client = OllamaClient(base_url=self.ollama_url, target_model=self.default_model)
            
        # Kiểm tra mô hình Ollama
        model_available, message = await self._ollama_client.check_model_availability()
        
        # Lưu thông tin trạng thái
        self.model_status = {
            "model_name": self.default_model,
            "available": model_available,
            "message": message
        }
        
        if not model_available:
            # Ghi log và trả về thông báo lỗi
            logger.warning(f"Mô hình Ollama không sẵn sàng: {message}")
        
        return model_available, message
    
    async def initialize(self) -> str:
        """
        Khởi tạo service phù hợp dựa trên cấu hình hoặc tính khả dụng
        
        Returns:
            Tên của service đang hoạt động
        """
        if self.service_type == LLMServiceType.AUTO:
            # Thử kết nối đến cả hai và ưu tiên llama.cpp
            is_llama_available = await self._check_llama_availability()
            is_ollama_available = await self._check_ollama_availability()
            
            if is_llama_available:
                self._active_service = LLMServiceType.LLAMA_CPP
                logger.info("Tự động chọn dịch vụ llama.cpp")
            elif is_ollama_available:
                self._active_service = LLMServiceType.OLLAMA
                logger.info("Tự động chọn dịch vụ Ollama")
                # Kiểm tra mô hình Ollama khi service được chọn
                await self.check_ollama_models()
            else:
                raise ConnectionError("Không thể kết nối đến bất kỳ dịch vụ LLM nào")
        else:
            self._active_service = self.service_type
            
            # Kiểm tra kết nối đến service đã chọn
            if self._active_service == LLMServiceType.LLAMA_CPP:
                is_available = await self._check_llama_availability()
                if not is_available:
                    # Fallback sang Ollama nếu llama.cpp không khả dụng
                    is_ollama_available = await self._check_ollama_availability()
                    if is_ollama_available:
                        logger.warning("Kết nối đến llama.cpp thất bại, chuyển sang Ollama")
                        self._active_service = LLMServiceType.OLLAMA
                        # Kiểm tra mô hình Ollama khi chuyển đổi service
                        await self.check_ollama_models()
                    else:
                        raise ConnectionError("Không thể kết nối đến bất kỳ dịch vụ LLM nào")
            else:  # OLLAMA
                is_available = await self._check_ollama_availability()
                if not is_available:
                    # Fallback sang llama.cpp nếu Ollama không khả dụng
                    is_llama_available = await self._check_llama_availability()
                    if is_llama_available:
                        logger.warning("Kết nối đến Ollama thất bại, chuyển sang llama.cpp")
                        self._active_service = LLMServiceType.LLAMA_CPP
                    else:
                        raise ConnectionError("Không thể kết nối đến bất kỳ dịch vụ LLM nào")
                else:
                    # Kiểm tra mô hình Ollama khi service được chọn
                    await self.check_ollama_models()
        
        return self._active_service.value
    
    def get_service(self):
        """Lấy instance của service đang hoạt động"""
        if self._active_service is None:
            raise RuntimeError("Service chưa được khởi tạo. Vui lòng gọi initialize() trước")
            
        if self._active_service == LLMServiceType.LLAMA_CPP:
            if self._llama_client is None:
                self._llama_client = LlamaClient(base_url=self.llama_url)
            return self._llama_client
        else:  # OLLAMA
            if self._ollama_client is None:
                self._ollama_client = OllamaClient(base_url=self.ollama_url)
            return self._ollama_client
    
    async def generate_response(self, messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
        """
        Gửi yêu cầu đến LLM service đã chọn và trả về phản hồi dạng streaming
        
        Args:
            messages: Danh sách tin nhắn theo định dạng [{"role": "user", "content": "..."}]
        
        Yields:
            Từng phần nội dung phản hồi từ AI
        """
        if self._active_service is None:
            try:
                await self.initialize()
            except ConnectionError as e:
                logger.error(f"Lỗi kết nối: {str(e)}")
                yield "Xin lỗi, không thể kết nối tới dịch vụ trí tuệ nhân tạo. Vui lòng thử lại sau."
                return
        
        service = self.get_service()
        try:
            async for chunk in service.generate_response(messages):
                yield chunk
        except Exception as e:
            logger.error(f"Lỗi khi gọi API: {str(e)}")
            yield "Xin lỗi, có lỗi xảy ra khi xử lý yêu cầu của bạn. Vui lòng thử lại sau."
    
    async def get_full_response(self, messages: List[Dict[str, str]]) -> str:
        """
        Gửi yêu cầu đến LLM service đã chọn và trả về phản hồi đầy đủ
        
        Args:
            messages: Danh sách tin nhắn theo định dạng [{"role": "user", "content": "..."}]
            
        Returns:
            Phản hồi đầy đủ từ AI
        """
        if self._active_service is None:
            try:
                await self.initialize()
            except ConnectionError as e:
                logger.error(f"Lỗi kết nối: {str(e)}")
                return "Xin lỗi, không thể kết nối tới dịch vụ trí tuệ nhân tạo. Vui lòng thử lại sau."
        
        service = self.get_service()
        try:
            return await service.get_full_response(messages)
        except Exception as e:
            logger.error(f"Lỗi khi gọi API: {str(e)}")
            return "Xin lỗi, có lỗi xảy ra khi xử lý yêu cầu của bạn. Vui lòng thử lại sau." 