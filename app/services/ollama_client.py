import httpx
from typing import List, Dict, Any, AsyncGenerator
import json
import logging
from app.config import settings

logger = logging.getLogger(__name__)


class OllamaClient:
    def __init__(self, base_url: str = None, target_model: str = None, timeout: float = 180.0):
        self.base_url = base_url or settings.OLLAMA_URL
        self.chat_endpoint = f"{self.base_url}/api/chat"  # Ollama native endpoint
        self.target_model = target_model or settings.MEDICHAT_MODEL
        self.timeout = timeout
        logger.info(f"Khởi tạo OllamaClient với model: {self.target_model}, URL cơ sở: {self.base_url}")
        
    async def check_model_availability(self, model_name: str = None) -> tuple[bool, str]:
        """
        Kiểm tra xem mô hình đã được cài đặt trên Ollama chưa
        
        Args:
            model_name: Tên mô hình cần kiểm tra, sử dụng target_model nếu không được cung cấp
            
        Returns:
            (bool, str): Tuple gồm trạng thái (True nếu tồn tại) và thông báo
        """
        model_name = model_name or self.target_model
        logger.info(f"Đang kiểm tra mô hình {model_name} trong Ollama...")
        
        try:
            # Kiểm tra danh sách mô hình đã cài đặt
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    url=f"{self.base_url}/api/tags",
                    timeout=10.0
                )
                
                # Kiểm tra kết nối tới Ollama
                if response.status_code != 200:
                    error_msg = f"Không thể kết nối tới Ollama API: HTTP {response.status_code}"
                    logger.error(error_msg)
                    return False, error_msg
                
                # Phân tích danh sách mô hình
                data = response.json()
                models = [model["name"] for model in data.get("models", [])]
                
                # Kiểm tra xem mô hình có tồn tại không
                if model_name in models:
                    logger.info(f"Mô hình {model_name} đã được cài đặt trong Ollama")
                    return True, f"Mô hình {model_name} đã sẵn sàng"
                else:
                    error_msg = f"Mô hình {model_name} chưa được cài đặt trong Ollama. Sử dụng lệnh: ollama pull {model_name}"
                    logger.warning(error_msg)
                    return False, error_msg
                    
        except httpx.TimeoutException:
            error_msg = "Timeout khi kiểm tra mô hình Ollama"
            logger.error(error_msg)
            return False, error_msg
            
        except Exception as e:
            error_msg = f"Lỗi khi kiểm tra mô hình Ollama: {str(e)}"
            logger.error(error_msg)
            return False, error_msg
        
    async def generate_response_no_stream(self, messages: List[Dict[str, str]], max_tokens: int = None) -> str:
        """
        Gửi yêu cầu đến API Ollama không sử dụng streaming để debug vấn đề
        
        Args:
            messages: Danh sách tin nhắn theo định dạng [{"role": "user", "content": "..."}]
            max_tokens: Số lượng token tối đa để sinh ra
        
        Returns:
            Phản hồi đầy đủ từ API
        """
        try:
            # Sử dụng giá trị mặc định từ settings nếu không có
            if max_tokens is None:
                max_tokens = settings.LLM_MAX_TOKENS
                
            # Thêm system message nếu chưa có
            if not any(msg.get("role") == "system" for msg in messages):
                system_message = {
                    "role": "system", 
                    "content": settings.MEDICHAT_SYSTEM_PROMPT
                }
                messages = [system_message] + messages
            
            # Cắt giảm nội dung tin nhắn nếu quá dài
            truncated_messages = []
            for msg in messages:
                content = msg.get("content", "")
                # Giới hạn nội dung mỗi tin nhắn tối đa 1000 ký tự
                if len(content) > 1000:
                    truncated_content = content[:997] + "..."
                    logger.warning(f"Tin nhắn đã bị cắt từ {len(content)} xuống còn 1000 ký tự")
                    msg = {**msg, "content": truncated_content}
                truncated_messages.append(msg)
            
            logger.debug(f"Gửi yêu cầu không streaming đến Ollama với {len(truncated_messages)} tin nhắn")
            
            # Kiểm tra và đảm bảo có dữ liệu người dùng trong messages
            if len(truncated_messages) < 2:  # Phải có ít nhất system message và user message
                logger.error("Không đủ tin nhắn để tạo hội thoại")
                return "Xin lỗi, không thể xử lý yêu cầu vì thiếu thông tin. Vui lòng nhập câu hỏi của bạn."
                
            # Kiểm tra xem tin nhắn cuối cùng có nội dung không
            last_user_msg = next((msg for msg in reversed(truncated_messages) if msg.get("role") == "user"), None)
            if last_user_msg is None or not last_user_msg.get("content", "").strip():
                logger.error("Tin nhắn người dùng trống hoặc không tồn tại")
                return "Xin lỗi, vui lòng nhập nội dung câu hỏi của bạn."
            
            # Sử dụng Ollama chat API
            logger.info(f"Đang kết nối với Ollama Chat API (timeout: {self.timeout}s)...")
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url=self.chat_endpoint,
                    json={
                        "model": self.target_model,
                        "messages": truncated_messages,
                        "stream": False,
                        "options": {
                            "temperature": 0.7,
                            "num_predict": max_tokens
                        }
                    },
                    timeout=self.timeout
                )
                
                logger.info(f"Chat API status: {response.status_code}")
                
                if response.status_code != 200:
                    error_msg = f"Lỗi từ API Ollama: {response.status_code}"
                    logger.error(error_msg)
                    try:
                        error_content = response.text
                        logger.error(f"Nội dung lỗi: {error_content}")
                    except:
                        pass
                    return "Xin lỗi, tôi đang gặp khó khăn trong việc kết nối tới hệ thống AI. Vui lòng thử lại sau."
                
                # Phân tích phản hồi
                try:
                    data = response.json()
                    logger.info(f"Nhận được phản hồi thành công từ model (thời gian xử lý: {data.get('total_duration', 0)/1000000:.2f}ms)")
                    if "message" in data and "content" in data["message"]:
                        return data["message"]["content"]
                    else:
                        logger.warning("Phản hồi từ chat API không chứa nội dung message.content")
                        return "Xin lỗi, tôi không thể xử lý yêu cầu của bạn. Vui lòng thử lại."
                except Exception as e:
                    logger.error(f"Lỗi khi xử lý phản hồi JSON: {str(e)}")
                    return "Lỗi xử lý phản hồi. Vui lòng thử lại sau."
                
        except httpx.TimeoutException:
            logger.error(f"Timeout khi kết nối đến Ollama sau {self.timeout} giây")
            return "Xin lỗi, tôi cần nhiều thời gian hơn để xử lý yêu cầu của bạn. Vui lòng thử lại với câu hỏi ngắn gọn hơn hoặc liên hệ quản trị viên để tăng thời gian chờ."
        
        except Exception as e:
            logger.error(f"Lỗi kết nối đến Ollama: {str(e)}")
            return "Xin lỗi, đã xảy ra lỗi không mong muốn. Vui lòng thử lại sau."
             
    async def generate_response(self, messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
        """
        Gửi yêu cầu đến API Ollama và trả về phản hồi (vô hiệu hóa streaming để debug)
        
        Args:
            messages: Danh sách tin nhắn theo định dạng [{"role": "user", "content": "..."}]
        
        Yields:
            Từng phần nội dung phản hồi từ AI
        """
        # Tạm thời vô hiệu hóa streaming và sử dụng phiên bản không streaming
        response = await self.generate_response_no_stream(messages)
        yield response
    
    async def get_full_response(self, messages: List[Dict[str, str]], max_tokens: int = None) -> str:
        """
        Gửi yêu cầu đến API Ollama và trả về phản hồi đầy đủ
        
        Args:
            messages: Danh sách tin nhắn theo định dạng [{"role": "user", "content": "..."}]
            max_tokens: Số lượng token tối đa để sinh ra
            
        Returns:
            Phản hồi đầy đủ từ AI
        """
        # Sử dụng giá trị mặc định từ settings nếu không có
        if max_tokens is None:
            max_tokens = settings.LLM_MAX_TOKENS
            
        logger.info(f"Đang gửi yêu cầu đến model {self.target_model} với {len(messages)} tin nhắn, max_tokens={max_tokens}")
        # Sử dụng phiên bản không streaming để lấy kết quả trực tiếp
        return await self.generate_response_no_stream(messages, max_tokens=max_tokens) 