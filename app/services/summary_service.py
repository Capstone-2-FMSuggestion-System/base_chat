import httpx
import json
import logging
from typing import List, Dict, Any, Optional
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential
from app.config import settings

logger = logging.getLogger(__name__)

# Khởi tạo biến global để theo dõi trạng thái
GOOGLE_AI_AVAILABLE = False

# Thử import google.generativeai, nếu không thành công thì dùng HTTP API
try:
    import google.generativeai as genai
    GOOGLE_AI_AVAILABLE = True
    logger.info("Đã import thành công thư viện google.generativeai")
except (ImportError, Exception) as e:
    logger.warning(f"Không thể import google.generativeai: {str(e)}. Sẽ sử dụng HTTP API.")


class SummaryService:
    """
    Dịch vụ tóm tắt lịch sử trò chuyện bằng Gemini API
    """
    
    def __init__(self, api_key: Optional[str] = None, api_url: Optional[str] = None):
        """
        Khởi tạo dịch vụ tóm tắt với API key và URL của Gemini
        
        Args:
            api_key: API key của Gemini (lấy từ cấu hình nếu không cung cấp)
            api_url: URL của Gemini API (lấy từ cấu hình nếu không cung cấp)
        """
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.api_url = api_url or settings.GEMINI_API_URL
        self.model_name = "gemini-2.0-flash-lite"  # Mô hình mặc định
        
        # Sử dụng biến global GOOGLE_AI_AVAILABLE
        global GOOGLE_AI_AVAILABLE
        
        # Khởi tạo Google Generative AI client nếu có thể
        if GOOGLE_AI_AVAILABLE and self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                logger.info("Đã khởi tạo Google Generative AI client")
            except Exception as e:
                logger.error(f"Lỗi khi khởi tạo Google Generative AI client: {str(e)}")
                GOOGLE_AI_AVAILABLE = False
        
        if not self.api_key:
            logger.warning("Không có API key cho Gemini, tính năng tóm tắt sẽ không hoạt động")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def summarize_conversation(self, messages: List[Dict[str, str]]) -> str:
        """
        Tóm tắt lịch sử cuộc trò chuyện sử dụng Gemini API
        
        Args:
            messages: Danh sách tin nhắn theo định dạng [{"role": "user", "content": "..."}]
            
        Returns:
            Bản tóm tắt lịch sử trò chuyện
        """
        if not self.api_key:
            logger.error("Không thể tóm tắt: Thiếu API key của Gemini")
            return "Không thể tóm tắt lịch sử trò chuyện."
        
        if not messages:
            logger.warning("Không có tin nhắn nào để tóm tắt")
            return ""
        
        # Loại bỏ system message khỏi nội dung cần tóm tắt
        conversation_messages = [msg for msg in messages if msg["role"] != "system"]
        
        # Nếu chỉ có ít tin nhắn, không cần tóm tắt
        if len(conversation_messages) <= 3:
            return ""
        
        # Tạo nội dung prompt để gửi đến Gemini
        prompt = self._create_summary_prompt(conversation_messages)
        
        # Sử dụng biến global GOOGLE_AI_AVAILABLE
        global GOOGLE_AI_AVAILABLE
        
        # Thử sử dụng thư viện Google Generative AI nếu có sẵn
        if GOOGLE_AI_AVAILABLE:
            try:
                summary = await self._summarize_with_google_client(prompt)
                logger.info(f"[GEMINI SUMMARY] {summary}")
                return summary
            except Exception as e:
                logger.warning(f"Lỗi khi sử dụng Google Generative AI client: {str(e)}. Thử sử dụng HTTP API trực tiếp.")
                GOOGLE_AI_AVAILABLE = False
        
        # Sử dụng HTTP API trực tiếp nếu thư viện không có sẵn hoặc gặp lỗi
        summary = await self._summarize_with_http_api(prompt, conversation_messages)
        logger.info(f"[GEMINI SUMMARY] {summary}")
        return summary
    
    async def _summarize_with_http_api(self, prompt: str, conversation_messages: List[Dict[str, str]]) -> str:
        """
        Tóm tắt sử dụng HTTP API trực tiếp
        
        Args:
            prompt: Nội dung prompt để tóm tắt
            conversation_messages: Danh sách tin nhắn để log
            
        Returns:
            Nội dung tóm tắt
        """
        try:
            logger.info(f"Gửi yêu cầu tóm tắt qua HTTP API cho {len(conversation_messages)} tin nhắn")
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url=f"{self.api_url}?key={self.api_key}",
                    json={
                        "contents": [{
                            "parts": [{
                                "text": prompt
                            }]
                        }],
                        "generationConfig": {
                            "temperature": 0.2,
                            "maxOutputTokens": 500,
                            "topP": 0.95
                        }
                    },
                    timeout=30.0
                )
                
                logger.debug(f"Phản hồi từ Gemini API: {response.status_code}")
                
                if response.status_code != 200:
                    logger.error(f"Lỗi khi gọi Gemini API: {response.status_code} - {response.text[:200]}")
                    return "Không thể tóm tắt lịch sử trò chuyện do lỗi API."
                
                result = response.json()
                
                try:
                    summary_text = result["candidates"][0]["content"]["parts"][0]["text"]
                    logger.info(f"Đã tóm tắt thành công qua HTTP API: {len(summary_text)} ký tự")
                    logger.info(f"[HTTP_API_SUMMARY] Nội dung tóm tắt:\n{summary_text}")
                    return summary_text
                except (KeyError, IndexError) as e:
                    logger.error(f"Lỗi khi xử lý kết quả từ Gemini: {str(e)}")
                    return "Không thể phân tích kết quả tóm tắt."
                
        except httpx.TimeoutException:
            logger.error("Timeout khi kết nối đến Gemini API")
            return "Tóm tắt thất bại do hết thời gian chờ."
        
        except Exception as e:
            logger.error(f"Lỗi không xác định khi tóm tắt qua HTTP API: {str(e)}")
            return "Đã xảy ra lỗi khi tóm tắt lịch sử trò chuyện."
    
    async def _summarize_with_google_client(self, prompt: str) -> str:
        """
        Tóm tắt sử dụng thư viện chính thức của Google
        
        Args:
            prompt: Nội dung prompt để tóm tắt
            
        Returns:
            Nội dung tóm tắt
        """
        try:
            # Sử dụng biến global
            global GOOGLE_AI_AVAILABLE
            
            # Lấy model
            model = genai.GenerativeModel(self.model_name)
            
            # Chuyển đổi sang coroutine để chạy bất đồng bộ
            def run_generation():
                try:
                    response = model.generate_content(
                        prompt,
                        generation_config=genai.GenerationConfig(
                            temperature=0.2,
                            max_output_tokens=500,
                            top_p=0.95
                        )
                    )
                    return response.text
                except Exception as e:
                    logger.error(f"Lỗi trong run_generation: {str(e)}")
                    raise
            
            # Chạy trong ThreadPoolExecutor
            loop = asyncio.get_event_loop()
            summary = await loop.run_in_executor(None, run_generation)
            
            logger.info(f"Đã tóm tắt thành công với Google client: {len(summary)} ký tự")
            logger.info(f"[GOOGLE_CLIENT_SUMMARY] Nội dung tóm tắt:\n{summary}")
            return summary
            
        except Exception as e:
            logger.error(f"Lỗi khi sử dụng Google client: {str(e)}")
            GOOGLE_AI_AVAILABLE = False
            raise
    
    def _create_summary_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Tạo prompt để gửi đến Gemini API
        
        Args:
            messages: Danh sách tin nhắn cần tóm tắt
            
        Returns:
            Prompt để tóm tắt cuộc trò chuyện
        """
        # Chuyển đổi các tin nhắn thành định dạng văn bản
        conversation_text = "\n\n"
        
        for msg in messages:
            role = "Người dùng" if msg["role"] == "user" else "Trợ lý"
            conversation_text += f"{role}: {msg['content']}\n\n"
        
        # Tạo prompt cho Gemini
        prompt = f"""Dưới đây là cuộc trò chuyện giữa một người dùng và trợ lý y tế.
Hãy tóm tắt những thông tin quan trọng từ cuộc trò chuyện này, bảo toàn:
1. Các triệu chứng, vấn đề sức khỏe hoặc bệnh lý được đề cập
2. Các câu hỏi chính của người dùng
3. Thông tin y tế quan trọng đã được trao đổi
4. Lời khuyên chính đã được đưa ra

Chú ý:
- Tóm tắt PHẢI DƯỚI 700 KÝ TỰ
- Giữ ngắn gọn và súc tích
- Chỉ liệt kê những điểm quan trọng nhất
- Giữ nguyên ngôn ngữ ban đầu (tiếng Việt)
- Duy trì tính chính xác của thông tin y tế
- Đảm bảo tóm tắt dễ hiểu, có cấu trúc rõ ràng
- Chỉ bao gồm thông tin đã được đề cập trong đoạn hội thoại

Đoạn hội thoại:
{conversation_text}

TÓM TẮT (DƯỚI 700 KÝ TỰ):
"""
        
        return prompt 