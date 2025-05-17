import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import json
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
import asyncio

from app.config import settings

logger = logging.getLogger(__name__)

# Khởi tạo biến global để theo dõi trạng thái
GOOGLE_AI_AVAILABLE = False

# Thử import google.generativeai, nếu không thành công thì dùng HTTP API
try:
    import google.generativeai as genai
    GOOGLE_AI_AVAILABLE = True
    logger.info("Đã import thành công thư viện google.generativeai cho GeminiPromptService")
except (ImportError, Exception) as e:
    logger.warning(f"Không thể import google.generativeai: {str(e)}. Sẽ sử dụng HTTP API.")


class GeminiPromptService:
    """
    Dịch vụ xử lý điều phối, phân tích nội dung chat và tạo prompt cho Medichat LLaMA3 
    sử dụng Gemini API
    """
    
    def __init__(self, api_key: Optional[str] = None, api_url: Optional[str] = None):
        """
        Khởi tạo dịch vụ với API key và URL của Gemini
        
        Args:
            api_key: API key của Gemini (lấy từ cấu hình nếu không cung cấp)
            api_url: URL của Gemini API (lấy từ cấu hình nếu không cung cấp)
        """
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.api_url = api_url or settings.GEMINI_API_URL
        self.model_name = "gemini-2.0-flash-lite"  # Mô hình mặc định
        self.max_prompt_length = 900  # Giới hạn độ dài prompt
        
        # Sử dụng biến global GOOGLE_AI_AVAILABLE
        global GOOGLE_AI_AVAILABLE
        
        # Khởi tạo Google Generative AI client nếu có thể
        if GOOGLE_AI_AVAILABLE and self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                logger.info("Đã khởi tạo Google Generative AI client cho GeminiPromptService")
            except Exception as e:
                logger.error(f"Lỗi khi khởi tạo Google Generative AI client: {str(e)}")
                GOOGLE_AI_AVAILABLE = False
        
        if not self.api_key:
            logger.warning("Không có API key cho Gemini, các tính năng phân tích và điều phối sẽ bị hạn chế")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def analyze_query(self, user_message: str, chat_history: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Phân tích nội dung chat của người dùng để xác định phạm vi hợp lệ và câu hỏi cần thiết
        
        Args:
            user_message: Nội dung tin nhắn của người dùng
            chat_history: Lịch sử chat trước đó
            
        Returns:
            Kết quả phân tích chứa thông tin về query, phạm vi, câu hỏi bổ sung
        """
        if not self.api_key:
            logger.error("Không thể phân tích: Thiếu API key của Gemini")
            return {
                "is_valid_scope": True,  # Fallback: mặc định là hợp lệ
                "need_more_info": False,
                "follow_up_question": None,
                "collected_info": {}
            }
        
        # Tạo prompt cho việc phân tích
        prompt = self._create_analysis_prompt(user_message, chat_history)
        
        try:
            # Sử dụng thư viện Google hoặc HTTP API
            if GOOGLE_AI_AVAILABLE:
                try:
                    analysis_result = await self._query_gemini_with_client(prompt)
                except Exception as e:
                    logger.warning(f"Lỗi khi sử dụng Google client: {str(e)}. Chuyển sang HTTP API.")
                    analysis_result = await self._query_gemini_with_http(prompt)
            else:
                analysis_result = await self._query_gemini_with_http(prompt)
            
            # Parse kết quả JSON
            try:
                # Xóa markdown code block nếu có
                clean_result = analysis_result
                if "```json" in analysis_result:
                    clean_result = analysis_result.split("```json")[1].split("```")[0].strip()
                elif "```" in analysis_result:
                    clean_result = analysis_result.split("```")[1].split("```")[0].strip()
                
                # Phân tích JSON
                result = json.loads(clean_result)
                logger.info(f"Phân tích thành công: {result}")
                return result
            except json.JSONDecodeError as json_err:
                logger.error(f"Không thể parse kết quả JSON: {analysis_result}")
                # Thử extract định dạng JSON từ phản hồi
                try:
                    import re
                    json_pattern = r'(\{.*\})'
                    match = re.search(json_pattern, analysis_result, re.DOTALL)
                    if match:
                        potential_json = match.group(1)
                        result = json.loads(potential_json)
                        logger.info(f"Đã trích xuất JSON thành công từ phản hồi: {result}")
                        return result
                except Exception as extract_err:
                    logger.error(f"Không thể trích xuất JSON từ phản hồi: {str(extract_err)}")
                
                # Fallback nếu không thể phân tích JSON
                if "chào" in analysis_result.lower() or "hello" in analysis_result.lower():
                    # Chào hỏi, cần phục vụ luôn
                    return {
                        "is_valid_scope": True,
                        "need_more_info": False,
                        "follow_up_question": analysis_result,
                        "collected_info": {}
                    }
                else:
                    return {
                        "is_valid_scope": True,
                        "need_more_info": False,
                        "follow_up_question": None,
                        "collected_info": {}
                    }
                
        except Exception as e:
            logger.error(f"Lỗi khi phân tích nội dung: {str(e)}")
            # Trả về kết quả mặc định nếu có lỗi
            return {
                "is_valid_scope": True,  # Fallback: mặc định là hợp lệ
                "need_more_info": False,
                "follow_up_question": None,
                "collected_info": {}
            }
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def create_medichat_prompt(self, messages: List[Dict[str, str]]) -> str:
        """
        Tạo prompt dưới 900 ký tự để gửi đến Medichat-LLaMA3-8B
        
        Args:
            messages: Danh sách tin nhắn theo định dạng [{"role": "user", "content": "..."}]
            
        Returns:
            Prompt được tối ưu hóa cho Medichat
        """
        if not self.api_key or not messages:
            logger.error("Không thể tạo prompt: Thiếu API key hoặc không có tin nhắn")
            return ""
        
        # Tạo prompt để gửi đến Gemini
        prompt = self._create_medichat_prompt_template(messages)
        
        try:
            # Sử dụng thư viện Google hoặc HTTP API
            if GOOGLE_AI_AVAILABLE:
                try:
                    result_prompt = await self._query_gemini_with_client(prompt)
                except Exception as e:
                    logger.warning(f"Lỗi khi sử dụng Google client: {str(e)}. Chuyển sang HTTP API.")
                    result_prompt = await self._query_gemini_with_http(prompt)
            else:
                result_prompt = await self._query_gemini_with_http(prompt)
            
            # Đảm bảo kết quả không vượt quá giới hạn
            if len(result_prompt) > self.max_prompt_length:
                result_prompt = result_prompt[:self.max_prompt_length]
                
            logger.info(f"Đã tạo prompt ({len(result_prompt)} ký tự): {result_prompt}")
            return result_prompt
                
        except Exception as e:
            logger.error(f"Lỗi khi tạo prompt: {str(e)}")
            # Trả về một prompt đơn giản trong trường hợp lỗi
            return "Cần tư vấn dinh dưỡng và món ăn phù hợp."
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def polish_response(self, medichat_response: str, original_prompt: str) -> str:
        """
        Kiểm tra và điều chỉnh phản hồi từ Medichat
        
        Args:
            medichat_response: Phản hồi từ Medichat
            original_prompt: Prompt ban đầu đã gửi đến Medichat
            
        Returns:
            Phản hồi đã được điều chỉnh
        """
        if not self.api_key:
            logger.error("Không thể điều chỉnh phản hồi: Thiếu API key của Gemini")
            return medichat_response
        
        # Tạo prompt để kiểm tra và điều chỉnh
        prompt = self._create_response_polish_prompt(medichat_response, original_prompt)
        
        try:
            # Sử dụng thư viện Google hoặc HTTP API
            if GOOGLE_AI_AVAILABLE:
                try:
                    polished_response = await self._query_gemini_with_client(prompt)
                except Exception as e:
                    logger.warning(f"Lỗi khi sử dụng Google client: {str(e)}. Chuyển sang HTTP API.")
                    polished_response = await self._query_gemini_with_http(prompt)
            else:
                polished_response = await self._query_gemini_with_http(prompt)
                
            # Xử lý để loại bỏ các metadata không cần thiết
            debug_patterns = [
                "**Đánh giá và Điều chỉnh Phản hồi:**",
                "**Đánh giá:**",
                "**Kiểm tra:**", 
                "**Điều chỉnh:**",
                "**Phản hồi đã được điều chỉnh:**",
                "**Phân tích phản hồi:**", 
                "**HỢP LỆ**", 
                "**KHÔNG HỢP LỆ**",
                "Dưới đây là phản hồi đã được điều chỉnh:"
            ]
            
            # Trường hợp phản hồi có cấu trúc điển hình với đánh giá ở đầu và phản hồi thực sự ở sau
            for pattern in debug_patterns:
                if pattern in polished_response:
                    parts = polished_response.split(pattern)
                    if len(parts) >= 2:
                        # Giữ lại phần sau pattern cuối cùng
                        polished_response = parts[-1].strip()
            
            # Xử lý trường hợp có định dạng số thứ tự và đánh dấu
            if polished_response.strip().startswith("1.") or polished_response.strip().startswith("*"):
                lines = polished_response.split("\n")
                filtered_lines = []
                in_debug_section = False
                
                for line in lines:
                    line_lower = line.lower().strip()
                    # Xác định dòng bắt đầu phần debug
                    if any(pattern.lower() in line_lower for pattern in debug_patterns):
                        in_debug_section = True
                        continue
                        
                    # Xác định kết thúc phần debug và bắt đầu nội dung thực
                    if in_debug_section and (
                        "phản hồi đã được điều chỉnh" in line_lower or 
                        "chào bạn" in line_lower or
                        line.strip() == ""
                    ):
                        in_debug_section = False
                    
                    # Chỉ thêm dòng nếu không nằm trong phần debug
                    if not in_debug_section:
                        filtered_lines.append(line)
                
                # Kết hợp các dòng đã lọc
                polished_response = "\n".join(filtered_lines).strip()
            
            # Loại bỏ phần đánh dấu còn sót
            polished_response = polished_response.replace("**Phản hồi:**", "").strip()
            
            # Xử lý trường hợp còn sót các phần cụ thể
            if "đã được điều chỉnh" in polished_response:
                parts = polished_response.split(":")
                if len(parts) > 1:  # Có dấu ":" trong phản hồi
                    polished_response = ":".join(parts[1:]).strip()
            
            # Loại bỏ các dấu xuống dòng thừa ở đầu
            while polished_response.startswith("\n"):
                polished_response = polished_response[1:]
            
            # Loại bỏ các dấu xuống dòng thừa ở cuối
            while polished_response.endswith("\n\n"):
                polished_response = polished_response[:-1]
            
            logger.info(f"Đã điều chỉnh phản hồi: {len(polished_response)} ký tự")
            return polished_response.strip()
                
        except Exception as e:
            logger.error(f"Lỗi khi điều chỉnh phản hồi: {str(e)}")
            # Trả về phản hồi gốc nếu có lỗi
            return medichat_response
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def generate_welcome_message(self) -> str:
        """
        Tạo lời chào và giới thiệu cho phiên trò chuyện mới
        
        Returns:
            Lời chào và giới thiệu
        """
        if not self.api_key:
            logger.error("Không thể tạo lời chào: Thiếu API key của Gemini")
            return "Xin chào! Tôi là trợ lý tư vấn dinh dưỡng và sức khỏe. Tôi có thể giúp gì cho bạn hôm nay?"
        
        prompt = """Viết một lời chào ngắn gọn, thân thiện nhưng chuyên nghiệp cho một chatbot tư vấn y tế, dinh dưỡng và món ăn phù hợp.

Lời chào cần:
1. Giới thiệu tên và chức năng (tư vấn dinh dưỡng, món ăn phù hợp với tình trạng sức khỏe)
2. Nhấn mạnh khả năng tư vấn các món ăn phù hợp với người có vấn đề sức khỏe hoặc chế độ dinh dưỡng đặc biệt
3. Mời người dùng chia sẻ về tình trạng sức khỏe, mục tiêu dinh dưỡng, hoặc món ăn họ quan tâm
4. Viết bằng tiếng Việt, tối đa 4 câu

Lưu ý: Ngắn gọn, mạch lạc, và thân thiện."""
        
        try:
            # Sử dụng thư viện Google hoặc HTTP API
            if GOOGLE_AI_AVAILABLE:
                try:
                    welcome_message = await self._query_gemini_with_client(prompt)
                except Exception as e:
                    logger.warning(f"Lỗi khi sử dụng Google client: {str(e)}. Chuyển sang HTTP API.")
                    welcome_message = await self._query_gemini_with_http(prompt)
            else:
                welcome_message = await self._query_gemini_with_http(prompt)
                
            logger.info(f"Đã tạo lời chào: {welcome_message}")
            return welcome_message
                
        except Exception as e:
            logger.error(f"Lỗi khi tạo lời chào: {str(e)}")
            # Trả về lời chào mặc định nếu có lỗi
            return "Xin chào! Tôi là trợ lý tư vấn dinh dưỡng và sức khỏe. Tôi có thể giúp gì cho bạn hôm nay?"
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def get_greeting_response(self, user_message: str) -> str:
        """
        Tạo phản hồi cho tin nhắn chào hỏi của người dùng
        
        Args:
            user_message: Nội dung tin nhắn chào hỏi của người dùng
            
        Returns:
            Phản hồi chào hỏi và giới thiệu
        """
        if not self.api_key:
            logger.error("Không thể tạo phản hồi chào hỏi: Thiếu API key của Gemini")
            return "Xin chào! Tôi là trợ lý tư vấn dinh dưỡng và sức khỏe. Tôi có thể giúp bạn tìm hiểu về các món ăn phù hợp với tình trạng sức khỏe, chế độ dinh dưỡng cân đối, hoặc tư vấn về thói quen ăn uống lành mạnh. Bạn cần hỗ trợ gì hôm nay?"
        
        prompt = f"""Người dùng gửi tin nhắn chào hỏi: "{user_message}"

Hãy viết một lời chào thân thiện và giới thiệu ngắn gọn về chức năng của trợ lý tư vấn sức khỏe và dinh dưỡng. 
Phản hồi cần:
1. Chào hỏi tương ứng với lời chào của người dùng
2. Giới thiệu khả năng tư vấn về món ăn phù hợp với tình trạng sức khỏe, dinh dưỡng và thói quen ăn uống
3. Khuyến khích người dùng chia sẻ về tình trạng sức khỏe hoặc mục tiêu dinh dưỡng
4. Ngắn gọn, tối đa 3-4 câu
5. Thân thiện nhưng chuyên nghiệp

Viết bằng tiếng Việt, trực tiếp phản hồi không có giải thích thêm."""
        
        try:
            # Sử dụng thư viện Google hoặc HTTP API
            if GOOGLE_AI_AVAILABLE:
                try:
                    greeting_response = await self._query_gemini_with_client(prompt)
                except Exception as e:
                    logger.warning(f"Lỗi khi sử dụng Google client: {str(e)}. Chuyển sang HTTP API.")
                    greeting_response = await self._query_gemini_with_http(prompt)
            else:
                greeting_response = await self._query_gemini_with_http(prompt)
                
            logger.info(f"Đã tạo phản hồi chào hỏi: {greeting_response[:50]}...")
            return greeting_response
                
        except Exception as e:
            logger.error(f"Lỗi khi tạo phản hồi chào hỏi: {str(e)}")
            # Trả về lời chào mặc định nếu có lỗi
            return "Xin chào! Tôi là trợ lý tư vấn dinh dưỡng và sức khỏe. Tôi có thể giúp bạn tìm hiểu về các món ăn phù hợp với tình trạng sức khỏe, chế độ dinh dưỡng cân đối, hoặc tư vấn về thói quen ăn uống lành mạnh. Bạn cần hỗ trợ gì hôm nay?"
    
    async def _query_gemini_with_http(self, prompt: str) -> str:
        """
        Gửi prompt đến Gemini API thông qua HTTP API
        
        Args:
            prompt: Nội dung prompt
            
        Returns:
            Phản hồi từ Gemini
        """
        try:
            logger.info(f"Gửi yêu cầu đến Gemini API HTTP, độ dài prompt: {len(prompt)}")
            
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
                            "maxOutputTokens": 1000,
                            "topP": 0.95
                        }
                    },
                    timeout=30.0
                )
                
                logger.debug(f"Phản hồi từ Gemini API: {response.status_code}")
                
                if response.status_code != 200:
                    logger.error(f"Lỗi khi gọi Gemini API: {response.status_code} - {response.text[:200]}")
                    raise Exception(f"API error: {response.status_code}")
                
                result = response.json()
                
                try:
                    response_text = result["candidates"][0]["content"]["parts"][0]["text"]
                    return response_text
                except (KeyError, IndexError) as e:
                    logger.error(f"Lỗi khi xử lý kết quả từ Gemini: {str(e)}")
                    raise Exception("Invalid response format")
                
        except httpx.TimeoutException:
            logger.error("Timeout khi kết nối đến Gemini API")
            raise Exception("API timeout")
        
        except Exception as e:
            logger.error(f"Lỗi không xác định khi gọi Gemini API: {str(e)}")
            raise
    
    async def _query_gemini_with_client(self, prompt: str) -> str:
        """
        Gửi prompt đến Gemini API sử dụng thư viện chính thức
        
        Args:
            prompt: Nội dung prompt
            
        Returns:
            Phản hồi từ Gemini
        """
        # Sử dụng biến global
        global GOOGLE_AI_AVAILABLE
        
        if not GOOGLE_AI_AVAILABLE:
            raise Exception("Google Generative AI client không khả dụng")
        
        try:
            # Lấy model
            model = genai.GenerativeModel(self.model_name)
            
            # Chuyển đổi sang coroutine để chạy bất đồng bộ
            def run_generation():
                try:
                    response = model.generate_content(
                        prompt,
                        generation_config=genai.GenerationConfig(
                            temperature=0.2,
                            max_output_tokens=1000,
                            top_p=0.95
                        )
                    )
                    return response.text
                except Exception as e:
                    logger.error(f"Lỗi trong run_generation: {str(e)}")
                    raise
            
            # Chạy trong ThreadPoolExecutor
            loop = asyncio.get_event_loop()
            response_text = await loop.run_in_executor(None, run_generation)
            
            logger.info(f"Nhận được phản hồi từ Gemini client: {len(response_text)} ký tự")
            return response_text
            
        except Exception as e:
            logger.error(f"Lỗi khi sử dụng Google client: {str(e)}")
            GOOGLE_AI_AVAILABLE = False
            raise
    
    def _create_analysis_prompt(self, user_message: str, chat_history: List[Dict[str, str]]) -> str:
        """
        Tạo prompt để phân tích nội dung chat của người dùng
        
        Args:
            user_message: Nội dung tin nhắn của người dùng
            chat_history: Lịch sử chat
            
        Returns:
            Prompt cho Gemini để phân tích
        """
        # Chuyển đổi lịch sử chat thành văn bản - sử dụng toàn bộ lịch sử
        # Nhưng tối ưu cho token - giảm độ dài nội dung nếu quá dài
        history_text = ""
        total_chars = 0
        max_history_chars = 14000  # Giới hạn ký tự cho lịch sử chat để tránh vượt quá giới hạn token
        
        if chat_history:
            # Sử dụng toàn bộ lịch sử chat nhưng có kiểm soát độ dài
            for msg in chat_history:
                role = "Người dùng" if msg["role"] == "user" else "Trợ lý"
                content = msg['content']
                
                # Cắt bớt nội dung nếu quá dài
                if len(content) > 500:
                    content = content[:500] + "... [nội dung đã cắt ngắn]"
                
                msg_text = f"{role}: {content}\n"
                
                # Kiểm tra xem có vượt quá giới hạn không
                if total_chars + len(msg_text) > max_history_chars:
                    history_text += "[...nhiều tin nhắn trước đó đã được bỏ qua...]\n"
                    break
                
                history_text += msg_text
                total_chars += len(msg_text)
        
        # Tạo prompt
        prompt = f"""Phân tích TOÀN BỘ cuộc trò chuyện dưới đây để xác định thông tin sức khỏe và nhu cầu dinh dưỡng của người dùng:

LỊCH SỬ CHAT ĐẦY ĐỦ:
{history_text}

TIN NHẮN NGƯỜI DÙNG MỚI NHẤT:
{user_message}

NHIỆM VỤ CỤ THỂ:

1. Phân tích kỹ lưỡng toàn bộ cuộc trò chuyện để xác định mọi thông tin liên quan đến:
   - Tình trạng sức khỏe hiện tại (tiểu đường, cao huyết áp, dị ứng, v.v.)
   - Bệnh lý đã biết (đặc biệt chú ý phát hiện các bệnh mãn tính)
   - Dị ứng thực phẩm (nêu cụ thể loại thực phẩm gây dị ứng)
   - Thói quen ăn uống (kiểu ăn, thời gian, sở thích)
   - Mục tiêu sức khỏe (giảm cân, kiểm soát đường huyết, v.v.)

2. Ghi lại chi tiết các yếu tố từ TIN NHẮN MỚI NHẤT và TOÀN BỘ LỊCH SỬ trước đó:
   - Chú ý đặc biệt đến các lần nhắc đến "tiểu đường", "dị ứng", "bệnh", "không dùng được", "không ăn được"
   - Liên kết các yêu cầu mới với thông tin sức khỏe đã chia sẻ trước đó
   - Xác định mong muốn về món ăn hoặc chế độ dinh dưỡng

3. Xác định phạm vi hỗ trợ và nhu cầu thông tin:
   - Nội dung có liên quan đến tư vấn dinh dưỡng/thực phẩm không?
   - Cần thu thập thêm thông tin gì để đưa ra gợi ý phù hợp?
   - Người dùng có từ chối cung cấp thông tin cụ thể nào không?

Trả về kết quả dưới dạng JSON với cấu trúc chính xác sau:
{{
  "is_valid_scope": true/false,
  "need_more_info": true/false,
  "follow_up_question": "Câu hỏi đơn giản, ngắn gọn nếu cần thêm thông tin",
  "collected_info": {{
    "health_condition": "Tình trạng sức khỏe hiện tại đã phát hiện (tiểu đường/cao huyết áp/v.v.)",
    "medical_history": "Bệnh lý đã biết (chi tiết từ lịch sử)",
    "allergies": "Dị ứng đã phát hiện (từ TOÀN BỘ cuộc trò chuyện)",
    "dietary_habits": "Thói quen ăn uống đã đề cập",
    "health_goals": "Mục tiêu sức khỏe đã đề cập (giảm cân/kiểm soát đường huyết/v.v.)"
  }}
}}

QUAN TRỌNG: Thu thập thông tin từ TOÀN BỘ cuộc trò chuyện, không chỉ tin nhắn mới nhất. Trích xuất mọi chi tiết về sức khỏe đã được đề cập."""
        
        return prompt
    
    def _create_medichat_prompt_template(self, messages: List[Dict[str, str]]) -> str:
        """
        Tạo template prompt để tóm tắt thông tin cho Medichat
        
        Args:
            messages: Danh sách tin nhắn
            
        Returns:
            Prompt cho Gemini để tạo prompt Medichat
        """
        # Chuyển đổi các tin nhắn thành văn bản - sử dụng toàn bộ lịch sử
        # Nhưng tối ưu cho token - giảm độ dài nội dung nếu quá dài
        conversation_text = "\n\n"
        total_chars = 0
        max_conversation_chars = 14000  # Giới hạn ký tự cho lịch sử chat
        
        # Sử dụng toàn bộ lịch sử chat
        for msg in messages:
            if msg["role"] != "system":  # Bỏ qua system message
                role = "Người dùng" if msg["role"] == "user" else "Trợ lý"
                content = msg['content']
                
                # Cắt bớt nội dung nếu quá dài
                if len(content) > 500:
                    content = content[:500] + "... [nội dung đã cắt ngắn]"
                
                msg_text = f"{role}: {content}\n\n"
                
                # Kiểm tra xem có vượt quá giới hạn không
                if total_chars + len(msg_text) > max_conversation_chars:
                    conversation_text += "[...nhiều tin nhắn trước đó đã được bỏ qua để đảm bảo không vượt quá giới hạn token...]\n\n"
                    break
                
                conversation_text += msg_text
                total_chars += len(msg_text)
        
        # Tạo prompt cho Gemini
        prompt = f"""Tóm tắt thông tin từ TOÀN BỘ cuộc trò chuyện sau để tạo prompt cho mô hình y tế Medichat-LLaMA3-8B:

{conversation_text}

Yêu cầu:
1. Tạo prompt ngắn gọn DƯỚI 900 KÝ TỰ tổng hợp các thông tin quan trọng về:
   - Yêu cầu chính/vấn đề mà người dùng đang hỏi (ƯU TIÊN giải quyết cái này)
   - Triệu chứng/tình trạng sức khỏe hiện tại
   - Bệnh lý nền/dị ứng đã biết
   - Thông tin về món ăn hoặc chế độ dinh dưỡng mà người dùng đang quan tâm
   - Mục tiêu dinh dưỡng/sức khỏe của người dùng
   - Thói quen ăn uống đã đề cập

2. Cấu trúc prompt theo dạng một yêu cầu rõ ràng: "Tôi cần gợi ý món ăn phù hợp cho [tình trạng sức khỏe] với các đặc điểm [liệt kê]. Tôi muốn món ăn [đặc điểm mong muốn]."

3. Cấu trúc prompt theo dạng một yêu cầu rõ ràng: "Tôi cần gợi ý món ăn phù hợp cho [tình trạng sức khỏe] với các đặc điểm [liệt kê]. Tôi muốn món ăn [đặc điểm mong muốn]."

4. Chỉ bao gồm thông tin đã được đề cập trong cuộc trò chuyện, không thêm thông tin không có thật.

5. Viết bằng ngôi thứ nhất, như thể người dùng đang trực tiếp hỏi Medichat.

PROMPT KẾT QUẢ (DƯỚI 900 KÝ TỰ):"""
        
        return prompt
    
    def _create_response_polish_prompt(self, medichat_response: str, original_prompt: str) -> str:
        """
        Tạo prompt để kiểm tra và điều chỉnh phản hồi từ Medichat
        
        Args:
            medichat_response: Phản hồi từ Medichat
            original_prompt: Prompt ban đầu đã gửi đến Medichat
            
        Returns:
            Prompt cho Gemini để kiểm tra và điều chỉnh
        """
        prompt = f"""Nhiệm vụ của bạn là đánh giá và điều chỉnh phản hồi từ mô hình y tế để đưa ra phản hồi CUỐI CÙNG hoàn chỉnh, sạch, không chứa metadata.

PROMPT BAN ĐẦU:
{original_prompt}

PHẢN HỒI TỪ MEDICHAT:
{medichat_response}

NHIỆM VỤ CỤ THỂ:
1. Đánh giá xem phản hồi hiện tại có cung cấp hướng dẫn dinh dưỡng hữu ích, phù hợp không
2. Nếu phản hồi tốt, chỉ cần làm sạch định dạng, loại bỏ mọi metadata
3. Nếu phản hồi chưa đầy đủ hoặc không phù hợp, viết phản hồi mới với nội dung hướng dẫn dinh dưỡng phù hợp
4. TẤT CẢ phản hồi của bạn sẽ được trả trực tiếp cho người dùng mà không qua bất kỳ xử lý nào nữa

HƯỚNG DẪN QUAN TRỌNG:
- KHÔNG BAO GIỜ bao gồm các từ như "Đánh giá", "Kiểm tra", "Điều chỉnh phản hồi" trong đầu ra
- KHÔNG BAO GIỜ chia phản hồi thành các phần có tiêu đề hoặc đánh số bước
- KHÔNG BAO GIỜ nhắc đến quá trình đánh giá hoặc sửa đổi
- LUÔN viết như thể bạn đang trực tiếp trả lời người dùng
- LUÔN sử dụng tiếng Việt thân thiện, mạch lạc
- LUÔN đảm bảo phản hồi ngắn gọn, súc tích và dễ hiểu
- LUÔN duy trì thông tin y tế chính xác và hữu ích

TRẢ VỀ NGAY LẬP TỨC CHỈ PHẦN NỘI DUNG PHẢN HỒI CUỐI CÙNG DÀNH CHO NGƯỜI DÙNG, KHÔNG CÓ BẤT KỲ METADATA HAY GIẢI THÍCH NÀO:"""
        
        return prompt 