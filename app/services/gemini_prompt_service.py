import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import json
import httpx
import re
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
        self.max_prompt_length = settings.GEMINI_MAX_PROMPT_LENGTH  # Từ settings
        self.max_prompt_length_with_recipes = 400  # Giới hạn cho prompt có recipes (từ)
        self.max_medichat_prompt_words_with_context = settings.GEMINI_MAX_PROMPT_WORDS_WITH_CONTEXT  # Từ settings
        
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
                "is_food_related": False,
                "requests_food": False,
                "requests_beverage": False,
                "need_more_info": False,
                "follow_up_question": None,
                "user_rejected_info": False,
                "suggest_general_options": False,
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
                
                # Đảm bảo các trường mới tồn tại với giá trị mặc định
                result.setdefault("requests_food", False)
                result.setdefault("requests_beverage", False)
                
                # Validation logic: Đảm bảo logic tính is_food_related
                if result.get("requests_food", False) or result.get("requests_beverage", False):
                    result["is_food_related"] = True
                
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
                        
                        # Đảm bảo các trường mới tồn tại với giá trị mặc định
                        result.setdefault("requests_food", False)
                        result.setdefault("requests_beverage", False)
                        
                        # Validation logic: Đảm bảo logic tính is_food_related
                        if result.get("requests_food", False) or result.get("requests_beverage", False):
                            result["is_food_related"] = True
                        
                        logger.info(f"Đã trích xuất JSON thành công từ phản hồi: {result}")
                        return result
                except Exception as extract_err:
                    logger.error(f"Không thể trích xuất JSON từ phản hồi: {str(extract_err)}")
                
                # Fallback nếu không thể phân tích JSON
                if "chào" in analysis_result.lower() or "hello" in analysis_result.lower():
                    # Chào hỏi, cần phục vụ luôn
                    return {
                        "is_valid_scope": True,
                        "is_food_related": False,
                        "requests_food": False,
                        "requests_beverage": False,
                        "need_more_info": False,
                        "follow_up_question": analysis_result,
                        "user_rejected_info": False,
                        "suggest_general_options": False,
                        "collected_info": {}
                    }
                else:
                    return {
                        "is_valid_scope": True,
                        "is_food_related": False,
                        "requests_food": False,
                        "requests_beverage": False,
                        "need_more_info": False,
                        "follow_up_question": None,
                        "user_rejected_info": False,
                        "suggest_general_options": False,
                        "collected_info": {}
                    }
                
        except Exception as e:
            logger.error(f"Lỗi khi phân tích nội dung: {str(e)}")
            # Trả về kết quả mặc định nếu có lỗi
            return {
                "is_valid_scope": True,  # Fallback: mặc định là hợp lệ
                "is_food_related": False,
                "requests_food": False,
                "requests_beverage": False,
                "need_more_info": False,
                "follow_up_question": None,
                "user_rejected_info": False,
                "suggest_general_options": False,
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
            
            # Xử lý trường hợp có định dạng số thứ tự và đánh dấu - CHỈ XỬ LÝ DEBUG, KHÔNG STRIP MARKDOWN
            if any(pattern.lower() in polished_response.lower() for pattern in debug_patterns):
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
                
                # Kết hợp các dòng đã lọc - GIỮ NGUYÊN MARKDOWN FORMATTING
                polished_response = "\n".join(filtered_lines).strip()
            
            # Loại bỏ phần đánh dấu còn sót
            polished_response = polished_response.replace("**Phản hồi:**", "").strip()
            
            # Xử lý trường hợp còn sót các phần cụ thể
            if "đã được điều chỉnh" in polished_response:
                parts = polished_response.split(":")
                if len(parts) > 1:  # Có dấu ":" trong phản hồi
                    polished_response = ":".join(parts[1:]).strip()
            
            # Bước 1: Chuẩn hóa tất cả các kiểu xuống dòng thành \n
            polished_response = polished_response.replace('\r\n', '\n').replace('\r', '\n')

            # Bước 2: Loại bỏ các khoảng trắng thừa ở đầu và cuối mỗi dòng
            lines = polished_response.split('\n')
            stripped_lines = [line.strip() for line in lines]
            polished_response = '\n'.join(stripped_lines)

            # Bước 3: Loại bỏ multiple line breaks - chỉ giữ single line break
            # (thay thế 2 hoặc nhiều \n liên tiếp bằng \n)
            polished_response = re.sub(r'\n{2,}', '\n', polished_response)

            # Loại bỏ các dấu xuống dòng thừa ở đầu
            while polished_response.startswith("\n"):
                polished_response = polished_response[1:]
            
            # Loại bỏ các dấu xuống dòng thừa ở cuối
            while polished_response.endswith("\n"):
                polished_response = polished_response[:-1]
            
            logger.info(f"Đã điều chỉnh và chuẩn hóa định dạng xuống dòng cho phản hồi: {len(polished_response)} ký tự")
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
        recent_history = chat_history[-10:] # Giữ nguyên giới hạn 10 tin nhắn gần nhất

        for msg in recent_history:
            role = "Người dùng" if msg["role"] == "user" else "Trợ lý"
            content = msg['content']
            if len(content) > 300:
                content = content[:300] + "... [nội dung đã cắt ngắn]"
            msg_text = f"{role}: {content}\n"
            if total_chars + len(msg_text) > 3000: #có thể thay 3000 bằng 14000
                history_text = "[...một số tin nhắn trước đó đã được bỏ qua...]\n" + history_text
                break
            history_text += msg_text
            total_chars += len(msg_text)
        
        # Tạo prompt
        prompt = f"""**NHIỆM VỤ CỦA BẠN:**
Bạn là một CHUYÊN VIÊN PHÂN TÍCH YÊU CẦU NGƯỜI DÙNG SIÊU CẤP, cực kỳ thông minh, tinh tế và có khả năng suy luận logic mạnh mẽ cho một chatbot tư vấn y tế, dinh dưỡng và ẩm thực. Nhiệm vụ của bạn là PHÂN TÍCH KỸ LƯỠNG TOÀN BỘ CUỘC TRÒ CHUYỆN (LỊCH SỬ và TIN NHẮN MỚI NHẤT), sau đó trả về một đối tượng JSON DUY NHẤT với cấu trúc được định nghĩa nghiêm ngặt ở cuối.

**PHẠM VI TƯ VẤN CỦA CHATBOT:**
(Giữ nguyên phần này)
- Tư vấn dinh dưỡng, sức khỏe tổng quát.
- Gợi ý món ăn và đồ uống phù hợp với tình trạng sức khỏe người dùng (nếu được cung cấp).
- Công thức nấu ăn/pha chế phù hợp bệnh lý.
- Dinh dưỡng cho các đối tượng đặc biệt.
- Thực phẩm nên dùng/tránh với các bệnh.
- Tư vấn chế độ ăn uống khoa học.

**THÔNG TIN ĐẦU VÀO ĐỂ PHÂN TÍCH:**

LỊCH SỬ CHAT GẦN ĐÂY (nếu có):
{history_text}

TIN NHẮN NGƯỜI DÙNG MỚI NHẤT CẦN PHÂN TÍCH:
Người dùng: {user_message}

**YÊU CẦU PHÂN TÍCH CHI TIẾT (TUÂN THỦ TUYỆT ĐỐI CÁC QUY TẮC LOGIC SAU):**

**1. PHÂN TÍCH PHẠM VI VÀ LOẠI YÊU CẦU (TỪ TIN NHẮN MỚI NHẤT):**
   - `is_valid_scope` (boolean): Yêu cầu có nằm trong PHẠM VI TƯ VẤN không? (Mặc định `true` trừ khi hoàn toàn không liên quan).
   - `is_food_related` (boolean): Yêu cầu có liên quan đến ẩm thực (món ăn, đồ uống, công thức, dinh dưỡng cụ thể) không?
   - `requests_food` (boolean): Người dùng có cụ thể hỏi MÓN ĂN không? (Ví dụ: "món ăn", "thực đơn", "công thức nấu ăn").
   - `requests_beverage` (boolean): Người dùng có cụ thể hỏi ĐỒ UỐNG không? (Ví dụ: "nước uống", "thức uống", "nước ép", "sinh tố").
   *Logic cho cờ `request_...`*:
     - Nếu người dùng nói "món ăn và đồ uống", đặt cả hai là `true`.
     - Nếu người dùng nói "ăn uống gì đó", và ngữ cảnh trước đó có nhắc đến loại cụ thể, hãy theo ngữ cảnh đó. Nếu không, có thể đặt `is_food_related=true` và cả `requests_food`, `requests_beverage` là `false` (để chatbot hỏi làm rõ).
     - `is_food_related` là `true` nếu `requests_food` hoặc `requests_beverage` là `true`.

**2. TRÍCH XUẤT THÔNG TIN NGƯỜI DÙNG (Từ TOÀN BỘ LỊCH SỬ và TIN NHẮN MỚI NHẤT - TÍCH LŨY THÔNG TIN):**
   - `collected_info` (object):
     - `health_condition` (string): Liệt kê TẤT CẢ tình trạng sức khỏe, bệnh lý (ví dụ: "tiểu đường, béo phì"). GHI NHẬN VÀ GIỮ LẠI thông tin này qua các lượt.
     - `medical_history` (string): (Tương tự `health_condition`)
     - `allergies` (string): (Tương tự `health_condition`)
     - `dietary_habits` (string): Các thói quen đặc biệt ("ăn chay", "ăn kiêng low-carb"). Nếu người dùng nói "bình thường", "không có gì đặc biệt" hoặc không cung cấp, để trống.
     - `food_preferences` (string): Sở thích cụ thể ("thích đồ ngọt", "thích món cay", "cần món giải nhiệt"). Nếu người dùng nói "gì cũng được", "tùy bạn" hoặc không cung cấp, để trống.
     - `food_dislikes` (string): Không thích/kiêng cữ ("không ăn hành", "không uống đồ có cồn", "tránh đồ nhiều dầu mỡ"). GHI NHẬN CẨN THẬN CÁC YÊU CẦU LOẠI TRỪ.
     - `health_goals` (string): (Tương tự `health_condition`)
   *QUAN TRỌNG*:
     - Các thông tin cốt lõi như `health_condition`, `allergies`, `health_goals` một khi đã được cung cấp phải được GIỮ LẠI trong `collected_info` qua các lượt phân tích sau, trừ khi người dùng nói rõ là thông tin đó đã thay đổi.
     - Đối với `dietary_habits`, `food_preferences`, `food_dislikes`: Nếu người dùng ở lượt TRƯỚC đã cung cấp, nhưng ở TIN NHẮN MỚI NHẤT lại nói "ăn gì cũng được", "bình thường", thì các trường này trong `collected_info` nên được làm rỗng hoặc phản ánh sự không chắc chắn đó, và đây là một tín hiệu cho `user_rejected_info`.

**3. XỬ LÝ INPUT KHÔNG MẠCH LẠC HOẶC NGOÀI PHẠM VI RÕ RÀNG:**
   - Nếu TIN NHẮN MỚI NHẤT hoàn toàn vô nghĩa, không phải câu hỏi, hoặc rõ ràng nằm ngoài phạm vi tư vấn (ví dụ: một từ đơn, một dãy số, tên riêng không kèm ngữ cảnh), hãy đặt:
     - `is_valid_scope`: `false`
     - `is_food_related`: `false`
     - `need_more_info`: `false`
     - `follow_up_question`: `null`
     - `collected_info`: giữ nguyên từ lịch sử nếu có, không thêm gì mới.

**4. ĐÁNH GIÁ THÁI ĐỘ NGƯỜI DÙNG VÀ QUYẾT ĐỊNH HƯỚNG HÀNH XỬ CỦA CHATBOT:**
   - `user_rejected_info` (boolean): Phân tích TIN NHẮN MỚI NHẤT. Người dùng có đang từ chối (rõ ràng hoặc ngầm) cung cấp THÊM THÔNG TIN CHI TIẾT về sở thích/thói quen, SAU KHI chatbot đã đặt câu hỏi gợi ý không?
     *   Các ví dụ từ chối bao gồm: "tôi không muốn nói", "tôi không rõ", "tôi không biết", "sao cũng được", "gì cũng được", "tùy bạn", "bạn cứ gợi ý đi", "cho tôi ví dụ", "gia đình tôi ăn uống bình thường", "không có yêu cầu gì đặc biệt", "không có sở thích cụ thể".
     *   **ĐẶC BIỆT QUAN TRỌNG**: Nếu chatbot hỏi "Bạn thích loại A, B, hay C?" và người dùng trả lời "Loại nào cũng được nhưng trừ X" hoặc "Tôi không biết chọn loại nào, bạn gợi ý đi", thì `user_rejected_info` (cho việc chọn loại cụ thể A,B,C) là `true`.
   - `suggest_general_options` (boolean): Đặt `true` NẾU ĐỒNG THỜI CÁC ĐIỀU KIỆN SAU ĐÚNG:
     *   a) `is_valid_scope` là true, VÀ
     *   b) `is_food_related` là true (người dùng quan tâm đến món ăn/đồ uống), VÀ
     *   c) ( `user_rejected_info` là `true` (người dùng không muốn/không thể cung cấp thêm chi tiết về SỞ THÍCH/THÓI QUEN ĂN UỐNG CỤ THỂ)
           HOẶC (thông tin về `dietary_habits`, `food_preferences`, `food_dislikes` trong `collected_info` là RẤT ÍT hoặc KHÔNG CÓ, VÀ người dùng không cung cấp thêm chi tiết khi được hỏi ở lượt trước)
           HOẶC (query đơn giản và rõ ràng muốn gợi ý ngay lập tức như "thêm món chay đi", "gợi ý món chay", "món chay nào ngon", mà không có yêu cầu cụ thể) ), VÀ
     *   d) Thông tin về SỞ THÍCH CÁ NHÂN cụ thể (ngoài các điều kiện như "không cồn") là KHÔNG ĐỦ để cá nhân hóa sâu sắc gợi ý món ăn/đồ uống.
     *   **QUY TẮC ƯU TIÊN THÔNG TIN SỨC KHỎE:** Nếu `collected_info.health_condition` hoặc `collected_info.health_goals` chứa thông tin chi tiết và cụ thể (ví dụ: nhiều hơn 2-3 từ mô tả bệnh/mục tiêu), thì `suggest_general_options` NÊN LÀ `false`, trừ khi người dùng RÕ RÀNG yêu cầu gợi ý chung hoặc từ chối cung cấp thêm chi tiết về món ăn.
     *Khi `suggest_general_options` là `true`, chatbot sẽ không hỏi thêm về sở thích chung nữa, mà sẽ đưa ra gợi ý dựa trên các tiêu chí phổ biến và các `health_condition` + `food_dislikes` (ví dụ "không cồn") đã biết.*
   - `need_more_info` (boolean):
     *   **QUY TẮC THÉP 1: NẾU `user_rejected_info` là `true`, thì `need_more_info` PHẢI LÀ `FALSE`.**
     *   **QUY TẮC THÉP 2: NẾU `suggest_general_options` là `true`, thì `need_more_info` PHẢI LÀ `FALSE`.**
     *   Chỉ đặt `true` nếu cả hai quy tắc trên không áp dụng VÀ thông tin hiện tại THỰC SỰ QUÁ MƠ HỒ hoặc THIẾU CỐT LÕI (ví dụ: thiếu hoàn toàn thông tin về tình trạng sức khỏe khi người dùng muốn tư vấn theo bệnh, hoặc chỉ nói "tôi muốn ăn" mà không có bất kỳ yêu cầu nào khác) để có thể đưa ra bất kỳ loại tư vấn nào (kể cả tư vấn chung có xem xét bệnh lý).
   - `follow_up_question` (string | null):
     *   **QUY TẮC THÉP: Chỉ được tạo khi `need_more_info` là `true`.** (Điều này đảm bảo các quy tắc thép của `need_more_info` được tuân thủ).
     *   Nếu tạo, câu hỏi phải TẬP TRUNG vào thông tin CÒN THIẾU QUAN TRỌNG NHẤT và CHƯA BỊ TỪ CHỐI.
     *   Nếu chatbot đã hỏi về sở thích A, B, C và người dùng nói "gì cũng được", thì `follow_up_question` tiếp theo KHÔNG ĐƯỢC hỏi lại về A, B, C.
     *   Ví dụ tình huống:
         - User: "Gợi ý món cho người tiểu đường."
         - Bot: "Bạn có sở thích cụ thể nào không (cay, ngọt, mặn)?"
         - User: "Tôi không có sở thích cụ thể, chỉ cần tốt cho người tiểu đường."
         - Phân tích cho lượt cuối của user: `user_rejected_info=true` (cho sở thích), `suggest_general_options=true`, `need_more_info=false`, `follow_up_question=null`. `collected_info.health_condition="tiểu đường"`.
     *   Ví dụ tình huống (cần hỏi thêm):
         - User: "Tôi muốn món ăn tốt cho sức khỏe."
         - Phân tích: `is_valid_scope=true`, `is_food_related=true`. `collected_info` rỗng. `user_rejected_info=false`, `suggest_general_options=false`. => `need_more_info=true`.
         - `follow_up_question`: "Tuyệt vời! Để tôi có thể gợi ý chính xác hơn, bạn có thể chia sẻ thêm về mục tiêu sức khỏe cụ thể của mình (ví dụ: giảm cân, tăng cường năng lượng) hoặc bạn có tình trạng sức khỏe nào cần lưu ý không ạ?"
     *   Nếu không cần hỏi thêm, PHẢI là `null`.

**CẤU TRÚC JSON OUTPUT (TUYỆT ĐỐI CHỈ TRẢ VỀ JSON NÀY, KHÔNG THÊM BẤT KỲ GIẢI THÍCH NÀO):**
```json
{{
  "is_valid_scope": boolean,
  "is_food_related": boolean,
  "requests_food": boolean,
  "requests_beverage": boolean,
  "user_rejected_info": boolean,
  "need_more_info": boolean,
  "suggest_general_options": boolean,
  "follow_up_question": string | null,
  "collected_info": {{
    "health_condition": "string",
    "medical_history": "string",
    "allergies": "string",
    "dietary_habits": "string",
    "food_preferences": "string",
    "food_dislikes": "string",
    "health_goals": "string"
  }}
}}
```"""
        
        return prompt
    
    def _create_medichat_prompt_template(self, messages: List[Dict[str, str]], recipes: List[Dict[str, Any]] = None, beverages: List[Dict[str, Any]] = None, suggest_general: bool = False, current_summary: Optional[str] = None) -> str:
        """
        Tạo template prompt để tóm tắt thông tin cho Medichat.
        Nếu có recipes hoặc beverages, đưa hết vào và giới hạn prompt tổng là 400 TỪ.
        Nếu suggest_general là true, yêu cầu Medichat gợi ý chung.
        
        Args:
            messages: Danh sách tin nhắn
            recipes: Danh sách công thức món ăn (nếu có)
            beverages: Danh sách đồ uống (nếu có)
            suggest_general: True nếu cần Medichat gợi ý theo tiêu chí chung.
            current_summary: Tóm tắt cuộc trò chuyện từ lượt trước (nếu có)
            
        Returns:
            Prompt cho Gemini để tạo prompt Medichat
        """
        # Xác định giới hạn từ dựa trên có recipes/beverages hay không hoặc suggest_general
        word_limit = self.max_medichat_prompt_words_with_context if (recipes or beverages or suggest_general) else 900
        
        # ⭐ XÂY DỰNG NGỮ CẢNH CUỘC TRÒ CHUYỆN DỰA TRÊN TÓM TẮT VÀ TIN NHẮN GẦN ĐÂY
        context_for_gemini = ""
        
        if current_summary:
            # Có tóm tắt từ lượt trước - sử dụng làm ngữ cảnh chính
            context_for_gemini += f"BẢN TÓM TẮT CUỘC TRÒ CHUYỆN HIỆN TẠI (ƯU TIÊN SỬ DỤNG LÀM NGỮ CẢNH CHÍNH):\n{current_summary}\n\n"
            context_for_gemini += "LỊCH SỬ CHAT CHI TIẾT GẦN ĐÂY (chỉ để tham khảo thêm hoặc làm rõ thông tin từ tóm tắt nếu cần, đặc biệt là tin nhắn mới nhất chưa có trong tóm tắt):\n"
            
            # Chỉ lấy vài tin nhắn cuối khi có tóm tắt
            num_recent_messages = 3
        else:
            # Không có tóm tắt - sử dụng lịch sử chat như cũ
            context_for_gemini += "LỊCH SỬ CHAT (vì chưa có tóm tắt, hãy dựa vào đây để lấy ngữ cảnh):\n"
            num_recent_messages = 10
        
        # Xây dựng phần tin nhắn gần đây
        conversation_text = "\n\n"
        total_chars = 0
        max_conversation_chars = 8000 if current_summary else 14000  # Giảm giới hạn khi có tóm tắt
        
        # Lấy N tin nhắn cuối cùng
        recent_messages = messages[-num_recent_messages:] if len(messages) > num_recent_messages else messages
        
        for msg in recent_messages:
            if msg["role"] != "system":  # Bỏ qua system message
                role = "Người dùng" if msg["role"] == "user" else "Trợ lý"
                content = msg['content']
                
                # Cắt bớt nội dung nếu quá dài
                max_content_length = 300 if current_summary else 500
                if len(content) > max_content_length:
                    content = content[:max_content_length] + "... [nội dung đã cắt ngắn]"
                
                msg_text = f"{role}: {content}\n\n"
                
                # Kiểm tra xem có vượt quá giới hạn không
                if total_chars + len(msg_text) > max_conversation_chars:
                    conversation_text += "[...một số tin nhắn đã được bỏ qua để đảm bảo không vượt quá giới hạn token...]\n\n"
                    break
                
                conversation_text += msg_text
                total_chars += len(msg_text)
        
        # Kết hợp ngữ cảnh hoàn chỉnh
        full_context = context_for_gemini + conversation_text
        
        # ⭐ TẠOS PHẦN RECIPES VỚI TIÊU ĐỀ RÕ RÀNG - ĐƯA TOÀN BỘ RECIPES VÀO
        recipe_section = ""
        if recipes:
            recipe_section = "\n\n### DANH SÁCH CÔNG THỨC MÓN ĂN THAM KHẢO TỪ DATABASE:\n"
            for i, recipe in enumerate(recipes, 1):  # Đưa TOÀN BỘ recipes vào (không giới hạn)
                recipe_id = recipe.get('id', f'R{i}')
                name = recipe.get('name', 'N/A')
                ingredients = recipe.get('ingredients_summary', 'N/A')
                url = recipe.get('url', '')
                
                # ⭐ GIỮ NGUYÊN INGREDIENTS_SUMMARY CHÍNH XÁC, KHÔNG CẮT NGẮN
                recipe_section += f"{i}. [ID: {recipe_id}] {name}\n   - Nguyên liệu: {ingredients}\n"
                if url and len(url) < 50:  # Chỉ thêm URL nếu không quá dài
                    recipe_section += f"   - Link: {url}\n"
        
        # ⭐ TẠO PHẦN BEVERAGES VỚI TIÊU ĐỀ RÕ RÀNG - ĐƯA TOÀN BỘ BEVERAGES VÀO
        beverage_section = ""
        if beverages:
            beverage_section = "\n\n### DANH SÁCH ĐỒ UỐNG THAM KHẢO TỪ DATABASE:\n"
            for i, bev in enumerate(beverages, 1):  # Đưa TOÀN BỘ beverages vào (không giới hạn)
                bev_id = bev.get('product_id', f'B{i}')
                name = bev.get('product_name', 'N/A')
                
                beverage_section += f"{i}. [ID: {bev_id}] {name}\n"
        
        # Tạo phần instruction cho suggest_general
        general_instruction = ""
        if suggest_general:
            general_instruction = "\n\nLƯU Ý QUAN TRỌNG CHO VIỆC TẠO PROMPT MEDICHAT:\n" \
                                "Người dùng không cung cấp đủ thông tin cụ thể. Hãy tạo một prompt yêu cầu Medichat gợi ý 2-3 MÓN ĂN HOẶC ĐỒ UỐNG CỤ THỂ dựa trên các tiêu chí chung sau:\n" \
                                "- Tính phổ biến: Món ăn/đồ uống được nhiều người biết đến và yêu thích\n" \
                                "- Tính đa dạng: Gợi ý các loại khác nhau nếu hợp lý (ví dụ: 1 món ăn chính, 1 đồ uống, 1 món tráng miệng)\n" \
                                "- Cân bằng dinh dưỡng cơ bản: Có đủ các nhóm chất dinh dưỡng thiết yếu\n" \
                                "- Ít gây dị ứng phổ biến: Tránh các thành phần dễ gây dị ứng như hải sản, đậu phộng\n" \
                                "- Dễ chế biến/dễ tìm: Nguyên liệu dễ kiếm, cách làm không quá phức tạp\n" \
                                "Prompt cho Medichat phải yêu cầu Medichat KHÔNG HỎI THÊM mà đưa ra gợi ý trực tiếp.\n\n" \
                                "🎯 YÊU CẦU TUYỆT ĐỐI ƯU TIÊN SỬ DỤNG DỮ LIỆU TỪ DATABASE:\n" \
                                "QUAN TRỌNG TUYỆT ĐỐI: Khi có danh sách món ăn (recipe_section) hoặc đồ uống (beverage_section) được cung cấp, " \
                                "hãy tạo prompt yêu cầu Medichat PHẢI **dựa vào và ưu tiên phân tích TẤT CẢ các items trong danh sách này trước tiên**. " \
                                "Medichat cần xác định những items nào trong danh sách này phù hợp nhất với các tiêu chí chung " \
                                "(phổ biến, đa dạng, cân bằng dinh dưỡng, ít dị ứng, dễ làm). " \
                                "Sau đó, Medichat có thể bổ sung bằng kiến thức của mình nếu danh sách không có gì hoàn toàn phù hợp hoặc cần thêm lựa chọn. " \
                                "Prompt cho Medichat phải rõ ràng rằng các recipes/beverages cung cấp là nguồn thông tin chính cần được khai thác tối đa."
        
        # Tạo prompt cho Gemini
        prompt = f""""Bạn là một trợ lý y tế thông minh, chuyên tóm tắt thông tin từ cuộc trò chuyện để tạo ra một prompt ngắn gọn, súc tích và đầy đủ thông tin nhất cho mô hình AI y tế chuyên sâu Medichat-LLaMA3-8B.

NGỮ CẢNH CUỘC TRÒ CHUYỆN ĐỂ TẠO PROMPT CHO MEDICHAT:
{full_context}

{recipe_section}{beverage_section}{general_instruction}

YÊU CẦU TẠO PROMPT CHO MEDICHAT (DỰA TRÊN NGỮ CẢNH TRÊN, ĐẶC BIỆT LÀ BẢN TÓM TẮT NẾU CÓ VÀ TIN NHẮN MỚI NHẤT):
1. Nội dung cốt lõi:
   - Phân tích yêu cầu chính từ TIN NHẮN MỚI NHẤT của người dùng (trong LỊCH SỬ CHAT CHI TIẾT GẦN ĐÂY).
   - Kết hợp với các thông tin quan trọng đã được đúc kết trong BẢN TÓM TẮT (nếu có) như tình trạng sức khỏe, mục tiêu, sở thích đã biết.
   - Nếu `general_instruction` có nội dung (suggest_general=true): Tạo prompt yêu cầu Medichat thực hiện gợi ý chung theo các tiêu chí đã nêu. Có thể tham khảo `recipe_section` nếu có món phù hợp với tiêu chí chung.
   - Nếu không có `general_instruction`: Tập trung vào yêu cầu chính/vấn đề mà người dùng đang hỏi, bao gồm triệu chứng/tình trạng sức khỏe, bệnh lý nền/dị ứng, thông tin về món ăn/chế độ dinh dưỡng quan tâm, mục tiêu dinh dưỡng/sức khỏe, và thói quen ăn uống đã đề cập.

2. Định dạng Prompt:
- Viết bằng NGÔI THỨ NHẤT, như thể người dùng đang trực tiếp đặt câu hỏi cho Medichat.
- Prompt phải là một YÊU CẦU RÕ RÀNG, dễ hiểu.
- Ví dụ cấu trúc (linh hoạt điều chỉnh tùy theo ngữ cảnh):
+ Nếu hỏi món ăn: "Tôi bị [tình trạng sức khỏe ví dụ: tiểu đường, dị ứng hải sản], muốn [mục tiêu ví dụ: kiểm soát đường huyết]. Xin gợi ý [số lượng] món [loại món ví dụ: canh, xào] phù hợp, [yêu cầu thêm ví dụ: ít gia vị, dễ làm]."
+ Nếu hỏi tư vấn chung: "Tôi bị [tình trạng sức khỏe], đang theo [thói quen ăn uống]. Tôi nên điều chỉnh chế độ ăn uống như thế nào để [mục tiêu sức khỏe]?"
+ Nếu gợi ý chung: "Tôi cần gợi ý món ăn/đồ uống [dựa trên tiêu chí từ general_instruction]. Xin đưa ra 2-3 lựa chọn cụ thể."

3. XỬ LÝ CÔNG THỨC MÓN ĂN/ĐỒ UỐNG - TẬN DỤNG TỐI ĐA DỮ LIỆU:

⭐ QUAN TRỌNG TUYỆT ĐỐI - YÊU CẦU NHẤT QUÁN MENU: Khi Medichat đưa ra gợi ý về một món ăn hoặc đồ uống, Medichat PHẢI TUÂN THỦ NGHIÊM NGẶT các quy tắc sau:
   a. CHỈ GỢI Ý CÁC MÓN ĂN/ĐỒ UỐNG CÓ TRONG DANH SÁCH ĐƯỢC CUNG CẤP: Medichat TUYỆT ĐỐI KHÔNG ĐƯỢC tự tạo ra món ăn/đồ uống mới ngoài danh sách recipes và beverages đã cung cấp.
   b. SỬ DỤNG CHÍNH XÁC NGUYÊN LIỆU TỪ DATABASE: Khi gợi ý một món từ danh sách, Medichat PHẢI trích xuất và sử dụng CHÍNH XÁC danh sách nguyên liệu từ trường ingredients_summary (đối với món ăn) hoặc tên sản phẩm (đối với đồ uống) được cung cấp.
   c. KHÔNG TỰ Ý THAY ĐỔI NGUYÊN LIỆU: Medichat TUYỆT ĐỐI KHÔNG ĐƯỢC thay đổi, thêm bớt, hay suy diễn danh sách nguyên liệu. Phải giữ nguyên như dữ liệu được cung cấp.
   d. ĐỊNH DẠNG TRÌNH BÀY CHUẨN: Trình bày danh sách nguyên liệu rõ ràng dưới dạng: **Nguyên liệu:** [nguyên liệu 1], [nguyên liệu 2], ...
   e. ƯU TIÊN SỬ DỤNG DỮ LIỆU CÓ SẴN: Nếu có cả recipes và beverages, Medichat phải ưu tiên phân tích và chọn từ danh sách này trước khi bổ sung kiến thức bên ngoài.
   f. Prompt bạn (Gemini) tạo ra cho Medichat PHẢI chứa chỉ dẫn rõ ràng: 'Bạn PHẢI chỉ gợi ý các món ăn/đồ uống có trong danh sách được cung cấp và sử dụng CHÍNH XÁC nguyên liệu từ trường ingredients_summary, không tự tạo ra món mới hay thay đổi nguyên liệu.'

- Khi suggest_general=True VÀ có recipe_section hoặc beverage_section:
  + Hướng dẫn Medichat PHẢI ƯU TIÊN TUYỆT ĐỐI các món ăn trong recipe_section và đồ uống trong beverage_section
  + Yêu cầu Medichat phân tích KỸ LƯỠNG TỪNG item trong DANH SÁCH CÔNG THỨC MÓN ĂN và DANH SÁCH ĐỒ UỐNG để chọn ra 2-3 items phù hợp nhất với tiêu chí gợi ý chung
  + ⭐ QUAN TRỌNG NHẤT QUÁN: Medichat CHỈ ĐƯỢC chọn từ danh sách được cung cấp và PHẢI sử dụng CHÍNH XÁC nguyên liệu từ trường ingredients_summary
  + ⭐ QUAN TRỌNG: Medichat phải bao gồm DANH SÁCH NGUYÊN LIỆU CHI TIẾT cho từng món được gợi ý, sử dụng ĐÚNG dữ liệu từ database
  + TUYỆT ĐỐI KHÔNG ĐƯỢC bổ sung món ăn/đồ uống ngoài danh sách đã cung cấp
  + VÍ DỤ PROMPT CHO MEDICHAT: "Tôi muốn gợi ý đồ uống giải nhiệt. Hãy chỉ chọn từ danh sách đồ uống sau: [Nước ép Xoài (ID:B1), Trà Chanh (ID:B2)]. Chọn 2-3 loại phù hợp nhất và sử dụng CHÍNH XÁC thành phần từ database. Bạn PHẢI chỉ gợi ý các món có trong danh sách và sử dụng CHÍNH XÁC nguyên liệu từ trường ingredients_summary, không tự tạo ra món mới hay thay đổi nguyên liệu."

- Khi KHÔNG phải suggest_general=True (người dùng có yêu cầu cụ thể) VÀ có recipe_section hoặc beverage_section:
  + Tạo prompt hướng dẫn Medichat PHẢI CHỈ SỬ DỤNG các món ăn từ recipe_section và/hoặc đồ uống từ beverage_section
  + Yêu cầu Medichat phân tích KỸ LƯỠNG TỪNG item trong DANH SÁCH để xem item nào khớp nhất với yêu cầu của người dùng, sau đó chọn ra 2-3 item phù hợp nhất
  + ⭐ QUAN TRỌNG NHẤT QUÁN: Medichat CHỈ ĐƯỢC chọn từ danh sách được cung cấp và PHẢI sử dụng CHÍNH XÁC nguyên liệu từ trường ingredients_summary
  + Medichat phải đánh giá từng item và giải thích chi tiết tại sao chúng phù hợp với yêu cầu cụ thể của người dùng
  + ⭐ QUAN TRỌNG: Medichat phải bao gồm DANH SÁCH NGUYÊN LIỆU CHI TIẾT cho từng món được gợi ý, sử dụng ĐÚNG dữ liệu từ database
  + TUYỆT ĐỐI KHÔNG ĐƯỢC tự tạo ra món ăn/đồ uống mới ngoài danh sách
  + VÍ DỤ PROMPT CHO MEDICHAT: "Tôi bị tiểu đường. Từ danh sách sau: [Canh bí đao (ID:R1, Nguyên liệu: bí đao, thịt nạc, hành), Gà kho gừng (ID:R2, Nguyên liệu: gà ta, gừng, nước mắm, đường)], món nào tốt nhất cho tôi? Hãy chỉ chọn từ danh sách này và sử dụng CHÍNH XÁC nguyên liệu đã cho. Bạn PHẢI chỉ gợi ý các món có trong danh sách và sử dụng CHÍNH XÁC nguyên liệu từ trường ingredients_summary, không tự tạo ra món mới hay thay đổi nguyên liệu."

- Khi có cả món ăn và đồ uống từ database:
  + Tạo prompt yêu cầu Medichat CHỈ ĐƯỢC phân tích và chọn từ recipe_section và beverage_section được cung cấp
  + Medichat phải đưa ra gợi ý kết hợp hài hòa từ cả hai danh sách, đảm bảo phù hợp với yêu cầu/tình trạng sức khỏe
  + ⭐ QUAN TRỌNG NHẤT QUÁN: Medichat CHỈ ĐƯỢC chọn từ danh sách được cung cấp và PHẢI sử dụng CHÍNH XÁC nguyên liệu từ trường ingredients_summary
  + TUYỆT ĐỐI KHÔNG ĐƯỢC tự tạo ra món ăn/đồ uống mới hoặc thay đổi nguyên liệu
  + KHÔNG ĐƯỢC bổ sung kiến thức bên ngoài - chỉ sử dụng dữ liệu có sẵn trong danh sách

4. Giới hạn:
- TOÀN BỘ prompt kết quả CHO MEDICHAT PHẢI DƯỚI {word_limit} TỪ.
- Cần cực kỳ súc tích và đúng trọng tâm. CHỈ bao gồm thông tin đã được đề cập trong cuộc trò chuyện. KHÔNG suy diễn, KHÔNG thêm thông tin không có.

5. Mục tiêu: Tạo ra prompt hiệu quả nhất để Medichat có thể đưa ra câu trả lời y tế chính xác và hữu ích

⭐ YÊU CẦU TUYỆT ĐỐI VỀ NGUYÊN LIỆU - QUAN TRỌNG NHẤT:
- BẮT BUỘC: Khi Medichat gợi ý bất kỳ món ăn hoặc đồ uống nào, PHẢI bao gồm danh sách nguyên liệu chi tiết
- 🚨 NGUYÊN LIỆU PHẢI TRÙNG KHỚP CHÍNH XÁC VỚI RECIPE-INDEX: Nếu món ăn có trong danh sách recipes được cung cấp, Medichat PHẢI sử dụng CHÍNH XÁC nguyên liệu từ trường "ingredients_summary" của recipe đó, KHÔNG ĐƯỢC tự tạo ra hoặc thay đổi
- Định dạng CHÍNH XÁC: "**Nguyên liệu:** [liệt kê từng nguyên liệu cách nhau bằng dấu phẩy]"
- Ví dụ CHUẨN: "**Nguyên liệu:** thịt bò, rau cải, tỏi, nước mắm, dầu ăn, tiêu"
- KHÔNG ĐƯỢC BỎ QUA: Điều này giúp người dùng biết chính xác cần mua gì để thực hiện món ăn được gợi ý
- PROMPT CHO MEDICHAT PHẢI BAO GỒM: "Hãy sử dụng CHÍNH XÁC nguyên liệu từ danh sách recipes được cung cấp, không tự tạo ra nguyên liệu khác"

🚨 LƯU Ý QUAN TRỌNG NHẤT - YÊU CẦU NHẤT QUÁN MENU:
Prompt bạn tạo ra PHẢI chứa các câu yêu cầu NGHIÊM NGẶT sau:
- "Bạn CHỈ ĐƯỢC gợi ý các món ăn/đồ uống có trong danh sách được cung cấp"
- "TUYỆT ĐỐI KHÔNG ĐƯỢC tự tạo ra món ăn/đồ uống mới ngoài danh sách"
- "Hãy sử dụng CHÍNH XÁC nguyên liệu từ trường 'ingredients_summary' của từng món trong danh sách"
- "KHÔNG ĐƯỢC tự tạo ra hoặc thay đổi nguyên liệu, phải dùng đúng như trong database"
- "Bao gồm danh sách nguyên liệu CHỈ THEO ĐÚNG thông tin đã cung cấp"
- "Khi bạn gợi ý một món từ danh sách, hãy đảm bảo bạn trích xuất và liệt kê chính xác danh sách nguyên liệu từ trường 'ingredients_summary', không tự ý thay đổi hay bổ sung."

CHỈ TRẢ VỀ PHẦN PROMPT ĐÃ ĐƯỢC TÓM TẮT VÀ TỐI ƯU HÓA CHO MEDICHAT, KHÔNG BAO GỒM BẤT KỲ LỜI GIẢI THÍCH HAY TIÊU ĐỀ NÀO KHÁC.
PROMPT KẾT QUẢ (DƯỚI {word_limit} TỪ):"""
        
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
        prompt = f"""Bạn là một chuyên gia biên tập nội dung y tế và dinh dưỡng. Nhiệm vụ của bạn là xem xét phản hồi từ một mô hình AI y tế (Medichat) và tinh chỉnh nó để tạo ra một câu trả lời HOÀN HẢO, SẠCH SẼ, và THÂN THIỆN cho người dùng.

PROMPT GỐC ĐÃ GỬI CHO MEDICHAT:
{original_prompt}

PHẢN HỒI THÔ TỪ MEDICHAT:
{medichat_response}

HƯỚNG DẪN BIÊN TẬP VÀ TINH CHỈNH:
1. Đánh giá chất lượng phản hồi thô:
- Nội dung có CHÍNH XÁC về mặt y tế/dinh dưỡng không?
- Có TRẢ LỜI TRỰC TIẾP và ĐẦY ĐỦ cho PROMPT GỐC không?
- 🚨 QUAN TRỌNG: Có bao gồm DANH SÁCH NGUYÊN LIỆU CHI TIẾT cho từng món ăn/đồ uống được gợi ý không?
- Ngôn ngữ có DỄ HIỂU, THÂN THIỆN, và PHÙ HỢP với người dùng không?
- Có chứa thông tin thừa, metadata, hoặc các cụm từ không tự nhiên (ví dụ: "dưới đây là...", "đánh giá của tôi...") không?
2. Hành động:
- Nếu phản hồi thô đã tốt (chính xác, đầy đủ, dễ hiểu) VÀ có đầy đủ nguyên liệu: Hãy loại bỏ TOÀN BỘ metadata, các cụm từ đánh giá, định dạng thừa. Giữ lại phần nội dung cốt lõi. ĐẢM BẢO NỘI DUNG CUỐI CÙNG SỬ DỤNG KÝ TỰ XUỐNG DÒNG `\n` MỘT CÁCH HỢP LÝ: một `\n` để ngắt dòng trong cùng một ý hoặc tạo danh sách, và hai `\n\n` để tạo khoảng cách rõ ràng giữa các đoạn văn, các mục lớn (ví dụ: giữa các món ăn được gợi ý, hoặc giữa phần mô tả và phần nguyên liệu).
- Nếu phản hồi thô THIẾU NGUYÊN LIỆU: Hãy BỔ SUNG danh sách nguyên liệu chi tiết cho từng món ăn/đồ uống được gợi ý theo định dạng "**Nguyên liệu:** [danh sách]". 🚨 QUAN TRỌNG: Chỉ sử dụng nguyên liệu từ kiến thức chung về món ăn đó, KHÔNG tự tạo ra nguyên liệu lạ hoặc không phù hợp
- Nếu phản hồi thô chưa tốt (lạc đề, không đầy đủ, khó hiểu, chứa thông tin sai lệch, hoặc quá máy móc): Hãy VIẾT LẠI HOÀN TOÀN một phản hồi mới dựa trên PROMPT GỐC. Phản hồi mới phải chính xác, đầy đủ, thân thiện, dễ hiểu, cung cấp giá trị thực sự cho người dùng VÀ ĐẢM BẢO SỬ DỤNG KÝ TỰ XUỐNG DÒNG `\n` MỘT CÁCH HỢP LÝ như đã mô tả ở trên VÀ BẮT BUỘC có nguyên liệu cho từng món.
- Nếu trình bày danh sách (ví dụ: nguyên liệu, cách làm), hãy sử dụng dấu gạch đầu dòng (`- ` hoặc `* ` bắt đầu mỗi mục) hoặc đánh số (`1. `, `2. `). Mỗi mục trong danh sách PHẢI được trình bày trên một dòng mới (sử dụng `\n` để ngắt dòng).
3. YÊU CẦU TUYỆT ĐỐI CHO ĐẦU RA CUỐI CÙNG:
- Đầu ra của bạn sẽ được gửi TRỰC TIẾP cho người dùng.
- KHÔNG BAO GIỜ bao gồm các từ/cụm từ như: "Đánh giá:", "Kiểm tra:", "Điều chỉnh:", "Phản hồi đã được điều chỉnh:", "Phân tích phản hồi:", "HỢP LỆ", "Dưới đây là...", "Theo tôi...", v.v.
- KHÔNG BAO GIỜ chia phản hồi thành các phần có tiêu đề kiểu "1. Đánh giá", "2. Điều chỉnh".
- KHÔNG BAO GIỜ nhắc đến quá trình đánh giá hay sửa đổi nội bộ.
- LUÔN viết như thể bạn đang trực tiếp trò chuyện và tư vấn cho người dùng.
- LUÔN sử dụng tiếng Việt tự nhiên, thân thiện, chuyên nghiệp và mạch lạc.
- LUÔN đảm bảo thông tin y tế/dinh dưỡng là chính xác và hữu ích.
- Đảm bảo phản hồi ngắn gọn, súc tích nhất có thể mà vẫn đủ ý.
- LUÔN đảm bảo định dạng xuống dòng nhất quán và dễ đọc, sử dụng \n cho ngắt dòng và \n\n cho ngắt đoạn.
TRẢ VỀ NGAY LẬP TỨC CHỈ PHẦN NỘI DUNG PHẢN HỒI CUỐI CÙNG DÀNH CHO NGƯỜI DÙNG. KHÔNG CÓ BẤT KỲ METADATA, GIẢI THÍCH, HAY BÌNH LUẬN NÀO.
"""
        
        return prompt
    
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
        prompt = self._create_medichat_prompt_template(conversation_messages)
        
        # Sử dụng biến global GOOGLE_AI_AVAILABLE
        global GOOGLE_AI_AVAILABLE
        
        # Thử sử dụng thư viện Google Generative AI nếu có sẵn
        if GOOGLE_AI_AVAILABLE:
            try:
                summary = await self._query_gemini_with_client(prompt)
                logger.info(f"[GEMINI SUMMARY] {summary}")
                return summary
            except Exception as e:
                logger.warning(f"Lỗi khi sử dụng Google Generative AI client: {str(e)}. Thử sử dụng HTTP API trực tiếp.")
                GOOGLE_AI_AVAILABLE = False
        
        # Sử dụng HTTP API trực tiếp nếu thư viện không có sẵn hoặc gặp lỗi
        summary = await self._query_gemini_with_http(prompt)
        logger.info(f"[GEMINI SUMMARY] {summary}")
        return summary
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def create_recipe_search_prompt(self, user_message: str, collected_info: Dict[str, Any], suggest_general_if_needed: bool = False) -> str:
        """
        Tạo prompt tối ưu cho recipe_tool từ yêu cầu người dùng và thông tin thu thập được
        
        Args:
            user_message: Tin nhắn của người dùng về món ăn
            collected_info: Thông tin sức khỏe đã thu thập được
            suggest_general_if_needed: True nếu cần tạo query tìm kiếm chung
            
        Returns:
            Prompt tối ưu cho recipe search
        """
        if not self.api_key:
            # Fallback prompt dựa trên suggest_general_if_needed
            if suggest_general_if_needed:
                return "các món ăn đồ uống phổ biến tốt cho sức khỏe dễ làm"
            
            if collected_info:
                conditions = []
                if collected_info.get('health_condition'):
                    conditions.append(f"phù hợp với {collected_info['health_condition']}")
                if collected_info.get('allergies'):
                    conditions.append(f"không có {collected_info['allergies']}")
                if collected_info.get('dietary_habits'):
                    conditions.append(f"theo chế độ {collected_info['dietary_habits']}")
                
                if conditions:
                    return f"{user_message}. {'. '.join(conditions)}"
            
            return f"{user_message}. Món ăn phổ biến, cân bằng dinh dưỡng, dễ chế biến"
        
        # Tạo prompt thông minh bằng Gemini dựa trên suggest_general_if_needed
        if suggest_general_if_needed:
            prompt = f"""Bạn là chuyên gia dinh dưỡng và ẩm thực. Nhiệm vụ của bạn là tạo ra một câu truy vấn chung để tìm kiếm công thức món ăn phù hợp với nhiều người.

YÊU CẦU CỦA NGƯỜI DÙNG:
"{user_message}"

THÔNG TIN SỨC KHỎE (có thể không đầy đủ):
{json.dumps(collected_info, ensure_ascii=False, indent=2) if collected_info else "Không có thông tin cụ thể"}

NHIỆM VỤ:
Tạo một câu truy vấn ngắn gọn để tìm kiếm các món ăn/đồ uống dựa trên các tiêu chí CHUNG sau:
- Tính phổ biến: Món ăn/đồ uống được nhiều người biết đến và yêu thích
- Tính đa dạng: Có thể bao gồm các loại khác nhau (món chính, đồ uống, tráng miệng)
- Cân bằng dinh dưỡng cơ bản: Có đủ các nhóm chất dinh dưỡng thiết yếu
- Ít gây dị ứng phổ biến: Tránh hải sản, đậu phộng, các thành phần dễ gây dị ứng
- Dễ chế biến/dễ tìm: Nguyên liệu dễ kiếm, cách làm không quá phức tạp

CẤU TRÚC QUERY MONG MUỐN:
"[Loại món/đồ uống chung] + [tiêu chí phổ biến] + [cân bằng dinh dưỡng] + [dễ làm]"

Ví dụ output:
- "gợi ý món ăn dinh dưỡng thông thường"
- "các món ăn đồ uống phổ biến tốt cho sức khỏe dễ làm"

CHỈ TRẢ VỀ QUERY CUỐI CÙNG, KHÔNG CÓ GIẢI THÍCH THÊM:"""
        else:
            prompt = f"""Bạn là chuyên gia dinh dưỡng và ẩm thực. Nhiệm vụ của bạn là tạo ra một câu truy vấn tìm kiếm công thức món ăn TỐI ƯU và CHI TIẾT NHẤT.

YÊU CẦU CỤ THỂ CỦA NGƯỜI DÙNG:
"{user_message}"

THÔNG TIN SỨC KHỎE VÀ SỞ THÍCH ĐÃ THU THẬP (QUAN TRỌNG - PHẢI SỬ DỤNG):
{json.dumps(collected_info, ensure_ascii=False, indent=2) if collected_info else "Chưa có thông tin cụ thể."}

NHIỆM VỤ:
Tạo một câu truy vấn NGẮN GỌN (tối đa 15-20 từ) nhưng ĐẦY ĐỦ THÔNG TIN NHẤT để tìm kiếm công thức món ăn. Query này PHẢI phản ánh chính xác yêu cầu trong `user_message` VÀ TẤT CẢ các điều kiện liên quan trong `THÔNG TIN SỨC KHỎE ĐÃ THU THẬP`.

QUY TẮC TẠO QUERY:
1. Bắt đầu bằng loại món hoặc yêu cầu chính từ `user_message`.
2. **TÍCH HỢP MỌI ĐIỀU KIỆN** từ `THÔNG TIN SỨC KHỎE ĐÃ THU THẬP`:
   - Nếu có `health_condition` (ví dụ: "tiểu đường, béo phì, suy dinh dưỡng"), query PHẢI bao gồm các từ khóa như "cho người tiểu đường", "người béo phì", "tăng cân", "suy dinh dưỡng".
   - Nếu có `allergies`, query phải có "không dị ứng [tên dị ứng]".
   - Nếu có `dietary_habits` (ví dụ: "ăn chay"), query phải có "món chay".
   - Nếu có `food_preferences` hoặc `food_dislikes`, cố gắng đưa vào (ví dụ: "ít cay", "không hành").
3. Bao gồm các từ khóa về lợi ích nếu có trong `health_goals` (ví dụ: "giảm cân", "tăng cơ", "tốt cho tim").
4. Giữ query tự nhiên nhưng súc tích.

VÍ DỤ:
User message: "bữa ăn cho gia đình tôi"
Collected_info: {{"health_condition": "bố tiểu đường, mẹ tiểu đường, anh suy dinh dưỡng, tôi béo phì"}}
Output Query mong muốn (ví dụ): "Bữa ăn gia đình phù hợp tiểu đường suy dinh dưỡng béo phì" hoặc "Thực đơn gia đình cho người tiểu đường béo phì và suy dinh dưỡng"

User message: "món canh giải nhiệt"
Collected_info: {{"allergies": "hải sản"}}
Output Query mong muốn: "Canh giải nhiệt không hải sản"

CHỈ TRẢ VỀ QUERY CUỐI CÙNG, KHÔNG CÓ GIẢI THÍCH THÊM:"""

        try:
            if GOOGLE_AI_AVAILABLE:
                try:
                    query = await self._query_gemini_with_client(prompt)
                except Exception as e:
                    logger.warning(f"Lỗi khi sử dụng Google client: {str(e)}. Chuyển sang HTTP API.")
                    query = await self._query_gemini_with_http(prompt)
            else:
                query = await self._query_gemini_with_http(prompt)
            
            # Làm sạch query
            query = query.strip().replace('\n', ' ')
            logger.info(f"Đã tạo recipe search query (suggest_general={suggest_general_if_needed}): {query}")
            return query
                
        except Exception as e:
            logger.error(f"Lỗi khi tạo recipe search prompt: {str(e)}")
            # Fallback
            if suggest_general_if_needed:
                return "các món ăn đồ uống phổ biến tốt cho sức khỏe dễ làm"
            
            if collected_info:
                conditions = []
                if collected_info.get('health_condition'):
                    conditions.append(f"phù hợp với {collected_info['health_condition']}")
                if collected_info.get('allergies'):
                    conditions.append(f"không có {collected_info['allergies']}")
                if collected_info.get('dietary_habits'):
                    conditions.append(f"theo chế độ {collected_info['dietary_habits']}")
                
                if conditions:
                    return f"{user_message}. {'. '.join(conditions)}"
            
            return f"{user_message}. Món ăn phổ biến, cân bằng dinh dưỡng, dễ chế biến"
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def filter_duplicate_recipes(self, recipes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Lọc các công thức trùng lặp bằng tên chuẩn hóa - XỬ LÝ TOÀN BỘ DANH SÁCH
        
        Args:
            recipes: Danh sách các công thức từ recipe_tool
            
        Returns:
            Danh sách công thức đã lọc trùng lặp
        """
        if not recipes or len(recipes) <= 1:
            return recipes
        
        # ⭐ LỌC TRÙNG LẶP BẰNG TÊN CHUẨN HÓA CHO TOÀN BỘ DANH SÁCH
        def normalize_recipe_name(name: str) -> str:
            """Chuẩn hóa tên recipe để so sánh trùng lặp"""
            if not name:
                return ""
            import unicodedata
            import re
            # Chuyển về lowercase, loại bỏ dấu cách, dấu gạch ngang, ký tự đặc biệt
            normalized = unicodedata.normalize('NFD', str(name).lower())
            normalized = ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')  # Loại bỏ dấu
            normalized = re.sub(r'[^a-z0-9]', '', normalized)  # Chỉ giữ chữ và số
            return normalized

        final_unique_recipes = []
        seen_normalized_names = set()
        
        for recipe_item in recipes:  # Xử lý TOÀN BỘ danh sách recipes
            if not isinstance(recipe_item, dict) or not recipe_item.get("name"):
                continue
                
            normalized_name = normalize_recipe_name(recipe_item["name"])
            if normalized_name and normalized_name not in seen_normalized_names:
                final_unique_recipes.append(recipe_item)
                seen_normalized_names.add(normalized_name)
            else:
                logger.debug(f"Đã lọc recipe trùng lặp: {recipe_item.get('name', 'Unknown')}")
        
        logger.info(f"Đã lọc từ {len(recipes)} xuống {len(final_unique_recipes)} recipes bằng tên chuẩn hóa.")
        return final_unique_recipes
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def create_product_search_prompt(self, medichat_response: str, recipes: List[Dict[str, Any]] = None, beverages: List[Dict[str, Any]] = None) -> str:
        """
        Tạo prompt cho product_find_tool dựa trên NGUYÊN LIỆU CỤ THỂ của các món ăn/đồ uống 
        ĐÃ ĐƯỢC CHỌN VÀ SẼ HIỂN THỊ CHO NGƯỜI DÙNG từ state['recipe_results'] và state['beverage_results'].
        
        Args:
            medichat_response: Phản hồi từ medichat (đã được polish và nhất quán với recipes/beverages)
            recipes: Danh sách recipes đã được lọc và chọn trong chat_flow.py
            beverages: Danh sách beverages đã được lọc và chọn trong chat_flow.py
            
        Returns:
            Query string tự nhiên để tìm sản phẩm/nguyên liệu cho các món đã được chọn
        """
        if not self.api_key:
            # Fallback cải thiện - ưu tiên recipes và beverages đã được chọn
            ingredients = []
            dish_names = []
            beverage_names = []
            
            # Lấy trực tiếp từ recipes đã được chọn (ưu tiên cao nhất)
            if recipes:
                for recipe in recipes:
                    recipe_name = recipe.get('name', '')
                    if recipe_name:
                        dish_names.append(recipe_name)
                    # Lấy nguyên liệu chi tiết từ ingredients_summary
                    if 'ingredients_summary' in recipe:
                        recipe_ingredients = [ing.strip() for ing in recipe['ingredients_summary'].split(',') if ing.strip()]
                        ingredients.extend(recipe_ingredients)
            
            # Lấy trực tiếp từ beverages đã được chọn
            if beverages:
                for beverage in beverages:
                    beverage_name = beverage.get('product_name', '')
                    if beverage_name:
                        beverage_names.append(beverage_name)
            
            # Tạo danh sách nguyên liệu duy nhất
            unique_ingredients = list(set(ingredients))[:15]
            all_items = dish_names + beverage_names
            
            if all_items and unique_ingredients:
                return f"Tôi cần mua nguyên liệu để làm {', '.join(all_items[:3])}, bao gồm: {', '.join(unique_ingredients)}."
            elif all_items:
                return f"Tôi cần mua nguyên liệu để làm {', '.join(all_items[:3])}."
            elif unique_ingredients:
                return f"Tôi cần mua các nguyên liệu sau: {', '.join(unique_ingredients)}."
            
            return "Tôi cần mua nguyên liệu để nấu ăn theo tư vấn dinh dưỡng."

        prompt = f"""Bạn là một KỸ SƯ AI CHUYÊN VỀ TRÍCH XUẤT NGUYÊN LIỆU cho hệ thống Chatbot Y tế. Nhiệm vụ của bạn là tạo ra một query mua sắm tự nhiên dựa trên NGUYÊN LIỆU CỤ THỂ của các món ăn/đồ uống ĐÃ ĐƯỢC CHỌN VÀ SẼ HIỂN THỊ CHO NGƯỜI DÙNG.

### NGUỒN DỮ LIỆU CHÍNH XÁC:

**PHẢN HỒI ĐÃ ĐƯỢC POLISH (nhất quán với recipes/beverages):**
```
{medichat_response}
```

**DANH SÁCH CÔNG THỨC ĐÃ ĐƯỢC CHỌN VÀ SẼ LƯU VÀO DATABASE:**
{json.dumps(recipes, ensure_ascii=False, indent=2) if recipes else "Không có công thức nào được chọn."}

**DANH SÁCH ĐỒ UỐNG ĐÃ ĐƯỢC CHỌN VÀ SẼ LƯU VÀO DATABASE:**
{json.dumps(beverages, ensure_ascii=False, indent=2) if beverages else "Không có đồ uống nào được chọn."}

### QUY TRÌNH TRÍCH XUẤT NGUYÊN LIỆU:

🎯 **NGUYÊN TẮC CHÍNH - SỬ DỤNG DỮ LIỆU ĐÃ ĐƯỢC CHỌN:**

Bạn cần tạo query mua sắm dựa trên CHÍNH XÁC các món ăn/đồ uống trong DANH SÁCH ĐÃ ĐƯỢC CHỌN ở trên. Đây là những món đã được hệ thống lọc và sẽ được lưu vào database.

**BƯỚC 1: TRÍCH XUẤT TÊN MÓN VÀ NGUYÊN LIỆU**
- Từ DANH SÁCH CÔNG THỨC: Lấy tên món từ trường "name" và nguyên liệu từ trường "ingredients_summary"
- Từ DANH SÁCH ĐỒ UỐNG: Lấy tên đồ uống từ trường "product_name"
- KHÔNG cần phân tích phản hồi Medichat để tìm món - chỉ cần dùng danh sách đã cho

**BƯỚC 2: XÂY DỰNG DANH SÁCH NGUYÊN LIỆU HOÀN CHỈNH**
- Từ mỗi recipe: Tách ingredients_summary thành danh sách nguyên liệu riêng biệt
- Từ mỗi beverage: Sử dụng tên sản phẩm làm nguyên liệu chính
- Loại bỏ trùng lặp và chuẩn hóa tên nguyên liệu

**BƯỚC 3: LÀM SẠCH VÀ CHUẨN HÓA NGUYÊN LIỆU**
- **Loại bỏ nguyên liệu quá chung chung:** "gia vị", "nước lọc", "dầu ăn" (trừ khi cụ thể như "dầu oliu", "muối hạt")
- **Chuẩn hóa tên gọi:** 
  + "Hành cây", "Hành lá" → "Hành lá"
  + "Thịt heo ba rọi", "Ba chỉ" → "Thịt ba chỉ" 
  + "Cà chua bi", "Cà chua" → "Cà chua"
- **Tạo danh sách duy nhất:** Loại bỏ trùng lặp, giữ tối đa 15-20 nguyên liệu quan trọng nhất

**BƯỚC 4: TẠO QUERY MUA SẮM TỰ NHIÊN**
Dựa trên danh sách món ăn/đồ uống và nguyên liệu đã trích xuất, tạo một YÊU CẦU MUA SẮM tự nhiên, ngắn gọn.

### VÍ DỤ HOÀN CHỈNH:

**Input:**
- Recipes: [{{"name": "Canh chua cá lóc", "ingredients_summary": "cá lóc, me cây, cà chua, dứa, đậu bắp, giá đỗ"}}]
- Beverages: [{{"product_name": "Nước ép dưa hấu"}}]

**Output mong đợi:**
"Tôi cần mua nguyên liệu để nấu Canh chua cá lóc và làm Nước ép dưa hấu, bao gồm: cá lóc, me cây, cà chua, dứa, đậu bắp, giá đỗ, dưa hấu."

### YÊU CẦU CUỐI CÙNG:
CHỈ TRẢ VỀ ĐOẠN VĂN BẢN YÊU CẦU MUA SẮM NGẮN GỌN (1-2 CÂU). KHÔNG TRẢ VỀ JSON, KHÔNG GIẢI THÍCH QUÁ TRÌNH, KHÔNG THÊM METADATA.

YÊU CẦU MUA SẮM:"""

        try:
            if GOOGLE_AI_AVAILABLE:
                try:
                    product_query = await self._query_gemini_with_client(prompt)
                except Exception as e:
                    logger.warning(f"Lỗi khi sử dụng Google client: {str(e)}. Chuyển sang HTTP API.")
                    product_query = await self._query_gemini_with_http(prompt)
            else:
                product_query = await self._query_gemini_with_http(prompt)
            
            # Làm sạch query - loại bỏ xuống dòng thừa và chuẩn hóa
            product_query = product_query.strip().replace('\n', ' ')
            
            # Loại bỏ các prefix thừa nếu Gemini thêm vào
            prefixes_to_remove = [
                "YÊU CẦU MUA SẮM:",
                "Đoạn văn bản:",
                "Kết quả:",
                "Output:",
                "Query mua sắm:",
                "Yêu cầu mua sắm:"
            ]
            
            for prefix in prefixes_to_remove:
                if product_query.startswith(prefix):
                    product_query = product_query[len(prefix):].strip()
            
            # Đảm bảo query không quá dài (giới hạn hợp lý cho product_find_tool)
            if len(product_query) > 300:
                # Cắt ngắn nhưng giữ ý nghĩa
                sentences = product_query.split('.')
                if len(sentences) > 1:
                    product_query = sentences[0] + '.'
                else:
                    product_query = product_query[:300] + '...'
            
            logger.info(f"Đã tạo FOCUSED product search query cho gợi ý cuối cùng ({len(product_query)} ký tự): {product_query}")
            return product_query
                
        except Exception as e:
            logger.error(f"Lỗi khi tạo product search prompt: {str(e)}")
            # Enhanced fallback tương tự như trong phần API key bị thiếu
            ingredients = []
            dish_names = []
            beverage_names = []
            
            # Trích xuất từ medichat_response trước (tập trung vào gợi ý cuối cùng)
            response_lower = medichat_response.lower()
            
            # Tìm kiếm tên món ăn cụ thể từ recipes trong medichat_response
            if recipes:
                for recipe in recipes[:3]:
                    recipe_name = recipe.get('name', '')
                    if recipe_name and recipe_name.lower() in response_lower:
                        dish_names.append(recipe_name)
                        # Lấy nguyên liệu chi tiết từ recipes
                        if 'ingredients_summary' in recipe:
                            ingredients.extend([ing.strip() for ing in recipe['ingredients_summary'].split(',')])
            
            # Tìm kiếm tên đồ uống từ beverages trong medichat_response
            if beverages:
                for beverage in beverages[:3]:
                    beverage_name = beverage.get('product_name', '')
                    if beverage_name and beverage_name.lower() in response_lower:
                        beverage_names.append(beverage_name)
            
            # Nếu không tìm thấy tên món cụ thể, lấy từ recipes/beverages làm fallback
            if not dish_names and not beverage_names:
                if recipes:
                    for recipe in recipes[:2]:
                        if 'name' in recipe:
                            dish_names.append(recipe['name'])
                        if 'ingredients_summary' in recipe:
                            ingredients.extend([ing.strip() for ing in recipe['ingredients_summary'].split(',')])
                
                if beverages:
                    for beverage in beverages[:2]:
                        if 'product_name' in beverage:
                            beverage_names.append(beverage['product_name'])
            
            unique_ingredients = list(set(ingredients))[:15]
            all_items = dish_names + beverage_names
            
            if all_items and unique_ingredients:
                return f"Tôi cần mua nguyên liệu để làm {', '.join(all_items[:3])}, bao gồm: {', '.join(unique_ingredients)}."
            elif all_items:
                return f"Tôi cần mua nguyên liệu để làm {', '.join(all_items[:3])}."
            elif unique_ingredients:
                return f"Tôi cần mua các nguyên liệu sau: {', '.join(unique_ingredients)}."
            
            # Fallback cuối cùng với thông tin từ medichat_response
            if "món" in response_lower or "nguyên liệu" in response_lower:
                return "Tôi cần mua các nguyên liệu chính từ các món ăn đã được gợi ý."
            
            return "Tôi cần mua nguyên liệu để nấu ăn theo tư vấn dinh dưỡng."
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def create_enhanced_medichat_prompt(self, messages: List[Dict[str, str]], recipes: List[Dict[str, Any]] = None, beverages: List[Dict[str, Any]] = None, suggest_general: bool = False, current_summary: Optional[str] = None) -> str:
        """
        Tạo prompt nâng cao cho Medichat với recipes và beverages (nếu có) và khả năng gợi ý chung
        
        Args:
            messages: Danh sách tin nhắn theo định dạng [{"role": "user", "content": "..."}]
            recipes: Danh sách recipes từ database (nếu có)
            beverages: Danh sách beverages từ database (nếu có)
            suggest_general: True nếu cần Medichat gợi ý theo tiêu chí chung
            current_summary: Tóm tắt cuộc trò chuyện từ lượt trước (nếu có)
            
        Returns:
            Prompt được tối ưu hóa cho Medichat
        """
        if not self.api_key or not messages:
            logger.error("Không thể tạo enhanced prompt: Thiếu API key hoặc không có tin nhắn")
            # Cải thiện fallback dựa trên suggest_general
            if suggest_general:
                return "Tôi muốn tìm một vài món ăn hoặc đồ uống giải nhiệt, phổ biến, cân bằng dinh dưỡng, dễ làm và ít gây dị ứng. Bạn có thể gợi ý được không?"
            else:
                return "Cần tư vấn dinh dưỡng và món ăn phù hợp."
        
        # Tạo prompt template với recipes, beverages, suggest_general và current_summary
        prompt_template = self._create_medichat_prompt_template(messages, recipes, beverages, suggest_general, current_summary)
        
        try:
            # Sử dụng thư viện Google hoặc HTTP API
            if GOOGLE_AI_AVAILABLE:
                try:
                    result_prompt = await self._query_gemini_with_client(prompt_template)
                except Exception as e:
                    logger.warning(f"Lỗi khi sử dụng Google client: {str(e)}. Chuyển sang HTTP API.")
                    result_prompt = await self._query_gemini_with_http(prompt_template)
            else:
                result_prompt = await self._query_gemini_with_http(prompt_template)
            
            # Logging chi tiết về độ dài prompt được tạo
            char_count = len(result_prompt)
            word_count_estimate = len(result_prompt.split())
            word_limit = self.max_medichat_prompt_words_with_context if (recipes or beverages or suggest_general) else 900
            
            logger.info(f"Đã tạo enhanced prompt: {char_count} ký tự, ~{word_count_estimate} từ (giới hạn: {word_limit} {'từ' if (recipes or beverages or suggest_general) else 'ký tự'})")
            logger.info(f"Prompt preview: {result_prompt[:100]}...")
            
            # Không cắt result_prompt theo ký tự nữa, tin tưởng Gemini tuân thủ giới hạn TỪ
            # Nếu Gemini thường xuyên vi phạm, chúng ta sẽ xem xét lại prompt cho Gemini
            
            return result_prompt
                
        except Exception as e:
            logger.error(f"Lỗi khi tạo enhanced prompt: {str(e)}")
            # Fallback được cải thiện dựa trên suggest_general
            if suggest_general:
                return "Tôi muốn tìm một vài món ăn hoặc đồ uống giải nhiệt, phổ biến, cân bằng dinh dưỡng, dễ làm và ít gây dị ứng. Bạn có thể gợi ý được không?"
            else:
                return "Cần tư vấn dinh dưỡng và món ăn phù hợp."
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def create_incremental_summary(
        self,
        previous_summary: Optional[str],
        new_user_message: str,
        new_assistant_message: str,
        full_chat_history_for_context: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Tạo bản tóm tắt tăng dần dựa trên tóm tắt trước đó và lượt tương tác mới.
        Sử dụng Gemini với vai trò Kỹ sư AI chuyên về Xử lý Ngôn ngữ Tự nhiên.

        Args:
            previous_summary: Bản tóm tắt của cuộc trò chuyện tính đến trước lượt tương tác này.
                              Có thể là None nếu đây là lần tóm tắt đầu tiên.
            new_user_message: Tin nhắn mới nhất của người dùng.
            new_assistant_message: Phản hồi mới nhất của trợ lý.
            full_chat_history_for_context: (Tùy chọn) 3-5 tin nhắn cuối cùng của lịch sử chat
                                           để cung cấp thêm ngữ cảnh cho Gemini nếu previous_summary quá cô đọng.

        Returns:
            Bản tóm tắt mới, bao gồm cả thông tin mới được tích hợp một cách thông minh.
        """
        if not self.api_key:
            logger.error("Không thể tạo tóm tắt tăng dần: Thiếu API key của Gemini.")
            # Fallback: nối chuỗi đơn giản nếu không có API key
            new_interaction = f"Người dùng: {new_user_message}\nTrợ lý: {new_assistant_message}\n"
            return f"{previous_summary}\n\n---\n\n{new_interaction}" if previous_summary else new_interaction

        # Xây dựng prompt chuyên nghiệp cho Gemini với vai trò Kỹ sư AI
        prompt_parts = [
            "Bạn là một KỸ SƯ AI CHUYÊN VỀ XỬ LÝ NGÔN NGỮ TỰ NHIÊN VÀ QUẢN LÝ NGỮ CẢNH HỘI THOẠI cho hệ thống Chatbot Y tế.",
            "Nhiệm vụ chuyên môn của bạn là tạo ra các bản tóm tắt 'cuộn' (incremental summary) để duy trì ngữ cảnh của toàn bộ cuộc trò chuyện một cách hiệu quả và thông minh.",
            "",
            "### PHÂN TÍCH NGUỒN DỮ LIỆU:"
        ]

        # Xử lý previous_summary
        if previous_summary:
            summary_word_count = len(previous_summary.split())
            prompt_parts.extend([
                "",
                "**BẢN TÓM TẮT CUỘC TRÒ CHUYỆN TÍNH ĐẾN THỜI ĐIỂM HIỆN TẠI:**",
                f"```text",
                f"{previous_summary}",
                f"```",
                f"(Độ dài hiện tại: ~{summary_word_count} từ)"
            ])
        else:
            prompt_parts.extend([
                "",
                "**BẢN TÓM TẮT TRƯỚC ĐÓ:** Không có (đây là lần tóm tắt đầu tiên)"
            ])

        # Thêm lượt tương tác mới
        user_preview = new_user_message[:150] + "..." if len(new_user_message) > 150 else new_user_message
        assistant_preview = new_assistant_message[:150] + "..." if len(new_assistant_message) > 150 else new_assistant_message
        
        prompt_parts.extend([
            "",
            "**LƯỢT TƯƠNG TÁC MỚI NHẤT CẦN TÍCH HỢP:**",
            f"Người dùng: {new_user_message}",
            f"Trợ lý: {new_assistant_message}"
        ])

        # Xử lý ngữ cảnh bổ sung nếu có
        if full_chat_history_for_context:
            context_messages = full_chat_history_for_context[-5:]  # Lấy tối đa 5 tin nhắn cuối
            context_text = ""
            for msg in context_messages:
                role_label = "Người dùng" if msg.get('role') == 'user' else "Trợ lý"
                content_preview = msg.get('content', '')[:200]  # Cắt ngắn 200 ký tự
                if len(msg.get('content', '')) > 200:
                    content_preview += "..."
                context_text += f"{role_label}: {content_preview}\n"
            
            if context_text.strip():
                prompt_parts.extend([
                    "",
                    "**NGỮ CẢNH BỔ SUNG TỪ VÀI LƯỢT TRAO ĐỔI GẦN ĐÂY:**",
                    f"```text",
                    f"{context_text.strip()}",
                    f"```",
                    "(Chỉ sử dụng nếu cần thiết để hiểu rõ hơn lượt tương tác mới)"
                ])

        # Hướng dẫn chuyên nghiệp cho Gemini
        summary_instructions = [
            "",
            "### NHIỆM VỤ CHUYÊN MÔN:",
            "",
            "Hãy cập nhật bản tóm tắt trên (hoặc tạo mới nếu chưa có) bằng cách tích hợp thông tin cốt lõi từ lượt tương tác mới nhất một cách THÔNG MINH và HIỆU QUẢ.",
            "",
            "**TIÊU CHÍ CHẤT LƯỢNG TÓM TẮT:**",
            "",
            "1. **Tính Súc Tích và Tập Trung:**",
            "   - Ngắn gọn, súc tích, tập trung vào các điểm chính, quyết định, thông tin quan trọng",
            "   - Ưu tiên thông tin sức khỏe, sở thích dinh dưỡng, mục tiêu của người dùng đã được xác nhận hoặc làm rõ",
            "   - Ghi nhận các món ăn, nguyên liệu, chế độ dinh dưỡng đã được thảo luận",
            "",
            "2. **Tính Mạch Lạc và Tự Nhiên:**",
            "   - Duy trì dòng chảy tự nhiên và logic của cuộc trò chuyện",
            "   - Sắp xếp thông tin theo thứ tự thời gian hoặc theo chủ đề một cách hợp lý",
            "",
            "3. **Tối Ưu Hóa Nội Dung:**",
            "   - Loại bỏ những chi tiết không cần thiết, lời chào hỏi lặp lại",
            "   - Tránh nhắc lại thông tin đã được tóm tắt đầy đủ ở `previous_summary` (trừ khi có thay đổi hoặc bổ sung ý nghĩa)",
            "   - Nếu lượt tương tác mới không thêm nhiều thông tin quan trọng, bản tóm tắt có thể không thay đổi nhiều",
            "",
            "4. **Quản Lý Độ Dài:**"
        ]

        # Thêm logic quản lý độ dài dựa trên previous_summary
        if previous_summary:
            current_word_count = len(previous_summary.split())
            if current_word_count > 700:
                summary_instructions.extend([
                    f"   - Previous_summary đã khá dài ({current_word_count} từ), hãy CÔ ĐỌNG NÓ một cách thông minh trước khi thêm thông tin mới",
                    "   - Đảm bảo bản tóm tắt cập nhật KHÔNG VƯỢT QUÁ 1000 từ",
                    "   - Ưu tiên giữ lại thông tin quan trọng nhất và mới nhất"
                ])
            else:
                summary_instructions.extend([
                    "   - Giữ bản tóm tắt ở mức độ hợp lý (tối đa khoảng 1000 từ)",
                    "   - Tích hợp thông tin mới một cách tự nhiên"
                ])
        else:
            summary_instructions.extend([
                "   - Tạo bản tóm tắt đầu tiên súc tích và đầy đủ",
                "   - Tập trung vào những thông tin cốt lõi từ lượt tương tác đầu tiên"
            ])

        summary_instructions.extend([
            "",
            "5. **Tính Chuyên Nghiệp:**",
            "   - Sử dụng ngôn ngữ chuyên nghiệp, rõ ràng, phù hợp với ngữ cảnh y tế/dinh dưỡng",
            "   - Duy trì tông giọng trung tính, khách quan",
            "",
            "### YÊU CẦU ĐẦU RA:",
            "",
            "CHỈ TRẢ VỀ NỘI DUNG BẢN TÓM TẮT MỚI ĐÃ ĐƯỢC CẬP NHẬT.",
            "KHÔNG GIẢI THÍCH QUÁ TRÌNH, KHÔNG TIÊU ĐỀ, KHÔNG METADATA, KHÔNG ĐỊNH DẠNG ĐẶC BIỆT.",
            "",
            "BẢN TÓM TẮT CẬP NHẬT:"
        ])

        # Kết hợp tất cả các phần
        prompt_parts.extend(summary_instructions)
        full_prompt = "\n".join(prompt_parts)
        
        try:
            # Gọi API Gemini với logging chi tiết
            prompt_char_count = len(full_prompt)
            logger.info(f"Tạo tóm tắt tăng dần - Prompt: {prompt_char_count} ký tự, Previous summary: {len(previous_summary) if previous_summary else 0} ký tự")
            
            if GOOGLE_AI_AVAILABLE:
                try:
                    updated_summary = await self._query_gemini_with_client(full_prompt)
                except Exception as e:
                    logger.warning(f"Lỗi khi sử dụng Google client cho tóm tắt: {str(e)}. Chuyển sang HTTP API.")
                    updated_summary = await self._query_gemini_with_http(full_prompt)
            else:
                updated_summary = await self._query_gemini_with_http(full_prompt)
            
            # Làm sạch kết quả
            updated_summary = updated_summary.strip()
            
            # Loại bỏ các prefix thừa nếu Gemini thêm vào
            prefixes_to_remove = [
                "BẢN TÓM TẮT CẬP NHẬT:",
                "Bản tóm tắt mới:",
                "Tóm tắt cập nhật:",
                "Kết quả:",
                "**Bản tóm tắt cập nhật:**"
            ]
            
            for prefix in prefixes_to_remove:
                if updated_summary.startswith(prefix):
                    updated_summary = updated_summary[len(prefix):].strip()
            
            # Logging kết quả
            final_word_count = len(updated_summary.split())
            logger.info(f"Đã tạo tóm tắt tăng dần: {final_word_count} từ, {len(updated_summary)} ký tự")
            logger.info(f"Preview tóm tắt: {updated_summary[:100]}...")
            
            return updated_summary
                
        except Exception as e:
            logger.error(f"Lỗi khi tạo tóm tắt tăng dần: {str(e)}", exc_info=True)
            # Fallback an toàn: nối chuỗi với format cải thiện
            logger.warning("Sử dụng fallback cho tóm tắt tăng dần")
            new_interaction_text = f"📝 Lượt tương tác mới:\n• Người dùng: {new_user_message}\n• Trợ lý: {new_assistant_message}"
            
            if previous_summary:
                return f"{previous_summary}\n\n---\n\n{new_interaction_text}"
            else:
                return new_interaction_text