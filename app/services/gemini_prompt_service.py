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
        self.max_prompt_length_with_recipes = 400  # Giới hạn cho prompt có recipes (từ)
        
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
        prompt = f"""Bạn là một chuyên viên phân tích y tế thông minh và tinh tế. Nhiệm vụ của bạn là phân tích CUỘC TRÒ CHUYỆN dưới đây để hiểu rõ ý định của người dùng, trích xuất thông tin và xác định các bước tiếp theo.

PHẠM VI HỖ TRỢ CỦA TRỢ LÝ: 
Tư vấn dinh dưỡng, sức khỏe, gợi ý món ăn và đồ uống phù hợp với tình trạng sức khỏe người dùng (nếu được cung cấp). Trợ lý có thể:
- Tư vấn món ăn, đồ uống tốt cho sức khỏe
- Gợi ý công thức nấu ăn phù hợp với tình trạng bệnh lý
- Tư vấn dinh dưỡng cho từng đối tượng (trẻ em, người cao tuổi, người bệnh)
- Tư vấn thực phẩm nên tránh với các bệnh lý cụ thể
- Gợi ý nguyên liệu và cách chế biến món ăn
- Tư vấn chế độ ăn uống khoa học

LỊCH SỬ CHAT GẦN ĐÂY:
{history_text}

TIN NHẮN NGƯỜI DÙNG MỚI NHẤT:
{user_message}

YÊU CẦU PHÂN TÍCH CHI TIẾT:

1. Xác định Phạm vi Yêu cầu:
   - is_valid_scope: (boolean) Yêu cầu có nằm trong PHẠM VI HỖ TRỢ không? Chỉ đặt false nếu yêu cầu hoàn toàn không liên quan đến dinh dưỡng, sức khỏe, món ăn, đồ uống.
   - is_food_related: (boolean) Yêu cầu có cụ thể về món ăn, đồ uống, công thức, nguyên liệu hoặc tư vấn dinh dưỡng không? (Cờ tổng quan cho cả ẩm thực nói chung)
   - requests_food: (boolean) Người dùng có cụ thể hỏi về món ăn, công thức nấu ăn, thực đơn món ăn không?
   - requests_beverage: (boolean) Người dùng có cụ thể hỏi về đồ uống, nước uống, công thức pha chế, trà, cà phê, nước ép, sinh tố không?

HƯỚNG DẪN CHI TIẾT CHO VIỆC ĐẶT CÁC CỜ:
- Nếu người dùng hỏi 'món ăn cho người tiểu đường', 'công thức phở bò', 'thực đơn bữa tối', 'cách nấu canh chua', đặt requests_food = true và requests_beverage = false (trừ khi họ cũng hỏi đồ uống).
- Nếu người dùng hỏi 'nước ép tốt cho da', 'cách pha trà gừng', 'đồ uống giải nhiệt', 'nước detox', 'sinh tố dinh dưỡng', đặt requests_beverage = true và requests_food = false (trừ khi họ cũng hỏi món ăn).
- Nếu người dùng hỏi 'gợi ý món ăn và đồ uống cho bữa tiệc', 'thực đơn đầy đủ cho ngày hôm nay', đặt cả requests_food = true và requests_beverage = true.
- Nếu người dùng hỏi chung chung 'tôi nên ăn uống gì hôm nay?' mà không rõ ràng món ăn hay đồ uống, dựa vào ngữ cảnh trước đó. Nếu không có ngữ cảnh rõ ràng, có thể đặt cả hai là false và dựa vào suggest_general_options hoặc follow_up_question.
- Cờ is_food_related sẽ là true nếu requests_food hoặc requests_beverage là true, hoặc nếu yêu cầu liên quan đến tư vấn dinh dưỡng nói chung.

2. Trích xuất Thông tin Sức khỏe và Yêu cầu:
   - collected_info: (object) {{
       "health_condition": "string (ví dụ: 'tiểu đường type 2', 'cao huyết áp', 'bệnh tim', để trống nếu không có)",
       "medical_history": "string (ví dụ: 'từng phẫu thuật dạ dày', 'có tiền sử dị ứng', để trống nếu không có)",
       "allergies": "string (ví dụ: 'hải sản', 'đậu phộng', 'gluten', để trống nếu không có)",
       "dietary_habits": "string (ví dụ: 'ăn chay', 'thích đồ ngọt', 'ăn ít muối', 'không uống sữa', để trống nếu không có)",
       "food_preferences": "string (ví dụ: 'thích ăn cá', 'thích vị ngọt', 'cần món nước', 'muốn món dễ làm', 'cần món nhanh gọn', 'thích món truyền thống', để trống nếu không có)",
       "food_dislikes": "string (ví dụ: 'không ăn được hành', 'ghét sầu riêng', 'không thích đồ chua', để trống nếu không có)",
       "health_goals": "string (ví dụ: 'giảm cân', 'kiểm soát đường huyết', 'hạ nhiệt', 'tăng cường miễn dịch', để trống nếu không có)"
     }}

3. Đánh giá Thái độ Từ chối và Gợi ý Chung:
   - user_rejected_info: (boolean) Người dùng có đang TỪ CHỐI RÕ RÀNG HOẶC NGẦM cung cấp thêm thông tin không? 
     Các ví dụ từ chối bao gồm:
     + Rõ ràng: "tôi không muốn nói", "tôi không thể cung cấp thông tin này", "tôi từ chối trả lời"
     + Ngầm: "tôi không biết nữa", "bạn cứ gợi ý đi", "cho tôi vài ví dụ", "tôi không rõ", "bạn chọn giúp tôi", "tùy bạn", "gì cũng được"
     
   - suggest_general_options: (boolean) Đặt TRUE khi:
     + is_valid_scope là true VÀ 
     + is_food_related là true VÀ
     + (user_rejected_info là true HOẶC thông tin trong collected_info + user_message quá ít để đưa ra gợi ý cá nhân hóa) VÀ
     + KHÔNG CÓ đủ thông tin cụ thể từ người dùng về tình trạng sức khỏe/sở thích cá nhân
     Khi TRUE: Trợ lý sẽ gợi ý dựa trên tiêu chí chung (phổ biến, đa dạng, cân bằng dinh dưỡng, ít gây dị ứng, dễ chế biến)

4. Đánh giá Nhu cầu Thông tin Bổ sung:
   - need_more_info: (boolean)
     + **QUY TẮC QUAN TRỌNG: Nếu user_rejected_info là true, thì need_more_info PHẢI LÀ FALSE.**
     + **QUY TẮC QUAN TRỌNG: Nếu suggest_general_options là true, thì need_more_info PHẢI LÀ FALSE.**
     + Nếu cả hai điều kiện trên là false, và thông tin trong collected_info + user_message QUÁ ÍT để đưa ra bất kỳ gợi ý nào (kể cả gợi ý chung), thì đặt là true.
     
   - follow_up_question: (string | null)
     + **QUAN TRỌNG: Chỉ tạo khi need_more_info là true VÀ user_rejected_info là false VÀ suggest_general_options là false**
     + Nếu cần tạo: Tạo câu hỏi NGẮN GỌN, LỊCH SỰ, CỤ THỂ và TRÁNH HỎI LẠI câu hỏi tương tự đã hỏi trước đó
     + Nếu người dùng không biết chọn gì, đưa ra 2-3 LỰA CHỌN CỤ THỂ để họ chọn
     + Ví dụ tốt: "Để gợi ý phù hợp, bạn có muốn thử: 1) Đồ uống giải khát (nước ép, trà thảo mộc), 2) Món ăn nhẹ (chè, bánh), hay 3) Món ăn chính (cơm, phở) không ạ?"
     + Nếu không cần hỏi thêm, trường này PHẢI là null

HÃY TRẢ VỀ KẾT QUẢ DƯỚI DẠNG MỘT ĐỐI TƯỢNG JSON DUY NHẤT, TUÂN THỦ NGHIÊM NGẶT CẤU TRÚC SAU. KHÔNG THÊM BẤT KỲ GIẢI THÍCH HAY VĂN BẢN NÀO BÊN NGOÀI CẤU TRÚC JSON:

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
}}"""
        
        return prompt
    
    def _create_medichat_prompt_template(self, messages: List[Dict[str, str]], recipes: List[Dict[str, Any]] = None, beverages: List[Dict[str, Any]] = None, suggest_general: bool = False) -> str:
        """
        Tạo template prompt để tóm tắt thông tin cho Medichat.
        Nếu có recipes hoặc beverages, đưa hết vào và giới hạn prompt tổng là 400 TỪ.
        Nếu suggest_general là true, yêu cầu Medichat gợi ý chung.
        
        Args:
            messages: Danh sách tin nhắn
            recipes: Danh sách công thức món ăn (nếu có)
            beverages: Danh sách đồ uống (nếu có)
            suggest_general: True nếu cần Medichat gợi ý theo tiêu chí chung.
            
        Returns:
            Prompt cho Gemini để tạo prompt Medichat
        """
        # Xác định giới hạn từ dựa trên có recipes/beverages hay không hoặc suggest_general
        word_limit = self.max_prompt_length_with_recipes if (recipes or beverages or suggest_general) else 900
        
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
        
        # Tạo phần recipes nếu có
        recipe_section = ""
        if recipes:
            recipe_section = "\n\nCÔNG THỨC MÓN ĂN CÓ SẴN TRONG DATABASE:\n"
            for i, recipe in enumerate(recipes, 1):  # Đưa toàn bộ recipes vào
                recipe_id = recipe.get('id', f'R{i}')
                name = recipe.get('name', 'N/A')
                ingredients = recipe.get('ingredients_summary', 'N/A')
                url = recipe.get('url', '')
                
                recipe_section += f"{i}. [ID: {recipe_id}] {name}\n   - Nguyên liệu: {ingredients}\n"
                if url:
                    recipe_section += f"   - Link: {url}\n"
        
        # Tạo phần beverages nếu có
        beverage_section = ""
        if beverages:
            beverage_section = "\n\nĐỒ UỐNG CÓ SẴN TRONG DATABASE:\n"
            for i, bev in enumerate(beverages, 1):
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
                                "🎯 YÊU CẦU ƯU TIÊN SỬ DỤNG DỮ LIỆU TỪ DATABASE:\n" \
                                "Nếu có danh sách món ăn (recipe_section) hoặc đồ uống (beverage_section) được cung cấp, " \
                                "hãy YÊU CẦU MEDICHAT ƯU TIÊN xem xét và lựa chọn từ danh sách này trước khi gợi ý các món/đồ uống khác, " \
                                "miễn là chúng phù hợp với các tiêu chí gợi ý chung (phổ biến, đa dạng, cân bằng, ít dị ứng, dễ làm). " \
                                "Prompt cho Medichat cần nhấn mạnh việc sử dụng dữ liệu có sẵn này làm ưu tiên số 1."
        
        # Tạo prompt cho Gemini
        prompt = f""""Bạn là một trợ lý y tế thông minh, chuyên tóm tắt thông tin từ cuộc trò chuyện để tạo ra một prompt ngắn gọn, súc tích và đầy đủ thông tin nhất cho mô hình AI y tế chuyên sâu Medichat-LLaMA3-8B.

TOÀN BỘ CUỘC TRÒ CHUYỆN ĐỂ TÓM TẮT:
{conversation_text}

{recipe_section}{beverage_section}{general_instruction}

YÊU CẦU TẠO PROMPT CHO MEDICHAT:
1. Nội dung cốt lõi:
   - Nếu `general_instruction` có nội dung (suggest_general=true): Tạo prompt yêu cầu Medichat thực hiện gợi ý chung theo các tiêu chí đã nêu. Có thể tham khảo `recipe_section` nếu có món phù hợp với tiêu chí chung.
   - Nếu không có `general_instruction`: Tập trung vào yêu cầu chính/vấn đề mà người dùng đang hỏi, bao gồm triệu chứng/tình trạng sức khỏe, bệnh lý nền/dị ứng, thông tin về món ăn/chế độ dinh dưỡng quan tâm, mục tiêu dinh dưỡng/sức khỏe, và thói quen ăn uống đã đề cập.

2. Định dạng Prompt:
- Viết bằng NGÔI THỨ NHẤT, như thể người dùng đang trực tiếp đặt câu hỏi cho Medichat.
- Prompt phải là một YÊU CẦU RÕ RÀNG, dễ hiểu.
- Ví dụ cấu trúc (linh hoạt điều chỉnh tùy theo ngữ cảnh):
+ Nếu hỏi món ăn: "Tôi bị [tình trạng sức khỏe ví dụ: tiểu đường, dị ứng hải sản], muốn [mục tiêu ví dụ: kiểm soát đường huyết]. Xin gợi ý [số lượng] món [loại món ví dụ: canh, xào] phù hợp, [yêu cầu thêm ví dụ: ít gia vị, dễ làm]."
+ Nếu hỏi tư vấn chung: "Tôi bị [tình trạng sức khỏe], đang theo [thói quen ăn uống]. Tôi nên điều chỉnh chế độ ăn uống như thế nào để [mục tiêu sức khỏe]?"
+ Nếu gợi ý chung: "Tôi cần gợi ý món ăn/đồ uống [dựa trên tiêu chí từ general_instruction]. Xin đưa ra 2-3 lựa chọn cụ thể."

3. XỬ LÝ CÔNG THỨC MÓN ĂN/ĐỒ UỐNG:
- Khi suggest_general=True VÀ có recipe_section hoặc beverage_section:
  + Hướng dẫn Medichat xem xét kỹ các món ăn trong recipe_section và đồ uống trong beverage_section
  + Yêu cầu Medichat CHỌN LỌC và gợi ý 2-3 items từ danh sách này nếu chúng đáp ứng các tiêu chí chung (phổ biến, cân bằng dinh dưỡng, ít dị ứng, dễ làm)
  + Nếu không có đủ lựa chọn phù hợp từ danh sách, Medichat có thể bổ sung bằng kiến thức của mình
  + VÍ DỤ PROMPT CHO MEDICHAT: "Tôi muốn vài gợi ý đồ uống giải nhiệt, ngọt ngào, phổ biến và dễ làm. Bạn có thể xem xét danh sách đồ uống sau đây (nếu có) và chọn ra 2-3 loại phù hợp nhất không: [Nước ép A (ID: B1), Trà B (ID: B2), Sinh tố C (ID: B3)]? Nếu không có gì phù hợp, xin hãy gợi ý các loại khác."

- Khi KHÔNG phải suggest_general=True (người dùng có yêu cầu cụ thể) VÀ có recipe_section hoặc beverage_section:
  + Tạo prompt hướng dẫn Medichat ƯU TIÊN SỬ DỤNG các món ăn từ recipe_section và/hoặc đồ uống từ beverage_section nếu chúng phù hợp với yêu cầu CỤ THỂ của người dùng (về tình trạng sức khỏe, sở thích đã được collected_info ghi nhận)
  + Yêu cầu Medichat giải thích tại sao chúng phù hợp. Nếu cần, Medichat có thể điều chỉnh (ví dụ: giảm gia vị) hoặc gợi ý món/đồ uống khác nếu danh sách cung cấp không có gì phù hợp
  + VÍ DỤ PROMPT CHO MEDICHAT: "Tôi bị tiểu đường và muốn một món canh ít đường. Trong danh sách món ăn này: [Canh X (ID: R1), Canh Y (ID: R2)], món nào phù hợp hơn cho tôi? Xin giải thích. Hoặc bạn có gợi ý nào khác không?"

- Khi có cả món ăn và đồ uống từ database:
  + Tạo prompt yêu cầu Medichat đưa ra gợi ý kết hợp từ recipe_section cho món ăn và từ beverage_section cho đồ uống, đảm bảo sự hài hòa và phù hợp với yêu cầu/tình trạng sức khỏe

4. Giới hạn:
- TOÀN BỘ prompt kết quả CHO MEDICHAT PHẢI DƯỚI {word_limit} TỪ.
- Cần cực kỳ súc tích và đúng trọng tâm. CHỈ bao gồm thông tin đã được đề cập trong cuộc trò chuyện. KHÔNG suy diễn, KHÔNG thêm thông tin không có.

5. Mục tiêu: Tạo ra prompt hiệu quả nhất để Medichat có thể đưa ra câu trả lời y tế chính xác và hữu ích

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
- Ngôn ngữ có DỄ HIỂU, THÂN THIỆN, và PHÙ HỢP với người dùng không?
- Có chứa thông tin thừa, metadata, hoặc các cụm từ không tự nhiên (ví dụ: "dưới đây là...", "đánh giá của tôi...") không?
2. Hành động:
- Nếu phản hồi thô đã tốt (chính xác, đầy đủ, dễ hiểu): Hãy loại bỏ TOÀN BỘ metadata, các cụm từ đánh giá, định dạng thừa. Giữ lại phần nội dung cốt lõi và đảm bảo nó mạch lạc, tự nhiên.
- Nếu phản hồi thô chưa tốt (lạc đề, không đầy đủ, khó hiểu, chứa thông tin sai lệch, hoặc quá máy móc): Hãy VIẾT LẠI HOÀN TOÀN một phản hồi mới dựa trên PROMPT GỐC. Phản hồi mới phải chính xác, đầy đủ, thân thiện, dễ hiểu, và cung cấp giá trị thực sự cho người dùng.
3. YÊU CẦU TUYỆT ĐỐI CHO ĐẦU RA CUỐI CÙNG:
- Đầu ra của bạn sẽ được gửi TRỰC TIẾP cho người dùng.
- KHÔNG BAO GIỜ bao gồm các từ/cụm từ như: "Đánh giá:", "Kiểm tra:", "Điều chỉnh:", "Phản hồi đã được điều chỉnh:", "Phân tích phản hồi:", "HỢP LỆ", "Dưới đây là...", "Theo tôi...", v.v.
- KHÔNG BAO GIỜ chia phản hồi thành các phần có tiêu đề kiểu "1. Đánh giá", "2. Điều chỉnh".
- KHÔNG BAO GIỜ nhắc đến quá trình đánh giá hay sửa đổi nội bộ.
- LUÔN viết như thể bạn đang trực tiếp trò chuyện và tư vấn cho người dùng.
- LUÔN sử dụng tiếng Việt tự nhiên, thân thiện, chuyên nghiệp và mạch lạc.
- LUÔN đảm bảo thông tin y tế/dinh dưỡng là chính xác và hữu ích.
- Đảm bảo phản hồi ngắn gọn, súc tích nhất có thể mà vẫn đủ ý.
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
            prompt = f"""Bạn là chuyên gia dinh dưỡng và ẩm thực. Nhiệm vụ của bạn là tạo ra một câu truy vấn tối ưu để tìm kiếm công thức món ăn phù hợp.

YÊU CẦU CỦA NGƯỜI DÙNG:
"{user_message}"

THÔNG TIN SỨC KHỎE ĐÃ THU THẬP:
{json.dumps(collected_info, ensure_ascii=False, indent=2) if collected_info else "Không có thông tin cụ thể"}

NHIỆM VỤ:
Tạo một câu truy vấn ngắn gọn, súc tích (tối đa 150 từ) để tìm kiếm công thức món ăn phù hợp nhất.

QUY TẮC TẠO QUERY:
1. **Nếu có thông tin sức khỏe cụ thể**: Kết hợp yêu cầu người dùng với các điều kiện sức khỏe
2. **Nếu không có thông tin**: Sử dụng tiêu chí mặc định:
   - Tính phổ biến và đa dạng
   - Cân bằng dinh dưỡng
   - Không gây dị ứng phổ biến
   - Dễ chế biến

CẤU TRÚC QUERY MONG MUỐN:
"[Loại món ăn/yêu cầu chính] + [điều kiện sức khỏe nếu có] + [ưu tiên dinh dưỡng] + [ưu tiên chế biến]"

Ví dụ:
- "Món canh dinh dưỡng cho người tiểu đường, ít đường, nhiều chất xơ, dễ nấu"
- "Món ăn sáng healthy, cân bằng dinh dưỡng, không gây dị ứng, nhanh chóng"

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
        Lọc các công thức trùng lặp bằng Gemini AI
        
        Args:
            recipes: Danh sách các công thức từ recipe_tool
            
        Returns:
            Danh sách công thức đã lọc trùng lặp
        """
        if not self.api_key or not recipes or len(recipes) <= 1:
            return recipes
        
        # Giới hạn số lượng recipes để tránh prompt quá dài
        limited_recipes = recipes[:20]
        
        # Tạo danh sách recipes cho prompt
        recipe_list = []
        for i, recipe in enumerate(limited_recipes, 1):
            name = recipe.get('name', 'N/A')
            ingredients = recipe.get('ingredients_summary', 'N/A')
            recipe_list.append(f"{i}. {name} - Nguyên liệu: {ingredients}")
        
        recipes_text = '\n'.join(recipe_list)
        
        prompt = f"""Bạn là chuyên gia ẩm thực. Nhiệm vụ của bạn là lọc các công thức món ăn trùng lặp từ danh sách dưới đây.

DANH SÁCH CÔNG THỨC:
{recipes_text}

YÊU CẦU:
1. Xác định các món ăn có tên giống nhau hoặc rất tương tự
2. Với mỗi nhóm món trùng lặp, chỉ giữ lại món đầu tiên (số thứ tự nhỏ nhất)
3. Trả về danh sách số thứ tự của các món cần GIỮ LẠI

QUY TẮC XÁC ĐỊNH TRÙNG LẶP:
- Tên hoàn toàn giống nhau: "Canh chua" và "Canh chua"
- Tên rất tương tự: "Canh chua cá" và "Canh chua cá lóc"
- Các biến thể của cùng món: "Phở bò" và "Phở bò tái"

TRẢ VỀ DƯỚI DẠNG JSON:
{{"selected_indices": [1, 3, 5, ...]}}

CHỈ TRẢ VỀ JSON, KHÔNG CÓ GIẢI THÍCH:"""

        try:
            if GOOGLE_AI_AVAILABLE:
                try:
                    response = await self._query_gemini_with_client(prompt)
                except Exception as e:
                    logger.warning(f"Lỗi khi sử dụng Google client: {str(e)}. Chuyển sang HTTP API.")
                    response = await self._query_gemini_with_http(prompt)
            else:
                response = await self._query_gemini_with_http(prompt)
            
            # Parse JSON response
            try:
                clean_response = response.strip()
                if "```json" in clean_response:
                    clean_response = clean_response.split("```json")[1].split("```")[0].strip()
                elif "```" in clean_response:
                    clean_response = clean_response.split("```")[1].split("```")[0].strip()
                
                result = json.loads(clean_response)
                selected_indices = result.get("selected_indices", [])
                
                # Lọc recipes dựa trên indices đã chọn
                filtered_recipes = []
                for i, recipe in enumerate(limited_recipes, 1):
                    if i in selected_indices:
                        filtered_recipes.append(recipe)
                
                logger.info(f"Đã lọc từ {len(limited_recipes)} xuống {len(filtered_recipes)} recipes")
                return filtered_recipes
                
            except json.JSONDecodeError as e:
                logger.error(f"Không thể parse JSON từ Gemini filter response: {e}")
                return limited_recipes
                
        except Exception as e:
            logger.error(f"Lỗi khi filter duplicate recipes: {str(e)}")
            return limited_recipes
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def create_product_search_prompt(self, medichat_response: str, recipes: List[Dict[str, Any]] = None, beverages: List[Dict[str, Any]] = None) -> str:
        """
        Tạo prompt cho product_find_tool từ phản hồi medichat, recipes và beverages.
        Gemini sẽ đóng vai trò Kỹ sư AI Trích xuất Thông tin Chính xác để tập trung vào 
        gợi ý cuối cùng của Medichat và phân biệt rõ món ăn/đồ uống.
        
        Args:
            medichat_response: Phản hồi từ medichat
            recipes: Danh sách recipes mà Medichat có thể đã tham khảo (nếu có)
            beverages: Danh sách beverages mà Medichat có thể đã tham khảo (nếu có)
            
        Returns:
            Query string tự nhiên để tìm sản phẩm/nguyên liệu
        """
        if not self.api_key:
            # Fallback được cải thiện - tập trung vào phân tích medichat_response trước
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
            
            # Fallback từ medichat_response với phân tích đơn giản
            if "món" in response_lower or "nguyên liệu" in response_lower:
                return "Tôi cần mua các nguyên liệu chính từ các món ăn đã được gợi ý."
            
            return "Tôi cần mua nguyên liệu để nấu ăn theo tư vấn dinh dưỡng."

        prompt = f"""Bạn là một KỸ SƯ AI CHUYÊN VỀ TRÍCH XUẤT THÔNG TIN CHÍNH XÁC cho hệ thống Chatbot Y tế. Nhiệm vụ cụ thể của bạn là phân tích phản hồi tư vấn y tế để trích xuất thông tin mua sắm nguyên liệu một cách CHÍNH XÁC và HIỆU QUẢ.

### ĐÁNH GIÁ NGUỒN DỮ LIỆU:

**PHẢN HỒI TƯ VẤN TỪ MEDICHAT:**
```
{medichat_response}
```

**DANH SÁCH CÔNG THỨC MÀ MEDICHAT CÓ THỂ ĐÃ THAM KHẢO:**
{json.dumps(recipes, ensure_ascii=False, indent=2) if recipes else "Không có danh sách công thức tham khảo kèm theo."}

**DANH SÁCH ĐỒ UỐNG MÀ MEDICHAT CÓ THỂ ĐÃ THAM KHẢO:**
{json.dumps(beverages, ensure_ascii=False, indent=2) if beverages else "Không có danh sách đồ uống tham khảo kèm theo."}

### QUY TRÌNH TRÍCH XUẤT CHUYÊN NGHIỆP:

🎯 **QUAN TRỌNG - TAPPING VÀO GỢI Ý CUỐI CÙNG:**
Chỉ trích xuất nguyên liệu cho những món ăn và đồ uống mà Medichat thực sự GỢI Ý CHO NGƯỜI DÙNG trong phần KẾT LUẬN hoặc PHẦN GỢI Ý CHÍNH của phản hồi. Bỏ qua các nguyên liệu được nhắc đến trong quá trình phân tích hoặc so sánh nếu chúng không phải là gợi ý cuối cùng.

**BƯỚC 1: XÁC ĐỊNH MÓN ĂN/ĐỒ UỐNG ĐƯỢC GỢI Ý CUỐI CÙNG**
- Đọc kỹ phần KẾT LUẬN/GỢI Ý CHÍNH của Medichat (thường ở cuối phản hồi hoặc có từ khóa như "gợi ý", "nên thử", "có thể làm")
- Phân biệt rõ ràng: Nếu Medichat gợi ý cả món ăn và đồ uống, hãy cố gắng tách biệt (nếu có thể) trong suy nghĩ của bạn, nhưng danh sách combined_unique_ingredients cuối cùng vẫn là tổng hợp
- Giới hạn tối đa 3-4 món được gợi ý thực sự để tránh phân tán

**BƯỚC 2: THAM CHIẾU RECIPES VÀ BEVERAGES MỘT CÁCH CẨN THẬN**
- Nếu Medichat đề cập đến một món ăn cụ thể có ID trong recipes đã cung cấp, hãy ưu tiên lấy danh sách nguyên liệu chi tiết từ recipes đó cho món ăn đó
- Tương tự với beverages: Nếu Medichat gợi ý đồ uống có trong danh sách beverages, tham chiếu đến thông tin chi tiết
- Nếu Medichat chỉ gợi ý tên chung (ví dụ: "nước ép cam") mà không có ID cụ thể, trích xuất nguyên liệu cơ bản từ kiến thức thông thường

**BƯỚC 3: TRÍCH XUẤT VÀ PHÂN LOẠI NGUYÊN LIỆU**
- Từ phản hồi Medichat: Thu thập nguyên liệu được đề cập trực tiếp trong phần gợi ý
- Từ `recipes` (nếu Medichat tham chiếu): Lấy nguyên liệu từ các món ăn được Medichat GỢI Ý
- Từ `beverages` (nếu Medichat tham chiếu): Lấy thành phần chính từ các đồ uống được Medichat GỢI Ý
- Phân biệt: food ingredients vs beverage ingredients trong quá trình tư duy nhưng kết hợp trong kết quả cuối

**BƯỚC 4: LÀM SẠCH VÀ CHUẨN HÓA NGUYÊN LIỆU**
- **Loại bỏ nguyên liệu quá chung chung:** "gia vị", "nước lọc", "dầu ăn" (trừ khi cụ thể như "dầu oliu", "muối hạt")
- **Chuẩn hóa tên gọi:** 
  + "Hành cây", "Hành lá" → "Hành lá"
  + "Thịt heo ba rọi", "Ba chỉ" → "Thịt ba chỉ" 
  + "Cà chua bi", "Cà chua" → "Cà chua"
- **Tạo danh sách duy nhất:** Loại bỏ trùng lặp, giữ tối đa 15-20 nguyên liệu quan trọng nhất

### CẤU TRÚC JSON TRUNG GIAN MONG MUỐN:

Trước khi tạo query string, hãy tạo một JSON để tổ chức thông tin:

```json
{{
  "suggested_items": [
    {{"item_name": "Canh chua cá lóc", "type": "food", "ingredients": ["cá lóc", "me", "cà chua", "dứa"]}},
    {{"item_name": "Nước ép dưa hấu", "type": "beverage", "ingredients": ["dưa hấu", "đường (tùy chọn)"]}}
  ],
  "combined_unique_ingredients_for_shopping": ["cá lóc", "me", "cà chua", "dứa", "dưa hấu", "đường"]
}}
```

**BƯỚC 5: TẠO QUERY MUA SẮM TỰ NHIÊN**
Dựa trên combined_unique_ingredients_for_shopping và suggested_items, hãy tạo một YÊU CẦU MUA SẮM tự nhiên, ngắn gọn.

### VÍ DỤ HOÀN CHỈNH:

**Input:**
- Medichat: "Tôi gợi ý bạn làm canh chua cá lóc và uống nước ép dưa hấu. Canh chua giúp giải nhiệt với cá lóc, me, cà chua. Dưa hấu rất tốt để bù nước."
- Recipes: [{{"name": "Canh chua cá lóc", "ingredients_summary": "cá lóc, me cây, cà chua, dứa, đậu bắp, giá đỗ"}}]

**Output mong đợi:**
"Tôi cần mua nguyên liệu để nấu Canh chua cá lóc và làm Nước ép dưa hấu, bao gồm: cá lóc, me cây, cà chua, dứa, đậu bắp, giá đỗ, dưa hấu."

### YÊU CẦU CUỐI CÙNG:
CHỈ TRẢ VỀ ĐOẠN VĂN BẢN YÊU CẦU MUA SẮM NGẮN GỌN (1-2 CÂU). KHÔNG TRẢ VỀ JSON TRUNG GIAN, KHÔNG GIẢI THÍCH QUÁ TRÌNH, KHÔNG THÊM METADATA.

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
            
            logger.info(f"Đã tạo enhanced product search query ({len(product_query)} ký tự): {product_query}")
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
    async def create_enhanced_medichat_prompt(self, messages: List[Dict[str, str]], recipes: List[Dict[str, Any]] = None, beverages: List[Dict[str, Any]] = None, suggest_general: bool = False) -> str:
        """
        Tạo prompt nâng cao cho Medichat với recipes và beverages (nếu có) và khả năng gợi ý chung
        
        Args:
            messages: Danh sách tin nhắn theo định dạng [{"role": "user", "content": "..."}]
            recipes: Danh sách recipes từ database (nếu có)
            beverages: Danh sách beverages từ database (nếu có)
            suggest_general: True nếu cần Medichat gợi ý theo tiêu chí chung
            
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
        
        # Tạo prompt template với recipes, beverages và suggest_general
        prompt_template = self._create_medichat_prompt_template(messages, recipes, beverages, suggest_general)
        
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
            word_limit = self.max_prompt_length_with_recipes if (recipes or beverages or suggest_general) else 900
            
            logger.info(f"Đã tạo enhanced prompt: {char_count} ký tự, ~{word_count_estimate} từ (giới hạn: {word_limit} {'từ' if (recipes or suggest_general) else 'ký tự'})")
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