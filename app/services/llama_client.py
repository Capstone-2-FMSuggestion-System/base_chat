import httpx
from typing import List, Dict, Any, AsyncGenerator
import json
import logging
from app.config import settings


logger = logging.getLogger(__name__)


class LlamaClient:
    def __init__(self, base_url: str = None):
        self.base_url = base_url or settings.LLAMA_CPP_URL
        self.chat_endpoint = f"{self.base_url}/v1/chat/completions"
        
    async def generate_response(self, messages: List[Dict[str, str]]) -> AsyncGenerator[str, None]:
        """
        Gửi yêu cầu đến API llama.cpp và trả về phản hồi dạng streaming
        
        Args:
            messages: Danh sách tin nhắn theo định dạng [{"role": "user", "content": "..."}]
        
        Yields:
            Từng phần nội dung phản hồi từ AI
        """
        try:
            # Kiểm tra xem có tin nhắn nào không
            if not messages:
                logger.error("Danh sách tin nhắn trống")
                yield "Xin lỗi, không tìm thấy tin nhắn nào để xử lý. Vui lòng thử lại."
                return
                
            # Kiểm tra tin nhắn của người dùng gần nhất
            last_user_msg = next((msg for msg in reversed(messages) if msg.get("role") == "user"), None)
            if not last_user_msg:
                logger.error("Không tìm thấy tin nhắn người dùng nào trong cuộc trò chuyện")
                yield "Xin lỗi, không tìm thấy câu hỏi của bạn. Vui lòng thử lại."
                return
                
            # Log nội dung tin nhắn để debug
            logger.info(f"Tin nhắn người dùng gần nhất: {last_user_msg}")
                
            # Thêm system message nếu chưa có
            if not any(msg.get("role") == "system" for msg in messages):
                system_message = {
                    "role": "system", 
                    "content": """You are an AI Medical Assistant with medical knowledge. Your mission is to provide general health information based on scientific evidence.

LIMITATIONS:
- DO NOT diagnose diseases or conclude medical conditions
- DO NOT prescribe medications or suggest specific dosages
- DO NOT give potentially dangerous advice
- DO NOT claim to be a doctor or medical professional
- DO NOT list too many prescription drugs

RESPONSE GUIDELINES:
1. Use simple, understandable and friendly language
2. Structure your answers as follows:
   - Brief introduction (1-2 sentences)
   - 3-5 main points in bullet form
   - End with advice to consult a doctor if necessary
3. Prioritize safe self-care measures before mentioning medications
4. For unclear questions, ask 1-2 clarifying questions before answering

ALWAYS respond in Vietnamese unless otherwise requested.
"""
                }
                messages = [system_message] + messages
            
            logger.debug(f"Sending request to llama.cpp: {json.dumps(messages)}")
            logger.info("Bắt đầu gửi yêu cầu đến API llama.cpp...")
            
            # Bộ lọc token đặc biệt
            special_tokens = ["<|im_ending|>", "<|im_start|>"]
            token_count = 0
            max_tokens = settings.LLM_MAX_TOKENS
            
            # Các mẫu thông báo lỗi cần phát hiện
            error_prefixes = [
                "Xin lỗi, tôi đang có vấn đề",
                "Xin lỗi, hiện tôi không thể kết nối",
                "Xin lỗi, tôi gặp khó khăn"
            ]
            
            # Biến để theo dõi nếu đã nhận thấy thông báo lỗi
            detected_error = False
            error_content = ""
            medical_content = ""
            is_collecting_error = False
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    url=self.chat_endpoint,
                    json={
                        "model": "QuantFactory/Medichat-Llama3-8B-GGUF",
                        "messages": messages,
                        "temperature": 0.7,
                        "max_tokens": max_tokens,
                        "stream": True
                    },
                    timeout=60.0
                )
                
                logger.info(f"Đã nhận phản hồi với status code: {response.status_code}")
                
                if response.status_code != 200:
                    error_msg = f"Lỗi từ API llama.cpp: {response.status_code}"
                    logger.error(error_msg)
                    yield "Xin lỗi, tôi đang gặp khó khăn trong việc xử lý yêu cầu của bạn. Vui lòng thử lại sau."
                    return
                
                logger.info("Đang xử lý phản hồi streaming...")
                collected_response = ""  # Theo dõi toàn bộ phản hồi
                
                async for line in response.aiter_lines():
                    if not line.strip():  # Bỏ qua dòng trống
                        continue
                    
                    logger.debug(f"Dòng phản hồi: {line}")
                    
                    if line.startswith("data: "):
                        # Xử lý đặc biệt cho thông điệp [DONE]
                        if line.strip() == "data: [DONE]":
                            logger.info("Nhận được thông điệp kết thúc: [DONE]")
                            continue
                        
                        try:
                            data = line[6:]  # Bỏ "data: " phía trước
                            
                            # Kiểm tra nếu đây là phản hồi không phải JSON
                            if any(token in data for token in special_tokens):
                                # Lọc bỏ các token đặc biệt
                                for token in special_tokens:
                                    data = data.replace(token, "")
                                if data.strip():  # Nếu còn nội dung sau khi lọc
                                    collected_response += data  # Cập nhật phản hồi đã thu thập
                                    
                                    # Kiểm tra nếu đây là phần đầu của thông báo lỗi
                                    if not is_collecting_error and not detected_error:
                                        for prefix in error_prefixes:
                                            if prefix in collected_response:
                                                is_collecting_error = True
                                                logger.warning(f"Phát hiện thông báo lỗi: {prefix}")
                                                break
                                    
                                    # Nếu đang thu thập lỗi và tìm thấy dấu chấm, kết thúc thu thập
                                    if is_collecting_error and "." in data:
                                        end_pos = data.find(".") + 1
                                        error_content += data[:end_pos]
                                        is_collecting_error = False
                                        detected_error = True
                                        logger.warning(f"Đã thu thập thông báo lỗi: {error_content}")
                                    elif is_collecting_error:
                                        error_content += data
                                    elif detected_error:
                                        # Đã phát hiện lỗi, nhưng còn chứa nội dung y tế
                                        medical_content += data
                                    else:
                                        # Phản hồi bình thường
                                        token_count += len(data.split())
                                        if token_count <= max_tokens:
                                            yield data
                                continue
                            
                            # Xử lý JSON thông thường
                            json_data = json.loads(data)
                            if "choices" in json_data and len(json_data["choices"]) > 0:
                                choice = json_data["choices"][0]
                                
                                # Kiểm tra xem có phải là phần cuối không
                                if choice.get("finish_reason") is not None:
                                    logger.info(f"Kết thúc phản hồi với lý do: {choice.get('finish_reason')}")
                                    continue
                                
                                # Lấy nội dung từ delta
                                if "delta" in choice and "content" in choice["delta"]:
                                    chunk = choice["delta"]["content"]
                                    if chunk:  # Chỉ yield khi có nội dung
                                        # Lọc bỏ các token đặc biệt
                                        for token in special_tokens:
                                            chunk = chunk.replace(token, "")
                                        
                                        collected_response += chunk  # Cập nhật phản hồi đã thu thập
                                        
                                        # Kiểm tra nếu đây là phần đầu của thông báo lỗi
                                        if not is_collecting_error and not detected_error:
                                            for prefix in error_prefixes:
                                                if prefix in collected_response and not detected_error:
                                                    is_collecting_error = True
                                                    logger.warning(f"Phát hiện thông báo lỗi: {prefix}")
                                                    break
                                        
                                        # Nếu đang thu thập lỗi và tìm thấy dấu chấm, kết thúc thu thập
                                        if is_collecting_error and "." in chunk:
                                            end_pos = chunk.find(".") + 1
                                            error_content += chunk[:end_pos]
                                            is_collecting_error = False
                                            detected_error = True
                                            logger.warning(f"Đã thu thập thông báo lỗi: {error_content}")
                                            # Phần còn lại của chunk thuộc về medical_content
                                            if end_pos < len(chunk):
                                                medical_content += chunk[end_pos:]
                                        elif is_collecting_error:
                                            error_content += chunk
                                        elif detected_error:
                                            # Đã phát hiện lỗi, nhưng còn chứa nội dung y tế
                                            medical_content += chunk
                                        else:
                                            # Phản hồi bình thường
                                            if chunk.strip():
                                                token_count += len(chunk.split())
                                                if token_count <= max_tokens:
                                        logger.debug(f"Đang trả về chunk: {chunk}")
                                        yield chunk
                                                else:
                                                    logger.info(f"Đã đạt giới hạn {max_tokens} token, dừng phản hồi")
                                                    break
                        except json.JSONDecodeError as e:
                            # Xử lý các line không phải JSON (có thể là text trực tiếp)
                            logger.warning(f"Không thể parse JSON, xử lý như plain text: {line}")
                            
                            # Lọc bỏ các token đặc biệt
                            content = line[6:]  # Bỏ "data: " phía trước
                            for token in special_tokens:
                                content = content.replace(token, "")
                            
                            if content.strip():
                                token_count += len(content.split())
                                if token_count <= max_tokens:
                                    yield content
                                else:
                                    logger.info(f"Đã đạt giới hạn {max_tokens} token, dừng phản hồi")
                                    break
                        except (KeyError, IndexError) as e:
                            logger.error(f"Lỗi cấu trúc JSON: {str(e)}, line: {line}")
                            continue
                
                logger.info("Đã hoàn thành xử lý phản hồi streaming")
                
                # Nếu phát hiện thông báo lỗi nhưng đã bỏ qua, trả về thông báo lỗi
                if detected_error and not error_content.strip():
                    logger.warning("Phát hiện lỗi không chứa nội dung phù hợp, sử dụng thông báo mặc định")
                    yield "Xin lỗi, tôi đang gặp khó khăn trong việc xử lý yêu cầu của bạn. Vui lòng thử lại sau."
                elif detected_error:
                    # Nếu chưa trả về thông báo lỗi, trả về ngay bây giờ
                    if token_count == 0:
                        logger.info(f"Trả về thông báo lỗi: {error_content}")
                        yield error_content
                
        except httpx.TimeoutException:
            logger.error("Timeout khi kết nối đến llama.cpp")
            yield "Xin lỗi, tôi cần nhiều thời gian hơn để xử lý yêu cầu của bạn. Vui lòng thử lại với câu hỏi ngắn gọn hơn."
        
        except Exception as e:
            logger.error(f"Lỗi kết nối đến llama.cpp: {str(e)}")
            yield "Xin lỗi, hiện tôi không thể kết nối tới hệ thống trí tuệ nhân tạo. Vui lòng thử lại sau."
    
    async def get_full_response(self, messages: List[Dict[str, str]]) -> str:
        """
        Gửi yêu cầu đến API llama.cpp và trả về phản hồi đầy đủ
        
        Args:
            messages: Danh sách tin nhắn theo định dạng [{"role": "user", "content": "..."}]
            
        Returns:
            Phản hồi đầy đủ từ AI
        """
        result = ""
        async for chunk in self.generate_response(messages):
            result += chunk
        return result
