import logging
from typing import Dict, List, Any, Optional, TypedDict, Annotated, Callable
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel, Field
import json
from datetime import datetime

from app.config import settings
from app.services.gemini_prompt_service import GeminiPromptService
from app.repositories.chat_repository import ChatRepository
from app.services.llm_service_factory import LLMServiceFactory
from app.services.summary_service import SummaryService
from app.db.models import Message, HealthData
from app.db.database import redis_client

logger = logging.getLogger(__name__)

# Định nghĩa trạng thái 
class ChatState(TypedDict):
    conversation_id: int
    user_id: int
    user_message: str
    messages: List[Dict[str, str]]
    is_valid_scope: bool
    need_more_info: bool
    collected_info: Dict[str, Any]
    medichat_prompt: Optional[str]
    medichat_response: Optional[str]
    final_response: Optional[str]
    follow_up_question: Optional[str]
    error: Optional[str]

# Các node xử lý
async def check_scope_node(state: ChatState) -> ChatState:
    """Kiểm tra xem nội dung tin nhắn có thuộc phạm vi hỗ trợ không"""
    logger.info(f"Đang kiểm tra phạm vi nội dung: {state['user_message'][:50]}...")
    
    try:
        # Tạo dịch vụ Gemini
        gemini_service = GeminiPromptService()
        
        # Phân tích nội dung từ Gemini
        analysis = await gemini_service.analyze_query(state['user_message'], state['messages'])
        
        # Cập nhật trạng thái
        state['is_valid_scope'] = analysis.get('is_valid_scope', True)
        state['need_more_info'] = analysis.get('need_more_info', False)
        state['follow_up_question'] = analysis.get('follow_up_question')
        
        # Xử lý trường hợp chào hỏi
        if "chào" in state['user_message'].lower() or "hello" in state['user_message'].lower():
            # Trường hợp chào hỏi, tự trả lời luôn
            state['need_more_info'] = False
            if state['follow_up_question'] and ("chào" in state['follow_up_question'].lower() or 
                                               "hello" in state['follow_up_question'].lower()):
                state['final_response'] = state['follow_up_question']
        
        # Lưu thông tin đã thu thập được
        collected_info = state.get('collected_info', {})
        new_info = analysis.get('collected_info', {})
        collected_info.update({k: v for k, v in new_info.items() if v})
        state['collected_info'] = collected_info
        
        logger.info(f"Kết quả kiểm tra phạm vi: valid={state['is_valid_scope']}, need_more={state['need_more_info']}")
        
        # Lưu thông tin sức khỏe vào Redis nếu có và thuộc phạm vi hợp lệ
        if state['is_valid_scope'] and collected_info:
            await save_health_data_to_cache(
                state['conversation_id'], 
                state['user_id'],
                collected_info
            )
    except Exception as e:
        logger.error(f"Lỗi khi kiểm tra phạm vi: {str(e)}")
        state['error'] = f"Lỗi khi kiểm tra phạm vi: {str(e)}"
        state['is_valid_scope'] = True  # Fallback cho phép tiếp tục
        state['need_more_info'] = False
    
    return state

async def collect_info_node(state: ChatState) -> ChatState:
    """Thu thập thêm thông tin từ người dùng"""
    logger.info("Đang chuẩn bị thu thập thêm thông tin...")
    
    # Nếu cần thêm thông tin, chuẩn bị câu hỏi tiếp theo
    if state['need_more_info'] and state['follow_up_question']:
        logger.info(f"Cần thu thập thêm thông tin: {state['follow_up_question']}")
        state['final_response'] = state['follow_up_question']
    else:
        # Nếu đã đủ thông tin, chỉ cập nhật trạng thái
        state['need_more_info'] = False
    
    return state

async def store_data_node(state: ChatState, repository) -> ChatState:
    """Lưu dữ liệu vào cơ sở dữ liệu"""
    logger.info("Đang lưu dữ liệu vào cơ sở dữ liệu...")
    
    try:
        if state['is_valid_scope'] and not state['need_more_info']:
            # Lưu thông điệp người dùng
            repository.add_message(state['conversation_id'], "user", state['user_message'])
            
            # Lưu thông tin sức khỏe vào cơ sở dữ liệu MySQL
            if state['collected_info']:
                await save_health_data_to_db(
                    repository,
                    state['conversation_id'],
                    state['user_id'],
                    state['collected_info']
                )
                
            logger.info(f"Đã lưu dữ liệu cho conversation_id={state['conversation_id']}")
    except Exception as e:
        logger.error(f"Lỗi khi lưu dữ liệu: {str(e)}")
        state['error'] = f"Lỗi khi lưu dữ liệu: {str(e)}"
    
    return state

async def medichat_call_node(state: ChatState, repository, llm_service) -> ChatState:
    """Gọi đến Medichat để lấy phản hồi"""
    logger.info("Đang chuẩn bị gọi đến Medichat...")
    
    # Kiểm tra nếu đã có lỗi hoặc không phải phạm vi hợp lệ thì bỏ qua
    if not state['is_valid_scope'] or state['need_more_info'] or state['error']:
        return state
    
    try:
        # Tạo dịch vụ Gemini
        gemini_service = GeminiPromptService()
        
        # Lấy lịch sử trò chuyện với tóm tắt nếu cần
        messages = repository.get_messages_with_summary(state['conversation_id'])
        
        # Tạo prompt tối ưu cho Medichat bằng Gemini
        medichat_prompt = await gemini_service.create_medichat_prompt(messages)
        state['medichat_prompt'] = medichat_prompt
        
        # Khởi tạo LLM service nếu cần
        if llm_service._active_service is None:
            await llm_service.initialize()
        
        # Tạo danh sách tin nhắn mới với prompt từ Gemini
        medichat_messages = [
            {
                "role": "system", 
                "content": settings.MEDICHAT_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": medichat_prompt
            }
        ]
        
        # Gọi đến Medichat để lấy phản hồi
        logger.info(f"Gửi prompt đến Medichat: {medichat_prompt}")
        medichat_response = await llm_service.get_full_response(medichat_messages)
        state['medichat_response'] = medichat_response
        
        logger.info(f"Đã nhận phản hồi từ Medichat: {medichat_response[:50]}...")
    except Exception as e:
        logger.error(f"Lỗi khi gọi đến Medichat: {str(e)}")
        state['error'] = f"Lỗi khi gọi đến Medichat: {str(e)}"
    
    return state

async def response_cleanup_node(state: ChatState, repository) -> ChatState:
    """Kiểm tra và tinh chỉnh phản hồi trước khi trả về"""
    logger.info("Đang kiểm tra và tinh chỉnh phản hồi...")
    
    # Xử lý các trường hợp
    try:
        if not state['is_valid_scope']:
            # Nếu không thuộc phạm vi hợp lệ, trả về thông báo từ chối
            state['final_response'] = "Xin lỗi, câu hỏi của bạn nằm ngoài phạm vi tư vấn dinh dưỡng và sức khỏe. Tôi chỉ có thể hỗ trợ về các vấn đề liên quan đến sức khỏe, dinh dưỡng, và món ăn phù hợp. Bạn có thể đặt câu hỏi khác không?"
        
        elif state['need_more_info'] and state['follow_up_question']:
            # Nếu cần thêm thông tin, trả về câu hỏi tiếp theo
            # (đã được gán trong collect_info_node)
            pass
            
        elif state['error']:
            # Nếu có lỗi, trả về thông báo lỗi
            state['final_response'] = "Xin lỗi, hiện tôi không thể kết nối tới hệ thống trí tuệ nhân tạo. Vui lòng thử lại sau."
            
        elif state['medichat_response']:
            # Nếu có phản hồi từ Medichat, kiểm tra và tinh chỉnh
            gemini_service = GeminiPromptService()
            if state['medichat_prompt']:
                polished_response = await gemini_service.polish_response(
                    state['medichat_response'], 
                    state['medichat_prompt']
                )
                state['final_response'] = polished_response
                
                # Lưu phản hồi đã tinh chỉnh
                repository.add_message(state['conversation_id'], "assistant", polished_response)
            else:
                state['final_response'] = state['medichat_response']
        else:
            # Nếu không có phản hồi, trả về thông báo lỗi
            state['final_response'] = "Xin lỗi, tôi không thể xử lý yêu cầu của bạn. Vui lòng thử lại."
                
        logger.info(f"Phản hồi cuối cùng: {state['final_response'][:50]}...")
    except Exception as e:
        logger.error(f"Lỗi khi tinh chỉnh phản hồi: {str(e)}")
        state['final_response'] = "Xin lỗi, có lỗi xảy ra khi xử lý phản hồi. Vui lòng thử lại sau."
    
    return state

# Utility functions để lưu dữ liệu
async def save_health_data_to_cache(conversation_id: int, user_id: int, data: Dict[str, Any]) -> None:
    """Lưu thông tin sức khỏe vào Redis cache"""
    try:
        # Tạo key cho Redis
        cache_key = f"session:{conversation_id}:health_info"
        
        # Chuyển đổi dữ liệu thành JSON và lưu vào Redis
        cache_data = {
            "user_id": user_id,
            "timestamp": datetime.now().isoformat(),
            "data": data
        }
        
        redis_client.set(
            cache_key, 
            json.dumps(cache_data),
            ex=86400  # Hết hạn sau 24 giờ
        )
        
        logger.info(f"Đã lưu thông tin sức khỏe vào cache: {cache_key}")
    except Exception as e:
        logger.error(f"Lỗi khi lưu thông tin sức khỏe vào cache: {str(e)}")

async def save_health_data_to_db(repo, conversation_id: int, user_id: int, data: Dict[str, Any]) -> None:
    """Lưu thông tin sức khỏe vào cơ sở dữ liệu MySQL"""
    try:
        # Gọi đến repository để lưu vào DB
        repo.save_health_data(
            conversation_id=conversation_id,
            user_id=user_id,
            health_condition=data.get("health_condition", ""),
            medical_history=data.get("medical_history", ""),
            allergies=data.get("allergies", ""),
            dietary_habits=data.get("dietary_habits", ""),
            health_goals=data.get("health_goals", "")
        )
        
        logger.info(f"Đã lưu thông tin sức khỏe vào DB cho conversation_id={conversation_id}")
    except Exception as e:
        logger.error(f"Lỗi khi lưu thông tin sức khỏe vào DB: {str(e)}")

# Chức năng chạy quy trình xử lý chat
async def run_chat_flow(
    user_message: str,
    user_id: int,
    conversation_id: int,
    messages: List[Dict[str, str]],
    repository = None,
    llm_service = None
) -> Dict[str, Any]:
    """Chạy luồng xử lý chat và trả về kết quả"""
    try:
        # Khởi tạo trạng thái
        state = ChatState(
            conversation_id=conversation_id,
            user_id=user_id,
            user_message=user_message,
            messages=messages,
            is_valid_scope=True,  # Giả định mặc định là hợp lệ
            need_more_info=False,
            collected_info={},
            medichat_prompt=None,
            medichat_response=None,
            final_response=None,
            follow_up_question=None,
            error=None
        )
        
        # Xử lý luồng thủ công thay vì dùng LangGraph
        state = await check_scope_node(state)
        
        # Nếu là chào hỏi đơn giản và đã có final_response, thì chỉ cần lưu tin nhắn
        if state.get("final_response") and (
            "chào" in user_message.lower() or "hello" in user_message.lower()):
            # Lưu tin nhắn người dùng
            repository.add_message(state['conversation_id'], "user", state['user_message'])
            # Lưu tin nhắn phản hồi
            repository.add_message(state['conversation_id'], "assistant", state['final_response'])
            # Thêm assistant_message vào state
            state["assistant_message"] = {
                "role": "assistant",
                "content": state.get("final_response")
            }
            # Trả về kết quả
            return state
        
        # Xử lý các trường hợp khác
        if not state["is_valid_scope"] or state["error"]:
            state = await response_cleanup_node(state, repository)
        elif state["need_more_info"] and state["follow_up_question"]:
            state = await collect_info_node(state)
            state = await response_cleanup_node(state, repository)
        else:
            state = await store_data_node(state, repository)
            state = await medichat_call_node(state, repository, llm_service)
            state = await response_cleanup_node(state, repository)
        
        # Đảm bảo luôn có final_response
        if not state.get("final_response"):
            if state.get("error"):
                state["final_response"] = "Xin lỗi, có lỗi xảy ra trong quá trình xử lý. Vui lòng thử lại sau."
            elif not state.get("is_valid_scope", True):
                state["final_response"] = "Xin lỗi, câu hỏi của bạn nằm ngoài phạm vi tư vấn dinh dưỡng và sức khỏe. Tôi chỉ có thể hỗ trợ về các vấn đề liên quan đến sức khỏe, dinh dưỡng, và món ăn phù hợp."
            else:
                state["final_response"] = "Xin lỗi, tôi không thể xử lý yêu cầu của bạn. Vui lòng thử lại."
        
        # Đảm bảo có assistant_message cho API trả về
        if not state.get("assistant_message"):
            state["assistant_message"] = {
                "role": "assistant",
                "content": state.get("final_response", "")
            }
            
        # Log kết quả
        logger.info(f"Kết quả từ xử lý: valid={state.get('is_valid_scope')}, need_more={state.get('need_more_info')}")
        
        return state
    except Exception as e:
        # Xử lý ngoại lệ
        logger.error(f"Lỗi khi xử lý luồng chat: {str(e)}")
        return {
            "conversation_id": conversation_id,
            "user_message": {"role": "user", "content": user_message},
            "assistant_message": {"role": "assistant", "content": "Xin lỗi, hệ thống gặp lỗi khi xử lý yêu cầu của bạn."},
            "is_valid_scope": True,
            "need_more_info": False,
            "error": str(e),
            "final_response": "Xin lỗi, hệ thống gặp lỗi khi xử lý yêu cầu của bạn."
        } 