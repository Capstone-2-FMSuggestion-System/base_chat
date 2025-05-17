import logging
from typing import Dict, List, Any, Optional, TypedDict, Annotated, Callable, Literal, Union
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel, Field
import json
from datetime import datetime
import asyncio
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

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
    is_greeting: bool  # Thêm trạng thái để đánh dấu tin nhắn chào hỏi
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
        
        # Kiểm tra xem có phải tin nhắn chào hỏi không - vẫn giữ xử lý cơ bản này
        greeting_words = ["chào", "hello", "hi", "xin chào", "hey", "good morning", "good afternoon", "good evening"]
        is_greeting = any(word in state['user_message'].lower() for word in greeting_words)
        state['is_greeting'] = is_greeting
        
        if is_greeting:
            # Xử lý tin nhắn chào hỏi đặc biệt
            state['is_valid_scope'] = True
            state['need_more_info'] = False
            greeting_response = await gemini_service.get_greeting_response(state['user_message'])
            state['final_response'] = greeting_response
            logger.info(f"Đã xác định là tin nhắn chào hỏi, phản hồi: {greeting_response[:50]}...")
            return state
            
        # Đảm bảo messages có tin nhắn hiện tại
        current_messages = state['messages'].copy()
        
        # Thêm tin nhắn hiện tại nếu chưa có
        current_user_message_exists = False
        for msg in current_messages:
            if msg["role"] == "user" and msg["content"] == state['user_message']:
                current_user_message_exists = True
                break
                
        if not current_user_message_exists:
            current_messages.append({"role": "user", "content": state['user_message']})
        
        # Phân tích nội dung từ Gemini dựa vào TOÀN BỘ lịch sử trò chuyện
        analysis = await gemini_service.analyze_query(state['user_message'], current_messages)
        
        # Cập nhật trạng thái từ phân tích của Gemini
        state['is_valid_scope'] = analysis.get('is_valid_scope', True)
        state['need_more_info'] = analysis.get('need_more_info', False)
        state['follow_up_question'] = analysis.get('follow_up_question')
        
        # Lấy thông tin sức khỏe từ phân tích của Gemini - không tự xử lý từ khóa nữa
        collected_info = analysis.get('collected_info', {})
        
        # Lọc bỏ các giá trị rỗng từ collected_info
        collected_info = {k: v for k, v in collected_info.items() if v}
        state['collected_info'] = collected_info
        
        # Kiểm tra từ chối cung cấp thông tin
        rejection_phrases = [
            "không thể cung cấp", 
            "không muốn chia sẻ", 
            "không thể đề cập", 
            "không nói", 
            "không tiết lộ",
            "chỉ muốn",
            "không biết",
            "không dùng"
        ]
        
        # Kiểm tra xem người dùng có từ chối cung cấp thông tin không
        user_rejected_info = any(phrase in state['user_message'].lower() for phrase in rejection_phrases)
        
        # Nếu người dùng từ chối cung cấp thông tin nhưng đã có thông tin cơ bản, không hỏi thêm
        if user_rejected_info and collected_info:
            state['need_more_info'] = False
        
        logger.info(f"Kết quả kiểm tra phạm vi: valid={state['is_valid_scope']}, need_more={state['need_more_info']}")
        
        # Lưu thông tin sức khỏe vào Redis nếu có và thuộc phạm vi hợp lệ
        if state['is_valid_scope'] and collected_info:
            await save_health_data_to_cache(
                state['conversation_id'], 
                state['user_id'],
                collected_info
            )
            logger.info(f"Đã lưu thông tin sức khỏe vào cache: session:{state['conversation_id']}:health_info")
    except Exception as e:
        logger.error(f"Lỗi khi kiểm tra phạm vi: {str(e)}")
        state['error'] = f"Lỗi khi kiểm tra phạm vi: {str(e)}"
        state['is_valid_scope'] = True  # Fallback cho phép tiếp tục
        state['need_more_info'] = False
    
    return state

def collect_info_node(state: ChatState) -> ChatState:
    """Thu thập thêm thông tin từ người dùng"""
    logger.info("Đang chuẩn bị thu thập thêm thông tin...")
    
    # Nếu là tin nhắn chào hỏi thì bỏ qua bước này
    if state['is_greeting']:
        return state
    
    # Nếu cần thêm thông tin, chuẩn bị câu hỏi tiếp theo
    if state['need_more_info'] and state['follow_up_question']:
        logger.info(f"Cần thu thập thêm thông tin: {state['follow_up_question']}")
        state['final_response'] = state['follow_up_question']
    else:
        # Nếu đã đủ thông tin, chỉ cập nhật trạng thái
        state['need_more_info'] = False
    
    return state

# Wrapper cho các hàm bất đồng bộ
def run_async(async_func):
    async def wrapped(*args, **kwargs):
        return await async_func(*args, **kwargs)
    
    def wrapper(*args, **kwargs):
        return asyncio.run(wrapped(*args, **kwargs))
    
    return wrapper

def store_data_node_wrapper(state: ChatState, repository) -> ChatState:
    """Wrapper đồng bộ cho hàm store_data_node bất đồng bộ"""
    async def _async_store_data():
        result_state = state.copy()
        try:
            # Tìm tin nhắn người dùng đã tồn tại trong database thay vì tạo mới
            from sqlalchemy import desc
            from app.db.models import Message
            
            db = repository.db
            existing_message = db.query(Message).filter(
                Message.conversation_id == result_state['conversation_id'],
                Message.role == "user",
                Message.content == result_state['user_message']
            ).order_by(desc(Message.created_at)).first()
            
            # Nếu không tìm thấy tin nhắn, mới thêm vào database
            if not existing_message:
                # LUÔN lưu tin nhắn người dùng vào cơ sở dữ liệu, bất kể phạm vi
                repository.add_message(result_state['conversation_id'], "user", result_state['user_message'])
                logger.info(f"Đã lưu tin nhắn người dùng: {result_state['user_message'][:50]}...")
            else:
                logger.info(f"Đã tìm thấy tin nhắn người dùng hiện có trong database, ID={existing_message.message_id}")
            
            # Kiểm tra thông tin sức khỏe từ phân tích của Gemini
            collected_info = result_state.get('collected_info', {})
            
            # Lọc ra các thông tin có giá trị (không None/rỗng)
            valid_health_info = {}
            if collected_info:
                if collected_info.get('health_condition'):
                    valid_health_info['health_condition'] = collected_info.get('health_condition')
                    logger.info(f"Phát hiện tình trạng sức khỏe: {collected_info.get('health_condition')}")
                
                if collected_info.get('medical_history'):
                    valid_health_info['medical_history'] = collected_info.get('medical_history')
                    logger.info(f"Phát hiện bệnh lý: {collected_info.get('medical_history')}")
                
                if collected_info.get('allergies'):
                    valid_health_info['allergies'] = collected_info.get('allergies')
                    logger.info(f"Phát hiện dị ứng: {collected_info.get('allergies')}")
                
                if collected_info.get('dietary_habits'):
                    valid_health_info['dietary_habits'] = collected_info.get('dietary_habits')
                    logger.info(f"Phát hiện thói quen ăn uống: {collected_info.get('dietary_habits')}")
                
                if collected_info.get('health_goals'):
                    valid_health_info['health_goals'] = collected_info.get('health_goals')
                    logger.info(f"Phát hiện mục tiêu sức khỏe: {collected_info.get('health_goals')}")
            
            # Lưu thông tin vào cơ sở dữ liệu nếu có thông tin hợp lệ
            # Ngay cả khi không thuộc phạm vi hỗ trợ (is_valid_scope = False), vẫn lưu thông tin
            # nếu Gemini đã phát hiện được
            if valid_health_info:
                try:
                    await save_health_data_to_db(
                        repository,
                        result_state['conversation_id'],
                        result_state['user_id'],
                        valid_health_info
                    )
                    logger.info(f"Đã lưu {len(valid_health_info)} thông tin sức khỏe vào DB cho conversation_id={result_state['conversation_id']}")
                except Exception as e:
                    logger.error(f"Lỗi khi lưu thông tin sức khỏe vào DB: {str(e)}")
        except Exception as e:
            logger.error(f"Lỗi khi lưu dữ liệu: {str(e)}")
            result_state['error'] = f"Lỗi khi lưu dữ liệu: {str(e)}"
        return result_state

    return asyncio.run(_async_store_data())

def medichat_call_node_wrapper(state: ChatState, repository, llm_service) -> ChatState:
    """Wrapper đồng bộ cho hàm medichat_call_node bất đồng bộ"""
    async def _async_medichat_call():
        result_state = state.copy()
        # Bỏ qua bước gọi Medichat nếu là tin nhắn chào hỏi hoặc không hợp lệ
        if not result_state['is_valid_scope'] or result_state['need_more_info'] or result_state['error'] or result_state['is_greeting']:
            return result_state
    
        try:
            # Tạo dịch vụ Gemini
            gemini_service = GeminiPromptService()
            
            # Lấy toàn bộ lịch sử trò chuyện, không giới hạn số lượng tin nhắn
            messages = repository.get_messages_with_summary(result_state['conversation_id'])
            
            # Cập nhật tin nhắn hiện tại của người dùng vào state nếu chưa được lưu
            current_user_message_exists = False
            for msg in messages:
                if msg["role"] == "user" and msg["content"] == result_state['user_message']:
                    current_user_message_exists = True
                    break
            
            # Nếu tin nhắn hiện tại chưa có trong lịch sử, thêm vào để Gemini có thể sử dụng
            if not current_user_message_exists:
                messages.append({"role": "user", "content": result_state['user_message']})
            
            # Tạo prompt tối ưu cho Medichat bằng Gemini
            medichat_prompt = await gemini_service.create_medichat_prompt(messages)
            result_state['medichat_prompt'] = medichat_prompt
            
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
            result_state['medichat_response'] = medichat_response
            
            logger.info(f"Đã nhận phản hồi từ Medichat: {medichat_response[:50]}...")
        except Exception as e:
            logger.error(f"Lỗi khi gọi đến Medichat: {str(e)}")
            result_state['error'] = f"Lỗi khi gọi đến Medichat: {str(e)}"
        
        return result_state

    return asyncio.run(_async_medichat_call())

def response_cleanup_node_wrapper(state: ChatState, repository) -> ChatState:
    """Wrapper đồng bộ cho hàm response_cleanup_node bất đồng bộ"""
    async def _async_response_cleanup():
        result_state = state.copy()
    
        # Xử lý các trường hợp
        try:
            if not result_state['is_valid_scope']:
                # Nếu không thuộc phạm vi hợp lệ, trả về thông báo từ chối
                result_state['final_response'] = "Xin lỗi, câu hỏi của bạn nằm ngoài phạm vi tư vấn dinh dưỡng và sức khỏe. Tôi chỉ có thể hỗ trợ về các vấn đề liên quan đến sức khỏe, dinh dưỡng, và món ăn phù hợp. Bạn có thể đặt câu hỏi khác không?"
            
            elif result_state['is_greeting']:
                # Trường hợp chào hỏi đã được xử lý trong check_scope_node
                # Đảm bảo có final_response
                if not result_state['final_response']:
                    result_state['final_response'] = "Xin chào! Tôi là trợ lý tư vấn dinh dưỡng và sức khỏe. Tôi có thể giúp gì cho bạn hôm nay?"
            
            elif result_state['need_more_info'] and result_state['follow_up_question']:
                # Nếu cần thêm thông tin, trả về câu hỏi tiếp theo
                # (đã được gán trong collect_info_node)
                pass
            
            elif result_state['error']:
                # Nếu có lỗi, trả về thông báo lỗi
                result_state['final_response'] = "Xin lỗi, hiện tôi không thể kết nối tới hệ thống trí tuệ nhân tạo. Vui lòng thử lại sau."
            
            elif result_state['medichat_response']:
                # Nếu có phản hồi từ Medichat, kiểm tra và tinh chỉnh
                gemini_service = GeminiPromptService()
                if result_state['medichat_prompt']:
                    polished_response = await gemini_service.polish_response(
                        result_state['medichat_response'], 
                        result_state['medichat_prompt']
                    )
                    result_state['final_response'] = polished_response
                else:
                    result_state['final_response'] = result_state['medichat_response']
            else:
                # Nếu không có phản hồi, trả về thông báo lỗi
                result_state['final_response'] = "Xin lỗi, tôi không thể xử lý yêu cầu của bạn. Vui lòng thử lại."
                
            logger.info(f"Phản hồi cuối cùng: {result_state['final_response'][:50]}...")
            
            # LUÔN lưu phản hồi vào cơ sở dữ liệu, bất kể phạm vi
            if result_state['final_response']:
                repository.add_message(result_state['conversation_id'], "assistant", result_state['final_response'])
                logger.info(f"Đã lưu phản hồi trợ lý: {result_state['final_response'][:50]}...")
                
                # Cập nhật trạng thái để API có thể truy cập
                result_state["assistant_message"] = {
                    "role": "assistant",
                    "content": result_state['final_response']
                }
                
        except Exception as e:
            logger.error(f"Lỗi khi tinh chỉnh phản hồi: {str(e)}")
            result_state['final_response'] = "Xin lỗi, có lỗi xảy ra khi xử lý phản hồi. Vui lòng thử lại sau."
            # Vẫn lưu phản hồi lỗi
            repository.add_message(result_state['conversation_id'], "assistant", result_state['final_response'])
        
        return result_state
        
    return asyncio.run(_async_response_cleanup())

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
        # Chuẩn bị dữ liệu sức khỏe từ thông tin đã thu thập
        health_data = {
            'conversation_id': conversation_id,
            'user_id': user_id
        }
        
        # Thêm các trường cụ thể nếu có
        for field in ['health_condition', 'medical_history', 'allergies', 'dietary_habits', 'health_goals']:
            if field in data and data[field]:
                health_data[field] = data[field]
                logger.debug(f"Đã thêm {field}={data[field]} vào dữ liệu để lưu")
        
        # Thêm các dữ liệu khác vào additional_info nếu có
        additional_info = {}
        for key, value in data.items():
            if key not in ['health_condition', 'medical_history', 'allergies', 'dietary_habits', 'health_goals'] and value:
                additional_info[key] = value
        
        if additional_info:
            health_data['additional_info'] = additional_info
            logger.debug(f"Đã thêm thông tin bổ sung: {additional_info}")
        
        # Gọi đến repository để lưu vào DB
        repo.save_health_data(**health_data)
        logger.info(f"Đã lưu thông tin sức khỏe vào DB cho conversation_id={conversation_id} với {len(health_data)-2} trường dữ liệu chính")
    except Exception as e:
        logger.error(f"Lỗi khi lưu thông tin sức khỏe vào DB: {str(e)}")
        raise

# Định nghĩa router cho đồ thị
def define_router(state: ChatState) -> str:
    """Router cho StateGraph để xác định node tiếp theo trong đồ thị"""
    # Xử lý trường hợp đặc biệt: tin nhắn chào hỏi
    if state.get("is_greeting", False):
        return "response_cleanup"
    
    # Kiểm tra các điều kiện để quyết định node tiếp theo
    if not state.get("is_valid_scope"):
        return "response_cleanup"
    elif state.get("need_more_info") and state.get("follow_up_question"):
        return "collect_info"
    else:
        return "store_data"

# Khởi tạo StateGraph cho luồng xử lý chat
def create_chat_flow_graph(repository=None, llm_service=None):
    """Tạo và cấu hình StateGraph cho luồng xử lý chat"""
    # Tạo đồ thị với trạng thái là ChatState
    builder = StateGraph(ChatState)
    
    # Thêm các node vào đồ thị
    builder.add_node("check_scope", run_async(check_scope_node))
    builder.add_node("collect_info", collect_info_node)
    builder.add_node("store_data", lambda state: store_data_node_wrapper(state, repository))
    builder.add_node("medichat_call", lambda state: medichat_call_node_wrapper(state, repository, llm_service))
    builder.add_node("response_cleanup", lambda state: response_cleanup_node_wrapper(state, repository))
    
    # Cấu hình điểm bắt đầu
    builder.set_entry_point("check_scope")
    
    # Cấu hình các cạnh có điều kiện sử dụng router
    builder.add_conditional_edges(
        "check_scope",
        define_router,
        {
            "collect_info": "collect_info",
            "response_cleanup": "response_cleanup",
            "store_data": "store_data"
        }
    )
    
    # Cấu hình cạnh từ collect_info đến response_cleanup
    builder.add_edge("collect_info", "response_cleanup")
    
    # Cấu hình cạnh từ store_data đến các node tiếp theo
    # Nếu là tin nhắn chào hỏi, đi thẳng đến response_cleanup sau khi lưu
    builder.add_conditional_edges(
        "store_data",
        lambda state: "response_cleanup" if state.get("is_greeting", False) else "medichat_call",
        {
            "medichat_call": "medichat_call",
            "response_cleanup": "response_cleanup"
        }
    )
    
    # Cấu hình cạnh từ medichat_call đến response_cleanup
    builder.add_edge("medichat_call", "response_cleanup")
    
    # Cấu hình điểm kết thúc
    builder.add_edge("response_cleanup", END)
    
    # Biên dịch graph thành dạng thực thi
    checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)

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
        # Kiểm tra giới hạn số tin nhắn trong phiên trò chuyện
        limits = await check_conversation_limits(conversation_id, repository)
        if limits["limit_reached"]:
            # Đã đạt giới hạn 30 tin nhắn
            # Không cần lưu tin nhắn người dùng vì đã được lưu ở API trước đó
            limit_message = "Bạn đã đạt đến giới hạn 30 tin nhắn trong phiên trò chuyện này. Vui lòng bắt đầu một phiên mới để tiếp tục."
            repository.add_message(conversation_id, "assistant", limit_message)
            
            logger.info(f"Đã đạt giới hạn 30 tin nhắn trong phiên trò chuyện {conversation_id}")
            
            return {
                "conversation_id": conversation_id,
                "user_message": {"role": "user", "content": user_message},
                "assistant_message": {"role": "assistant", "content": limit_message},
                "is_valid_scope": True,
                "need_more_info": False,
                "final_response": limit_message,
                "limit_reached": True,
                "message_count": limits["message_count"]
            }
        
        # Khởi tạo trạng thái
        state = ChatState(
            conversation_id=conversation_id,
            user_id=user_id,
            user_message=user_message,
            messages=messages,
            is_valid_scope=True,  # Giả định mặc định là hợp lệ
            is_greeting=False,
            need_more_info=False,
            collected_info={},
            medichat_prompt=None,
            medichat_response=None,
            final_response=None,
            follow_up_question=None,
            error=None
        )
        
        # Tạo graph xử lý chat
        chat_graph = create_chat_flow_graph(repository, llm_service)
        
        # Tạo thread id duy nhất cho conversation
        thread_id = f"conversation_{conversation_id}"
        
        # Chạy đồ thị với trạng thái đã khởi tạo
        result = await chat_graph.ainvoke(
            state, 
            config={"configurable": {"thread_id": thread_id}},
        )
        
        # Đảm bảo luôn có final_response
        if not result.get("final_response"):
            if result.get("error"):
                result["final_response"] = "Xin lỗi, có lỗi xảy ra trong quá trình xử lý. Vui lòng thử lại sau."
            elif not result.get("is_valid_scope", True):
                result["final_response"] = "Xin lỗi, câu hỏi của bạn nằm ngoài phạm vi tư vấn dinh dưỡng và sức khỏe. Tôi chỉ có thể hỗ trợ về các vấn đề liên quan đến sức khỏe, dinh dưỡng, và món ăn phù hợp."
            else:
                result["final_response"] = "Xin lỗi, tôi không thể xử lý yêu cầu của bạn. Vui lòng thử lại."
        
        # Đảm bảo có assistant_message cho API trả về
        if not result.get("assistant_message"):
            result["assistant_message"] = {
                "role": "assistant",
                "content": result.get("final_response", "")
            }
        
        # Thêm thông tin về số lượng tin nhắn
        result["limit_reached"] = False
        result["message_count"] = limits["message_count"] + 2  # +2 cho tin nhắn mới (user + assistant)
        
        # Log kết quả
        logger.info(f"Kết quả từ xử lý: valid={result.get('is_valid_scope')}, need_more={result.get('need_more_info')}")
        
        return result
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
            "final_response": "Xin lỗi, hệ thống gặp lỗi khi xử lý yêu cầu của bạn.",
            "limit_reached": False,
            "message_count": 0
        }

async def check_conversation_limits(conversation_id: int, repository) -> Dict[str, Any]:
    """
    Kiểm tra giới hạn số tin nhắn trong phiên trò chuyện
    
    Args:
        conversation_id: ID của cuộc trò chuyện
        repository: Repository để truy vấn dữ liệu
        
    Returns:
        Dict chứa thông tin về giới hạn: {'limit_reached': bool, 'message_count': int}
    """
    try:
        # Lấy danh sách tin nhắn hiện tại
        messages = repository.get_messages(conversation_id)
        message_count = len(messages)
        
        # Kiểm tra nếu vượt quá giới hạn 30 tin nhắn
        limit_reached = message_count >= 30
        
        logger.info(f"Kiểm tra giới hạn tin nhắn: conversation_id={conversation_id}, số tin nhắn={message_count}, vượt giới hạn={limit_reached}")
        
        return {
            "limit_reached": limit_reached,
            "message_count": message_count
        }
    except Exception as e:
        logger.error(f"Lỗi khi kiểm tra giới hạn tin nhắn: {str(e)}")
        # Fallback nếu có lỗi: không giới hạn
        return {
            "limit_reached": False,
            "message_count": 0
        } 