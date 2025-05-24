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
from app.services.cache_service import CacheService
from app.db.models import Message, HealthData

logger = logging.getLogger(__name__)

# Import các tool cần thiết
from app.tools.recipe_tool import search_and_filter_recipes
from app.tools.product_find_tool import process_user_request_async
from app.tools.product_beverage import fetch_and_filter_drinks_in_batches_async, init_services

# Định nghĩa trạng thái với đầy đủ các trường cần thiết
class ChatState(TypedDict):
    # Thông tin cơ bản về cuộc trò chuyện và người dùng
    conversation_id: int
    user_id: int
    user_message: str  # Tin nhắn hiện tại của người dùng

    # Lịch sử tin nhắn (có thể dùng để tái tạo ngữ cảnh nếu cần)
    messages: List[Dict[str, str]]

    # Cờ và thông tin từ bước phân tích (check_scope_node)
    is_valid_scope: bool  # Tin nhắn có nằm trong phạm vi hỗ trợ không?
    is_greeting: bool     # Tin nhắn có phải là lời chào hỏi không?
    is_food_related: bool # Tin nhắn có liên quan đến món ăn/đồ uống không?

    user_rejected_info: bool # Người dùng có từ chối cung cấp thêm thông tin không?
    need_more_info: bool     # Có cần hỏi thêm thông tin từ người dùng không?
    suggest_general_options: bool # Có nên gợi ý các lựa chọn chung chung không (do thiếu thông tin/từ chối)?
    
    follow_up_question: Optional[str] # Câu hỏi tiếp theo nếu need_more_info là true

    collected_info: Dict[str, Any] # Thông tin sức khỏe, sở thích đã thu thập được từ người dùng

    # Thông tin liên quan đến việc gọi mô hình LLM (Medichat/LLaMA3)
    medichat_prompt: Optional[str]    # Prompt đã được tạo để gửi cho Medichat
    medichat_response: Optional[str]  # Phản hồi thô từ Medichat

    # Kết quả từ các tool (nếu có)
    recipe_results: Optional[List[Dict[str, Any]]] # Kết quả từ recipe_tool
    beverage_results: Optional[List[Dict[str, Any]]] # Kết quả từ product_beverage
    product_results: Optional[Dict[str, Any]]    # Kết quả từ product_find_tool
    
    # Các cờ phân loại yêu cầu (từ Nhiệm vụ E.1)
    requests_food: Optional[bool] # Yêu cầu cụ thể về món ăn
    requests_beverage: Optional[bool] # Yêu cầu cụ thể về đồ uống

    # Phản hồi cuối cùng và lỗi
    final_response: Optional[str] # Phản hồi cuối cùng sẽ được gửi cho người dùng
    error: Optional[str]          # Thông báo lỗi nếu có sự cố xảy ra trong quá trình xử lý
    
    # ID của tin nhắn trong database (để ChatService có thể truy cập)
    user_message_id_db: Optional[int] # ID của tin nhắn người dùng trong DB
    assistant_message_id_db: Optional[int] # ID của tin nhắn trợ lý trong DB

# Các node xử lý
async def check_scope_node(state: ChatState) -> ChatState:
    """
    Kiểm tra xem nội dung tin nhắn có thuộc phạm vi hỗ trợ không.
    Cập nhật state với đầy đủ các cờ từ analyze_query và logic xử lý need_more_info.
    """
    logger.info(f"🔍 Đang kiểm tra phạm vi nội dung: {state['user_message'][:50]}...")
    
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
            state['is_food_related'] = False
            state['need_more_info'] = False
            state['user_rejected_info'] = False # Không từ chối khi chào
            state['suggest_general_options'] = False
            state['follow_up_question'] = None
            greeting_response = await gemini_service.get_greeting_response(state['user_message'])
            state['final_response'] = greeting_response # Sẽ được gán vào assistant_message ở cleanup
            logger.info(f"✅ Đã xác định là tin nhắn chào hỏi, phản hồi: {greeting_response[:50]}...")
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
        state['is_food_related'] = analysis.get('is_food_related', False)
        state['user_rejected_info'] = analysis.get('user_rejected_info', False)
        state['suggest_general_options'] = analysis.get('suggest_general_options', False)
        
        # ⭐ CÁC CỜ MỚI TỪ NHIỆM VỤ E.1: Phân loại món ăn và đồ uống
        state['requests_food'] = analysis.get('requests_food', False)
        state['requests_beverage'] = analysis.get('requests_beverage', False)
        
        # ⭐ LOGIC QUAN TRỌNG: Xử lý need_more_info dựa trên user_rejected_info và suggest_general_options
        if state['user_rejected_info'] or state['suggest_general_options']:
            state['need_more_info'] = False
            state['follow_up_question'] = None
            logger.info(f"🎯 User từ chối hoặc cần gợi ý chung → need_more_info = False")
        else:
            state['need_more_info'] = analysis.get('need_more_info', False)
            state['follow_up_question'] = analysis.get('follow_up_question')
        
        # Lấy thông tin sức khỏe từ phân tích của Gemini - không tự xử lý từ khóa nữa
        collected_info = analysis.get('collected_info', {})
        
        # Lọc bỏ các giá trị rỗng từ collected_info
        collected_info = {k: v for k, v in collected_info.items() if v}
        state['collected_info'] = collected_info
        
        logger.info(f"✅ Kết quả phân tích scope:")
        logger.info(f"   - is_valid_scope: {state['is_valid_scope']}")
        logger.info(f"   - is_food_related: {state['is_food_related']}")
        logger.info(f"   - user_rejected_info: {state['user_rejected_info']}")
        logger.info(f"   - suggest_general_options: {state['suggest_general_options']}")
        logger.info(f"   - need_more_info: {state['need_more_info']}")
        logger.info(f"   - requests_food: {state['requests_food']}")
        logger.info(f"   - requests_beverage: {state['requests_beverage']}")
        
        # Lưu thông tin sức khỏe vào Redis nếu có và thuộc phạm vi hợp lệ
        if state['is_valid_scope'] and collected_info:
            await save_health_data_to_cache(
                state['conversation_id'], 
                state['user_id'],
                collected_info
            )
            logger.info(f"💾 Đã lưu thông tin sức khỏe vào cache: session:{state['conversation_id']}:health_info")
            
    except Exception as e:
        logger.error(f"💥 Lỗi khi kiểm tra phạm vi: {str(e)}", exc_info=True)
        state['error'] = f"Lỗi khi kiểm tra phạm vi: {str(e)}"
        state['is_valid_scope'] = True  # Fallback cho phép tiếp tục
        state['is_food_related'] = False
        state['need_more_info'] = False
        state['user_rejected_info'] = False
        state['suggest_general_options'] = False
    
    return state

def persist_user_interaction_node_wrapper(state: ChatState, repository) -> ChatState:
    """
    ⭐ NODE MỚI: Luôn lưu user_message và cập nhật user_message_id_db vào state.
    Node này chạy ngay sau check_scope_node để đảm bảo user_message_id_db luôn có.
    """
    async def _async_persist_user_interaction():
        result_state = state.copy()
        try:
            logger.info("💾 Persist user interaction node - đang lưu user message...")
            
            # Tìm tin nhắn người dùng đã tồn tại trong database
            from sqlalchemy import desc
            from app.db.models import Message
            
            db = repository.db
            existing_message = db.query(Message).filter(
                Message.conversation_id == result_state['conversation_id'],
                Message.role == "user",
                Message.content == result_state['user_message']
            ).order_by(desc(Message.created_at)).first()
            
            if existing_message:
                # Tin nhắn đã tồn tại, chỉ cập nhật ID
                result_state['user_message_id_db'] = existing_message.message_id
                logger.info(f"📌 User message đã tồn tại với ID: {existing_message.message_id}")
            else:
                # LUÔN lưu tin nhắn người dùng vào cơ sở dữ liệu, bất kể phạm vi
                user_message_db_obj = repository.add_message(
                    result_state['conversation_id'], 
                    "user", 
                    result_state['user_message']
                )
                result_state['user_message_id_db'] = user_message_db_obj.message_id
                logger.info(f"💾 Đã lưu user message với ID: {user_message_db_obj.message_id}")
            
            # Tạo user_message object để đồng nhất với response format
            result_state["user_message"] = {
                "role": "user",
                "content": result_state['user_message']
            }
            
            # Lưu thông tin sức khỏe vào database nếu có collected_info và thuộc phạm vi hợp lệ
            if (result_state.get('collected_info') and 
                result_state.get('is_valid_scope') and 
                not result_state.get('is_greeting')):
                
                try:
                    await save_health_data_to_db(
                        repository, 
                        result_state['conversation_id'], 
                        result_state['user_id'], 
                        result_state['collected_info']
                    )
                    logger.info("💾 Đã lưu collected_info vào HealthData database")
                except Exception as health_error:
                    logger.error(f"⚠️ Lỗi khi lưu health_data: {health_error}")
                    # Không fail toàn bộ process vì lỗi health_data
            
            logger.info(f"✅ Persist user interaction hoàn tất. user_message_id_db: {result_state.get('user_message_id_db')}")
            
        except Exception as e:
            logger.error(f"💥 Lỗi trong persist_user_interaction_node: {e}", exc_info=True)
            result_state['error'] = f"Lỗi lưu user message: {str(e)}"
            # Không fail hard, để luồng tiếp tục
        
        return result_state

    return asyncio.run(_async_persist_user_interaction())

def collect_info_node(state: ChatState) -> ChatState:
    """Thu thập thêm thông tin từ người dùng"""
    logger.info("📝 Đang chuẩn bị thu thập thêm thông tin...")
    
    # Nếu là tin nhắn chào hỏi thì bỏ qua bước này
    if state['is_greeting']:
        return state
    
    # Nếu cần thêm thông tin, chuẩn bị câu hỏi tiếp theo
    if state['need_more_info'] and state['follow_up_question']:
        logger.info(f"❓ Cần thu thập thêm thông tin: {state['follow_up_question']}")
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
    """
    Wrapper đồng bộ cho hàm store_data_node bất đồng bộ.
    ⚠️ CHÚ Ý: user_message_id_db đã được xử lý trong persist_user_interaction_node.
    Node này giờ chỉ xử lý các logic còn lại nếu cần.
    """
    async def _async_store_data():
        result_state = state.copy()
        try:
            logger.info("📂 Store data node - xử lý logic bổ sung...")
            
            # Logic bổ sung có thể thêm ở đây nếu cần
            # user_message_id_db đã được xử lý trong persist_user_interaction_node
            
            # Xác nhận user_message_id_db đã có
            if not result_state.get('user_message_id_db'):
                logger.warning("⚠️ user_message_id_db chưa có trong store_data_node")
            else:
                logger.info(f"✅ Store data node: user_message_id_db = {result_state['user_message_id_db']}")
                
        except Exception as e:
            logger.error(f"💥 Lỗi trong store_data_node: {e}", exc_info=True)
            result_state['error'] = f"Lỗi trong store_data: {str(e)}"
        
        return result_state

    return asyncio.run(_async_store_data())

def medichat_call_node_wrapper(state: ChatState, repository, llm_service) -> ChatState:
    """Wrapper đồng bộ cho hàm medichat_call_node bất đồng bộ"""
    async def _async_medichat_call():
        result_state = state.copy()
        
        # Bỏ qua nếu không hợp lệ hoặc cần thêm thông tin
        if not result_state['is_valid_scope'] or result_state['need_more_info'] or result_state['error'] or result_state['is_greeting']:
            return result_state

        try:
            gemini_service = GeminiPromptService()
            
            # Lấy toàn bộ lịch sử trò chuyện
            messages = repository.get_messages_with_summary(result_state['conversation_id'])
            
            # Thêm tin nhắn hiện tại nếu chưa có
            current_user_message_exists = False
            for msg in messages:
                if msg["role"] == "user" and msg["content"] == result_state['user_message']:
                    current_user_message_exists = True
                    break
            
            if not current_user_message_exists:
                messages.append({"role": "user", "content": result_state['user_message']})
            
            # Tạo prompt cho Medichat
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
                    "content": result_state['medichat_prompt']
                }
            ]
            
            # Gọi đến Medichat để lấy phản hồi
            logger.info(f"📞 Gửi prompt đến Medichat: {result_state['medichat_prompt'][:100]}...")
            medichat_response = await llm_service.get_full_response(medichat_messages)
            result_state['medichat_response'] = medichat_response
            
            logger.info(f"✅ Đã nhận phản hồi từ Medichat: {medichat_response[:50]}...")
        except Exception as e:
            logger.error(f"💥 Lỗi khi gọi đến Medichat: {str(e)}", exc_info=True)
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
                # Cập nhật thông báo từ chối để phản ánh đúng phạm vi tư vấn
                result_state['final_response'] = ("Xin lỗi, câu hỏi của bạn nằm ngoài phạm vi tư vấn của tôi. "
                                                "Tôi chỉ có thể hỗ trợ về các vấn đề liên quan đến dinh dưỡng, sức khỏe, "
                                                "món ăn và đồ uống. Bạn có thể đặt câu hỏi khác trong phạm vi này không?")
            
            elif result_state['is_greeting']:
                if not result_state['final_response']:
                    result_state['final_response'] = "Xin chào! Tôi là trợ lý tư vấn dinh dưỡng và sức khỏe. Tôi có thể giúp gì cho bạn hôm nay?"
            
            elif result_state['need_more_info'] and result_state['follow_up_question']:
                pass
            
            elif result_state['error']:
                result_state['final_response'] = "Xin lỗi, hiện tôi không thể kết nối tới hệ thống trí tuệ nhân tạo. Vui lòng thử lại sau."
            
            elif result_state['medichat_response']:
                # Polish response
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
                result_state['final_response'] = "Xin lỗi, tôi không thể xử lý yêu cầu của bạn. Vui lòng thử lại."
                
            logger.info(f"📝 Phản hồi cuối cùng: {result_state['final_response'][:50]}...")
            
            # LUÔN lưu phản hồi vào cơ sở dữ liệu và cập nhật assistant_message_id_db
            if result_state['final_response']:
                assistant_message_db_obj = repository.add_message(result_state['conversation_id'], "assistant", result_state['final_response'])
                result_state['assistant_message_id_db'] = assistant_message_db_obj.message_id
                logger.info(f"💾 Đã lưu phản hồi trợ lý với ID={assistant_message_db_obj.message_id}: {result_state['final_response'][:50]}...")
                
                result_state["assistant_message"] = {
                    "role": "assistant",
                    "content": result_state['final_response']
                }
        except Exception as e:
            logger.error(f"💥 Lỗi khi xử lý phản hồi: {str(e)}", exc_info=True)
            result_state['error'] = f"Lỗi khi xử lý phản hồi: {str(e)}"
            result_state['final_response'] = "Xin lỗi, có lỗi xảy ra khi xử lý phản hồi. Vui lòng thử lại sau."

        return result_state

    return asyncio.run(_async_response_cleanup())

async def save_health_data_to_cache(conversation_id: int, user_id: int, data: Dict[str, Any]) -> None:
    """Lưu thông tin sức khỏe vào Redis Cache với TTL"""
    try:
        cache_key = f"session:{conversation_id}:health_info"
        cache_data = {
            "user_id": user_id,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        
        CacheService.set_cache(cache_key, cache_data, ttl=CacheService.TTL_LONG)
        logger.info(f"💾 Đã lưu thông tin sức khỏe vào cache: {cache_key}")
        
    except Exception as e:
        logger.error(f"💥 Lỗi khi lưu thông tin sức khỏe vào cache: {str(e)}")

async def save_health_data_to_db(repo, conversation_id: int, user_id: int, data: Dict[str, Any]) -> None:
    """Lưu thông tin sức khỏe vào cơ sở dữ liệu"""
    try:
        # Lọc dữ liệu theo các trường được hỗ trợ
        health_condition = data.get('health_condition')
        medical_history = data.get('medical_history')
        allergies = data.get('allergies')
        dietary_habits = data.get('dietary_habits')
        health_goals = data.get('health_goals')
        
        # Lưu additional_info cho các thông tin khác
        additional_info = {k: v for k, v in data.items() 
                         if k not in ['health_condition', 'medical_history', 'allergies', 'dietary_habits', 'health_goals']}
        
        health_data = repo.save_health_data(
            conversation_id=conversation_id,
            user_id=user_id,
            health_condition=health_condition,
            medical_history=medical_history,
            allergies=allergies,
            dietary_habits=dietary_habits,
            health_goals=health_goals,
            additional_info=additional_info
        )
        
        logger.info(f"💾 Đã lưu thông tin sức khỏe vào DB: conversation_id={conversation_id}")
        
    except Exception as e:
        logger.error(f"💥 Lỗi khi lưu thông tin sức khỏe vào DB: {str(e)}")

async def recipe_search_node(state: ChatState) -> ChatState:
    """Tìm kiếm công thức món ăn từ database"""
    logger.info("🔍 Bắt đầu tìm kiếm công thức món ăn...")
    
    try:
        gemini_service = GeminiPromptService()
        
        # Tạo query tìm kiếm với suggest_general_if_needed
        suggest_general_if_needed = state.get('suggest_general_options', False)
        
        search_query = await gemini_service.create_recipe_search_prompt(
            state['user_message'], 
            state.get('collected_info', {}),
            suggest_general_if_needed=suggest_general_if_needed
        )
        
        logger.info(f"🔍 Recipe search query: {search_query}")
        
        # Gọi recipe tool (synchronous function - không await)
        recipe_json_str = search_and_filter_recipes(search_query)
        
        # Parse JSON result với error handling mạnh mẽ
        recipes = []
        if recipe_json_str:
            try:
                recipes_data = json.loads(recipe_json_str)
                
                # Kiểm tra định dạng trả về từ recipe_tool
                if isinstance(recipes_data, list) and all(isinstance(item, dict) for item in recipes_data):
                    recipes = recipes_data
                elif isinstance(recipes_data, dict) and "recipes" in recipes_data and isinstance(recipes_data["recipes"], list):
                    # Trường hợp recipe_tool trả về {"recipes": [...], "errors": [...]}
                    recipes = recipes_data["recipes"]
                    if "errors" in recipes_data and recipes_data["errors"]:
                        logger.warning(f"⚠️ Lỗi từ recipe_tool: {recipes_data['errors']}")
                elif isinstance(recipes_data, dict) and "error" in recipes_data:
                    logger.error(f"💥 Recipe tool trả về lỗi: {recipes_data['error']}")
                    recipes = []
                else:
                    logger.warning(f"⚠️ Recipe tool trả về định dạng không mong muốn: {type(recipes_data)}")
                    recipes = []
                    
            except json.JSONDecodeError as json_error:
                logger.error(f"💥 Lỗi parse JSON từ recipe_tool: {str(json_error)}")
                logger.error(f"Raw response (first 200 chars): {recipe_json_str[:200]}")
                recipes = []
                # Không set error vì recipe search không phải critical
        else:
            logger.warning("⚠️ Recipe tool không trả về kết quả.")
            recipes = []

        if recipes:
            # Lọc trùng lặp bằng Gemini (hàm này là async)
            try:
                filtered_recipes = await gemini_service.filter_duplicate_recipes(recipes)
                state['recipe_results'] = filtered_recipes[:10]  # Giới hạn 10 recipes tốt nhất
                logger.info(f"✅ Đã lọc từ {len(recipes)} xuống {len(filtered_recipes)} recipes, lưu {len(state['recipe_results'])} recipes")
            except Exception as filter_error:
                logger.error(f"💥 Lỗi khi lọc recipes: {str(filter_error)}")
                # Fallback: sử dụng recipes chưa lọc, giới hạn 10
                state['recipe_results'] = recipes[:10]
                logger.info(f"✅ Sử dụng {len(state['recipe_results'])} recipes chưa lọc (fallback)")
        else:
            state['recipe_results'] = []
            logger.info("❌ Không tìm thấy công thức món ăn phù hợp sau khi parse.")
            
    except Exception as e:
        logger.error(f"💥 Lỗi nghiêm trọng khi tìm kiếm công thức: {str(e)}", exc_info=True)
        state['recipe_results'] = []
        # Không set error để không chặn luồng xử lý tiếp theo
        logger.info("🔄 Tiếp tục xử lý mà không có recipes")
    
    return state

async def product_search_node(state: ChatState) -> ChatState:
    """Tìm kiếm sản phẩm từ medichat response và recipe results"""
    logger.info("🛒 Bắt đầu tìm kiếm sản phẩm...")
    
    try:
        # Chỉ tìm kiếm sản phẩm nếu có medichat_response
        if not state.get('medichat_response'):
            logger.info("❌ Không có medichat_response để tìm kiếm sản phẩm")
            state['product_results'] = {}
            return state
        
        gemini_service = GeminiPromptService()
        
        # Tạo product search prompt với cả recipes và beverages
        product_search_prompt = await gemini_service.create_product_search_prompt(
            state['medichat_response'],
            state.get('recipe_results'),
            state.get('beverage_results')
        )
        
        logger.info(f"🛒 Product search prompt: {product_search_prompt}")
        
        # Gọi product tool (async function)
        product_results = await process_user_request_async(product_search_prompt)
        
        if product_results:
            state['product_results'] = product_results
            logger.info(f"✅ Tìm thấy thông tin sản phẩm")
        else:
            state['product_results'] = {}
            logger.info("❌ Không tìm thấy sản phẩm phù hợp")
            
    except Exception as e:
        logger.error(f"💥 Lỗi khi tìm kiếm sản phẩm: {str(e)}", exc_info=True)
        state['product_results'] = {}
        # Không set error ở đây vì product search là optional
    
    return state

async def beverage_search_node(state: ChatState) -> ChatState:
    """
    ⭐ NODE MỚI: Tìm kiếm đồ uống từ product_beverage tool với async optimization.
    Sử dụng fetch_and_filter_drinks_in_batches_async để lấy danh sách đồ uống.
    """
    logger.info("🥤 Bắt đầu tìm kiếm đồ uống...")
    
    try:
        # ⭐ KHỞI TẠO SERVICES (CẬP NHẬT CHO VERSION MỚI)
        loop = asyncio.get_event_loop()
        pinecone_index, vector_dimension = await loop.run_in_executor(
            None, init_services
        )
        
        logger.info(f"🔧 Đã khởi tạo services: vector_dim={vector_dimension}")
        
        # ⭐ GỌI HÀM ASYNC MỚI TRỰC TIẾP
        beverages_data = await fetch_and_filter_drinks_in_batches_async(pinecone_index, vector_dimension)
        
        if beverages_data and isinstance(beverages_data, list):
            state['beverage_results'] = beverages_data
            logger.info(f"✅ Tìm thấy {len(beverages_data)} đồ uống.")
            # Log một vài sản phẩm đầu để debug
            for i, beverage in enumerate(beverages_data[:3]):
                logger.info(f"   - Đồ uống {i+1}: {beverage.get('product_name', 'N/A')}")
        else:
            state['beverage_results'] = []
            logger.info("❌ Không tìm thấy đồ uống phù hợp.")
            
    except Exception as e:
        logger.error(f"💥 Lỗi khi tìm kiếm đồ uống: {str(e)}", exc_info=True)
        state['beverage_results'] = []
        # Không set state['error'] để luồng có thể tiếp tục
        logger.info("🔄 Tiếp tục xử lý mà không có beverage results")
    
    return state

def enhanced_medichat_call_node_wrapper(state: ChatState, repository, llm_service) -> ChatState:
    """
    Enhanced medichat call với recipes nếu có.
    Truyền đúng cờ suggest_general_options khi tạo prompt.
    """
    async def _async_enhanced_medichat_call():
        result_state = state.copy()
        
        # Bỏ qua nếu không hợp lệ hoặc cần thêm thông tin
        if not result_state['is_valid_scope'] or result_state['need_more_info'] or result_state['error'] or result_state['is_greeting']:
            return result_state

        try:
            gemini_service = GeminiPromptService()
            
            # Lấy toàn bộ lịch sử trò chuyện
            messages = repository.get_messages_with_summary(result_state['conversation_id'])
            
            # Thêm tin nhắn hiện tại nếu chưa có
            current_user_message_exists = False
            for msg in messages:
                if msg["role"] == "user" and msg["content"] == result_state['user_message']:
                    current_user_message_exists = True
                    break
            
            if not current_user_message_exists:
                messages.append({"role": "user", "content": result_state['user_message']})
            
            # Lấy cờ suggest_general từ state
            suggest_general = result_state.get('suggest_general_options', False)
            
            # Lấy recipes và beverages từ state với kiểm tra an toàn
            recipe_list = result_state.get('recipe_results')
            if recipe_list is None:
                recipe_list = []
            
            beverage_list = result_state.get('beverage_results')
            if beverage_list is None:
                beverage_list = []
            
            # Nếu có recipe results hoặc beverage results, tạo enhanced prompt
            if result_state.get('is_food_related') and (recipe_list or beverage_list):
                # Tạo prompt với recipes, beverages và suggest_general sử dụng method enhanced
                medichat_prompt = await gemini_service.create_enhanced_medichat_prompt(messages, recipe_list, beverage_list, suggest_general)
                result_state['medichat_prompt'] = medichat_prompt
                logger.info(f"📝 Tạo enhanced prompt với {len(recipe_list)} recipes, {len(beverage_list)} beverages, suggest_general={suggest_general}")
                
            elif suggest_general:
                # Trường hợp suggest_general=true nhưng không có recipes/beverages
                medichat_prompt = await gemini_service.create_enhanced_medichat_prompt(messages, None, None, suggest_general)
                result_state['medichat_prompt'] = medichat_prompt
                logger.info(f"📝 Tạo enhanced prompt với suggest_general=True, không có recipes/beverages")
                
            else:
                # Tạo prompt thông thường
                medichat_prompt = await gemini_service.create_medichat_prompt(messages)
                result_state['medichat_prompt'] = medichat_prompt
                logger.info(f"📝 Tạo prompt thông thường")
            
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
                    "content": result_state['medichat_prompt']
                }
            ]
            
            # Gọi đến Medichat để lấy phản hồi
            logger.info(f"📞 Gửi enhanced prompt đến Medichat: {result_state['medichat_prompt'][:100]}...")
            medichat_response = await llm_service.get_full_response(medichat_messages)
            result_state['medichat_response'] = medichat_response
            
            logger.info(f"✅ Đã nhận phản hồi từ Enhanced Medichat: {medichat_response[:50]}...")
        except Exception as e:
            logger.error(f"💥 Lỗi khi gọi đến Enhanced Medichat: {str(e)}", exc_info=True)
            result_state['error'] = f"Lỗi khi gọi đến Medichat: {str(e)}"
        
        return result_state

    return asyncio.run(_async_enhanced_medichat_call())

def enhanced_response_cleanup_node_wrapper(state: ChatState, repository) -> ChatState:
    """
    Enhanced response cleanup với thông tin recipes và products.
    Có fallback tạo gợi ý chung nếu suggest_general_options=true nhưng Medichat không phản hồi.
    Lưu assistant_message_id_db vào state.
    """
    async def _async_enhanced_response_cleanup():
        result_state = state.copy()

        # Xử lý các trường hợp
        try:
            if not result_state['is_valid_scope']:
                # Cập nhật thông báo từ chối để phản ánh đúng phạm vi tư vấn
                result_state['final_response'] = ("Xin lỗi, câu hỏi của bạn nằm ngoài phạm vi tư vấn của tôi. "
                                                "Tôi chỉ có thể hỗ trợ về các vấn đề liên quan đến dinh dưỡng, sức khỏe, "
                                                "món ăn và đồ uống. Bạn có thể đặt câu hỏi khác trong phạm vi này không?")
            
            elif result_state['is_greeting']:
                if not result_state['final_response']:
                    result_state['final_response'] = "Xin chào! Tôi là trợ lý tư vấn dinh dưỡng và sức khỏe. Tôi có thể giúp gì cho bạn hôm nay?"
            
            elif result_state['need_more_info'] and result_state['follow_up_question']:
                pass
            
            elif result_state['error']:
                result_state['final_response'] = "Xin lỗi, hiện tôi không thể kết nối tới hệ thống trí tuệ nhân tạo. Vui lòng thử lại sau."
            
            elif result_state['medichat_response']:
                # Xây dựng comprehensive response cho food/beverage-related queries
                if result_state.get('is_food_related') and (result_state.get('recipe_results') or result_state.get('beverage_results') or result_state.get('product_results')):
                    comprehensive_parts = [result_state['medichat_response']]
                    
                    # ⭐ THÊM THÔNG TIN BEVERAGE RESULTS NẾU CÓ
                    beverages = result_state.get('beverage_results')
                    if beverages is None:
                        beverages = []
                    if beverages:
                        beverage_section = "\n\n🥤 **ĐỒ UỐNG GỢI Ý TỪ DATABASE**"
                        for i, beverage in enumerate(beverages[:8]):  # Giới hạn 8 đồ uống
                            product_name = beverage.get('product_name', 'N/A')
                            product_id = beverage.get('product_id', 'N/A')
                            beverage_section += f"\n{i+1}. {product_name} (ID: {product_id})"
                        
                        if len(beverages) > 8:
                            beverage_section += f"\n... và {len(beverages) - 8} đồ uống khác"
                        
                        comprehensive_parts.append(beverage_section)
                    
                    # Thêm thông tin recipes nếu có  
                    recipes = result_state.get('recipe_results')
                    if recipes is None:
                        recipes = []
                    if recipes:
                        recipe_section = "\n\n📝 **CÔNG THỨC GỢI Ý TỪ DATABASE**"
                        for i, recipe in enumerate(recipes[:5], 1):
                            name = recipe.get('name', 'N/A')
                            url = recipe.get('url', '')
                            ingredients = recipe.get('ingredients_summary', 'N/A')
                            recipe_section += f"\n{i}. **{name}**"
                            if ingredients and ingredients != 'N/A':
                                recipe_section += f"\n   - Nguyên liệu: {ingredients}"
                            if url:
                                recipe_section += f"\n   - Link: {url}"
                        comprehensive_parts.append(recipe_section)
                    
                    # Thêm thông tin products nếu có
                    products = result_state.get('product_results', {})
                    if products and products.get('ingredient_mapping_results'):
                        available_products = []
                        unavailable_ingredients = []
                        
                        for mapping in products['ingredient_mapping_results']:
                            ingredient = mapping.get('requested_ingredient', '')
                            product_id = mapping.get('product_id')
                            product_name = mapping.get('product_name')
                            
                            if product_id and product_name:
                                available_products.append(f"• {ingredient} → {product_name}")
                            elif ingredient:
                                unavailable_ingredients.append(f"• {ingredient}")
                        
                        if available_products:
                            products_section = "\n\n🛒 **SẢN PHẨM CÓ SẴN TRONG CỬA HÀNG**\n" + "\n".join(available_products[:10])
                            comprehensive_parts.append(products_section)
                        
                        if unavailable_ingredients:
                            unavailable_section = "\n\n⚠️ **NGUYÊN LIỆU CẦN TÌM NGUỒN KHÁC**\n" + "\n".join(unavailable_ingredients[:5])
                            comprehensive_parts.append(unavailable_section)
                    
                    # Ghép thành response hoàn chỉnh
                    raw_comprehensive_response = "\n".join(comprehensive_parts)
                    
                    # Polish response
                    gemini_service = GeminiPromptService()
                    if result_state['medichat_prompt']:
                        polished_response = await gemini_service.polish_response(
                            raw_comprehensive_response, 
                            result_state['medichat_prompt']
                        )
                        result_state['final_response'] = polished_response
                    else:
                        result_state['final_response'] = raw_comprehensive_response
                        
                else:
                    # Polish response thông thường
                    gemini_service = GeminiPromptService()
                    if result_state['medichat_prompt']:
                        polished_response = await gemini_service.polish_response(
                            result_state['medichat_response'], 
                            result_state['medichat_prompt']
                        )
                        result_state['final_response'] = polished_response
                    else:
                        result_state['final_response'] = result_state['medichat_response']
                        # ⭐ KIỂM TRA: Nếu chưa có final_response, tạo fallback
            if not result_state.get('final_response'):
                if result_state.get('suggest_general_options', False) and result_state.get('is_valid_scope', True):
                    logger.info("🎯 Enhanced Fallback: Tạo gợi ý chung phong phú vì suggest_general_options=True")
                    
                    # ⭐ SỬ DỤNG TEMPLATE CỐ ĐỊNH CHẤT LƯỢNG CAO 
                    logger.info("📝 Tạo gợi ý chung từ template có sẵn")
                    
                    # Template cố định chất lượng cao
                    import random
                    suggestion_templates = [
                        ("Dạ, tôi hiểu bạn muốn có một số gợi ý chung về món ăn tốt cho sức khỏe. "
                        "Dựa trên các tiêu chí phổ biến, cân bằng dinh dưỡng và dễ chế biến, "
                        "tôi xin đề xuất một số lựa chọn:\n\n"
                        "🥗 **Salad rau củ quả** - Nhiều chất xơ, vitamin, khoáng chất tự nhiên\n"
                        "🍲 **Canh chua cá** - Giàu protein, vitamin C, dễ tiêu hóa\n"
                        "🥣 **Cháo gà** - Dễ ăn, bổ dưỡng, phù hợp nhiều lứa tuổi\n"
                        "🥤 **Nước ép trái cây tươi** - Vitamin tự nhiên, tăng cường miễn dịch\n\n"
                        "Những món này thường dễ tìm nguyên liệu, tốt cho sức khỏe và ít gây dị ứng. "
                        "Bạn có muốn tôi tư vấn cụ thể hơn về món nào không?"),
                        
                        ("Dạ, để gợi ý các món ăn phù hợp chung, tôi có thể đề xuất một số lựa chọn "
                        "dựa trên tính cân bằng dinh dưỡng và độ phổ biến:\n\n"
                        "🍜 **Phở gà** - Nhẹ, dễ ăn, đầy đủ chất dinh dưỡng\n"
                        "🥙 **Bánh mì kẹp rau** - Tiện lợi, có thể tùy chỉnh nguyên liệu\n"
                        "🍯 **Sữa chua mật ong** - Probiotics tốt cho tiêu hóa\n"
                        "🥞 **Bánh yến mạch chuối** - Chất xơ cao, năng lượng bền vững\n\n"
                        "Đây là những lựa chọn an toàn và được nhiều người yêu thích. "
                        "Bạn có thể chia sẻ thêm về sở thích hoặc nhu cầu cụ thể để tôi tư vấn chính xác hơn không?")
                    ]
                    
                    selected_template = random.choice(suggestion_templates)
                    result_state['final_response'] = selected_template
                else:
                    # Fallback cuối cùng nếu không có gì khác
                    result_state['final_response'] = ("Xin lỗi, hiện tôi không thể xử lý yêu cầu của bạn vào lúc này. "
                                                    "Vui lòng thử lại sau hoặc đặt câu hỏi cụ thể hơn về dinh dưỡng, "
                                                    "món ăn hoặc sức khỏe để tôi có thể hỗ trợ bạn tốt hơn.")
                
            logger.info(f"📝 Phản hồi cuối cùng: {result_state['final_response'][:50]}...")
            
            # LUÔN lưu phản hồi vào cơ sở dữ liệu và cập nhật assistant_message_id_db
            if result_state['final_response']:
                assistant_message_db_obj = repository.add_message(result_state['conversation_id'], "assistant", result_state['final_response'])
                result_state['assistant_message_id_db'] = assistant_message_db_obj.message_id
                logger.info(f"💾 Đã lưu phản hồi trợ lý với ID={assistant_message_db_obj.message_id}: {result_state['final_response'][:50]}...")
                
                result_state["assistant_message"] = {
                    "role": "assistant",
                    "content": result_state['final_response']
                }
                
                # Lưu công thức món ăn vào database nếu có food-related results
                if (result_state.get('is_food_related') and 
                    result_state.get('recipe_results') and 
                    result_state.get('product_results')):
                    
                    try:
                        saved_menu_ids = repository.save_multiple_recipes_to_menu(
                            result_state['recipe_results'],
                            result_state['product_results']
                        )
                        
                        if saved_menu_ids:
                            logger.info(f"💾 Đã lưu {len(saved_menu_ids)} công thức món ăn vào database: {saved_menu_ids}")
                    except Exception as recipe_save_error:
                        logger.error(f"💥 Lỗi khi lưu recipes: {recipe_save_error}")
        except Exception as e:
            logger.error(f"💥 Lỗi khi xử lý phản hồi: {str(e)}", exc_info=True)
            result_state['error'] = f"Lỗi khi xử lý phản hồi: {str(e)}"
            result_state['final_response'] = "Xin lỗi, có lỗi xảy ra khi xử lý phản hồi. Vui lòng thử lại sau."

        return result_state

    return asyncio.run(_async_enhanced_response_cleanup())

def define_router(state: ChatState) -> str:
    """
    ⭐ ROUTER CẬP NHẬT: Sử dụng requests_food và requests_beverage để định tuyến chính xác.
    Có logging chi tiết cho các cờ quan trọng.
    """
    # Logging các giá trị cờ quan trọng ngay đầu hàm
    logger.info("🧭 MAIN ROUTER DECISION:")
    logger.info(f"   - is_greeting: {state.get('is_greeting', False)}")
    logger.info(f"   - is_valid_scope: {state.get('is_valid_scope', True)}")
    logger.info(f"   - user_rejected_info: {state.get('user_rejected_info', False)}")
    logger.info(f"   - need_more_info: {state.get('need_more_info', False)}")
    logger.info(f"   - suggest_general_options: {state.get('suggest_general_options', False)}")
    logger.info(f"   - is_food_related: {state.get('is_food_related', False)}")
    logger.info(f"   - requests_food: {state.get('requests_food', False)}")
    logger.info(f"   - requests_beverage: {state.get('requests_beverage', False)}")
    logger.info(f"   - error: {state.get('error', None)}")
    logger.info(f"   - user_message_id_db: {state.get('user_message_id_db', None)}")
    
    # Xử lý trường hợp đặc biệt: tin nhắn chào hỏi
    if state.get("is_greeting", False):
        logger.info("🎯 Router decision: enhanced_response_cleanup (greeting)")
        return "enhanced_response_cleanup"
    
    # Kiểm tra error
    if state.get("error"):
        logger.info("🎯 Router decision: enhanced_response_cleanup (error)")
        return "enhanced_response_cleanup"
    
    # Kiểm tra is_valid_scope
    if not state.get("is_valid_scope"):
        logger.info("🎯 Router decision: enhanced_response_cleanup (invalid scope)")
        return "enhanced_response_cleanup"
    
    # Ưu tiên kiểm tra need_more_info trước
    if state.get("need_more_info") and state.get("follow_up_question"):
        logger.info("🎯 Router decision: collect_info (need more info)")
        return "collect_info"
    
    # ⭐ LOGIC MỚI: Phân nhánh dựa trên requests_food và requests_beverage
    if state.get("need_more_info") == False:  # Đã đủ thông tin hoặc gợi ý chung
        requests_food = state.get("requests_food", False)
        requests_beverage = state.get("requests_beverage", False)
        
        if requests_food and not requests_beverage:
            # Chỉ yêu cầu món ăn
            logger.info("🎯 Router decision: recipe_search (food only)")
            return "recipe_search"
        elif requests_beverage and not requests_food:
            # Chỉ yêu cầu đồ uống
            logger.info("🎯 Router decision: beverage_search (beverage only)")
            return "beverage_search"
        elif requests_food and requests_beverage:
            # Yêu cầu cả hai - ưu tiên recipe_search trước, sẽ xử lý beverage trong enhanced_medichat_call
            logger.info("🎯 Router decision: recipe_search (mixed - prioritize food first)")
            return "recipe_search"
        elif state.get("suggest_general_options", False) and state.get("is_food_related", False):
            # Gợi ý chung về dinh dưỡng - đi qua recipe_search với query chung
            logger.info("🎯 Router decision: recipe_search (general food suggestions)")
            return "recipe_search"
        elif state.get("is_food_related", False):
            # Fallback cho food-related queries
            logger.info("🎯 Router decision: recipe_search (food-related fallback)")
            return "recipe_search"
        else:
            # Không liên quan đến món ăn/đồ uống
            logger.info("🎯 Router decision: store_data (non-food/beverage)")
            return "store_data"
    
    # Fallback cho need_more_info != False
    if state.get("is_food_related", False):
        logger.info("🎯 Router decision: recipe_search (fallback food-related)")
        return "recipe_search"
    else:
        logger.info("🎯 Router decision: store_data (fallback non-food)")
        return "store_data"

# Router sau khi collect_info
def define_post_collect_info_router(state: ChatState) -> str:
    """
    Router sau collect_info để quyết định tiếp tục hay dừng.
    Sử dụng requests_food và requests_beverage.
    """
    # Logging các giá trị cờ quan trọng ngay đầu hàm
    logger.info("🧭 POST-COLLECT-INFO ROUTER DECISION:")
    logger.info(f"   - need_more_info: {state.get('need_more_info', False)}")
    logger.info(f"   - is_food_related: {state.get('is_food_related', False)}")
    logger.info(f"   - requests_food: {state.get('requests_food', False)}")
    logger.info(f"   - requests_beverage: {state.get('requests_beverage', False)}")
    logger.info(f"   - error: {state.get('error', None)}")
    
    # Nếu vẫn cần thêm thông tin hoặc có lỗi
    if state.get("need_more_info", False) or state.get("error"):
        logger.info("🎯 Post-collect router decision: enhanced_response_cleanup (still need info or error)")
        return "enhanced_response_cleanup"
    
    # ⭐ SỬ DỤNG LOGIC TƯƠNG TỰ MAIN ROUTER
    requests_food = state.get("requests_food", False)
    requests_beverage = state.get("requests_beverage", False)
    
    if requests_food and not requests_beverage:
        logger.info("🎯 Post-collect router decision: recipe_search (food only)")
        return "recipe_search"
    elif requests_beverage and not requests_food:
        logger.info("🎯 Post-collect router decision: beverage_search (beverage only)")
        return "beverage_search"
    elif requests_food and requests_beverage:
        logger.info("🎯 Post-collect router decision: recipe_search (mixed - prioritize food first)")
        return "recipe_search"
    elif state.get("is_food_related", False):
        logger.info("🎯 Post-collect router decision: recipe_search (food-related fallback)")
        return "recipe_search"
    else:
        logger.info("🎯 Post-collect router decision: store_data (non-food)")
        return "store_data"

# Khởi tạo StateGraph cho luồng xử lý chat
def create_chat_flow_graph(repository=None, llm_service=None):
    """⭐ CẬP NHẬT: Tạo và cấu hình StateGraph với persist_user_interaction_node mới"""
    # Tạo đồ thị với trạng thái là ChatState
    builder = StateGraph(ChatState)
    
    # Thêm các node cơ bản
    builder.add_node("check_scope", run_async(check_scope_node))
    builder.add_node("persist_user_interaction", lambda state: persist_user_interaction_node_wrapper(state, repository))
    builder.add_node("collect_info", collect_info_node)
    builder.add_node("store_data", lambda state: store_data_node_wrapper(state, repository))
    builder.add_node("medichat_call", lambda state: medichat_call_node_wrapper(state, repository, llm_service))
    builder.add_node("response_cleanup", lambda state: response_cleanup_node_wrapper(state, repository))
    
    # Thêm các node cho food flow 
    builder.add_node("recipe_search", run_async(recipe_search_node))
    builder.add_node("product_search", run_async(product_search_node))
    # ⭐ NODE MỚI: Beverage search
    builder.add_node("beverage_search", run_async(beverage_search_node))
    builder.add_node("enhanced_medichat_call", lambda state: enhanced_medichat_call_node_wrapper(state, repository, llm_service))
    builder.add_node("enhanced_response_cleanup", lambda state: enhanced_response_cleanup_node_wrapper(state, repository))
    
    # ⭐ LUỒNG MỚI: check_scope → persist_user_interaction → router
    builder.set_entry_point("check_scope")
    builder.add_edge("check_scope", "persist_user_interaction")
    
    # Router chạy sau persist_user_interaction để đảm bảo user_message_id_db luôn có
    builder.add_conditional_edges(
        "persist_user_interaction",
        define_router,
        {
            "collect_info": "collect_info",
            "enhanced_response_cleanup": "enhanced_response_cleanup",
            "store_data": "store_data",
            "recipe_search": "recipe_search",
            "beverage_search": "beverage_search"  # ⭐ THÊM EDGE CHO BEVERAGE SEARCH
        }
    )
    
    # Cấu hình cạnh từ collect_info
    builder.add_conditional_edges(
        "collect_info",
        define_post_collect_info_router,
        {
            "enhanced_response_cleanup": "enhanced_response_cleanup",
            "recipe_search": "recipe_search",
            "beverage_search": "beverage_search",  # ⭐ THÊM EDGE CHO BEVERAGE SEARCH
            "store_data": "store_data"
        }
    )
    
    # Food/Beverage flow sequences
    builder.add_edge("recipe_search", "enhanced_medichat_call")
    builder.add_edge("beverage_search", "enhanced_medichat_call")  # ⭐ BEVERAGE → MEDICHAT
    builder.add_edge("enhanced_medichat_call", "product_search")
    builder.add_edge("product_search", "enhanced_response_cleanup")
    
    # Cấu hình cạnh từ store_data đến các node tiếp theo
    def store_data_router(state: ChatState) -> str:
        """Router sau store_data để quyết định luồng tiếp theo"""
        logger.info("🧭 STORE_DATA ROUTER DECISION:")
        logger.info(f"   - is_greeting: {state.get('is_greeting', False)}")
        logger.info(f"   - is_food_related: {state.get('is_food_related', False)}")
        
        if state.get("is_greeting", False):
            logger.info("🎯 Store_data router: enhanced_response_cleanup (greeting)")
            return "enhanced_response_cleanup"
        elif state.get("is_food_related", False):
            logger.info("🎯 Store_data router: recipe_search (food-related)")
            return "recipe_search"
        else:
            logger.info("🎯 Store_data router: medichat_call (non-food)")
            return "medichat_call"
    
    builder.add_conditional_edges(
        "store_data",
        store_data_router,
        {
            "medichat_call": "medichat_call",
            "enhanced_response_cleanup": "enhanced_response_cleanup",
            "recipe_search": "recipe_search"
        }
    )
    
    # Cấu hình cạnh từ medichat_call đến response_cleanup
    builder.add_edge("medichat_call", "response_cleanup")
    
    # Cấu hình điểm kết thúc
    builder.add_edge("response_cleanup", END)
    builder.add_edge("enhanced_response_cleanup", END)
    
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
    """Chạy luồng xử lý chat và trả về kết quả với đầy đủ message IDs"""
    try:
        # Kiểm tra giới hạn số tin nhắn trong phiên trò chuyện
        limits = await check_conversation_limits(conversation_id, repository)
        if limits["limit_reached"]:
            # Đã đạt giới hạn 30 tin nhắn
            # Không cần lưu tin nhắn người dùng vì đã được lưu ở API trước đó
            limit_message = "Bạn đã đạt đến giới hạn 30 tin nhắn trong phiên trò chuyện này. Vui lòng bắt đầu một phiên mới để tiếp tục."
            assistant_message_db_obj = repository.add_message(conversation_id, "assistant", limit_message)
            
            logger.info(f"⚠️ Đã đạt giới hạn 30 tin nhắn trong phiên trò chuyện {conversation_id}")
            
            return {
                "conversation_id": conversation_id,
                "user_message": {"role": "user", "content": user_message},
                "assistant_message": {"role": "assistant", "content": limit_message},
                "is_valid_scope": True,
                "need_more_info": False,
                "final_response": limit_message,
                "limit_reached": True,
                "message_count": limits["message_count"],
                "assistant_message_id_db": assistant_message_db_obj.message_id
            }
        
        # Khởi tạo trạng thái với đầy đủ các trường
        state = ChatState(
            conversation_id=conversation_id,
            user_id=user_id,
            user_message=user_message,
            messages=messages,
            is_valid_scope=True,
            is_greeting=False,
            is_food_related=False,
            user_rejected_info=False,
            need_more_info=False,
            suggest_general_options=False,
            follow_up_question=None,
            collected_info={},
            medichat_prompt=None,
            medichat_response=None,
            recipe_results=None,
            beverage_results=None,  # ⭐ KHỞI TẠO BEVERAGE_RESULTS
            product_results=None,
            final_response=None,
            error=None,
            user_message_id_db=None,
            assistant_message_id_db=None,
            requests_food=None,      # ⭐ KHỞI TẠO REQUESTS_FOOD
            requests_beverage=None   # ⭐ KHỞI TẠO REQUESTS_BEVERAGE
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
                result["final_response"] = ("Xin lỗi, câu hỏi của bạn nằm ngoài phạm vi tư vấn của tôi. "
                                          "Tôi chỉ có thể hỗ trợ về các vấn đề liên quan đến dinh dưỡng, sức khỏe, "
                                          "món ăn và đồ uống.")
            else:
                result["final_response"] = "Xin lỗi, tôi không thể xử lý yêu cầu của bạn. Vui lòng thử lại."
        
        # Đảm bảo user_message luôn là dictionary
        if "user_message" not in result or not isinstance(result["user_message"], dict):
            result["user_message"] = {"role": "user", "content": user_message}
        
        # Đảm bảo có assistant_message cho API trả về và luôn là dictionary
        if not result.get("assistant_message") or not isinstance(result["assistant_message"], dict):
            result["assistant_message"] = {
                "role": "assistant",
                "content": result.get("final_response", "")
            }
        
        # Log kết quả với message IDs
        logger.info("🎯 CHAT FLOW RESULT:")
        logger.info(f"   - user_message_id_db: {result.get('user_message_id_db')}")
        logger.info(f"   - assistant_message_id_db: {result.get('assistant_message_id_db')}")
        logger.info(f"   - is_valid_scope: {result.get('is_valid_scope')}")
        logger.info(f"   - suggest_general_options: {result.get('suggest_general_options')}")
        logger.info(f"   - final_response length: {len(result.get('final_response', ''))}")
        
        return result
        
    except Exception as e:
        logger.error(f"💥 Lỗi nghiêm trọng trong run_chat_flow: {str(e)}", exc_info=True)
        return {
            "conversation_id": conversation_id,
            "user_message": {"role": "user", "content": user_message},
            "assistant_message": {"role": "assistant", "content": "Xin lỗi, có lỗi xảy ra trong hệ thống. Vui lòng thử lại sau."},
            "is_valid_scope": False,
            "need_more_info": False,
            "final_response": "Xin lỗi, có lỗi xảy ra trong hệ thống. Vui lòng thử lại sau.",
            "error": str(e),
            "user_message_id_db": None,
            "assistant_message_id_db": None
        }

async def check_conversation_limits(conversation_id: int, repository) -> Dict[str, Any]:
    """Kiểm tra giới hạn số tin nhắn trong cuộc trò chuyện"""
    try:
        messages = repository.get_messages(conversation_id)
        message_count = len(messages)
        limit_reached = message_count >= settings.MAX_HISTORY_MESSAGES
        
        return {
            "message_count": message_count,
            "limit_reached": limit_reached,
            "max_messages": settings.MAX_HISTORY_MESSAGES
        }
    except Exception as e:
        logger.error(f"💥 Lỗi khi kiểm tra giới hạn tin nhắn: {str(e)}")
        return {
            "message_count": 0,
            "limit_reached": False,
            "max_messages": settings.MAX_HISTORY_MESSAGES
        } 