from fastapi import APIRouter, Depends, HTTPException, Body, Query, BackgroundTasks
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session
from fastapi.responses import StreamingResponse
import logging
import json

from app.db.database import get_db
from app.services.chat_service import ChatService
from app.services.chat_flow import run_chat_flow
# from app.middleware.auth import get_current_user_id
from app.schemas.chat import ChatRequest, ChatResponse, NewChatResponse, ChatContentResponse
from app.config import settings


router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    # user_id: int = Depends(get_current_user_id),
    user_id: int = 1,  # Giá trị mặc định để test
    db: Session = Depends(get_db)
):
    """Gửi tin nhắn và nhận phản hồi từ AI"""
    chat_service = ChatService(db)
    
    # Lưu tin nhắn người dùng trước khi xử lý
    if request.conversation_id:
        chat_service.repository.add_message(request.conversation_id, "user", request.message)
        logger.info(f"Đã lưu tin nhắn người dùng trước khi xử lý: {request.message[:50]}...")
    
    result = await chat_service.process_message(
        user_id=user_id,
        message=request.message,
        conversation_id=request.conversation_id
    )
    
    return result


@router.post("/stream-chat")
async def stream_chat(
    request: ChatRequest,
    # user_id: int = Depends(get_current_user_id),
    user_id: int = 1,  # Giá trị mặc định để test
    db: Session = Depends(get_db)
):
    """Gửi tin nhắn và nhận phản hồi từ AI dạng streaming"""
    chat_service = ChatService(db)
    
    # Khởi tạo logger ở phạm vi rộng hơn để có thể truy cập từ khối try và except
    logger = logging.getLogger(__name__)
    
    # Lưu tin nhắn người dùng và chuẩn bị cuộc trò chuyện
    # Nếu không có conversation_id, lấy cuộc trò chuyện mới nhất hoặc tạo mới
    if not request.conversation_id:
        conversation = chat_service.repository.get_latest_conversation(user_id)
        if not conversation:
            conversation = chat_service.repository.create_conversation(user_id)
            # Thêm lời chào khi bắt đầu cuộc trò chuyện mới
            welcome_message = await chat_service.gemini_service.generate_welcome_message()
            chat_service.repository.add_message(conversation.conversation_id, "assistant", welcome_message)
            # Stream lời chào ngay lập tức
            async def welcome_generator():
                yield f"data: {welcome_message}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(welcome_generator(), media_type="text/event-stream")
        conversation_id = conversation.conversation_id
    else:
        # Kiểm tra người dùng có quyền truy cập cuộc trò chuyện không
        if not chat_service.repository.is_user_owner_of_conversation(user_id, request.conversation_id):
            raise HTTPException(status_code=403, detail="Không có quyền truy cập cuộc trò chuyện này")
        
        # Kiểm tra xem cuộc trò chuyện có tồn tại không
        conversation = chat_service.repository.get_conversation_by_id(request.conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Cuộc trò chuyện không tồn tại")
        conversation_id = request.conversation_id
    
    # Lưu tin nhắn người dùng vào database ngay từ đầu
    chat_service.repository.add_message(conversation_id, "user", request.message)
    logger.info(f"Đã lưu tin nhắn người dùng trước khi xử lý: {request.message[:50]}...")
    
    # Lấy lịch sử trò chuyện hiện tại để phân tích
    chat_history = chat_service.repository.get_messages(conversation_id, limit=10)
    
    # Hàm generator để stream phản hồi
    async def response_generator():
        special_tokens = ["<|im_ending|>", "<|im_start|>"]
        full_response = ""
        error_indicators = ["Xin lỗi, tôi đang có vấn đề", "Xin lỗi, hiện tôi không thể kết nối"]
        
        try:
            # Khởi tạo service nếu cần
            if chat_service.llm_service._active_service is None:
                active_service = await chat_service.llm_service.initialize()
                yield f"data: {{\"info\": \"Đã kết nối thành công đến dịch vụ: {active_service}\"}}\n\n"
            else:
                active_service = chat_service.llm_service._active_service.value
                yield f"data: {{\"info\": \"Đang sử dụng dịch vụ: {active_service}\"}}\n\n"
            
            try:
                # Chạy luồng xử lý với LangGraph
                result = await run_chat_flow(
                    user_message=request.message,
                    user_id=user_id,
                    conversation_id=conversation_id,
                    messages=chat_history,
                    repository=chat_service.repository,
                    llm_service=chat_service.llm_service
                )
            except Exception as e:
                logger.error(f"Lỗi khi chạy LangGraph flow: {str(e)}")
                # Fallback - lưu tin nhắn người dùng
                chat_service.repository.add_message(conversation_id, "user", request.message)
                error_message = "Xin lỗi, có lỗi xử lý trong hệ thống. Tôi sẽ ghi nhận câu hỏi của bạn và trả lời sau."
                yield f"data: {error_message}\n\n"
                yield "data: [DONE]\n\n"
                return
            
            # Xử lý kết quả từ LangGraph
            if "final_response" in result and result["final_response"]:
                # Phản hồi cuối cùng đã có sẵn trong result
                response = result["final_response"]
                yield f"data: {response}\n\n"
                yield "data: [DONE]\n\n"
                return
            
            # Các trường hợp xử lý khác nếu final_response không có
            if "need_more_info" in result and result["need_more_info"] and "assistant_message" in result:
                # Nếu cần thu thập thêm thông tin
                follow_up = result["assistant_message"]["content"]
                yield f"data: {follow_up}\n\n"
                yield "data: [DONE]\n\n"
                return
            
            # Nếu không, stream với Medichat như thông thường
            # Trường hợp này xảy ra nếu LangGraph chuyển việc tạo phản hồi cho Medichat
            # Lấy prompt từ LangGraph hoặc tạo mới
            medichat_prompt = result.get("medichat_prompt")
            if not medichat_prompt:
                medichat_prompt = await chat_service.gemini_service.create_medichat_prompt(messages=chat_history)
            
            # Tạo tin nhắn cho Medichat
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
            
            # Stream từng phần phản hồi
            logger.info(f"Đang gửi prompt đến Medichat: {medichat_prompt}")
            async for chunk in chat_service.llm_service.generate_response(medichat_messages):
                # Log từng chunk
                logger.debug(f"Nhận được chunk: {chunk}")
                
                # Loại bỏ các token đặc biệt từ chunk
                for token in special_tokens:
                    chunk = chunk.replace(token, "")
                
                if chunk.strip():  # Chỉ xử lý các chunk có nội dung sau khi lọc
                    full_response += chunk
                    yield f"data: {chunk}\n\n"
            
            # Kiểm tra xem phản hồi có phải là thông báo lỗi không
            is_error_response = any(indicator in full_response for indicator in error_indicators)
            if is_error_response:
                logger.warning(f"Phát hiện thông báo lỗi trong phản hồi: {full_response}")
                # Nếu còn có thêm nội dung y tế sau thông báo lỗi, tách và chỉ giữ lại phần đầu
                for indicator in error_indicators:
                    if indicator in full_response:
                        error_end_pos = full_response.find(".", full_response.find(indicator)) + 1
                        if error_end_pos > 0:
                            full_response = full_response[:error_end_pos].strip()
                            break
            
            # Sau khi hoàn thành, điều chỉnh phản hồi bằng Gemini và lưu vào database
            if full_response.strip() and not is_error_response:
                # Điều chỉnh phản hồi
                polished_response = await chat_service.gemini_service.polish_response(full_response, medichat_prompt)
                
                # Lưu phản hồi đã điều chỉnh vào database
                chat_service.repository.add_message(conversation_id, "assistant", polished_response)
                logger.info(f"Đã lưu phản hồi đã điều chỉnh: {polished_response[:50]}...")
                
                # Gửi phản hồi đã điều chỉnh nếu khác với full_response
                if polished_response != full_response:
                    yield f"data: {{\"replace\": \"{json.dumps(polished_response)}\"}}\n\n"
            else:
                if not full_response.strip():
                    error_message = "Xin lỗi, không nhận được phản hồi từ hệ thống AI. Vui lòng thử lại sau."
                else:
                    # Nếu là lỗi đã được phát hiện, sử dụng thông báo đó
                    error_message = full_response
                
                yield f"data: {error_message}\n\n"
                chat_service.repository.add_message(conversation_id, "assistant", error_message)
                logger.warning(f"Đã lưu thông báo lỗi: {error_message}")
                
        except Exception as e:
            # Xử lý lỗi, trả về thông báo lỗi
            logger.error(f"Lỗi khi xử lý phản hồi: {str(e)}")
            error_message = "Xin lỗi, hiện tôi không thể kết nối tới hệ thống trí tuệ nhân tạo. Vui lòng thử lại sau."
            yield f"data: {error_message}\n\n"
            
            # Lưu thông báo lỗi vào database
            chat_service.repository.add_message(conversation_id, "assistant", error_message)
        
        # Gửi event hoàn thành
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        response_generator(),
        media_type="text/event-stream"
    )


@router.post("/newChat", response_model=NewChatResponse)
async def new_chat(
    # user_id: int = Depends(get_current_user_id),
    user_id: int = 1,  # Giá trị mặc định để test
    db: Session = Depends(get_db)
):
    """Tạo cuộc trò chuyện mới"""
    chat_service = ChatService(db)
    result = chat_service.create_new_chat(user_id)
    return result


@router.get("/chatContent", response_model=ChatContentResponse)
async def get_chat_content(
    conversation_id: Optional[int] = None,
    # user_id: int = Depends(get_current_user_id),
    user_id: int = 1,  # Giá trị mặc định để test
    db: Session = Depends(get_db)
):
    """Lấy nội dung cuộc trò chuyện"""
    chat_service = ChatService(db)
    result = chat_service.get_chat_content(user_id, conversation_id)
    return result 