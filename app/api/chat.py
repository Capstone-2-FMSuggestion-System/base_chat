from fastapi import APIRouter, Depends, HTTPException, Body, Query, BackgroundTasks
from typing import Dict, Any, Optional
from sqlalchemy.orm import Session
from fastapi.responses import StreamingResponse
import logging

from app.db.database import get_db
from app.services.chat_service import ChatService
# from app.middleware.auth import get_current_user_id
from app.schemas.chat import ChatRequest, ChatResponse, NewChatResponse, ChatContentResponse


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
    
    # Lưu tin nhắn người dùng và chuẩn bị cuộc trò chuyện
    # Nếu không có conversation_id, lấy cuộc trò chuyện mới nhất hoặc tạo mới
    if not request.conversation_id:
        conversation = chat_service.repository.get_latest_conversation(user_id)
        if not conversation:
            conversation = chat_service.repository.create_conversation(user_id)
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
    
    # Lưu tin nhắn của người dùng
    chat_service.repository.add_message(conversation_id, "user", request.message)
    
    # Lấy lịch sử trò chuyện đã được tóm tắt (nếu cần)
    messages = chat_service.repository.get_messages_with_summary(conversation_id)
    
    # Khởi tạo logger ở phạm vi rộng hơn để có thể truy cập từ khối try và except
    logger = logging.getLogger(__name__)
    
    # Hàm generator để stream phản hồi
    async def response_generator():
        full_response = ""
        special_tokens = ["<|im_ending|>", "<|im_start|>"]
        error_indicators = ["Xin lỗi, tôi đang có vấn đề", "Xin lỗi, hiện tôi không thể kết nối"]
        
        try:
            # Khởi tạo service nếu cần
            if chat_service.llm_service._active_service is None:
                active_service = await chat_service.llm_service.initialize()
                yield f"data: {{\"info\": \"Đã kết nối thành công đến dịch vụ: {active_service}\"}}\n\n"
            else:
                active_service = chat_service.llm_service._active_service.value
                yield f"data: {{\"info\": \"Đang sử dụng dịch vụ: {active_service}\"}}\n\n"
            
            # Log tin nhắn gửi đi để debug
            logger.info(f"Đang gửi tin nhắn đến LLM: {request.message}")
            logger.info(f"Số lượng tin nhắn trong cuộc trò chuyện: {len(messages)}")
            
            # Stream từng phần phản hồi
            async for chunk in chat_service.llm_service.generate_response(messages):
                # Log từng chunk để debug
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
            
            # Sau khi hoàn thành, lưu phản hồi vào database
            if full_response.strip() and not is_error_response:  # Kiểm tra xem có phản hồi không và không phải lỗi
                chat_service.repository.add_message(conversation_id, "assistant", full_response)
                logger.info(f"Đã lưu phản hồi thành công: {full_response[:50]}...")
            else:
                if not full_response.strip():
                    error_message = "Xin lỗi, không nhận được phản hồi từ hệ thống AI. Vui lòng thử lại sau."
                else:
                    # Nếu là lỗi đã được phát hiện, sử dụng thông báo đó
                    error_message = full_response
                
                yield f"data: {error_message}\n\n"
                chat_service.repository.add_message(conversation_id, "assistant", error_message)
                logger.warning(f"Đã lưu thông báo lỗi: {error_message}")
                full_response = error_message
        except Exception as e:
            # Xử lý lỗi, trả về thông báo lỗi
            logger.error(f"Lỗi khi xử lý phản hồi: {str(e)}")
            error_message = "Xin lỗi, hiện tôi không thể kết nối tới hệ thống trí tuệ nhân tạo. Vui lòng thử lại sau."
            yield f"data: {error_message}\n\n"
            
            # Lưu thông báo lỗi vào database
            chat_service.repository.add_message(conversation_id, "assistant", error_message)
            full_response = error_message
        
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