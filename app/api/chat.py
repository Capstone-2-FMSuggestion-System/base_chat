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
from app.middleware.auth import get_verified_user_from_backend, VerifiedUserInfo
from app.schemas.chat import ChatRequest, ChatResponse, NewChatResponse, ChatContentResponse
from app.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    verified_user: VerifiedUserInfo = Depends(get_verified_user_from_backend),
    db: Session = Depends(get_db)
):
    """Gửi tin nhắn và nhận phản hồi từ AI với background DB operations"""
    chat_service = ChatService(db)
    
    # Kiểm tra quyền truy cập conversation nếu có
    if request.conversation_id:
        if not chat_service.repository.is_user_owner_of_conversation(verified_user.user_id, request.conversation_id):
            raise HTTPException(status_code=403, detail="Không có quyền truy cập cuộc trò chuyện này")
    
    # Xử lý message với background DB operations
    result, background_task_ids = await chat_service.process_message_with_background(
        user_id=verified_user.user_id,
        message=request.message,
        conversation_id=request.conversation_id
    )
    
    # Thêm background tasks để execute các DB operations
    if background_task_ids:
        background_tasks.add_task(chat_service.execute_background_tasks, background_task_ids)
        logger.info(f"🚀 Đã thêm {len(background_task_ids)} background DB tasks")
    
    return result


@router.post("/stream-chat")
async def stream_chat(
    request: ChatRequest,
    verified_user: VerifiedUserInfo = Depends(get_verified_user_from_backend),
    db: Session = Depends(get_db)
):
    """Gửi tin nhắn và nhận phản hồi từ AI dạng streaming"""
    chat_service = ChatService(db)
    
    logger = logging.getLogger(__name__)
    
    if not request.conversation_id:
        conversation = chat_service.repository.get_latest_conversation(verified_user.user_id)
        if not conversation:
            conversation = chat_service.repository.create_conversation(verified_user.user_id)
            welcome_message = await chat_service.gemini_service.generate_welcome_message()
            chat_service.repository.add_message(conversation.conversation_id, "assistant", welcome_message)
            async def welcome_generator():
                yield f"data: {welcome_message}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(welcome_generator(), media_type="text/event-stream")
        conversation_id = conversation.conversation_id
    else:
        if not chat_service.repository.is_user_owner_of_conversation(verified_user.user_id, request.conversation_id):
            raise HTTPException(status_code=403, detail="Không có quyền truy cập cuộc trò chuyện này")
        conversation = chat_service.repository.get_conversation_by_id(request.conversation_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Cuộc trò chuyện không tồn tại")
        conversation_id = request.conversation_id
    
    chat_service.repository.add_message(conversation_id, "user", request.message)
    logger.info(f"Đã lưu tin nhắn người dùng trước khi xử lý: {request.message[:50]}...")
    
    chat_history = chat_service.repository.get_messages(conversation_id, limit=10)
    
    async def response_generator():
        special_tokens = ["<|im_ending|>", "<|im_start|>"]
        full_response = ""
        error_indicators = ["Xin lỗi, tôi đang có vấn đề", "Xin lỗi, hiện tôi không thể kết nối"]
        
        try:
            if chat_service.llm_service._active_service is None:
                active_service = await chat_service.llm_service.initialize()
                yield f"data: {{\"info\": \"Đã kết nối thành công đến dịch vụ: {active_service}\"}}\n\n"
            else:
                active_service = chat_service.llm_service._active_service.value
                yield f"data: {{\"info\": \"Đang sử dụng dịch vụ: {active_service}\"}}\n\n"
            
            try:
                result = await run_chat_flow(
                    user_message=request.message,
                    user_id=verified_user.user_id,
                    conversation_id=conversation_id,
                    messages=chat_history,
                    repository=chat_service.repository,
                    llm_service=chat_service.llm_service
                )
            except Exception as e:
                logger.error(f"Lỗi khi chạy LangGraph flow: {str(e)}")
                chat_service.repository.add_message(conversation_id, "user", request.message)
                error_message = "Xin lỗi, có lỗi xử lý trong hệ thống. Tôi sẽ ghi nhận câu hỏi của bạn và trả lời sau."
                yield f"data: {error_message}\n\n"
                yield "data: [DONE]\n\n"
                return
            
            if "final_response" in result and result["final_response"]:
                response = result["final_response"]
                yield f"data: {response}\n\n"
                yield "data: [DONE]\n\n"
                return
            
            if "need_more_info" in result and result["need_more_info"] and "assistant_message" in result:
                follow_up = result["assistant_message"]["content"]
                yield f"data: {follow_up}\n\n"
                yield "data: [DONE]\n\n"
                return
            
            medichat_prompt = result.get("medichat_prompt")
            if not medichat_prompt:
                medichat_prompt = await chat_service.gemini_service.create_medichat_prompt(messages=chat_history)
            
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
            
            logger.info(f"Đang gửi prompt đến Medichat: {medichat_prompt}")
            async for chunk in chat_service.llm_service.generate_response(medichat_messages):
                logger.debug(f"Nhận được chunk: {chunk}")
                
                for token in special_tokens:
                    chunk = chunk.replace(token, "")
                
                if chunk.strip():
                    full_response += chunk
                    yield f"data: {chunk}\n\n"
            
            is_error_response = any(indicator in full_response for indicator in error_indicators)
            if is_error_response:
                logger.warning(f"Phát hiện thông báo lỗi trong phản hồi: {full_response}")
                for indicator in error_indicators:
                    if indicator in full_response:
                        error_end_pos = full_response.find(".", full_response.find(indicator)) + 1
                        if error_end_pos > 0:
                            full_response = full_response[:error_end_pos].strip()
                            break
            
            if full_response.strip() and not is_error_response:
                polished_response = await chat_service.gemini_service.polish_response(full_response, medichat_prompt)
                
                chat_service.repository.add_message(conversation_id, "assistant", polished_response)
                logger.info(f"Đã lưu phản hồi đã điều chỉnh: {polished_response[:50]}...")
                
                if polished_response != full_response:
                    yield f"data: {{\"replace\": \"{json.dumps(polished_response)}\"}}\n\n"
            else:
                if not full_response.strip():
                    error_message = "Xin lỗi, không nhận được phản hồi từ hệ thống AI. Vui lòng thử lại sau."
                else:
                    error_message = full_response
                
                yield f"data: {error_message}\n\n"
                chat_service.repository.add_message(conversation_id, "assistant", error_message)
                logger.warning(f"Đã lưu thông báo lỗi: {error_message}")
                
        except Exception as e:
            logger.error(f"Lỗi khi xử lý phản hồi: {str(e)}")
            error_message = "Xin lỗi, hiện tôi không thể kết nối tới hệ thống trí tuệ nhân tạo. Vui lòng thử lại sau."
            yield f"data: {error_message}\n\n"
            
            chat_service.repository.add_message(conversation_id, "assistant", error_message)
        
        yield "data: [DONE]\n\n"
    
    return StreamingResponse(
        response_generator(),
        media_type="text/event-stream"
    )


@router.post("/newChat", response_model=NewChatResponse)
async def new_chat(
    verified_user: VerifiedUserInfo = Depends(get_verified_user_from_backend),
    db: Session = Depends(get_db)
):
    """Tạo cuộc trò chuyện mới"""
    chat_service = ChatService(db)
    result = chat_service.create_new_chat(verified_user.user_id)
    return result


@router.get("/chatContent", response_model=ChatContentResponse)
async def get_chat_content(
    conversation_id: Optional[int] = None,
    verified_user: VerifiedUserInfo = Depends(get_verified_user_from_backend),
    db: Session = Depends(get_db)
):
    """Lấy nội dung cuộc trò chuyện"""
    chat_service = ChatService(db)
    result = chat_service.get_chat_content(verified_user.user_id, conversation_id)
    return result


@router.get("/background-task-status/{task_id}")
async def get_background_task_status(
    task_id: str,
    verified_user: VerifiedUserInfo = Depends(get_verified_user_from_backend),
    db: Session = Depends(get_db)
):
    """
    Lấy trạng thái của background task.
    Endpoint này dùng cho monitoring/debugging.
    """
    chat_service = ChatService(db)
    status = chat_service.get_background_task_status(task_id)
    
    if status is None:
        raise HTTPException(status_code=404, detail="Task không tồn tại")
        
    return {
        "task_id": task_id,
        "status": status["status"],
        "details": status
    } 