import logging
from sqlalchemy.orm import Session
from typing import Dict, List, Optional, Any
from fastapi import HTTPException
import asyncio
from datetime import datetime

from app.repositories.chat_repository import ChatRepository
from app.services.llm_service_factory import LLMServiceFactory
from app.services.summary_service import SummaryService
from app.config import settings
from app.db.models import Message


logger = logging.getLogger(__name__)


class ChatService:
    def __init__(self, db: Session):
        self.db = db
        self.repository = ChatRepository(db)
        self.llm_service = LLMServiceFactory(
            service_type=settings.LLM_SERVICE_TYPE,
            llama_url=settings.LLAMA_CPP_URL,
            ollama_url=settings.OLLAMA_URL
        )
        self.summary_service = SummaryService()
    
    async def process_message(self, user_id: int, message: str, conversation_id: Optional[int] = None) -> Dict[str, Any]:
        """Xử lý tin nhắn từ người dùng và trả về phản hồi từ AI"""
        
        # Nếu không có conversation_id, lấy cuộc trò chuyện mới nhất hoặc tạo mới
        if not conversation_id:
            conversation = self.repository.get_latest_conversation(user_id)
            if not conversation:
                conversation = self.repository.create_conversation(user_id)
            conversation_id = conversation.conversation_id
        else:
            # Kiểm tra người dùng có quyền truy cập cuộc trò chuyện không
            if not self.repository.is_user_owner_of_conversation(user_id, conversation_id):
                raise HTTPException(status_code=403, detail="Không có quyền truy cập cuộc trò chuyện này")
            
            # Kiểm tra xem cuộc trò chuyện có tồn tại không
            conversation = self.repository.get_conversation_by_id(conversation_id)
            if not conversation:
                raise HTTPException(status_code=404, detail="Cuộc trò chuyện không tồn tại")
        
        # Lưu tin nhắn của người dùng
        user_message = self.repository.add_message(conversation_id, "user", message)
        
        # Kiểm tra xem có cần tạo tóm tắt mới không
        await self._check_and_create_summary(conversation_id, user_message.message_id)
        
        # Lấy lịch sử trò chuyện với tóm tắt nếu cần
        messages = self.repository.get_messages_with_summary(conversation_id)
        
        try:
            # Khởi tạo service nếu cần
            if self.llm_service._active_service is None:
                await self.llm_service.initialize()
                
            # Lấy phản hồi từ AI
            ai_response = await self.llm_service.get_full_response(messages)
            
            # Lưu phản hồi từ AI
            self.repository.add_message(conversation_id, "assistant", ai_response)
            
            return {
                "conversation_id": conversation_id,
                "user_message": {"role": "user", "content": message},
                "assistant_message": {"role": "assistant", "content": ai_response}
            }
        except Exception as e:
            logger.error(f"Lỗi khi xử lý tin nhắn: {str(e)}")
            # Lưu thông báo lỗi thay vì phản hồi từ AI
            error_message = "Xin lỗi, hiện tôi không thể kết nối tới hệ thống trí tuệ nhân tạo. Vui lòng thử lại sau."
            self.repository.add_message(conversation_id, "assistant", error_message)
            
            return {
                "conversation_id": conversation_id,
                "user_message": {"role": "user", "content": message},
                "assistant_message": {"role": "assistant", "content": error_message}
            }
    
    async def _check_and_create_summary(self, conversation_id: int, message_id: int) -> None:
        """
        Kiểm tra xem cuộc trò chuyện có cần tạo tóm tắt mới không và tạo tóm tắt nếu cần
        
        Args:
            conversation_id: ID của cuộc trò chuyện
            message_id: ID của tin nhắn người dùng mới nhất
        """
        # Đếm số lượng tin nhắn chưa được tóm tắt
        unsummarized_count = self.db.query(Message).filter(
            Message.conversation_id == conversation_id,
            Message.is_summarized == False,
            Message.role != "system"
        ).count()
        
        # Đếm tổng số tin nhắn non-system
        total_messages = self.db.query(Message).filter(
            Message.conversation_id == conversation_id,
            Message.role != "system"
        ).count()
        
        logger.debug(f"Kiểm tra tóm tắt: total={total_messages}, unsummarized={unsummarized_count}")
        
        # Nếu có đủ tin nhắn chưa được tóm tắt, tạo tóm tắt mới
        if (unsummarized_count >= settings.SUMMARY_THRESHOLD or 
            total_messages >= settings.MAX_HISTORY_MESSAGES):
            # Tạo tóm tắt mới
            await self._create_message_summary(conversation_id, message_id)
    
    async def _create_message_summary(self, conversation_id: int, message_id: int) -> None:
        """
        Tạo tóm tắt cho tin nhắn mới nhất của cuộc trò chuyện
        
        Args:
            conversation_id: ID của cuộc trò chuyện
            message_id: ID của tin nhắn cần tóm tắt
        """
        try:
            # Lấy tất cả tin nhắn của cuộc trò chuyện (không giới hạn số lượng)
            messages = self.repository.get_messages(conversation_id, limit=0)
            
            if not messages:
                logger.warning(f"Không có tin nhắn nào để tóm tắt cho cuộc trò chuyện ID={conversation_id}")
                return
                
            # Tạo tóm tắt bằng Gemini API
            summary = await self.summary_service.summarize_conversation(messages)
            
            if not summary:
                logger.warning(f"Không tạo được tóm tắt cho tin nhắn ID={message_id}")
                return
                
            # Cập nhật tóm tắt vào cơ sở dữ liệu
            self.repository.update_message_summary(message_id, summary)
            logger.info(f"Đã tóm tắt thành công cho tin nhắn ID={message_id}: {summary[:50]}...")
            
        except Exception as e:
            logger.error(f"Lỗi khi tạo tóm tắt cho tin nhắn ID={message_id}: {str(e)}")
    
    def create_new_chat(self, user_id: int) -> Dict[str, Any]:
        """Tạo cuộc trò chuyện mới"""
        conversation = self.repository.create_conversation(user_id)
        return {
            "conversation_id": conversation.conversation_id,
            "created_at": conversation.created_at
        }
    
    def get_chat_content(self, user_id: int, conversation_id: Optional[int] = None) -> Dict[str, Any]:
        """Lấy nội dung cuộc trò chuyện"""
        if conversation_id:
            # Kiểm tra người dùng có quyền truy cập cuộc trò chuyện không
            if not self.repository.is_user_owner_of_conversation(user_id, conversation_id):
                raise HTTPException(status_code=403, detail="Không có quyền truy cập cuộc trò chuyện này")
            
            # Kiểm tra xem cuộc trò chuyện có tồn tại không
            conversation = self.repository.get_conversation_by_id(conversation_id)
            if not conversation:
                raise HTTPException(status_code=404, detail="Cuộc trò chuyện không tồn tại")
        else:
            # Lấy cuộc trò chuyện mới nhất
            conversation = self.repository.get_latest_conversation(user_id)
            
            if not conversation:
                # Nếu chưa có cuộc trò chuyện nào, tạo mới
                conversation = self.repository.create_conversation(user_id)
                messages = []
                return {
                    "conversation_id": conversation.conversation_id,
                    "messages": messages
                }
        
        # Lấy tin nhắn từ cuộc trò chuyện hiện có (giới hạn tin nhắn)
        messages = self.repository.get_messages(conversation.conversation_id)
        
        # Tìm tin nhắn gần nhất có summary
        summarized_message = self.db.query(Message).filter(
            Message.conversation_id == conversation.conversation_id,
            Message.summary.isnot(None),
            Message.summary != ""
        ).order_by(Message.created_at.desc()).first()
        
        # Thêm thông tin tóm tắt nếu có
        result = {
            "conversation_id": conversation.conversation_id,
            "messages": messages
        }
        
        if summarized_message:
            result["has_summary"] = True
            result["summary"] = summarized_message.summary
            result["summary_updated_at"] = summarized_message.created_at
        else:
            result["has_summary"] = False
        
        return result 