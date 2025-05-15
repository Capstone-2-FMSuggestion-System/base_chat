import json
import logging
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from sqlalchemy import desc, func

from app.db.models import Conversation, Message, User
from app.db.database import redis_client
from app.config import settings


logger = logging.getLogger(__name__)


class ChatRepository:
    def __init__(self, db: Session):
        self.db = db
    
    def create_conversation(self, user_id: int, title: str = None) -> Conversation:
        """Tạo cuộc trò chuyện mới"""
        conversation = Conversation(user_id=user_id, title=title)
        self.db.add(conversation)
        self.db.commit()
        self.db.refresh(conversation)
        
        # Lưu conversation mới vào Redis
        self._cache_conversation_metadata(conversation)
        
        return conversation
    
    def add_message(self, conversation_id: int, role: str, content: str) -> Message:
        """Thêm tin nhắn mới vào cuộc trò chuyện"""
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content
        )
        self.db.add(message)
        
        # Cập nhật thời gian cập nhật cuộc trò chuyện
        conversation = self.get_conversation_by_id(conversation_id)
        if conversation:
            conversation.updated_at = datetime.now()
        
        self.db.commit()
        self.db.refresh(message)
        
        # Cập nhật cache
        self._update_conversation_cache(conversation_id)
        
        return message
    
    def update_message_summary(self, message_id: int, summary: str) -> bool:
        """Cập nhật tóm tắt cho tin nhắn"""
        message = self.db.query(Message).filter(Message.message_id == message_id).first()
        if not message:
            logger.error(f"Không tìm thấy tin nhắn ID={message_id} để cập nhật tóm tắt")
            return False
            
        try:
            message.summary = summary
            message.is_summarized = True
            
            self.db.commit()
            
            # Cập nhật cache
            self._update_conversation_cache(message.conversation_id)
            
            logger.info(f"Đã cập nhật tóm tắt cho tin nhắn ID={message_id}")
            return True
            
        except Exception as e:
            logger.error(f"Lỗi khi cập nhật tóm tắt: {str(e)}")
            self.db.rollback()
            return False
    
    def get_conversation_by_id(self, conversation_id: int) -> Optional[Conversation]:
        """Lấy thông tin cuộc trò chuyện theo ID"""
        return self.db.query(Conversation).filter(Conversation.conversation_id == conversation_id).first()
    
    def get_latest_conversation(self, user_id: int) -> Optional[Conversation]:
        """Lấy cuộc trò chuyện mới nhất của người dùng"""
        # Kiểm tra trong Redis trước
        cached_id = redis_client.get(f"user:{user_id}:latest_conversation")
        
        if cached_id:
            conversation = self.get_conversation_by_id(int(cached_id))
            if conversation:
                return conversation
        
        # Nếu không có trong cache, truy vấn từ DB
        conversation = self.db.query(Conversation).filter(
            Conversation.user_id == user_id
        ).order_by(Conversation.updated_at.desc()).first()
        
        # Cập nhật cache nếu tìm thấy
        if conversation:
            redis_client.set(f"user:{user_id}:latest_conversation", conversation.conversation_id, ex=3600)  # TTL 1 giờ
        
        return conversation
    
    def get_messages(self, conversation_id: int, limit: int = None) -> List[Dict[str, str]]:
        """
        Lấy tất cả tin nhắn của một cuộc trò chuyện
        
        Args:
            conversation_id: ID của cuộc trò chuyện
            limit: Số lượng tin nhắn tối đa cần lấy (nếu None, sử dụng giá trị từ cấu hình)
            
        Returns:
            Danh sách tin nhắn theo định dạng cho LLM
        """
        if limit is None:
            limit = settings.MAX_HISTORY_MESSAGES
            
        # Kiểm tra cache trước
        cache_key = f"conversation:{conversation_id}:messages"
        cached_messages = redis_client.get(cache_key)
        
        if cached_messages:
            try:
                messages = json.loads(cached_messages)
                logger.debug(f"Lấy {len(messages)} tin nhắn từ cache cho conversation_id={conversation_id}")
                
                # Nếu số lượng tin nhắn vượt quá giới hạn, áp dụng giới hạn
                if limit > 0 and len(messages) > limit:
                    # Đảm bảo giữ lại system message
                    system_messages = [msg for msg in messages if msg.get("role") == "system"]
                    non_system_messages = [msg for msg in messages if msg.get("role") != "system"]
                    
                    # Lấy n tin nhắn gần nhất
                    limited_non_system = non_system_messages[-limit:] if len(non_system_messages) > limit else non_system_messages
                    
                    # Kết hợp lại
                    messages = system_messages + limited_non_system
                    logger.info(f"Đã giới hạn xuống {len(messages)} tin nhắn từ cache")
                
                # Kiểm tra cấu trúc tin nhắn
                self._validate_messages(messages)
                return messages
                
            except json.JSONDecodeError:
                logger.error(f"Lỗi giải mã JSON từ cache: {cached_messages[:100]}...")
                # Tiếp tục để lấy từ DB
            except Exception as e:
                logger.error(f"Lỗi xử lý tin nhắn từ cache: {str(e)}")
                # Tiếp tục để lấy từ DB
        
        # Nếu không có trong cache, truy vấn từ DB
        messages_query = self.db.query(Message).filter(
            Message.conversation_id == conversation_id
        ).order_by(Message.created_at)
        
        # Đếm tổng số tin nhắn
        total_messages = messages_query.count()
        logger.info(f"Tổng số tin nhắn trong DB: {total_messages} cho conversation_id={conversation_id}")
        
        # Tải tất cả tin nhắn
        all_messages = messages_query.all()
        
        # Tách system messages
        system_messages = [msg for msg in all_messages if msg.role == "system"]
        non_system_messages = [msg for msg in all_messages if msg.role != "system"]
        
        # Giới hạn số lượng tin nhắn non-system
        if limit > 0 and len(non_system_messages) > limit:
            non_system_messages = non_system_messages[-limit:]
            logger.info(f"Đã giới hạn xuống {len(non_system_messages)} tin nhắn non-system + {len(system_messages)} system messages")
        
        # Kết hợp lại
        limited_messages = system_messages + non_system_messages
        
        # Sắp xếp theo thời gian
        limited_messages.sort(key=lambda msg: msg.created_at)
        
        # Format lại kết quả cho phù hợp với định dạng của LLM
        formatted_messages = [
            {"role": msg.role, "content": msg.content} 
            for msg in limited_messages
        ]
        
        # Kiểm tra và log dữ liệu
        logger.info(f"Trả về {len(formatted_messages)} tin nhắn cho conversation_id={conversation_id}")
        
        # Kiểm tra cấu trúc tin nhắn
        self._validate_messages(formatted_messages)
        
        # Cập nhật cache
        try:
            redis_client.set(cache_key, json.dumps(formatted_messages), ex=3600)  # TTL 1 giờ
        except Exception as e:
            logger.error(f"Lỗi lưu cache: {str(e)}")
        
        return formatted_messages
    
    def get_messages_with_summary(self, conversation_id: int, limit: int = None) -> List[Dict[str, str]]:
        """
        Lấy tin nhắn của cuộc trò chuyện với tóm tắt nếu cần
        
        Args:
            conversation_id: ID của cuộc trò chuyện
            limit: Số lượng tin nhắn tối đa cần lấy (nếu None, sử dụng giá trị từ cấu hình)
            
        Returns:
            Danh sách tin nhắn, có bao gồm tóm tắt nếu cần
        """
        if limit is None:
            limit = settings.MAX_HISTORY_MESSAGES
            
        # Lấy tổng số tin nhắn non-system
        total_messages = self.db.query(func.count(Message.message_id)).filter(
            Message.conversation_id == conversation_id,
            Message.role != "system"
        ).scalar()
        
        # Kiểm tra xem có cần tóm tắt không
        need_summary = (total_messages > settings.SUMMARY_THRESHOLD)
        
        # Tìm tin nhắn gần nhất có summary
        summarized_message = self.db.query(Message).filter(
            Message.conversation_id == conversation_id,
            Message.summary.isnot(None),
            Message.summary != ""
        ).order_by(Message.created_at.desc()).first()
        
        has_summary = summarized_message is not None
        
        # Kiểm tra xem có tin nhắn mới chưa được tóm tắt
        has_new_messages = self.db.query(func.count(Message.message_id)).filter(
            Message.conversation_id == conversation_id,
            Message.is_summarized == False,
            Message.role != "system"
        ).scalar() > 0
        
        # Cần tạo tóm tắt mới nếu: 
        # - Số tin nhắn vượt ngưỡng VÀ
        # - (Chưa có tóm tắt HOẶC Có tin nhắn mới chưa được tóm tắt)
        need_new_summary = need_summary and (not has_summary or has_new_messages)
        
        logger.info(f"Kiểm tra tóm tắt: total_msgs={total_messages}, need_summary={need_summary}, has_summary={has_summary}, need_new={need_new_summary}")
        
        # Lấy messages thông thường (có giới hạn)
        messages = self.get_messages(conversation_id, limit)
        
        # Nếu có tóm tắt và cần sử dụng, thêm tóm tắt vào đầu danh sách
        if has_summary and need_summary and not need_new_summary:
            system_messages = [msg for msg in messages if msg["role"] == "system"]
            non_system_messages = [msg for msg in messages if msg["role"] != "system"]
            
            # Tạo tin nhắn tóm tắt
            summary_message = {
                "role": "system",
                "content": f"TÓM TẮT CUỘC TRÒ CHUYỆN TRƯỚC ĐÂY: {summarized_message.summary}"
            }
            
            # Ghép lại, đảm bảo tóm tắt nằm sau system message gốc nhưng trước các tin nhắn user/assistant
            result_messages = system_messages + [summary_message] + non_system_messages
            
            logger.info(f"Đã thêm tóm tắt vào tin nhắn, tổng cộng {len(result_messages)} tin nhắn")
            return result_messages
            
        # Trả về tin nhắn đã lấy, không có tóm tắt
        return messages
    
    def get_unsummarized_conversation_ids(self, threshold: int = None) -> List[int]:
        """
        Lấy danh sách các cuộc trò chuyện cần được tóm tắt
        
        Args:
            threshold: Ngưỡng số lượng tin nhắn để cần tóm tắt
            
        Returns:
            Danh sách ID của các cuộc trò chuyện cần tóm tắt
        """
        if threshold is None:
            threshold = settings.SUMMARY_THRESHOLD
            
        # Lấy các cuộc trò chuyện có nhiều hơn threshold tin nhắn chưa được tóm tắt
        conversation_ids = self.db.query(Message.conversation_id).filter(
            Message.is_summarized == False,
            Message.role != "system"
        ).group_by(Message.conversation_id).having(
            func.count(Message.message_id) > threshold
        ).all()
        
        return [conv_id[0] for conv_id in conversation_ids]
    
    def is_user_owner_of_conversation(self, user_id: int, conversation_id: int) -> bool:
        """Kiểm tra xem người dùng có phải là chủ của cuộc trò chuyện không"""
        conversation = self.get_conversation_by_id(conversation_id)
        return conversation is not None and conversation.user_id == user_id
    
    def _cache_conversation_metadata(self, conversation: Conversation):
        """Lưu metadata của cuộc trò chuyện vào Redis"""
        user_id = conversation.user_id
        conversation_id = conversation.conversation_id
        
        # Lưu conversation ID mới nhất của user
        redis_client.set(f"user:{user_id}:latest_conversation", conversation_id, ex=3600)  # TTL 1 giờ
    
    def _update_conversation_cache(self, conversation_id: int):
        """Cập nhật cache khi có tin nhắn mới"""
        # Xóa cache cũ để buộc cập nhật mới khi get lần sau
        cache_key = f"conversation:{conversation_id}:messages"
        redis_client.delete(cache_key) 
    
    def _validate_messages(self, messages: List[Dict[str, str]]) -> None:
        """Kiểm tra tính hợp lệ của danh sách tin nhắn"""
        if not isinstance(messages, list):
            logger.error(f"Kiểu dữ liệu không hợp lệ: {type(messages)}, mong đợi kiểu list")
            raise ValueError("Định dạng tin nhắn không hợp lệ: mong đợi một danh sách")
        
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                logger.error(f"Tin nhắn {i} không phải kiểu dict: {type(msg)}")
                raise ValueError(f"Tin nhắn {i} không có định dạng đúng")
            
            if "role" not in msg:
                logger.error(f"Tin nhắn {i} thiếu trường 'role'")
                raise ValueError(f"Tin nhắn {i} thiếu thông tin role")
                
            if "content" not in msg:
                logger.error(f"Tin nhắn {i} thiếu trường 'content'")
                raise ValueError(f"Tin nhắn {i} thiếu nội dung")
                
            if msg["role"] not in ["system", "user", "assistant"]:
                logger.warning(f"Tin nhắn {i} có role không chuẩn: {msg['role']}")
        
        # Kiểm tra xem có ít nhất một tin nhắn người dùng
        user_messages = [msg for msg in messages if msg.get("role") == "user"]
        if not user_messages:
            logger.warning("Không có tin nhắn nào từ người dùng trong cuộc trò chuyện") 