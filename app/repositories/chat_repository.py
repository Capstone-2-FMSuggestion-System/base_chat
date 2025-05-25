import json
import logging
from sqlalchemy.orm import Session
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from sqlalchemy import desc, func

from app.db.models import Conversation, Message, User, HealthData, Menu, MenuItem
from app.services.cache_service import CacheService, invalidate_cache_on_update
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

        CacheService.cache_conversation_metadata(
            conversation.conversation_id,
            user_id,
            title,
            conversation.created_at,
            conversation.updated_at
        )
        return conversation

    def add_message(self, conversation_id: int, role: str, content: str, is_summarized: bool = False) -> Message:
        """
        Thêm tin nhắn mới vào cuộc trò chuyện.
        
        Args:
            conversation_id: ID cuộc trò chuyện
            role: Vai trò (user/assistant/system)
            content: Nội dung tin nhắn
            is_summarized: Mặc định False - chỉ True nếu tin nhắn này TỰ NÓ là một bản tóm tắt
            
        Returns:
            Message object đã được tạo
            
        Note:
            - Trường 'summary' luôn được đặt là None khi tạo tin nhắn mới
            - Việc cập nhật summary sẽ được thực hiện riêng bởi save_conversation_summary()
            - is_summarized=False đảm bảo tin nhắn sẽ được xem xét cho việc tóm tắt sau này
        """
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            is_summarized=False,  # Luôn False cho tin nhắn mới - sẽ được cập nhật khi có tóm tắt
            summary=None  # Luôn None ban đầu - sẽ được cập nhật bởi save_conversation_summary
        )
        self.db.add(message)

        # Cập nhật timestamp cuộc trò chuyện
        conversation = self.get_conversation_by_id(conversation_id)
        if conversation:
            conversation.updated_at = datetime.now()

        self.db.commit()
        self.db.refresh(message)

        # Invalidate caches liên quan để đảm bảo consistency
        self._rebuild_messages_cache(conversation_id)
        self._sync_related_caches(conversation_id)
        
        # Invalidate cache tóm tắt vì có tin nhắn mới
        summary_cache_key = CacheService._get_cache_key(
            CacheService.CONVERSATION_METADATA, 
            conversation_id=f"{conversation_id}_latest_summary"
        )
        CacheService.delete_cache(summary_cache_key)

        return message

    def get_latest_summary(self, conversation_id: int) -> Optional[str]:
        """
        Lấy bản tóm tắt "cuộn" gần nhất của cuộc trò chuyện.
        
        Logic:
        1. Thử lấy từ Redis cache trước (O(1))
        2. Nếu cache miss: Query DB tìm assistant message gần nhất có summary
        3. Cache kết quả để tối ưu cho lần sau
        4. Nếu không tìm thấy: cache empty string với TTL ngắn để tránh query liên tục
        
        Args:
            conversation_id: ID cuộc trò chuyện
            
        Returns:
            Bản tóm tắt cuối cùng hoặc None nếu chưa có tóm tắt nào
        """
        # Tạo cache key nhất quán cho latest summary
        cache_key = CacheService._get_cache_key(
            CacheService.CONVERSATION_METADATA, 
            conversation_id=f"{conversation_id}_latest_summary"
        )

        # Bước 1: Thử lấy từ cache trước
        cached_summary = CacheService.get_cache(cache_key, expected_type=str)
        if cached_summary is not None:
            logger.debug(f"✅ Cache hit - Lấy tóm tắt từ cache cho conversation_id={conversation_id}")
            # Trả về None nếu cache chứa empty string (đánh dấu không có tóm tắt)
            return cached_summary if cached_summary else None

        # Bước 2: Cache miss - Query DB để tìm assistant message gần nhất có summary
        logger.debug(f"🔍 Cache miss - Query DB tìm tóm tắt cho conversation_id={conversation_id}")
        
        latest_assistant_message_with_summary = self.db.query(Message)\
            .filter(
                Message.conversation_id == conversation_id,
                Message.role == "assistant",
                Message.summary.isnot(None),
                Message.summary != ""
            )\
            .order_by(desc(Message.created_at))\
            .first()

        if latest_assistant_message_with_summary:
            summary_text = latest_assistant_message_with_summary.summary
            logger.info(f"📝 Tìm thấy tóm tắt từ DB cho conversation_id={conversation_id} (message_id={latest_assistant_message_with_summary.message_id})")
            
            # Bước 3: Cache kết quả với TTL medium
            CacheService.set_cache(cache_key, summary_text, ttl=CacheService.TTL_MEDIUM)
            return summary_text
        
        # Bước 4: Không tìm thấy tóm tắt - cache empty string với TTL ngắn
        logger.debug(f"❌ Không tìm thấy tóm tắt cho conversation_id={conversation_id}")
        CacheService.set_cache(cache_key, "", ttl=CacheService.TTL_SHORT)
        return None

    def save_conversation_summary(self, conversation_id: int, assistant_message_id: int, summary_text: str) -> bool:
        """
        Lưu bản tóm tắt tăng dần vào trường summary của tin nhắn assistant.
        Đảm bảo tính nhất quán giữa DB và cache.
        
        Logic:
        1. Tìm assistant message theo ID và conversation_id
        2. Cập nhật summary và đánh dấu is_summarized=True cho assistant message
        3. Tìm user message ngay trước đó và đánh dấu is_summarized=True
        4. Commit DB transaction
        5. Cập nhật cache với summary mới
        
        Args:
            conversation_id: ID cuộc trò chuyện
            assistant_message_id: ID tin nhắn assistant cần lưu tóm tắt
            summary_text: Nội dung tóm tắt
            
        Returns:
            True nếu thành công, False nếu có lỗi
        """
        try:
            # Bước 1: Tìm assistant message
            assistant_message = self.db.query(Message).filter(
                Message.message_id == assistant_message_id,
                Message.conversation_id == conversation_id,
                Message.role == "assistant"
            ).first()

            if not assistant_message:
                logger.error(f"❌ Không tìm thấy assistant_message_id={assistant_message_id} trong conversation_id={conversation_id}")
                return False

            # Bước 2: Cập nhật summary cho assistant message
            assistant_message.summary = summary_text
            assistant_message.is_summarized = True
            assistant_message.updated_at = datetime.now()
            
            # Bước 3: Tìm và đánh dấu user message ngay trước đó
            user_message_before_assistant = self.db.query(Message).filter(
                Message.conversation_id == conversation_id,
                Message.role == "user",
                Message.created_at < assistant_message.created_at
            ).order_by(desc(Message.created_at)).first()

            if user_message_before_assistant and not user_message_before_assistant.is_summarized:
                user_message_before_assistant.is_summarized = True
                user_message_before_assistant.updated_at = datetime.now()
                logger.debug(f"✅ Đánh dấu user_message_id={user_message_before_assistant.message_id} đã được tóm tắt")

            # Bước 4: Commit transaction
            self.db.commit()
            logger.info(f"💾 Đã lưu tóm tắt cho assistant_message_id={assistant_message_id}, độ dài: {len(summary_text)} ký tự")

            # Bước 5: Cập nhật cache
            cache_key = CacheService._get_cache_key(
                CacheService.CONVERSATION_METADATA, 
                conversation_id=f"{conversation_id}_latest_summary"
            )
            cache_success = CacheService.set_cache(cache_key, summary_text, ttl=CacheService.TTL_MEDIUM)
            
            if cache_success:
                logger.debug(f"🔄 Đã cập nhật cache tóm tắt cho conversation_id={conversation_id}")
            else:
                logger.warning(f"⚠️ Không thể cập nhật cache tóm tắt - DB đã lưu thành công")
            
            # Invalidate các cache liên quan
            self._invalidate_summary_related_caches(conversation_id)
            
            return True

        except Exception as e:
            # Rollback transaction nếu có lỗi
            self.db.rollback()
            logger.error(f"💥 Lỗi khi lưu tóm tắt cho assistant_message_id={assistant_message_id}: {str(e)}", exc_info=True)
            return False

    def get_messages_for_summary_context(self, conversation_id: int, limit: int = 3) -> List[Dict[str, str]]:
        """
        Lấy tin nhắn gần nhất để làm ngữ cảnh cho việc tạo tóm tắt tăng dần.
        
        Logic:
        1. Lấy limit*2 tin nhắn cuối cùng từ DB (để đảm bảo có đủ cả user và assistant)
        2. Sắp xếp theo thứ tự thời gian tăng dần (giữ đúng flow hội thoại)
        3. Giới hạn kết quả theo limit
        
        Args:
            conversation_id: ID cuộc trò chuyện
            limit: Số lượng tin nhắn tối đa cần lấy
            
        Returns:
            List các dict {"role": "...", "content": "..."} theo thứ tự thời gian
        """
        try:
            # Lấy limit*2 tin nhắn cuối để đảm bảo có đủ context
            messages_db = self.db.query(Message)\
                .filter(Message.conversation_id == conversation_id)\
                .order_by(desc(Message.created_at))\
                .limit(limit * 2)\
                .all()
            
            if not messages_db:
                logger.debug(f"Không có tin nhắn nào cho conversation_id={conversation_id}")
                return []

            # Đảo ngược để có thứ tự thời gian tăng dần (đúng flow hội thoại)
            formatted_messages = []
            for msg in reversed(messages_db):
                formatted_messages.append({
                    "role": msg.role, 
                    "content": msg.content,
                    "created_at": msg.created_at.isoformat() if hasattr(msg, 'created_at') else None
                })

            # Giới hạn kết quả theo limit
            result = formatted_messages[-limit:] if len(formatted_messages) > limit else formatted_messages
            
            logger.debug(f"📋 Lấy {len(result)} tin nhắn context cho conversation_id={conversation_id}")
            return result

        except Exception as e:
            logger.error(f"💥 Lỗi khi lấy tin nhắn context cho conversation_id={conversation_id}: {str(e)}", exc_info=True)
            return []

    def get_conversation_by_id(self, conversation_id: int) -> Optional[Conversation]:
        cached_metadata = CacheService.get_conversation_metadata(conversation_id)
        if cached_metadata:
            try:
                conv = Conversation()
                for key, value in cached_metadata.items():
                    if hasattr(conv, key) and value is not None:
                        if key in ['created_at', 'updated_at']:
                            setattr(conv, key, datetime.fromisoformat(value))
                        else:
                            setattr(conv, key, value)
                return conv
            except (KeyError, ValueError, TypeError) as e:
                logger.error(f"Lỗi parse conversation cache: {str(e)}")
                CacheService.delete_cache(CacheService._get_cache_key(CacheService.CONVERSATION_METADATA, conversation_id=conversation_id))

        conversation = self.db.query(Conversation).filter(Conversation.conversation_id == conversation_id).first()
        if conversation:
            CacheService.cache_conversation_metadata(
                conversation.conversation_id, conversation.user_id, conversation.title,
                conversation.created_at, conversation.updated_at
            )
        return conversation

    def get_latest_conversation(self, user_id: int) -> Optional[Conversation]:
        cached_conversation_id = CacheService.get_latest_conversation_id(user_id)
        if cached_conversation_id:
            conversation = self.get_conversation_by_id(cached_conversation_id)
            if conversation:
                return conversation

        conversation = self.db.query(Conversation).filter(
            Conversation.user_id == user_id
        ).order_by(desc(Conversation.updated_at)).first()

        if conversation:
            CacheService.cache_conversation_metadata(
                conversation.conversation_id,
                user_id,
                conversation.title,
                conversation.created_at,
                conversation.updated_at
            )
        return conversation

    def get_messages(self, conversation_id: int, limit: int = None) -> List[Dict[str, str]]:
        if limit is None:
            limit = settings.MAX_HISTORY_MESSAGES

        cache_key = CacheService._get_cache_key(CacheService.CONVERSATION_MESSAGES, conversation_id=conversation_id)

        def rebuild_messages_from_db():
            logger.debug(f"Rebuilding messages từ DB cho conversation_id={conversation_id}")
            messages_query = self.db.query(Message).filter(
                Message.conversation_id == conversation_id
            ).order_by(Message.created_at)
            
            all_messages_db = messages_query.all()
            formatted_msgs = [{"role": msg.role, "content": msg.content} for msg in all_messages_db]
            logger.info(f"Đã rebuild {len(formatted_msgs)} tin nhắn từ DB")
            return formatted_msgs

        all_cached_messages = CacheService.get_or_rebuild_cache(
            key=cache_key,
            rebuild_func=rebuild_messages_from_db,
            expected_type=list,
            ttl=CacheService.TTL_MEDIUM
        )

        if not all_cached_messages:
            return []

        if limit > 0 and len(all_cached_messages) > limit:
            return all_cached_messages[-limit:]
        return all_cached_messages

    def get_messages_with_summary(self, conversation_id: int, limit: int = None) -> List[Dict[str, str]]:
        """
        Đơn giản hóa - chỉ trả về tin nhắn thô.
        ChatService sẽ chịu trách nhiệm kết hợp tin nhắn và tóm tắt nếu cần.
        
        Note: Hàm này được giữ lại để tương thích ngược,
        nhưng logic tóm tắt tăng dần được xử lý bởi ChatService.
        """
        return self.get_messages(conversation_id, limit)

    def get_unsummarized_conversation_ids(self, threshold: int = None) -> List[int]:
        if threshold is None:
            threshold = settings.SUMMARY_THRESHOLD
            
        cached_result = CacheService.get_unsummarized_conversations(threshold)
        if cached_result is not None:
            return cached_result
        
        try:
            conversation_ids_query = self.db.query(Message.conversation_id).filter(
                Message.is_summarized == False,
                Message.role != "system"
            ).group_by(Message.conversation_id).having(
                func.count(Message.message_id) >= threshold
            )
            conversation_ids = conversation_ids_query.all()
            result = [conv_id[0] for conv_id in conversation_ids]
        except Exception as e:
            logger.error(f"Lỗi query unsummarized conversations: {e}")
            all_convs_with_unsummarized = self.db.query(Message.conversation_id).filter(
                Message.is_summarized == False, Message.role != "system"
            ).distinct().all()
            result = []
            for conv_id_tuple in all_convs_with_unsummarized:
                conv_id = conv_id_tuple[0]
                count = self.db.query(func.count(Message.message_id)).filter(
                    Message.conversation_id == conv_id,
                    Message.is_summarized == False,
                    Message.role != "system"
                ).scalar()
                if count >= threshold:
                    result.append(conv_id)

        CacheService.cache_unsummarized_conversations(threshold, result)
        return result

    def is_user_owner_of_conversation(self, user_id: int, conversation_id: int) -> bool:
        cached_owner = CacheService.get_conversation_owner(conversation_id)
        if cached_owner is not None:
            return cached_owner == user_id
            
        conversation = self.db.query(Conversation).filter(Conversation.conversation_id == conversation_id).first()
        if conversation:
            CacheService.cache_conversation_metadata(
                conversation.conversation_id, conversation.user_id, conversation.title,
                conversation.created_at, conversation.updated_at
            )
            return conversation.user_id == user_id
        return False

    def save_health_data(self, conversation_id: int, user_id: int, health_condition: str = None, 
                        medical_history: str = None, allergies: str = None, dietary_habits: str = None, 
                        health_goals: str = None, additional_info: Dict = None, data: Dict = None) -> HealthData:
        """
        Lưu thông tin sức khỏe với xử lý thông minh các giá trị rỗng từ collected_info.
        
        Args:
            conversation_id: ID cuộc trò chuyện
            user_id: ID người dùng
            health_condition: Tình trạng sức khỏe
            medical_history: Lịch sử bệnh án
            allergies: Dị ứng
            dietary_habits: Thói quen ăn uống
            health_goals: Mục tiêu sức khỏe
            additional_info: Thông tin bổ sung
            data: Dictionary chứa dữ liệu từ collected_info (ưu tiên cao hơn các tham số riêng lẻ)
        
        Returns:
            HealthData object đã được lưu
        """
        try:
            logger.info(f"Lưu thông tin sức khỏe: conv_id={conversation_id}")
            existing_data = self.db.query(HealthData).filter(HealthData.conversation_id == conversation_id).first()
            
            # Nếu có data từ collected_info, ưu tiên sử dụng nó
            if data:
                health_condition = data.get('health_condition', health_condition)
                medical_history = data.get('medical_history', medical_history)
                allergies = data.get('allergies', allergies)
                dietary_habits = data.get('dietary_habits', dietary_habits)
                health_goals = data.get('health_goals', health_goals)
                
                # Xử lý additional_info từ data
                if not additional_info:
                    additional_info = {}
                
                # Thêm food_preferences và food_dislikes từ data vào additional_info
                if data.get('food_preferences') is not None:
                    additional_info['food_preferences'] = data.get('food_preferences')
                if data.get('food_dislikes') is not None:
                    additional_info['food_dislikes'] = data.get('food_dislikes')
                
                # Thêm các trường khác từ data vào additional_info (ngoài các trường cơ bản)
                excluded_fields = ['health_condition', 'medical_history', 'allergies', 'dietary_habits', 'health_goals', 'food_preferences', 'food_dislikes']
                for k, v in data.items():
                    if k not in excluded_fields and v is not None:
                        additional_info[k] = v
            
            if existing_data:
                # Hàm helper để cập nhật trường với logic tránh ghi đè bằng chuỗi rỗng
                def update_field_safely(field_name: str, new_value: str, current_value: str) -> str:
                    """
                    Cập nhật trường một cách an toàn:
                    - Nếu new_value là None: cho phép xóa (trả về None)
                    - Nếu new_value là chuỗi rỗng và current_value có giá trị: giữ nguyên current_value
                    - Ngược lại: cập nhật với new_value
                    """
                    if new_value is None:
                        return None  # Cho phép xóa giá trị
                    elif isinstance(new_value, str) and new_value.strip() == "" and current_value:
                        logger.debug(f"Giữ nguyên {field_name} hiện có (không ghi đè bằng chuỗi rỗng)")
                        return current_value  # Giữ nguyên giá trị hiện có
                    else:
                        return new_value  # Cập nhật với giá trị mới
                
                # Áp dụng logic cập nhật an toàn cho các trường cơ bản
                existing_data.health_condition = update_field_safely(
                    "health_condition", health_condition, existing_data.health_condition
                )
                existing_data.medical_history = update_field_safely(
                    "medical_history", medical_history, existing_data.medical_history
                )
                existing_data.allergies = update_field_safely(
                    "allergies", allergies, existing_data.allergies
                )
                existing_data.dietary_habits = update_field_safely(
                    "dietary_habits", dietary_habits, existing_data.dietary_habits
                )
                existing_data.health_goals = update_field_safely(
                    "health_goals", health_goals, existing_data.health_goals
                )
                
                # Xử lý additional_info với merge thông minh
                if additional_info:
                    # Đảm bảo existing additional_info tồn tại
                    if not existing_data.additional_info:
                        existing_data.additional_info = {}
                    
                    # Merge từng trường trong additional_info với logic an toàn
                    for key, value in additional_info.items():
                        if value is not None:
                            if isinstance(value, str) and value.strip() == "" and existing_data.additional_info.get(key):
                                # Không ghi đè giá trị hiện có bằng chuỗi rỗng
                                logger.debug(f"Giữ nguyên additional_info['{key}'] hiện có (không ghi đè bằng chuỗi rỗng)")
                            else:
                                existing_data.additional_info[key] = value
                        
                existing_data.updated_at = datetime.now()
                self.db.commit()
                
                health_data_dict = self._format_health_data_for_cache(existing_data)
                CacheService.cache_health_data(conversation_id, health_data_dict)
                logger.info(f"✅ Đã cập nhật thông tin sức khỏe cho conversation_id={conversation_id}")
                return existing_data
            else:
                health_data = HealthData(
                    conversation_id=conversation_id, user_id=user_id,
                    health_condition=health_condition,
                    medical_history=medical_history,
                    allergies=allergies,
                    dietary_habits=dietary_habits,
                    health_goals=health_goals,
                    additional_info=additional_info or {}
                )
                self.db.add(health_data)
                self.db.commit()
                self.db.refresh(health_data)
                
                health_data_dict = self._format_health_data_for_cache(health_data)
                CacheService.cache_health_data(conversation_id, health_data_dict)
                logger.info(f"✅ Đã tạo mới thông tin sức khỏe cho conversation_id={conversation_id}")
                return health_data
                
        except Exception as e:
            logger.error(f"Lỗi khi lưu thông tin sức khỏe: {str(e)}", exc_info=True)
            self.db.rollback()
            raise

    def get_health_data(self, conversation_id: int) -> Optional[Dict[str, Any]]:
        cached_data = CacheService.get_health_data(conversation_id)
        if cached_data:
            return cached_data
            
        health_data = self.db.query(HealthData).filter(
            HealthData.conversation_id == conversation_id
        ).order_by(desc(HealthData.updated_at)).first()
        
        if not health_data:
            return None
            
        result = self._format_health_data_for_cache(health_data)
        CacheService.cache_health_data(conversation_id, result)
        return result

    def _format_health_data_for_cache(self, health_data: HealthData) -> Dict[str, Any]:
        return {
            "user_id": health_data.user_id,
            "health_condition": health_data.health_condition,
            "medical_history": health_data.medical_history,
            "allergies": health_data.allergies,
            "dietary_habits": health_data.dietary_habits,
            "health_goals": health_data.health_goals,
            "additional_info": health_data.additional_info or {},
            "created_at": health_data.created_at.isoformat() if health_data.created_at else None,
            "updated_at": health_data.updated_at.isoformat() if health_data.updated_at else None
        }

    def _rebuild_messages_cache(self, conversation_id: int) -> bool:
        """
        Rebuild cache tin nhắn từ DB để đảm bảo consistency.
        
        Args:
            conversation_id: ID cuộc trò chuyện cần rebuild cache
            
        Returns:
            True nếu rebuild thành công, False nếu có lỗi
        """
        try:
            # Query tất cả tin nhắn từ DB theo thứ tự thời gian
            messages_query = self.db.query(Message).filter(
                Message.conversation_id == conversation_id
            ).order_by(Message.created_at)
            
            all_messages_db = messages_query.all()
            formatted_messages = [
                {"role": msg.role, "content": msg.content} 
                for msg in all_messages_db
            ]
            
            # Cache với key nhất quán
            cache_key = CacheService._get_cache_key(
                CacheService.CONVERSATION_MESSAGES, 
                conversation_id=conversation_id
            )
            success = CacheService.set_cache(cache_key, formatted_messages, CacheService.TTL_MEDIUM)
            
            if success:
                logger.debug(f"🔄 Đã rebuild messages cache cho conversation_id={conversation_id} ({len(formatted_messages)} tin nhắn)")
                return True
            else:
                logger.error(f"❌ Lỗi set cache khi rebuild cho conversation_id={conversation_id}")
                return False
            
        except Exception as e:
            logger.error(f"💥 Lỗi khi rebuild cache tin nhắn cho conversation_id={conversation_id}: {str(e)}", exc_info=True)
            return False

    def _sync_related_caches(self, conversation_id: int) -> None:
        """
        Đồng bộ các cache liên quan khi có thay đổi conversation.
        
        Args:
            conversation_id: ID cuộc trò chuyện cần sync cache
        """
        try:
            conversation = self.db.query(Conversation).filter(
                Conversation.conversation_id == conversation_id
            ).first()
            
            if conversation:
                # Cache conversation metadata
                CacheService.cache_conversation_metadata(
                    conversation.conversation_id, conversation.user_id, conversation.title,
                    conversation.created_at, conversation.updated_at
                )
                
                # Cache latest conversation cho user
                latest_key = CacheService._get_cache_key(
                    CacheService.USER_LATEST_CONVERSATION, 
                    user_id=conversation.user_id
                )
                CacheService.set_cache(latest_key, conversation_id, CacheService.TTL_MEDIUM)
                
                logger.debug(f"🔄 Đã sync related caches cho conversation_id={conversation_id}")
            else:
                logger.warning(f"⚠️ Không tìm thấy conversation_id={conversation_id} để sync cache")
            
        except Exception as e:
            logger.error(f"💥 Lỗi khi sync related caches cho conversation_id={conversation_id}: {str(e)}", exc_info=True)

    @invalidate_cache_on_update(["recent_recipes:*", "recipe:*:details"])
    def save_recipe_to_menu(self, recipe_name: str, recipe_description: str, 
                           ingredients_with_products: List[Dict[str, Any]]) -> Optional[int]:
        try:
            menu = Menu(name=recipe_name, description=recipe_description)
            self.db.add(menu)
            self.db.flush()
            
            menu_id = menu.menu_id
            logger.info(f"Đã tạo menu '{recipe_name}' với ID={menu_id}")
            
            menu_items_data = []
            ingredients_saved_count = 0
            
            # ⭐ LƯU TẤT CẢ NGUYÊN LIỆU, PRODUCT_ID CÓ THỂ NULL
            for ing in ingredients_with_products:
                ingredient_name = ing.get('ingredient_name', '').strip()
                product_id = ing.get('product_id')
                quantity = ing.get('quantity', 1)
                
                if ingredient_name:  # Chỉ cần có tên nguyên liệu
                    menu_item = MenuItem(
                        menu_id=menu_id,
                        product_id=product_id,  # Có thể là None
                        quantity=quantity
                    )
                    self.db.add(menu_item)
                    ingredients_saved_count += 1
                    
                    menu_items_data.append({
                        'product_id': product_id,
                        'quantity': quantity,
                        'ingredient_name': ingredient_name
                    })
                    
            self.db.commit()
            
            recipe_data_cache = {
                'menu_id': menu_id,
                'name': recipe_name,
                'description': recipe_description,
                'created_at': datetime.now().isoformat(),
                'ingredients': menu_items_data
            }
            CacheService.cache_recipe_data(menu_id, recipe_data_cache)
            
            logger.info(f"Đã lưu công thức '{recipe_name}' với {ingredients_saved_count} nguyên liệu")
            return menu_id
            
        except Exception as e:
            logger.error(f"Lỗi khi lưu công thức '{recipe_name}': {str(e)}", exc_info=True)
            self.db.rollback()
            return None

    def save_multiple_recipes_to_menu(self, recipes_data: List[Dict[str, Any]], 
                                     product_mapping: Dict[str, Any]) -> List[int]:
        created_menu_ids = []
        if not recipes_data:
            return created_menu_ids
            
        # ⭐ XÂY DỰNG MAPPING TỪ NGUYÊN LIỆU ĐẾN PRODUCT_ID
        ingredient_to_product = {}
        if product_mapping and product_mapping.get('ingredient_mapping_results'):
            for mapping in product_mapping['ingredient_mapping_results']:
                name_low = mapping.get('requested_ingredient', '').lower().strip()
                pid = mapping.get('product_id')
                if name_low and pid:
                    ingredient_to_product[name_low] = {
                        'product_id': int(pid),
                        'product_name': mapping.get('product_name', '')
                    }
        
        for recipe in recipes_data:
            try:
                name = recipe.get('name', 'Món ăn không tên')
                ing_sum = recipe.get('ingredients_summary', '')
                url = recipe.get('url', '')
                
                desc_parts = []
                if ing_sum:
                    desc_parts.append(f"Nguyên liệu: {ing_sum}")
                if url:
                    desc_parts.append(f"Nguồn: {url}")
                    
                recipe_desc = "\n".join(desc_parts)
                ings_with_prods = []
                
                # ⭐ LƯU TẤT CẢ NGUYÊN LIỆU, KHÔNG CHỈ NHỮNG CÁI CÓ PRODUCT_ID
                if ing_sum:
                    for ing_item_str in ing_sum.split(','):
                        ing_clean = ing_item_str.strip()
                        if ing_clean:
                            ing_low = ing_clean.lower()
                            prod_info = None
                            
                            # Tìm product_id nếu có
                            if ing_low in ingredient_to_product:
                                prod_info = ingredient_to_product[ing_low]
                            else:
                                for mapped_ing, info_map in ingredient_to_product.items():
                                    if mapped_ing in ing_low or ing_low in mapped_ing:
                                        prod_info = info_map
                                        break
                            
                            # ⭐ LUÔN THÊM NGUYÊN LIỆU VÀO DANH SÁCH, PRODUCT_ID CÓ THỂ NULL
                            ings_with_prods.append({
                                'ingredient_name': ing_clean,
                                'product_id': prod_info['product_id'] if prod_info else None,
                                'quantity': 1
                            })
                
                menu_id = self.save_recipe_to_menu(name, recipe_desc, ings_with_prods)
                if menu_id:
                    created_menu_ids.append(menu_id)
                else:
                    logger.warning(f"Không thể lưu recipe '{name}' từ batch")
                    
            except Exception as e:
                logger.error(f"Lỗi khi xử lý recipe '{recipe.get('name', 'Unknown')}': {str(e)}", exc_info=True)
                continue
        
        if created_menu_ids:
            batch_data_cache = {
                'menu_ids': created_menu_ids,
                'recipes_count': len(recipes_data),
                'recipes_summary': [
                    {
                        'name': r.get('name', ''),
                        'menu_id': created_menu_ids[i] if i < len(created_menu_ids) else None
                    }
                    for i, r in enumerate(recipes_data)
                ]
            }
            CacheService.cache_batch_operation("batch_recipe_save", batch_data_cache)
            
        logger.info(f"Đã lưu {len(created_menu_ids)}/{len(recipes_data)} recipes vào database")
        return created_menu_ids

    def get_cached_recipes(self, limit: int = 20) -> Optional[List[Dict[str, Any]]]:
        cache_key = CacheService._get_cache_key(CacheService.RECENT_RECIPES, limit=limit)
        return CacheService.get_cache(cache_key, list)

    def get_recipe_by_id(self, menu_id: int) -> Optional[Dict[str, Any]]:
        cached_recipe = CacheService.get_recipe_data(menu_id)
        if cached_recipe:
            return cached_recipe
            
        menu = self.db.query(Menu).filter(Menu.menu_id == menu_id).first()
        if not menu:
            return None
            
        menu_items_db = self.db.query(MenuItem).filter(MenuItem.menu_id == menu_id).all()
        
        # ⭐ TRÍCH XUẤT INGREDIENT_NAME TỪ DESCRIPTION NẾU CÓ
        ingredients_list = []
        ingredient_names_from_desc = []
        
        # Parse ingredient names từ description nếu có
        if menu.description and "Nguyên liệu:" in menu.description:
            desc_lines = menu.description.split('\n')
            for line in desc_lines:
                if line.startswith("Nguyên liệu:"):
                    ingredients_text = line.replace("Nguyên liệu:", "").strip()
                    ingredient_names_from_desc = [ing.strip() for ing in ingredients_text.split(',') if ing.strip()]
                    break
        
        # Tạo danh sách ingredients với tên nếu có
        for i, item in enumerate(menu_items_db):
            ingredient_name = None
            if i < len(ingredient_names_from_desc):
                ingredient_name = ingredient_names_from_desc[i]
            
            ingredients_list.append({
                'product_id': item.product_id,
                'quantity': item.quantity,
                'ingredient_name': ingredient_name
            })
        
        recipe_data_db = {
            'menu_id': menu.menu_id,
            'name': menu.name,
            'description': menu.description,
            'created_at': menu.created_at.isoformat() if menu.created_at else None,
            'ingredients': ingredients_list
        }
        
        CacheService.cache_recipe_data(menu_id, recipe_data_db)
        return recipe_data_db 

    def _invalidate_summary_related_caches(self, conversation_id: int) -> None:
        """Invalidate tất cả cache liên quan đến tóm tắt khi có cập nhật"""
        try:
            summary_cache_key = CacheService._get_cache_key(
                CacheService.CONVERSATION_METADATA, 
                conversation_id=f"{conversation_id}_latest_summary"
            )
            CacheService.delete_cache(summary_cache_key)
            
            self._rebuild_messages_cache(conversation_id)
            self._sync_related_caches(conversation_id)
            
            logger.debug(f"🔄 Đã invalidate cache tóm tắt cho conversation_id={conversation_id}")
        except Exception as e:
            logger.warning(f"⚠️ Lỗi khi invalidate summary cache: {str(e)}")

    # === BACKGROUND DB OPERATIONS METHODS ===
    
    def prepare_add_message(self, conversation_id: int, role: str, content: str) -> Message:
        """
        Chuẩn bị message object mà không commit vào DB ngay.
        Dùng cho background processing.
        
        Returns:
            Message object đã được add vào session nhưng chưa commit
        """
        message = Message(
            conversation_id=conversation_id,
            role=role,
            content=content,
            is_summarized=False,
            summary=None
        )
        self.db.add(message)
        
        # Cập nhật timestamp cuộc trò chuyện
        conversation = self.get_conversation_by_id(conversation_id)
        if conversation:
            conversation.updated_at = datetime.now()
        
        # Chưa commit - sẽ được commit ở background
        logger.debug(f"📝 Prepared add_message for conversation_id={conversation_id}, role={role}")
        return message
    
    def commit_and_refresh_message(self, message: Message) -> Message:
        """
        Commit message đã được prepare và refresh để lấy ID.
        Sau đó invalidate caches.
        
        Args:
            message: Message object đã được prepare
            
        Returns:
            Message object đã commit và refresh
        """
        self.db.commit()
        self.db.refresh(message)
        
        # Invalidate caches sau khi commit
        self._rebuild_messages_cache(message.conversation_id)
        self._sync_related_caches(message.conversation_id)
        
        # Invalidate cache tóm tắt vì có tin nhắn mới
        summary_cache_key = CacheService._get_cache_key(
            CacheService.CONVERSATION_METADATA, 
            conversation_id=f"{message.conversation_id}_latest_summary"
        )
        CacheService.delete_cache(summary_cache_key)
        
        logger.debug(f"✅ Committed and refreshed message_id={message.message_id}")
        return message
    
    def add_message_immediate(self, conversation_id: int, role: str, content: str) -> Message:
        """
        Version không thay đổi của add_message - commit ngay lập tức.
        Dùng cho trường hợp cần response ngay (như user message trước khi gọi LLM).
        """
        message = self.prepare_add_message(conversation_id, role, content)
        return self.commit_and_refresh_message(message)
    
    def prepare_health_data(self, conversation_id: int, user_id: int, 
                           health_condition: str = None, medical_history: str = None,
                           allergies: str = None, dietary_habits: str = None,
                           health_goals: str = None, additional_info: Dict = None,
                           data: Dict = None) -> Optional[HealthData]:
        """
        Chuẩn bị HealthData object mà không commit vào DB ngay.
        Dùng cho background processing.
        """
        try:
            existing_health_data = self.db.query(HealthData).filter(
                HealthData.conversation_id == conversation_id,
                HealthData.user_id == user_id
            ).first()

            if existing_health_data:
                # Update existing
                def update_field_safely(field_name: str, new_value: str, current_value: str) -> str:
                    if not new_value or new_value.strip() == "":
                        return current_value
                    if not current_value or current_value.strip() == "":
                        return new_value
                    if new_value not in current_value:
                        return f"{current_value}. {new_value}"
                    return current_value

                if health_condition:
                    existing_health_data.health_condition = update_field_safely(
                        "health_condition", health_condition, existing_health_data.health_condition or ""
                    )
                if medical_history:
                    existing_health_data.medical_history = update_field_safely(
                        "medical_history", medical_history, existing_health_data.medical_history or ""
                    )
                if allergies:
                    existing_health_data.allergies = update_field_safely(
                        "allergies", allergies, existing_health_data.allergies or ""
                    )
                if dietary_habits:
                    existing_health_data.dietary_habits = update_field_safely(
                        "dietary_habits", dietary_habits, existing_health_data.dietary_habits or ""
                    )
                if health_goals:
                    existing_health_data.health_goals = update_field_safely(
                        "health_goals", health_goals, existing_health_data.health_goals or ""
                    )

                if additional_info:
                    current_additional = existing_health_data.additional_info or {}
                    if isinstance(current_additional, str):
                        try:
                            current_additional = json.loads(current_additional)
                        except:
                            current_additional = {}
                    current_additional.update(additional_info)
                    existing_health_data.additional_info = current_additional

                if data:
                    current_data = existing_health_data.data or {}
                    if isinstance(current_data, str):
                        try:
                            current_data = json.loads(current_data)
                        except:
                            current_data = {}
                    current_data.update(data)
                    existing_health_data.data = current_data

                existing_health_data.updated_at = datetime.now()
                logger.debug(f"📝 Prepared update health_data for conversation_id={conversation_id}")
                return existing_health_data
            else:
                # Create new
                health_data = HealthData(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    health_condition=health_condition,
                    medical_history=medical_history,
                    allergies=allergies,
                    dietary_habits=dietary_habits,
                    health_goals=health_goals,
                    additional_info=additional_info,
                    data=data
                )
                self.db.add(health_data)
                logger.debug(f"📝 Prepared create new health_data for conversation_id={conversation_id}")
                return health_data

        except Exception as e:
            logger.error(f"💥 Lỗi khi prepare health_data: {str(e)}")
            return None
    
    def commit_and_refresh_health_data(self, health_data: HealthData) -> HealthData:
        """
        Commit health_data đã được prepare và refresh để lấy ID.
        Sau đó invalidate caches.
        """
        self.db.commit()
        self.db.refresh(health_data)
        
        # Invalidate health data cache
        health_cache_key = CacheService._get_cache_key(
            CacheService.HEALTH_DATA, 
            conversation_id=health_data.conversation_id
        )
        CacheService.delete_cache(health_cache_key)
        
        logger.debug(f"✅ Committed and refreshed health_data id={health_data.id}")
        return health_data 