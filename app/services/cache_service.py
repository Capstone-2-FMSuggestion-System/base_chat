import json
import logging
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timedelta
import asyncio
from functools import wraps

from app.db.database import redis_client
from app.config import settings


logger = logging.getLogger(__name__)


class CacheService:
    """Service quản lý Redis cache tập trung"""
    
    # Cache key patterns
    CONVERSATION_METADATA = "conversation:{conversation_id}:metadata"
    CONVERSATION_MESSAGES = "conversation:{conversation_id}:messages"
    CONVERSATION_OWNER = "conversation:{conversation_id}:owner"
    USER_LATEST_CONVERSATION = "user:{user_id}:latest_conversation"
    HEALTH_DATA = "session:{conversation_id}:health_data"
    MESSAGE_SUMMARY = "message:{message_id}:summary"
    RECIPE_DETAILS = "recipe:{menu_id}:details"
    RECENT_RECIPES = "recent_recipes:limit_{limit}"
    UNSUMMARIZED_CONVERSATIONS = "unsummarized_conversations:threshold_{threshold}"
    BATCH_RECIPE_SAVE = "batch_recipe_save:{timestamp}"
    
    # Default TTL values (in seconds)
    TTL_SHORT = 600      # 10 minutes
    TTL_MEDIUM = 3600    # 1 hour  
    TTL_LONG = 86400     # 24 hours
    TTL_EXTRA_LONG = 604800  # 7 days
    
    @staticmethod
    def _get_cache_key(pattern: str, **kwargs) -> str:
        """Tạo cache key từ pattern và params"""
        return pattern.format(**kwargs)
    
    @staticmethod
    def _serialize_data(data: Any) -> str:
        """Serialize data để lưu vào Redis"""
        if isinstance(data, (dict, list)):
            return json.dumps(data, ensure_ascii=False)
        return str(data)
    
    @staticmethod
    def _deserialize_data(data: str, data_type: type = dict) -> Any:
        """Deserialize data từ Redis"""
        try:
            if data_type in (dict, list):
                return json.loads(data)
            elif data_type == int:
                return int(data)
            elif data_type == float:
                return float(data)
            return data
        except (json.JSONDecodeError, ValueError) as e:
            logger.error(f"Lỗi deserialize data: {str(e)}")
            return None
    
    @classmethod
    def set_cache(cls, key: str, value: Any, ttl: int = TTL_MEDIUM) -> bool:
        """Lưu data vào cache với error handling"""
        try:
            serialized_value = cls._serialize_data(value)
            redis_client.set(key, serialized_value, ex=ttl)
            logger.debug(f"Cached: {key} (TTL: {ttl}s)")
            return True
        except Exception as e:
            logger.error(f"Lỗi set cache {key}: {str(e)}")
            return False
    
    @classmethod
    def get_cache(cls, key: str, expected_type: type = None) -> Any:
        """
        Lấy dữ liệu từ cache với type validation
        
        Args:
            key: Cache key
            expected_type: Kiểu dữ liệu mong đợi (dict, list, str, int, etc.)
        
        Returns:
            Dữ liệu từ cache hoặc None nếu không tìm thấy
        """
        try:
            if not redis_client:
                logger.error("Redis client chưa được khởi tạo")
                return None
                
            cached_data = redis_client.get(key)
            if cached_data is None:
                return None
                
            # Deserialize JSON
            data = cls._deserialize_data(cached_data, expected_type)
            
            # Type validation nếu được chỉ định
            if expected_type and not isinstance(data, expected_type):
                logger.warning(f"Cache data type mismatch. Expected: {expected_type}, Got: {type(data)}")
                return None
                
            logger.debug(f"Cache hit cho key: {key}")
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"Lỗi decode JSON cho key {key}: {str(e)}")
            # Xóa cache bị lỗi
            redis_client.delete(key)
            return None
        except Exception as e:
            logger.error(f"Lỗi khi lấy cache key {key}: {str(e)}")
            return None
    
    @classmethod
    def delete_cache(cls, key: str) -> bool:
        """Xóa cache với error handling"""
        try:
            result = redis_client.delete(key)
            logger.debug(f"Deleted cache: {key}")
            return result > 0
        except Exception as e:
            logger.error(f"Lỗi delete cache {key}: {str(e)}")
            return False
    
    @classmethod
    def delete_pattern(cls, pattern: str) -> int:
        """Xóa tất cả cache keys matching pattern"""
        deleted_count = 0
        try:
            for key in redis_client.scan_iter(match=pattern):
                redis_client.delete(key)
                deleted_count += 1
            logger.debug(f"Deleted {deleted_count} cache keys matching: {pattern}")
            return deleted_count
        except Exception as e:
            logger.error(f"Lỗi delete pattern {pattern}: {str(e)}")
            return 0
    
    @classmethod
    def cache_conversation_metadata(cls, conversation_id: int, user_id: int, 
                                  title: str = None, created_at: datetime = None, 
                                  updated_at: datetime = None) -> bool:
        """Cache conversation metadata"""
        metadata = {
            'conversation_id': conversation_id,
            'user_id': user_id,
            'title': title,
            'created_at': created_at.isoformat() if created_at else None,
            'updated_at': updated_at.isoformat() if updated_at else None
        }
        
        # Cache metadata và owner info
        metadata_key = cls._get_cache_key(cls.CONVERSATION_METADATA, conversation_id=conversation_id)
        owner_key = cls._get_cache_key(cls.CONVERSATION_OWNER, conversation_id=conversation_id)
        latest_key = cls._get_cache_key(cls.USER_LATEST_CONVERSATION, user_id=user_id)
        
        success = True
        success &= cls.set_cache(metadata_key, metadata, cls.TTL_MEDIUM)
        success &= cls.set_cache(owner_key, user_id, cls.TTL_MEDIUM)
        success &= cls.set_cache(latest_key, conversation_id, cls.TTL_MEDIUM)
        
        return success
    
    @classmethod
    def get_conversation_metadata(cls, conversation_id: int) -> Optional[Dict[str, Any]]:
        """Lấy conversation metadata từ cache"""
        key = cls._get_cache_key(cls.CONVERSATION_METADATA, conversation_id=conversation_id)
        return cls.get_cache(key, dict)
    
    @classmethod
    def cache_health_data(cls, conversation_id: int, health_data: Dict[str, Any]) -> bool:
        """Cache health data"""
        key = cls._get_cache_key(cls.HEALTH_DATA, conversation_id=conversation_id)
        return cls.set_cache(key, health_data, cls.TTL_LONG)
    
    @classmethod
    def get_health_data(cls, conversation_id: int) -> Optional[Dict[str, Any]]:
        """Lấy health data từ cache"""
        key = cls._get_cache_key(cls.HEALTH_DATA, conversation_id=conversation_id)
        return cls.get_cache(key, dict)
    
    @classmethod
    def cache_recipe_data(cls, menu_id: int, recipe_data: Dict[str, Any]) -> bool:
        """Cache recipe data"""
        key = cls._get_cache_key(cls.RECIPE_DETAILS, menu_id=menu_id)
        return cls.set_cache(key, recipe_data, cls.TTL_MEDIUM)
    
    @classmethod
    def get_recipe_data(cls, menu_id: int) -> Optional[Dict[str, Any]]:
        """Lấy recipe data từ cache"""
        key = cls._get_cache_key(cls.RECIPE_DETAILS, menu_id=menu_id)
        return cls.get_cache(key, dict)
    
    @classmethod
    def cache_message_summary(cls, message_id: int, summary: str) -> bool:
        """Cache message summary"""
        key = cls._get_cache_key(cls.MESSAGE_SUMMARY, message_id=message_id)
        return cls.set_cache(key, summary, cls.TTL_LONG)
    
    @classmethod
    def get_message_summary(cls, message_id: int) -> Optional[str]:
        """Lấy message summary từ cache"""
        key = cls._get_cache_key(cls.MESSAGE_SUMMARY, message_id=message_id)
        return cls.get_cache(key, str)
    
    @classmethod
    def invalidate_conversation_cache(cls, conversation_id: int) -> None:
        """Invalidate tất cả cache liên quan đến conversation"""
        patterns = [
            f"conversation:{conversation_id}:*",
            "unsummarized_conversations:*"
        ]
        
        for pattern in patterns:
            cls.delete_pattern(pattern)
    
    @classmethod
    def invalidate_recipes_cache(cls) -> None:
        """Invalidate tất cả cache liên quan đến recipes"""
        patterns = [
            "recent_recipes:*",
            "recipe:*:details"
        ]
        
        for pattern in patterns:
            cls.delete_pattern(pattern)
    
    @classmethod
    def cache_batch_operation(cls, operation_type: str, data: Dict[str, Any], ttl: int = TTL_LONG) -> bool:
        """Cache kết quả của batch operations"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        key = f"{operation_type}:{timestamp}"
        batch_data = {
            'operation_type': operation_type,
            'timestamp': datetime.now().isoformat(),
            'data': data
        }
        return cls.set_cache(key, batch_data, ttl)
    
    @classmethod
    def get_conversation_owner(cls, conversation_id: int) -> Optional[int]:
        """Lấy owner của conversation từ cache"""
        key = cls._get_cache_key(cls.CONVERSATION_OWNER, conversation_id=conversation_id)
        return cls.get_cache(key, int)
    
    @classmethod
    def get_latest_conversation_id(cls, user_id: int) -> Optional[int]:
        """Lấy conversation ID mới nhất của user từ cache"""
        key = cls._get_cache_key(cls.USER_LATEST_CONVERSATION, user_id=user_id)
        return cls.get_cache(key, int)
    
    @classmethod
    def cache_unsummarized_conversations(cls, threshold: int, conversation_ids: List[int]) -> bool:
        """Cache danh sách conversation chưa summarized"""
        key = cls._get_cache_key(cls.UNSUMMARIZED_CONVERSATIONS, threshold=threshold)
        return cls.set_cache(key, conversation_ids, cls.TTL_SHORT)
    
    @classmethod
    def get_unsummarized_conversations(cls, threshold: int) -> Optional[List[int]]:
        """Lấy danh sách conversation chưa summarized từ cache"""
        key = cls._get_cache_key(cls.UNSUMMARIZED_CONVERSATIONS, threshold=threshold)
        return cls.get_cache(key, list)

    @classmethod
    def get_or_rebuild_cache(cls, key: str, rebuild_func: callable, expected_type: type = None, ttl: int = None) -> Any:
        """
        Lấy data từ cache, nếu không có thì rebuild bằng function
        Đảm bảo consistency với distributed locking
        
        Args:
            key: Cache key
            rebuild_func: Function để rebuild cache (không có params)
            expected_type: Kiểu dữ liệu mong đợi
            ttl: Time to live, mặc định là TTL_MEDIUM
            
        Returns:
            Dữ liệu từ cache hoặc sau khi rebuild
        """
        try:
            # Thử lấy từ cache trước
            cached_data = cls.get_cache(key, expected_type)
            if cached_data is not None:
                return cached_data
            
            # Nếu không có cache, sử dụng distributed lock để rebuild
            lock_key = f"lock:{key}"
            lock_ttl = 30  # 30 seconds lock timeout
            
            # Thử acquire lock
            if redis_client.set(lock_key, "1", nx=True, ex=lock_ttl):
                try:
                    logger.debug(f"Acquired lock để rebuild cache: {key}")
                    
                    # Double check - có thể thread khác đã rebuild
                    cached_data = cls.get_cache(key, expected_type)
                    if cached_data is not None:
                        return cached_data
                    
                    # Rebuild cache
                    rebuilt_data = rebuild_func()
                    
                    if rebuilt_data is not None:
                        # Cache với TTL
                        used_ttl = ttl or cls.TTL_MEDIUM
                        success = cls.set_cache(key, rebuilt_data, used_ttl)
                        if success:
                            logger.info(f"✅ Đã rebuild cache thành công: {key}")
                            return rebuilt_data
                        else:
                            logger.error(f"❌ Lỗi khi set rebuilt cache: {key}")
                    
                    return rebuilt_data
                    
                finally:
                    # Release lock
                    redis_client.delete(lock_key)
                    logger.debug(f"Released lock cho cache: {key}")
            else:
                # Không lấy được lock - có thread khác đang rebuild
                # Wait và retry một lần
                import time
                time.sleep(0.1)  # Wait 100ms
                
                cached_data = cls.get_cache(key, expected_type)
                if cached_data is not None:
                    logger.debug(f"Cache đã được rebuild bởi thread khác: {key}")
                    return cached_data
                
                # Nếu vẫn không có, fallback to direct rebuild
                logger.warning(f"Fallback rebuild cho cache: {key}")
                return rebuild_func()
                
        except Exception as e:
            logger.error(f"Lỗi trong get_or_rebuild_cache: {str(e)}")
            # Fallback to direct rebuild
            try:
                return rebuild_func()
            except Exception as rebuild_error:
                logger.error(f"Lỗi khi fallback rebuild: {str(rebuild_error)}")
                return None


def cache_with_fallback(cache_key_func, fallback_func, ttl: int = CacheService.TTL_MEDIUM, data_type: type = dict):
    """
    Decorator để tự động cache kết quả với fallback to database
    
    Args:
        cache_key_func: Function để tạo cache key từ args
        fallback_func: Function để lấy data từ DB nếu cache miss
        ttl: Time to live cho cache
        data_type: Kiểu dữ liệu để deserialize
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Tạo cache key
            cache_key = cache_key_func(*args, **kwargs)
            
            # Thử lấy từ cache trước
            cached_result = CacheService.get_cache(cache_key, data_type)
            if cached_result is not None:
                return cached_result
            
            # Cache miss - lấy từ DB
            try:
                result = fallback_func(*args, **kwargs)
                if result is not None:
                    # Cache kết quả
                    CacheService.set_cache(cache_key, result, ttl)
                return result
            except Exception as e:
                logger.error(f"Lỗi fallback function: {str(e)}")
                return None
        
        return wrapper
    return decorator


def invalidate_cache_on_update(cache_patterns: List[str]):
    """
    Decorator để tự động invalidate cache khi update data
    
    Args:
        cache_patterns: List các pattern cache cần xóa
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Thực hiện update
            result = func(*args, **kwargs)
            
            # Invalidate cache nếu update thành công
            if result:
                for pattern in cache_patterns:
                    # Format pattern với args nếu cần
                    try:
                        formatted_pattern = pattern.format(*args, **kwargs)
                        CacheService.delete_pattern(formatted_pattern)
                    except (IndexError, KeyError):
                        # Nếu không format được, xóa trực tiếp
                        CacheService.delete_pattern(pattern)
            
            return result
        
        return wrapper
    return decorator 