"""
API Key Manager Service cho Gemini API
Quản lý và cung cấp API keys theo cơ chế round-robin với thread safety
"""

import os
import threading
import logging
from typing import List, Optional
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class ApiKeyManager:
    """
    Singleton service để quản lý và xoay vòng API keys của Gemini
    Thread-safe và hỗ trợ round-robin key rotation
    """
    
    _instance = None
    _lock = threading.Lock()  # Lock cho việc khởi tạo singleton

    def __init__(self):
        """Private constructor - sử dụng get_instance() để lấy singleton instance"""
        if hasattr(self, "_initialized"):
            return
        
        self.api_keys: List[str] = []
        self.current_index: int = 0
        self.index_lock = threading.Lock()  # Lock cho việc truy cập current_index
        
        # Thống kê sử dụng keys (optional, để monitoring)
        self.key_usage_count = {}
        self.last_rotation_time = datetime.now()
        
        self._load_keys()
        self._validate_keys()
        self._initialized = True

    def _load_keys(self):
        """Đọc API keys từ biến môi trường theo nhiều cách khác nhau"""
        logger.info("🔑 Đang tải API keys của Gemini từ biến môi trường...")
        
        # Cách 1: Đọc từ danh sách phân tách bằng dấu phẩy
        keys_str = os.getenv("GEMINI_API_KEYS_LIST")
        if keys_str:
            raw_keys = [key.strip() for key in keys_str.split(',') if key.strip()]
            self.api_keys.extend(raw_keys)
            logger.info(f"✅ Tải được {len(raw_keys)} keys từ GEMINI_API_KEYS_LIST")
        
        # Cách 2: Đọc các key riêng lẻ GEMINI_API_KEY_1, GEMINI_API_KEY_2, ...
        if not self.api_keys:
            i = 1
            while True:
                key = os.getenv(f"GEMINI_API_KEY_{i}")
                if key and key.strip():
                    self.api_keys.append(key.strip())
                    i += 1
                else:
                    break
            
            if self.api_keys:
                logger.info(f"✅ Tải được {len(self.api_keys)} keys từ GEMINI_API_KEY_X pattern")
        
        # Cách 3: Fallback về key đơn lẻ GEMINI_API_KEY (backward compatibility)
        if not self.api_keys:
            single_key = os.getenv("GEMINI_API_KEY")
            if single_key and single_key.strip():
                self.api_keys.append(single_key.strip())
                logger.info("✅ Tải được 1 key từ GEMINI_API_KEY (fallback)")
        
        # Loại bỏ duplicates và keys rỗng
        unique_keys = []
        seen = set()
        for key in self.api_keys:
            if key and key not in seen:
                unique_keys.append(key)
                seen.add(key)
        
        self.api_keys = unique_keys
        
        # Khởi tạo usage counter
        for key in self.api_keys:
            self.key_usage_count[key] = 0

    def _validate_keys(self):
        """Validate và log thông tin về API keys"""
        if not self.api_keys:
            error_msg = (
                "❌ KHÔNG CÓ API KEY NÀO CỦA GEMINI ĐƯỢC CẤU HÌNH!\n"
                "Vui lòng set một trong các biến môi trường sau:\n"
                "- GEMINI_API_KEYS_LIST='key1,key2,key3'\n"
                "- GEMINI_API_KEY_1='key1', GEMINI_API_KEY_2='key2', ...\n"
                "- GEMINI_API_KEY='single_key' (fallback)"
            )
            logger.error(error_msg)
            # Có thể raise exception nếu muốn app dừng lại khi không có key
            # raise ValueError("Không có API key nào của Gemini được cấu hình")
        else:
            logger.info(f"🎯 ApiKeyManager đã sẵn sàng với {len(self.api_keys)} API keys")
            for i, key in enumerate(self.api_keys, 1):
                # Log an toàn - chỉ hiển thị 4 ký tự đầu và 4 ký tự cuối
                masked_key = f"{key[:4]}...{key[-4:]}" if len(key) > 8 else "***"
                logger.info(f"  Key {i}: {masked_key}")

    @classmethod
    def get_instance(cls) -> 'ApiKeyManager':
        """
        Lấy singleton instance của ApiKeyManager
        Thread-safe implementation
        """
        if not cls._instance:
            with cls._lock:
                if not cls._instance:  # Double-checked locking
                    cls._instance = cls()
        return cls._instance

    def get_next_key(self) -> Optional[str]:
        """
        Lấy API key tiếp theo theo cơ chế round-robin
        Thread-safe và tự động xoay vòng
        
        Returns:
            str: API key tiếp theo, hoặc None nếu không có key nào
        """
        if not self.api_keys:
            logger.warning("⚠️ Không có API key nào để trả về từ ApiKeyManager")
            return None

        with self.index_lock:  # Thread-safe access
            key_to_return = self.api_keys[self.current_index]
            
            # Cập nhật usage counter
            self.key_usage_count[key_to_return] += 1
            
            # Round-robin: chuyển sang key tiếp theo
            self.current_index = (self.current_index + 1) % len(self.api_keys)
            
            # Log rotation info (chỉ khi quay về key đầu tiên)
            if self.current_index == 0:
                time_since_last = datetime.now() - self.last_rotation_time
                logger.debug(f"🔄 Hoàn thành chu kỳ rotation ({time_since_last.total_seconds():.1f}s)")
                self.last_rotation_time = datetime.now()

        # Log cẩn thận - không log toàn bộ key
        masked_key = f"{key_to_return[:4]}...{key_to_return[-4:]}" if len(key_to_return) > 8 else "***"
        logger.debug(f"🔑 Cung cấp key: {masked_key} (usage: {self.key_usage_count[key_to_return]})")
        
        return key_to_return

    def get_all_keys(self) -> List[str]:
        """
        Trả về bản sao của danh sách tất cả API keys
        
        Returns:
            List[str]: Danh sách tất cả API keys
        """
        return list(self.api_keys)

    def total_keys(self) -> int:
        """
        Số lượng API keys hiện có
        
        Returns:
            int: Tổng số API keys
        """
        return len(self.api_keys)

    def get_current_key_index(self) -> int:
        """
        Lấy index của key sẽ được sử dụng tiếp theo
        
        Returns:
            int: Index hiện tại
        """
        with self.index_lock:
            return self.current_index

    def get_usage_statistics(self) -> dict:
        """
        Lấy thống kê sử dụng các API keys
        
        Returns:
            dict: Thống kê usage count cho từng key
        """
        with self.index_lock:
            # Return masked statistics for security
            masked_stats = {}
            for key, count in self.key_usage_count.items():
                masked_key = f"{key[:4]}...{key[-4:]}" if len(key) > 8 else "***"
                masked_stats[masked_key] = count
            return masked_stats

    def reset_usage_statistics(self):
        """Reset bộ đếm sử dụng keys"""
        with self.index_lock:
            for key in self.key_usage_count:
                self.key_usage_count[key] = 0
            logger.info("📊 Đã reset usage statistics cho tất cả API keys")

    def is_healthy(self) -> bool:
        """
        Kiểm tra trạng thái healthy của ApiKeyManager
        
        Returns:
            bool: True nếu có ít nhất 1 API key
        """
        return len(self.api_keys) > 0

# ===== GLOBAL INSTANCE =====
# Tạo instance toàn cục để dễ sử dụng trong toàn bộ ứng dụng
# Sẽ được khởi tạo lazy khi module này được import lần đầu
api_key_manager = None

def get_api_key_manager() -> ApiKeyManager:
    """
    Factory function để lấy global instance của ApiKeyManager
    
    Returns:
        ApiKeyManager: Global singleton instance
    """
    global api_key_manager
    if api_key_manager is None:
        api_key_manager = ApiKeyManager.get_instance()
    return api_key_manager

# ===== TESTING & DEBUGGING =====
if __name__ == "__main__":
    # Test code - chỉ chạy khi file được execute trực tiếp
    import sys
    
    # Setup logging for testing
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)s - %(message)s')
    
    print("🧪 Testing ApiKeyManager...")
    
    # Test với environment variables giả lập
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Simulate environment variables
        os.environ["GEMINI_API_KEYS_LIST"] = "test_key_A,test_key_B,test_key_C"
        
        print("\n1. Testing với GEMINI_API_KEYS_LIST")
        manager = ApiKeyManager.get_instance()
        
        if manager.is_healthy():
            print(f"✅ Tổng số keys: {manager.total_keys()}")
            print("🔄 Testing round-robin rotation:")
            
            # Test rotation
            for i in range(manager.total_keys() * 2 + 1):
                key = manager.get_next_key()
                print(f"  Lần {i+1}: {key}")
            
            print(f"\n📊 Usage statistics: {manager.get_usage_statistics()}")
        else:
            print("❌ ApiKeyManager không healthy")
    else:
        print("Để test, chạy: python api_key_manager.py --test") 