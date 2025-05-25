"""
Gemini Model Pool để quản lý thread-safe access tới multiple API keys
Giải quyết vấn đề genai.configure() global state
"""

import threading
import logging
from typing import Dict, Optional
import google.generativeai as genai
from app.services.api_key_manager import get_api_key_manager

logger = logging.getLogger(__name__)

class GeminiModelPool:
    """
    Pool các GenerativeModel instances, mỗi instance được cấu hình với 1 API key riêng
    Thread-safe và tránh xung đột global genai.configure()
    """
    
    def __init__(self, model_name: str = "gemini-2.0-flash-lite"):
        self.model_name = model_name
        self.model_pool: Dict[str, genai.GenerativeModel] = {}
        self.pool_lock = threading.Lock()
        self.api_key_manager = get_api_key_manager()
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Khởi tạo pool với tất cả API keys có sẵn"""
        if not self.api_key_manager.is_healthy():
            logger.error("❌ Không thể khởi tạo GeminiModelPool: ApiKeyManager không healthy")
            return
        
        all_keys = self.api_key_manager.get_all_keys()
        logger.info(f"🏗️ Khởi tạo GeminiModelPool với {len(all_keys)} keys...")
        
        for i, api_key in enumerate(all_keys, 1):
            try:
                # Cấu hình genai cho key này
                genai.configure(api_key=api_key)
                
                # Tạo model instance
                model_instance = genai.GenerativeModel(self.model_name)
                
                # Lưu vào pool với key làm identifier
                key_id = f"key_{i}"  # Sử dụng index thay vì API key để bảo mật
                self.model_pool[key_id] = {
                    'model': model_instance,
                    'api_key': api_key,
                    'usage_count': 0
                }
                
                masked_key = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "***"
                logger.info(f"✅ Pool {key_id}: Đã tạo model cho key {masked_key}")
                
            except Exception as e:
                masked_key = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "***"
                logger.error(f"❌ Lỗi tạo model cho key {masked_key}: {e}")
        
        logger.info(f"🎯 GeminiModelPool đã sẵn sàng với {len(self.model_pool)} models")
    
    def get_model_for_key(self, api_key: str) -> Optional[genai.GenerativeModel]:
        """
        Lấy model instance cho một API key cụ thể
        
        Args:
            api_key: API key cần lấy model
            
        Returns:
            GenerativeModel instance hoặc None nếu không tìm thấy
        """
        with self.pool_lock:
            for key_id, pool_item in self.model_pool.items():
                if pool_item['api_key'] == api_key:
                    pool_item['usage_count'] += 1
                    return pool_item['model']
        
        logger.warning(f"⚠️ Không tìm thấy model cho API key ...{api_key[-4:]}")
        return None
    
    def get_next_model(self) -> tuple[Optional[genai.GenerativeModel], Optional[str]]:
        """
        Lấy model tiếp theo theo round-robin (sử dụng ApiKeyManager)
        
        Returns:
            Tuple (model_instance, api_key) hoặc (None, None) nếu không có
        """
        api_key = self.api_key_manager.get_next_key()
        if not api_key:
            logger.warning("⚠️ Không có API key từ ApiKeyManager")
            return None, None
        
        model = self.get_model_for_key(api_key)
        return model, api_key
    
    def get_pool_statistics(self) -> dict:
        """Lấy thống kê sử dụng pool"""
        with self.pool_lock:
            stats = {}
            for key_id, pool_item in self.model_pool.items():
                masked_key = f"{pool_item['api_key'][:4]}...{pool_item['api_key'][-4:]}"
                stats[masked_key] = pool_item['usage_count']
            return stats
    
    def total_models(self) -> int:
        """Số lượng models trong pool"""
        return len(self.model_pool)

# Global instance để dùng chung
_gemini_model_pool = None
_pool_lock = threading.Lock()

def get_gemini_model_pool(model_name: str = "gemini-2.0-flash-lite") -> GeminiModelPool:
    """
    Factory function để lấy global instance của GeminiModelPool
    
    Args:
        model_name: Tên model Gemini (default: gemini-2.0-flash-lite)
        
    Returns:
        GeminiModelPool instance
    """
    global _gemini_model_pool
    
    if _gemini_model_pool is None:
        with _pool_lock:
            if _gemini_model_pool is None:  # Double-checked locking
                _gemini_model_pool = GeminiModelPool(model_name)
    
    return _gemini_model_pool

if __name__ == "__main__":
    # Test code
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    
    print("🧪 Testing GeminiModelPool...")
    
    pool = get_gemini_model_pool()
    print(f"📊 Pool size: {pool.total_models()}")
    
    # Test get_next_model
    for i in range(5):
        model, api_key = pool.get_next_model()
        if model:
            masked_key = f"...{api_key[-4:]}" if api_key else "None"
            print(f"Test {i+1}: Model OK, Key: {masked_key}")
        else:
            print(f"Test {i+1}: Model None")
    
    print(f"📈 Pool statistics: {pool.get_pool_statistics()}") 