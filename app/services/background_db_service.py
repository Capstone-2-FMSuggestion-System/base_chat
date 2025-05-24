import logging
import asyncio
from typing import Dict, List, Any, Optional, Callable, Tuple
from sqlalchemy.orm import Session
from dataclasses import dataclass
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import threading

logger = logging.getLogger(__name__)

@dataclass
class DBTask:
    """Represents a database task to be executed in background"""
    task_id: str
    operation: str
    data: Dict[str, Any]
    callback: Optional[Callable] = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

class BackgroundDBService:
    """
    Service quản lý các tác vụ DB background để tối ưu hóa I/O bất đồng bộ.
    Sử dụng ThreadPoolExecutor để chạy các tác vụ DB đồng bộ trong background.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(BackgroundDBService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="bg_db")
            self.pending_tasks: Dict[str, DBTask] = {}
            self.completed_tasks: Dict[str, Dict[str, Any]] = {}
            self.initialized = True
            logger.info("🔧 BackgroundDBService initialized with ThreadPoolExecutor")
    
    def add_message_task(self, conversation_id: int, role: str, content: str, 
                        repository_instance: Any) -> str:
        """Tạo task để add message vào DB"""
        task_id = f"add_msg_{conversation_id}_{role}_{datetime.now().timestamp()}"
        
        task = DBTask(
            task_id=task_id,
            operation="add_message",
            data={
                "conversation_id": conversation_id,
                "role": role,
                "content": content,
                "repository": repository_instance
            }
        )
        
        self.pending_tasks[task_id] = task
        logger.debug(f"🔄 Tạo task add_message: {task_id}")
        return task_id
    
    def save_conversation_summary_task(self, conversation_id: int, assistant_message_id: int,
                                     summary_text: str, repository_instance: Any) -> str:
        """Tạo task để save conversation summary vào DB"""
        task_id = f"save_summary_{conversation_id}_{assistant_message_id}_{datetime.now().timestamp()}"
        
        task = DBTask(
            task_id=task_id,
            operation="save_conversation_summary",
            data={
                "conversation_id": conversation_id,
                "assistant_message_id": assistant_message_id,
                "summary_text": summary_text,
                "repository": repository_instance
            }
        )
        
        self.pending_tasks[task_id] = task
        logger.debug(f"🔄 Tạo task save_conversation_summary: {task_id}")
        return task_id
    
    def save_health_data_task(self, conversation_id: int, user_id: int,
                             health_data: Dict[str, Any], repository_instance: Any) -> str:
        """Tạo task để save health data vào DB"""
        task_id = f"save_health_{conversation_id}_{user_id}_{datetime.now().timestamp()}"
        
        task = DBTask(
            task_id=task_id,
            operation="save_health_data",
            data={
                "conversation_id": conversation_id,
                "user_id": user_id,
                "health_data": health_data,
                "repository": repository_instance
            }
        )
        
        self.pending_tasks[task_id] = task
        logger.debug(f"🔄 Tạo task save_health_data: {task_id}")
        return task_id
    
    def execute_task(self, task_id: str) -> None:
        """Execute một DB task cụ thể trong background"""
        if task_id not in self.pending_tasks:
            logger.error(f"❌ Task {task_id} không tồn tại trong pending_tasks")
            return
        
        task = self.pending_tasks[task_id]
        
        future = self.executor.submit(self._execute_db_operation, task)
        
        def task_completion_callback(fut):
            try:
                result = fut.result()
                self.completed_tasks[task_id] = {
                    "status": "success",
                    "result": result,
                    "completed_at": datetime.now()
                }
                logger.info(f"✅ Background task hoàn thành: {task_id}")
            except Exception as e:
                logger.error(f"💥 Background task thất bại: {task_id}, lỗi: {str(e)}")
                self.completed_tasks[task_id] = {
                    "status": "error",
                    "error": str(e),
                    "completed_at": datetime.now()
                }
            finally:
                self.pending_tasks.pop(task_id, None)
        
        future.add_done_callback(task_completion_callback)
    
    def _execute_db_operation(self, task: DBTask) -> Any:
        """Thực hiện thao tác DB cụ thể"""
        operation = task.operation
        data = task.data
        repository = data["repository"]
        
        try:
            if operation == "add_message":
                result = repository.add_message(
                    conversation_id=data["conversation_id"],
                    role=data["role"],
                    content=data["content"]
                )
                logger.debug(f"📝 Background add_message thành công: message_id={result.message_id}")
                return {"message_id": result.message_id}
                
            elif operation == "save_conversation_summary":
                result = repository.save_conversation_summary(
                    conversation_id=data["conversation_id"],
                    assistant_message_id=data["assistant_message_id"],
                    summary_text=data["summary_text"]
                )
                logger.debug(f"📝 Background save_conversation_summary thành công: {result}")
                return {"success": result}
                
            elif operation == "save_health_data":
                health_data = data["health_data"]
                result = repository.save_health_data(
                    conversation_id=data["conversation_id"],
                    user_id=data["user_id"],
                    **health_data
                )
                logger.debug(f"📝 Background save_health_data thành công: {result.id if result else 'None'}")
                return {"health_data_id": result.id if result else None}
                
            else:
                raise ValueError(f"Unknown operation: {operation}")
                
        except Exception as e:
            logger.error(f"💥 Lỗi thực hiện DB operation {operation}: {str(e)}")
            raise
    
    def execute_multiple_tasks(self, task_ids: List[str]) -> None:
        """Execute nhiều DB tasks cùng lúc"""
        for task_id in task_ids:
            self.execute_task(task_id)
        logger.info(f"🚀 Đã khởi động {len(task_ids)} background DB tasks")
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Lấy trạng thái của một task"""
        if task_id in self.pending_tasks:
            return {"status": "pending", "created_at": self.pending_tasks[task_id].created_at}
        elif task_id in self.completed_tasks:
            return self.completed_tasks[task_id]
        else:
            return None
    
    def cleanup_completed_tasks(self, max_age_hours: int = 24) -> int:
        """Cleanup các completed tasks cũ"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        removed_count = 0
        
        for task_id in list(self.completed_tasks.keys()):
            if self.completed_tasks[task_id]["completed_at"] < cutoff_time:
                self.completed_tasks.pop(task_id)
                removed_count += 1
        
        if removed_count > 0:
            logger.info(f"🧹 Cleaned up {removed_count} completed background tasks")
        
        return removed_count
    
    def shutdown(self):
        """Shutdown executor gracefully"""
        logger.info("🔄 Shutting down BackgroundDBService...")
        self.executor.shutdown(wait=True)
        logger.info("✅ BackgroundDBService shutdown completed")

# Global instance
background_db_service = BackgroundDBService() 