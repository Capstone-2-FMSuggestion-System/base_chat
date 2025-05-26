import logging
from sqlalchemy.orm import Session
from typing import Dict, List, Optional, Any, Union, Tuple
from fastapi import HTTPException
import asyncio
from datetime import datetime
from sqlalchemy import desc

from app.repositories.chat_repository import ChatRepository
from app.services.llm_service_factory import LLMServiceFactory
from app.services.gemini_prompt_service import GeminiPromptService
from app.services.chat_flow import run_chat_flow
from app.services.background_db_service import background_db_service
from app.services.product_service import ProductService
from app.config import settings
from app.db.models import Message
from app.schemas.chat import ChatResponse, NewChatResponse


logger = logging.getLogger(__name__)


class ChatService:
    """
    ChatService - Kiến trúc sư Hệ thống Chatbot
    
    Chịu trách nhiệm điều phối toàn bộ quy trình xử lý tin nhắn,
    bao gồm tạo và lưu trữ tóm tắt tăng dần sau mỗi lượt tương tác.
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.repository = ChatRepository(db)
        self.llm_service = LLMServiceFactory(
            service_type=settings.LLM_SERVICE_TYPE,
            llama_url=settings.LLAMA_CPP_URL,
            ollama_url=settings.OLLAMA_URL
        )
        self.gemini_service = GeminiPromptService()
        self.product_service = ProductService()
        self.medichat_model = "medichat-llama3:8b_q4_K_M"  # Model Medichat từ Ollama
    
    async def process_message(self, user_id: int, message_content: str, conversation_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Điều phối quy trình xử lý tin nhắn với tóm tắt tăng dần.
        
        Flow:
        1. Chuẩn bị conversation và lấy lịch sử chat
        2. Gọi run_chat_flow (LangGraph) để xử lý logic chính
        3. Kiểm tra điều kiện và tạo tóm tắt tăng dần
        4. Lưu tóm tắt và trả về kết quả hoàn chỉnh
        
        Args:
            user_id: ID người dùng
            message_content: Nội dung tin nhắn
            conversation_id: ID cuộc trò chuyện (optional)
            
        Returns:
            Dict chứa kết quả xử lý và tóm tắt hiện tại
        """
        # Bước 1: Chuẩn bị conversation
        if not conversation_id:
            conversation = self.repository.get_latest_conversation(user_id)
            if not conversation:
                # Tạo conversation mới và trả về với welcome message
                return await self._handle_new_conversation(user_id, message_content)
            conversation_id = conversation.conversation_id
        else:
            conversation = self.repository.get_conversation_by_id(conversation_id)
            if not conversation or conversation.user_id != user_id:
                raise ValueError("Cuộc trò chuyện không hợp lệ hoặc không có quyền truy cập.")

        # Bước 2: Lấy lịch sử chat TRƯỚC khi tin nhắn hiện tại được thêm
        chat_history_before_current_message = self.repository.get_messages(conversation_id)
        logger.info(f"🔄 Bắt đầu xử lý tin nhắn cho conversation_id={conversation_id}, user_id={user_id}")

        try:
            # Bước 3: Gọi LangGraph để xử lý logic chính
            langgraph_result = await run_chat_flow(
                user_message=message_content,
                user_id=user_id,
                conversation_id=conversation_id,
                messages=chat_history_before_current_message,
                repository=self.repository,
                llm_service=self.llm_service
            )
            
            # Bước 4: Xử lý sản phẩm có sẵn nếu có menu được tạo
            await self._handle_available_products(langgraph_result)
            
            # Bước 5: Xử lý tóm tắt tăng dần
            current_summary = await self._handle_incremental_summary(
                conversation_id=conversation_id,
                message_content=message_content,
                langgraph_result=langgraph_result
            )
            
            # Bước 6: Chuẩn bị và trả về response
            response_payload = self._build_response_payload(
                conversation_id=conversation_id,
                message_content=message_content,
                langgraph_result=langgraph_result,
                current_summary=current_summary
            )
            
            logger.info(f"✅ Xử lý thành công tin nhắn cho conversation_id={conversation_id}")
            return response_payload
            
        except Exception as e:
            logger.error(f"💥 Lỗi nghiêm trọng khi xử lý tin nhắn (ChatService): {str(e)}", exc_info=True)
            return await self._handle_error_response(conversation_id, message_content, e)

    async def _handle_available_products(self, langgraph_result: Dict[str, Any]) -> None:
        """
        Xử lý việc lấy thông tin sản phẩm có sẵn sau khi lưu menu.
        
        Args:
            langgraph_result: Kết quả từ LangGraph chứa thông tin menu đã tạo
        """
        try:
            # Kiểm tra xem có menu_ids được tạo không
            menu_ids = langgraph_result.get("menu_ids", [])
            if not menu_ids:
                logger.debug("Không có menu nào được tạo, bỏ qua việc lấy sản phẩm")
                return
            
            logger.info(f"🛒 Bắt đầu lấy sản phẩm có sẵn cho {len(menu_ids)} menu")
            
            # Danh sách tất cả sản phẩm có sẵn từ tất cả menu
            all_available_products = []
            processed_product_ids = set()  # Tránh trùng lặp sản phẩm
            
            # Xử lý từng menu
            for menu_id in menu_ids:
                try:
                    # Lấy thông tin recipe từ repository
                    recipe_data = self.repository.get_recipe_by_id(menu_id)
                    if not recipe_data:
                        logger.warning(f"Không tìm thấy recipe với menu_id={menu_id}")
                        continue
                    
                    # Lấy danh sách ingredients có product_id
                    ingredients = recipe_data.get('ingredients', [])
                    logger.debug(f"📋 Menu {menu_id} có {len(ingredients)} ingredients")
                    
                    # Lấy sản phẩm có sẵn cho menu này
                    available_products = await self.product_service.get_available_products_from_menu_items(ingredients)
                    
                    if available_products:
                        # Lọc ra sản phẩm chưa được xử lý (tránh trùng lặp)
                        new_products = []
                        for product in available_products:
                            product_id = product.get('id')
                            if product_id and product_id not in processed_product_ids:
                                new_products.append(product)
                                processed_product_ids.add(product_id)
                        
                        all_available_products.extend(new_products)
                        
                        # Cập nhật cache cho recipe này
                        recipe_data['available_products'] = available_products
                        from app.services.cache_service import CacheService
                        CacheService.cache_recipe_data(menu_id, recipe_data)
                        
                        logger.info(f"✅ Menu {menu_id}: Thêm {len(new_products)} sản phẩm mới ({len(available_products)} total)")
                    else:
                        logger.debug(f"❌ Menu {menu_id}: Không có sản phẩm nào có sẵn")
                        
                except Exception as e:
                    logger.error(f"💥 Lỗi khi xử lý sản phẩm cho menu_id={menu_id}: {str(e)}")
                    continue
            
            # Thêm tất cả sản phẩm có sẵn vào langgraph_result
            if all_available_products:
                langgraph_result['available_products'] = all_available_products
                logger.info(f"🎯 Tổng cộng: {len(all_available_products)} sản phẩm có sẵn được thêm vào response")
                
                # Log chi tiết các sản phẩm
                for product in all_available_products:
                    logger.debug(f"🛍️ Available product: {product.get('name')} (ID: {product.get('id')}) - Stock: {product.get('stock_quantity')}")
            else:
                logger.info("ℹ️ Không có sản phẩm nào có sẵn trong kho cho các menu được tạo")
                langgraph_result['available_products'] = []
                    
        except Exception as e:
            logger.error(f"💥 Lỗi nghiêm trọng khi xử lý sản phẩm có sẵn: {str(e)}", exc_info=True)
            # Đảm bảo luôn có key này trong response
            langgraph_result['available_products'] = []

    async def _handle_new_conversation(self, user_id: int, message_content: str) -> Dict[str, Any]:
        """
        Xử lý tin nhắn đầu tiên trong conversation mới.
        
        Returns:
            Dict với conversation_id mới và welcome message
        """
        try:
            conversation = self.repository.create_conversation(user_id)
            conversation_id = conversation.conversation_id
            
            # Tạo welcome message
            welcome_message_content = await self.gemini_service.generate_welcome_message()
            self.repository.add_message(conversation_id, "assistant", welcome_message_content)
            
            # Thêm tin nhắn user
            self.repository.add_message(conversation_id, "user", message_content)
            
            logger.info(f"🆕 Tạo conversation mới: {conversation_id} cho user_id={user_id}")
            
            return {
                "conversation_id": conversation_id,
                "user_message": {"role": "user", "content": message_content},
                "assistant_message": {"role": "assistant", "content": welcome_message_content},
                "current_summary": None,
                "is_new_conversation": True
            }
            
        except Exception as e:
            logger.error(f"💥 Lỗi khi tạo conversation mới: {str(e)}", exc_info=True)
            raise

    async def _handle_incremental_summary(self, conversation_id: int, message_content: str, langgraph_result: Dict[str, Any]) -> Optional[str]:
        """
        Xử lý việc tạo và lưu tóm tắt tăng dần.
        
        Logic:
        - Chỉ tạo tóm tắt nếu: is_valid_scope=True, need_more_info=False, có final_response
        - Cần có user_message_id_db và assistant_message_id_db từ LangGraph
        
        Args:
            conversation_id: ID cuộc trò chuyện
            message_content: Nội dung tin nhắn user
            langgraph_result: Kết quả từ LangGraph
            
        Returns:
            Tóm tắt hiện tại (mới hoặc cũ)
        """
        # Lấy message IDs từ LangGraph result
        user_message_id = langgraph_result.get("user_message_id_db")
        assistant_message_id = langgraph_result.get("assistant_message_id_db")

        if not user_message_id or not assistant_message_id:
            logger.warning(f"⚠️ Không nhận được message IDs từ chat_flow (user: {user_message_id}, assistant: {assistant_message_id}) - bỏ qua tạo tóm tắt")
            return self.repository.get_latest_summary(conversation_id)

        # Kiểm tra điều kiện tạo tóm tắt
        should_create_summary = (
            langgraph_result.get("is_valid_scope", False) and 
            not langgraph_result.get("need_more_info", True) and
            langgraph_result.get("final_response")
        )

        if not should_create_summary:
            logger.debug(f"🔍 Không đủ điều kiện tạo tóm tắt cho conversation_id={conversation_id}")
            logger.debug(f"   - is_valid_scope: {langgraph_result.get('is_valid_scope')}")
            logger.debug(f"   - need_more_info: {langgraph_result.get('need_more_info')}")
            logger.debug(f"   - has_final_response: {bool(langgraph_result.get('final_response'))}")
            return self.repository.get_latest_summary(conversation_id)

        # Tạo tóm tắt tăng dần
        try:
            logger.info(f"📝 Bắt đầu tạo tóm tắt tăng dần cho conversation_id={conversation_id}")
            
            # Lấy dữ liệu cần thiết cho tóm tắt
            previous_summary_text = self.repository.get_latest_summary(conversation_id)
            summary_context_messages = self.repository.get_messages_for_summary_context(conversation_id, limit=3)
            final_response = langgraph_result.get("final_response", "")

            logger.debug(f"   - Previous summary: {'Có' if previous_summary_text else 'Không'} ({len(previous_summary_text or '')} ký tự)")
            logger.debug(f"   - Context messages: {len(summary_context_messages)} tin nhắn")
            logger.debug(f"   - Final response: {len(final_response)} ký tự")

            # Gọi Gemini để tạo tóm tắt
            new_summary_text = await self.gemini_service.create_incremental_summary(
                previous_summary=previous_summary_text,
                new_user_message=message_content,
                new_assistant_message=final_response,
                full_chat_history_for_context=summary_context_messages
            )

            if not new_summary_text:
                logger.warning(f"⚠️ Gemini trả về summary rỗng cho conversation_id={conversation_id}")
                return previous_summary_text

            # Lưu tóm tắt vào DB
            save_success = self.repository.save_conversation_summary(
                conversation_id=conversation_id,
                assistant_message_id=assistant_message_id,
                summary_text=new_summary_text
            )

            if save_success:
                logger.info(f"💾 Đã lưu tóm tắt mới: {len(new_summary_text)} ký tự cho conversation_id={conversation_id}")
                return new_summary_text
            else:
                logger.error(f"❌ Không thể lưu tóm tắt cho conversation_id={conversation_id} - giữ nguyên tóm tắt cũ")
                return previous_summary_text

        except Exception as e:
            logger.error(f"💥 Lỗi khi tạo tóm tắt tăng dần cho conversation_id={conversation_id}: {str(e)}", exc_info=True)
            # Fallback: trả về tóm tắt cũ nếu có lỗi
            return self.repository.get_latest_summary(conversation_id)

    def _build_response_payload(self, conversation_id: int, message_content: str, langgraph_result: Dict[str, Any], current_summary: Optional[str]) -> Dict[str, Any]:
        """
        Xây dựng response payload hoàn chỉnh.
        
        Args:
            conversation_id: ID cuộc trò chuyện
            message_content: Nội dung tin nhắn user gốc
            langgraph_result: Kết quả từ LangGraph
            current_summary: Tóm tắt hiện tại
            
        Returns:
            Dict response hoàn chỉnh
        """
        response_payload = {
            "conversation_id": conversation_id,
            "user_message": langgraph_result.get("user_message", {"role": "user", "content": message_content}),
            "assistant_message": langgraph_result.get("assistant_message", {
                "role": "assistant", 
                "content": langgraph_result.get("final_response", "")
            }),
            "current_summary": current_summary
        }
        
        # Thêm các metadata từ LangGraph
        metadata_keys = [
            "is_valid_scope", "need_more_info", "is_food_related", 
            "user_rejected_info", "suggest_general_options", 
            "limit_reached", "message_count", "available_products"
        ]
        
        for key in metadata_keys:
            if key in langgraph_result:
                response_payload[key] = langgraph_result[key]
        
        return response_payload

    async def _handle_error_response(self, conversation_id: Optional[int], message_content: str, error: Exception) -> Dict[str, Any]:
        """
        Xử lý response khi có lỗi nghiêm trọng.
        
        Args:
            conversation_id: ID cuộc trò chuyện (có thể None)
            message_content: Nội dung tin nhắn user
            error: Exception đã xảy ra
            
        Returns:
            Dict error response an toàn
        """
        error_response_content = "Xin lỗi, hệ thống gặp lỗi khi xử lý yêu cầu của bạn. Vui lòng thử lại sau."
        
        # Cố gắng lưu error message vào DB nếu có conversation_id
        try:
            if conversation_id:
                self.repository.add_message(conversation_id, "assistant", error_response_content)
        except Exception as db_error:
            logger.error(f"💥 Lỗi khi lưu error message vào DB: {db_error}")

        # Lấy tóm tắt cũ nhất nếu có thể
        current_summary = None
        try:
            if conversation_id:
                current_summary = self.repository.get_latest_summary(conversation_id)
        except Exception as summary_error:
            logger.error(f"💥 Không thể lấy tóm tắt trong error handler: {summary_error}")

        return {
            "conversation_id": conversation_id,
            "user_message": {"role": "user", "content": message_content},
            "assistant_message": {"role": "assistant", "content": error_response_content},
            "current_summary": current_summary,
            "error": "Lỗi hệ thống - vui lòng thử lại",
            "error_details": str(error) if settings.DEBUG else None
        }

    def create_new_chat(self, user_id: int) -> Dict[str, Any]:
        """
        Tạo cuộc trò chuyện mới với welcome message.
        
        Args:
            user_id: ID người dùng
            
        Returns:
            Dict với thông tin conversation mới
        """
        try:
            conversation = self.repository.create_conversation(user_id)
            conversation_id = conversation.conversation_id
            
            # Tạo welcome message
            try:
                welcome_message = asyncio.run(self.gemini_service.generate_welcome_message())
            except Exception as e:
                logger.error(f"💥 Lỗi khi tạo welcome message với Gemini: {str(e)}")
                welcome_message = "Xin chào! Tôi là trợ lý tư vấn dinh dưỡng và sức khỏe. Tôi có thể giúp gì cho bạn hôm nay?"
            
            # Lưu welcome message
            self.repository.add_message(conversation_id, "assistant", welcome_message)
            
            logger.info(f"🆕 Tạo chat mới thành công: conversation_id={conversation_id}, user_id={user_id}")
            
            return {
                "conversation_id": conversation_id,
                "created_at": conversation.created_at.isoformat(),
                "welcome_message": welcome_message,
                "current_summary": None  # Chat mới chưa có tóm tắt
            }
            
        except Exception as e:
            logger.error(f"💥 Lỗi khi tạo chat mới cho user_id={user_id}: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail="Không thể tạo cuộc trò chuyện mới")

    async def get_chat_content(self, user_id: int, conversation_id: Optional[int] = None) -> Dict[str, Any]:
        """
        Lấy nội dung cuộc trò chuyện bao gồm tóm tắt hiện tại và sản phẩm có sẵn.
        
        Args:
            user_id: ID người dùng
            conversation_id: ID cuộc trò chuyện (optional - lấy latest nếu None)
            
        Returns:
            Dict với messages, summary, health_data và available_products
        """
        try:
            # Xác định conversation
            if conversation_id:
                # Kiểm tra quyền truy cập
                if not self.repository.is_user_owner_of_conversation(user_id, conversation_id):
                    raise HTTPException(status_code=403, detail="Không có quyền truy cập cuộc trò chuyện này")
                
                conversation = self.repository.get_conversation_by_id(conversation_id)
                if not conversation:
                    raise HTTPException(status_code=404, detail="Cuộc trò chuyện không tồn tại")
            else:
                # Lấy conversation gần nhất
                conversation = self.repository.get_latest_conversation(user_id)
                if not conversation:
                    return {
                        "conversation_id": None,
                        "messages": [],
                        "current_summary": None,
                        "health_data": None,
                        "available_products": []
                    }

            # Lấy dữ liệu conversation
            messages_from_db = self.repository.get_messages(conversation.conversation_id)
            current_summary_text = self.repository.get_latest_summary(conversation.conversation_id)
            health_data_db = self.repository.get_health_data(conversation.conversation_id)
            
            # Lấy sản phẩm có sẵn từ các menu đã được lưu trong conversation
            available_products = await self._get_available_products_for_conversation(conversation.conversation_id)
            
            logger.debug(f"📖 Lấy chat content: conversation_id={conversation.conversation_id}, messages={len(messages_from_db)}, summary={'Có' if current_summary_text else 'Không'}, products={len(available_products)}")
            
            result = {
                "conversation_id": conversation.conversation_id,
                "messages": messages_from_db,
                "current_summary": current_summary_text,
                "health_data": health_data_db if health_data_db else None,
                "available_products": available_products
            }
            
            return result
            
        except HTTPException:
            # Re-raise HTTPException để FastAPI xử lý
            raise
        except Exception as e:
            logger.error(f"💥 Lỗi khi lấy chat content cho user_id={user_id}, conversation_id={conversation_id}: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail="Không thể lấy nội dung cuộc trò chuyện")

    async def _get_available_products_for_conversation(self, conversation_id: int) -> List[Dict[str, Any]]:
        """
        Lấy danh sách sản phẩm có sẵn từ các menu đã được tạo trong conversation.
        
        Args:
            conversation_id: ID cuộc trò chuyện
            
        Returns:
            List các sản phẩm có sẵn
        """
        try:
            # Lấy tất cả menu đã được tạo trong conversation này
            menu_data_list = self.repository.get_menu_data_by_conversation(conversation_id)
            
            if not menu_data_list:
                logger.debug(f"📭 Không tìm thấy menu nào cho conversation_id={conversation_id}")
                return []
            
            all_available_products = []
            processed_product_ids = set()
            
            for menu_data in menu_data_list:
                try:
                    menu_id = menu_data.get('menu_id')
                    if not menu_id:
                        continue
                    
                    # Lấy thông tin recipe từ repository
                    recipe_data = self.repository.get_recipe_by_id(menu_id)
                    if not recipe_data:
                        logger.warning(f"Không tìm thấy recipe với menu_id={menu_id}")
                        continue
                    
                    # Lấy danh sách ingredients có product_id
                    ingredients = recipe_data.get('ingredients', [])
                    if not ingredients:
                        continue
                    
                    # Lấy sản phẩm có sẵn cho menu này
                    available_products = await self.product_service.get_available_products_from_menu_items(ingredients)
                    
                    if available_products:
                        # Lọc ra sản phẩm chưa được xử lý (tránh trùng lặp)
                        for product in available_products:
                            product_id = product.get('id')
                            if product_id and product_id not in processed_product_ids:
                                all_available_products.append(product)
                                processed_product_ids.add(product_id)
                                
                        logger.debug(f"✅ Menu {menu_id}: Tìm thấy {len(available_products)} sản phẩm có sẵn")
                    
                except Exception as e:
                    logger.error(f"💥 Lỗi khi xử lý menu trong conversation: {str(e)}")
                    continue
            
            logger.info(f"🛒 Tổng cộng: {len(all_available_products)} sản phẩm có sẵn cho conversation_id={conversation_id}")
            return all_available_products
            
        except Exception as e:
            logger.error(f"💥 Lỗi khi lấy sản phẩm có sẵn cho conversation_id={conversation_id}: {str(e)}", exc_info=True)
            return []

    # === BACKGROUND DB OPERATIONS METHODS ===
    
    async def process_message_with_background(self, user_id: int, message_content: str, 
                                           conversation_id: Optional[int] = None) -> Tuple[Dict[str, Any], List[str]]:
        """
        Version của process_message với background DB operations.
        
        Returns:
            Tuple[response_data, background_task_ids]
        """
        background_task_ids = []
        
        # Bước 1: Chuẩn bị conversation
        if not conversation_id:
            conversation = self.repository.get_latest_conversation(user_id)
            if not conversation:
                # Tạo conversation mới và trả về với welcome message
                return await self._handle_new_conversation_with_background(user_id, message_content)
            conversation_id = conversation.conversation_id
        else:
            conversation = self.repository.get_conversation_by_id(conversation_id)
            if not conversation or conversation.user_id != user_id:
                raise ValueError("Cuộc trò chuyện không hợp lệ hoặc không có quyền truy cập.")

        # Bước 2: Lấy lịch sử chat TRƯỚC khi tin nhắn hiện tại được thêm
        chat_history_before_current_message = self.repository.get_messages(conversation_id)
        logger.info(f"🔄 Bắt đầu xử lý tin nhắn với background DB cho conversation_id={conversation_id}, user_id={user_id}")

        try:
            # Bước 3: Gọi LangGraph để xử lý logic chính
            langgraph_result = await run_chat_flow(
                user_message=message_content,
                user_id=user_id,
                conversation_id=conversation_id,
                messages=chat_history_before_current_message,
                repository=self.repository,
                llm_service=self.llm_service
            )
            
            # Bước 4: Chuẩn bị background tasks cho DB operations
            background_task_ids.extend(
                await self._prepare_background_db_tasks(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    message_content=message_content,
                    langgraph_result=langgraph_result
                )
            )
            
            # Bước 5: Chuẩn bị response (không chờ DB commit)
            current_summary = self.repository.get_latest_summary(conversation_id)
            response_payload = self._build_response_payload(
                conversation_id=conversation_id,
                message_content=message_content,
                langgraph_result=langgraph_result,
                current_summary=current_summary
            )
            
            logger.info(f"✅ Xử lý thành công tin nhắn với {len(background_task_ids)} background tasks cho conversation_id={conversation_id}")
            return response_payload, background_task_ids
            
        except Exception as e:
            logger.error(f"💥 Lỗi nghiêm trọng khi xử lý tin nhắn với background (ChatService): {str(e)}", exc_info=True)
            error_response = await self._handle_error_response(conversation_id, message_content, e)
            return error_response, background_task_ids

    async def _handle_new_conversation_with_background(self, user_id: int, message_content: str) -> Tuple[Dict[str, Any], List[str]]:
        """
        Xử lý tin nhắn đầu tiên trong conversation mới với background operations.
        """
        try:
            conversation = self.repository.create_conversation(user_id)
            conversation_id = conversation.conversation_id
            
            # Tạo welcome message
            welcome_message_content = await self.gemini_service.generate_welcome_message()
            
            # Tạo background tasks cho messages
            task_ids = []
            
            # Task cho welcome message
            welcome_task_id = background_db_service.add_message_task(
                conversation_id=conversation_id,
                role="assistant",
                content=welcome_message_content,
                repository_instance=self.repository
            )
            task_ids.append(welcome_task_id)
            
            # Task cho user message
            user_task_id = background_db_service.add_message_task(
                conversation_id=conversation_id,
                role="user", 
                content=message_content,
                repository_instance=self.repository
            )
            task_ids.append(user_task_id)
            
            logger.info(f"🆕 Tạo conversation mới với background tasks: {conversation_id} cho user_id={user_id}")
            
            response_data = {
                "conversation_id": conversation_id,
                "user_message": {"role": "user", "content": message_content},
                "assistant_message": {"role": "assistant", "content": welcome_message_content},
                "current_summary": None,
                "is_new_conversation": True
            }
            
            return response_data, task_ids
            
        except Exception as e:
            logger.error(f"💥 Lỗi khi tạo conversation mới với background: {str(e)}", exc_info=True)
            raise

    async def _prepare_background_db_tasks(self, conversation_id: int, user_id: int,
                                         message_content: str, langgraph_result: Dict[str, Any]) -> List[str]:
        """
        Chuẩn bị các background DB tasks dựa trên kết quả từ LangGraph.
        
        Returns:
            List task IDs đã được tạo
        """
        task_ids = []
        
        try:
            # Task 1: Save user message nếu chưa có
            user_message_id = langgraph_result.get("user_message_id_db")
            if not user_message_id:
                user_task_id = background_db_service.add_message_task(
                    conversation_id=conversation_id,
                    role="user",
                    content=message_content,
                    repository_instance=self.repository
                )
                task_ids.append(user_task_id)
                logger.debug(f"📝 Tạo background task cho user message: {user_task_id}")
            
            # Task 2: Save assistant message nếu có final_response
            final_response = langgraph_result.get("final_response")
            assistant_message_id = langgraph_result.get("assistant_message_id_db")
            
            if final_response and not assistant_message_id:
                assistant_task_id = background_db_service.add_message_task(
                    conversation_id=conversation_id,
                    role="assistant",
                    content=final_response,
                    repository_instance=self.repository
                )
                task_ids.append(assistant_task_id)
                logger.debug(f"📝 Tạo background task cho assistant message: {assistant_task_id}")
            
            # Task 3: Save conversation summary nếu đủ điều kiện
            should_create_summary = (
                langgraph_result.get("is_valid_scope", False) and 
                not langgraph_result.get("need_more_info", True) and
                final_response and assistant_message_id
            )
            
            if should_create_summary:
                # Tạo summary với Gemini
                previous_summary = self.repository.get_latest_summary(conversation_id)
                summary_context_messages = self.repository.get_messages_for_summary_context(conversation_id, limit=3)
                
                new_summary = await self.gemini_service.create_incremental_summary(
                    previous_summary=previous_summary,
                    new_user_message=message_content,
                    new_assistant_message=final_response,
                    full_chat_history_for_context=summary_context_messages
                )
                
                if new_summary:
                    summary_task_id = background_db_service.save_conversation_summary_task(
                        conversation_id=conversation_id,
                        assistant_message_id=assistant_message_id,
                        summary_text=new_summary,
                        repository_instance=self.repository
                    )
                    task_ids.append(summary_task_id)
                    logger.debug(f"📝 Tạo background task cho conversation summary: {summary_task_id}")
            
            # Task 4: Save health data nếu có
            extracted_health_data = langgraph_result.get("extracted_health_data")
            if extracted_health_data:
                health_task_id = background_db_service.save_health_data_task(
                    conversation_id=conversation_id,
                    user_id=user_id,
                    health_data=extracted_health_data,
                    repository_instance=self.repository
                )
                task_ids.append(health_task_id)
                logger.debug(f"📝 Tạo background task cho health data: {health_task_id}")
            
            logger.info(f"🔄 Đã chuẩn bị {len(task_ids)} background DB tasks cho conversation_id={conversation_id}")
            return task_ids
            
        except Exception as e:
            logger.error(f"💥 Lỗi khi chuẩn bị background DB tasks: {str(e)}", exc_info=True)
            return task_ids

    def execute_background_tasks(self, task_ids: List[str]) -> None:
        """
        Execute các background DB tasks.
        Được gọi từ FastAPI BackgroundTasks.
        """
        if not task_ids:
            return
            
        logger.info(f"🚀 Bắt đầu execute {len(task_ids)} background DB tasks")
        background_db_service.execute_multiple_tasks(task_ids)
        
    def get_background_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Lấy trạng thái của một background task.
        Dùng cho monitoring/debugging.
        """
        return background_db_service.get_task_status(task_id) 