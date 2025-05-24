import asyncio
import sys
import os
from dotenv import load_dotenv

# Add the app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

load_dotenv()

async def test_persist_and_fallback():
    """Test persist user interaction và fallback suggest_general_options"""
    print("🧪 Hi Nam - Đang test persist user interaction và fallback suggest_general_options...")
    
    try:
        from app.services.chat_flow import run_chat_flow
        from app.repositories.chat_repository import ChatRepository
        from app.services.llm_service_factory import LLMServiceFactory
        from app.db.database import get_db
        
        # Tạo mock objects
        db = next(get_db())
        repository = ChatRepository(db)
        llm_service = LLMServiceFactory.create_llm_service()
        
        # Test cases cho suggest_general_options
        test_cases = [
            {
                "user_message": "Tôi cần gợi ý dinh dưỡng chung",
                "description": "Yêu cầu gợi ý chung - nên trigger suggest_general_options",
                "expected_fallback": True
            },
            {
                "user_message": "Gợi ý đồ uống tốt cho sức khỏe",
                "description": "Yêu cầu đồ uống chung - test template đồ uống",
                "expected_fallback": True
            },
            {
                "user_message": "Tôi có bệnh tim mạch, nên ăn gì?",
                "description": "Có vấn đề sức khỏe - test template sức khỏe",
                "expected_fallback": False
            },
            {
                "user_message": "Tôi muốn giảm cân, gợi ý thực đơn",
                "description": "Giảm cân - test template giảm cân", 
                "expected_fallback": True
            }
        ]
        
        conversation_id = 888
        user_id = 1
        
        for i, test_case in enumerate(test_cases):
            print(f"\n{'='*70}")
            print(f"📝 TEST CASE {i+1}: {test_case['description']}")
            print(f"📨 User message: {test_case['user_message']}")
            print(f"🔀 Expected fallback: {test_case['expected_fallback']}")
            
            try:
                # Chạy chat flow
                result = await run_chat_flow(
                    user_message=test_case['user_message'],
                    user_id=user_id,
                    conversation_id=conversation_id + i,  # Unique conversation
                    messages=[],
                    repository=repository,
                    llm_service=llm_service
                )
                
                print(f"✅ Test case {i+1} hoàn thành:")
                
                # Kiểm tra user_message_id_db
                user_msg_id = result.get('user_message_id_db')
                print(f"   - user_message_id_db: {user_msg_id}")
                if user_msg_id and user_msg_id > 0:
                    print("   ✅ user_message_id_db hợp lệ")
                else:
                    print("   ❌ user_message_id_db không hợp lệ")
                
                # Kiểm tra assistant_message_id_db  
                assistant_msg_id = result.get('assistant_message_id_db')
                print(f"   - assistant_message_id_db: {assistant_msg_id}")
                if assistant_msg_id and assistant_msg_id > 0:
                    print("   ✅ assistant_message_id_db hợp lệ")
                else:
                    print("   ❌ assistant_message_id_db không hợp lệ")
                
                # Kiểm tra final_response
                final_response = result.get('final_response', '')
                print(f"   - Response length: {len(final_response)} chars")
                print(f"   - Response preview: {final_response[:100]}...")
                
                # Kiểm tra suggest_general_options
                suggest_general = result.get('suggest_general_options', False)
                print(f"   - suggest_general_options: {suggest_general}")
                
                # Kiểm tra chất lượng response
                if final_response:
                    if any(emoji in final_response for emoji in ['🥗', '🍲', '🥣', '🍜', '🥤', '🍵', '🥛']):
                        print("   ✅ Response có emoji phù hợp")
                    else:
                        print("   ⚠️ Response thiếu emoji")
                        
                    if len(final_response) > 200:
                        print("   ✅ Response đủ chi tiết")
                    else:
                        print("   ⚠️ Response có thể quá ngắn")
                        
                    if '?' in final_response[-50:]:  # Kiểm tra có câu hỏi cuối không
                        print("   ✅ Response có câu hỏi mời tiếp tục")
                    else:
                        print("   ⚠️ Response thiếu câu hỏi mời tiếp tục")
                        
                else:
                    print("   ❌ Không có final_response")
                
                # Kiểm tra error
                error = result.get('error')
                if error:
                    print(f"   ⚠️ Có error: {error}")
                else:
                    print("   ✅ Không có error")
                    
            except Exception as e:
                print(f"💥 Lỗi trong test case {i+1}: {str(e)}")
                import traceback
                traceback.print_exc()
                
        print(f"\n{'='*70}")
        print("✅ Hoàn thành tất cả test cases!")
        
    except Exception as e:
        print(f"💥 Lỗi tổng quát trong test: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_persist_and_fallback()) 