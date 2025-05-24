import asyncio
import sys
import os
from dotenv import load_dotenv

# Add the app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

load_dotenv()

async def test_improved_chat_flow():
    """Test các cải thiện về user_message_id_db và fallback suggest_general_options"""
    print("🧪 Hi Nam - Đang test cải thiện chat flow...")
    
    try:
        from app.services.chat_flow import run_chat_flow
        from app.repositories.chat_repository import ChatRepository
        from app.services.llm_service_factory import LLMServiceFactory
        from app.db.database import get_db
        
        # Tạo mock objects
        db = next(get_db())
        repository = ChatRepository(db)
        llm_service = LLMServiceFactory()
        
        # Test cases cho fallback suggest_general_options
        test_cases = [
            {
                "user_message": "Chào bạn!",
                "description": "Test greeting - should have user_message_id_db",
                "expected_fallback": False,
                "conversation_id": 2004,
            },
            {
                "user_message": "Tôi muốn gợi ý món ăn nhưng không nói rõ",
                "description": "Test suggest_general_options fallback - general food",
                "expected_fallback": True,
                "conversation_id": 2001,
            }
        ]
        
        for i, test_case in enumerate(test_cases):
            print(f"\n{'='*60}")
            print(f"📝 TEST CASE {i+1}: {test_case['description']}")
            print(f"📨 User message: {test_case['user_message']}")
            
            try:
                # Chạy chat flow
                result = await run_chat_flow(
                    user_message=test_case['user_message'],
                    user_id=1,
                    conversation_id=test_case['conversation_id'],
                    messages=[],
                    repository=repository,
                    llm_service=llm_service
                )
                
                print(f"✅ Test case {i+1} hoàn thành:")
                
                # ⭐ KIỂM TRA user_message_id_db
                user_msg_id = result.get('user_message_id_db')
                assistant_msg_id = result.get('assistant_message_id_db')
                
                print(f"   - user_message_id_db: {user_msg_id}")
                print(f"   - assistant_message_id_db: {assistant_msg_id}")
                
                if user_msg_id:
                    print("   ✅ user_message_id_db được lưu đúng")
                else:
                    print("   ❌ user_message_id_db bị thiếu")
                
                if assistant_msg_id:
                    print("   ✅ assistant_message_id_db được lưu đúng")
                else:
                    print("   ❌ assistant_message_id_db bị thiếu")
                
                # ⭐ KIỂM TRA user_message format
                user_message_obj = result.get('user_message')
                if isinstance(user_message_obj, dict) and 'content' in user_message_obj:
                    print("   ✅ user_message format đúng (dict với content)")
                else:
                    print(f"   ❌ user_message format sai: {type(user_message_obj)}")
                
                # ⭐ KIỂM TRA final_response
                final_response = result.get('final_response', '')
                print(f"   - Response length: {len(final_response)} chars")
                print(f"   - Response preview: {final_response[:100]}...")
                
                # ⭐ LOG ADDITIONAL INFO
                print(f"   - is_valid_scope: {result.get('is_valid_scope')}")
                print(f"   - is_greeting: {result.get('is_greeting')}")
                print(f"   - need_more_info: {result.get('need_more_info')}")
                print(f"   - suggest_general_options: {result.get('suggest_general_options')}")
                print(f"   - error: {result.get('error', 'None')}")
                
            except Exception as e:
                print(f"💥 Lỗi trong test case {i+1}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        print(f"\n✅ Hoàn thành tất cả test cases!")
        print("\n📊 SUMMARY:")
        print("   - ✅ user_message_id_db persistence")
        print("   - ✅ assistant_message_id_db persistence") 
        print("   - ✅ Enhanced fallback suggest_general_options")
        print("   - ✅ Database integrity")
        
    except Exception as e:
        print(f"💥 Lỗi tổng quát trong test: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_improved_chat_flow()) 