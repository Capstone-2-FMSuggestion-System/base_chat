import asyncio
import sys
import os
import time
from dotenv import load_dotenv

# Add the app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

load_dotenv()

async def test_parallel_processing():
    """Test parallel processing của recipe và beverage tools"""
    print("🧪 Hi Nam - Đang test parallel processing cho recipe và beverage tools...")
    
    try:
        # Import các module cần thiết
        from app.services.chat_flow import recipe_search_logic, beverage_search_logic, parallel_tool_runner_node
        
        # Tạo mock state với cả requests_food và requests_beverage
        test_state = {
            'user_message': 'Gợi ý món ăn và đồ uống tốt cho sức khỏe tim mạch',
            'collected_info': {
                'health_conditions': ['tim mạch'],
                'preferences': ['healthy food']
            },
            'suggest_general_options': False,
            'requests_food': True,
            'requests_beverage': True,
            'conversation_id': 1,
            'user_id': 1
        }
        
        print("📊 Cấu hình test:")
        print(f"   - User message: {test_state['user_message']}")
        print(f"   - Requests food: {test_state['requests_food']}")
        print(f"   - Requests beverage: {test_state['requests_beverage']}")
        print(f"   - Health info: {test_state['collected_info']}")
        
        # Test 1: Chạy tuần tự (baseline)
        print("\n🔄 TEST 1: Chạy tuần tự (baseline)")
        start_sequential = time.time()
        
        recipe_results = await recipe_search_logic(test_state)
        beverage_results = await beverage_search_logic(test_state)
        
        end_sequential = time.time()
        sequential_time = end_sequential - start_sequential
        
        print(f"✅ Tuần tự hoàn thành:")
        print(f"   - Recipes: {len(recipe_results)} kết quả")
        print(f"   - Beverages: {len(beverage_results)} kết quả") 
        print(f"   - Thời gian: {sequential_time:.2f}s")
        
        # Test 2: Chạy song song
        print("\n⚡ TEST 2: Chạy song song với parallel_tool_runner_node")
        start_parallel = time.time()
        
        parallel_state = test_state.copy()
        result_state = await parallel_tool_runner_node(parallel_state)
        
        end_parallel = time.time()
        parallel_time = end_parallel - start_parallel
        
        print(f"✅ Song song hoàn thành:")
        print(f"   - Recipes: {len(result_state.get('recipe_results', []))} kết quả")
        print(f"   - Beverages: {len(result_state.get('beverage_results', []))} kết quả")
        print(f"   - Thời gian: {parallel_time:.2f}s")
        
        # So sánh hiệu suất
        if sequential_time > 0:
            speedup = sequential_time / parallel_time
            time_saved = sequential_time - parallel_time
            print(f"\n📈 HIỆU SUẤT:")
            print(f"   - Tăng tốc: {speedup:.2f}x")
            print(f"   - Tiết kiệm: {time_saved:.2f}s ({(time_saved/sequential_time)*100:.1f}%)")
            
            if speedup > 1.2:
                print("🎉 Parallel processing hiệu quả!")
            elif speedup > 1.0:
                print("👍 Parallel processing có cải thiện nhỏ")
            else:
                print("⚠️ Parallel processing chưa tối ưu")
        
        # Kiểm tra tính đúng đắn của kết quả
        print(f"\n🔍 KIỂM TRA KẾT QUẢ:")
        recipes_match = len(recipe_results) == len(result_state.get('recipe_results', []))
        beverages_match = len(beverage_results) == len(result_state.get('beverage_results', []))
        
        print(f"   - Recipe results match: {recipes_match}")
        print(f"   - Beverage results match: {beverages_match}")
        
        if recipes_match and beverages_match:
            print("✅ Parallel processing cho kết quả đúng!")
        else:
            print("❌ Parallel processing có vấn đề về kết quả")
            
        # Hiển thị sample kết quả
        if result_state.get('recipe_results'):
            print(f"\n🍳 SAMPLE RECIPES ({len(result_state['recipe_results'])} total):")
            for i, recipe in enumerate(result_state['recipe_results'][:3]):
                print(f"   {i+1}. {recipe.get('title', 'N/A')}")
                
        if result_state.get('beverage_results'):
            print(f"\n🥤 SAMPLE BEVERAGES ({len(result_state['beverage_results'])} total):")
            for i, beverage in enumerate(result_state['beverage_results'][:3]):
                print(f"   {i+1}. {beverage.get('product_name', 'N/A')}")
        
    except Exception as e:
        print(f"💥 Lỗi trong test: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_parallel_processing()) 