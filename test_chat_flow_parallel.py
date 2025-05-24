import asyncio
import sys
import os
from dotenv import load_dotenv

# Add the app directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'app'))

load_dotenv()

async def test_parallel_logic_functions():
    """Test parallel logic functions trực tiếp mà không cần database"""
    print("🧪 Hi Nam - Đang test parallel logic functions...")
    
    try:
        from app.services.chat_flow import parallel_tool_runner_node, recipe_search_logic, beverage_search_logic
        
        # Tạo mock state cho test
        test_state = {
            'conversation_id': 1,
            'user_id': 1,
            'user_message': 'Tôi muốn gợi ý món ăn và đồ uống tốt cho tim mạch',
            'is_valid_scope': True,
            'is_food_related': True,
            'requests_food': True,
            'requests_beverage': True,
            'user_rejected_info': False,
            'need_more_info': False,
            'suggest_general_options': False,
            'collected_info': {'health_condition': 'bệnh tim'},
            'messages': [
                {"role": "user", "content": "Tôi muốn gợi ý món ăn và đồ uống tốt cho tim mạch"}
            ]
        }
        
        print(f"\n{'='*60}")
        print("📝 TEST PARALLEL PROCESSING")
        print(f"📨 User message: {test_state['user_message']}")
        print(f"🔀 Requests food: {test_state['requests_food']}")
        print(f"🔀 Requests beverage: {test_state['requests_beverage']}")
        
        # Test 1: Chạy tuần tự để có baseline
        print(f"\n🚀 TEST 1: Sequential processing (baseline)")
        start_time = asyncio.get_event_loop().time()
        
        recipe_results = await recipe_search_logic(test_state.copy())
        beverage_results = await beverage_search_logic(test_state.copy())
        
        sequential_time = asyncio.get_event_loop().time() - start_time
        
        print(f"✅ Sequential hoàn thành:")
        print(f"   - Recipe results: {len(recipe_results) if recipe_results else 0} items")
        print(f"   - Beverage results: {len(beverage_results) if beverage_results else 0} items")
        print(f"   - Time taken: {sequential_time:.2f}s")
        
        # Test 2: Chạy song song với parallel_tool_runner_node
        print(f"\n⚡ TEST 2: Parallel processing")
        start_time = asyncio.get_event_loop().time()
        
        result_state = await parallel_tool_runner_node(test_state.copy())
        
        parallel_time = asyncio.get_event_loop().time() - start_time
        
        print(f"✅ Parallel hoàn thành:")
        print(f"   - Recipe results: {len(result_state.get('recipe_results', [])) if result_state.get('recipe_results') else 0} items")
        print(f"   - Beverage results: {len(result_state.get('beverage_results', [])) if result_state.get('beverage_results') else 0} items")
        print(f"   - Time taken: {parallel_time:.2f}s")
        
        # Tính toán hiệu suất
        if sequential_time > 0 and parallel_time > 0:
            speedup = sequential_time / parallel_time
            time_saved = sequential_time - parallel_time
            percentage_saved = (time_saved / sequential_time) * 100
            
            print(f"\n📊 PERFORMANCE COMPARISON:")
            print(f"   - Sequential: {sequential_time:.2f}s")
            print(f"   - Parallel: {parallel_time:.2f}s")
            print(f"   - Speedup: {speedup:.2f}x")
            print(f"   - Time saved: {time_saved:.2f}s ({percentage_saved:.1f}%)")
            
            if speedup > 1.0:
                print("🎉 Parallel processing hoạt động hiệu quả!")
            else:
                print("⚠️ Parallel processing chậm hơn tuần tự (có thể do overhead)")
        
        # Test 3: Test edge cases
        print(f"\n🧪 TEST 3: Edge cases")
        
        # Test với chỉ requests_food = True
        food_only_state = test_state.copy()
        food_only_state['requests_beverage'] = False
        result = await parallel_tool_runner_node(food_only_state)
        
        print(f"   - Food only: recipes={len(result.get('recipe_results', [])) if result.get('recipe_results') else 0}, beverages={len(result.get('beverage_results', [])) if result.get('beverage_results') else 0}")
        
        # Test với chỉ requests_beverage = True  
        beverage_only_state = test_state.copy()
        beverage_only_state['requests_food'] = False
        result = await parallel_tool_runner_node(beverage_only_state)
        
        print(f"   - Beverage only: recipes={len(result.get('recipe_results', [])) if result.get('recipe_results') else 0}, beverages={len(result.get('beverage_results', [])) if result.get('beverage_results') else 0}")
        
        # Test với cả hai = False
        neither_state = test_state.copy()
        neither_state['requests_food'] = False
        neither_state['requests_beverage'] = False
        result = await parallel_tool_runner_node(neither_state)
        
        print(f"   - Neither: recipes={len(result.get('recipe_results', [])) if result.get('recipe_results') else 0}, beverages={len(result.get('beverage_results', [])) if result.get('beverage_results') else 0}")
        
        print(f"\n{'='*60}")
        print("✅ Hoàn thành tất cả tests cho parallel processing!")
        
    except Exception as e:
        print(f"💥 Lỗi trong test: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_parallel_logic_functions()) 