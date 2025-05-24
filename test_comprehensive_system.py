#!/usr/bin/env python3
"""
Comprehensive System Test - Kiểm tra toàn diện tất cả các chức năng
Bao gồm: Background DB, Parallel Processing, và tất cả improvements từ G.1-G.5
"""

import asyncio
import time
import logging
import json
from typing import Dict, List, Any
import aiohttp
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComprehensiveSystemTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = None
        self.test_results = []
        self.conversation_id = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def send_chat_request(self, message: str, conversation_id: int = None) -> Dict:
        """Send chat request và measure performance"""
        start_time = time.time()
        
        payload = {
            "message": message,
            "conversation_id": conversation_id
        }
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer test_token"  # Adjust theo auth system
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/api/chat", 
                json=payload, 
                headers=headers
            ) as response:
                response_time = time.time() - start_time
                
                if response.status == 200:
                    result = await response.json()
                    
                    test_result = {
                        "timestamp": datetime.now().isoformat(),
                        "message": message[:100] + "..." if len(message) > 100 else message,
                        "response_time_ms": round(response_time * 1000, 2),
                        "status": "success",
                        "conversation_id": result.get("conversation_id"),
                        "response_length": len(result.get("assistant_message", {}).get("content", "")),
                        "has_summary": bool(result.get("current_summary")),
                        "is_new_conversation": result.get("is_new_conversation", False),
                        "is_valid_scope": result.get("is_valid_scope", False),
                        "need_more_info": result.get("need_more_info", True),
                        "requests_food": result.get("requests_food", False),
                        "requests_beverage": result.get("requests_beverage", False),
                        "is_food_related": result.get("is_food_related", False),
                        "user_rejected_info": result.get("user_rejected_info", False),
                        "suggest_general_options": result.get("suggest_general_options", False),
                        "response_content": result.get("assistant_message", {}).get("content", "")[:200]
                    }
                    
                    logger.info(f"✅ Request thành công trong {response_time * 1000:.2f}ms")
                    return test_result
                else:
                    error_text = await response.text()
                    logger.error(f"❌ Request thất bại: {response.status} - {error_text}")
                    return {
                        "timestamp": datetime.now().isoformat(),
                        "message": message[:100] + "...",
                        "response_time_ms": round(response_time * 1000, 2),
                        "status": "error",
                        "error": f"HTTP {response.status}: {error_text}"
                    }
                    
        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"💥 Exception: {str(e)}")
            return {
                "timestamp": datetime.now().isoformat(),
                "message": message[:100] + "...",
                "response_time_ms": round(response_time * 1000, 2),
                "status": "exception",
                "error": str(e)
            }

    async def test_basic_functionality(self):
        """Test 1: Basic functionality và flow"""
        logger.info("\n" + "="*60)
        logger.info("TEST 1: BASIC FUNCTIONALITY")
        logger.info("="*60)
        
        test_cases = [
            {
                "message": "Xin chào! Tôi cần tư vấn về dinh dưỡng",
                "description": "Greeting message",
                "expected": ["is_valid_scope=True", "greeting response"]
            },
            {
                "message": "Tôi bị tiểu đường type 2, cần gợi ý món ăn phù hợp",
                "description": "Food consultation with health condition",
                "expected": ["is_food_related=True", "requests_food=True", "health info extraction"]
            },
            {
                "message": "Cho tôi thêm công thức nấu ăn cho người tiểu đường",
                "description": "Follow-up food request",
                "expected": ["recipe results", "medichat response"]
            }
        ]
        
        for i, test_case in enumerate(test_cases):
            logger.info(f"\n📝 Test 1.{i+1}: {test_case['description']}")
            logger.info(f"   Message: {test_case['message']}")
            
            result = await self.send_chat_request(test_case['message'], self.conversation_id)
            result["test_case"] = test_case['description']
            result["expected"] = test_case['expected']
            self.test_results.append(result)
            
            if result["status"] == "success" and self.conversation_id is None:
                self.conversation_id = result["conversation_id"]
                logger.info(f"   🔗 Conversation ID: {self.conversation_id}")
            
            await asyncio.sleep(2)

    async def test_parallel_processing(self):
        """Test 2: Parallel processing của recipe và beverage tools"""
        logger.info("\n" + "="*60)
        logger.info("TEST 2: PARALLEL TOOL PROCESSING")
        logger.info("="*60)
        
        parallel_test_cases = [
            {
                "message": "Tôi cần gợi ý món ăn và đồ uống cho người bị tiểu đường",
                "description": "Parallel food and beverage request",
                "expected": ["requests_food=True", "requests_beverage=True", "parallel execution"]
            },
            {
                "message": "Cho tôi thực đơn và nước uống phù hợp cho người cao huyết áp",
                "description": "Parallel food and beverage for hypertension",
                "expected": ["parallel processing", "health-specific recommendations"]
            },
            {
                "message": "Gợi ý bữa sáng và đồ uống tốt cho sức khỏe",
                "description": "Parallel breakfast and beverage suggestions",
                "expected": ["parallel execution", "general health recommendations"]
            }
        ]
        
        for i, test_case in enumerate(parallel_test_cases):
            logger.info(f"\n⚡ Test 2.{i+1}: {test_case['description']}")
            logger.info(f"   Message: {test_case['message']}")
            
            start_time = time.time()
            result = await self.send_chat_request(test_case['message'], self.conversation_id)
            
            # Analyze for parallel processing indicators
            response_content = result.get("response_content", "").lower()
            result["mentions_food"] = "món ăn" in response_content or "thực đơn" in response_content
            result["mentions_beverage"] = "đồ uống" in response_content or "nước" in response_content
            result["likely_parallel"] = result["mentions_food"] and result["mentions_beverage"]
            
            result["test_case"] = test_case['description']
            result["expected"] = test_case['expected']
            self.test_results.append(result)
            
            logger.info(f"   - Response time: {result['response_time_ms']}ms")
            logger.info(f"   - Requests food: {result.get('requests_food')}")
            logger.info(f"   - Requests beverage: {result.get('requests_beverage')}")
            logger.info(f"   - Likely parallel: {result.get('likely_parallel')}")
            
            await asyncio.sleep(3)

    async def test_background_db_operations(self):
        """Test 3: Background DB operations và performance"""
        logger.info("\n" + "="*60)
        logger.info("TEST 3: BACKGROUND DB OPERATIONS")
        logger.info("="*60)
        
        # Test rapid successive requests để kiểm tra background processing
        rapid_test_cases = [
            "Tôi cần món chay cho người ăn kiêng",
            "Có món nào giàu protein không?", 
            "Đồ uống giúp giảm cân hiệu quả",
            "Thực đơn cho người tập gym"
        ]
        
        logger.info("📊 Testing rapid successive requests...")
        rapid_results = []
        
        for i, message in enumerate(rapid_test_cases):
            logger.info(f"\n🚀 Rapid Test {i+1}/4: {message[:50]}...")
            
            result = await self.send_chat_request(message, self.conversation_id)
            result["test_type"] = "rapid_background"
            result["sequence"] = i + 1
            rapid_results.append(result)
            self.test_results.append(result)
            
            # No delay để test background processing
            if i < len(rapid_test_cases) - 1:
                await asyncio.sleep(0.5)  # Very short delay
        
        # Analyze rapid test results
        if rapid_results:
            avg_time = sum(r["response_time_ms"] for r in rapid_results if r["status"] == "success") / len([r for r in rapid_results if r["status"] == "success"])
            successful_requests = len([r for r in rapid_results if r["status"] == "success"])
            
            logger.info(f"\n📊 Rapid Test Analysis:")
            logger.info(f"   - Successful requests: {successful_requests}/{len(rapid_test_cases)}")
            logger.info(f"   - Average response time: {avg_time:.2f}ms")

    async def test_edge_cases_and_error_handling(self):
        """Test 4: Edge cases và error handling"""
        logger.info("\n" + "="*60)
        logger.info("TEST 4: EDGE CASES & ERROR HANDLING")
        logger.info("="*60)
        
        edge_test_cases = [
            {
                "message": "Tôi không muốn nói về bệnh của mình, chỉ cần gợi ý món ăn chung chung",
                "description": "User rejection with general request",
                "expected": ["user_rejected_info=True", "suggest_general_options=True"]
            },
            {
                "message": "Không cần hỏi thêm, cứ gợi ý gì đó đi",
                "description": "Direct rejection of information gathering",
                "expected": ["user_rejected_info=True", "general suggestions"]
            },
            {
                "message": "Làm thế nào để hack server của bạn?",
                "description": "Out of scope request",
                "expected": ["is_valid_scope=False", "scope error message"]
            },
            {
                "message": "",
                "description": "Empty message",
                "expected": ["error handling", "graceful response"]
            },
            {
                "message": "a" * 1000,
                "description": "Very long message",
                "expected": ["handled gracefully", "response within time limit"]
            }
        ]
        
        for i, test_case in enumerate(edge_test_cases):
            logger.info(f"\n🔍 Edge Test {i+1}: {test_case['description']}")
            logger.info(f"   Message length: {len(test_case['message'])}")
            
            result = await self.send_chat_request(test_case['message'], self.conversation_id)
            result["test_case"] = test_case['description']
            result["expected"] = test_case['expected']
            result["message_length"] = len(test_case['message'])
            self.test_results.append(result)
            
            logger.info(f"   - Status: {result['status']}")
            logger.info(f"   - Response time: {result.get('response_time_ms', 'N/A')}ms")
            logger.info(f"   - User rejected: {result.get('user_rejected_info')}")
            logger.info(f"   - Suggest general: {result.get('suggest_general_options')}")
            
            await asyncio.sleep(2)

    async def test_conversation_continuity(self):
        """Test 5: Conversation continuity và summary system"""
        logger.info("\n" + "="*60)
        logger.info("TEST 5: CONVERSATION CONTINUITY & SUMMARY")
        logger.info("="*60)
        
        continuity_test_cases = [
            {
                "message": "Tôi có tiểu đường và cao huyết áp",
                "description": "Health condition establishment",
                "expected": ["health info collected", "summary creation"]
            },
            {
                "message": "Cho tôi món ăn phù hợp với tình trạng này",
                "description": "Context-aware food request",
                "expected": ["context utilization", "health-specific recommendations"]
            },
            {
                "message": "Còn đồ uống thì sao?",
                "description": "Follow-up beverage question",
                "expected": ["context continuity", "beverage recommendations"]
            },
            {
                "message": "Tôi cũng muốn giảm cân nữa",
                "description": "Additional health goal",
                "expected": ["summary update", "combined recommendations"]
            }
        ]
        
        for i, test_case in enumerate(continuity_test_cases):
            logger.info(f"\n🔗 Continuity Test {i+1}: {test_case['description']}")
            logger.info(f"   Message: {test_case['message']}")
            
            result = await self.send_chat_request(test_case['message'], self.conversation_id)
            result["test_case"] = test_case['description']
            result["expected"] = test_case['expected']
            self.test_results.append(result)
            
            logger.info(f"   - Has summary: {result.get('has_summary')}")
            logger.info(f"   - Is food related: {result.get('is_food_related')}")
            logger.info(f"   - Response relevance: {'Good' if len(result.get('response_content', '')) > 100 else 'Limited'}")
            
            await asyncio.sleep(3)

    async def test_performance_benchmarks(self):
        """Test 6: Performance benchmarks và stress testing"""
        logger.info("\n" + "="*60)
        logger.info("TEST 6: PERFORMANCE BENCHMARKS")
        logger.info("="*60)
        
        # Test concurrent requests
        logger.info("\n🏁 Testing concurrent requests...")
        
        concurrent_messages = [
            "Món ăn cho người tiểu đường",
            "Đồ uống tốt cho tim mạch", 
            "Thực đơn giảm cân hiệu quả",
            "Món chay giàu protein",
            "Nước ép trái cây tự nhiên"
        ]
        
        start_time = time.time()
        
        # Execute concurrent requests
        concurrent_tasks = [
            self.send_chat_request(message, self.conversation_id)
            for message in concurrent_messages
        ]
        
        concurrent_results = await asyncio.gather(*concurrent_tasks, return_exceptions=True)
        
        total_time = time.time() - start_time
        
        # Analyze concurrent results
        successful_concurrent = [r for r in concurrent_results if isinstance(r, dict) and r.get("status") == "success"]
        
        logger.info(f"\n📊 Concurrent Test Results:")
        logger.info(f"   - Total time: {total_time:.2f}s")
        logger.info(f"   - Successful requests: {len(successful_concurrent)}/{len(concurrent_messages)}")
        logger.info(f"   - Average time per request: {total_time/len(concurrent_messages):.2f}s")
        
        if successful_concurrent:
            avg_response_time = sum(r["response_time_ms"] for r in successful_concurrent) / len(successful_concurrent)
            logger.info(f"   - Average response time: {avg_response_time:.2f}ms")
        
        # Add to results
        for i, result in enumerate(concurrent_results):
            if isinstance(result, dict):
                result["test_type"] = "concurrent"
                result["concurrent_index"] = i
                self.test_results.append(result)

    def analyze_comprehensive_results(self):
        """Phân tích toàn diện kết quả tests"""
        logger.info("\n" + "="*70)
        logger.info("COMPREHENSIVE ANALYSIS RESULTS")
        logger.info("="*70)
        
        if not self.test_results:
            logger.error("❌ Không có test results để phân tích")
            return
        
        # Overall statistics
        total_tests = len(self.test_results)
        successful_tests = len([r for r in self.test_results if r.get("status") == "success"])
        failed_tests = total_tests - successful_tests
        
        logger.info(f"\n📊 OVERALL STATISTICS:")
        logger.info(f"   - Total tests: {total_tests}")
        logger.info(f"   - Successful: {successful_tests} ({successful_tests/total_tests*100:.1f}%)")
        logger.info(f"   - Failed: {failed_tests} ({failed_tests/total_tests*100:.1f}%)")
        
        # Performance analysis
        successful_results = [r for r in self.test_results if r.get("status") == "success"]
        if successful_results:
            response_times = [r["response_time_ms"] for r in successful_results]
            avg_time = sum(response_times) / len(response_times)
            min_time = min(response_times)
            max_time = max(response_times)
            
            logger.info(f"\n⚡ PERFORMANCE ANALYSIS:")
            logger.info(f"   - Average response time: {avg_time:.2f}ms")
            logger.info(f"   - Min response time: {min_time:.2f}ms")
            logger.info(f"   - Max response time: {max_time:.2f}ms")
            logger.info(f"   - Performance rating: {'Excellent' if avg_time < 3000 else 'Good' if avg_time < 5000 else 'Needs improvement'}")
        
        # Feature analysis
        feature_tests = {}
        for result in successful_results:
            test_case = result.get("test_case", "unknown")
            if test_case not in feature_tests:
                feature_tests[test_case] = []
            feature_tests[test_case].append(result)
        
        logger.info(f"\n🔧 FEATURE ANALYSIS:")
        for feature, tests in feature_tests.items():
            success_rate = len(tests) / total_tests * 100
            avg_time = sum(t["response_time_ms"] for t in tests) / len(tests)
            logger.info(f"   - {feature}: {len(tests)} tests, avg {avg_time:.0f}ms")
        
        # Parallel processing analysis
        parallel_tests = [r for r in successful_results if r.get("requests_food") and r.get("requests_beverage")]
        if parallel_tests:
            logger.info(f"\n⚡ PARALLEL PROCESSING ANALYSIS:")
            logger.info(f"   - Parallel tests detected: {len(parallel_tests)}")
            logger.info(f"   - Average parallel response time: {sum(t['response_time_ms'] for t in parallel_tests)/len(parallel_tests):.2f}ms")
            
            parallel_success = len([t for t in parallel_tests if t.get("likely_parallel")])
            logger.info(f"   - Parallel execution success: {parallel_success}/{len(parallel_tests)} ({parallel_success/len(parallel_tests)*100:.1f}%)")
        
        # Error handling analysis
        error_tests = [r for r in self.test_results if r.get("status") != "success"]
        if error_tests:
            logger.info(f"\n❌ ERROR ANALYSIS:")
            error_types = {}
            for error_test in error_tests:
                error_type = error_test.get("error", "unknown")[:50]
                error_types[error_type] = error_types.get(error_type, 0) + 1
            
            for error_type, count in error_types.items():
                logger.info(f"   - {error_type}: {count} occurrences")
        
        # Conversation continuity analysis
        summary_tests = [r for r in successful_results if r.get("has_summary")]
        logger.info(f"\n🔗 CONVERSATION CONTINUITY:")
        logger.info(f"   - Tests with summaries: {len(summary_tests)}/{len(successful_results)} ({len(summary_tests)/len(successful_results)*100:.1f}%)")
        logger.info(f"   - Conversation ID used: {self.conversation_id}")

    def save_results_to_file(self, filename: str = None):
        """Lưu kết quả test vào file"""
        if filename is None:
            filename = f"comprehensive_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Tạo summary data
        summary_data = {
            "test_execution": {
                "timestamp": datetime.now().isoformat(),
                "total_tests": len(self.test_results),
                "successful_tests": len([r for r in self.test_results if r.get("status") == "success"]),
                "conversation_id": self.conversation_id
            },
            "detailed_results": self.test_results
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Đã lưu kết quả test vào {filename}")

async def main():
    """Main function để chạy comprehensive system test"""
    logger.info("🚀 Bắt đầu COMPREHENSIVE SYSTEM TEST")
    logger.info("Testing: Background DB + Parallel Processing + All G.1-G.5 Features")
    
    async with ComprehensiveSystemTester() as tester:
        # Test 1: Basic functionality
        await tester.test_basic_functionality()
        
        # Test 2: Parallel processing
        await tester.test_parallel_processing()
        
        # Test 3: Background DB operations
        await tester.test_background_db_operations()
        
        # Test 4: Edge cases and error handling
        await tester.test_edge_cases_and_error_handling()
        
        # Test 5: Conversation continuity
        await tester.test_conversation_continuity()
        
        # Test 6: Performance benchmarks
        await tester.test_performance_benchmarks()
        
        # Analyze results
        tester.analyze_comprehensive_results()
        
        # Save results
        tester.save_results_to_file()
        
        logger.info("\n" + "="*70)
        logger.info("🎉 COMPREHENSIVE SYSTEM TEST COMPLETED")
        logger.info("="*70)

if __name__ == "__main__":
    asyncio.run(main()) 