# Comprehensive Testing Guide

## Hi Nam! 🚀 Hướng dẫn kiểm tra toàn diện hệ thống

### 📋 **Tổng quan Testing Framework**

Đây là guide hoàn chỉnh để test tất cả các tính năng đã implement từ G.1 đến G.5, bao gồm:
- ⚡ Background DB Operations (G.2)
- 🎯 Parallel Tool Processing (G.3) 
- 🛡️ Advanced Error Handling (G.4)
- 📊 Enhanced Monitoring (G.5)
- 💨 Cache Optimization (G.1)

---

## 🔧 **Chuẩn bị Environment**

### 1. Install Dependencies
```bash
pip install aiohttp pytest pytest-asyncio
```

### 2. Start Server
```bash
# Terminal 1: Start main server
python main.py

# Terminal 2: Check server health
curl http://localhost:8000/health
```

### 3. Verify Database & Redis
```bash
# Check database connection
python -c "from app.db.database import get_db; print('DB OK')"

# Check Redis connection  
python -c "from app.services.cache_service import CacheService; print('Redis OK')"
```

---

## 🧪 **Test Execution**

### **Option 1: Comprehensive System Test (Recommended)**
```bash
# Chạy full test suite
python test_comprehensive_system.py
```

**Kết quả mong đợi:**
- ✅ 25-30 test cases
- ✅ >90% success rate
- ✅ <3000ms average response time
- ✅ Parallel processing detection
- ✅ Background task completion

### **Option 2: Manual API Testing**

#### A. Basic Functionality Test
```bash
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer test_token" \
  -d '{
    "message": "Xin chào! Tôi cần tư vấn về dinh dưỡng",
    "conversation_id": null
  }'
```

#### B. Parallel Tool Processing Test
```bash
curl -X POST "http://localhost:8000/api/chat" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer test_token" \
  -d '{
    "message": "Tôi cần gợi ý món ăn và đồ uống cho người tiểu đường",
    "conversation_id": 1
  }'
```

#### C. Background Task Status Check
```bash
curl "http://localhost:8000/api/background-task-status/{task_id}" \
  -H "Authorization: Bearer test_token"
```

---

## 📊 **Test Categories & Scenarios**

### **1. Basic Functionality Tests**
- ✅ Greeting và scope validation
- ✅ Food consultation requests
- ✅ Recipe search integration
- ✅ Health condition processing

### **2. Parallel Processing Tests** ⭐
- ✅ Simultaneous food + beverage requests
- ✅ Response time comparison (single vs parallel)
- ✅ Result completeness verification
- ✅ Error handling in parallel execution

### **3. Background DB Operations Tests** ⭐
- ✅ Rapid successive requests
- ✅ Non-blocking response verification
- ✅ Task completion monitoring
- ✅ Data consistency checks

### **4. Edge Cases & Error Handling**
- ✅ User information rejection
- ✅ Out-of-scope requests
- ✅ Empty/invalid messages
- ✅ Very long messages
- ✅ Network timeout scenarios

### **5. Conversation Continuity Tests**
- ✅ Context preservation across messages
- ✅ Summary system validation
- ✅ Health condition memory
- ✅ Follow-up question handling

### **6. Performance Benchmarks**
- ✅ Concurrent request handling
- ✅ Response time under load
- ✅ Memory usage tracking
- ✅ Cache hit rate verification

---

## 🎯 **Expected Performance Metrics**

### **Response Times:**
- ⚡ Basic requests: <2000ms
- ⚡ Parallel processing: <3000ms
- ⚡ Background tasks: Response <500ms, Complete <5000ms
- ⚡ Cache hits: <100ms

### **Success Rates:**
- 🎯 Overall: >95%
- 🎯 Parallel processing: >90%
- 🎯 Background tasks: >98%
- 🎯 Error recovery: >99%

### **Concurrency:**
- 🔄 5 concurrent requests: All successful
- 🔄 Response time degradation: <20%
- 🔄 No resource leaks
- 🔄 Graceful error handling

---

## 🔍 **Monitoring & Debugging**

### **1. Log Analysis**
```bash
# Monitor real-time logs
tail -f logs/app.log | grep -E "(⚡|🎯|⭐|🔧)"
```

### **2. Key Log Indicators**
- ⚡ `PARALLEL_PROCESSING`: Parallel execution detected
- 🎯 `BACKGROUND_TASK`: Background operation created
- ⭐ `ROUTER_DECISION`: Flow routing decisions
- 🔧 `CACHE_HIT/MISS`: Cache performance
- 📊 `PERFORMANCE_METRIC`: Response time tracking

### **3. Health Check Endpoints**
```bash
# System health
curl http://localhost:8000/health

# Database status
curl http://localhost:8000/api/health/db

# Redis status  
curl http://localhost:8000/api/health/redis
```

---

## 🚨 **Troubleshooting Common Issues**

### **Issue 1: Slow Response Times**
```bash
# Check database connections
python -c "from app.db.database import engine; print(engine.pool.size())"

# Check Redis latency
redis-cli --latency
```

### **Issue 2: Background Tasks Not Completing**
```bash
# Check background service status
curl http://localhost:8000/api/background-tasks/status

# Monitor task queue
python -c "from app.services.background_db_service import background_db_service; print(len(background_db_service.task_queue))"
```

### **Issue 3: Parallel Processing Not Working**
```bash
# Check router logic
grep "parallel_tool_runner" logs/app.log

# Verify asyncio.gather execution
grep "asyncio.gather" logs/app.log
```

### **Issue 4: Cache Problems**
```bash
# Clear cache
redis-cli FLUSHDB

# Check cache service
python -c "from app.services.cache_service import CacheService; print(CacheService.test_connection())"
```

---

## 📈 **Performance Optimization Tips**

### **1. Database Optimization**
- Ensure proper indexing on message, conversation tables
- Monitor connection pool usage
- Use EXPLAIN for slow queries

### **2. Redis Configuration**
```bash
# Recommended Redis settings
redis-cli CONFIG SET maxmemory-policy allkeys-lru
redis-cli CONFIG SET save "60 1000"
```

### **3. Application Settings**
```python
# In settings.py
BACKGROUND_TASK_TIMEOUT = 30  # seconds
REDIS_TTL_DEFAULT = 3600      # 1 hour
DB_POOL_SIZE = 20             # connections
```

---

## ✅ **Test Completion Checklist**

### **Before Testing:**
- [ ] Server is running on port 8000
- [ ] Database is connected and migrated
- [ ] Redis is running and accessible
- [ ] All dependencies installed
- [ ] Environment variables configured

### **During Testing:**
- [ ] Monitor logs for errors
- [ ] Check response times
- [ ] Verify data consistency
- [ ] Watch memory usage
- [ ] Validate all features work

### **After Testing:**
- [ ] Review test results JSON file
- [ ] Analyze performance metrics
- [ ] Check for memory leaks
- [ ] Validate background task completion
- [ ] Ensure no lingering processes

---

## 🎉 **Success Criteria**

✅ **PASS CONDITIONS:**
- All test categories achieve >90% success rate
- Average response time <3000ms
- Parallel processing efficiency >40% improvement
- Background tasks complete successfully
- No critical errors in logs
- System remains stable under load

❌ **FAIL CONDITIONS:**
- >10% test failure rate
- Response times >5000ms consistently
- Memory leaks detected
- Database connection issues
- Redis connectivity problems
- Critical unhandled exceptions

---

## 📞 **Support & Next Steps**

### **If Tests Pass:** 🎉
System is ready for production deployment!

### **If Tests Fail:** 🔧
1. Check troubleshooting section
2. Review logs for error patterns
3. Verify environment configuration
4. Test individual components
5. Apply fixes and retest

### **Performance Tuning:**
1. Analyze bottlenecks from test results
2. Optimize database queries
3. Fine-tune Redis settings
4. Adjust concurrency parameters
5. Monitor production metrics

**Happy Testing! 🚀** 