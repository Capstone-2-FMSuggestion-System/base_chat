import os
import logging
import json
import pinecone
import asyncio
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_huggingface import HuggingFaceEmbeddings
import math
import re
from typing import Tuple, Optional

# ⭐ IMPORT API KEY MANAGER để xoay vòng Gemini API keys
from app.services.api_key_manager import get_api_key_manager
# ⭐ IMPORT GEMINI MODEL POOL để thread-safe API access
from app.tools.gemini_model_pool import get_gemini_model_pool

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Tải biến môi trường
load_dotenv()

PINECONE_API_KEY = os.getenv("RECIPE_DB_PINECONE_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# ⭐ KHỞI TẠO API KEY MANAGER VÀ GEMINI MODEL POOL
api_key_manager = get_api_key_manager()
gemini_model_pool = get_gemini_model_pool(GEMINI_MODEL_NAME)

PINECONE_INDEX_NAME = "recipe-index"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
TEXT_KEY_IN_PINECONE = "text"

# ⭐ TỐI ƯU HÓA MẠNH: Giảm đáng kể batch size để tránh lỗi token limit
DOCUMENTS_PER_GEMINI_CALL = 80   # Giảm từ 120 xuống 80 để an toàn hơn
TOTAL_DOCS_IN_PINECONE = 2351

# ⭐ GIỚI HẠN KẾT QUẢ: Dừng khi đạt số lượng công thức mong muốn
MAX_RECIPES_RESULT = 20          # Giới hạn tối đa 20 công thức trả về

# ⭐ Dynamic batching constants - giảm mạnh để tránh lỗi JSON
MAX_CHAR_PER_BATCH = 200000      # Giảm từ 350k xuống 200k (~67k tokens)
MAX_SAFE_BATCH_SIZE = 60         # Giảm từ 100 xuống 60
MIN_BATCH_SIZE = 15              # Giảm từ 20 xuống 15

# ⭐ GEMINI OUTPUT TOKEN LIMIT - giảm để đảm bảo JSON response ổn định
MAX_GEMINI_OUTPUT_TOKENS = 4096  # Giảm từ 8192 xuống 4096

# ⭐ WORKER POOL CONFIGURATION: Số lượng Gemini Worker để xử lý song song
# Sẽ được tính động dựa trên số API key có sẵn, tối đa 7 worker
MAX_GEMINI_WORKERS = 7

# ⭐ CONFIGURATION CHO PARALLEL PROCESSING
WORKER_SLEEP_BETWEEN_TASKS = 0.1
WORKER_ERROR_SLEEP = 2
WORKER_TIMEOUT_SECONDS = 120

# ⭐ SANITIZATION CONFIGURATION
SANITIZE_INPUT_TEXT = True

# ⭐ RETRY CONFIGURATION
MAX_RETRIES_PER_BATCH = 3
RETRY_DELAY_SECONDS = 2

# ⭐ LOGGING CONFIGURATION
LOG_BATCH_DETAILS = True
LOG_WORKER_DETAILS = True

# ⭐ PERFORMANCE MONITORING
ENABLE_PERFORMANCE_LOGGING = True

def get_embedding_model():
    """⭐ Lấy embedding model từ global cache hoặc tạo mới nếu cần"""
    try:
        # Import function để lấy global embedding model
        from main import get_global_embedding_model
        global_model = get_global_embedding_model()
        
        if global_model is not None:
            logger.info("✅ Sử dụng pre-loaded embedding model từ global cache")
            return global_model
        else:
            logger.warning("⚠️ Global embedding model chưa được load, tạo mới...")
            return HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_NAME,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
    except ImportError:
        logger.warning("⚠️ Không thể import global embedding model, tạo mới...")
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

def init_connections() -> Tuple[Optional[pinecone.Index], Optional[HuggingFaceEmbeddings]]:
    """Khởi tạo kết nối với Pinecone client và embedding model. Gemini sẽ được config trong mỗi API call."""
    try:
        from pinecone import Pinecone
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        try:
            index_info = pc.describe_index(PINECONE_INDEX_NAME)
            logger.info(f"Pinecone index '{PINECONE_INDEX_NAME}' đã tồn tại.")
        except Exception as e:
            logger.error(f"Pinecone index '{PINECONE_INDEX_NAME}' không tồn tại: {str(e)}")
            raise ValueError(f"Pinecone index '{PINECONE_INDEX_NAME}' không tồn tại.") from e

        index = pc.Index(PINECONE_INDEX_NAME)
        logger.info(f"Đã tạo đối tượng Index cho: {PINECONE_INDEX_NAME}")

        # ⭐ SỬ DỤNG GLOBAL EMBEDDING MODEL
        embeddings_model = get_embedding_model()
        if embeddings_model:
            logger.info("✅ Đã sử dụng embedding model (pre-loaded hoặc mới tạo)")
        else:
            logger.error("❌ Không thể tạo embedding model")
            raise ValueError("Không thể tạo embedding model")

        # ⭐ KIỂM TRA API KEY MANAGER READINESS
        if api_key_manager.total_keys() == 0:
            logger.error("❌ Không có API key nào của Gemini được cấu hình trong ApiKeyManager. Recipe tool có thể không hoạt động.")
            # Có thể raise exception nếu muốn dừng hẳn
        else:
            logger.info(f"✅ Gemini integration ready - ApiKeyManager có {api_key_manager.total_keys()} keys")

        return index, embeddings_model
    except Exception as e:
        logger.error(f"Lỗi khi khởi tạo kết nối: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise

def sanitize_text_for_llm(text: str) -> str:
    """Làm sạch văn bản cải thiện để loại bỏ các ký tự có thể gây lỗi JSON."""
    if not text:
        return ""
    
    # Loại bỏ các ký tự điều khiển C0 (00-1F) và DEL (7F), ngoại trừ TAB, LF, CR
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
    
    # Thay thế dấu ngoặc kép thông minh bằng dấu ngoặc kép chuẩn
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace("'", "'").replace("'", "'")
    
    # Loại bỏ các escape sequence không cần thiết
    text = re.sub(r'\\[rn]', ' ', text)
    
    # Cắt bớt nếu quá dài (mỗi recipe không nên vượt quá 2000 chars)
    if len(text) > 2000:
        text = text[:1997] + "..."
        
    return text

def estimate_tokens(text: str) -> int:
    """Ước tính số token dựa trên character count (3 chars ≈ 1 token)"""
    return len(text) // 3

def create_dynamic_batches(matches: list) -> list:
    """
    ⭐ DYNAMIC BATCHING: Tạo batches dựa trên character count thay vì số lượng cố định
    """
    batches = []
    current_batch = []
    current_batch_chars = 0
    
    for match in matches:
        if TEXT_KEY_IN_PINECONE not in match.metadata:
            continue
            
        page_content = match.metadata[TEXT_KEY_IN_PINECONE]
        if SANITIZE_INPUT_TEXT:
            page_content = sanitize_text_for_llm(page_content)
            
        if not page_content:
            continue
            
        # Ước tính character count cho document này (bao gồm wrapper)
        doc_char_count = len(f"--- CÔNG THỨC ID: {match.id} ---\n{page_content}\n--- KẾT THÚC CÔNG THỨC ID: {match.id} ---\n\n")
        
        # Kiểm tra xem có thể thêm vào batch hiện tại không
        if (current_batch_chars + doc_char_count > MAX_CHAR_PER_BATCH and current_batch) or len(current_batch) >= MAX_SAFE_BATCH_SIZE:
            # Batch hiện tại đã đủ, lưu và tạo batch mới
            if current_batch:  # Chỉ lưu nếu có content
                batches.append(current_batch)
            current_batch = [match]
            current_batch_chars = doc_char_count
        else:
            # Thêm vào batch hiện tại
            current_batch.append(match)
            current_batch_chars += doc_char_count
    
    # Thêm batch cuối cùng nếu có
    if current_batch:
        batches.append(current_batch)
    
    return batches

def clean_json_response(response_text: str) -> str:
    """⭐ CẢI THIỆN: Làm sạch response từ Gemini trước khi parse JSON với nhiều fix patterns hơn"""
    try:
        cleaned = response_text.strip()
        
        # Loại bỏ markdown wrapper
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
            
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
            
        cleaned = cleaned.strip()
        
        # ⭐ CẢI THIỆN: Thêm nhiều pattern fix JSON phổ biến hơn
        
        # 1. Trailing comma trước } hoặc ]
        cleaned = re.sub(r',(\s*[}\]])', r'\1', cleaned)
        
        # 2. Missing quotes cho keys (chỉ fix nếu chưa có quotes)
        cleaned = re.sub(r'(\w+):\s*(["\[])', r'"\1": \2', cleaned)
        
        # 3. Single quotes thay vì double quotes cho keys và strings
        cleaned = re.sub(r"'([^']*?)'(\s*:\s*)", r'"\1"\2', cleaned)  # Keys
        cleaned = re.sub(r':\s*\'([^\']*?)\'', r': "\1"', cleaned)    # String values
        
        # 4. Thêm dấu phẩy thiếu giữa objects trong array
        cleaned = re.sub(r'}\s*{', r'}, {', cleaned)
        
        # 5. Loại bỏ control characters có thể gây lỗi JSON
        cleaned = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', cleaned)
        
        # 6. Fix escaped quotes thừa
        cleaned = re.sub(r'\\"', '"', cleaned)
        
        # 7. Đảm bảo JSON bắt đầu và kết thúc đúng format
        if not cleaned.startswith('[') and not cleaned.startswith('{'):
            # Tìm JSON array hoặc object đầu tiên
            array_match = re.search(r'(\[[\s\S]*\])', cleaned)
            object_match = re.search(r'(\{[\s\S]*\})', cleaned)
            
            if array_match:
                cleaned = array_match.group(1)
            elif object_match:
                cleaned = object_match.group(1)
        
        return cleaned
    except Exception as e:
        logger.warning(f"Lỗi khi clean JSON response: {e}")
        return response_text.strip()

def parse_json_with_fallback(response_text: str) -> dict:
    """⭐ MỚI: Parse JSON với multiple fallback strategies"""
    try:
        cleaned = clean_json_response(response_text)
        
        # Thử parse trực tiếp
        try:
            result = json.loads(cleaned)
            return {"success": True, "data": result}
        except json.JSONDecodeError as e:
            logger.warning(f"Direct JSON parse failed: {e}")
        
        # ⭐ Fallback 1: Tìm và extract JSON object/array từ text
        json_patterns = [
            r'\[[\s\S]*?\]',  # JSON array
            r'\{[\s\S]*?\}',  # JSON object
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, cleaned)
            for match in matches:
                try:
                    result = json.loads(match)
                    logger.info("✅ JSON parsed successfully với pattern matching")
                    return {"success": True, "data": result}
                except json.JSONDecodeError:
                    continue
        
        # ⭐ Fallback 2: Nếu là object nhưng mong đợi array, extract array từ object
        try:
            obj_result = json.loads(cleaned)
            if isinstance(obj_result, dict):
                # Tìm key chứa array
                for key, value in obj_result.items():
                    if isinstance(value, list):
                        logger.info(f"✅ Extracted array từ object key: {key}")
                        return {"success": True, "data": value}
        except json.JSONDecodeError:
            pass
        
        # ⭐ Fallback 3: Return structured error
        return {
            "success": False, 
            "error": "Invalid JSON from Gemini", 
            "raw_response": response_text[:500],  # Giới hạn độ dài
            "cleaned_response": cleaned[:500]
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"JSON parsing exception: {str(e)}",
            "raw_response": response_text[:500]
        }

async def call_gemini_api_async(prompt: str, api_key_for_this_call: str, task_description: str, max_retries: int = 3) -> str:
    """⭐ ASYNC VERSION với API KEY được truyền vào từ Worker Pool - SỬ DỤNG MODEL POOL"""
    prompt_char_count = len(prompt)
    estimated_tokens = estimate_tokens(prompt)
    
    if not api_key_for_this_call:
        logger.error("❌ Không có API key Gemini được truyền vào")
        return json.dumps({"error": "Không có API key Gemini khả dụng."})
    
    # Log key rotation (an toàn)
    masked_key = f"{api_key_for_this_call[:8]}..." if len(api_key_for_this_call) > 8 else "***"
    logger.info(f"🔍 {task_description}: Gọi Gemini với key {masked_key} - Chars: {prompt_char_count:,}, Est. tokens: {estimated_tokens:,}")
    
    loop = asyncio.get_running_loop()
    
    for attempt in range(max_retries):
        try:
            # ⭐ SỬ DỤNG MODEL POOL THAY VÌ genai.configure() (thread-safe)
            temp_gemini_model = gemini_model_pool.get_model_for_key(api_key_for_this_call)
            if not temp_gemini_model:
                logger.error(f"❌ Không tìm thấy model cho key {masked_key}")
                return json.dumps({"error": "Không tìm thấy model cho API key"})
            
            # ⭐ CHẠY GEMINI TRONG EXECUTOR để không block event loop
            response = await loop.run_in_executor(
                None,
                lambda: temp_gemini_model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=MAX_GEMINI_OUTPUT_TOKENS,
                        temperature=0.1
                    )
                )
            )
            
            if response.parts:
                response_text = response.text
                
                # ⭐ SỬ DỤNG PARSE_JSON_WITH_FALLBACK
                parse_result = parse_json_with_fallback(response_text)
                
                if parse_result["success"]:
                    # Trả về JSON string của data đã parse thành công
                    return json.dumps(parse_result["data"], ensure_ascii=False)
                else:
                    # Trả về structured error
                    logger.error(f"❌ JSON parse failed: {parse_result.get('error')}")
                    if attempt < max_retries - 1:
                        logger.info(f"🔄 Retry attempt {attempt + 1}/{max_retries} due to JSON parse error")
                        await asyncio.sleep(5)
                        continue
                    return json.dumps({
                        "error": parse_result["error"],
                        "raw_snippet": parse_result.get("raw_response", "")[:200]
                    })
            else:
                logger.warning("Gemini không trả về nội dung.")
                logger.warning(f"Gemini response: {response}")
                if response.prompt_feedback and response.prompt_feedback.block_reason:
                    logger.warning(f"Bị chặn bởi: {response.prompt_feedback.block_reason}")
                return "[]"
                
        except Exception as e:
            error_str = str(e).lower()
            
            if "429" in error_str or "resource_exhausted" in error_str or "quota" in error_str:
                retry_delay = 15 + attempt * 10
                delay_match = re.search(r'retry in (\d+)s', error_str)
                if delay_match: 
                    retry_delay = max(int(delay_match.group(1)), retry_delay)
                logger.warning(f"🚫 Quota limit Gemini, retry sau {retry_delay}s (#{attempt+1}/{max_retries})")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    
            elif "invalid_argument" in error_str and ("token" in error_str or "size" in error_str):
                logger.error(f"💥 Prompt quá lớn: {estimated_tokens:,} tokens, {prompt_char_count:,} chars")
                return json.dumps({"error_too_large": True, "estimated_tokens": estimated_tokens})
                
            elif "500" in error_str or "internal" in error_str or "unavailable" in error_str:
                retry_delay = 10 + attempt * 15
                logger.warning(f"🔧 Server error Gemini, retry sau {retry_delay}s (#{attempt+1}/{max_retries})")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
            else:
                logger.error(f"❌ Lỗi không xác định Gemini (#{attempt+1}/{max_retries}): {str(e)}")
                if attempt == max_retries - 1:
                    return json.dumps({"error": f"Gemini API error: {str(e)}"})
                await asyncio.sleep(10)
                
    logger.error(f"💥 Thất bại hoàn toàn sau {max_retries} lần thử với Gemini")
    return json.dumps({"error": "Không thể kết nối Gemini sau nhiều lần thử"})

async def search_and_filter_recipes_async(user_query: str) -> str:
    """
    ⭐ ASYNC VERSION: Search với dynamic batching và xử lý bất đồng bộ đồng thời
    """
    try:
        # ⭐ CHẠY INIT_CONNECTIONS TRONG EXECUTOR
        loop = asyncio.get_running_loop()
        pinecone_index, embeddings_model = await loop.run_in_executor(None, init_connections)
    except Exception as e:
        logger.error(f"💥 Không thể khởi tạo kết nối: {str(e)}")
        return json.dumps({"error": f"Connection init failed: {str(e)}"})

    all_selected_recipes = []
    error_reports = []

    try:
        logger.info(f"🔍 Đang embed query: '{user_query}'")
        
        # ⭐ CHẠY EMBEDDING TRONG EXECUTOR
        query_vector = await loop.run_in_executor(None, embeddings_model.embed_query, user_query)
        logger.info("✅ Đã embed query thành công")

        # ⭐ GIẢM PINECONE TOP_K để giảm tải
        actual_top_k = min(TOTAL_DOCS_IN_PINECONE, 3000)  # Giảm từ 10000 xuống 3000
        logger.info(f"🔍 Query Pinecone với top_k={actual_top_k}")
        
        # ⭐ CHẠY PINECONE QUERY TRONG EXECUTOR
        query_response = await loop.run_in_executor(
            None,
            lambda: pinecone_index.query(
                vector=query_vector,
                top_k=actual_top_k,
                include_metadata=True,
            )
        )
        
        retrieved_matches = query_response.matches
        logger.info(f"✅ Lấy được {len(retrieved_matches)} documents từ Pinecone")

        if not retrieved_matches:
            return json.dumps([])

        # ⭐ SỬ DỤNG DYNAMIC BATCHING
        dynamic_batches = create_dynamic_batches(retrieved_matches)
        logger.info(f"🔄 Chia thành {len(dynamic_batches)} dynamic batches")

        # ⭐ KHỞI TẠO WORKER POOL CONFIGURATION
        NUM_GEMINI_WORKERS = min(api_key_manager.total_keys(), MAX_GEMINI_WORKERS)
        if NUM_GEMINI_WORKERS == 0: 
            NUM_GEMINI_WORKERS = 1  # Ít nhất 1 worker nếu có key
        
        logger.info(f"🤖 Khởi tạo Worker Pool với {NUM_GEMINI_WORKERS} workers (có {api_key_manager.total_keys()} API keys)")
        logger.info(f"📊 Sẽ xử lý {len(dynamic_batches)} batches với worker pool parallelism")
        
        # ⭐ KHỞI TẠO QUEUE VÀ RESULT STORAGE
        work_queue = asyncio.Queue()
        results_list = []  # Thread-safe với coroutine
        error_reports_list = []
        
        # ⭐ TẠO CÁC TASK_DATA VÀ ĐƯA VÀO QUEUE
        for batch_idx, batch_matches in enumerate(dynamic_batches):
            logger.info(f"📦 Chuẩn bị batch {batch_idx + 1}/{len(dynamic_batches)} ({len(batch_matches)} docs)")
            
            context_parts = []
            
            for match in batch_matches:
                doc_id = match.id
                if TEXT_KEY_IN_PINECONE in match.metadata:
                    page_content = match.metadata[TEXT_KEY_IN_PINECONE]
                    if SANITIZE_INPUT_TEXT:
                        page_content = sanitize_text_for_llm(page_content)
                    
                    if page_content:
                        context_parts.append(f"--- CÔNG THỨC ID: {doc_id} ---\n{page_content}\n--- KẾT THÚC CÔNG THỨC ID: {doc_id} ---")

            if not context_parts:
                logger.info(f"⏭️ Batch {batch_idx + 1} rỗng, bỏ qua")
                continue

            batch_context = "\n\n".join(context_parts)
            estimated_tokens = estimate_tokens(batch_context)
            
            logger.info(f"📊 Batch {batch_idx + 1}: {len(batch_context):,} chars, ~{estimated_tokens:,} tokens")
            
            # Kiểm tra an toàn token
            if estimated_tokens > 300000:  # 300k tokens limit cho an toàn
                logger.error(f"⚠️ Batch {batch_idx + 1} quá lớn ({estimated_tokens:,} tokens), bỏ qua")
                error_reports_list.append({
                    "error_batch": batch_idx + 1,
                    "message": f"Batch quá lớn: {estimated_tokens:,} tokens",
                    "doc_count": len(batch_matches)
                })
                continue

            # ⭐ PROMPT TỐI ƯU HÓA MẠNH CHO JSON - YÊU CẦU CỰC KỲ CHẶT CHẼ
            optimized_prompt = f'''NHIỆM VỤ: Phân tích query "{user_query}" và chọn các công thức phù hợp NHẤT.

🚨 QUY TẮC TUYỆT ĐỐI:
- CHỈ trả về JSON array hợp lệ
- KHÔNG thêm text, giải thích, markdown
- KHÔNG sử dụng dấu ngoặc kép thông minh (" ")
- CHỈ sử dụng dấu ngoặc kép ASCII chuẩn (")
- KHÔNG có trailing comma
- Nếu không có kết quả: []

📋 ĐỊNH DẠNG BẮT BUỘC:
[{{"id":"recipe_id","name":"tên công thức","url":"link hoặc null","ingredients_summary":"nguyên liệu chính"}}]

✅ VÍ DỤ CHÍNH XÁC:
[{{"id":"recipe_123","name":"Salad giảm cân","url":"https://example.com","ingredients_summary":"Rau xanh, cà chua"}},{{"id":"recipe_456","name":"Sinh tố rau củ","url":null,"ingredients_summary":"Cải bó xôi, chuối"}}]

❌ TUYỆT ĐỐI KHÔNG:
- Không markdown: ```json
- Không text thêm: "Dưới đây là..."
- Không trailing comma: }},]
- Không smart quotes: "text"

DỮLIỆU CÔNG THỨC:
{batch_context}

TRẢ VỀ JSON ARRAY:'''

            # ⭐ ĐƯA TASK VÀO QUEUE
            if optimized_prompt.strip():
                task_data = {
                    "prompt": optimized_prompt,
                    "batch_num": batch_idx + 1,
                    "doc_count": len(batch_matches)
                }
                await work_queue.put(task_data)

        # ⭐ HÀM GEMINI WORKER VỚI EARLY STOPPING
        async def gemini_worker(worker_id: int):
            logger.info(f"🤖 Worker {worker_id}: Bắt đầu hoạt động.")
            while True:
                try:
                    # ⭐ KIỂM TRA EARLY STOPPING: Nếu đã đủ 20 recipes thì dừng worker
                    if len(results_list) >= MAX_RECIPES_RESULT:
                        logger.info(f"🛑 Worker {worker_id}: Đã đạt {MAX_RECIPES_RESULT} recipes, dừng worker.")
                        # Đánh dấu task done và thoát
                        try:
                            task_data = work_queue.get_nowait()
                            work_queue.task_done()
                        except:
                            pass
                        break
                    
                    task_data = await work_queue.get()
                    if task_data is None:  # Tín hiệu dừng
                        work_queue.task_done()
                        logger.info(f"🤖 Worker {worker_id}: Nhận tín hiệu dừng.")
                        break

                    # ⭐ KIỂM TRA LẠI SAU KHI LẤY TASK (vì có thể worker khác đã đủ)
                    if len(results_list) >= MAX_RECIPES_RESULT:
                        logger.info(f"🛑 Worker {worker_id}: Đã đạt {MAX_RECIPES_RESULT} recipes sau khi lấy task, bỏ qua.")
                        work_queue.task_done()
                        break

                    prompt_to_process = task_data["prompt"]
                    batch_num_log = task_data["batch_num"]
                    doc_count_log = task_data["doc_count"]
                    
                    task_description = f"Worker {worker_id} - Batch {batch_num_log}"
                    
                    # Mỗi worker tự lấy key mới cho mỗi task nó xử lý
                    api_key_for_call = api_key_manager.get_next_key()
                    if not api_key_for_call:
                        logger.error(f"🤖 Worker {worker_id}: Không có API key, bỏ qua batch {batch_num_log}")
                        error_reports_list.append({
                            "error_batch": batch_num_log, "error_type": "no_api_key",
                            "message": "Không có API key khả dụng", "doc_count": doc_count_log
                        })
                        work_queue.task_done()
                        continue

                    logger.info(f"🤖 Worker {worker_id}: Đang xử lý Batch {batch_num_log} ({doc_count_log} docs) với key ...{api_key_for_call[-4:]}")
                    
                    # Gọi API Gemini
                    gemini_response_text = await call_gemini_api_async(
                        prompt_to_process, 
                        api_key_for_call,
                        task_description
                    )
                    
                    # Xử lý response (parse JSON, v.v...)
                    parse_attempt = parse_json_with_fallback(gemini_response_text)
                    if parse_attempt["success"]:
                        parsed_data = parse_attempt["data"]
                        if isinstance(parsed_data, list):
                            valid_items = [item for item in parsed_data if isinstance(item, dict) and "id" in item]
                            if valid_items:
                                # ⭐ KIỂM TRA VÀ GIỚI HẠN SỐ LƯỢNG KHI THÊM VÀO RESULTS
                                current_count = len(results_list)
                                remaining_slots = MAX_RECIPES_RESULT - current_count
                                
                                if remaining_slots > 0:
                                    # Chỉ thêm số lượng recipes còn thiếu
                                    items_to_add = valid_items[:remaining_slots]
                                    results_list.extend(items_to_add)
                                    
                                    logger.info(f"🤖 Worker {worker_id}: Batch {batch_num_log} thành công, thêm {len(items_to_add)}/{len(valid_items)} items. Tổng: {len(results_list)}/{MAX_RECIPES_RESULT}")
                                    
                                    # Nếu đã đủ, log thông báo early stopping
                                    if len(results_list) >= MAX_RECIPES_RESULT:
                                        logger.info(f"🎯 Worker {worker_id}: Đã đạt giới hạn {MAX_RECIPES_RESULT} recipes. Early stopping!")
                                else:
                                    logger.info(f"🛑 Worker {worker_id}: Đã đủ {MAX_RECIPES_RESULT} recipes, bỏ qua batch {batch_num_log}")
                            else:
                                logger.info(f"🤖 Worker {worker_id}: Batch {batch_num_log} không có valid items.")
                        else:
                            logger.warning(f"🤖 Worker {worker_id}: Batch {batch_num_log} trả về định dạng không phải list: {type(parsed_data)}")
                            error_reports_list.append({"error_batch": batch_num_log, "error_type": "invalid_gemini_response_format", "doc_count": doc_count_log})
                    else:
                        logger.error(f"🤖 Worker {worker_id}: Batch {batch_num_log} lỗi parse JSON: {parse_attempt.get('error')}")
                        error_reports_list.append({
                            "error_batch": batch_num_log, "error_type": "json_parse_failed", 
                            "message": parse_attempt.get('error'), "doc_count": doc_count_log,
                            "raw_snippet": parse_attempt.get('raw_response')
                        })
                    
                    work_queue.task_done()
                    
                    # ⭐ KIỂM TRA EARLY STOPPING SAU KHI XONG TASK
                    if len(results_list) >= MAX_RECIPES_RESULT:
                        logger.info(f"🎯 Worker {worker_id}: Đã đạt {MAX_RECIPES_RESULT} recipes, dừng worker sớm.")
                        break
                    
                    # Thời gian nghỉ nhỏ sau mỗi batch của một worker
                    await asyncio.sleep(WORKER_SLEEP_BETWEEN_TASKS)

                except asyncio.CancelledError:
                    logger.info(f"🤖 Worker {worker_id}: Bị cancel.")
                    break
                except Exception as e:
                    logger.error(f"🤖 Worker {worker_id}: Lỗi không mong muốn: {e}", exc_info=True)
                    if 'task_data' in locals() and task_data and 'batch_num' in task_data:
                         error_reports_list.append({"error_batch": task_data['batch_num'], "error_type": "worker_exception", "message": str(e), "doc_count": task_data.get('doc_count',0)})
                    if 'task_data' in locals() and task_data is not None:
                         work_queue.task_done()
                    await asyncio.sleep(WORKER_ERROR_SLEEP)

        # ⭐ KHỞI TẠO VÀ CHẠY CÁC WORKER VỚI EARLY STOPPING
        worker_tasks = []
        for i in range(NUM_GEMINI_WORKERS):
            worker_tasks.append(asyncio.create_task(gemini_worker(i + 1)))

        # ⭐ GIÁM SÁT EARLY STOPPING: Chờ queue xử lý hoặc đạt giới hạn
        try:
            while not work_queue.empty() and len(results_list) < MAX_RECIPES_RESULT:
                await asyncio.sleep(0.5)  # Kiểm tra định kỳ
            
            # Nếu đã đạt giới hạn, cancel các worker còn lại
            if len(results_list) >= MAX_RECIPES_RESULT:
                logger.info(f"🎯 Đã đạt giới hạn {MAX_RECIPES_RESULT} recipes, dừng toàn bộ workers sớm.")
                
                # Cancel các worker tasks
                for task in worker_tasks:
                    if not task.done():
                        task.cancel()
                
                # Clear remaining queue items
                try:
                    while not work_queue.empty():
                        await work_queue.get()
                        work_queue.task_done()
                except:
                    pass
            else:
                # Chờ tất cả các item trong queue được xử lý bình thường
                await work_queue.join()
        
        except Exception as e:
            logger.error(f"💥 Lỗi trong giám sát early stopping: {e}")
            await work_queue.join()  # Fallback to normal join

        # Gửi tín hiệu dừng cho tất cả worker (nếu chưa cancel)
        for _ in range(NUM_GEMINI_WORKERS):
            try:
                await work_queue.put(None)
            except:
                pass

        # Chờ tất cả worker hoàn thành hoặc cancel
        await asyncio.gather(*worker_tasks, return_exceptions=True)

        logger.info(f"🏁 Tất cả các worker đã hoàn thành. Tổng hợp kết quả từ {len(results_list)} recipes...")

        # ⭐ LỌC TRÙNG LẶP BẰNG TÊN CHUẨN HÓA
        def normalize_recipe_name(name: str) -> str:
            """Chuẩn hóa tên recipe để so sánh trùng lặp"""
            if not name:
                return ""
            # Chuyển về lowercase, loại bỏ dấu cách, dấu gạch ngang, ký tự đặc biệt
            import unicodedata
            normalized = unicodedata.normalize('NFD', str(name).lower())
            normalized = ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')  # Loại bỏ dấu
            normalized = re.sub(r'[^a-z0-9]', '', normalized)  # Chỉ giữ chữ và số
            return normalized

        if results_list:
            logger.info(f"🔄 Bắt đầu lọc trùng lặp từ {len(results_list)} recipes...")
            
            final_unique_recipes = []
            seen_normalized_names = set()
            
            for recipe_item in results_list:
                if not isinstance(recipe_item, dict) or not recipe_item.get("name"):
                    continue
                    
                normalized_name = normalize_recipe_name(recipe_item["name"])
                if normalized_name and normalized_name not in seen_normalized_names:
                    final_unique_recipes.append(recipe_item)
                    seen_normalized_names.add(normalized_name)
                    
                    # ⭐ EARLY STOPPING TRONG LỌC TRÙNG LẶP: Dừng khi đủ 20 recipes
                    if len(final_unique_recipes) >= MAX_RECIPES_RESULT:
                        logger.info(f"🎯 Đã đạt {MAX_RECIPES_RESULT} recipes unique, dừng lọc trùng lặp sớm.")
                        break
                else:
                    logger.debug(f"Đã lọc recipe trùng lặp: {recipe_item.get('name', 'Unknown')}")
            
            results_list = final_unique_recipes
            logger.info(f"✅ Sau khi lọc trùng lặp: {len(results_list)} recipes duy nhất (giới hạn tối đa {MAX_RECIPES_RESULT})")

        # ⭐ XỬ LÝ KẾT QUẢ CUỐI CÙNG
        if not results_list and error_reports_list:
            return json.dumps({
                "message": "Không có công thức nào được chọn và có lỗi xảy ra trong quá trình xử lý.",
                "errors": error_reports_list[:5]  # Giới hạn số lỗi hiển thị
            }, ensure_ascii=False, indent=2)
        elif not results_list:
            return json.dumps([])
        elif error_reports_list:
            # Có recipes nhưng cũng có lỗi - chỉ trả về recipes
            logger.warning(f"⚠️ Có {len(error_reports_list)} lỗi nhưng vẫn tìm được {len(results_list)} recipes")
            return json.dumps(results_list, ensure_ascii=False, indent=2)
        else:
            return json.dumps(results_list, ensure_ascii=False, indent=2)

    except Exception as e:
        logger.error(f"💥 Critical error trong search_and_filter_recipes_async: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return json.dumps({"error": f"System error: {str(e)}"})

# ⭐ WRAPPER ĐỒNG BỘ để tương thích với code hiện tại
def search_and_filter_recipes(user_query: str) -> str:
    """
    Wrapper đồng bộ cho search_and_filter_recipes_async để tương thích với code hiện tại
    """
    return asyncio.run(search_and_filter_recipes_async(user_query))

async def main_test_async():
    """⭐ ASYNC TEST FUNCTION"""
    try:
        query = "Món ăn cho giảm cân, ít calo, nhiều rau xanh"
        
        print(f"\n🔍 Testing ASYNC với query: '{query}'")
        
        import time
        start_time = time.time()
        
        result = await search_and_filter_recipes_async(query)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n⚡ ASYNC PERFORMANCE: {duration:.2f} seconds")
        print("\n✅ KẾT QUẢ:")
        try:
            parsed = json.loads(result)
            if isinstance(parsed, list):
                print(f"Tìm được {len(parsed)} recipes")
            else:
                print("Response không phải list recipes")
            print(json.dumps(parsed, indent=2, ensure_ascii=False))
        except json.JSONDecodeError:
            print("Raw result:", result)

    except ValueError as ve: 
        print(f"💥 Lỗi cấu hình: {ve}")
    except Exception as e:
        logger.error(f"💥 Lỗi không mong muốn: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    # ⭐ CHẠY ASYNC TEST
    asyncio.run(main_test_async())