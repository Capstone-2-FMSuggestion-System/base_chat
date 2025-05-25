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

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Tải biến môi trường
load_dotenv()

PINECONE_API_KEY = os.getenv("RECIPE_DB_PINECONE_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

# ⭐ KHỞI TẠO API KEY MANAGER
api_key_manager = get_api_key_manager()

PINECONE_INDEX_NAME = "recipe-index"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
TEXT_KEY_IN_PINECONE = "text"

# ⭐ TỐI ƯU HÓA: Giảm mạnh batch size để tránh lỗi token limit
DOCUMENTS_PER_GEMINI_CALL = 120  # Giảm từ 500 xuống 120 để an toàn hơn
TOTAL_DOCS_IN_PINECONE = 2351

# ⭐ Dynamic batching constants - ưu tiên character count
MAX_CHAR_PER_BATCH = 350000  # ~116k tokens (giả sử 3 chars = 1 token), an toàn cho gemini-1.5-flash
MAX_SAFE_BATCH_SIZE = 100     # Fallback limit nếu dynamic batching fails
MIN_BATCH_SIZE = 20          # Minimum documents per batch

# ⭐ CONCURRENCY CONTROL: Giới hạn số lượng Gemini API calls đồng thời
MAX_CONCURRENT_GEMINI_CALLS = 7  # Bắt đầu với 4 calls đồng thời để an toàn

# Cờ để bật/tắt việc làm sạch văn bản
SANITIZE_INPUT_TEXT = True

# Kiểm tra API keys
if not PINECONE_API_KEY:
    logger.error("PINECONE_API_KEY không tìm thấy trong file .env")
    raise ValueError("PINECONE_API_KEY không tìm thấy trong file .env")

# ⭐ KIỂM TRA API KEY MANAGER
if not api_key_manager.is_healthy():
    logger.error("❌ KHÔNG CÓ GEMINI API KEY NÀO ĐƯỢC CẤU HÌNH! Recipe tool sẽ không hoạt động.")
    logger.error("Vui lòng cấu hình ít nhất một API key trong .env file")
else:
    logger.info(f"✅ ApiKeyManager ready với {api_key_manager.total_keys()} API keys")

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

        logger.info(f"Đang tải mô hình embedding từ {EMBEDDING_MODEL_NAME}...")
        embeddings_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info("Đã tải thành công mô hình embedding.")

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

async def call_gemini_api_async(prompt: str, max_retries: int = 3) -> str:
    """⭐ ASYNC VERSION với API KEY ROTATION và cải thiện prompt engineering cho JSON"""
    prompt_char_count = len(prompt)
    estimated_tokens = estimate_tokens(prompt)
    
    # ⭐ LẤY API KEY TỪ KEY MANAGER
    current_api_key = api_key_manager.get_next_key()
    if current_api_key is None:
        logger.error("❌ Không có API key Gemini khả dụng từ ApiKeyManager")
        return json.dumps({"error": "Không có API key Gemini khả dụng."})
    
    # Log key rotation (an toàn)
    masked_key = f"{current_api_key[:8]}..." if len(current_api_key) > 8 else "***"
    logger.info(f"🔍 Gọi Gemini ASYNC với key {masked_key} - Chars: {prompt_char_count:,}, Est. tokens: {estimated_tokens:,}")
    
    loop = asyncio.get_running_loop()
    
    for attempt in range(max_retries):
        try:
            # ⭐ CẤU HÌNH GEMINI VỚI KEY HIỆN TẠI VÀ TẠO MODEL MỚI
            genai.configure(api_key=current_api_key)
            temp_gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
            
            # ⭐ CHẠY GEMINI TRONG EXECUTOR để không block event loop
            response = await loop.run_in_executor(
                None,
                lambda: temp_gemini_model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=8192,
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

        # ⭐ KHỞI TẠO SEMAPHORE ĐỂ KIỂM SOÁT CONCURRENCY
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_GEMINI_CALLS)
        logger.info(f"🚦 Khởi tạo semaphore với giới hạn {MAX_CONCURRENT_GEMINI_CALLS} concurrent calls")
        
        # ⭐ HÀM WRAPPER VỚI SEMAPHORE CONTROL
        async def process_batch_with_semaphore(batch_prompt: str, batch_num_for_log: int) -> str:
            async with semaphore:
                logger.info(f"🚦 Batch {batch_num_for_log}: Bắt đầu xử lý (semaphore acquired, {MAX_CONCURRENT_GEMINI_CALLS - semaphore._value} slots used)")
                try:
                    result = await call_gemini_api_async(batch_prompt)
                    # ⭐ TÙY CHỌN: Thêm delay nhỏ để giảm tải API thêm nữa
                    await asyncio.sleep(0.1)  # 0.1 giây delay giữa các calls
                    logger.info(f"🚦 Batch {batch_num_for_log}: Hoàn thành xử lý (semaphore released)")
                    return result
                except Exception as e:
                    logger.error(f"🚦 Batch {batch_num_for_log}: Lỗi trong semaphore block - {str(e)}")
                    return json.dumps({"error": f"Batch {batch_num_for_log} failed: {str(e)}"})

        # ⭐ TẠO TASKS CHO TẤT CẢ BATCHES ĐỂ XỬ LÝ ĐỒNG THỜI VỚI SEMAPHORE
        gemini_tasks = []
        batch_prompts = []
        
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
                error_reports.append({
                    "error_batch": batch_idx + 1,
                    "message": f"Batch quá lớn: {estimated_tokens:,} tokens",
                    "doc_count": len(batch_matches)
                })
                continue

            # ⭐ PROMPT TỐI ƯU HÓA CHO JSON - CẢI THIỆN THEO YÊU CẦU
            optimized_prompt = f'''Bạn là AI chuyên gia ẩm thực. Phân tích query "{user_query}" và chọn các công thức phù hợp NHẤT.

⚠️ TUYỆT ĐỐI CHỈ TRẢ VỀ MỘT DANH SÁCH JSON (JSON ARRAY) HỢP LỆ. KHÔNG THÊM bất kỳ văn bản nào trước hoặc sau danh sách JSON.

📋 YÊU CẦU CHÍNH XÁC:
1. Chỉ chọn công thức thực sự liên quan đến query
2. Trích xuất: id, name, url, ingredients_summary  
3. Đảm bảo tất cả các chuỗi trong JSON được đặt trong dấu ngoặc kép chuẩn (\")
4. Đảm bảo không có dấu phẩy thừa ở cuối danh sách hoặc cuối đối tượng
5. Nếu không có kết quả nào, trả về một danh sách JSON rỗng: []

✅ VÍ DỤ VỀ OUTPUT JSON HỢP LỆ:
[
  {{
    "id": "recipe_123", 
    "name": "Salad giảm cân với rau xanh",
    "url": "https://example.com/recipe_123",
    "ingredients_summary": "Rau xanh, cà chua, dưa chuột, dầu oliu"
  }},
  {{
    "id": "recipe_456",
    "name": "Sinh tố rau củ ít calo", 
    "url": null,
    "ingredients_summary": "Cải bó xôi, chuối, táo, nước"
  }}
]

DỮLIỆU CÔNG THỨC:
{batch_context}

🔥 CHỈ TRẢ VỀ JSON ARRAY - KHÔNG GIẢI THÍCH, KHÔNG MARKDOWN, KHÔNG VĂN BẢN THÊM:'''

            # ⭐ THÊM TASK VÀO DANH SÁCH ĐỂ XỬ LÝ ĐỒNG THỜI VỚI SEMAPHORE CONTROL
            if optimized_prompt.strip():
                batch_number_for_logging = batch_idx + 1
                gemini_tasks.append(process_batch_with_semaphore(optimized_prompt, batch_number_for_logging))
                batch_prompts.append((batch_number_for_logging, len(batch_matches)))  # Track batch info

        # ⭐ XỬ LÝ TẤT CẢ BATCHES ĐỒNG THỜI với asyncio.gather VÀ SEMAPHORE CONTROL
        if gemini_tasks:
            logger.info(f"🚀 Bắt đầu xử lý {len(gemini_tasks)} batches với SEMAPHORE (max {MAX_CONCURRENT_GEMINI_CALLS} concurrent)...")
            
            # Gọi tất cả tasks đồng thời với return_exceptions=True
            # Semaphore sẽ tự động kiểm soát số lượng calls thực sự được thực hiện đồng thời
            gemini_responses_or_exceptions = await asyncio.gather(*gemini_tasks, return_exceptions=True)
            
            # ⭐ XỬ LÝ KẾT QUẢ TỪ asyncio.gather
            for task_idx, (result, batch_info) in enumerate(zip(gemini_responses_or_exceptions, batch_prompts)):
                batch_num, doc_count = batch_info
                
                # Kiểm tra xem result có phải là Exception không
                if isinstance(result, Exception):
                    logger.error(f"💥 Batch {batch_num} failed với exception: {str(result)}")
                    error_reports.append({
                        "error_batch": batch_num,
                        "message": f"Task exception: {str(result)}",
                        "doc_count": doc_count
                    })
                    continue
                
                # result là gemini_response_text thành công
                gemini_response = result
                
                # ⭐ XỬ LÝ RESPONSE CẢI THIỆN VỚI ERROR_TOO_LARGE HANDLING
                try:
                    # Kiểm tra error_too_large từ call_gemini_api_async trước
                    if gemini_response.startswith('{"error_too_large":'):
                        error_data = json.loads(gemini_response)
                        logger.error(f"💥 Batch {batch_num} quá lớn cho Gemini: {error_data.get('estimated_tokens', 'unknown')} tokens")
                        logger.warning(f"⚠️ Cần giảm MAX_CHAR_PER_BATCH hiện tại ({MAX_CHAR_PER_BATCH:,}) hoặc DOCUMENTS_PER_GEMINI_CALL hiện tại ({DOCUMENTS_PER_GEMINI_CALL})")
                        error_reports.append({
                            "error_batch": batch_num,
                            "error_type": "batch_too_large",
                            "message": f"Batch quá lớn: {error_data.get('estimated_tokens', 'unknown')} tokens",
                            "doc_count": doc_count,
                            "suggestion": "Giảm MAX_CHAR_PER_BATCH hoặc batch size"
                        })
                        continue
                    
                    # Kiểm tra các error responses khác từ call_gemini_api_async
                    if gemini_response.startswith('{"error":'):
                        error_data = json.loads(gemini_response)
                        error_msg = error_data.get('error', 'Unknown Gemini error')
                        logger.error(f"💥 Batch {batch_num} Gemini error: {error_msg}")
                        error_reports.append({
                            "error_batch": batch_num,
                            "error_type": "gemini_api_error",
                            "message": error_msg,
                            "doc_count": doc_count
                        })
                        continue
                    
                    # Parse JSON với fallback handling
                    parse_result = parse_json_with_fallback(gemini_response)
                    
                    if parse_result["success"]:
                        batch_recipes = parse_result["data"]
                        
                        # ⭐ XỬ LÝ TRƯỜNG HỢP GEMINI TRẢ VỀ OBJECT THAY VÌ LIST
                        if isinstance(batch_recipes, dict):
                            logger.warning(f"⚠️ Batch {batch_num}: Gemini trả về object thay vì array")
                            # Thử extract array từ object
                            extracted_array = None
                            for key, value in batch_recipes.items():
                                if isinstance(value, list):
                                    logger.info(f"✅ Batch {batch_num}: Extracted array từ key '{key}'")
                                    extracted_array = value
                                    break
                            
                            if extracted_array:
                                batch_recipes = extracted_array
                            else:
                                logger.warning(f"⚠️ Batch {batch_num}: Không tìm thấy array trong object, bỏ qua")
                                error_reports.append({
                                    "error_batch": batch_num,
                                    "error_type": "object_instead_of_array",
                                    "message": "Gemini trả về object thay vì array và không có array con",
                                    "doc_count": doc_count
                                })
                                continue
                        
                        if isinstance(batch_recipes, list):
                            if batch_recipes:
                                valid_recipes = [r for r in batch_recipes if isinstance(r, dict) and 'id' in r and 'name' in r]
                                all_selected_recipes.extend(valid_recipes)
                                logger.info(f"✅ Batch {batch_num}: {len(valid_recipes)} valid recipes từ {len(batch_recipes)} items")
                            else:
                                logger.info(f"📭 Batch {batch_num}: Không tìm thấy recipe phù hợp (empty array)")
                        else:
                            logger.warning(f"⚠️ Batch {batch_num}: Response vẫn không phải array sau xử lý: {type(batch_recipes)}")
                            error_reports.append({
                                "error_batch": batch_num,
                                "error_type": "invalid_response_type",
                                "message": f"Response type không hợp lệ: {type(batch_recipes)}",
                                "doc_count": doc_count
                            })
                    else:
                        # Parse failed với structured error
                        logger.error(f"💥 Batch {batch_num} JSON parse failed: {parse_result.get('error')}")
                        error_reports.append({
                            "error_batch": batch_num,
                            "error_type": "json_parse_failed",
                            "message": parse_result.get('error', 'JSON parse failed'),
                            "raw_snippet": parse_result.get('raw_response', '')[:200],
                            "doc_count": doc_count
                        })

                except json.JSONDecodeError as e:
                    logger.error(f"💥 Batch {batch_num} Critical JSON decode error: {e}")
                    logger.error(f"Raw response sample: {gemini_response[:300]}...")
                    error_reports.append({
                        "error_batch": batch_num,
                        "error_type": "critical_json_error",
                        "message": f"Critical JSON decode error: {str(e)}",
                        "raw_snippet": gemini_response[:200],
                        "doc_count": doc_count
                    })
                except Exception as e:
                    logger.error(f"💥 Batch {batch_num} Unexpected error: {e}")
                    error_reports.append({
                        "error_batch": batch_num,
                        "error_type": "unexpected_error",
                        "message": f"Unexpected error: {str(e)}",
                        "doc_count": doc_count
                    })
        
        # ⭐ TRẢ VỀ KẾT QUẢ THÔNG MINH
        logger.info(f"🎯 FINAL RESULT: {len(all_selected_recipes)} recipes, {len(error_reports)} errors")
        
        if not all_selected_recipes and error_reports:
            return json.dumps({
                "message": "Không tìm thấy recipe và có lỗi xảy ra",
                "errors": error_reports[:3]  # Chỉ report 3 lỗi đầu
            }, ensure_ascii=False, indent=2)
        elif not all_selected_recipes:
            return json.dumps([])
        elif error_reports:
            # Có recipes nhưng cũng có lỗi - chỉ trả về recipes
            logger.warning(f"⚠️ Có {len(error_reports)} lỗi nhưng vẫn tìm được {len(all_selected_recipes)} recipes")
            return json.dumps(all_selected_recipes, ensure_ascii=False, indent=2)
        else:
            return json.dumps(all_selected_recipes, ensure_ascii=False, indent=2)

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