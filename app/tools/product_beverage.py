import os
import logging
import asyncio
import json
import re
from pinecone import Pinecone
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_huggingface import HuggingFaceEmbeddings
import numpy as np

# ⭐ Import ApiKeyManager CHÍNH XÁC
from app.services.api_key_manager import get_api_key_manager
# ⭐ IMPORT GEMINI MODEL POOL để thread-safe API access
from app.tools.gemini_model_pool import get_gemini_model_pool

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Tải biến môi trường
load_dotenv()

# --- Cấu hình ---
PINECONE_API_KEY = os.getenv("PRODUCT_DB_PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "product-index")
EMBEDDING_MODEL_FOR_DIM_ONLY = "sentence-transformers/all-mpnet-base-v2"
# ⭐ TỐI ƯU HÓA MẠNH: Giảm batch size để tránh lỗi JSON và token limit
PRODUCTS_TO_FETCH_FROM_PINECONE = 1610
PRODUCTS_PER_GEMINI_BATCH = 60           # Giảm từ 100 xuống 60
MAX_GEMINI_OUTPUT_TOKENS = 4096          # Giảm từ 8192 xuống 4096

# ⭐ CẤU HÌNH ASYNC VÀ CONCURRENCY - Tăng để tận dụng tối đa API keys
MAX_CONCURRENT_GEMINI_BEVERAGE_CLASSIFICATION_CALLS = 7  # Tăng từ 6 lên 7 để match với số API keys

# ⭐ DYNAMIC BATCHING CONSTANTS - thêm để tương đồng với recipe_tool
MAX_CHAR_PER_BATCH = 150000              # Giới hạn ký tự cho mỗi batch
MAX_SAFE_BATCH_SIZE = 50                 # Fallback limit nếu dynamic batching fails

# ⭐ CONFIGURATION CHO PARALLEL PROCESSING
WORKER_SLEEP_BETWEEN_TASKS = 0.1  # Giảm để tăng throughput
WORKER_ERROR_SLEEP = 2  # Sleep khi có lỗi

# ⭐ MODEL NAME CONSTANT
GEMINI_MODEL_NAME = "gemini-2.0-flash-lite"

# ⭐ KHỞI TẠO GEMINI MODEL POOL
gemini_model_pool = get_gemini_model_pool(GEMINI_MODEL_NAME)

def get_embedding_model_for_dimension():
    """⭐ Lấy embedding model để xác định dimension vector"""
    try:
        # Import function để lấy global embedding model
        from main import get_global_embedding_model
        global_model = get_global_embedding_model()
        
        if global_model is not None:
            logger.info("✅ Sử dụng pre-loaded embedding model để xác định dimension")
            return global_model
        else:
            logger.warning("⚠️ Global embedding model chưa được load, tạo mới để xác định dimension...")
            return HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL_FOR_DIM_ONLY,
                model_kwargs={'device': 'cpu'}
            )
    except ImportError:
        logger.warning("⚠️ Không thể import global embedding model, tạo mới...")
        return HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_FOR_DIM_ONLY,
            model_kwargs={'device': 'cpu'}
        )

def init_services():
    """⭐ Khởi tạo kết nối Pinecone và lấy dimension của vector. KHÔNG trả về gemini_model."""
    try:
        # ⭐ KIỂM TRA API KEY MANAGER
        api_key_manager = get_api_key_manager()
        if not api_key_manager.is_healthy():
            logger.error("ApiKeyManager không khỏe mạnh hoặc không có API key")
            raise ValueError("ApiKeyManager không khỏe mạnh")
        logger.info(f"✅ ApiKeyManager đã được xác nhận khỏe mạnh với {api_key_manager.total_keys()} keys")
        
        pc = Pinecone(api_key=PINECONE_API_KEY)
        
        # Cải thiện logic kiểm tra index
        try:
            index_list = pc.list_indexes()
            index_names = []
            
            if hasattr(index_list, 'indexes') and isinstance(index_list.indexes, list):
                index_names = [idx.get('name') for idx in index_list.indexes if isinstance(idx, dict) and idx.get('name')]
            elif hasattr(index_list, 'names'):
                if callable(index_list.names):
                    index_names = index_list.names()
                else:
                    index_names = index_list.names
            
            if PINECONE_INDEX_NAME not in index_names:
                logger.error(f"Pinecone index '{PINECONE_INDEX_NAME}' không tồn tại trong danh sách: {index_names}")
                raise ValueError(f"Pinecone index '{PINECONE_INDEX_NAME}' không tồn tại.")
                
        except Exception as e_list:
            logger.warning(f"Không thể kiểm tra danh sách index: {e_list}. Thử describe_index trực tiếp...")

        index_description = pc.describe_index(PINECONE_INDEX_NAME)
        logger.info(f"Pinecone index '{PINECONE_INDEX_NAME}' đã tồn tại. Thông tin: {index_description}")
        pinecone_index = pc.Index(PINECONE_INDEX_NAME)
        vector_dimension = index_description.dimension
        
        if not vector_dimension:
            logger.info("Không lấy được dimension từ describe_index, thử tải embedding model...")
            try:
                # ⭐ SỬ DỤNG GLOBAL EMBEDDING MODEL
                temp_embeddings_model = get_embedding_model_for_dimension()
                sample_embedding = temp_embeddings_model.embed_query("sample text")
                vector_dimension = len(sample_embedding)
                logger.info(f"Dimension của vector xác định từ embedding model: {vector_dimension}")
            except Exception as e_embed:
                logger.error(f"Không thể tự động xác định dimension: {e_embed}")
                raise ValueError("Không thể xác định dimension của vector.") from e_embed
        else:
            logger.info(f"Dimension của vector từ Pinecone: {vector_dimension}")

        logger.info(f"🚀 Đã khởi tạo Pinecone index với vector dimension: {vector_dimension}")
        # ⭐ CHỈ TRẢ VỀ pinecone_index VÀ vector_dimension (KHÔNG TRẢ VỀ gemini_model)
        return pinecone_index, vector_dimension
        
    except Exception as e:
        logger.error(f"Lỗi khi khởi tạo dịch vụ: {str(e)}")
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

async def call_gemini_api_batch_classification_async(products_batch: list[dict], api_key_for_this_call: str, task_description: str, max_retries: int = 2) -> list[dict]:
    """
    ⭐ ASYNC VERSION với API KEY được truyền vào từ Worker Pool
    
    Args:
        products_batch: list of {"id": "product_id", "name": "product_name"}
        api_key_for_this_call: API key được truyền từ worker
        task_description: Mô tả task để logging
        max_retries: Số lần thử lại tối đa
        
    Returns: 
        list of {"product_id": "...", "product_name": "..."} cho các sản phẩm là đồ uống
        Hoặc [{"error_too_large": True}] nếu lô quá lớn
    """
    if not products_batch:
        return []

    if not api_key_for_this_call:
        logger.error("❌ Không có API key Gemini được truyền vào")
        return []

    # Xây dựng phần danh sách sản phẩm cho prompt
    product_list_str = ""
    for i, p_data in enumerate(products_batch):
        safe_id = json.dumps(p_data['id'], ensure_ascii=False)
        safe_name = json.dumps(p_data['name'], ensure_ascii=False)
        product_list_str += f'{i+1}. ID: {safe_id}, Tên: {safe_name}\n'

    # ⭐ PROMPT TỐI ƯU HÓA MẠNH CHO JSON - YÊU CẦU CỰC KỲ CHẶT CHẼ
    prompt = f"""NHIỆM VỤ: Phân loại CHỈ những sản phẩm thực sự là ĐỒ UỐNG hoặc NGUYÊN LIỆU PHA CHẾ.

🚨 QUY TẮC TUYỆT ĐỐI:
- CHỈ trả về JSON array hợp lệ
- KHÔNG thêm text, giải thích, markdown
- KHÔNG sử dụng dấu ngoặc kép thông minh (" ")
- CHỈ sử dụng dấu ngoặc kép ASCII chuẩn (")
- KHÔNG có trailing comma
- Nếu không có kết quả: []

📋 ĐỊNH DẠNG BẮT BUỘC:
[{{"product_id":"id","product_name":"tên"}}]

⭐ CHỈ CHỌN:
- Đồ uống sẵn sàng: nước, trà, cà phê, sữa, nước ép
- Nguyên liệu pha chế: bột cà phê, trà lá, siro, matcha

❌ TUYỆT ĐỐI KHÔNG CHỌN:
- Gia vị nấu ăn: nước tương, dấm, muối
- Thực phẩm khô: bánh, kẹo, snack
- Nguyên liệu nấu ăn: hành, tỏi, gia vị

✅ VÍ DỤ CHÍNH XÁC:
[{{"product_id":"123","product_name":"Trà xanh"}},{{"product_id":"456","product_name":"Cà phê đen"}}]

❌ TUYỆT ĐỐI KHÔNG:
- Không markdown: ```json
- Không text thêm: "Dưới đây là..."
- Không trailing comma: }},]
- Không smart quotes: "text"

DANH SÁCH SẢN PHẨM:
{product_list_str}

TRẢ VỀ JSON ARRAY:"""

    # Log key rotation (an toàn)
    masked_key = f"{api_key_for_this_call[:8]}..." if len(api_key_for_this_call) > 8 else "***"
    logger.info(f"🔍 {task_description}: Gọi Gemini với key {masked_key} - {len(products_batch)} sản phẩm")

    for attempt in range(max_retries):
        try:
            # ⭐ SỬ DỤNG MODEL POOL THAY VÌ genai.configure() (thread-safe)
            temp_gemini_model = gemini_model_pool.get_model_for_key(api_key_for_this_call)
            if not temp_gemini_model:
                logger.error(f"❌ Không tìm thấy model cho key {masked_key}")
                return []

            # ⭐ CHẠY GENERATE_CONTENT TRONG EXECUTOR
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: temp_gemini_model.generate_content(
                    prompt,
                    generation_config={"max_output_tokens": MAX_GEMINI_OUTPUT_TOKENS, "temperature": 0.0}
                )
            )

            if response.parts:
                response_text = response.text

                # ⭐ SỬ DỤNG PARSE_JSON_WITH_FALLBACK THAY VÌ MANUAL PARSING
                parse_result = parse_json_with_fallback(response_text)
                
                if parse_result["success"]:
                    classified_drinks = parse_result["data"]
                    
                    # ⭐ XỬ LÝ TRƯỜNG HỢP GEMINI TRẢ VỀ OBJECT THAY VÌ LIST
                    if isinstance(classified_drinks, dict):
                        logger.warning(f"⚠️ {task_description}: Gemini trả về object thay vì array")
                        # Thử extract array từ object
                        extracted_array = None
                        for key, value in classified_drinks.items():
                            if isinstance(value, list):
                                logger.info(f"✅ {task_description}: Extracted array từ key '{key}'")
                                extracted_array = value
                                break
                        
                        if extracted_array:
                            classified_drinks = extracted_array
                        else:
                            logger.warning(f"⚠️ {task_description}: Không tìm thấy array trong object, retry")
                            if attempt < max_retries - 1:
                                await asyncio.sleep(5 + attempt * 5)
                                continue
                            return []
                    
                    if not isinstance(classified_drinks, list):
                        logger.warning(f"⚠️ {task_description}: Response vẫn không phải list sau xử lý: {type(classified_drinks)}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(5 + attempt * 5)
                            continue
                        return []
                    
                    # Xác thực thêm: mỗi item trong list phải là dict có product_id và product_name
                    valid_drinks = []
                    original_ids = {p['id'] for p in products_batch}
                    seen_product_ids = set()
                    
                    for drink in classified_drinks:
                        if isinstance(drink, dict) and "product_id" in drink and "product_name" in drink:
                            product_id = drink["product_id"]
                            
                            if product_id in seen_product_ids:
                                logger.warning(f"Gemini trả về product_id '{product_id}' bị trùng lặp trong cùng response. Bỏ qua.")
                                continue
                            
                            if product_id in original_ids:
                                valid_drinks.append(drink)
                                seen_product_ids.add(product_id)
                                
                                # Break sớm nếu đã đủ số lượng sản phẩm đầu vào
                                if len(valid_drinks) >= len(products_batch):
                                    logger.info(f"Đã xử lý đủ {len(products_batch)} sản phẩm, dừng để tránh hallucination")
                                    break
                            else:
                                logger.warning(f"Gemini trả về product_id '{product_id}' không có trong lô sản phẩm gốc. Bỏ qua.")
                        else:
                            logger.warning(f"Mục không hợp lệ trong danh sách JSON từ Gemini: {drink}")
                    
                    if len(classified_drinks) > len(products_batch):
                        logger.warning(f"Gemini trả về {len(classified_drinks)} items nhưng chỉ có {len(products_batch)} sản phẩm trong lô. "
                                       f"Có thể có hallucination.")
                    
                    # Lọc lần cuối để đảm bảo không có duplicate
                    final_unique_drinks = []
                    final_seen_ids = set()
                    for drink_item in valid_drinks:
                        if drink_item['product_id'] not in final_seen_ids:
                            final_unique_drinks.append(drink_item)
                            final_seen_ids.add(drink_item['product_id'])
                    
                    logger.info(f"✅ {task_description}: Thành công phân loại được {len(final_unique_drinks)} đồ uống từ {len(products_batch)} sản phẩm")
                    return final_unique_drinks
                    
                else:
                    # Parse failed với structured error
                    logger.error(f"💥 {task_description} JSON parse failed: {parse_result.get('error')}")
                    if attempt < max_retries - 1:
                        logger.info(f"🔄 Retry {task_description} attempt {attempt + 1}/{max_retries} due to JSON parse error")
                        await asyncio.sleep(10 + attempt * 10)
                        continue
                    return []

        except Exception as e:
            error_str = str(e).lower()
            if "token" in error_str or "size limit" in error_str or "request payload" in error_str or "too large" in error_str:
                logger.error(f"💥 Lỗi kích thước prompt/request khi gọi Gemini ({task_description}): {str(e)}. "
                             f"Lô hiện tại có {len(products_batch)} sản phẩm. Hãy thử giảm PRODUCTS_PER_GEMINI_BATCH.")
                logger.warning(f"⚠️ Cần giảm PRODUCTS_PER_GEMINI_BATCH hiện tại ({PRODUCTS_PER_GEMINI_BATCH}) để tránh lỗi này")
                return [{"error_too_large": True, "batch_size": len(products_batch), "suggestion": "Giảm PRODUCTS_PER_GEMINI_BATCH"}]

            if "429" in error_str or "resource_exhausted" in error_str or "too many requests" in error_str or "quota" in error_str:
                retry_delay = 15 + attempt * 10
                logger.warning(f"⚠️ Lỗi quota/rate limit Gemini ({task_description}), thử lại sau {retry_delay} giây...")
                await asyncio.sleep(retry_delay)
            elif "500" in error_str or "internal server error" in error_str or "service temporarily unavailable" in error_str:
                retry_delay = 20 + attempt * 10
                logger.warning(f"⚠️ Lỗi server Gemini (5xx) ({task_description}), thử lại sau {retry_delay} giây...")
                await asyncio.sleep(retry_delay)
            else:
                logger.error(f"💥 Lỗi không xác định khi gọi API Gemini ({task_description}) (Lần {attempt+1}/{max_retries}): {str(e)}")
                if attempt == max_retries - 1:
                    break
                await asyncio.sleep(15 + attempt * 5)

    logger.error(f"❌ Vẫn gặp lỗi sau {max_retries} lần thử lại với Gemini ({task_description}) hoặc không thể parse kết quả.")
    return []

async def fetch_and_filter_drinks_in_batches_async(pinecone_index, vector_dimension: int) -> list[dict]:
    """
    ⭐ ASYNC VERSION: Tìm kiếm và phân loại đồ uống từ Pinecone với xử lý bất đồng bộ.
    
    Args:
        pinecone_index: Pinecone index instance
        vector_dimension: Dimension của vector trong Pinecone
        
    Returns:
        list[dict]: Danh sách các đồ uống đã phân loại
    """
    identified_drinks_overall = []
    query_vector = np.random.rand(vector_dimension).tolist()

    logger.info(f"🔍 Bắt đầu quét sản phẩm từ Pinecone (tối đa {PRODUCTS_TO_FETCH_FROM_PINECONE} sản phẩm)...")
    
    try:
        # ⭐ CHẠY PINECONE QUERY TRONG EXECUTOR
        loop = asyncio.get_event_loop()
        query_response = await loop.run_in_executor(
            None,
            lambda: pinecone_index.query(
                vector=query_vector,
                top_k=PRODUCTS_TO_FETCH_FROM_PINECONE,
                include_metadata=True
            )
        )
        
        products_from_pinecone = []
        if query_response.matches:
            for match in query_response.matches:
                prod_id = match.metadata.get("doc_id", match.id) if match.metadata else match.id
                prod_name = match.metadata.get("name") if match.metadata else None
                if prod_id and prod_name:
                    products_from_pinecone.append({"id": prod_id, "name": prod_name})
                else:
                    logger.warning(f"Sản phẩm từ Pinecone (ID vector: {match.id}) thiếu 'name' hoặc 'doc_id'. Bỏ qua.")
        
        if not products_from_pinecone:
            logger.info("❌ Không có sản phẩm nào được tìm thấy từ Pinecone.")
            return []

        logger.info(f"📦 Lấy được {len(products_from_pinecone)} sản phẩm từ Pinecone.")
        
        # ⭐ KHỞI TẠO WORKER POOL CONFIGURATION
        api_key_manager = get_api_key_manager()
        NUM_GEMINI_WORKERS = min(api_key_manager.total_keys(), MAX_CONCURRENT_GEMINI_BEVERAGE_CLASSIFICATION_CALLS)
        if NUM_GEMINI_WORKERS == 0: 
            NUM_GEMINI_WORKERS = 1  # Ít nhất 1 worker nếu có key
        
        logger.info(f"🤖 Khởi tạo Worker Pool với {NUM_GEMINI_WORKERS} workers (có {api_key_manager.total_keys()} API keys)")
        
        # Tính toán số batches sẽ được tạo
        total_batches = (len(products_from_pinecone) + PRODUCTS_PER_GEMINI_BATCH - 1) // PRODUCTS_PER_GEMINI_BATCH
        logger.info(f"📊 Sẽ xử lý {total_batches} batches với worker pool parallelism")
        
        # ⭐ KHỞI TẠO QUEUE VÀ RESULT STORAGE
        work_queue = asyncio.Queue()
        results_list = []  # Thread-safe với coroutine
        error_reports_list = []
        
        # ⭐ CHIA THÀNH CÁC LÔ VÀ ĐƯA VÀO QUEUE
        current_batch_size = PRODUCTS_PER_GEMINI_BATCH
        
        for i in range(0, len(products_from_pinecone), current_batch_size):
            batch_to_send = products_from_pinecone[i : i + current_batch_size]
            if batch_to_send:
                batch_num = i // current_batch_size + 1
                task_data = {
                    "products_batch": batch_to_send,
                    "batch_num": batch_num,
                    "batch_size": len(batch_to_send)
                }
                await work_queue.put(task_data)

        # ⭐ HÀM GEMINI WORKER
        async def gemini_worker(worker_id: int):
            logger.info(f"🤖 Worker {worker_id}: Bắt đầu hoạt động.")
            while True:
                try:
                    task_data = await work_queue.get()
                    if task_data is None:  # Tín hiệu dừng
                        work_queue.task_done()
                        logger.info(f"🤖 Worker {worker_id}: Nhận tín hiệu dừng.")
                        break

                    products_batch = task_data["products_batch"]
                    batch_num_log = task_data["batch_num"]
                    batch_size_log = task_data["batch_size"]
                    
                    task_description = f"Worker {worker_id} - Batch {batch_num_log}"
                    
                    # Mỗi worker tự lấy key mới cho mỗi task nó xử lý
                    api_key_for_call = api_key_manager.get_next_key()
                    if not api_key_for_call:
                        logger.error(f"🤖 Worker {worker_id}: Không có API key, bỏ qua batch {batch_num_log}")
                        error_reports_list.append({
                            "error_batch": batch_num_log, "error_type": "no_api_key",
                            "message": "Không có API key khả dụng", "batch_size": batch_size_log
                        })
                        work_queue.task_done()
                        continue

                    logger.info(f"🤖 Worker {worker_id}: Đang xử lý Batch {batch_num_log} ({batch_size_log} sản phẩm) với key ...{api_key_for_call[-4:]}")
                    
                    start_time = asyncio.get_event_loop().time()
                    
                    # Gọi API Gemini
                    classified_drinks = await call_gemini_api_batch_classification_async(
                        products_batch, 
                        api_key_for_call,
                        task_description
                    )
                    
                    end_time = asyncio.get_event_loop().time()
                    elapsed = end_time - start_time
                    
                    # ⭐ XỬ LÝ TRƯỜNG HỢP PROMPT QUÁ LỚN VỚI LOGGING RÕ RÀNG HỢN
                    if classified_drinks and isinstance(classified_drinks[0], dict) and classified_drinks[0].get("error_too_large"):
                        error_info = classified_drinks[0]
                        logger.error(f"💥 Batch {batch_num_log}: Lô quá lớn cho Gemini API")
                        logger.error(f"   - Batch size hiện tại: {batch_size_log} sản phẩm")
                        logger.error(f"   - PRODUCTS_PER_GEMINI_BATCH hiện tại: {PRODUCTS_PER_GEMINI_BATCH}")
                        logger.warning(f"⚠️ Cần điều chỉnh cấu hình để tránh lỗi này tiếp tục xảy ra")
                        error_reports_list.append({
                            "error_batch": batch_num_log, "error_type": "batch_too_large",
                            "message": "Batch quá lớn cho Gemini", "batch_size": batch_size_log
                        })
                    elif classified_drinks:
                        results_list.extend(classified_drinks)
                        logger.info(f"✅ Worker {worker_id}: Batch {batch_num_log} xác định được {len(classified_drinks)} đồ uống trong {elapsed:.2f}s")
                    else:
                        logger.info(f"⚪ Worker {worker_id}: Batch {batch_num_log} không có đồ uống hoặc có lỗi ({elapsed:.2f}s)")
                        error_reports_list.append({
                            "error_batch": batch_num_log, "error_type": "empty_result",
                            "message": "Không có kết quả", "batch_size": batch_size_log
                        })
                    
                    work_queue.task_done()
                    # Thời gian nghỉ nhỏ sau mỗi batch của một worker
                    await asyncio.sleep(WORKER_SLEEP_BETWEEN_TASKS)

                except asyncio.CancelledError:
                    logger.info(f"🤖 Worker {worker_id}: Bị cancel.")
                    break
                except Exception as e:
                    logger.error(f"🤖 Worker {worker_id}: Lỗi không mong muốn: {e}", exc_info=True)
                    if 'task_data' in locals() and task_data and 'batch_num' in task_data:
                         error_reports_list.append({"error_batch": task_data['batch_num'], "error_type": "worker_exception", "message": str(e), "batch_size": task_data.get('batch_size',0)})
                    if 'task_data' in locals() and task_data is not None:
                         work_queue.task_done()
                    await asyncio.sleep(WORKER_ERROR_SLEEP)

        # ⭐ KHỞI TẠO VÀ CHẠY CÁC WORKER
        worker_tasks = []
        for i in range(NUM_GEMINI_WORKERS):
            worker_tasks.append(asyncio.create_task(gemini_worker(i + 1)))

        # Chờ tất cả các item trong queue được xử lý
        await work_queue.join()

        # Gửi tín hiệu dừng cho tất cả worker
        for _ in range(NUM_GEMINI_WORKERS):
            await work_queue.put(None)

        # Chờ tất cả worker hoàn thành
        await asyncio.gather(*worker_tasks, return_exceptions=True)

        logger.info(f"🏁 Tất cả các worker đã hoàn thành. Tổng hợp kết quả...")
        
        # ⭐ LỌC TRÙNG LẶP BẰNG TÊN CHUẨN HÓA
        def normalize_product_name(name: str) -> str:
            """Chuẩn hóa tên sản phẩm để so sánh trùng lặp"""
            if not name:
                return ""
            # Chuyển về lowercase, loại bỏ dấu cách, dấu gạch ngang, ký tự đặc biệt
            import unicodedata
            normalized = unicodedata.normalize('NFD', str(name).lower())
            normalized = ''.join(c for c in normalized if unicodedata.category(c) != 'Mn')  # Loại bỏ dấu
            normalized = re.sub(r'[^a-z0-9]', '', normalized)  # Chỉ giữ chữ và số
            return normalized

        if results_list:
            logger.info(f"🔄 Bắt đầu lọc trùng lặp từ {len(results_list)} beverages...")
            
            final_unique_beverages = []
            seen_normalized_names = set()
            seen_product_ids = set()
            
            for beverage_item in results_list:
                if not isinstance(beverage_item, dict):
                    continue
                    
                product_id = beverage_item.get("product_id")
                product_name = beverage_item.get("product_name")
                
                if not product_id or not product_name:
                    continue
                
                # Lọc theo ID trước (ưu tiên cao nhất)
                if product_id in seen_product_ids:
                    logger.debug(f"Đã lọc beverage trùng ID: {product_id}")
                    continue
                
                # Lọc theo tên chuẩn hóa
                normalized_name = normalize_product_name(product_name)
                if normalized_name and normalized_name not in seen_normalized_names:
                    final_unique_beverages.append(beverage_item)
                    seen_normalized_names.add(normalized_name)
                    seen_product_ids.add(product_id)
                else:
                    logger.debug(f"Đã lọc beverage trùng tên: {product_name}")
            
            results_list = final_unique_beverages
            logger.info(f"✅ Sau khi lọc trùng lặp: {len(results_list)} beverages duy nhất")

        # ⭐ XỬ LÝ KẾT QUẢ CUỐI CÙNG
        identified_drinks_overall.extend(results_list)
        
        if error_reports_list:
            logger.warning(f"⚠️ Có {len(error_reports_list)} batch lỗi nhưng vẫn tìm được {len(results_list)} đồ uống")
            # Log một số lỗi đầu
            for error in error_reports_list[:3]:
                logger.warning(f"   - Batch {error['error_batch']}: {error['message']}")
        
        logger.info(f"📊 Kết quả: {len(results_list)} đồ uống được tìm thấy, {len(error_reports_list)} batch lỗi")
        
    except Exception as e:
        logger.error(f"💥 Lỗi khi query Pinecone hoặc xử lý lô sản phẩm: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

    return identified_drinks_overall

# ⭐ WRAPPER ĐỒNG BỘ CHO BACKWARD COMPATIBILITY
def fetch_and_filter_drinks_in_batches_sync_wrapper() -> list[dict]:
    """
    ⭐ WRAPPER ĐỒNG BỘ: Để chat_flow.py có thể gọi từ synchronous context.
    """
    try:
        # Khởi tạo các dịch vụ cần thiết
        pinecone_index, vector_dimension = init_services()
        
        # Kiểm tra xem có event loop đang chạy không
        try:
            loop = asyncio.get_running_loop()
            # Nếu có event loop đang chạy, tạo task mới
            logger.info("💡 Phát hiện event loop đang chạy, sử dụng create_task")
            task = loop.create_task(fetch_and_filter_drinks_in_batches_async(pinecone_index, vector_dimension))
            # Wait cho task hoàn thành (blocking)
            import concurrent.futures
            import threading
            
            def run_in_new_loop():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(
                        fetch_and_filter_drinks_in_batches_async(pinecone_index, vector_dimension)
                    )
                finally:
                    new_loop.close()
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_new_loop)
                return future.result()
                
        except RuntimeError:
            # Không có event loop đang chạy, có thể dùng asyncio.run
            logger.info("💡 Không có event loop đang chạy, sử dụng asyncio.run")
            return asyncio.run(fetch_and_filter_drinks_in_batches_async(pinecone_index, vector_dimension))
            
    except Exception as e:
        logger.error(f"💥 Lỗi trong wrapper đồng bộ của fetch_and_filter_drinks: {e}", exc_info=True)
        return []

# ⭐ GIỮ LẠI HÀM CŨ ĐỂ TƯƠNG THÍCH NGƯỢC (Deprecated)
def fetch_and_filter_drinks_in_batches(pinecone_index, gemini_model, vector_dimension: int) -> list[dict]:
    """Hàm cũ để tương thích ngược - DEPRECATED: sử dụng async version"""
    logger.warning("⚠️ Sử dụng hàm fetch_and_filter_drinks_in_batches CŨ - khuyến nghị sử dụng async version")
    return asyncio.run(fetch_and_filter_drinks_in_batches_async(pinecone_index, vector_dimension))

if __name__ == "__main__":
    async def main_beverage_test_async():
        """⭐ HÀM TEST ASYNC CHÍNH"""
        try:
            print("🔧 Đang khởi tạo các dịch vụ...")
            pinecone_index, vector_dim = init_services()
            
            print(f"\n🚀 Bắt đầu quá trình tìm kiếm đồ uống từ Pinecone index '{PINECONE_INDEX_NAME}' (ASYNC VERSION)...")
            print(f"📊 Cấu hình:")
            print(f"   - Sản phẩm tối đa từ Pinecone: {PRODUCTS_TO_FETCH_FROM_PINECONE}")
            print(f"   - Kích thước mỗi batch: {PRODUCTS_PER_GEMINI_BATCH}")
            print(f"   - Đồng thời tối đa: {MAX_CONCURRENT_GEMINI_BEVERAGE_CLASSIFICATION_CALLS} batch")
            print(f"   - Vector dimension: {vector_dim}")
            print(f"   - Gemini model: {GEMINI_MODEL_NAME}")
            
            # Lấy thống kê API key
            api_key_manager = get_api_key_manager()
            print(f"   - API keys available: {api_key_manager.total_keys()}")
            
            import time
            start_time = time.time()
            all_drinks_found = await fetch_and_filter_drinks_in_batches_async(pinecone_index, vector_dim)
            end_time = time.time()
            
            print("\n📋 --- DANH SÁCH ĐỒ UỐNG ĐƯỢC TÌM THẤY ---")
            if all_drinks_found:
                # Loại bỏ trùng lặp nếu có
                unique_drinks = []
                seen_ids = set()
                for drink in all_drinks_found:
                    if drink['product_id'] not in seen_ids:
                        unique_drinks.append(drink)
                        seen_ids.add(drink['product_id'])
                
                print(json.dumps(unique_drinks[:10], indent=2, ensure_ascii=False))  # Chỉ hiển thị 10 đầu tiên
                if len(unique_drinks) > 10:
                    print(f"... và {len(unique_drinks) - 10} đồ uống khác")
                
                print(f"\n✅ Tổng cộng tìm thấy {len(unique_drinks)} sản phẩm đồ uống duy nhất.")
                
                # Thống kê hiệu suất
                total_processing_time = end_time - start_time
                products_per_second = len(unique_drinks) / total_processing_time if total_processing_time > 0 else 0
                print(f"⚡ Hiệu suất: {products_per_second:.2f} đồ uống/giây")
                
            else:
                print("❌ Không tìm thấy sản phẩm đồ uống nào hoặc không có sản phẩm nào được phân loại là đồ uống.")
                
            print(f"⏱️ Tổng thời gian thực hiện: {end_time - start_time:.2f} giây.")
            
            # Hiển thị thống kê sử dụng API key
            usage_stats = api_key_manager.get_usage_statistics()
            print(f"\n📊 Thống kê sử dụng API key:")
            for key_masked, usage_count in usage_stats.items():
                print(f"   - Key {key_masked}: {usage_count} lần sử dụng")

        except ValueError as ve: 
            print(f"❌ Lỗi cấu hình hoặc giá trị: {ve}")
        except Exception as e:
            logger.error(f"❌ Lỗi không mong muốn ở main: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())

    # ⭐ SỬ DỤNG ASYNCIO.RUN CHO MAIN
    asyncio.run(main_beverage_test_async())