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

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Tải biến môi trường
load_dotenv()

# --- Cấu hình ---
PINECONE_API_KEY = os.getenv("PRODUCT_DB_PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "product-index")
EMBEDDING_MODEL_FOR_DIM_ONLY = "sentence-transformers/all-mpnet-base-v2"
PRODUCTS_TO_FETCH_FROM_PINECONE = 1610
PRODUCTS_PER_GEMINI_BATCH = 100
MAX_GEMINI_OUTPUT_TOKENS = 8192

# ⭐ CẤU HÌNH ASYNC VÀ CONCURRENCY - Đúng tên theo yêu cầu
MAX_CONCURRENT_GEMINI_BEVERAGE_CLASSIFICATION_CALLS = 3

# ⭐ MODEL NAME CONSTANT
GEMINI_MODEL_NAME = "gemini-2.0-flash-lite"

# Kiểm tra API keys cơ bản
if not PINECONE_API_KEY or not PINECONE_INDEX_NAME:
    missing_keys = [
        key for key, value in {
            "PINECONE_API_KEY": PINECONE_API_KEY,
            "PINECONE_INDEX_NAME": PINECONE_INDEX_NAME
        }.items() if not value
    ]
    logger.error(f"Các biến môi trường sau không được thiết lập: {', '.join(missing_keys)}")
    raise ValueError(f"Các biến môi trường sau không được thiết lập: {', '.join(missing_keys)}")

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
                temp_embeddings_model = HuggingFaceEmbeddings(
                    model_name=EMBEDDING_MODEL_FOR_DIM_ONLY, model_kwargs={'device': 'cpu'}
                )
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

async def call_gemini_api_batch_classification_async(products_batch: list[dict], max_retries: int = 2) -> list[dict]:
    """
    ⭐ ASYNC VERSION: Gửi một lô sản phẩm đến Gemini để phân loại và trả về danh sách đồ uống.
    
    Args:
        products_batch: list of {"id": "product_id", "name": "product_name"}
        max_retries: Số lần thử lại tối đa
        
    Returns: 
        list of {"product_id": "...", "product_name": "..."} cho các sản phẩm là đồ uống
        Hoặc [{"error_too_large": True}] nếu lô quá lớn
    """
    if not products_batch:
        return []

    # ⭐ LẤY API KEY MANAGER
    api_key_manager = get_api_key_manager()
    if not api_key_manager.is_healthy():
        logger.error("ApiKeyManager không khỏe mạnh, không thể gọi Gemini API")
        return []

    # Xây dựng phần danh sách sản phẩm cho prompt
    product_list_str = ""
    for i, p_data in enumerate(products_batch):
        safe_id = json.dumps(p_data['id'], ensure_ascii=False)
        safe_name = json.dumps(p_data['name'], ensure_ascii=False)
        product_list_str += f'{i+1}. ID: {safe_id}, Tên: {safe_name}\n'

    prompt = f"""
Bạn được cung cấp một danh sách các sản phẩm. Hãy phân loại CHỈ những sản phẩm thực sự là ĐỒ UỐNG hoặc NGUYÊN LIỆU CHÍNH để pha chế đồ uống.

⭐ CHỈ CHỌN CÁC SẢN PHẨM SAU:
- ĐỒ UỐNG SẴN SÀNG: nước giải khát, trà, cà phê pha sẵn, sữa, nước ép, sinh tố, bia, rượu vang, nước lọc
- NGUYÊN LIỆU PHA CHẾ CHÍNH: bột cà phê, trà túi lọc, trà lá, siro pha chế, sữa đặc, bột cacao, matcha, coffee bean

⭐ TUYỆT ĐỐI KHÔNG CHỌN:
- Gia vị nấu ăn: nước màu dừa, nước tương, dấm, muối
- Thực phẩm khô: bánh kẹo, snack, mì tôm
- Đường phèn (trừ khi có bối cảnh "Trà đường phèn" hoặc đồ uống cụ thể)
- Nguyên liệu nấu ăn khác: hành tây, tỏi, gia vị

Danh sách sản phẩm:
{product_list_str}

⚠️ YÊU CẦU QUAN TRỌNG:
1. Mỗi product_id trong danh sách JSON trả về phải là DUY NHẤT - KHÔNG lặp lại sản phẩm
2. Chỉ trả về JSON array format chính xác
3. Mỗi object phải có đúng 2 trường: "product_id" và "product_name"
4. ID và name phải khớp chính xác với input

Ví dụ format JSON (nếu sản phẩm 1 và 3 là đồ uống):
[
  {{
    "product_id": "ID_SAN_PHAM_1",
    "product_name": "TEN_SAN_PHAM_1"
  }},
  {{
    "product_id": "ID_SAN_PHAM_3",
    "product_name": "TEN_SAN_PHAM_3"
  }}
]

Nếu không có đồ uống nào: []
CHỈ TRẢ VỀ JSON ARRAY - KHÔNG GIẢI THÍCH THÊM.
"""
    task_desc = f"Phân loại lô {len(products_batch)} sản phẩm (async)"

    for attempt in range(max_retries):
        try:
            # ⭐ LẤY API KEY TỪ API_KEY_MANAGER
            api_key = api_key_manager.get_next_key()
            if not api_key:
                logger.error("Không thể lấy API key từ api_key_manager")
                return []

            # ⭐ LOG API KEY USAGE (AN TOÀN)
            key_info = f"...{api_key[-6:]}" if len(api_key) > 6 else "short_key"
            logger.debug(f"🔑 Đang sử dụng API key kết thúc bằng {key_info} cho {task_desc}")

            # ⭐ CẤU HÌNH GEMINI VỚI API KEY MỚI CHO MỖI CALL
            genai.configure(api_key=api_key)
            temp_gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)

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

                # Cố gắng trích xuất JSON từ phản hồi
                match_json = re.search(r'```json\s*([\s\S]*?)\s*```', response_text, re.DOTALL)
                if match_json:
                    json_str = match_json.group(1).strip()
                else:
                    json_str = response_text.strip()
                    if not (json_str.startswith('[') and json_str.endswith(']')):
                         match_obj_json = re.search(r'\{\s*[\s\S]*?\s*\}', response_text, re.DOTALL)
                         if match_obj_json and not (json_str.startswith('[') and json_str.endswith(']')):
                              logger.warning(f"Gemini trả về JSON object thay vì list cho lô. Response: {response_text}")
                              if attempt < max_retries - 1:
                                  logger.info(f"Thử lại ({attempt+1}/{max_retries}) do định dạng không mong đợi.")
                                  await asyncio.sleep(5 + attempt * 5)
                                  continue
                              return []

                try:
                    classified_drinks = json.loads(json_str)
                    if not isinstance(classified_drinks, list):
                        logger.warning(f"Gemini không trả về một danh sách JSON. Response: {json_str}. Thử lại...")
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
                    
                    logger.info(f"✅ {task_desc}: Thành công phân loại được {len(final_unique_drinks)} đồ uống từ {len(products_batch)} sản phẩm")
                    return final_unique_drinks
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Lỗi giải mã JSON từ Gemini ({task_desc}): {e}. Response: {json_str}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(10 + attempt * 10)
            else:
                logger.warning(f"Gemini không trả về nội dung ({task_desc}). Response: {response}")
                if response.prompt_feedback and response.prompt_feedback.block_reason:
                    logger.warning(f"Bị chặn bởi ({task_desc}): {response.prompt_feedback.block_reason_message}")
                    if "SAFETY" in str(response.prompt_feedback.block_reason).upper():
                        logger.error("Prompt bị chặn do an toàn. Sẽ không thử lại với prompt này.")
                        return []
                if attempt < max_retries - 1:
                     await asyncio.sleep(10 + attempt * 5)

        except Exception as e:
            error_str = str(e).lower()
            if "token" in error_str or "size limit" in error_str or "request payload" in error_str or "too large" in error_str:
                logger.error(f"💥 Lỗi kích thước prompt/request khi gọi Gemini ({task_desc}): {str(e)}. "
                             f"Lô hiện tại có {len(products_batch)} sản phẩm. Hãy thử giảm PRODUCTS_PER_GEMINI_BATCH.")
                return [{"error_too_large": True}]

            if "429" in error_str or "resource_exhausted" in error_str or "too many requests" in error_str or "quota" in error_str:
                retry_delay = 15 + attempt * 10
                logger.warning(f"⚠️ Lỗi quota/rate limit Gemini ({task_desc}), thử lại sau {retry_delay} giây...")
                await asyncio.sleep(retry_delay)
            elif "500" in error_str or "internal server error" in error_str or "service temporarily unavailable" in error_str:
                retry_delay = 20 + attempt * 10
                logger.warning(f"⚠️ Lỗi server Gemini (5xx) ({task_desc}), thử lại sau {retry_delay} giây...")
                await asyncio.sleep(retry_delay)
            else:
                logger.error(f"💥 Lỗi không xác định khi gọi API Gemini ({task_desc}) (Lần {attempt+1}/{max_retries}): {str(e)}")
                if attempt == max_retries - 1:
                    break
                await asyncio.sleep(15 + attempt * 5)

    logger.error(f"❌ Vẫn gặp lỗi sau {max_retries} lần thử lại với Gemini ({task_desc}) hoặc không thể parse kết quả.")
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
        
        # ⭐ CHIA THÀNH CÁC LÔ VÀ XỬ LÝ BẰNG ASYNCIO.GATHER VỚI SEMAPHORE
        current_batch_size = PRODUCTS_PER_GEMINI_BATCH
        batch_tasks = []
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_GEMINI_BEVERAGE_CLASSIFICATION_CALLS)
        
        async def process_beverage_classification_batch_with_semaphore(products_in_batch, batch_num_log):
            """⭐ Hàm xử lý một batch với semaphore control"""
            async with semaphore:
                logger.info(f"🥤 Batch {batch_num_log}: Đang xử lý {len(products_in_batch)} sản phẩm với Gemini...")
                start_time = asyncio.get_event_loop().time()
                
                classified_drinks = await call_gemini_api_batch_classification_async(products_in_batch)
                
                end_time = asyncio.get_event_loop().time()
                elapsed = end_time - start_time
                
                # Xử lý trường hợp prompt quá lớn
                if classified_drinks and isinstance(classified_drinks[0], dict) and classified_drinks[0].get("error_too_large"):
                    logger.warning(f"⚠️ Batch {batch_num_log}: Lô quá lớn cho Gemini (kích thước: {current_batch_size}). Bỏ qua lô này.")
                    return []
                
                if classified_drinks:
                    logger.info(f"✅ Batch {batch_num_log}: Xác định được {len(classified_drinks)} đồ uống trong {elapsed:.2f}s")
                else:
                    logger.info(f"⚪ Batch {batch_num_log}: Không có đồ uống nào hoặc có lỗi ({elapsed:.2f}s)")
                
                return classified_drinks
        
        # ⭐ TẠO TASKS CHO TẤT CẢ CÁC BATCH
        for i in range(0, len(products_from_pinecone), current_batch_size):
            batch_to_send = products_from_pinecone[i : i + current_batch_size]
            if batch_to_send:
                batch_num = i // current_batch_size + 1
                task = process_beverage_classification_batch_with_semaphore(batch_to_send, batch_num)
                batch_tasks.append(task)
        
        # ⭐ CHẠY TẤT CẢ BATCH ĐỒNG THỜI VỚI ASYNCIO.GATHER
        if batch_tasks:
            total_batches = len(batch_tasks)
            logger.info(f"🚀 Đang xử lý {total_batches} batch đồng thời với giới hạn {MAX_CONCURRENT_GEMINI_BEVERAGE_CLASSIFICATION_CALLS} batch cùng lúc...")
            
            start_gather_time = asyncio.get_event_loop().time()
            results_from_gather = await asyncio.gather(*batch_tasks, return_exceptions=True)
            end_gather_time = asyncio.get_event_loop().time()
            
            logger.info(f"⚡ Hoàn thành tất cả {total_batches} batch trong {end_gather_time - start_gather_time:.2f}s")
            
            # ⭐ XỬ LÝ KẾT QUẢ TỪ GATHER
            successful_batches = 0
            failed_batches = 0
            for i, result in enumerate(results_from_gather):
                if isinstance(result, Exception):
                    logger.error(f"❌ Lỗi trong batch {i+1}: {str(result)}")
                    failed_batches += 1
                elif isinstance(result, list):
                    identified_drinks_overall.extend(result)
                    successful_batches += 1
                else:
                    logger.warning(f"⚠️ Kết quả không mong đợi từ batch {i+1}: {type(result)}")
                    failed_batches += 1
            
            logger.info(f"📊 Kết quả: {successful_batches} batch thành công, {failed_batches} batch thất bại")
        
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