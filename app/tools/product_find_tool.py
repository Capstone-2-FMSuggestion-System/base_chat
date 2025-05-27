import os
import sys
import logging
import asyncio
import json
import re
from dotenv import load_dotenv
import google.generativeai as genai
from typing import Optional, Tuple, List, Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from app.services.api_key_manager import get_api_key_manager
from app.services.cache_service import CacheService

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Tải biến môi trường
load_dotenv()

# --- Cấu hình ---
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite")
PRODUCTS_PER_GEMINI_BATCH = 200  # Giảm để tránh rate limit, vẫn hiệu quả hơn 80
MAX_CONCURRENT_GEMINI_PRODUCT_CALLS = min(get_api_key_manager().total_keys(), 3)  # Giảm để tránh quota overwhelm
REQUEST_DELAY = 0.5  # Delay 500ms giữa các requests để tránh rate limit

api_key_manager = get_api_key_manager()

# Kiểm tra API keys
if not api_key_manager.is_healthy():
    logger.error("ApiKeyManager không khả dụng. Kiểm tra cấu hình API keys trong file .env")
    raise ValueError("ApiKeyManager không khả dụng. Kiểm tra cấu hình API keys trong file .env")

logger.info(f"✅ Product Find Tool (Cache Version) đã khởi tạo với {len(api_key_manager.get_all_keys())} API keys có sẵn")

# Các functions Pinecone không còn cần thiết nữa

def deduplicate_products_cache(products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Lọc bỏ sản phẩm trùng lặp theo product_id và tên tương tự
    """
    if not products:
        return []
    
    # Step 1: Loại bỏ trùng product_id
    seen_ids = set()
    unique_by_id = []
    for product in products:
        product_id = str(product.get("product_id", ""))
        if product_id and product_id not in seen_ids:
            seen_ids.add(product_id)
            unique_by_id.append(product)
    
    # Step 2: Loại bỏ trùng tên gần giống (fuzzy matching)
    import difflib
    final_products = []
    seen_names = []
    
    for product in unique_by_id:
        name = product.get("name", "").lower().strip()
        if not name:
            continue
            
        # Kiểm tra similarity với các tên đã có
        is_duplicate = False
        for existing_name in seen_names:
            similarity = difflib.SequenceMatcher(None, name, existing_name).ratio()
            if similarity >= 0.85:  # 85% giống nhau = duplicate
                is_duplicate = True
                break
        
        if not is_duplicate:
            seen_names.append(name)
            final_products.append(product)
    
    logger.info(f"DEDUP: {len(products)} → {len(final_products)} sản phẩm (loại bỏ {len(products) - len(final_products)} trùng lặp)")
    return final_products

async def ensure_products_cache_available() -> List[Dict[str, Any]]:
    """
    Đảm bảo cache sản phẩm có sẵn, nếu chưa có thì tạo ngay lập tức.
    Được gọi tự động khi người dùng bắt đầu chat và cần tìm sản phẩm.
    """
    logger.info("PRODUCT_FIND_TOOL: Kiểm tra cache sản phẩm...")
    
    # Thử lấy từ cache trước
    cached_products = CacheService.get_all_products_list()
    if cached_products and len(cached_products) > 0:
        logger.info(f"PRODUCT_FIND_TOOL: Cache sản phẩm đã có sẵn với {len(cached_products)} sản phẩm")
        return deduplicate_products_cache(cached_products)
    
    logger.info("PRODUCT_FIND_TOOL: Cache sản phẩm chưa có, đang tạo ngay lập tức...")
    
    # Import background task function
    try:
        from app.services.background_products_cache import refresh_products_cache_task
        
        # Chạy task refresh cache ngay lập tức (đồng bộ)
        cache_result = await refresh_products_cache_task()
        
        if cache_result.get("success"):
            logger.info(f"PRODUCT_FIND_TOOL: Đã tạo cache thành công với {cache_result.get('products_count', 0)} sản phẩm")
            # Lấy lại từ cache sau khi đã tạo và dedup
            raw_products = CacheService.get_all_products_list() or []
            return deduplicate_products_cache(raw_products)
        else:
            logger.error(f"PRODUCT_FIND_TOOL: Tạo cache thất bại: {cache_result.get('error')}")
            return []
            
    except Exception as e:
        logger.error(f"PRODUCT_FIND_TOOL: Lỗi khi tạo cache sản phẩm: {e}", exc_info=True)
        return []


async def call_gemini_api_generic_async(prompt: str, task_description: str, max_retries: int = 3) -> str:
    """Hàm gọi Gemini bất đồng bộ với smart rate limiting."""
    loop = asyncio.get_event_loop()
    
    # Smart delay để tránh overwhelm quota
    await asyncio.sleep(REQUEST_DELAY)
    
    current_api_key = api_key_manager.get_next_key()
    if not current_api_key:
        error_msg = f"Không có API key khả dụng cho {task_description}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg})
    
    logger.info(f"🔑 {task_description} sử dụng key: {current_api_key[-8:]}...")
    
    for attempt in range(max_retries):
        try:
            genai.configure(api_key=current_api_key)
            temp_gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
            
            response = await loop.run_in_executor(
                None,
                lambda: temp_gemini_model.generate_content(
                    prompt,
                    generation_config={"max_output_tokens": 2048}  # Giảm output để tiết kiệm quota
                )
            )
            
            if response.parts:
                return response.text
            else:
                logger.warning(f"Gemini không trả về nội dung ({task_description}). Response: {response}")
                if response.prompt_feedback and response.prompt_feedback.block_reason:
                    logger.warning(f"Bị chặn bởi ({task_description}): {response.prompt_feedback.block_reason_message}")
                return json.dumps({"error": f"Gemini không trả về nội dung ({task_description})."})
                
        except Exception as e:
            error_str = str(e).lower()
            
            if "429" in error_str or "resource_exhausted" in error_str or "too many requests" in error_str:
                # Exponential backoff thông minh hơn
                base_delay = 15  # Base delay lâu hơn
                retry_delay = base_delay * (2 ** attempt) + (attempt * 5)  # Exponential + linear
                logger.warning(f"⚠️ Rate limit ({task_description}), đợi {retry_delay}s trước retry {attempt+1}/{max_retries}")
                await asyncio.sleep(retry_delay)
                
                # Thử key khác nếu có
                current_api_key = api_key_manager.get_next_key()
                if not current_api_key:
                    logger.error("Hết API keys khả dụng")
                    return json.dumps({"error": "Hết API keys khả dụng"})
                
            elif "invalid_argument" in error_str and ("token" in error_str or "request payload" in error_str or "size limit" in error_str):
                logger.error(f"Lỗi: Prompt quá lớn ({task_description}). {str(e)}")
                return json.dumps({"error": f"Prompt quá lớn cho model xử lý ({task_description})."})
                
            else:
                logger.error(f"Lỗi không xác định khi gọi API Gemini ({task_description}) (Lần {attempt+1}/{max_retries}): {str(e)}")
                if attempt == max_retries - 1:
                    return json.dumps({"error": f"Lỗi không xác định và không thể phục hồi từ Gemini ({task_description})."})
                await asyncio.sleep(10 + attempt * 5)
                
    logger.error(f"Vẫn gặp lỗi sau {max_retries} lần thử lại với Gemini ({task_description}).")
    return json.dumps({"error": f"Không thể kết nối với Gemini sau nhiều lần thử ({task_description})."})


async def analyze_user_request_with_gemini_async(user_request_text: str) -> dict:
    """Bước 1: Sử dụng Gemini để phân tích yêu cầu người dùng."""
    prompt = f"""
    Phân tích đoạn văn bản để tìm **YÊU CẦU THỰC TẾ** của người dùng về việc mua nguyên liệu nấu ăn.

    **VĂN BẢN CẦN PHÂN TÍCH:**
    "{user_request_text}"

    **QUY TẮC QUAN TRỌNG:**
    1. **CHỈ EXTRACT NGUYÊN LIỆU TỪ YÊU CẦU THỰC TẾ** của người dùng (câu hỏi, đặt hàng, mua sắm)
    2. **BỎ QUA PHẦN GỢI Ý/MÔ TẢ CÔNG THỨC** từ AI assistant hoặc người khác
    3. **TÌM CÁC KEYWORD:** "tôi cần", "mua", "làm món", "nấu", "tôi muốn"
    4. **PHÂN BIỆT NGỮ CẢNH:** Gợi ý vs Yêu cầu thực tế

    **CÁCH NHẬN DIỆN:**
    - **YÊU CẦU THỰC TẾ:** "Tôi muốn nấu canh chua, cần mua cá lóc"
    - **GỢI Ý (BỎ QUA):** "Canh bí đỏ đậu xanh - Nguyên liệu: Thịt băm, bí đỏ..."

    **OUTPUT JSON:**
    {{
        "dish_name": "Tên món ăn người dùng THỰC SỰ muốn nấu (null nếu chỉ hỏi gợi ý)",
        "requested_ingredients": ["Danh sách nguyên liệu người dùng THỰC SỰ yêu cầu mua"]
    }}

    **VÍ DỤ:**

    Input 1: "Tôi muốn nấu canh chua cá lóc, cần mua cá lóc, me, cà chua"
    {{
        "dish_name": "Canh chua cá lóc", 
        "requested_ingredients": ["cá lóc", "me", "cà chua"]
    }}

    Input 2: "Gợi ý món ăn: Canh bí đỏ - Nguyên liệu: thịt băm, bí đỏ. Cơm sen - Nguyên liệu: gạo lứt, hạt sen"
    {{
        "dish_name": null,
        "requested_ingredients": []
    }}

    Input 3: "Cảm ơn gợi ý! Tôi sẽ làm canh bí đỏ, cần mua thịt băm và bí đỏ"
    {{
        "dish_name": "Canh bí đỏ",
        "requested_ingredients": ["thịt băm", "bí đỏ"]
    }}

    CHỈ TRẢ VỀ JSON - KHÔNG GIẢI THÍCH THÊM.
    """
    logger.info("Gửi yêu cầu phân tích nguyên liệu đến Gemini...")
    response_text = await call_gemini_api_generic_async(prompt, "Phân tích yêu cầu người dùng")
    
    try:
        cleaned_response_text = response_text.strip()
        if cleaned_response_text.startswith("```json"):
            cleaned_response_text = cleaned_response_text[7:]
        if cleaned_response_text.endswith("```"):
            cleaned_response_text = cleaned_response_text[:-3]
        
        analysis_result = json.loads(cleaned_response_text)
        if isinstance(analysis_result, list) and len(analysis_result) > 0:
            # Nếu Gemini trả về list, lấy phần tử đầu tiên
            analysis_result = analysis_result[0]
            
        if "error" in analysis_result:
            logger.error(f"Lỗi từ Gemini khi phân tích yêu cầu: {analysis_result['error']}")
            return {
                "dish_name": None,
                "requested_ingredients": [],
                "error": analysis_result['error']
            }
        
        # Đảm bảo format đúng
        result = {
            "dish_name": analysis_result.get("dish_name"),
            "requested_ingredients": analysis_result.get("requested_ingredients", [])
        }
        
        logger.info(f"Kết quả phân tích từ Gemini: {result}")
        return result
        
    except json.JSONDecodeError as e:
        logger.error(f"Lỗi giải mã JSON từ phân tích yêu cầu của Gemini: {e}")
        logger.error(f"Phản hồi gốc từ Gemini (phân tích): {response_text}")
        return {
            "dish_name": None,
            "requested_ingredients": [],
            "error": "Lỗi parse JSON phân tích yêu cầu."
        }
    except Exception as e:
        logger.error(f"Lỗi không xác định khi phân tích yêu cầu: {str(e)}")
        return {
            "dish_name": None,
            "requested_ingredients": [],
            "error": f"Lỗi không xác định: {str(e)}"
        }


async def find_product_for_ingredient_async_from_cache(
    ingredient_name: str,
    all_cached_products: List[Dict[str, Any]]
) -> Tuple[Optional[str], Optional[str]]:
    """
    Tìm và chọn product_id cho một nguyên liệu cụ thể từ danh sách sản phẩm đã cache.
    Sử dụng Gemini để chọn sản phẩm phù hợp nhất từ các lô (batches).
    Trả về (product_id, product_name) hoặc (None, None) nếu không tìm thấy.
    """
    if not all_cached_products:
        logger.warning(f"PRODUCT_FIND_TOOL: Không có sản phẩm trong cache để tìm '{ingredient_name}'")
        return None, None

    logger.info(f"PRODUCT_FIND_TOOL: Bắt đầu tìm sản phẩm cho '{ingredient_name}' từ {len(all_cached_products)} sản phẩm cache")

    # Chia thành các lô hoặc gửi toàn bộ nếu cache nhỏ
    if len(all_cached_products) <= PRODUCTS_PER_GEMINI_BATCH:
        # Cache nhỏ - gửi toàn bộ trong 1 lô
        num_batches = 1
        logger.info(f"PRODUCT_FIND_TOOL: Cache nhỏ ({len(all_cached_products)} sản phẩm) - gửi toàn bộ trong 1 lô")
    else:
        # Cache lớn - chia lô
        num_batches = (len(all_cached_products) + PRODUCTS_PER_GEMINI_BATCH - 1) // PRODUCTS_PER_GEMINI_BATCH
        logger.info(f"PRODUCT_FIND_TOOL: Chia thành {num_batches} lô (mỗi lô {PRODUCTS_PER_GEMINI_BATCH} sản phẩm)")

    for i in range(num_batches):
        if num_batches == 1:
            # Gửi toàn bộ cache
            current_product_batch = all_cached_products
        else:
            # Chia lô thông thường
            batch_start_index = i * PRODUCTS_PER_GEMINI_BATCH
            current_product_batch = all_cached_products[batch_start_index : batch_start_index + PRODUCTS_PER_GEMINI_BATCH]

        if not current_product_batch:
            continue

        # Tạo chuỗi sản phẩm cho prompt
        product_batch_str = "\n".join(
            [f"- ID: {p['product_id']}, Tên Sản Phẩm: {p['name']}" for p in current_product_batch]
        )
        
        # Prompt tối ưu cho Top 3 với confidence score
        prompt_for_selection = f"""
        Tìm sản phẩm tương ứng với nguyên liệu: "{ingredient_name}".
        Lô sản phẩm (ID và Tên):
        {product_batch_str}

        **NHIỆM VỤ:** Chọn TỐI ĐA 3 sản phẩm PHÙ HỢP NHẤT, sắp xếp theo mức độ phù hợp giảm dần.

        **QUY TẮC:**
        1. **KHỚP TÊN:** Ưu tiên sản phẩm có tên chứa đúng hoặc gần tên nguyên liệu "{ingredient_name}"
        2. **LOẠI TRỪ CHẾ BIẾN:** KHÔNG chọn sản phẩm chế biến sẵn như "Mì gói vị {ingredient_name}"
        3. **ƯU TIÊN NGUYÊN LIỆU THÔ/SƠ CHẾ**
        4. **CONFIDENCE:** Đánh giá độ tin cậy từ 0.0-1.0 (1.0 = hoàn toàn chắc chắn)

        **TRẢ VỀ JSON ARRAY (KHÔNG MARKDOWN, KHÔNG GIẢI THÍCH):**
        [
            {{
                "selected_product_id": "ID_1",
                "selected_product_name": "TÊN_1", 
                "confidence_score": 0.95,
                "reason": "Khớp chính xác tên nguyên liệu"
            }},
            {{
                "selected_product_id": "ID_2",
                "selected_product_name": "TÊN_2",
                "confidence_score": 0.8,
                "reason": "Gần đúng, dạng sơ chế"
            }}
        ]
        Nếu KHÔNG tìm thấy sản phẩm phù hợp: []
        """
        
        task_desc = f"PRODUCT_FIND_TOOL: Chọn sản phẩm cho '{ingredient_name}' từ cache batch {i+1}/{num_batches}"
        logger.info(f"{task_desc} - Gửi {len(current_product_batch)} sản phẩm cho Gemini")
        
        response_text = await call_gemini_api_generic_async(prompt_for_selection, task_desc)
        
        try:
            # Clean and parse JSON response cho Top 3 results
            cleaned_response_text = response_text.strip()
            
            # Tìm JSON array trong response
            array_match = re.search(r'\[.*\]', cleaned_response_text, re.DOTALL)
            if array_match:
                cleaned_response_text = array_match.group(0)
            else:
                logger.warning(f"PRODUCT_FIND_TOOL: Không tìm thấy JSON array trong response (batch {i+1} cho '{ingredient_name}')")
                continue

            selection_results = json.loads(cleaned_response_text)
            
            # Kiểm tra nếu có lỗi hoặc array rỗng
            if not isinstance(selection_results, list) or not selection_results:
                logger.info(f"PRODUCT_FIND_TOOL: Gemini không tìm thấy sản phẩm phù hợp trong batch {i+1} cho '{ingredient_name}'")
                continue
            
            # Xử lý top 3 results, tìm result tốt nhất
            best_result = None
            for result in selection_results:
                if not isinstance(result, dict):
                    continue
                    
                product_id_raw = result.get("selected_product_id")
                product_name = result.get("selected_product_name")
                confidence = result.get("confidence_score", 0.0)
                
                # Validate product_id
                product_id = None
                if product_id_raw is not None and product_id_raw != "null":
                    try:
                        product_id = str(int(product_id_raw))
                    except (ValueError, TypeError):
                        continue
                
                if product_id and product_name and confidence > 0.0:
                    if best_result is None or confidence > best_result["confidence"]:
                        best_result = {
                            "product_id": product_id,
                            "product_name": product_name,
                            "confidence": confidence,
                            "reason": result.get("reason", "")
                        }
            
                            # Xử lý multiple results để tìm alternative nếu best bị duplicate
            if best_result:
                confidence = best_result["confidence"]
                
                # Thử tất cả results theo thứ tự confidence giảm dần
                for result_candidate in sorted(selection_results, key=lambda x: x.get("confidence_score", 0), reverse=True):
                    if not isinstance(result_candidate, dict):
                        continue
                        
                    candidate_id_raw = result_candidate.get("selected_product_id")
                    candidate_name = result_candidate.get("selected_product_name")
                    candidate_confidence = result_candidate.get("confidence_score", 0.0)
                    
                    # Validate candidate_id
                    candidate_id = None
                    if candidate_id_raw is not None and candidate_id_raw != "null":
                        try:
                            candidate_id = str(int(candidate_id_raw))
                        except (ValueError, TypeError):
                            continue
                    
                    if candidate_id and candidate_name and candidate_confidence >= 0.5:
                        if candidate_confidence >= 0.75:  # Ngưỡng tin cậy cao - dừng ngay
                            logger.info(f"PRODUCT_FIND_TOOL: ⚡ EARLY STOP - Gemini chọn ID: {candidate_id}, Tên: {candidate_name} cho '{ingredient_name}' (batch {i+1}, confidence: {candidate_confidence:.2f})")
                            return candidate_id, candidate_name
                        elif i == num_batches - 1:  # Batch cuối, chấp nhận kết quả này
                            logger.info(f"PRODUCT_FIND_TOOL: Chọn candidate cuối ID: {candidate_id}, Tên: {candidate_name} cho '{ingredient_name}' (confidence: {candidate_confidence:.2f})")
                            return candidate_id, candidate_name
                        else:  # Chưa phải batch cuối, tiếp tục tìm
                            logger.info(f"PRODUCT_FIND_TOOL: Tìm thấy candidate ID: {candidate_id} (confidence: {candidate_confidence:.2f}), tiếp tục tìm batch tiếp theo...")
                            break  # Thoát khỏi vòng lặp candidates, chuyển sang batch tiếp theo
        
        except json.JSONDecodeError as e:
            logger.warning(f"PRODUCT_FIND_TOOL: Lỗi parse JSON từ Gemini (batch {i+1} cho '{ingredient_name}'): {e}")
            logger.warning(f"PRODUCT_FIND_TOOL: Raw response: {response_text[:200]}...")
            continue
        except Exception as e:
            logger.error(f"PRODUCT_FIND_TOOL: Lỗi xử lý response Gemini (batch {i+1} cho '{ingredient_name}'): {e}")
            continue

    logger.info(f"PRODUCT_FIND_TOOL: Không tìm thấy sản phẩm phù hợp cho '{ingredient_name}' sau khi duyệt hết cache")
    return None, None


async def process_user_request_async(user_request_text: str) -> dict:
    """
    Quy trình chính: Phân tích yêu cầu, tìm sản phẩm cho từng nguyên liệu từ cache.
    """
    analysis = await analyze_user_request_with_gemini_async(user_request_text)
    
    if "error" in analysis:  # Kiểm tra lỗi từ bước phân tích
        return {
            "error": analysis["error"],
            "dish_name_identified": analysis.get("dish_name"),
            "ingredient_mapping_results": [],
            "ingredients_not_found_product_id": analysis.get("requested_ingredients", [])
        }

    if not analysis.get("requested_ingredients"):
        logger.info("PRODUCT_FIND_TOOL: Không có nguyên liệu nào được yêu cầu sau phân tích")
        return {
            "dish_name_identified": analysis.get("dish_name"),
            "processed_request": user_request_text,
            "ingredient_mapping_results": [],
            "ingredients_not_found_product_id": []
        }

    dish_name = analysis.get("dish_name")
    requested_ingredients = list(set(analysis.get("requested_ingredients", [])))  # Loại bỏ trùng lặp
    
    logger.info(f"PRODUCT_FIND_TOOL: Món ăn: {dish_name}, Nguyên liệu yêu cầu (đã lọc trùng): {requested_ingredients}")

    # Lấy toàn bộ danh sách sản phẩm từ cache, nếu chưa có thì tạo ngay
    all_cached_products = await ensure_products_cache_available()
    if not all_cached_products:
        logger.error("PRODUCT_FIND_TOOL: Lỗi nghiêm trọng - Không thể lấy hoặc tạo cache sản phẩm")
        return {
            "error": "Lỗi hệ thống: Không thể truy cập dữ liệu sản phẩm.",
            "dish_name_identified": dish_name,
            "ingredient_mapping_results": [],
            "ingredients_not_found_product_id": requested_ingredients
        }
    logger.info(f"PRODUCT_FIND_TOOL: Sẽ tìm kiếm từ {len(all_cached_products)} sản phẩm trong cache")

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_GEMINI_PRODUCT_CALLS)
    used_product_ids = set()  # Track để tránh trùng lặp product_id
    used_product_ids_lock = asyncio.Lock()  # Thread-safe access to used_product_ids
    
    async def process_single_ingredient_with_semaphore(ingredient_name: str):
        async with semaphore:
            logger.info(f"PRODUCT_FIND_TOOL: 🚀 Bắt đầu xử lý '{ingredient_name}' (concurrency: {MAX_CONCURRENT_GEMINI_PRODUCT_CALLS})")
            start_time = asyncio.get_event_loop().time()
            
            product_id, product_name = await find_product_for_ingredient_async_from_cache(
                ingredient_name, 
                all_cached_products
            )
            
            elapsed = asyncio.get_event_loop().time() - start_time
            
            # Tạo kết quả cho nguyên liệu này
            mapping_result_for_ingredient = {
                "requested_ingredient": ingredient_name,
                "product_id": None,
                "product_name": None,
                "status": "Không tìm thấy sản phẩm phù hợp"
            }

            if product_id and product_name:
                # Thread-safe check và add product_id
                async with used_product_ids_lock:
                    if product_id in used_product_ids:
                        logger.warning(f"PRODUCT_FIND_TOOL: ⚠️ Product ID {product_id} đã được dùng cho nguyên liệu khác, bỏ qua cho '{ingredient_name}'")
                        mapping_result_for_ingredient["status"] = f"Sản phẩm đã được gán cho nguyên liệu khác"
                    else:
                        used_product_ids.add(product_id)  # Đánh dấu đã dùng
                        mapping_result_for_ingredient.update({
                            "product_id": product_id,
                            "product_name": product_name,
                            "status": "Đã tìm thấy sản phẩm"
                        })
                        logger.info(f"PRODUCT_FIND_TOOL: ✅ '{ingredient_name}' -> ID: {product_id}, Tên: {product_name} ({elapsed:.2f}s)")
            else:
                logger.info(f"PRODUCT_FIND_TOOL: ❌ Không tìm thấy sản phẩm cho '{ingredient_name}' ({elapsed:.2f}s)")
            
            return mapping_result_for_ingredient

    tasks = [process_single_ingredient_with_semaphore(ing) for ing in requested_ingredients]
    logger.info(f"PRODUCT_FIND_TOOL: 🛒 Tạo {len(tasks)} tasks song song để tìm sản phẩm (MAX_CONCURRENT: {MAX_CONCURRENT_GEMINI_PRODUCT_CALLS})")
    
    # Chạy các task song song với 7 API keys
    all_mapping_results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Xử lý kết quả cuối cùng với counting chính xác
    final_results_for_frontend = []
    ingredients_not_found = []
    successful_mappings = 0
    duplicated_mappings = 0
    error_mappings = 0
    
    for i, res_item in enumerate(all_mapping_results):
        if isinstance(res_item, Exception):
            # Lỗi trong task xử lý nguyên liệu
            ing_name = requested_ingredients[i]
            logger.error(f"PRODUCT_FIND_TOOL: 💥 Lỗi trong task '{ing_name}': {res_item}")
            final_results_for_frontend.append({
                "requested_ingredient": ing_name,
                "product_id": None,
                "product_name": None,
                "status": f"Lỗi xử lý: {str(res_item)}"
            })
            ingredients_not_found.append(ing_name)
            error_mappings += 1
            continue

        req_ing = res_item["requested_ingredient"]
        prod_id = res_item.get("product_id")
        status = res_item.get("status", "")
        
        final_results_for_frontend.append(res_item)
        
        # Phân loại kết quả để counting chính xác
        if prod_id:  # Có product_id = thành công
            successful_mappings += 1
        elif "đã được gán cho nguyên liệu khác" in status:  # Duplicate
            duplicated_mappings += 1
            # Không add vào ingredients_not_found vì đã tìm thấy sản phẩm, chỉ bị trùng
        else:  # Thực sự không tìm thấy
            ingredients_not_found.append(req_ing)
            
    # Tính toán metrics chi tiết
    total_processed = len(requested_ingredients)
    actual_products_found = successful_mappings + duplicated_mappings  # Tổng số sản phẩm thực sự được tìm thấy
    
    logger.info(f"PRODUCT_FIND_TOOL: 🎯 HOÀN THÀNH XỬ LÝ")
    logger.info(f"  📊 Tổng nguyên liệu: {total_processed}")
    logger.info(f"  ✅ Mapping thành công: {successful_mappings}")
    logger.info(f"  🔄 Tìm thấy nhưng trùng lặp: {duplicated_mappings}")
    logger.info(f"  ❌ Không tìm thấy: {len(ingredients_not_found)}")
    logger.info(f"  💥 Lỗi xử lý: {error_mappings}")
    logger.info(f"  🎯 Tổng sản phẩm tìm thấy: {actual_products_found}")
    logger.info(f"  📈 Tỷ lệ tìm thấy sản phẩm: {actual_products_found/total_processed*100:.1f}%")
    logger.info(f"  📈 Tỷ lệ mapping thành công: {successful_mappings/total_processed*100:.1f}%")
    logger.info(f"  🏃 Concurrency: {MAX_CONCURRENT_GEMINI_PRODUCT_CALLS} API keys")
    logger.info(f"  📦 Batch size: {PRODUCTS_PER_GEMINI_BATCH} sản phẩm/lô")
    
    # Validation check - đảm bảo số lượng kết quả trả về = số nguyên liệu đầu vào
    assert len(final_results_for_frontend) == total_processed, f"CRITICAL: Số lượng results ({len(final_results_for_frontend)}) != số nguyên liệu ({total_processed})"
    assert (successful_mappings + duplicated_mappings + len(ingredients_not_found) + error_mappings) == total_processed, "CRITICAL: Tổng các loại kết quả không bằng tổng nguyên liệu"

    # Lấy danh sách unique product IDs đã được map thành công
    unique_product_ids = list(used_product_ids)
    
    return {
        "dish_name_identified": dish_name,
        "processed_request": user_request_text,
        "ingredient_mapping_results": final_results_for_frontend,
        "ingredients_not_found_product_id": list(set(ingredients_not_found)),  # Đảm bảo unique
        "summary": {
            "total_ingredients_requested": total_processed,
            "successful_mappings": successful_mappings,
            "duplicated_mappings": duplicated_mappings,
            "not_found": len(ingredients_not_found),
            "errors": error_mappings,
            "unique_products_found": len(unique_product_ids),
            "unique_product_ids": unique_product_ids,
            "success_rate": round(successful_mappings/total_processed*100, 1),
            "product_discovery_rate": round(actual_products_found/total_processed*100, 1)
        }
    }


async def main_test_product_async():
    """Test function for async product finding."""
    # Mock CacheService cho testing
    class MockCacheService:
        _products = []  # Bắt đầu với cache trống để test auto-loading
        _mock_db_products = [
            {"product_id": "100", "name": "Gạo thơm Jasmine túi 5kg"},
            {"product_id": "101", "name": "Gạo nếp cái hoa vàng"},
            {"product_id": "200", "name": "Giò heo rút xương CP"},
            {"product_id": "201", "name": "Thịt ba chỉ heo MeatDeli"},
            {"product_id": "300", "name": "Cá lóc phi lê làm sạch"},
            {"product_id": "301", "name": "Cá diêu hồng tươi"},
            {"product_id": "1073", "name": "Nấm đông cô tươi VietGap 150g"},
            {"product_id": "1074", "name": "Thùng 30 gói mì Gấu Đỏ rau nấm 62g"}, # Sản phẩm gây nhiễu
            {"product_id": "1075", "name": "Nấm đùi gà"},
            {"product_id": "1076", "name": "Nấm kim châm"},
        ]
        
        @classmethod
        def get_all_products_list(cls):
            logger.info(f"MOCK_CACHE: get_all_products_list called. Current cache size: {len(cls._products)}")
            return cls._products if cls._products else None
        
        @classmethod
        def cache_all_products_list(cls, products, ttl=None):
            logger.info(f"MOCK_CACHE: cache_all_products_list called with {len(products)} products.")
            cls._products = products
            return True

    # Mock refresh_products_cache_task
    async def mock_refresh_products_cache_task():
        logger.info("MOCK_REFRESH: Simulating database fetch...")
        # Simulate cache refresh bằng cách load mock data
        MockCacheService.cache_all_products_list(MockCacheService._mock_db_products)
        return {
            "success": True,
            "products_count": len(MockCacheService._mock_db_products),
            "message": "Mock cache refresh thành công"
        }
    
    # Mock CacheService methods
    original_get_method = CacheService.get_all_products_list
    original_cache_method = CacheService.cache_all_products_list
    CacheService.get_all_products_list = MockCacheService.get_all_products_list
    CacheService.cache_all_products_list = MockCacheService.cache_all_products_list
    
    # Mock refresh function
    import app.services.background_products_cache as bg_cache_module
    original_refresh_task = bg_cache_module.refresh_products_cache_task
    bg_cache_module.refresh_products_cache_task = mock_refresh_products_cache_task
    
    try:
        # Test cache availability function
        print("\n=== TEST CACHE AVAILABILITY ===")
        cache_test = await ensure_products_cache_available()
        print(f"Cache test result: {len(cache_test)} sản phẩm")

        # Test case 1: Yêu cầu nhiều nguyên liệu để test concurrency
        user_input_text_1 = "Tôi muốn nấu Cháo cá giò heo, tôi cần mua Gạo thơm, Gạo nếp, Giò heo rút xương, Nạc cá lóc, Nấm đông cô, Nấm đùi gà."
        print(f"\n🧪 PERFORMANCE TEST - Xử lý yêu cầu: '{user_input_text_1}'")
        print(f"📊 Config: MAX_CONCURRENT={MAX_CONCURRENT_GEMINI_PRODUCT_CALLS}, BATCH_SIZE={PRODUCTS_PER_GEMINI_BATCH}")
        start_time_1 = asyncio.get_event_loop().time()
        result_1 = await process_user_request_async(user_input_text_1)
        end_time_1 = asyncio.get_event_loop().time()
        print(f"\n⚡ Thời gian xử lý: {end_time_1 - start_time_1:.2f} giây")
        print("\n--- KẾT QUẢ XỬ LÝ HIỆU SUẤT ---")
        print(json.dumps(result_1, indent=2, ensure_ascii=False))

        # Test case 2: Nguyên liệu có thể gây nhầm lẫn
        user_input_text_2 = "Tôi cần nấm đông cô và nấm đùi gà."
        print(f"\nĐang xử lý yêu cầu 2: '{user_input_text_2}'")
        start_time_2 = asyncio.get_event_loop().time()
        result_2 = await process_user_request_async(user_input_text_2)
        end_time_2 = asyncio.get_event_loop().time()
        print(f"\n⏱️ Thời gian xử lý 2: {end_time_2 - start_time_2:.2f} giây")
        print("\n--- KẾT QUẢ XỬ LÝ YÊU CẦU 2 ---")
        print(json.dumps(result_2, indent=2, ensure_ascii=False))
        
        # Test case 3: Nguyên liệu không có trong cache
        user_input_text_3 = "Tôi cần yến sào Khánh Hòa."
        print(f"\nĐang xử lý yêu cầu 3: '{user_input_text_3}'")
        start_time_3 = asyncio.get_event_loop().time()
        result_3 = await process_user_request_async(user_input_text_3)
        end_time_3 = asyncio.get_event_loop().time()
        print(f"\n⏱️ Thời gian xử lý 3: {end_time_3 - start_time_3:.2f} giây")
        print("\n--- KẾT QUẢ XỬ LÝ YÊU CẦU 3 ---")
        print(json.dumps(result_3, indent=2, ensure_ascii=False))

    except ValueError as ve: 
        print(f"Lỗi cấu hình: {ve}")
    except Exception as e:
        logger.error(f"Lỗi không mong muốn ở main (product_find_tool): {str(e)}", exc_info=True)
    finally:
        # Khôi phục tất cả mock functions
        CacheService.get_all_products_list = original_get_method
        CacheService.cache_all_products_list = original_cache_method
        bg_cache_module.refresh_products_cache_task = original_refresh_task


if __name__ == "__main__":
    asyncio.run(main_test_product_async())