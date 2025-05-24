import os
import sys
import logging
import asyncio
import json
from pinecone import Pinecone
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_huggingface import HuggingFaceEmbeddings
from typing import Optional, Tuple

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from app.services.api_key_manager import get_api_key_manager

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Tải biến môi trường
load_dotenv()

# --- Cấu hình ---
PINECONE_API_KEY = os.getenv("PRODUCT_DB_PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "product-index")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-lite")
EMBEDDING_MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
TOP_K_PRODUCTS_PER_INGREDIENT = 500
MAX_CONCURRENT_GEMINI_PRODUCT_CALLS = 3

api_key_manager = get_api_key_manager()

# Kiểm tra API keys
if not PINECONE_API_KEY or not PINECONE_INDEX_NAME:
    missing_keys = [
        key for key, value in {
            "PRODUCT_DB_PINECONE_API_KEY": PINECONE_API_KEY,
            "PINECONE_INDEX_NAME": PINECONE_INDEX_NAME
        }.items() if not value
    ]
    logger.error(f"Các biến môi trường sau không được thiết lập: {', '.join(missing_keys)}")
    raise ValueError(f"Các biến môi trường sau không được thiết lập: {', '.join(missing_keys)}")

if not api_key_manager.is_healthy():
    logger.error("ApiKeyManager không khả dụng. Kiểm tra cấu hình API keys trong file .env")
    raise ValueError("ApiKeyManager không khả dụng. Kiểm tra cấu hình API keys trong file .env")

logger.info(f"✅ Product Find Tool đã khởi tạo với {len(api_key_manager.get_all_keys())} API keys có sẵn")


def init_services():
    """Khởi tạo kết nối Pinecone và embedding model."""
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        try:
            index_info = pc.describe_index(PINECONE_INDEX_NAME)
            logger.info(f"Pinecone index '{PINECONE_INDEX_NAME}' đã tồn tại")
        except Exception as e:
            logger.error(f"Pinecone index '{PINECONE_INDEX_NAME}' không tồn tại hoặc lỗi khi truy cập: {str(e)}")
            raise ValueError(f"Pinecone index '{PINECONE_INDEX_NAME}' không tồn tại.") from e

        pinecone_index = pc.Index(PINECONE_INDEX_NAME)
        logger.info(f"Đã tạo đối tượng Index cho: {PINECONE_INDEX_NAME}")

        logger.info(f"Đang tải mô hình embedding từ {EMBEDDING_MODEL_NAME}...")
        embeddings_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info("Đã tải thành công mô hình embedding.")

        return pinecone_index, embeddings_model
    except Exception as e:
        logger.error(f"Lỗi khi khởi tạo kết nối dịch vụ: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise


async def call_gemini_api_generic_async(prompt: str, task_description: str, max_retries: int = 3) -> str:
    """Hàm gọi Gemini bất đồng bộ với API key rotation."""
    loop = asyncio.get_event_loop()
    
    current_api_key = api_key_manager.get_next_key()
    if not current_api_key:
        error_msg = f"Không có API key khả dụng cho {task_description}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg})
    
    logger.info(f"🔑 {task_description} sử dụng key: {current_api_key[:10]}...")
    
    for attempt in range(max_retries):
        try:
            genai.configure(api_key=current_api_key)
            temp_gemini_model = genai.GenerativeModel(GEMINI_MODEL_NAME)
            
            response = await loop.run_in_executor(
                None,
                lambda: temp_gemini_model.generate_content(
                    prompt,
                    generation_config={"max_output_tokens": 4096}
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
                retry_delay = 10 + attempt * 10
                logger.warning(f"Lỗi quota/rate limit Gemini ({task_description}), thử lại sau {retry_delay} giây...")
                await asyncio.sleep(retry_delay)
                
            elif "invalid_argument" in error_str and ("token" in error_str or "request payload" in error_str or "size limit" in error_str):
                logger.error(f"Lỗi: Prompt quá lớn ({task_description}). {str(e)}")
                return json.dumps({"error": f"Prompt quá lớn cho model xử lý ({task_description})."})
                
            else:
                logger.error(f"Lỗi không xác định khi gọi API Gemini ({task_description}) (Lần {attempt+1}/{max_retries}): {str(e)}")
                if attempt == max_retries - 1:
                    return json.dumps({"error": f"Lỗi không xác định và không thể phục hồi từ Gemini ({task_description})."})
                await asyncio.sleep(15)
                
    logger.error(f"Vẫn gặp lỗi sau {max_retries} lần thử lại với Gemini ({task_description}).")
    return json.dumps({"error": f"Không thể kết nối với Gemini sau nhiều lần thử ({task_description})."})


async def analyze_user_request_with_gemini_async(user_request_text: str) -> dict:
    """Bước 1: Sử dụng Gemini để phân tích yêu cầu người dùng."""
    prompt = f"""
    Phân tích đoạn văn bản yêu cầu sau của người dùng để xác định các nguyên liệu họ cần mua và tên món ăn (nếu có).
    Yêu cầu: "{user_request_text}"

    Hãy trả lời dưới dạng một đối tượng JSON DUY NHẤT có các trường sau:
    - "dish_name": (string) Tên món ăn chính mà người dùng muốn nấu (nếu có thể xác định, nếu không thì để là null).
    - "requested_ingredients": (list of strings) Danh sách các tên nguyên liệu mà người dùng đề cập. Cố gắng chuẩn hóa tên gọi nếu có thể (ví dụ: "hành cây" -> "hành lá", "bột canh" -> "gia vị nêm").

    Ví dụ: Nếu input là "Tôi muốn nấu canh chua cá lóc, cần mua cá lóc, me, cà chua, giá và ít rau thơm."
    Output mong muốn:
    {{
        "dish_name": "Canh chua cá lóc",
        "requested_ingredients": ["cá lóc", "me", "cà chua", "giá đỗ", "rau thơm"]
    }}

    Nếu input là "Cho tôi xin ít thịt ba chỉ, vài quả cà chua và hành lá."
    Output mong muốn:
    {{
        "dish_name": null,
        "requested_ingredients": ["thịt ba chỉ", "cà chua", "hành lá"]
    }}
    Chỉ trả về đối tượng JSON.
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
        if "error" in analysis_result:
            logger.error(f"Lỗi từ Gemini khi phân tích yêu cầu: {analysis_result['error']}")
            return {"dish_name": None, "requested_ingredients": [], "error": analysis_result['error']}
        
        analysis_result.setdefault("dish_name", None)
        analysis_result.setdefault("requested_ingredients", [])
        logger.info(f"Kết quả phân tích từ Gemini: {analysis_result}")
        return analysis_result
        
    except json.JSONDecodeError as e:
        logger.error(f"Lỗi giải mã JSON từ phân tích yêu cầu của Gemini: {e}")
        logger.error(f"Phản hồi gốc từ Gemini (phân tích): {response_text}")
        return {"dish_name": None, "requested_ingredients": [], "error": "Lỗi parse JSON phân tích yêu cầu."}


async def find_product_for_ingredient_async(pinecone_index, embeddings_model, ingredient_name: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Bước 2 & 3: Tìm và chọn product_id cho một nguyên liệu cụ thể.
    Trả về (product_id, product_name) hoặc (None, None) nếu không tìm thấy.
    """
    loop = asyncio.get_event_loop()
    pinecone_query_text = f"Sản phẩm cho nguyên liệu: {ingredient_name}"
    
    try:
        query_vector = await loop.run_in_executor(None, embeddings_model.embed_query, pinecone_query_text)
    except Exception as e:
        logger.error(f"Lỗi khi nhúng query cho nguyên liệu '{ingredient_name}': {str(e)}")
        return None, None

    potential_products_context = []
    try:
        query_response = await loop.run_in_executor(
            None,
            lambda: pinecone_index.query(
                vector=query_vector,
                top_k=TOP_K_PRODUCTS_PER_INGREDIENT,
                include_metadata=True
            )
        )
        
        if query_response.matches:
            for match in query_response.matches:
                prod_id = match.metadata.get("doc_id")
                prod_name = match.metadata.get("name")
                if prod_id and prod_name:
                    potential_products_context.append({"id": prod_id, "name": prod_name})
                    
    except Exception as e:
        logger.error(f"Lỗi khi query Pinecone cho nguyên liệu '{ingredient_name}': {str(e)}")
        return None, None

    if not potential_products_context:
        logger.info(f"Không tìm thấy sản phẩm tiềm năng nào trên Pinecone cho nguyên liệu: '{ingredient_name}'")
        return None, None

    potential_products_str = "\n".join([f"- ID: {p['id']}, Tên Sản Phẩm: {p['name']}" for p in potential_products_context])
    prompt_for_selection = f"""
    Người dùng cần tìm sản phẩm cho nguyên liệu: "{ingredient_name}".
    Dưới đây là danh sách các sản phẩm (ID và Tên) có thể liên quan được tìm thấy:
    {potential_products_str}

    Hãy chọn ra sản phẩm (ID và Tên) PHÙ HỢP NHẤT cho nguyên liệu "{ingredient_name}".
    Một sản phẩm được coi là phù hợp nếu tên của nó khớp hoặc là một biến thể/loại cụ thể của nguyên liệu đó.
    Ví dụ, nếu người dùng cần "gà", sản phẩm "Gà ta nguyên con" hoặc "Ức gà" có thể phù hợp.

    Trả lời dưới dạng một đối tượng JSON DUY NHẤT có các trường:
    - "selected_product_id": (string) ID của sản phẩm bạn chọn. Nếu không có sản phẩm nào phù hợp, để là null.
    - "selected_product_name": (string) Tên của sản phẩm bạn chọn. Nếu không có, để là null.

    Ví dụ phản hồi: {{"selected_product_id": "123", "selected_product_name": "Gà ta nguyên con"}}
    Hoặc nếu không có: {{"selected_product_id": null, "selected_product_name": null}}
    Chỉ trả về đối tượng JSON.
    """
    
    # Gọi Gemini để chọn sản phẩm (logging đã có trong process_single_ingredient_with_semaphore)
    response_text = await call_gemini_api_generic_async(prompt_for_selection, f"Chọn sản phẩm cho '{ingredient_name}'")
    
    try:
        cleaned_response_text = response_text.strip()
        if cleaned_response_text.startswith("```json"): 
            cleaned_response_text = cleaned_response_text[7:]
        if cleaned_response_text.endswith("```"): 
            cleaned_response_text = cleaned_response_text[:-3]
        
        selection_result = json.loads(cleaned_response_text)
        if "error" in selection_result:
            logger.error(f"Lỗi từ Gemini khi chọn sản phẩm cho '{ingredient_name}': {selection_result['error']}")
            return None, None
        
        return selection_result.get("selected_product_id"), selection_result.get("selected_product_name")
        
    except json.JSONDecodeError as e:
        logger.error(f"Lỗi giải mã JSON khi chọn sản phẩm cho '{ingredient_name}': {e}")
        logger.error(f"Phản hồi gốc từ Gemini (chọn sản phẩm cho '{ingredient_name}'): {response_text}")
        return None, None


async def process_user_request_async(user_request_text: str) -> dict:
    """
    Quy trình chính: Phân tích yêu cầu, tìm sản phẩm cho từng nguyên liệu với concurrency control.
    """
    loop = asyncio.get_event_loop()
    
    try:
        pinecone_index, embeddings_model = await loop.run_in_executor(None, init_services)
    except Exception as e:
        return {"error": f"Lỗi khởi tạo dịch vụ: {str(e)}", "results": []}

    analysis = await analyze_user_request_with_gemini_async(user_request_text)
    if "error" in analysis or not analysis.get("requested_ingredients"):
        return {
            "error": analysis.get("error", "Không thể phân tích yêu cầu người dùng hoặc không tìm thấy nguyên liệu."),
            "dish_name_identified": analysis.get("dish_name"),
            "results": []
        }

    dish_name = analysis.get("dish_name")
    requested_ingredients = analysis.get("requested_ingredients", [])
    logger.info(f"Món ăn xác định (nếu có): {dish_name}")
    logger.info(f"Các nguyên liệu yêu cầu (đã chuẩn hóa bởi Gemini): {requested_ingredients}")

    if not requested_ingredients:
        return {
            "dish_name_identified": dish_name,
            "processed_request": user_request_text,
            "ingredient_mapping_results": [],
            "ingredients_not_found_product_id": []
        }

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_GEMINI_PRODUCT_CALLS)
    
    async def process_single_ingredient_with_semaphore(ingredient_name: str):
        async with semaphore:
            logger.info(f"🔄 BẮT ĐẦU xử lý '{ingredient_name}' (semaphore acquired)...")
            start_time = asyncio.get_event_loop().time()
            
            product_id, product_name = await find_product_for_ingredient_async(pinecone_index, embeddings_model, ingredient_name)
            
            end_time = asyncio.get_event_loop().time()
            elapsed = end_time - start_time
            
            if product_id:
                result = {
                    "requested_ingredient": ingredient_name,
                    "product_id": product_id,
                    "product_name": product_name,
                    "status": "Đã tìm thấy sản phẩm"
                }
                logger.info(f"✅ KẾT THÚC xử lý '{ingredient_name}': Tìm thấy sản phẩm ID: {product_id}, Tên: {product_name} ({elapsed:.2f}s)")
            else:
                result = {
                    "requested_ingredient": ingredient_name,
                    "product_id": None,
                    "product_name": None,
                    "status": "Không tìm thấy sản phẩm phù hợp"
                }
                logger.info(f"❌ KẾT THÚC xử lý '{ingredient_name}': Không tìm thấy sản phẩm phù hợp ({elapsed:.2f}s)")
            
            return result

    # Log chuẩn bị cho tất cả nguyên liệu trước khi tạo tasks
    logger.info(f"🛒 CHUẨN BỊ tìm sản phẩm cho {len(requested_ingredients)} nguyên liệu: {', '.join(requested_ingredients)}")
    
    # Tạo tasks mà không log từng nguyên liệu riêng lẻ
    tasks = []
    for ingredient in requested_ingredients:
        tasks.append(process_single_ingredient_with_semaphore(ingredient))
    
    logger.info(f"🚀 Bắt đầu xử lý {len(tasks)} nguyên liệu với concurrency tối đa {MAX_CONCURRENT_GEMINI_PRODUCT_CALLS}")
    
    ingredient_processing_results = await asyncio.gather(*tasks, return_exceptions=True)
    
    final_results = []
    ingredients_not_found = []
    
    for i, result in enumerate(ingredient_processing_results):
        if isinstance(result, Exception):
            ingredient_name = requested_ingredients[i]
            logger.error(f"❌ Lỗi khi xử lý nguyên liệu '{ingredient_name}': {str(result)}")
            final_results.append({
                "requested_ingredient": ingredient_name,
                "product_id": None,
                "product_name": None,
                "status": f"Lỗi xử lý: {str(result)}"
            })
            ingredients_not_found.append(ingredient_name)
        else:
            final_results.append(result)
            if not result.get("product_id"):
                ingredients_not_found.append(result["requested_ingredient"])

    logger.info(f"🎯 Hoàn thành xử lý: {len(final_results)} nguyên liệu, {len(ingredients_not_found)} không tìm thấy sản phẩm")

    return {
        "dish_name_identified": dish_name,
        "processed_request": user_request_text,
        "ingredient_mapping_results": final_results,
        "ingredients_not_found_product_id": ingredients_not_found
    }


async def main_test_product_async():
    """Test function for async product finding."""
    try:
        user_input_text = "Tôi muốn nấu Cháo cá giò heo, tôi cần mua Gạo thơm, Gạo nếp, Giò heo rút xương, Nạc cá lóc."
        
        print(f"\nĐang xử lý yêu cầu: '{user_input_text}'")
        
        import time
        start_time = time.time()
        result = await process_user_request_async(user_input_text)
        end_time = time.time()
        
        print(f"\n⏱️ Thời gian xử lý: {end_time - start_time:.2f} giây")
        print("\n--- KẾT QUẢ XỬ LÝ YÊU CẦU ---")
        print(json.dumps(result, indent=2, ensure_ascii=False))

    except ValueError as ve: 
        print(f"Lỗi cấu hình: {ve}")
    except Exception as e:
        logger.error(f"Lỗi không mong muốn ở main (product_find_tool): {str(e)}")
        import traceback
        logger.error(traceback.format_exc())


if __name__ == "__main__":
    asyncio.run(main_test_product_async())