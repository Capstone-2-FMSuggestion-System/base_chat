import logging
from typing import Dict, Any
from app.services.cache_service import CacheService
from app.repositories.product_repository import ProductRepository
from app.db.database import SessionLocal

logger = logging.getLogger(__name__)

async def refresh_products_cache_task() -> Dict[str, Any]:
    """
    Background task để làm mới cache danh sách sản phẩm
    Được gọi bất đồng bộ khi người dùng đăng nhập
    """
    logger.info("BACKGROUND_CACHE: Bắt đầu làm mới cache danh sách sản phẩm")
    db_session = None
    try:
        db_session = SessionLocal()
        product_repo = ProductRepository(db_session)
        all_products = product_repo.get_all_product_ids_and_names()
        
        if all_products:
            success = CacheService.cache_all_products_list(all_products)
            if success:
                logger.info(f"BACKGROUND_CACHE: Đã cache thành công {len(all_products)} sản phẩm")
                return {
                    "success": True,
                    "products_count": len(all_products),
                    "message": "Cache sản phẩm thành công"
                }
            else:
                logger.error("BACKGROUND_CACHE: Lỗi khi lưu vào cache")
                return {
                    "success": False,
                    "error": "Lỗi khi lưu vào cache"
                }
        else:
            logger.warning("BACKGROUND_CACHE: Không có sản phẩm nào từ DB để cache")
            return {
                "success": False,
                "error": "Không có sản phẩm nào từ DB"
            }
    except Exception as e:
        logger.error(f"BACKGROUND_CACHE: Lỗi khi làm mới cache sản phẩm: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e)
        }
    finally:
        if db_session:
            db_session.close()

def should_refresh_products_cache() -> bool:
    """
    Kiểm tra xem có cần refresh cache sản phẩm không
    """
    try:
        cached_products = CacheService.get_all_products_list()
        return cached_products is None or len(cached_products) == 0
    except Exception as e:
        logger.error(f"BACKGROUND_CACHE: Lỗi khi kiểm tra cache: {e}")
        return True 