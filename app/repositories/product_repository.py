import logging
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from app.db.models import Product

logger = logging.getLogger(__name__)

class ProductRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_all_product_ids_and_names(self) -> List[Dict[str, Any]]:
        """
        Lấy toàn bộ product_id và name từ bảng products
        """
        try:
            logger.info("PRODUCT_REPO: Lấy toàn bộ product_id và name từ DB")
            products = self.db.query(Product.product_id, Product.name).order_by(Product.product_id).all()
            result = [{"product_id": p.product_id, "name": p.name} for p in products]
            logger.info(f"PRODUCT_REPO: Đã lấy {len(result)} sản phẩm từ DB")
            return result
        except Exception as e:
            logger.error(f"PRODUCT_REPO: Lỗi khi lấy danh sách sản phẩm: {e}", exc_info=True)
            return [] 