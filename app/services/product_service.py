import httpx
import logging
from typing import List, Dict, Any, Optional
from app.config import settings

logger = logging.getLogger(__name__)

class ProductService:
    def __init__(self):
        self.backend_url = settings.BACKEND_URL
        self.timeout = 30.0
    
    async def get_products_by_ids(self, product_ids: List[int]) -> List[Dict[str, Any]]:
        """Lấy thông tin sản phẩm từ backend theo danh sách product_ids"""
        if not product_ids:
            logger.debug("Danh sách product_ids trống")
            return []
        
        logger.info(f"🛒 Bắt đầu lấy thông tin {len(product_ids)} sản phẩm từ backend")
        products = []
        
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            for product_id in product_ids:
                try:
                    response = await client.get(
                        f"{self.backend_url}/api/e-commerce/products/{product_id}"
                    )
                    if response.status_code == 200:
                        product_data_raw = None
                        try:
                            product_data_raw = response.json()
                        except Exception as json_error:
                            logger.error(f"💥 Lỗi khi parse JSON cho sản phẩm {product_id}: {str(json_error)}")
                            logger.error(f"Response text: {response.text}")
                            continue # Bỏ qua sản phẩm này nếu không parse được JSON
                        
                        # Kiểm tra xem product_data_raw có phải là dictionary không
                        if not isinstance(product_data_raw, dict):
                            logger.error(f"💥 Dữ liệu sản phẩm {product_id} không phải là dictionary. Nhận được: {type(product_data_raw)}")
                            logger.error(f"Data: {product_data_raw}")
                            continue # Bỏ qua sản phẩm này

                        try:
                            formatted_product = self._format_product_for_chat(product_data_raw)
                            products.append(formatted_product)
                            logger.debug(f"✅ Lấy thành công sản phẩm {product_id}: {formatted_product['name']}")
                        except Exception as format_error:
                            logger.error(f"💥 Lỗi khi format sản phẩm {product_id} (dữ liệu: {product_data_raw}): {str(format_error)}")
                            continue
                    else:
                        logger.warning(f"❌ Không thể lấy thông tin sản phẩm {product_id}: HTTP {response.status_code} - Response: {response.text}")
                except httpx.RequestError as req_err:
                    logger.error(f"💥 Lỗi request HTTP khi lấy sản phẩm {product_id}: {str(req_err)}")
                    continue
                except Exception as e:
                    logger.error(f"💥 Lỗi không xác định khi lấy sản phẩm {product_id}: {str(e)}")
                    continue
        
        logger.info(f"🎯 Hoàn thành lấy thông tin sản phẩm: {len(products)}/{len(product_ids)} thành công")
        return products
    
    def _format_product_for_chat(self, product_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format dữ liệu sản phẩm để hiển thị trong chat"""
        image_url = None
        images_raw = product_data.get('images') # Không cung cấp giá trị mặc định để có thể kiểm tra None

        if isinstance(images_raw, list) and images_raw: # Kiểm tra là list và không rỗng
            first_image_item = images_raw[0]
            if isinstance(first_image_item, dict):
                image_url = first_image_item.get('image_url')
            elif isinstance(first_image_item, str):
                image_url = first_image_item # Nếu phần tử đầu tiên là string, giả sử đó là URL
                logger.warning(f"Product ID {product_data.get('product_id')}: 'images' list contains a string instead of a dict: {first_image_item}")
            else:
                logger.warning(f"Product ID {product_data.get('product_id')}: First item in 'images' is not a dict or string: {type(first_image_item)}")
        elif images_raw is not None: # Nếu images_raw không phải list nhưng không phải None
             logger.warning(f"Product ID {product_data.get('product_id')}: 'images' field is not a list: {type(images_raw)}, data: {images_raw}")

        formatted_product = {
            'id': product_data.get('product_id'),
            'name': product_data.get('name'),
            'price': float(product_data.get('price', 0.0)), # Đảm bảo float
            'original_price': float(product_data.get('original_price', 0.0)), # Đảm bảo float
            'description': product_data.get('description'),
            'image': image_url,
            'unit': product_data.get('unit'),
            'stock_quantity': product_data.get('stock_quantity', 0),
            'category_id': product_data.get('category_id')
        }
        
        logger.debug(f"📦 Formatted product: {formatted_product.get('name')} - {formatted_product.get('price')}đ")
        return formatted_product
    
    async def get_available_products_from_menu_items(self, menu_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Lấy danh sách sản phẩm có sẵn từ menu items.
        Chỉ lấy những sản phẩm có product_id và có trong kho.
        
        Args:
            menu_items: Danh sách các nguyên liệu với product_id
            
        Returns:
            List[Dict]: Danh sách sản phẩm có sẵn trong kho
        """
        if not menu_items:
            logger.debug("Danh sách menu items trống")
            return []
        
        logger.info(f"🔍 Bắt đầu tìm sản phẩm có sẵn từ {len(menu_items)} menu items")
        
        # Lọc ra các product_id có giá trị
        product_ids = []
        for item in menu_items:
            product_id = item.get('product_id')
            ingredient_name = item.get('ingredient_name', 'Unknown')
            
            if product_id:
                product_ids.append(product_id)
                logger.debug(f"📝 Found product_id={product_id} for ingredient: {ingredient_name}")
            else:
                logger.debug(f"❓ No product_id for ingredient: {ingredient_name}")
        
        if not product_ids:
            logger.info("⚠️ Không có product_id nào trong menu items")
            return []
        
        logger.info(f"🎯 Tìm thấy {len(product_ids)} product_ids để lấy thông tin: {product_ids}")
        
        # Lấy thông tin sản phẩm từ backend
        products = await self.get_products_by_ids(product_ids)
        
        # Lọc ra những sản phẩm có trong kho (stock_quantity > 0)
        available_products = []
        for product in products:
            stock_quantity = product.get('stock_quantity', 0)
            if stock_quantity > 0:
                available_products.append(product)
                logger.debug(f"✅ Product có sẵn: {product['name']} (Stock: {stock_quantity})")
            else:
                logger.debug(f"❌ Product hết hàng: {product['name']} (Stock: {stock_quantity})")
        
        logger.info(f"🛒 Kết quả cuối cùng: {len(available_products)} sản phẩm có sẵn trong kho")
        
        return available_products
    
    async def check_product_availability(self, product_id: int) -> bool:
        """Kiểm tra xem sản phẩm có còn hàng hay không"""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.backend_url}/api/e-commerce/products/{product_id}"
                )
                if response.status_code == 200:
                    product_data = response.json()
                    stock_quantity = product_data.get('stock_quantity', 0)
                    return stock_quantity > 0
                else:
                    logger.warning(f"Không thể kiểm tra tồn kho cho sản phẩm {product_id}")
                    return False
        except Exception as e:
            logger.error(f"Lỗi khi kiểm tra tồn kho sản phẩm {product_id}: {str(e)}")
            return False 