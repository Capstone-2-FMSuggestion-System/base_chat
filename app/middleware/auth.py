from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import User
from app.config import settings
import httpx
from pydantic import BaseModel
import logging

# Thiết lập logger
logger = logging.getLogger(__name__)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="http://localhost:8000/api/auth/login")


class VerifiedUserInfo(BaseModel):
    user_id: int
    username: str
    role: str
    email: Optional[str] = None


async def get_verified_user_from_backend(token: str = Depends(oauth2_scheme)) -> VerifiedUserInfo:
    """
    Xác thực token bằng cách gọi API của backend và trả về thông tin người dùng.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Không thể xác thực thông tin đăng nhập. Token không hợp lệ hoặc đã hết hạn.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    service_unavailable_exception = HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Dịch vụ xác thực tạm thời không khả dụng. Vui lòng thử lại sau.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    internal_server_error_exception = HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Lỗi máy chủ nội bộ trong quá trình xác thực.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if not token:
        logger.warning("Token không được cung cấp")
        raise credentials_exception

    async with httpx.AsyncClient(timeout=settings.API_TIMEOUT) as client:
        try:
            logger.debug(f"Gửi yêu cầu xác thực đến {settings.BACKEND_AUTH_VERIFY_URL}")
            response = await client.get(
                settings.BACKEND_AUTH_VERIFY_URL,
                headers={"Authorization": f"Bearer {token}"}
            )
            
            # Kiểm tra lỗi xác thực rõ ràng
            if response.status_code == 401:
                logger.warning("Lỗi xác thực: Token không hợp lệ hoặc đã hết hạn")
                raise credentials_exception
            
            if response.status_code == 403:
                logger.warning("Lỗi xác thực: Không có quyền truy cập")
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Không có quyền truy cập vào tài nguyên này.",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            # Kiểm tra lỗi khác
            if response.status_code >= 400:
                logger.error(f"Lỗi từ backend: {response.status_code} - {response.text}")
                raise service_unavailable_exception
            
            user_data = response.json()
            
            if not isinstance(user_data, dict) or not all(key in user_data for key in ["user_id", "username", "role"]):
                logger.error(f"Backend auth response có cấu trúc không đúng: {user_data}")
                raise internal_server_error_exception
            
            logger.debug(f"Xác thực thành công cho người dùng: {user_data['username']}")
            return VerifiedUserInfo(**user_data)
            
        except httpx.HTTPStatusError as exc:
            error_msg = f"Backend auth verification failed: {exc.response.status_code} - {exc.response.text}"
            logger.error(error_msg)
            
            if exc.response.status_code == 401:
                raise credentials_exception
            elif exc.response.status_code == 403:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Không có quyền truy cập vào tài nguyên này.",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            else:
                raise service_unavailable_exception
            
        except httpx.RequestError as exc:
            error_msg = f"Error connecting to backend auth service: {str(exc)}"
            logger.error(error_msg)
            raise service_unavailable_exception
            
        except Exception as e:
            error_msg = f"An unexpected error occurred during backend auth verification: {str(e)}"
            logger.exception(error_msg)
            # Lỗi không mong muốn sẽ trả về 401 thay vì 500 để tránh lộ thông tin lỗi nội bộ
            # và giúp client biết họ cần đăng nhập lại
            raise credentials_exception
