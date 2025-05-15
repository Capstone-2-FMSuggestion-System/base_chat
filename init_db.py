import mysql.connector
import sys
import os
from dotenv import load_dotenv
from passlib.context import CryptContext

# Load biến môi trường từ .env
load_dotenv()

# Lấy thông tin kết nối DB từ biến môi trường
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME", "family_menu_db")

# Mã hóa mật khẩu
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def init_database():
    # Kết nối tới MySQL server (không chọn database)
    try:
        print("Đang kết nối đến MySQL server...")
        connection = mysql.connector.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            port=int(DB_PORT)
        )
        
        cursor = connection.cursor()
        
        # Kiểm tra xem database đã tồn tại chưa
        cursor.execute(f"SHOW DATABASES LIKE '{DB_NAME}'")
        database_exists = cursor.fetchone()
        
        if not database_exists:
            print(f"Đang tạo database {DB_NAME}...")
            cursor.execute(f"CREATE DATABASE {DB_NAME}")
            print(f"Database {DB_NAME} đã được tạo thành công!")
        else:
            print(f"Database {DB_NAME} đã tồn tại.")
        
        # Chọn database
        cursor.execute(f"USE {DB_NAME}")
        
        # Tạo bảng users nếu chưa tồn tại
        print("Đang kiểm tra và tạo bảng users...")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INT AUTO_INCREMENT PRIMARY KEY,
            username VARCHAR(50) UNIQUE NOT NULL,
            password VARCHAR(255) NOT NULL,
            email VARCHAR(100) UNIQUE NOT NULL,
            full_name VARCHAR(100),
            avatar_url VARCHAR(255),
            preferences JSON,
            location VARCHAR(100),
            role VARCHAR(20) DEFAULT 'user',
            status VARCHAR(20) DEFAULT 'active',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Tạo bảng conversations
        print("Đang kiểm tra và tạo bảng conversations...")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS conversations (
            conversation_id INT AUTO_INCREMENT PRIMARY KEY,
            title VARCHAR(255),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            user_id INT,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
        """)
        
        # Tạo bảng messages
        print("Đang kiểm tra và tạo bảng messages...")
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            message_id INT AUTO_INCREMENT PRIMARY KEY,
            conversation_id INT,
            role VARCHAR(20),
            content TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
        )
        """)
        
        # Kiểm tra xem có người dùng admin chưa
        cursor.execute("SELECT COUNT(*) FROM users WHERE user_id = 1")
        admin_exists = cursor.fetchone()[0]
        
        if not admin_exists:
            # Tạo người dùng admin với user_id = 1
            print("Đang tạo người dùng admin với user_id = 1...")
            hashed_password = pwd_context.hash("admin123")
            
            # Xóa tất cả dữ liệu hiện có trong bảng users (chỉ dành cho mục đích test)
            cursor.execute("DELETE FROM users")
            
            # Reset auto_increment để đảm bảo user_id = 1
            cursor.execute("ALTER TABLE users AUTO_INCREMENT = 1")
            
            cursor.execute("""
            INSERT INTO users (username, password, email, full_name, role)
            VALUES (%s, %s, %s, %s, %s)
            """, ("admin", hashed_password, "admin@example.com", "Admin User", "admin"))
            
            print("Người dùng admin đã được tạo thành công với user_id = 1!")
        else:
            print("Người dùng admin với user_id = 1 đã tồn tại.")
        
        connection.commit()
        print("Khởi tạo cơ sở dữ liệu thành công!")
        
    except mysql.connector.Error as error:
        print(f"Lỗi khi kết nối hoặc khởi tạo cơ sở dữ liệu: {error}")
        sys.exit(1)
    finally:
        if 'connection' in locals() and connection.is_connected():
            cursor.close()
            connection.close()
            print("Đã đóng kết nối MySQL.")

if __name__ == "__main__":
    print("Bắt đầu khởi tạo cơ sở dữ liệu...")
    init_database()
    print("Hoàn thành!") 