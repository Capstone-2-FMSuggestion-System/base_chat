from sqlalchemy import Column, Integer, String, Text, ForeignKey, TIMESTAMP, JSON, Boolean, DECIMAL
from sqlalchemy.sql import text
from app.db.database import Base


class User(Base):
    __tablename__ = "users"
    
    user_id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, nullable=False)
    password = Column(String(255), nullable=False)
    email = Column(String(100), unique=True, nullable=False)
    full_name = Column(String(100))
    avatar_url = Column(String(255))
    preferences = Column(JSON)
    location = Column(String(100))
    role = Column(String(20), default="user")
    status = Column(String(20), default="active")
    created_at = Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"))


class Conversation(Base):
    __tablename__ = "conversations"
    
    conversation_id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=True)
    created_at = Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"))
    updated_at = Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"), onupdate=text("CURRENT_TIMESTAMP"))
    user_id = Column(Integer, ForeignKey("users.user_id"))


class Message(Base):
    __tablename__ = "messages"
    
    message_id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.conversation_id"))
    role = Column(String(20))  # "user" hoặc "assistant" hoặc "system"
    content = Column(Text)
    created_at = Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP")) 
    is_summarized = Column(Boolean, default=False)  # Đánh dấu tin nhắn đã được tóm tắt hay chưa
    summary = Column(Text, nullable=True)  # Tóm tắt nội dung tin nhắn 


class HealthData(Base):
    __tablename__ = "health_data"
    
    data_id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.conversation_id"))
    user_id = Column(Integer, ForeignKey("users.user_id"))
    
    # Các thông tin sức khỏe
    health_condition = Column(Text, nullable=True)  # Tình trạng sức khỏe hiện tại
    medical_history = Column(Text, nullable=True)  # Bệnh lý đã biết
    allergies = Column(Text, nullable=True)  # Dị ứng
    dietary_habits = Column(Text, nullable=True)  # Thói quen ăn uống
    health_goals = Column(Text, nullable=True)  # Mục tiêu sức khỏe
    
    # Thông tin bổ sung
    additional_info = Column(JSON, nullable=True)  # Thông tin bổ sung dạng JSON
    created_at = Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"))
    updated_at = Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"), onupdate=text("CURRENT_TIMESTAMP"))
    
    # Dánh dấu trạng thái
    is_processed = Column(Boolean, default=False)  # Đánh dấu dữ liệu đã được xử lý chưa


class Menu(Base):
    __tablename__ = "menus"
    
    menu_id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.conversation_id"), nullable=True)  # Liên kết với conversation
    name = Column(String(100), nullable=False)  # Tên công thức món ăn
    description = Column(String(500), nullable=True)  # Mô tả công thức
    created_at = Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP"))


class MenuItem(Base):
    __tablename__ = "menu_items"
    
    menu_item_id = Column(Integer, primary_key=True, index=True)
    menu_id = Column(Integer, ForeignKey("menus.menu_id"), nullable=False)
    product_id = Column(Integer, nullable=True)  # Product ID từ product database
    quantity = Column(Integer, nullable=False, default=1)  # Số lượng nguyên liệu cần


class Category(Base):
    __tablename__ = "categories"
    
    category_id = Column(Integer, primary_key=True, index=True)
    parent_id = Column(Integer, ForeignKey("categories.category_id"), nullable=True)
    name = Column(String(50), nullable=False)
    description = Column(String(500), nullable=True)
    level = Column(Integer, nullable=False)


class Product(Base):
    __tablename__ = "products"
    
    product_id = Column(Integer, primary_key=True, index=True)
    category_id = Column(Integer, ForeignKey("categories.category_id"), nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(String(1000), nullable=True)
    price = Column(DECIMAL(10, 2), nullable=False)
    original_price = Column(DECIMAL(10, 2), nullable=False)
    unit = Column(String(20), nullable=True)
    stock_quantity = Column(Integer, default=0)
    is_featured = Column(Boolean, default=False)
    created_at = Column(TIMESTAMP, server_default=text("CURRENT_TIMESTAMP")) 