from sqlalchemy import Column, Integer, String, Text, ForeignKey, TIMESTAMP, JSON, Boolean
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