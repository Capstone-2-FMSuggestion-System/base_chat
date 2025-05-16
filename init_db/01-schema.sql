-- Tạo database nếu chưa tồn tại và đặt rõ ràng charset
CREATE DATABASE IF NOT EXISTS family_menu_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE family_menu_db;

-- Tạo bảng users
CREATE TABLE IF NOT EXISTS users (
    user_id INT AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    password VARCHAR(255) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    full_name VARCHAR(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci,
    avatar_url VARCHAR(255),
    preferences JSON,
    location VARCHAR(100) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci,
    role VARCHAR(20) DEFAULT 'user',
    status VARCHAR(20) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Tạo bảng conversations
CREATE TABLE IF NOT EXISTS conversations (
    conversation_id INT AUTO_INCREMENT PRIMARY KEY,
    title VARCHAR(255) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    user_id INT,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;

-- Tạo bảng messages
CREATE TABLE IF NOT EXISTS messages (
    message_id INT AUTO_INCREMENT PRIMARY KEY,
    conversation_id INT,
    role VARCHAR(20) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci, -- Quan trọng cho vai trò user/assistant
    content TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci, -- Rất quan trọng cho nội dung chat
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_summarized BOOLEAN DEFAULT FALSE,
    summary TEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci, -- Quan trọng cho nội dung tóm tắt
    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci; 