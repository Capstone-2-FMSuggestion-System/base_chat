-- Tạo người dùng admin mẫu
-- Mật khẩu đã hash với bcrypt: "admin123"
INSERT INTO users (username, password, email, full_name, role)
VALUES ('admin', '$2b$12$EixZaYVK1fsbw1ZfbX3OXePaWxn96p36WQoeG6Lruj3vjPGga31lW', 'admin@example.com', 'Admin User', 'admin')
ON DUPLICATE KEY UPDATE
    username = VALUES(username),
    email = VALUES(email);

-- Tạo người dùng bác sĩ mẫu
-- Mật khẩu đã hash với bcrypt: "doctor123"
INSERT INTO users (username, password, email, full_name, role, location)
VALUES ('doctor', '$2b$12$51tSr5wd1.q8O3ToSxwX8.XtUVOdbq4RUbQ1R50u7R60nV.UQ3iCC', 'doctor@example.com', 'Bác Sĩ Nguyễn', 'doctor', 'Hà Nội')
ON DUPLICATE KEY UPDATE
    username = VALUES(username),
    email = VALUES(email);

-- Tạo người dùng bệnh nhân mẫu
-- Mật khẩu đã hash với bcrypt: "patient123"
INSERT INTO users (username, password, email, full_name, role, location)
VALUES ('patient', '$2b$12$9NqPnhI/GZnfLT5jG/Ulu.F21rWpCPdXlqJ4BG3bZ1b2WwfTgvIGe', 'patient@example.com', 'Nguyễn Văn A', 'user', 'Hồ Chí Minh')
ON DUPLICATE KEY UPDATE
    username = VALUES(username),
    email = VALUES(email); 