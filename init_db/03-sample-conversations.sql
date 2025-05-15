-- Tạo cuộc trò chuyện mẫu cho người dùng bệnh nhân
INSERT INTO conversations (title, user_id, created_at, updated_at)
VALUES 
('Tư vấn về triệu chứng cảm cúm', 3, NOW(), NOW()),
('Câu hỏi về sức khỏe tim mạch', 3, NOW(), NOW());

-- Tạo tin nhắn mẫu cho cuộc trò chuyện 1
INSERT INTO messages (conversation_id, role, content, created_at)
VALUES
(1, 'user', 'Tôi bị đau đầu và sốt cao từ hai ngày nay.', NOW() - INTERVAL 30 MINUTE),
(1, 'assistant', 'Tôi rất tiếc khi nghe bạn không khỏe. Đau đầu kèm sốt cao có thể do nhiều nguyên nhân khác nhau như cảm cúm, viêm xoang, nhiễm trùng khác, hoặc đôi khi là các vấn đề nghiêm trọng hơn. Bạn có các triệu chứng nào khác không?', NOW() - INTERVAL 28 MINUTE),
(1, 'user', 'Tôi cũng bị đau họng và sổ mũi. Tôi nên uống thuốc gì?', NOW() - INTERVAL 26 MINUTE),
(1, 'assistant', 'Các triệu chứng của bạn nghe giống như cảm cúm hoặc nhiễm virus đường hô hấp. Để giảm các triệu chứng:\n\n1. Hạ sốt và giảm đau: Paracetamol (như Panadol, Tylenol) hoặc Ibuprofen (như Advil)\n2. Thuốc trị cảm lạnh: Có thể dùng các thuốc phối hợp có chứa thành phần giảm đau, kháng histamine và chống nghẹt mũi\n3. Xịt mũi nước muối sinh lý để giảm nghẹt mũi\n4. Húng chanh, mật ong ấm để giảm đau họng\n\nTuy nhiên, nếu sốt cao trên 39°C kéo dài hoặc triệu chứng nặng hơn sau 2-3 ngày, bạn nên đến gặp bác sĩ. Bạn đã thử biện pháp nào chưa?', NOW() - INTERVAL 24 MINUTE);

-- Cập nhật tin nhắn đầu tiên có summary
UPDATE messages 
SET is_summarized = TRUE, 
    summary = 'Người dùng bị đau đầu, sốt cao, đau họng, sổ mũi. Được tư vấn sử dụng thuốc giảm đau, hạ sốt như Paracetamol hoặc Ibuprofen, xịt mũi nước muối, uống nước ấm với mật ong. Cần đi khám nếu sốt cao kéo dài hoặc triệu chứng nặng hơn.'
WHERE conversation_id = 1 AND role = 'user' ORDER BY created_at LIMIT 1;

-- Tạo tin nhắn mẫu cho cuộc trò chuyện 2
INSERT INTO messages (conversation_id, role, content, created_at)
VALUES
(2, 'user', 'Làm sao để biết mình có nguy cơ bị bệnh tim mạch không?', NOW() - INTERVAL 10 MINUTE),
(2, 'assistant', 'Để đánh giá nguy cơ bệnh tim mạch, bạn nên quan tâm đến các yếu tố sau:\n\n1. Chỉ số sức khỏe: Huyết áp, chỉ số BMI, mức cholesterol, đường huyết\n2. Lối sống: Chế độ ăn, tập thể dục, hút thuốc, uống rượu\n3. Tiền sử gia đình: Người thân có bị bệnh tim sớm không\n4. Tuổi và giới tính: Nam giới và người trên 45 tuổi có nguy cơ cao hơn\n\nBạn nên khám sức khỏe định kỳ để đánh giá các chỉ số này. Bạn đã kiểm tra sức khỏe gần đây chưa?', NOW() - INTERVAL 8 MINUTE); 