# FMChat - Hỗ Trợ Y Tế

Hệ thống chatbot hỗ trợ y tế sử dụng mô hình ngôn ngữ lớn (LLM) được tinh chỉnh cho các tri thức y khoa, tích hợp khả năng tóm tắt lịch sử trò chuyện bằng Gemini API.

## Tổng Quan

FMChat là một hệ thống chatbot y tế thông minh giúp người dùng tìm kiếm thông tin y tế, tư vấn sơ bộ về các vấn đề sức khỏe, và cung cấp hướng dẫn chung về các loại bệnh và phương pháp điều trị. Hệ thống này sử dụng các mô hình ngôn ngữ lớn (LLM) chuyên về y tế để đưa ra thông tin đáng tin cậy.

**Lưu ý quan trọng**: Hệ thống này chỉ cung cấp thông tin tham khảo và không thay thế cho tư vấn y tế chuyên nghiệp.

## Các tính năng

- Tư vấn y tế sơ bộ dựa trên triệu chứng.
- Trả lời câu hỏi về các loại bệnh, thuốc và phương pháp điều trị.
- Hỗ trợ nhiều cuộc trò chuyện đồng thời.
- Lưu trữ lịch sử trò chuyện.
- Tự động tóm tắt lịch sử trò chuyện dài để tối ưu hóa ngữ cảnh cho LLM.

## Công nghệ sử dụng

- **Backend**: FastAPI, Python 3.11+
- **Cơ sở dữ liệu**: MySQL
- **ORM**: SQLAlchemy
- **Deployment**: Docker, Docker Compose
- **LLM Service**: Hỗ trợ hai hệ thống LLM:
  - **llama.cpp**: Triển khai mô hình trên CPU/GPU thông qua llama.cpp.
  - **Ollama**: Triển khai thay thế, dễ dàng hơn trong việc cài đặt và sử dụng.
- **Model LLM**: Tương thích với các mô hình như MediChat-Llama3 (8B, 70B) hoặc các mô hình khác hỗ trợ bởi llama.cpp/Ollama.
- **Dịch vụ tóm tắt**: Google Gemini API (ví dụ: `gemini-2.0-flash-lite`).

## Quickstart

### 1. Cài đặt với Docker (Khuyến nghị)

```bash
# Clone repository
git clone https://github.com/yourusername/fmchat.git
cd fmchat

# Cấu hình môi trường
cp .env.example .env.docker
# Chỉnh sửa file .env.docker với các thông tin cần thiết

# Khởi chạy ứng dụng
docker-compose up -d
```

### 2. Cài đặt thủ công

```bash
# Clone repository
git clone https://github.com/yourusername/fmchat.git
cd fmchat

# Tạo môi trường ảo
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
.\venv\Scripts\activate  # Windows

# Cài đặt thư viện
pip install -r requirements.txt

# Cấu hình môi trường
cp .env.example .env
# Chỉnh sửa file .env với các thông tin cần thiết

# Khởi tạo database
python init_db.py

# Khởi chạy ứng dụng
python main.py
```

## Cấu hình LLM Service và Gemini API

Hệ thống sử dụng biến môi trường để cấu hình kết nối đến các dịch vụ LLM và Gemini API:

### Biến môi trường chính

- `LLAMA_CPP_URL`: URL của llama.cpp API
- `OLLAMA_URL`: URL của Ollama API
- `LLM_SERVICE_TYPE`: Loại dịch vụ LLM (`llama_cpp`, `ollama`, hoặc `auto`)
- `GEMINI_API_KEY`: API Key của Google Gemini API
- `GEMINI_API_URL`: URL của Gemini API
- `MAX_HISTORY_MESSAGES`: Số lượng tin nhắn tối đa trong lịch sử
- `SUMMARY_THRESHOLD`: Ngưỡng số tin nhắn để kích hoạt tóm tắt

## API Endpoints

Hệ thống cung cấp các API endpoints chính sau:

### Authentication

- `POST /api/login`: Đăng nhập và lấy token
- `POST /api/register`: Đăng ký tài khoản mới

### Chat

- `POST /api/chat/`: Gửi tin nhắn và nhận phản hồi
- `POST /api/chat/stream-chat`: Gửi tin nhắn và nhận phản hồi theo stream
- `GET /api/chat/conversations`: Lấy danh sách cuộc trò chuyện
- `GET /api/chat/conversations/{conversation_id}`: Lấy chi tiết cuộc trò chuyện

### System

- `GET /`: Kiểm tra trạng thái hệ thống
- `GET /api/llm/status`: Kiểm tra trạng thái dịch vụ LLM

## Tài liệu bổ sung

Xem thêm các tài liệu chi tiết trong thư mục `docs`:

- [Tổng quan hệ thống](docs/SYSTEM_OVERVIEW.md): Mô tả chi tiết về các chức năng và luồng thực thi
- [Hướng dẫn sử dụng LLM Services](docs/LLM_SERVICES_GUIDE.md): Cách cấu hình và sử dụng các dịch vụ LLM

## Đóng góp

Nếu bạn muốn đóng góp cho dự án, vui lòng xem xét các quy tắc đóng góp và gửi Pull Request.

## Giấy phép

Dự án này được phân phối dưới giấy phép MIT. Xem file `LICENSE` để biết thêm chi tiết.