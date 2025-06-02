HƯỚNG DẪN CÀI ĐẶT HỆ THỐNG FAMILY MENU SUGGESTION SYSTEM
===============================================================

MÔ TẢ TỔNG QUAN
Hệ thống gồm 3 thành phần chính:
- Backend (FastAPI): API chính cho ứng dụng với MySQL database
- Frontend (React): Giao diện người dùng  
- Base_chat (FastAPI): Service chat AI với LLM và vector database

YÊU CẦU HỆ THỐNG

1. Software Requirements
- Python 3.11+ (khuyến nghị 3.10+)
- Node.js 16+ và npm/yarn
- MySQL 8.0+
- Redis 6.0+
- Git

2. Hardware Requirements  
- RAM: Tối thiểu 8GB (khuyến nghị 16GB cho LLM)
- Storage: 10GB trống
- CPU: 4 cores trở lên

3. Optional AI Tools (cho base_chat)
- Ollama (để chạy local LLM)
- llama.cpp server
- Gemini API key

CÀI ĐẶT CHI TIẾT

BƯỚC 1: CHUẨN BỊ MÔITORTRƯỜNG

1.1 Clone repositories
git clone <repository-url>
cd Capstone-2-FMSuggestion-System

1.2 Cài đặt MySQL và Redis
Windows:
- Tải và cài MySQL 8.0 từ mysql.com
- Tải và cài Redis for Windows hoặc dùng WSL
- Khởi động các services

Linux/Mac:
MySQL:
sudo apt install mysql-server  # Ubuntu/Debian
brew install mysql            # macOS

Redis:
sudo apt install redis-server # Ubuntu/Debian  
brew install redis           # macOS

1.3 Tạo database
CREATE DATABASE family_menu_db;
CREATE USER 'family_user'@'localhost' IDENTIFIED BY 'your_password';
GRANT ALL PRIVILEGES ON family_menu_db.* TO 'family_user'@'localhost';
FLUSH PRIVILEGES;

BƯỚC 2: CÀI ĐẶT BACKEND

2.1 Setup Python environment
cd backend
python -m venv venv

Windows:
venv\Scripts\activate

Linux/Mac:  
source venv/bin/activate

2.2 Cài đặt dependencies
pip install -r requirements.txt

2.3 Cấu hình environment
cp .env.example .env
# Chỉnh sửa file .env với thông tin thực tế:

File backend/.env cần thiết:

Database Configuration
DB_USER=family_user
DB_PASSWORD=your_password
DB_HOST=localhost
DB_PORT=3306
DB_NAME=family_menu_db

App Configuration  
SECRET_KEY=generate_random_secret_key_here
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
DEBUG=True

Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=

Server Configuration
HOST=0.0.0.0
PORT=8000

Cloudinary Configuration (cho upload ảnh)
CLOUDINARY_CLOUD_NAME=your_cloud_name
CLOUDINARY_API_KEY=your_api_key
CLOUDINARY_API_SECRET=your_api_secret

Payment Configuration (PayOS)
PAYOS_CLIENT_ID=your-payos-client-id
PAYOS_API_KEY=your-payos-api-key
PAYOS_CHECKSUM_KEY=your-payos-checksum-key
FRONTEND_URL=http://localhost:3000

2.4 Tạo secret key
python generate_secret_key.py

2.5 Chạy database migrations
python migrate.py
hoặc
alembic upgrade head

2.6 Khởi động backend server
python run.py
hoặc
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

Server sẽ chạy tại: http://localhost:8000

BƯỚC 3: CÀI ĐẶT FRONTEND

3.1 Setup Node.js environment
cd frontend
npm install
hoặc
yarn install

3.2 Cấu hình environment  
File frontend/.env đã có sẵn, kiểm tra và điều chỉnh:

File frontend/.env:

REACT_APP_API_URL=http://localhost:8000
REACT_APP_BASE_CHAT_API_URL=http://localhost:8002/api

Cloudinary Configuration (phải khớp với backend)
REACT_APP_CLOUDINARY_CLOUD_NAME=your_cloud_name
REACT_APP_CLOUDINARY_API_KEY=your_api_key
REACT_APP_CLOUDINARY_UPLOAD_PRESET=your_upload_preset
REACT_APP_CLOUDINARY_FOLDER=fm_products

3.3 Khởi động frontend server
npm start
hoặc  
yarn start

Ứng dụng sẽ chạy tại: http://localhost:3000

BƯỚC 4: CÀI ĐẶT BASE_CHAT (AI CHAT SERVICE)

4.1 Setup Python environment
cd base_chat
python -m venv venv

Windows:
venv\Scripts\activate

Linux/Mac:
source venv/bin/activate

4.2 Cài đặt dependencies
pip install -r requirements.txt

4.3 Cài đặt AI Tools (Tùy chọn)

Option A: Ollama (Local LLM - Khuyến nghị)
Tải từ ollama.ai
Windows: Tải installer
Linux/Mac:
curl -fsSL https://ollama.ai/install.sh | sh

Pull model:
ollama pull llama3.1:8b
hoặc model khác tùy theo phần cứng

Option B: llama.cpp (Alternative)
Build llama.cpp từ source:
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp
make
Tải GGUF model và chạy server

Option C: Gemini API (Cloud)
- Đăng ký tại Google AI Studio
- Lấy API key

4.4 Cấu hình environment
cp .env.example .env
Chỉnh sửa file .env:

File base_chat/.env cần thiết:

Database (dùng chung với backend)
DB_USER=family_user
DB_PASSWORD=your_password
DB_HOST=localhost
DB_PORT=3306
DB_NAME=family_menu_db

Redis
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0
REDIS_PASSWORD=

Security
SECRET_KEY=same_as_backend_secret_key
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=60

LLM Service Configuration
LLM_SERVICE_TYPE=auto
OLLAMA_URL=http://localhost:11434
LLAMA_CPP_URL=http://localhost:8080

API Configuration
API_HOST=0.0.0.0
API_PORT=8002
DEBUG_MODE=True

Gemini API (nếu sử dụng)
GEMINI_API_KEY=your_gemini_api_key
GEMINI_API_KEYS_LIST=key1,key2,key3  # multiple keys cho load balancing

Vector Database (Pinecone)
PRODUCT_DB_PINECONE_API_KEY=your_pinecone_api_key
RECIPE_DB_PINECONE_API_KEY=your_pinecone_api_key  
PINECONE_INDEX_NAME=product-index
PINECONE_ENVIRONMENT=gcp-starter

Backend Integration
BACKEND_AUTH_VERIFY_URL=http://localhost:8000/api/auth/verify-token

4.5 Khởi tạo database
python init_db.py

4.6 Khởi động chat service
python main.py

Chat service sẽ chạy tại: http://localhost:8002

Frontend build:
cd frontend  
npm run build
Deploy build folder lên web server

BƯỚC 5: KIỂM TRA VÀ XÁC THỰC

5.1 Kiểm tra Backend
curl http://localhost:8000/docs  # Swagger UI
curl http://localhost:8000/health

5.2 Kiểm tra Frontend
- Truy cập http://localhost:3000
- Đăng ký/đăng nhập account
- Test các chức năng cơ bản

5.3 Kiểm tra Chat Service
curl http://localhost:8002/
curl http://localhost:8002/api/llm/status

BƯỚC 6: TROUBLESHOOTING

6.1 Lỗi thường gặp

Database connection:
Kiểm tra MySQL running:
sudo systemctl status mysql

Test connection:
mysql -u family_user -p -h localhost family_menu_db

Redis connection:
Kiểm tra Redis running:  
redis-cli ping

Python dependencies:
Nếu lỗi import, cài lại:
pip install --force-reinstall -r requirements.txt

Node.js dependencies:
Clear cache và reinstall:
rm -rf node_modules package-lock.json
npm install

AI Model issues:
Kiểm tra Ollama:
ollama list
ollama run llama3.1:8b

Test local model:
curl http://localhost:11434/api/generate -d '{
  "model": "llama3.1:8b",
  "prompt": "Hello world"
}'

6.2 Performance tuning

Backend:
- Điều chỉnh connection pool size trong SQLAlchemy
- Enable Redis caching
- Tối ưu database indexes

Frontend:
- Build production: npm run build
- Enable service worker
- Optimize bundle size

Chat Service:
- Sử dụng GPU cho LLM nếu có
- Điều chỉnh model size theo RAM
- Enable model caching

BƯỚC 7: PRODUCTION DEPLOYMENT

7.1 Environment setup
Set production environment variables:
DEBUG=False
CORS_ORIGINS=["https://yourdomain.com"]
DATABASE_URL=mysql://user:pass@prod-host/db

7.2 Security checklist
- Change all default passwords
- Use HTTPS certificates
- Set up firewall rules
- Enable rate limiting
- Configure backup strategy
- Set up monitoring/logging

7.3 Monitoring
- Database performance monitoring
- API response time tracking  
- Error logging và alerting
- Resource usage monitoring

SUPPORT VÀ LIÊN HỆ

Documentation:
- Backend API: http://localhost:8000/docs
- Frontend: Xem src/README.md
- Chat API: http://localhost:8002/docs

================== HẾT HƯỚNG DẪN ==================
