FROM python:3.11-slim

WORKDIR /app

# Cài đặt các gói hệ thống cần thiết
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    default-mysql-client \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt các dependencies Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Sao chép script đợi MySQL và thêm quyền thực thi
COPY wait-for-mysql.sh .
RUN chmod +x wait-for-mysql.sh

# Sao chép file .env.docker
COPY .env.docker ./.env

# Sao chép mã nguồn vào container
COPY . .

# Expose cổng mà ứng dụng sẽ chạy
EXPOSE 8000

# Khởi động ứng dụng khi container chạy
CMD ["./wait-for-mysql.sh", "mysql", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] 