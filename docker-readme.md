# Hướng dẫn sử dụng Docker

## Cài đặt yêu cầu

1. [Docker](https://www.docker.com/products/docker-desktop/)
2. [Docker Compose](https://docs.docker.com/compose/install/) (thường được cài cùng Docker Desktop)

## Biến môi trường

Hệ thống sử dụng biến môi trường từ hai nguồn:

1. **File .env.docker** (mặc định trong container)
2. **Biến môi trường trong docker-compose.yml**

Trong đó, các biến trong docker-compose.yml sẽ ghi đè lên các biến trong .env.docker.

Bạn cần tạo một file `.env` ở thư mục gốc với ít nhất khóa API của Google AI:

```
# Google AI API
GOOGLE_AI_API_KEY=your_google_api_key_here
```

Sau đó, biến môi trường này sẽ được truyền vào container thông qua docker-compose.

## Khởi động ứng dụng

### Chạy trong môi trường development

```bash
docker-compose up -d
```

Ứng dụng sẽ khởi động và chạy ở background. Bạn có thể truy cập API tại địa chỉ [http://localhost:8000](http://localhost:8000).

### Xem logs

```bash
docker-compose logs -f app
```

### Dừng ứng dụng

```bash
docker-compose down
```

## Các cổng dịch vụ

- **API FastAPI**: `localhost:8000`
- **MySQL**: `localhost:3307` (port được map từ 3306 bên trong container)
- **Redis**: `localhost:6380` (port được map từ 6379 bên trong container)

## Cách truy cập vào container

### Truy cập container ứng dụng chính

```bash
docker exec -it medical_chat_app bash
```

### Truy cập container MySQL

```bash
docker exec -it medical_chat_mysql mysql -umedical_user -pmedical_password -Dmedical_chat_db
```

### Truy cập container Redis

```bash
docker exec -it medical_chat_redis redis-cli -a redis_password
```

## Rebuild image sau khi có thay đổi

Nếu bạn thay đổi code mà không thay đổi dependencies (trong requirements.txt), thì không cần rebuild:

```bash
docker-compose restart app
```

Nếu thay đổi dependencies hoặc Dockerfile:

```bash
docker-compose build app
docker-compose up -d
```

## Xóa tất cả dữ liệu

Để xóa hoàn toàn các containers, volumes và dữ liệu:

```bash
docker-compose down -v
```

**Lưu ý**: Lệnh trên sẽ xóa tất cả dữ liệu trong MySQL và Redis. 