#!/usr/bin/env python3
import os
import subprocess
import argparse

def run_command(command):
    """Chạy lệnh shell và hiển thị output"""
    print(f"Đang chạy: {command}")
    try:
        result = subprocess.run(command, shell=True, check=True, text=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Lỗi: {e}")
        return False

def build_docker():
    """Build Docker image"""
    return run_command("docker-compose build")

def start_containers():
    """Khởi động tất cả containers"""
    return run_command("docker-compose up -d")

def stop_containers():
    """Dừng tất cả containers"""
    return run_command("docker-compose down")

def show_logs(service=None):
    """Hiển thị logs"""
    command = "docker-compose logs"
    if service:
        command += f" {service}"
    return run_command(command)

def follow_logs(service=None):
    """Theo dõi logs liên tục"""
    command = "docker-compose logs -f"
    if service:
        command += f" {service}"
    return run_command(command)

def check_docker():
    """Kiểm tra xem Docker và Docker Compose đã được cài đặt chưa"""
    docker_ok = run_command("docker --version")
    compose_ok = run_command("docker-compose --version")
    return docker_ok and compose_ok

def main():
    """Hàm chính"""
    parser = argparse.ArgumentParser(description="Công cụ quản lý Docker cho ứng dụng Medical Chat")
    parser.add_argument("action", choices=["build", "start", "stop", "logs", "follow", "check"], 
                        help="Hành động cần thực hiện")
    parser.add_argument("--service", "-s", help="Dịch vụ cụ thể để thao tác (app, mysql, redis)")
    
    # Xử lý trường hợp service được truyền như tham số vị trí
    args, unknown = parser.parse_known_args()
    if unknown and args.action in ["logs", "follow"]:
        args.service = unknown[0]
    
    if args.action == "check":
        if check_docker():
            print("Docker và Docker Compose đã được cài đặt đúng cách.")
        else:
            print("Docker hoặc Docker Compose chưa được cài đặt hoặc cài đặt không đúng.")
    elif args.action == "build":
        build_docker()
    elif args.action == "start":
        start_containers()
    elif args.action == "stop":
        stop_containers()
    elif args.action == "logs":
        show_logs(args.service)
    elif args.action == "follow":
        follow_logs(args.service)

if __name__ == "__main__":
    main() 