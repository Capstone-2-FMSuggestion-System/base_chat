#!/usr/bin/env python3
"""
Debug Unicode encoding issue
"""
import json
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

def decode_unicode_string(encoded_str):
    """Decode Unicode escape sequences trong string"""
    print(f"🔍 Decoding: {encoded_str[:100]}...")
    
    try:
        # Method 1: JSON decode
        try:
            decoded_json = json.loads(f'"{encoded_str}"')
            print(f"✅ JSON decode: {decoded_json[:100]}...")
            return decoded_json
        except:
            pass
        
        # Method 2: Unicode escape decode
        try:
            decoded_unicode = encoded_str.encode('utf-8').decode('unicode_escape')
            print(f"✅ Unicode escape decode: {decoded_unicode[:100]}...")
            return decoded_unicode
        except:
            pass
            
        # Method 3: Encode/decode
        try:
            decoded_encode = encoded_str.encode().decode('unicode_escape')
            print(f"✅ Encode/decode: {decoded_encode[:100]}...")
            return decoded_encode
        except:
            pass
        
        print("❌ Không thể decode")
        return encoded_str
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return encoded_str

def test_unicode_cases():
    """Test các trường hợp Unicode khác nhau"""
    print("=== TEST UNICODE CASES ===")
    
    test_cases = [
        "Ch\\u00e0o b\\u1ea1n! R\\u1ea5t vui \\u0111\\u01b0\\u1ee3c g\\u1eb7p b\\u1ea1n.",
        "T\\u00f4i l\\u00e0 tr\\u1ee3 l\\u00fd t\\u01b0 v\\u1ea5n s\\u1ee9c kh\\u1ecfe",
        "Chào bạn! Rất vui được gặp bạn.",  # Normal Vietnamese
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n🧪 Test case {i}:")
        decode_unicode_string(case)

def find_encoding_source():
    """Tìm nguồn gốc encoding"""
    print("\n=== TÌM NGUỒN GỐC ENCODING ===")
    
    # Test JSON với tiếng Việt
    test_data = {
        "message": "Chào bạn! Rất vui được gặp bạn.",
        "assistant": "Tôi là trợ lý tư vấn sức khỏe và dinh dưỡng"
    }
    
    print("📝 Original data:")
    print(test_data)
    
    print("\n📝 JSON với ensure_ascii=True:")
    json_true = json.dumps(test_data, ensure_ascii=True, indent=2)
    print(json_true)
    
    print("\n📝 JSON với ensure_ascii=False:")
    json_false = json.dumps(test_data, ensure_ascii=False, indent=2)
    print(json_false)
    
    # Test load lại
    print("\n📝 Load lại JSON (ensure_ascii=True):")
    loaded_true = json.loads(json_true)
    print(loaded_true)
    
    print("\n📝 Load lại JSON (ensure_ascii=False):")
    loaded_false = json.loads(json_false)
    print(loaded_false)

if __name__ == "__main__":
    print("🐛 DEBUG UNICODE ENCODING ISSUE\n")
    
    # Decode the problematic string
    problematic_string = "Ch\\u00e0o b\\u1ea1n! R\\u1ea5t vui \\u0111\\u01b0\\u1ee3c g\\u1eb7p b\\u1ea1n. T\\u00f4i l\\u00e0 tr\\u1ee3 l\\u00fd t\\u01b0 v\\u1ea5n s\\u1ee9c kh\\u1ecfe v\\u00e0 dinh d\\u01b0\\u1ee1ng, c\\u00f3 th\\u1ec3 gi\\u00fap b\\u1ea1n t\\u00ecm hi\\u1ec3u v\\u1ec1 c\\u00e1c m\\u00f3n \\u0103n ph\\u00f9 h\\u1ee3p v\\u1edbi t\\u00ecnh tr\\u1ea1ng s\\u1ee9c kh\\u1ecfe, dinh d\\u01b0\\u1ee1ng v\\u00e0 th\\u00f3i quen \\u0103n u\\u1ed1ng c\\u1ee7a b\\u1ea1n. B\\u1ea1n c\\u00f3 th\\u1ec3 chia s\\u1ebb v\\u1ec1 t\\u00ecnh tr\\u1ea1ng s\\u1ee9c kh\\u1ecfe ho\\u1eb7c m\\u1ee5c ti\\u00eau dinh d\\u01b0\\u1ee1ng c\\u1ee7a m\\u00ecnh \\u0111\\u1ec3 t\\u00f4i h\\u1ed7 tr\\u1ee3 b\\u1ea1n nh\\u00e9!"
    
    print("🎯 DECODING PROBLEMATIC STRING:")
    decoded = decode_unicode_string(problematic_string)
    print(f"\n✨ Final result:\n{decoded}")
    
    test_unicode_cases()
    find_encoding_source() 