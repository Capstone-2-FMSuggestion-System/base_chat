#!/usr/bin/env python3
"""
Test script đơn giản để kiểm tra counting logic mà không cần database dependencies
"""

def test_counting_logic():
    """Test counting logic simulation"""
    
    print("🧪 KIỂM TRA COUNTING LOGIC SIMULATION")
    print("=" * 50)
    
    # Simulate different scenarios
    test_scenarios = [
        {
            "name": "Perfect Mapping - Không duplicate",
            "ingredients": ["gạo", "thịt", "cá", "rau"],
            "found_products": ["101", "202", "303", "404"],  # All unique
            "expected": {
                "total": 4, "successful": 4, "duplicated": 0, 
                "not_found": 0, "errors": 0, "unique_products": 4
            }
        },
        {
            "name": "With Duplicates - Có trùng lặp product_id",
            "ingredients": ["gạo tẻ", "gạo thơm", "thịt heo", "thịt ba chỉ"],
            "found_products": ["101", "101", "202", "202"],  # Duplicates
            "expected": {
                "total": 4, "successful": 2, "duplicated": 2,
                "not_found": 0, "errors": 0, "unique_products": 2
            }
        },
        {
            "name": "Mixed Results - Kết hợp các trường hợp",
            "ingredients": ["gạo", "thịt", "yến sào", "caviar", "cá"],
            "found_products": ["101", "202", None, None, "101"],  # Mix of found/not found/duplicate
            "expected": {
                "total": 5, "successful": 2, "duplicated": 1,
                "not_found": 2, "errors": 0, "unique_products": 2
            }
        }
    ]
    
    all_passed = True
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n📋 SCENARIO {i}: {scenario['name']}")
        print(f"📝 Ingredients: {scenario['ingredients']}")
        print(f"🔍 Found products: {scenario['found_products']}")
        
        # Simulate logic
        used_product_ids = set()
        successful_mappings = 0
        duplicated_mappings = 0
        not_found_count = 0
        error_mappings = 0
        
        results = []
        
        for j, ingredient in enumerate(scenario['ingredients']):
            product_id = scenario['found_products'][j]
            
            result = {
                "requested_ingredient": ingredient,
                "product_id": None,
                "product_name": None,
                "status": "Không tìm thấy sản phẩm phù hợp"
            }
            
            if product_id is not None:  # Found a product
                if product_id in used_product_ids:  # Duplicate
                    result["status"] = "Sản phẩm đã được gán cho nguyên liệu khác"
                    duplicated_mappings += 1
                else:  # Successful
                    used_product_ids.add(product_id)
                    result.update({
                        "product_id": product_id,
                        "product_name": f"Product_{product_id}",
                        "status": "Đã tìm thấy sản phẩm"
                    })
                    successful_mappings += 1
            else:  # Not found
                not_found_count += 1
            
            results.append(result)
        
        # Calculate metrics
        total_processed = len(scenario['ingredients'])
        unique_products_found = len(used_product_ids)
        
        actual = {
            "total": total_processed,
            "successful": successful_mappings,
            "duplicated": duplicated_mappings,
            "not_found": not_found_count,
            "errors": error_mappings,
            "unique_products": unique_products_found
        }
        
        expected = scenario['expected']
        
        print(f"\n📊 KẾT QUẢ:")
        print(f"  Expected: {expected}")
        print(f"  Actual  : {actual}")
        
        # Validate
        scenario_passed = True
        for key in expected:
            if actual[key] != expected[key]:
                print(f"❌ FAIL: {key} - Expected: {expected[key]}, Actual: {actual[key]}")
                scenario_passed = False
                all_passed = False
        
        # Additional validations
        if len(results) != total_processed:
            print(f"❌ FAIL: Results count ({len(results)}) != total ingredients ({total_processed})")
            scenario_passed = False
            all_passed = False
        
        if (successful_mappings + duplicated_mappings + not_found_count + error_mappings) != total_processed:
            print(f"❌ FAIL: Sum of categories != total ingredients")
            scenario_passed = False
            all_passed = False
        
        if scenario_passed:
            print(f"✅ PASS: Scenario {i} validation successful!")
        else:
            print(f"❌ FAIL: Scenario {i} validation failed!")
    
    print(f"\n{'='*50}")
    if all_passed:
        print("🎉 TẤT CẢ SCENARIOS PASSED! Counting logic chính xác.")
        print("\n📝 NHẬN XÉT:")
        print("✅ Logic đếm số lượng hoạt động chính xác")
        print("✅ Xử lý duplicate product_id đúng")
        print("✅ Phân loại kết quả chính xác") 
        print("✅ Validation constraints được đảm bảo")
    else:
        print("💥 CÓ SCENARIOS FAILED! Cần sửa logic.")
    
    return all_passed

if __name__ == "__main__":
    test_counting_logic() 