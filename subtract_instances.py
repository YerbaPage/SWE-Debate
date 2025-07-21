#!/usr/bin/env python3
"""
从verified_dataset_ids.txt中减去test_instance.txt中的实例，输出到新的txt文件
"""

def subtract_instances():
    # 读取verified_dataset_ids.txt
    with open('/home/swebench/SWE-Search/verified_dataset_ids.txt', 'r') as f:
        verified_ids = set(line.strip() for line in f if line.strip())
    
    # 读取test_instance.txt
    with open('/home/swebench/SWE-Search/test_instance.txt', 'r') as f:
        test_ids = set(line.strip() for line in f if line.strip())
    
    # 计算差集 (verified - test)
    remaining_ids = verified_ids - test_ids
    
    # 按字母顺序排序
    remaining_ids_sorted = sorted(remaining_ids)
    
    # 输出到新文件
    output_file = '/home/swebench/SWE-Search/remaining_instances.txt'
    with open(output_file, 'w') as f:
        for instance_id in remaining_ids_sorted:
            f.write(instance_id + '\n')
    
    # 打印统计信息
    print(f"原始verified实例数: {len(verified_ids)}")
    print(f"test实例数: {len(test_ids)}")
    print(f"剩余实例数: {len(remaining_ids)}")
    print(f"结果已保存到: {output_file}")
    
    # 验证没有重复
    overlap = verified_ids & test_ids
    print(f"重叠的实例数: {len(overlap)}")
    if overlap:
        print("重叠的实例:", sorted(overlap)[:10], "..." if len(overlap) > 10 else "")

if __name__ == "__main__":
    subtract_instances()
