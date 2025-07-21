#!/usr/bin/env python3
"""
通用的SWE-Search评测结果处理脚本
支持处理任意experience目录，提取patch信息和统计评测结果
"""

import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Set

def process_experience_directory(experience_dir: Path) -> tuple:
    """
    处理experience目录，提取所有实例的patch和评测结果
    
    Args:
        experience_dir: experience目录的路径
    
    Returns:
        tuple: (patches_data, stats_data)
    """
    patches_data = []
    stats = {
        "total_instances": 0,
        "submitted_instances": 0,
        "completed_instances": 0,
        "resolved_instances": 0,
        "unresolved_instances": 0,
        "empty_patch_instances": 0,
        "error_instances": 0,
        "completed_ids": [],
        "incomplete_ids": [],
        "empty_patch_ids": [],
        "submitted_ids": [],
        "resolved_ids": [],
        "unresolved_ids": [],
        "error_ids": [],
        "schema_version": 2
    }
    
    # 获取所有实例目录
    instance_dirs = [d for d in experience_dir.iterdir() if d.is_dir()]
    stats["total_instances"] = len(instance_dirs)
    
    print(f"发现 {len(instance_dirs)} 个实例目录")
    
    for instance_dir in sorted(instance_dirs):
        instance_id = instance_dir.name
        
        # 查找报告文件（支持不同的日期格式）
        report_files = list(instance_dir.glob("*_report.json"))
        if not report_files:
            print(f"Warning: Report file not found for {instance_id}")
            stats["incomplete_ids"].append(instance_id)
            continue
            
        # 使用找到的第一个报告文件
        report_file = report_files[0]
        
        try:
            with open(report_file, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
                
            # 检查必要字段
            if "instance_id" not in report_data:
                print(f"Warning: Missing instance_id in {instance_id}")
                stats["error_ids"].append(instance_id)
                continue
                
            # 提取patch信息
            patch = report_data.get("patch", "")
            if patch:
                patches_data.append({
                    "instance_id": instance_id,
                    "model_patch": patch
                })
                stats["submitted_ids"].append(instance_id)
                stats["submitted_instances"] += 1
            else:
                stats["empty_patch_ids"].append(instance_id)
                stats["empty_patch_instances"] += 1
                
            # 统计评测结果
            if "patch_applied" in report_data and "resolved" in report_data:
                stats["completed_ids"].append(instance_id)
                stats["completed_instances"] += 1
                
                if report_data.get("resolved", False):
                    stats["resolved_ids"].append(instance_id)
                    stats["resolved_instances"] += 1
                else:
                    stats["unresolved_ids"].append(instance_id)
                    stats["unresolved_instances"] += 1
            else:
                stats["incomplete_ids"].append(instance_id)
                
        except json.JSONDecodeError as e:
            print(f"Error reading JSON for {instance_id}: {e}")
            stats["error_ids"].append(instance_id)
            stats["error_instances"] += 1
        except Exception as e:
            print(f"Unexpected error processing {instance_id}: {e}")
            stats["error_ids"].append(instance_id)
            stats["error_instances"] += 1
    
    return patches_data, stats

def main():
    parser = argparse.ArgumentParser(description='处理SWE-Search评测结果')
    parser.add_argument('experience_dir', nargs='?', 
                       help='experience目录路径 (默认: tmp/SWE_Search_deepseek0324_verified75_REACT/experience)')
    parser.add_argument('-o', '--output-prefix', default='swe_search_verified75',
                       help='输出文件前缀 (默认: swe_search_verified75)')
    parser.add_argument('--base-dir', default='.',
                       help='基础目录 (默认: 当前目录)')
    
    args = parser.parse_args()
    
    # 设置路径
    base_dir = Path(args.base_dir).resolve()
    
    if args.experience_dir:
        experience_dir = Path(args.experience_dir)
        if not experience_dir.is_absolute():
            experience_dir = base_dir / experience_dir
    else:
        experience_dir = base_dir / "tmp/SWE_Search_deepseek0324_verified75_REACT/experience"
    
    if not experience_dir.exists():
        print(f"Error: Experience directory not found: {experience_dir}")
        return 1
    
    print(f"Processing experience directory: {experience_dir}")
    
    # 处理数据
    patches_data, stats = process_experience_directory(experience_dir)
    
    # 保存patches到JSONL文件
    patches_file = base_dir / f"{args.output_prefix}.jsonl"
    with open(patches_file, 'w', encoding='utf-8') as f:
        for patch_entry in patches_data:
            f.write(json.dumps(patch_entry, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(patches_data)} patches to {patches_file}")
    
    # 保存统计结果到JSON文件
    stats_file = base_dir / f"{args.output_prefix}_result.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"Saved statistics to {stats_file}")
    
    # 打印统计信息
    print("\n=== 统计信息 ===")
    print(f"总实例数: {stats['total_instances']}")
    print(f"提交实例数: {stats['submitted_instances']}")
    print(f"完成实例数: {stats['completed_instances']}")
    print(f"解决实例数: {stats['resolved_instances']}")
    print(f"未解决实例数: {stats['unresolved_instances']}")
    print(f"空patch实例数: {stats['empty_patch_instances']}")
    print(f"错误实例数: {stats['error_instances']}")
    
    if stats['resolved_instances'] > 0 and stats['submitted_instances'] > 0:
        success_rate = stats['resolved_instances'] / stats['submitted_instances'] * 100
        print(f"成功率: {success_rate:.2f}% ({stats['resolved_instances']}/{stats['submitted_instances']})")
    
    return 0

if __name__ == "__main__":
    exit(main()) 