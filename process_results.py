#!/usr/bin/env python3
"""
Universal SWE-Search evaluation result processing script
Supports processing arbitrary experience directories, extracting patch information and statistical evaluation results
"""

import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Set

def process_experience_directory(experience_dir: Path) -> tuple:
    """
    Process experience directory, extract all instance patches and evaluation results
    
    Args:
        experience_dir: Path to the experience directory
    
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
    
    # Get all instance directories
    instance_dirs = [d for d in experience_dir.iterdir() if d.is_dir()]
    stats["total_instances"] = len(instance_dirs)
    
    print(f"Found {len(instance_dirs)} instance directories")
    
    for instance_dir in sorted(instance_dirs):
        instance_id = instance_dir.name
        
        # Look for report files (supports different date formats)
        report_files = list(instance_dir.glob("*_report.json"))
        if not report_files:
            print(f"Warning: Report file not found for {instance_id}")
            stats["incomplete_ids"].append(instance_id)
            continue
            
        # Use the first report file found
        report_file = report_files[0]
        
        try:
            with open(report_file, 'r', encoding='utf-8') as f:
                report_data = json.load(f)
                
            # Check required fields
            if "instance_id" not in report_data:
                print(f"Warning: Missing instance_id in {instance_id}")
                stats["error_ids"].append(instance_id)
                continue
                
            # Extract patch information
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
                
            # Statistics for evaluation results
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
    parser = argparse.ArgumentParser(description='Process SWE-Search evaluation results')
    parser.add_argument('experience_dir', nargs='?', 
                       help='Experience directory path (default: tmp/SWE_Search_deepseek0324_verified75_REACT/experience)')
    parser.add_argument('-o', '--output-prefix', default='swe_search_verified75',
                       help='Output file prefix (default: swe_search_verified75)')
    parser.add_argument('--base-dir', default='.',
                       help='Base directory (default: current directory)')
    
    args = parser.parse_args()
    
    # Set paths
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
    
    # Process data
    patches_data, stats = process_experience_directory(experience_dir)
    
    # Save patches to JSONL file
    patches_file = base_dir / f"{args.output_prefix}.jsonl"
    with open(patches_file, 'w', encoding='utf-8') as f:
        for patch_entry in patches_data:
            f.write(json.dumps(patch_entry, ensure_ascii=False) + '\n')
    
    print(f"Saved {len(patches_data)} patches to {patches_file}")
    
    # Save statistics to JSON file
    stats_file = base_dir / f"{args.output_prefix}_result.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"Saved statistics to {stats_file}")
    
    # Print statistics
    print("\n=== Statistics ===")
    print(f"Total instances: {stats['total_instances']}")
    print(f"Submitted instances: {stats['submitted_instances']}")
    print(f"Completed instances: {stats['completed_instances']}")
    print(f"Resolved instances: {stats['resolved_instances']}")
    print(f"Unresolved instances: {stats['unresolved_instances']}")
    print(f"Empty patch instances: {stats['empty_patch_instances']}")
    print(f"Error instances: {stats['error_instances']}")
    
    if stats['resolved_instances'] > 0 and stats['submitted_instances'] > 0:
        success_rate = stats['resolved_instances'] / stats['submitted_instances'] * 100
        print(f"Success rate: {success_rate:.2f}% ({stats['resolved_instances']}/{stats['submitted_instances']})")
    
    return 0

if __name__ == "__main__":
    exit(main()) 