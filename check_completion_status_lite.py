#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import glob
import logging
from typing import List, Tuple

# 设置日志
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def is_instance_completed(instance_id: str, max_iterations: int, base_dir: str = "tmp") -> bool:
    """
    判断instance是否已经完成处理
    
    修正后的逻辑：
    1. 优先检查是否有成功的patch和report文件（patch_applied=True且有patch内容）
    2. 如果有成功的patch，直接认为完成
    3. 如果没有成功的patch，再检查是否达到max_iterations
    4. 达到max_iterations但没有finish node也算完成
    
    Args:
        instance_id: 实例ID
        max_iterations: 最大迭代次数
        base_dir: 基础目录路径
        
    Returns:
        bool: 如果instance已经完成返回True，否则返回False
    """
    # 查找trajectory文件
    trajectory_files = glob.glob(f"{base_dir}/trajectory/{instance_id}/*_trajectory.json")
    
    if not trajectory_files:
        logger.info(f"No trajectory file found for {instance_id}")
        return False
    
    # 使用最新的trajectory文件
    latest_trajectory = max(trajectory_files)
    
    try:
        with open(latest_trajectory, 'r', encoding='utf-8') as f:
            trajectory_data = json.load(f)
        
        # 递归查找所有节点
        def find_all_nodes(node_data, nodes_list=None):
            if nodes_list is None:
                nodes_list = []
            if isinstance(node_data, dict):
                nodes_list.append(node_data)
                if "children" in node_data:
                    for child in node_data["children"]:
                        find_all_nodes(child, nodes_list)
            return nodes_list
        
        # 从root节点开始查找所有节点
        all_nodes = []
        if "root" in trajectory_data:
            all_nodes = find_all_nodes(trajectory_data["root"])
        
        # 步骤1: 优先检查是否有成功的patch和report文件
        report_files = glob.glob(f"{base_dir}/experience/{instance_id}/*_report.json")
        
        if report_files:
            # 检查最新的report文件
            latest_report = max(report_files)
            
            try:
                with open(latest_report, 'r', encoding='utf-8') as f:
                    report_data = json.load(f)
                
                # 检查是否成功生成patch并测试执行
                patch_applied = report_data.get("patch_applied", False)
                patch = report_data.get("patch", "")
                
                if patch_applied and patch:
                    logger.info(f"Instance {instance_id}: Successfully completed with patch applied and tested")
                    return True
                    
            except Exception as e:
                logger.error(f"Error reading report file {latest_report}: {e}")
        
        # 步骤2: 检查是否达到max_iterations
        target_node_id = max_iterations - 1
        target_node = None
        
        for node in all_nodes:
            if node.get("node_id") == target_node_id:
                target_node = node
                break
        
        if target_node:
            # 检查目标节点是否完整执行
            visits = target_node.get("visits", 0)
            has_observation = False
            
            # 检查action_steps中是否有观察结果
            if "action_steps" in target_node:
                for step in target_node["action_steps"]:
                    if step.get("observation") is not None:
                        has_observation = True
                        break
            
            if visits > 0 and has_observation:
                logger.info(f"Instance {instance_id}: Max iterations completed")
                return True
        
        # 步骤3: 检查是否因为其他原因（如达到max_finished_nodes）而完成
        # 查找finish nodes
        finish_nodes = []
        for node in all_nodes:
            if "action_steps" in node:
                for step in node["action_steps"]:
                    action = step.get("action", {})
                    if action.get("action_name") == "Finish" or action.get("name") == "Finish":
                        finish_nodes.append(node)
                        break
        
        if finish_nodes:
            logger.info(f"Instance {instance_id}: Found {len(finish_nodes)} finish nodes but no successful patch generated")
            # 如果有finish nodes但没有成功的patch，认为没有完成
            return False
        
        # 步骤4: 没有finish nodes且没有达到max_iterations，检查是否有足够的节点表明搜索已经运行
        if len(all_nodes) >= max_iterations * 0.8:  # 如果节点数达到max_iterations的80%，可能是因为其他原因提前终止
            logger.info(f"Instance {instance_id}: Search appears to have run significantly ({len(all_nodes)} nodes) but no completion criteria met")
            return False
        
        logger.info(f"Instance {instance_id}: Search not completed (only {len(all_nodes)} nodes, target: {max_iterations})")
        return False
            
    except Exception as e:
        logger.error(f"Error checking completion status for {instance_id}: {e}")
        return False


def has_trajectory_file(instance_id: str, base_dir: str) -> bool:
    """
    检查实例是否有trajectory文件（即已开始运行）
    
    Args:
        instance_id: 实例ID
        base_dir: 基础目录路径
        
    Returns:
        bool: 如果有trajectory文件返回True，否则返回False
    """
    trajectory_files = glob.glob(f"{base_dir}/trajectory/{instance_id}/*_trajectory.json")
    return len(trajectory_files) > 0


def get_resolved_status(instance_id: str, base_dir: str) -> Tuple[bool, bool]:
    """
    获取实例的resolved状态
    
    Args:
        instance_id: 实例ID
        base_dir: 基础目录路径
        
    Returns:
        Tuple[has_report, resolved]: (是否有报告文件, 是否已解决)
    """
    report_files = glob.glob(f"{base_dir}/experience/{instance_id}/*_report.json")
    
    if not report_files:
        return False, False
    
    # 使用最新的report文件
    latest_report = max(report_files)
    
    try:
        with open(latest_report, 'r', encoding='utf-8') as f:
            report_data = json.load(f)
        
        resolved = report_data.get("resolved", False)
        return True, resolved
        
    except Exception as e:
        logger.error(f"Error reading report file {latest_report}: {e}")
        return False, False


def check_completion_status():
    """
    使用is_instance_completed函数统计已完成的实例数量，并计算正确率 - LITE版本
    """
    # 设置参数 - 专门针对lite数据集
    base_dir = "/data/swebench/workspace/SWE-Search/tmp/SWE_Search_deepseek0324_lite"#_REACT"
    max_iterations = 20
    instance_list_file = "/data/swebench/workspace/SWE-Search/lite.txt"
    
    # 读取所有实例ID
    with open(instance_list_file, 'r', encoding='utf-8') as f:
        instance_ids = [line.strip() for line in f if line.strip()]
    
    print(f"📋 总共需要检查 {len(instance_ids)} 个实例 (LITE数据集)")
    print(f"📁 基础目录: {base_dir}")
    print(f"🔄 最大迭代数: {max_iterations}")
    print("-" * 80)
    
    # 统计完成状态和解决状态
    completed_instances = []
    incomplete_instances = []
    resolved_instances = []
    completed_but_not_resolved = []
    started_but_incomplete = []  # 已运行但未完成
    not_started = []  # 未开始运行
    
    for i, instance_id in enumerate(instance_ids, 1):
        try:
            is_completed = is_instance_completed(instance_id, max_iterations, base_dir)
            has_report, resolved = get_resolved_status(instance_id, base_dir)
            has_trajectory = has_trajectory_file(instance_id, base_dir)
            
            if is_completed:
                completed_instances.append(instance_id)
                if resolved:
                    resolved_instances.append(instance_id)
                    status = "✅ 已完成 (resolved: True)"
                else:
                    completed_but_not_resolved.append(instance_id)
                    status = "✅ 已完成 (resolved: False)"
            else:
                incomplete_instances.append(instance_id)
                if has_trajectory:
                    started_but_incomplete.append(instance_id)
                    status = "🔄 已运行但未完成"
                else:
                    not_started.append(instance_id)
                    status = "⭕ 未开始运行"
            
            print(f"[{i:2d}/{len(instance_ids)}] {instance_id:<25} {status}")
            
        except Exception as e:
            incomplete_instances.append(instance_id)
            not_started.append(instance_id)
            print(f"[{i:2d}/{len(instance_ids)}] {instance_id:<25} ⚠️ 检查失败: {str(e)}")
    
    # 计算各种率
    completion_rate = len(completed_instances) / len(instance_ids) * 100
    accuracy_rate = len(resolved_instances) / len(instance_ids) * 100
    success_rate_in_completed = len(resolved_instances) / len(completed_instances) * 100 if completed_instances else 0
    started_rate = (len(completed_instances) + len(started_but_incomplete)) / len(instance_ids) * 100
    
    # 输出统计结果
    print("-" * 80)
    print(f"📊 统计结果 (LITE数据集):")
    print(f"   总实例数: {len(instance_ids)}")
    print(f"   已完成数: {len(completed_instances)}")
    print(f"   已运行但未完成数: {len(started_but_incomplete)}")
    print(f"   未开始运行数: {len(not_started)}")
    print(f"   已解决数 (resolved=True): {len(resolved_instances)}")
    print(f"   已完成但未解决数: {len(completed_but_not_resolved)}")
    print()
    print(f"📈 率计算:")
    print(f"   已开始运行率: {started_rate:.1f}% ({len(completed_instances) + len(started_but_incomplete)}/{len(instance_ids)})")
    print(f"   完成率: {completion_rate:.1f}% ({len(completed_instances)}/{len(instance_ids)})")
    print(f"   正确率 (总体): {accuracy_rate:.1f}% ({len(resolved_instances)}/{len(instance_ids)})")
    print(f"   成功率 (已完成中): {success_rate_in_completed:.1f}% ({len(resolved_instances)}/{len(completed_instances)})")
    
    if not_started:
        print(f"\n⭕ 未开始运行的实例 ({len(not_started)}个):")
        for instance_id in not_started:
            print(f"   - {instance_id}")
    
    if started_but_incomplete:
        print(f"\n🔄 已运行但未完成的实例 ({len(started_but_incomplete)}个):")
        for instance_id in started_but_incomplete:
            print(f"   - {instance_id}")
    
    if completed_but_not_resolved:
        print(f"\n⚠️ 已完成但未解决的实例 ({len(completed_but_not_resolved)}个):")
        for instance_id in completed_but_not_resolved:
            print(f"   - {instance_id}")
    
    if resolved_instances:
        print(f"\n🎉 已解决的实例 ({len(resolved_instances)}个):")
        for instance_id in resolved_instances:
            print(f"   - {instance_id}")
    
    return len(completed_instances), len(incomplete_instances), len(resolved_instances), len(instance_ids), len(started_but_incomplete)

if __name__ == "__main__":
    completed_count, incomplete_count, resolved_count, total_count, started_but_incomplete_count = check_completion_status()
    print(f"\n🏁 最终结果 (LITE数据集):")
    print(f"   总数: {total_count} 个")
    print(f"   已完成: {completed_count} 个")
    print(f"   已运行但未完成: {started_but_incomplete_count} 个") 
    print(f"   未开始运行: {total_count - completed_count - started_but_incomplete_count} 个")
    print(f"   已解决: {resolved_count} 个")
    if completed_count > 0:
        print(f"   正确率 (已完成中): {resolved_count}/{completed_count} = {resolved_count/completed_count*100:.1f}%")
    else:
        print(f"   正确率 (已完成中): 0/0 = 0.0%") 