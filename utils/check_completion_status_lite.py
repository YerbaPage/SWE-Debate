#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import glob
import logging
from typing import List, Tuple

# Set up logging
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def is_instance_completed(instance_id: str, max_iterations: int, base_dir: str = "tmp") -> bool:
    """
    Determine if an instance has completed processing
    
    Revised logic:
    1. Prioritize checking if there are successful patch and report files (patch_applied=True and has patch content)
    2. If there is a successful patch, consider it completed
    3. If there is no successful patch, then check if max_iterations is reached
    4. Reaching max_iterations without finish node is also considered completed
    
    Args:
        instance_id: Instance ID
        max_iterations: Maximum number of iterations
        base_dir: Base directory path
        
    Returns:
        bool: Returns True if instance is completed, otherwise returns False
    """
    # Find trajectory files
    trajectory_files = glob.glob(f"{base_dir}/trajectory/{instance_id}/*_trajectory.json")
    
    if not trajectory_files:
        logger.info(f"No trajectory file found for {instance_id}")
        return False
    
    # Use the latest trajectory file
    latest_trajectory = max(trajectory_files)
    
    try:
        with open(latest_trajectory, 'r', encoding='utf-8') as f:
            trajectory_data = json.load(f)
        
        # Recursively find all nodes
        def find_all_nodes(node_data, nodes_list=None):
            if nodes_list is None:
                nodes_list = []
            if isinstance(node_data, dict):
                nodes_list.append(node_data)
                if "children" in node_data:
                    for child in node_data["children"]:
                        find_all_nodes(child, nodes_list)
            return nodes_list
        
        # Start finding all nodes from root node
        all_nodes = []
        if "root" in trajectory_data:
            all_nodes = find_all_nodes(trajectory_data["root"])
        
        # Step 1: Prioritize checking if there are successful patch and report files
        report_files = glob.glob(f"{base_dir}/experience/{instance_id}/*_report.json")
        
        if report_files:
            # Check the latest report file
            latest_report = max(report_files)
            
            try:
                with open(latest_report, 'r', encoding='utf-8') as f:
                    report_data = json.load(f)
                
                # Check if patch was successfully generated and test executed
                patch_applied = report_data.get("patch_applied", False)
                patch = report_data.get("patch", "")
                
                if patch_applied and patch:
                    logger.info(f"Instance {instance_id}: Successfully completed with patch applied and tested")
                    return True
                    
            except Exception as e:
                logger.error(f"Error reading report file {latest_report}: {e}")
        
        # Step 2: Check if max_iterations is reached
        target_node_id = max_iterations - 1
        target_node = None
        
        for node in all_nodes:
            if node.get("node_id") == target_node_id:
                target_node = node
                break
        
        if target_node:
            # Check if target node is fully executed
            visits = target_node.get("visits", 0)
            has_observation = False
            
# Check if there are observation results in action_steps
            if "action_steps" in target_node:
                for step in target_node["action_steps"]:
                    if step.get("observation") is not None:
                        has_observation = True
                        break
            
            if visits > 0 and has_observation:
                logger.info(f"Instance {instance_id}: Max iterations completed")
                return True
        
        # Step 3: Check if completed for other reasons (such as reaching max_finished_nodes)
# Find finish nodes
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
            # If there are finish nodes but no successful patch, consider it not completed
            return False
        
        # Step 4: No finish nodes and max_iterations not reached, check if there are enough nodes to indicate search has run
  # If node count reaches 80% of max_iterations, it may have terminated early for other reasons
            logger.info(f"Instance {instance_id}: Search appears to have run significantly ({len(all_nodes)} nodes) but no completion criteria met")
            return False
        
        logger.info(f"Instance {instance_id}: Search not completed (only {len(all_nodes)} nodes, target: {max_iterations})")
        return False
            
    except Exception as e:
        logger.error(f"Error checking completion status for {instance_id}: {e}")
        return False


def has_trajectory_file(instance_id: str, base_dir: str) -> bool:
    """
    Check if instance has trajectory file (i.e., has started running)
    
    Args:
        instance_id: Instance ID
        base_dir: Base directory path
        
    Returns:
        bool: Returns True if trajectory file exists, otherwise False
    """
    trajectory_files = glob.glob(f"{base_dir}/trajectory/{instance_id}/*_trajectory.json")
    return len(trajectory_files) > 0


def get_resolved_status(instance_id: str, base_dir: str) -> Tuple[bool, bool]:
    """
    Get the resolved status of instance
    
    Args:
        instance_id: Instance ID
        base_dir: Base directory path
        
    Returns:
        Tuple[has_report, resolved]: (whether has report file, whether resolved)
    """
    report_files = glob.glob(f"{base_dir}/experience/{instance_id}/*_report.json")
    
    if not report_files:
        return False, False
    
    # Use the latest report file
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
    Use is_instance_completed function to count completed instances and calculate accuracy - LITE version
    """
    # Set parameters - specifically for lite dataset
    base_dir = "/data/swebench/workspace/SWE-Search/tmp/SWE_Search_deepseek0324_lite"#_REACT"
    max_iterations = 20
    instance_list_file = "/data/swebench/workspace/SWE-Search/lite.txt"
    
    # Read all instance IDs
    with open(instance_list_file, 'r', encoding='utf-8') as f:
        instance_ids = [line.strip() for line in f if line.strip()]
    
    print(f"📋 Total instances to check: {len(instance_ids)} (LITE dataset)")
    print(f"📁 Base directory: {base_dir}")
    print(f"🔄 Max iterations: {max_iterations}")
    print("-" * 80)
    
    # Statistics for completion status and resolution status
    completed_instances = []
    incomplete_instances = []
    resolved_instances = []
    completed_but_not_resolved = []
    started_but_incomplete = []  # Started but not completed
    not_started = []  # Not started yet
    
    for i, instance_id in enumerate(instance_ids, 1):
        try:
            is_completed = is_instance_completed(instance_id, max_iterations, base_dir)
            has_report, resolved = get_resolved_status(instance_id, base_dir)
            has_trajectory = has_trajectory_file(instance_id, base_dir)
            
            if is_completed:
                completed_instances.append(instance_id)
                if resolved:
                    resolved_instances.append(instance_id)
                    status = "✅ Completed (resolved: True)"
                else:
                    completed_but_not_resolved.append(instance_id)
                    status = "✅ Completed (resolved: False)"
            else:
                incomplete_instances.append(instance_id)
                if has_trajectory:
                    started_but_incomplete.append(instance_id)
                    status = "🔄 Started but not completed"
                else:
                    not_started.append(instance_id)
                    status = "⭕ Not started yet"
            
            print(f"[{i:2d}/{len(instance_ids)}] {instance_id:<25} {status}")
            
        except Exception as e:
            incomplete_instances.append(instance_id)
            not_started.append(instance_id)
            print(f"[{i:2d}/{len(instance_ids)}] {instance_id:<25} ⚠️ Check failed: {str(e)}")
    
    # Calculate various rates
    completion_rate = len(completed_instances) / len(instance_ids) * 100
    accuracy_rate = len(resolved_instances) / len(instance_ids) * 100
    success_rate_in_completed = len(resolved_instances) / len(completed_instances) * 100 if completed_instances else 0
    started_rate = (len(completed_instances) + len(started_but_incomplete)) / len(instance_ids) * 100
    
    # Output statistical results
    print("-" * 80)
    print(f"📊 Statistical Results (LITE dataset):")
    print(f"   Total instances: {len(instance_ids)}")
    print(f"   Completed: {len(completed_instances)}")
    print(f"   Started but not completed: {len(started_but_incomplete)}")
    print(f"   Not started: {len(not_started)}")
    print(f"   Resolved (resolved=True): {len(resolved_instances)}")
    print(f"   Completed but not resolved: {len(completed_but_not_resolved)}")
    print()
    print(f"📈 Rate Calculations:")
    print(f"   Started rate: {started_rate:.1f}% ({len(completed_instances) + len(started_but_incomplete)}/{len(instance_ids)})")
    print(f"   Completion rate: {completion_rate:.1f}% ({len(completed_instances)}/{len(instance_ids)})")
    print(f"   Accuracy rate (overall): {accuracy_rate:.1f}% ({len(resolved_instances)}/{len(instance_ids)})")
    print(f"   Success rate (among completed): {success_rate_in_completed:.1f}% ({len(resolved_instances)}/{len(completed_instances)})")
    
    if not_started:
        print(f"\n⭕ Instances not started ({len(not_started)} instances):")
        for instance_id in not_started:
            print(f"   - {instance_id}")
    
    if started_but_incomplete:
        print(f"\n🔄 Instances started but not completed ({len(started_but_incomplete)} instances):")
        for instance_id in started_but_incomplete:
            print(f"   - {instance_id}")
    
    if completed_but_not_resolved:
        print(f"\n⚠️ Instances completed but not resolved ({len(completed_but_not_resolved)} instances):")
        for instance_id in completed_but_not_resolved:
            print(f"   - {instance_id}")
    
    if resolved_instances:
        print(f"\n🎉 Resolved instances ({len(resolved_instances)} instances):")
        for instance_id in resolved_instances:
            print(f"   - {instance_id}")
    
    return len(completed_instances), len(incomplete_instances), len(resolved_instances), len(instance_ids), len(started_but_incomplete)

if __name__ == "__main__":
    completed_count, incomplete_count, resolved_count, total_count, started_but_incomplete_count = check_completion_status()
    print(f"\n🏁 Final Results (LITE dataset):")
    print(f"   Total: {total_count} instances")
    print(f"   Completed: {completed_count} instances")
    print(f"   Started but not completed: {started_but_incomplete_count} instances") 
    print(f"   Not started: {total_count - completed_count - started_but_incomplete_count} instances")
    print(f"   Resolved: {resolved_count} instances")
    if completed_count > 0:
        print(f"   Accuracy rate (among completed): {resolved_count}/{completed_count} = {resolved_count/completed_count*100:.1f}%")
    else:
        print(f"   Accuracy rate (among completed): 0/0 = 0.0%") 