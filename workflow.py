from dotenv import load_dotenv
load_dotenv(".env")
import os
import sys
import json
import random
import argparse
import logging
import glob
import time
import multiprocessing as mp
from datetime import datetime
from typing import Dict, List, Tuple
from tqdm import tqdm

# Add localization directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'localization'))

from moatless.benchmark.utils import get_moatless_instance
from moatless.benchmark.swebench import create_repository
from moatless.index.code_index import CodeIndex
from moatless.index.settings import IndexSettings
# from moatless.runtime.testbed import TestbedEnvironment

from moatless.file_context import FileContext
from moatless.feedback.feedback_agent import FeedbackAgent
from moatless.value_function.base import ValueFunction

from moatless.actions import FindClass, FindFunction, FindCodeSnippet, SemanticSearch, ViewCode, Finish, Reject, RunTests, StringReplace, CreateFile
from moatless.agent.code_agent import CodingAgent
from moatless.agent.code_prompts import *
from moatless.search_tree import SearchTree
from moatless.completion.completion import (
    LLMResponseFormat,
    CompletionModel,
)

from moatless.discriminator import AgentDiscriminator


import traceback
import re
from enum import Enum

logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class RetryReason(Enum):
    JSON_PARSE_ERROR = "json_parse_error"
    PATCH_GENERATION_FAILED = "patch_generation_failed"  
    NONE_TYPE_ACTION = "none_type_action"
    TESTBED_TIMEOUT = "testbed_timeout"
    UNKNOWN_ERROR = "unknown_error"

def should_retry_error(error_message: str, exception: Exception = None) -> Tuple[bool, RetryReason]:

    error_lower = error_message.lower()
    
    # 1. django__django-11815: 'NoneType' object has no attribute 'name'
    if "'nonetype' object has no attribute 'name'" in error_lower:
        return True, RetryReason.NONE_TYPE_ACTION
    
    # 2. sympy__sympy-18189: TestbedTimeoutError: Request to ... timed out after 3 retries  
    if "testbedtimeouterror" in error_lower or "apply-patch timed out" in error_lower:
        return True, RetryReason.TESTBED_TIMEOUT
    
    # 3. Has True finish nodes but Unable to generate patch
    if "Unable to generate patch" in error_lower and "The real finished_node exists but there is no patch" in error_lower:
        return True, RetryReason.PATCH_GENERATION_FAILED
    
    return False, RetryReason.UNKNOWN_ERROR


def is_instance_completed(instance_id: str, max_iterations: int, base_dir: str = "tmp") -> bool:
    exp_name = os.path.basename(base_dir)
    patches_file = os.path.join(base_dir, f"model_patches_{exp_name}.jsonl")
    
    if os.path.exists(patches_file):
        with open(patches_file, 'r') as f:
            for line in f:
                try:
                    patch_record = json.loads(line)
                    if patch_record["instance_id"] == instance_id:
                        return True
                except json.JSONDecodeError:
                    continue
    
    trajectory_files = glob.glob(f"{base_dir}/trajectory/{instance_id}/*_trajectory.json")
    
    if not trajectory_files:
        logger.info(f"No trajectory file found for {instance_id}")
        return False
    
    latest_trajectory = max(trajectory_files)
    
    try:
        with open(latest_trajectory, 'r', encoding='utf-8') as f:
            trajectory_data = json.load(f)
        
        def find_all_nodes(node_data, nodes_list=None):
            if nodes_list is None:
                nodes_list = []
            if isinstance(node_data, dict):
                nodes_list.append(node_data)
                if "children" in node_data:
                    for child in node_data["children"]:
                        find_all_nodes(child, nodes_list)
            return nodes_list
        
        all_nodes = []
        if "root" in trajectory_data:
            all_nodes = find_all_nodes(trajectory_data["root"])
        
        report_files = glob.glob(f"{base_dir}/experience/{instance_id}/*_report.json")
        
        if report_files:
            latest_report = max(report_files)
            
            try:
                with open(latest_report, 'r', encoding='utf-8') as f:
                    report_data = json.load(f)
                
                patch_applied = report_data.get("patch_applied", False)
                patch = report_data.get("patch", "")
                
                if patch_applied and patch:
                    logger.info(f"Instance {instance_id}: Successfully completed with patch applied and tested")
                    return True
                    
            except Exception as e:
                logger.error(f"Error reading report file {latest_report}: {e}")

        current_date = datetime.now().strftime("%Y-%m-%d")
        execution_log_path = f'{base_dir}/trajectory/{instance_id}/{current_date}_execution.log'
        
        if os.path.exists(execution_log_path):
            try:
                with open(execution_log_path, 'r', encoding='utf-8') as f:
                    log_content = f.read()
                    
                testbed_errors = [
                    "Health check failed",
                    "TestbedTimeoutError", 
                    "Error running tests for instance",
                    "apply-patch timed out",
                    "Request failed, retrying",
                    "Operation timed out after"
                ]
                
                for error_pattern in testbed_errors:
                    if error_pattern in log_content:
                        logger.info(f"Instance {instance_id}: Detected testbeds failure ({error_pattern}), marking as incomplete for retry")
                        return False
                        
            except Exception as e:
                logger.warning(f"Error reading execution log for {instance_id}: {e}")
        
        target_node_id = max_iterations - 1
        target_node = None
        
        for node in all_nodes:
            if node.get("node_id") == target_node_id:
                target_node = node
                break
        
        if target_node:
            visits = target_node.get("visits", 0)
            has_observation = False

            if "action_steps" in target_node:
                for step in target_node["action_steps"]:
                    if step.get("observation") is not None:
                        has_observation = True
                        break
            
            if visits > 0 and has_observation:
                logger.info(f"Instance {instance_id}: Max iterations completed")
                return True

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

            return False

        if len(all_nodes) >= max_iterations * 0.8: 
            logger.info(f"Instance {instance_id}: Search appears to have run significantly ({len(all_nodes)} nodes) but no completion criteria met")
            return False
        
        logger.info(f"Instance {instance_id}: Search not completed (only {len(all_nodes)} nodes, target: {max_iterations})")
        return False
            
    except Exception as e:
        logger.error(f"Error checking completion status for {instance_id}: {e}")
        return False


def load_jsonl(file_path):
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data_list.append(json.loads(line.strip()))
    return data_list


def save2json(data, path):
    directory = os.path.dirname(path)
    
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
        
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def save_to_jsonl(data_list, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data_list:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def add_data_to_jsonl(file_path, new_data):
    if file_path and os.path.exists(file_path):
        data_list = load_jsonl(file_path)
    else:
        data_list = []

    if isinstance(new_data, list):
        data_list.extend(new_data)
    else:
        data_list.append(new_data)

    save_to_jsonl(data_list, file_path)


def get_leaves_with_patch(tree) -> Dict:
    leaves = tree.get_leaf_nodes()
    parent_ids = set()
    tmp = dict()
    counter = 0
    for leaf in leaves:
        if leaf.file_context.generate_git_patch():
            if leaf.is_terminal() and leaf.parent.node_id not in parent_ids:
                parent_ids.add(leaf.parent.node_id)
                patch = leaf.file_context.generate_git_patch()
                # result = leaf.file_context.run_evaluation()
                report_ = {
                    "leaf_id": leaf.node_id,
                    # "patch_applied": result.patch_applied,
                    # "resolved": result.resolved,
                    "patch": patch,
                }
                tmp[str(leaf.node_id)] = report_
                counter += 1
        else:
            report_ = {
                "leaf_id": leaf.node_id,
                # "patch_applied": False,
                # "resolved": False,
                "patch": None,
            }
            tmp[str(leaf.node_id)] = report_
            counter += 1
    return tmp

def get_path_to_leaf(leaf_node_id, tree):
    try:
        target_node = None
        
        def find_node(node):
            nonlocal target_node
            if node.node_id == leaf_node_id:
                target_node = node
                return
            for child in node.children:
                find_node(child)
        
        find_node(tree.root)
        
        if not target_node:
            return f"Node ID not found: {leaf_node_id}"
        
        path_nodes = []
        current = target_node
        
        while current:
            path_nodes.append(current)
            current = current.parent

        path_nodes.reverse()

        result_parts = []
        for i, node in enumerate(path_nodes):
            if hasattr(node, 'action_steps') and node.action_steps:
                action_step = node.action_steps[0]
                
                action_str = f"{action_step.action.__class__.__name__}: {action_step.action}"
                observation_str = action_step.observation if action_step.observation else "No observation"
                
                result_parts.append(f"Action{node.node_id}:")
                result_parts.append(f"{action_str}")
                result_parts.append(f"Observation: {observation_str}")
                result_parts.append("---")
        
        return "\n".join(result_parts)
        
    except Exception as e:
        return f"error: {e}"


def main(instance_id, max_iterations, max_finish_nodes, result_path=None,use_testbed=False):
    if result_path is None:
        base_dir = os.path.abspath("tmp")
    else:
        base_dir = result_path
    
    instance_logger = setup_instance_logging(instance_id, base_dir)

    import sys
    from contextlib import redirect_stdout, redirect_stderr
    current_date = datetime.now().strftime("%Y-%m-%d")
    log_file_path = f'{base_dir}/trajectory/{instance_id}/{current_date}_execution.log'

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    instance_logger.info(f"🚀 Processing instance: {instance_id}")
    instance_logger.info(f"📁 Base directory: {base_dir}")
    instance_logger.info(f"🔄 Max iterations: {max_iterations}")
    instance_logger.info(f"🎯 Max finished nodes: {max_finish_nodes}")

    retry_count = 0
    MAX_RETRIES = 3 

    class LogFileRedirect:
        def __init__(self, log_file_path):
            self.log_file_path = log_file_path
            
        def write(self, text):
            if text.strip(): 
                with open(self.log_file_path, 'a', encoding='utf-8') as f:
                    f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - STDOUT - INFO - {text}")
                    
        def flush(self):
            pass
    
    log_redirect = LogFileRedirect(log_file_path)
    
    while True:  
        try:
            sys.stdout = log_redirect

            from moatless.completion.react import ReActCompletionModel
            react_completion_model = ReActCompletionModel(model="deepseek/deepseek-chat", temperature=0.7)
            completion_model = CompletionModel(model="deepseek/deepseek-chat", temperature=0.7)
            discriminator_model = CompletionModel(model="deepseek/deepseek-chat", temperature=1)
            value_model = CompletionModel(model="deepseek/deepseek-chat", temperature=0.2)

            completion_model = CompletionModel(model="deepseek/deepseek-chat", temperature=0.7)

            react_completion_model.response_format = LLMResponseFormat.REACT
            completion_model.response_format = LLMResponseFormat.REACT
            discriminator_model.response_format = LLMResponseFormat.REACT
            value_model.response_format = LLMResponseFormat.REACT

            instance = get_moatless_instance(instance_id=instance_id)  
            
            repository = create_repository(instance)
        
            # Set up index store directory
            index_store_dir = os.getenv("INDEX_STORE_DIR", os.path.abspath("tmp/index_store"))
            
            instance_logger.info("🧠 Loading Code Index...")
            code_index = CodeIndex.from_index_name(
                instance["instance_id"], file_repo=repository, index_store_dir=index_store_dir
            )

            instance_logger.info("📄 Initializing file context...")
            file_context = FileContext(repo=repository)

            current_date = datetime.now().strftime("%Y-%m-%d")
            instance_path = f'{base_dir}/trajectory/{instance_id}/'
            persist_path = f'{base_dir}/trajectory/{instance_id}/{current_date}_trajectory.json'
            
            instance_logger.info(f"📊 Trajectory: {persist_path}")

            instance_logger.info("⚙️ Setting up actions and system prompt...")
            value_function = ValueFunction(completion_model=value_model)
            actions = [
                FindClass(completion_model=react_completion_model, code_index=code_index, repository=repository),
                FindFunction(completion_model=react_completion_model, code_index=code_index, repository=repository),
                FindCodeSnippet(completion_model=react_completion_model, code_index=code_index, repository=repository),
                SemanticSearch(completion_model=react_completion_model, code_index=code_index, repository=repository),
                ViewCode(completion_model=react_completion_model, repository=repository),
                StringReplace(repository=repository, code_index=code_index),
                CreateFile(repository=repository, code_index=code_index),
                Finish(),
            ]

            system_prompt = AGENT_ROLE
            if completion_model.response_format == LLMResponseFormat.REACT:
                system_prompt += REACT_CORE_OPERATION_RULES
            elif completion_model.response_format == LLMResponseFormat.TOOLS:
                system_prompt += REACT_GUIDELINES
            workflow_prompt = generate_workflow_prompt(actions, has_runtime=True)
            system_prompt += workflow_prompt + generate_guideline_prompt(has_runtime=True) + ADDITIONAL_NOTES

            instance_logger.info(f"system_prompt:\n{system_prompt}")
            instance_logger.info("🤖 Initializing agent...")
            agent = CodingAgent(system_prompt=system_prompt, actions=actions, completion=react_completion_model)

            discriminator = AgentDiscriminator(
                completion=discriminator_model,
                n_agents=5,
                n_rounds=3,
            )

            feedback_generator = FeedbackAgent(
                completion_model=completion_model, instance_dir=instance_path
            )

            instance_logger.info("🌳 Creating search tree...")

            from entity_localization_pipeline import EntityLocalizationPipeline

            # pipeline = EntityLocalizationPipeline(model_name=completion_model.model)
            pipeline = EntityLocalizationPipeline()
            results = pipeline.run_pipeline(
                instance,"context",
                max_initial_entities=5
            )
          

            search_tree = SearchTree.create(
                message=f'{results}',
                agent=agent,
                # assistant=agent,
                file_context=file_context,
                value_function=value_function,
                discriminator=discriminator,
                feedback_generator=feedback_generator,
                max_finished_nodes=max_finish_nodes,
                max_iterations=max_iterations,
                max_expansions=3,
                max_depth=20,
                max_duplicate_count=5, 
                persist_path=persist_path,
            )

            instance_logger.info("🔍 Starting search...")
            finished_node = search_tree.run_search()
            
            instance_logger.info("💾 Saving Search Tree...")
            search_tree.persist(persist_path)

            if finished_node:
                # if finished_node.is_finished():
                if finished_node.file_context.generate_git_patch():

                    logger.info(f"{instance} patch: {finished_node.file_context.generate_git_patch()}")

                    eva2rollout = get_leaves_with_patch(search_tree)
                    
                    eva2rollout['source_tree_path'] = persist_path
                    eva2rollout['debate_node'] = str(finished_node.node_id)
                    save2json(eva2rollout, f'/tmp/trajectory/{instance_id}/eval2rollout.json')

                    if not search_tree.get_finished_nodes() and finished_node.file_context.generate_git_patch():
                        tmp = {
                            "model_name_or_path": "DeepSeek_IA",
                            "instance_id": instance_id,
                            "model_patch": finished_node.file_context.generate_git_patch(),
                            "leaf_id": finished_node.node_id,
                            'source_tree_path': persist_path,
                            'debate_node': str(finished_node.node_id),
                        }
                        add_data_to_jsonl('/tmp_patch_1.jsonl', tmp)
                        add_data_to_jsonl('/tmp_patch_2.jsonl', tmp)

                    new_eval_objects = []
                    for i in search_tree.get_finished_nodes():
                        trajectory = f'Issue: {instance["problem_statement"]}\nTrajectory:\n'
                        trajectory += get_path_to_leaf(i.node_id, search_tree)
                        trajectory += f"\nGenerated Patch:\n{i.file_context.generate_git_patch()}"
                        tmp = {
                            "model_name_or_path": "DeepSeek_IA",
                            "instance_id": instance_id,
                            "model_patch": i.file_context.generate_git_patch(),
                            "leaf_id": i.node_id,
                            'source_tree_path': persist_path,
                            'debate_node': str(finished_node.node_id),
                            'trajectory': trajectory,
                        }
                        new_eval_objects.append(tmp)

                    if len(new_eval_objects) > 1:
                        add_data_to_jsonl('/tmp_patch_1.jsonl', new_eval_objects[0])
                        add_data_to_jsonl('/tmp_patch_2.jsonl', new_eval_objects[1])
                    elif len(new_eval_objects) == 1:
                        add_data_to_jsonl('/tmp_patch_1.jsonl', new_eval_objects)
                        add_data_to_jsonl('/tmp_patch_2.jsonl', new_eval_objects)
                    

                    tmp = {
                        "model_name_or_path": "DeepSeek_IA",
                        "instance_id": instance_id,
                        "model_patch": finished_node.file_context.generate_git_patch(),
                    }   
                    add_data_to_jsonl('/tmp_prediction_patch.jsonl', tmp)

            else:
                instance_logger.warning("⚠️ Finished node not found - the search ended normally but no solution was found")
                return False
                
        except Exception as e:
            error_msg = str(e)
            instance_logger.error(f"❌ Instance {instance_id} error: {error_msg}")
            import traceback
            full_traceback = traceback.format_exc()
            instance_logger.error(f"📋 Detailed traceback:\n{full_traceback}")
            
            if "MCTS_DUPLICATES_EXCEEDED" in error_msg:
                if retry_count >= MAX_RETRIES:
                    instance_logger.error(f"❌ Maximum retry limit ({MAX_RETRIES}) reached, abandoning this instance")
                    return False
                retry_count += 1
                instance_logger.warning(f"⚠️ Too many duplicate nodes detected, retrying attempt {retry_count}...")
                continue

            should_retry, retry_reason = should_retry_error(error_msg, e)
            if should_retry:
                if retry_count >= MAX_RETRIES:
                    instance_logger.error(f"❌ Maximum retry limit ({MAX_RETRIES}) reached, abandoning this instance")
                    return False
                retry_count += 1
                instance_logger.warning(f"⚠️ Retryable error ({retry_reason.value}) occurred, retry attempt {retry_count}...")
                continue

            return False
            
        finally:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            
    return False  


def parse_slice(slice_str: str) -> slice:
    if not slice_str:
        return slice(None)
    
    try:
        parts = slice_str.split(':')
        
        if len(parts) > 3:
            raise ValueError(f"Invalid slice syntax: {slice_str}")
        
        while len(parts) < 3:
            parts.append('')
        
        def parse_int(s):
            return int(s) if s else None
        
        start = parse_int(parts[0])
        stop = parse_int(parts[1])
        step = parse_int(parts[2])
        
        return slice(start, stop, step)
        
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid slice syntax '{slice_str}': {e}")


def setup_instance_logging(instance_id: str, base_dir: str):
    current_date = datetime.now().strftime("%Y-%m-%d")
    log_dir = f'{base_dir}/trajectory/{instance_id}/'
    os.makedirs(log_dir, exist_ok=True)
    

    log_file = f'{log_dir}/{current_date}_execution.log'
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    instance_logger = logging.getLogger(f"instance_{instance_id}")
    instance_logger.setLevel(logging.INFO)
    
    for handler in instance_logger.handlers[:]:
        instance_logger.removeHandler(handler)
    
    instance_logger.addHandler(file_handler)
    
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter(f'[{instance_id}] %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(logging.WARNING) 
    instance_logger.addHandler(console_handler)

    litellm_logger = logging.getLogger("LiteLLM")
    litellm_logger.setLevel(logging.INFO)

    for handler in litellm_logger.handlers[:]:
        litellm_logger.removeHandler(handler)

    litellm_logger.addHandler(file_handler)
    litellm_logger.propagate = False 
    

    moatless_loggers = [
        "moatless",
        "moatless.search_tree", 
        "moatless.agent",
        "moatless.actions",
        "moatless.completion",
        "moatless.runtime",
        "moatless.file_context",
        "moatless.feedback",
        "moatless.discriminator",
        "moatless.node",
        "moatless.tree",
        "moatless.value_function",
        "moatless.benchmark"
    ]
    
    for logger_name in moatless_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)

        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        logger.addHandler(file_handler)
        logger.propagate = False 
    
    other_loggers = [
        "openai",
        "httpx", 
        "anthropic",
        "deepseek",
        "testbeds"
    ]
    
    for logger_name in other_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)

        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        logger.addHandler(file_handler)
        logger.propagate = False 

    root_logger = logging.getLogger()
    

    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):

            handler.setLevel(logging.WARNING)
        elif isinstance(handler, logging.FileHandler):

            root_logger.removeHandler(handler)
    

    root_logger.setLevel(logging.WARNING)
    
    return instance_logger


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Process some arguments.")


    parser.add_argument("--instance_ids", type=str, required=True,
                        help="The file path to instance ID(s)")

    parser.add_argument("--max_iterations", type=int, default=10, help="Max iteration for tree search")

    parser.add_argument("--max_finished_nodes", type=int, default=3, help="Max finished nodes for tree search")

    parser.add_argument("--resume", action="store_true", help="Resume from the last instance")

    parser.add_argument("--num_processes", type=int, default=4, help="Number of processes for parallel processing")

    parser.add_argument("--result_path", type=str, default=None, help="Custom result directory path (default: tmp/experience)")

    parser.add_argument("--slice", type=str, default=None, help="Python slice syntax to select instances (e.g., '0:10', '10:', ':5', '10:20:2')")

    args = parser.parse_args()

    with open(args.instance_ids, "r", encoding='utf-8') as f:
        instance_ids = [line.strip() for line in f if line.strip()]

    if args.slice:
        try:
            slice_obj = parse_slice(args.slice)
            original_count = len(instance_ids)
            instance_ids = instance_ids[slice_obj]
            print(f"🔢 Applying slice '{args.slice}': {original_count} -> {len(instance_ids)} instances")
            if len(instance_ids) == 0:
                print("⚠️ No instances left after slicing, program exits")
                exit(0)
        except ValueError as e:
            print(f"❌ Slice parameter error: {e}")
            exit(1)
    
    print(f"📋 Finally processing {len(instance_ids)} instances")
    if args.result_path is None:
        base_dir = os.path.abspath("tmp")
    else:
        base_dir = args.result_path
    
    pass_instances = []

    if len(instance_ids) == 1:

        instance_id = instance_ids[0]

        args_tuple = (instance_id, args.max_iterations, args.max_finished_nodes, base_dir, args.resume, args.result_path)
        instance_id_result, success, message = process_single_instance(args_tuple)
        
        if success:
            if "Already completed" not in message:
                pass_instances.append(instance_id)
                print("🎉 Pass@1: 1")
            else:
                print("✅ Pass@1: Already completed")
        else:
            print(f"❌ Pass@1: 0 - {message}")
            if "retry" in message:
                print(f"📊 retry: {message}")
    else:
        pass_instances = process_instances_parallel(instance_ids, args.max_iterations, args.max_finished_nodes, base_dir, args.resume, args.num_processes, args.result_path)
        success_rate = len(pass_instances) / len(instance_ids)
        print(f"\n🎯 Final Result:")
        print(f"   Success Rate: {success_rate:.2%} ({len(pass_instances)}/{len(instance_ids)})")
        print(f"   Successful Instances: {pass_instances}")

        retry_count = sum(1 for instance_id in pass_instances if any(
            "retry" in log_line for log_line in get_retry_info_for_instance(instance_id, base_dir)
        ))
        if retry_count > 0:
            print(f" 💫 Instances that successfully passed retry: {retry_count}/{len(pass_instances)}")

    print('\n🏁 All done!')


def get_retry_info_for_instance(instance_id: str, base_dir: str) -> List[str]:
    try:
        current_date = datetime.now().strftime("%Y-%m-%d")
        log_file_path = f'{base_dir}/trajectory/{instance_id}/{current_date}_execution.log'
        if os.path.exists(log_file_path):
            with open(log_file_path, 'r', encoding='utf-8') as f:
                return [line for line in f.readlines() if "RETRY" in line or "retry" in line]
    except Exception:
        pass
    return []