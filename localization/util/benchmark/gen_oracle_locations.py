import os
import json
import re
import subprocess
import tempfile
import collections
import argparse
import logging
import logging.handlers
from datetime import datetime
from collections import defaultdict
from tqdm import tqdm
from queue import Empty
import uuid
import shutil
from util.utils import load_jsonl, append_to_jsonl
from util.benchmark.setup_repo import setup_repo
from util.benchmark.parse_patch import (
    get_oracle_filenames, parse_patch,
)
from util.benchmark.parse_python_file import (
    parse_python_file,
    parse_class_docstrings, is_docstring,
    parse_import_nodes, is_import_statement,
    parse_comment_nodes, is_comment,
    parse_global_var_from_file, is_global_var
)
import torch.multiprocessing as mp
from datasets import load_dataset


def parse_module_name(code_str: str):
    # Regular expression to match the function definition and extract the name
    match = re.search(r'\bdef\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', code_str)

    if match:
        function_name = match.group(1)
        return function_name
    else:
        # print("No function definition found.")
        return None


def check_moduel_existed(module, file_structure):
    s = file_structure
    module_type = module.split(':')[0].strip()
    module_name = module.split(':')[-1].strip()
    
    if module_type == 'function' and '.' not in module_name:
        for func in s['functions']:
            if func['name'] == module_name:
                return True
    elif module_type == 'function' and '.' in module_name:
        class_name = module_name.split('.')[0]
        method_name = module_name.split('.')[-1]
        cls = [cls for cls in s['classes'] if cls['name'] == class_name]
        if cls:
            method = [method for method in cls[0]['methods'] if method['name'] == method_name]
            if method:
                return True
    elif module_type == 'class':
        cls = [cls for cls in s['classes'] if cls['name'] == module_name]
        if cls:
            return True
        
    return False


# def get_module_from_line_number_with_file_structure(line, file_structure, include_class=False, merge_init=True):
def get_module_from_line_number_with_file_structure(line, file_structure, 
                                                    include_class=False, 
                                                    merge_init=False
                                                    ):
    s = file_structure
    for txt in s['classes']:
        for func in txt['methods']:
            if line >= func['start_line'] and line <= func['end_line']:
                if merge_init and func['name'] == '__init__':
                    desc = f"class: {txt['name']}"
                    return desc
                else:
                    desc = f"function: {txt['name']}.{func['name']}"
                    return desc
                
        # don't belong to any methods
        if line >= txt['start_line'] and line <= txt['end_line']:
            desc = f"class: {txt['name']}"
            # if not txt['methods'] or include_class:
            if include_class:
                return desc
            else:
                return None
            
    for txt in s['functions']:
        if line >= txt['start_line'] and line <= txt['end_line']:
            desc = f"function: {txt['name']}"
            return desc
    
    return None


def apply_patch_str(patch, apply_file_path, hunk_size):
    # Write the patch string to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, mode='w') as temp_patch_file:
        temp_patch_file.write(patch)
        temp_patch_file_path = temp_patch_file.name

    # Apply the patch using
    try:
        result = subprocess.run(
            ['patch', '-p1', '-i', temp_patch_file_path, apply_file_path],
            check=True,
            text=True,
            capture_output=True
        )
        # print("Patch applied successfully.")
        # logging.debug(result.stdout)
        offsets = [0 for i in range(hunk_size)]
        for out in str(result.stdout).splitlines():
            # if out.startswith('patching file'):
            #     offsets.append(0)
            # else:
                # process offset
                # Regular expression to extract offset (including negative values)
            pattern = r"Hunk #(\d+) succeeded at (\d+) \(offset ([+-]?\d+) lines\)"
            match = re.search(pattern, str(out))
            # Extracting the values if a match is found
            if match:
                hunk_id = int(match.group(1))
                offset = int(match.group(3))
                offsets[hunk_id-1] = offset
                
        # logging.debug('offsets', offsets)
        return (True, offsets)
    except subprocess.CalledProcessError as e:
        # logging.warning(f"Error applying patch: {e.stderr}")
        return (False, [])
    finally:
        # Clean up the temporary file
        import os
        os.remove(temp_patch_file_path)


def map_import_lines(codes):
    in_import_statement = False
    open_parens = 0
    line_labels = {}  # Dictionary to store line number and its label (True/False)
    for code in codes:
        content = code['content']
        line_num = code['line']
        stripped_line = content.strip()
        if not in_import_statement:
            if stripped_line.startswith('import ') or stripped_line.startswith('from '):
                in_import_statement = True
                open_parens += stripped_line.count('(') - stripped_line.count(')')
                line_labels[line_num] = True
                if open_parens == 0:
                    in_import_statement = False
            else:
                # Not an import statement
                line_labels[line_num] = False
        else:
            # Inside a multi-line import statement
            open_parens += stripped_line.count('(') - stripped_line.count(')')
            line_labels[line_num] = True
            if open_parens == 0:
                in_import_statement = False
    return line_labels


def group_patch_by_file(patch):
    """
    Groups a patch string by file.

    Args:
        patch (str): The patch content as a string.

    Returns:
        dict: A dictionary where the keys are file paths, and the values are the corresponding patch content.
    """
    patch_by_file = defaultdict(list)
    patch_lines = patch.splitlines()

    current_file = None
    file_header_pattern = r"^(---|\+\+\+) (.+)"

    for line in patch_lines:
        match = re.match(file_header_pattern, line)
        if match:
            current_file = re.sub(r"^(a/|b/)", "", match.group(2))
            patch_by_file[current_file].append(f"{line}\n")
        else:
            if current_file:
                patch_by_file[current_file].append(f"{line}\n")

    return {file: "".join(hunks) for file, hunks in patch_by_file.items()}


def extract_module_from_patch(instance, repo_dir, max_edit_file_num=1,
                              logger=None, 
                              include_gvar=False,
                              rank=0):
    edit_files = get_oracle_filenames(instance['patch'])
    # print(len(edit_files))
    
    # filter python files and limit the number of files
    filtered_edit_files = []
    for fle in edit_files:
        if fle.endswith('.py'):
            filtered_edit_files.append(fle)
    if not filtered_edit_files: return None
    if len(filtered_edit_files) > max_edit_file_num:
        return None
    
    file_changes = parse_patch(instance['patch'])
    # Group the patch by file
    patch_by_file = group_patch_by_file(instance['patch'])
    
    updated_file_changes = []
    for file_change in file_changes:
        file = file_change['file']
        if not file.endswith('.py'): continue
        target_file_path = os.path.join(repo_dir, file)
        
        # initial file structure
        class_info, function_names, file_lines = parse_python_file(target_file_path)
        old_file_structure = {
            "classes": class_info,
            "functions": function_names,
            "text": file_lines,
        }
        old_global_vars = parse_global_var_from_file(target_file_path)
        old_import_nodes = parse_import_nodes(target_file_path)
        old_comment_nodes = parse_comment_nodes(target_file_path)
        old_docstring_nodes = parse_class_docstrings(target_file_path)
        
        # Extract the partial patch for this file
        partial_patch = patch_by_file.get(file)
        if not partial_patch:
            # logging.warning(f"No patch found for {file}")
            continue
        
        # Apply the patch
        success, offsets = apply_patch_str(partial_patch, target_file_path, len(file_change['hunks']))
        if not success:
            # TODO: assert
            return None
        
        # new file structure
        class_info, function_names, file_lines = parse_python_file(target_file_path)
        new_file_structure = {
            "classes": class_info,
            "functions": function_names,
            "text": file_lines,
        }
        new_global_vars = parse_global_var_from_file(target_file_path)
        new_import_nodes = parse_import_nodes(target_file_path)
        new_comment_nodes = parse_comment_nodes(target_file_path)
        new_docstring_nodes = parse_class_docstrings(target_file_path)
        
        changes = collections.defaultdict(list)
        for i, hunk in enumerate(file_change['hunks']):
            # if i == len(offsets): offsets.append(0) # align with hunk size
            
            # process edited lines
            delete_change = hunk['changes']['delete']
            add_change = hunk['changes']['add']
            # deleted_lines, added_lines = [], []
            
            for delete in delete_change:
                line = delete['line'] + offsets[i]
                # is_comment(line, old_comment_nodes) or \
                if is_import_statement(line, old_import_nodes) or \
                    delete['content'].strip().startswith('#') or \
                    is_docstring(line, old_docstring_nodes):
                    continue
                
                # check is global var
                variable = is_global_var(line, old_global_vars)
                if variable:
                    if include_gvar and variable not in changes['edited_modules']:
                        changes['edited_modules'].append(f'variable: {variable}')
                    continue
                
                # check is module
                module = get_module_from_line_number_with_file_structure(line, old_file_structure)
                if module and not module in changes['edited_modules']:
                    changes['edited_modules'].append(module)
                # elif not module and delete['content'].strip():
                #     deleted_lines.append(delete)
                    
            for add in add_change:
                # is_comment(line, new_comment_nodes) or \
                line = add['line'] + offsets[i]
                if is_import_statement(line, new_import_nodes) or \
                    add['content'].strip().startswith('#') or \
                    is_docstring(line, new_docstring_nodes):
                    continue
                
                # check is global var
                variable = is_global_var(line, new_global_vars)
                if variable:
                    if not include_gvar: continue
                    if variable in old_global_vars and f'variable: {variable}' not in changes['edited_modules']:
                        changes['edited_modules'].append(f'variable: {variable}')
                    elif variable not in old_global_vars and f'variable: {variable}' not in changes['added_modules']:
                        changes['added_modules'].append(f'variable: {variable}')
                    continue
                
                # check is module
                module = get_module_from_line_number_with_file_structure(line, new_file_structure)
                if module and \
                    module not in changes['edited_modules'] and \
                    module not in changes['added_modules']:
                    
                    # check if the module in old file
                    if check_moduel_existed(module, old_file_structure):
                        changes['edited_modules'].append(module)
                    else:
                        changes['added_modules'].append(module)
        
        _changes = collections.defaultdict(list)
        for mode, change in changes.items():
            if mode in ['added_lines', 'edited_lines']:
                continue
            for c in change:
                if c.startswith("variable:"):
                    continue
                if mode in ['added_modules', 'edited_modules']:
                    _mode = mode.replace('_modules', '_entities')
                    _changes[_mode].append(f'{file}:{c.split(':')[-1].strip()}')
                
                if c.startswith("function:") and '.' in c:
                    _c = c.split(':')[-1].strip().split('.')[0]
                    if f'{file}:{_c.strip()}' not in _changes[mode]:
                        _changes[mode].append(f'{file}:{_c.strip()}')
                else:
                    if f'{file}:{c.split(':')[-1].strip()}' not in _changes[mode]:
                        _changes[mode].append(f'{file}:{c.split(':')[-1].strip()}')
        
        updated_file_changes.append({
            'file': file,
            # 'changes': changes
            'changes': _changes
        })
    
    return updated_file_changes


def generate_oracle_locations_for_dataset(dataset, split,
                                     output_dir='evaluation/gt_location',
                                     repo_base_dir='playground',
                                     selected_list=None):
    bench_data = load_dataset(dataset, split=split)
    # current_date = datetime.now().strftime('%Y-%m-%d')
    # output_file = f'evaluation/gt_data/SWE-bench_Lite/gt_modules_data_{current_date}.jsonl',
    output_file = os.path.join(output_dir, dataset.split('/')[-1], split, 'gt_location.jsonl')
    os.makedirs(os.path.join(output_dir, dataset.split('/')[-1], split), exist_ok=True)
    processed_instances = []
    if os.path.exists(output_file):
        processed_instances = [data['instance_id'] for data in load_jsonl(output_file)]

    error_list, empty_edit_list = [], []
    for instance in tqdm(bench_data):
        if instance['instance_id'] in processed_instances:
            continue
        if selected_list and instance['instance_id'] not in selected_list:
            continue
        
        try:
            # pull the repo
            os.makedirs(repo_base_dir, exist_ok=True)
            # repo_dir = setup_repo(instance_data=instance, repo_base_dir=repo_base_dir)
            repo_dir = setup_repo(instance_data=instance, repo_base_dir=repo_base_dir,
                                dataset=dataset, split=split)
        
            file_changes = extract_module_from_patch(instance, repo_dir)
            if not file_changes:
                empty_edit_list.append(instance['instance_id'])
                # continue
            # else:
            #     for fchange in file_changes:
            #         if not fchange['changes']: # or \
            #             # "edited_modules" not in fchange['changes'] or \
            #             # not fchange['changes']["edited_modules"]:
            #             empty_edit_list.append(instance['instance_id'])
            #             # continue
            append_to_jsonl({
                    'instance_id': instance['instance_id'],
                    'file_changes': file_changes,
                    'repo': instance['repo'],
                    'base_commit': instance['base_commit'],
                    'problem_statement': instance['problem_statement'],
                    'patch': instance['patch']
                }, output_file)
        except FileNotFoundError as e:
            logging.info(e)
            error_list.append(instance['instance_id'])
            break
    print(empty_edit_list)
    print(error_list)
    return output_file


def run_extract_locations_from_patch(rank, 
                                  queue, log_queue, output_file_lock,
                                  repo_playground, output_file, max_edit_file_num
                                  ):
    queue_handler = logging.handlers.QueueHandler(log_queue)
    logger = logging.getLogger()
    logger.setLevel(logging.getLevelName("DEBUG"))
    logger.handlers = []
    logger.addHandler(queue_handler)

    logger.debug(f"------ rank {rank} start ------")
    
    while True:
        try:
            instance = queue.get_nowait()
        except Empty:
            break
        
        try:
            # pull the repo
            repo_playground = os.path.join(repo_playground, str(uuid.uuid4()))
            os.makedirs(repo_playground, exist_ok=True)
            repo_dir = setup_repo(instance_data=instance, 
                                repo_base_dir=repo_playground,
                                dataset=None, split=None
                                )
            file_changes = extract_module_from_patch(instance, repo_dir, 
                                                     logger=logger,
                                                     max_edit_file_num=max_edit_file_num, rank=rank)
            if not file_changes:
                continue
            # else:
            #     for fchange in file_changes:
            #         if not fchange['changes']: continue
            with output_file_lock:
                with open(output_file, 'a') as f:
                    f.write(json.dumps({
                        'instance_id': instance['instance_id'],
                        'file_changes': file_changes,
                        'repo': instance['repo'],
                        'base_commit': instance['base_commit'],
                        'problem_statement': instance['problem_statement'],
                        'patch': instance['patch']
                    }) + '\n')
        except FileNotFoundError:
            logger.debug(f"rank {rank}: FileNotFoundError.")
            # error_list.append(instance['instance_id'])
        except subprocess.CalledProcessError as e:
            logger.debug(f"rank {rank}: {e}")
            # error_list.append(instance['instance_id'])
        except Exception as e:
            logger.debug(f"rank {rank}: {e}")
        finally:
            if os.path.exists(repo_playground):
                shutil.rmtree(repo_playground)
            # error_list.append(instance['instance_id'])


def generate_oracle_locations_for_data_file(dataset_file, n_limit,
                                          max_edit_file_num=1, 
                                          repo_base_dir='playground/loc_bench',
                                          num_processes=1):
    logging.basicConfig(
        # filename=f"{args.output_folder}/localize.log",
        level=logging.getLevelName('DEBUG'),
        format="%(asctime)s %(filename)s %(levelname)s %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(f"evaluation/gt_data/LOC-bench/gen_gt.log"),
            logging.StreamHandler()
        ]
    )
    
    current_date = datetime.now().strftime('%Y-%m-%d')
    output_file = f'evaluation/gt_data/LOC-bench/gt_modules_data_{max_edit_file_num}file_{current_date}.jsonl'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    processed_instances = []
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            for line in f:
                processed_instances.append(json.loads(line)['instance_id'])     
    
    bench_data = load_jsonl(dataset_file)
    manager = mp.Manager()
    queue = manager.Queue()
    output_file_lock = manager.Lock()
    
    num_instances = 0
    for instance in bench_data[:n_limit]:
        if not instance['instance_id'] in processed_instances:
            queue.put(instance)
            num_instances += 1
    
    log_queue = manager.Queue()
    queue_listener = logging.handlers.QueueListener(log_queue, *logging.getLogger().handlers)
    queue_listener.start()
    mp.spawn(
        run_extract_locations_from_patch,
        nprocs=min(num_instances, num_processes) if num_processes > 0 else num_instances,
        args=(queue, log_queue, output_file_lock,
              repo_base_dir, output_file, max_edit_file_num
              ),
        join=True
    )
    queue_listener.stop()
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo_base_dir', type=str, default='playground/repo_base')
    parser.add_argument('--output_dir', type=str, default='evaluation/gt_location')
    parser.add_argument("--dataset", type=str, default="princeton-nlp/SWE-bench_Lite")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument('--selected_list_file', type=str, default='playground/repo_base')
    parser.add_argument('--loc_bench', action='store_true')
    parser.add_argument("--max_edit_file_num", type=int, default=1)
    parser.add_argument("--num_processes", type=int, default=1)
    parser.add_argument("--gen_n_limit", type=int, default=0)
    # parser.add_argument('--merge_init', action='store_true')
    args = parser.parse_args()
    # # for test/debug
    # bench_data = load_jsonl(args.dataset)
    # instance = [data for data in bench_data if data['instance_id'] == 'Chainlit__chainlit-1441'][0]
    # repo_dir = setup_repo(instance_data=instance, repo_base_dir=args.repo_base_dir,
    #                             dataset=None, split=None)
    # result = extract_module_from_patch(instance, repo_dir,
    #                                    max_edit_file_num=args.max_edit_file_num)
    # print(result)
    
    
    if args.dataset == 'princeton-nlp/SWE-bench_Lite' and args.split == 'test':
        generate_oracle_locations_for_dataset(args.dataset, args.split, 
                                              args.output_dir, args.repo_base_dir)
    elif args.dataset == 'princeton-nlp/SWE-bench' and args.split == 'train':
        with open(args.selected_list_file, 'r') as f:
            selected_list = json.loads(f.read())
        generate_oracle_locations_for_dataset(args.dataset, args.split, 
                                              args.output_dir, args.repo_base_dir, 
                                              selected_list)
        
    if args.loc_bench:
        generate_oracle_locations_for_data_file(args.dataset, args.gen_n_limit,
                                              args.max_edit_file_num, 
                                              args.repo_base_dir, args.num_processes)