import json
import logging
import time
import torch.multiprocessing as mp
from typing import List, Dict, Tuple, Optional
from pathlib import Path

from LocAgent.auto_search_main import (
    auto_search_process, 
    get_loc_results_from_raw_outputs,
    get_task_instruction
)
from LocAgent.util.runtime import function_calling
from LocAgent.util.prompts.pipelines import auto_search_prompt as auto_search

# 添加正确的LLM调用导入
import litellm

# Entity提取的prompt模板
ENTITY_EXTRACTION_PROMPT = """
You are a code analysis expert. Given a context containing code entities and an issue description, your task is to identify the most relevant entry entities that are likely related to the issue.

**Context:**
{context}

**Issue Description:**
{issue_description}

**Instructions:**
1. Analyze the issue description to understand what functionality or components are involved
2. From the provided context, identify entities (functions, classes, methods) that are most likely to be entry points for investigating this issue
3. Focus on entities that are directly mentioned in the issue or are likely to be the main interfaces for the problematic functionality
4. Return a prioritized list of entity names (maximum 5) that would be good starting points for investigation
5. Only return the simple entity names (function names, class names, method names), not full paths

**Output Format:**
Return a JSON list of simple entity names in order of relevance (most relevant first):
["entity_name1", "entity_name2", "entity_name3", ...]

**Example:**
If the issue is about pagination iteration, you might return:
["__iter__", "page_range", "Paginator"]

Note: Return only the simple names like "__iter__", "page_range", "MyClass", "my_function", etc. Do not include file paths or full qualified names.
"""

# 相关性评分的prompt模板
RELEVANCE_SCORING_PROMPT = """
You are a code analysis expert. Given an issue description and code entities found during a search, evaluate how relevant these entities are to solving the issue.

**Issue Description:**
{issue_description}

**Found Entities and Code:**
{entity_code_info}

**Instructions:**
1. Analyze each entity's code to understand its functionality
2. Evaluate how directly each entity relates to the issue described
3. Consider if the entities contain the root cause or solution to the issue
4. Provide a relevance score from 0-100 where:
   - 90-100: Directly contains the issue/solution
   - 70-89: Highly relevant, key component
   - 50-69: Moderately relevant, supporting component
   - 30-49: Somewhat relevant, peripheral component
   - 0-29: Not relevant or unrelated

**Output Format:**
Return a JSON object with overall score and explanation:
{{
    "overall_score": <0-100>,
    "explanation": "Detailed explanation of why these entities are relevant to the issue",
    "entity_scores": {{
        "entity_id_1": {{"score": <0-100>, "reason": "explanation"}},
        "entity_id_2": {{"score": <0-100>, "reason": "explanation"}},
        ...
    }}
}}
"""

# RELEVANCE_SCORING_PROMPT = """
# You are an experienced software engineer and code reviewer. Given an issue description and code entities found during a search, thoroughly evaluate how critical these entities are to solving the issue.

# **Issue Description:**
# {issue_description}

# **Found Entities and Code:**
# {entity_code_info}

# **Evaluation Dimensions:**
# 1. **Functional Relevance** (0-30 points):
#    - Does the code directly implement the feature/bug mentioned in the issue?
#    - Is it in the execution path of the described problem?

# 2. **Impact Potential** (0-25 points):
#    - How significantly would modifying this code affect the issue?
#    - Is this a high-leverage point for solving the problem?

# 3. **Contextual Alignment** (0-20 points):
#    - Does the code contain keywords/concepts from the issue?
#    - Is it mentioned in stack traces or error messages?

# 4. **Structural Criticality** (0-15 points):
#    - Is this a core component or infrastructure code?
#    - Does it contain critical business logic or algorithms?

# 5. **Modification Feasibility** (0-10 points):
#    - Is this code modular enough to be safely modified?
#    - Does it have clear inputs/outputs for testing changes?

# **Scoring Rubric:**
# - 95-100: Clearly contains root cause/solution; minimal changes needed
# - 85-94: Core component directly related to issue; high impact potential
# - 75-84: Important component with clear connection to issue
# - 60-74: Relevant component but may require additional changes
# - 40-59: Somewhat related but not likely the key solution point
# - 20-39: Peripheral connection; unlikely to solve issue directly
# - 0-19: No apparent relevance to the issue

# **Additional Considerations:**
# - Penalize scores for overly complex code that would be risky to modify
# - Note any dependencies that might be affected by changes

# **Output Format:**
# Return a JSON object with overall score and explanation:
# {{
#     "overall_score": <0-100>,
#     "explanation": "Detailed explanation of why these entities are relevant to the issue",
#     "entity_scores": {{
#         "entity_id_1": {{"score": <0-100>, "reason": "explanation"}},
#         "entity_id_2": {{"score": <0-100>, "reason": "explanation"}},
#         ...
#     }}
# }}
# """
# # Return a JSON object with detailed analysis:
# # {{
# #     "overall_assessment": {{
# #         "score": <0-100>,
# #         "confidence": <0-100>,  # How confident you are in this evaluation
# #         "summary": "Concise problem-solution fit analysis"
# #     }},
# #     "detailed_scores": {{
# #         "entity_id_1": {{
# #             "score": <0-100>,
# #             "dimension_scores": {{
# #                 "functional": <0-30>,
# #                 "impact": <0-25>,
# #                 "contextual": <0-20>,
# #                 "structural": <0-15>,
# #                 "feasibility": <0-10>
# #             }},
# #             "strengths": ["list of positive aspects"],
# #             "risks": ["potential drawbacks or risks"],
# #             "recommendation": "Should this be modified first? (High/Medium/Low priority)"
# #         }},
# #         ...
# #     }},
# #     "comparative_analysis": "How the entities relate to each other in solving the issue",
# #     "potential_omissions": "Any important code that seems missing based on the issue"
# # }}

def call_llm_simple(
    model_name: str,
    messages: List[Dict],
    temp: float = 0.1,
    max_tokens: int = 1000
) -> str:
    """
    简单的LLM调用函数
    
    Args:
        model_name: 模型名称
        messages: 消息列表
        temp: 温度参数
        max_tokens: 最大token数
    
    Returns:
        模型响应文本
    """
    try:
        response = litellm.completion(
            model=model_name,
            messages=messages,
            temperature=temp,
            max_tokens=max_tokens,
            timeout=60
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"LLM调用失败: {e}")
        raise e

def extract_entities_from_context(
    context: str, 
    issue_description: str, 
    model_name: str = "deepseek/deepseek-chat"
) -> List[str]:
    """
    从context中提取与issue相关的entity列表
    
    Args:
        context: 包含可能entity的上下文
        issue_description: issue描述
        model_name: 使用的模型名称
    
    Returns:
        提取出的entity列表
    """
    prompt = ENTITY_EXTRACTION_PROMPT.format(
        context=context,
        issue_description=issue_description
    )
    
    messages = [{"role": "user", "content": prompt}]
    
    try:
        response = call_llm_simple(
            model_name=model_name,
            messages=messages,
            temp=0.1,
            max_tokens=1000
        )
        
        # 解析JSON响应
        response_text = response.strip()
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        
        entities = json.loads(response_text)
        logging.info(f"提取到的entities: {entities}")
        return entities
        
    except Exception as e:
        logging.error(f"Entity提取失败: {e}")
        return []

def search_single_entity(
    entity_id: str,
    instance_data: Dict,
    model_name: str = "deepseek/deepseek-chat",
    max_iteration_num: int = 10,
    max_retries: int = 3,
    retry_delay: float = 5.0
) -> Optional[Dict]:
    """
    对单个entity进行搜索，包含重试机制
    
    Args:
        entity_id: 要搜索的entity标识（简单名称，如 "__iter__", "page_range" 等）
        instance_data: 实例数据
        model_name: 模型名称
        max_iteration_num: 最大迭代次数
        max_retries: 最大重试次数
        retry_delay: 重试间隔时间（秒）
    
    Returns:
        搜索结果，包含entity_with_code等信息
    """
    # 构建针对特定entity的搜索指令
    search_instruction = f"""
Please investigate the entity `{entity_id}` and its related components in the context of this issue:

{instance_data.get('problem_statement', '')}

Focus on:
1. Understanding the implementation of {entity_id} (this could be a function, method, or class name)
2. Finding related methods, classes, or functions that interact with it
3. Identifying any issues or areas that need modification
4. Search for this entity name across the codebase to find all relevant locations

Provide detailed analysis of the relevant code locations containing "{entity_id}".
"""
    
    # 准备消息
    system_prompt = function_calling.SYSTEM_PROMPT
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": get_task_instruction(instance_data, include_pr=True, include_hint=True)}
    ]
    
    # 设置结果队列
    ctx = mp.get_context('fork')
    
    # 获取工具配置
    tools = function_calling.get_tools(
        codeact_enable_search_keyword=True,
        codeact_enable_search_entity=True,
        codeact_enable_tree_structure_traverser=True,
        simple_desc=False
    )
    
    # 重试循环
    for retry_count in range(max_retries):
        logging.info(f"Entity {entity_id} 搜索尝试 {retry_count + 1}/{max_retries}")
        
        try:
            # 为每次重试创建新的队列
            result_queue = ctx.Manager().Queue()
            
            # 启动搜索进程
            process = ctx.Process(target=auto_search_process, kwargs={
                'result_queue': result_queue,
                'model_name': model_name,
                'messages': messages.copy(),  # 使用消息副本
                'fake_user_msg': auto_search.FAKE_USER_MSG_FOR_LOC,
                'temp': 1.0,
                'tools': tools,
                'use_function_calling': True,
                'max_iteration_num': max_iteration_num
            })
            
            timeout = 300  # 5分钟超时
            process.start()
            process.join(timeout=timeout)
            
            if process.is_alive():
                logging.warning(f"Entity {entity_id} 第{retry_count + 1}次搜索超时")
                process.terminate()
                process.join()
                if retry_count < max_retries - 1:
                    logging.info(f"等待 {retry_delay} 秒后重试...")
                    time.sleep(retry_delay)
                continue
            
            # 获取结果
            if result_queue.empty():
                logging.warning(f"Entity {entity_id} 第{retry_count + 1}次搜索队列为空")
                if retry_count < max_retries - 1:
                    logging.info(f"等待 {retry_delay} 秒后重试...")
                    time.sleep(retry_delay)
                continue
                
            result = result_queue.get()
            
            if isinstance(result, dict) and 'error' in result:
                logging.warning(f"Entity {entity_id} 第{retry_count + 1}次搜索出错: {result['error']}")
                if retry_count < max_retries - 1:
                    logging.info(f"等待 {retry_delay} 秒后重试...")
                    time.sleep(retry_delay)
                continue
            
            loc_result, messages_result, traj_data = result
            
            if not loc_result or not loc_result.strip():
                logging.warning(f"Entity {entity_id} 第{retry_count + 1}次搜索结果为空")
                if retry_count < max_retries - 1:
                    logging.info(f"等待 {retry_delay} 秒后重试...")
                    time.sleep(retry_delay)
                continue
            
            # 解析搜索结果
            raw_output = [loc_result]
            all_found_files, all_found_modules, all_found_entities, entity_with_code = get_loc_results_from_raw_outputs(
                instance_data["instance_id"], raw_output
            )
            
            # 检查是否找到有效的实体代码
            if not entity_with_code or len(entity_with_code) == 0:
                logging.warning(f"Entity {entity_id} 第{retry_count + 1}次搜索未找到有效的实体代码")
                if retry_count < max_retries - 1:
                    logging.info(f"等待 {retry_delay} 秒后重试...")
                    time.sleep(retry_delay)
                continue
            
            # 成功获取有效结果
            search_result = {
                "entity_id": entity_id,
                "found_files": all_found_files[0] if all_found_files else [],
                "found_entities": all_found_entities[0] if all_found_entities else [],
                "entity_with_code": entity_with_code,
                "raw_output": loc_result,
                "usage": traj_data.get('usage', {}),
                "retry_count": retry_count + 1
            }
            
            logging.info(f"Entity {entity_id} 第{retry_count + 1}次搜索成功，找到 {len(entity_with_code)} 个实体代码")
            return search_result
            
        except Exception as e:
            logging.error(f"Entity {entity_id} 第{retry_count + 1}次搜索异常: {e}")
            if retry_count < max_retries - 1:
                logging.info(f"等待 {retry_delay} 秒后重试...")
                time.sleep(retry_delay)
            continue
    
    # 所有重试都失败
    logging.error(f"Entity {entity_id} 经过 {max_retries} 次重试后仍然失败")
    return None

def score_search_results(
    search_results: Dict,
    issue_description: str,
    model_name: str = "deepseek/deepseek-chat"
) -> Dict:
    """
    对搜索结果进行相关性评分
    
    Args:
        search_results: 搜索结果
        issue_description: issue描述
        model_name: 模型名称
    
    Returns:
        评分结果
    """
    # 构建entity代码信息
    entity_info_list = []
    for entity in search_results.get("entity_with_code", []):
        entity_info = f"""
Entity: {entity['entity_id']}
Type: {entity['type']}
Code:
{entity.get('code', 'No code available')}
"""
        entity_info_list.append(entity_info)
    
    entity_code_info = "\n".join(entity_info_list)
    
    prompt = RELEVANCE_SCORING_PROMPT.format(
        issue_description=issue_description,
        entity_code_info=entity_code_info
    )

    logging.info(f"评分Prompt: {prompt}")
    
    messages = [{"role": "user", "content": prompt}]
    
    try:
        response = call_llm_simple(
            model_name=model_name,
            messages=messages,
            temp=0.7,
            max_tokens=2000
        )
        
        # 解析JSON响应
        response_text = response.strip()
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        
        score_result = json.loads(response_text)
        return score_result
        
    except Exception as e:
        logging.error(f"评分失败: {e}")
        return {"overall_score": 0, "explanation": f"评分失败: {e}", "entity_scores": {}}

def entity_based_search_and_score(
    context: str,
    issue_description: str,
    instance_data: Optional[Dict] = None,
    model_name: str = "deepseek/deepseek-chat",
    max_entities: int = 5,
    max_iteration_per_entity: int = 10,
    max_retries_per_entity: int = 3
) -> Dict:
    """
    基于entity的搜索和评分主函数
    
    Args:
        context: 包含可能entity的上下文
        issue_description: issue描述
        instance_data: 实例数据，如果为None则使用当前issue
        model_name: 模型名称
        max_entities: 最大处理的entity数量
        max_iteration_per_entity: 每个entity的最大搜索迭代次数
        max_retries_per_entity: 每个entity的最大重试次数
    
    Returns:
        包含最佳结果的字典
    """
    
    logging.info(f"开始entity搜索和评分流程，instance: {instance_data.get('instance_id', 'unknown')}")
    
    # 1. 提取entities
    logging.info("步骤1: 从context中提取entities")
    entities = extract_entities_from_context(context, issue_description, model_name)
    
    if not entities:
        logging.warning("未提取到任何entities")
        return {
            "success": False,
            "error": "No entities extracted from context",
            "results": []
        }
    
    # 限制entity数量
    entities = entities[:max_entities]
    logging.info(f"将处理 {len(entities)} 个entities: {entities}")
    
    # 2. 对每个entity进行搜索
    logging.info("步骤2: 对每个entity进行搜索")
    search_results_list = []
    
    for i, entity in enumerate(entities):
        logging.info(f"搜索entity {i+1}/{len(entities)}: {entity}")
        
        search_result = search_single_entity(
            entity, 
            instance_data, 
            model_name, 
            max_iteration_per_entity,
            max_retries_per_entity
        )
        
        if search_result:
            search_results_list.append(search_result)
            logging.info(f"Entity {entity} 搜索完成，找到 {len(search_result.get('entity_with_code', []))} 个代码实体，重试了 {search_result.get('retry_count', 1)} 次")
            logging.info(f"搜索到的具体entity_with_code: {search_result.get('entity_with_code', [])}")
        else:
            logging.warning(f"Entity {entity} 搜索失败，已达到最大重试次数")
    
    if not search_results_list:
        logging.warning("所有entity搜索都失败了")
        return {
            "success": False,
            "error": "All entity searches failed after retries",
            "results": []
        }
    
    # 3. 对每个搜索结果进行评分
    logging.info("步骤3: 对搜索结果进行相关性评分")
    scored_results = []
    
    for search_result in search_results_list:
        logging.info(f"评分entity: {search_result['entity_id']}")
        
        score_result = score_search_results(search_result, issue_description, model_name)
        
        scored_results.append({
            "entity_id": search_result["entity_id"],
            "search_result": search_result,
            "score_result": score_result,
            "overall_score": score_result.get("overall_score", 0)
        })
        
        logging.info(f"Entity {search_result['entity_id']} 得分: {score_result.get('overall_score', 0)}")
    
    # 4. 选择最高分的结果
    if scored_results:
        best_result = max(scored_results, key=lambda x: x["overall_score"])
        logging.info(f"最佳结果: {best_result['entity_id']} (得分: {best_result['overall_score']})")
        
        return {
            "success": True,
            "best_result": best_result,
            "all_results": scored_results,
            "best_entity_with_code": best_result["search_result"].get("entity_with_code", [])
        }
    else:
        return {
            "success": False,
            "error": "No valid scored results",
            "results": []
        }

def save_search_results(results: Dict, output_file: Optional[str] = None) -> str:
    """
    保存搜索结果到文件
    
    Args:
        results: 搜索结果
        output_file: 输出文件路径，如果为None则自动生成
    
    Returns:
        保存的文件路径
    """
    if output_file is None:
        output_dir = Path(__file__).parent / "outputs"
        output_dir.mkdir(exist_ok=True)
        timestamp = int(time.time())
        output_file = output_dir / f"entity_search_results_{timestamp}.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    logging.info(f"搜索结果已保存到: {output_file}")
    return str(output_file)
